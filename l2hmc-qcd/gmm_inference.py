"""
gmm_inference.py

Run inference using a trained (L2HMC) model.

Author: Sam Foreman (github: @saforem2)
Date: 09/18/2019
"""
from __future__ import absolute_import, division, print_function

import os
import time
import pickle

from config import HAS_HOROVOD, HAS_MATPLOTLIB, HAS_MEMORY_PROFILER, NP_FLOAT
from update import set_precision
from inference import _log_inference_header
from runners.gmm_runner import GaussianMixtureModelRunner
from plotters.plot_utils import _gmm_plot, gmm_plot, _gmm_plot3d
from loggers.run_logger import RunLogger
from loggers.summary_utils import create_summaries

import numpy as np
import tensorflow as tf

from scipy.stats import sem

import utils.file_io as io

from utils.data_utils import block_resampling, calc_avg_vals_errors
from utils.distributions import GMM
from utils.parse_inference_args import parse_args as parse_inference_args

if HAS_HOROVOD:
    import horovod.tensorflow as hvd

if HAS_MEMORY_PROFILER:
    import memory_profiler

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)


SEP_STR = 80 * '-'


def create_config(params):
    """Helper method for creating a tf.ConfigProto object."""
    config = tf.ConfigProto(allow_soft_placement=True)

    if params.get('gpu', False):
        # Horovod: pin GPU to be used to process local rank 
        # (one GPU per process)
        config.gpu_options.allow_growth = True
        #  config.allow_soft_placement = True
        if HAS_HOROVOD and params['horovod']:
            config.gpu_options.visible_device_list = str(hvd.local_rank())

    if HAS_MATPLOTLIB:
        params['_plot'] = True

    if params['theta']:
        params['_plot'] = False
        io.log("Training on Theta @ ALCF...")
        params['data_format'] = 'channels_last'
        os.environ["KMP_BLOCKTIME"] = str(0)
        os.environ["KMP_AFFINITY"] = (
            "granularity=fine,verbose,compact,1,0"
        )
        # NOTE: KMP affinity taken care of by passing -cc depth to aprun call
        OMP_NUM_THREADS = 62
        config.allow_soft_placement = True
        config.intra_op_parallelism_threads = OMP_NUM_THREADS
        config.inter_op_parallelism_threads = 0

    return config, params


def parse_flags(FLAGS, log_file=None):
    """Parse command line FLAGS and create dictionary of parameters."""
    hmc = getattr(FLAGS, 'hmc', False)
    if not hmc:
        return

    try:
        kv_pairs = FLAGS.__dict__.items()
    except AttributeError:
        kv_pairs = FLAGS.items()

    params = {
        k: v for k, v in kv_pairs
    }

    params['log_dir'] = io.create_log_dir(FLAGS, log_file=log_file)
    params['summaries'] = not getattr(FLAGS, 'no_summaries', False)
    if 'no_summaries' in params:
        del params['no_summaries']

    save_steps = getattr(FLAGS, 'save_steps', None)
    train_steps = getattr(FLAGS, 'train_steps', 5000)
    if save_steps is None and train_steps is not None:
        params['train_steps'] = train_steps
        params['save_steps'] = train_steps // 4

    float64 = getattr(FLAGS, 'float64', False)
    if float64:
        io.log(f'INFO: Setting floating point precision to `float64`.')
        set_precision('float64')

    return params


def load_params(pkl_file=None, log_file=None):
    if pkl_file is None:
        alt_file = os.path.join(os.getcwd(), 'params.pkl')
        if os.path.isfile(alt_file):
            pkl_file = alt_file

    if not os.path.isfile(pkl_file):
        raise FileNotFoundError(
            'Unable to locate the `.pkl` file containing params. Exiting.'
        )

    with open(pkl_file, 'rb') as f:
        params = pickle.load(f)

    return params


def _load_distribution_params(_dir):
    """Load `mus`, `sigmas` and `pis` to reconstruct GMM distribution."""
    mus_file = os.path.join(_dir, 'mus.pkl')
    sigmas_file = os.path.join(_dir, 'sigmas.pkl')
    pis_file = os.path.join(_dir, 'pis.pkl')

    def _load(in_file):
        with open(in_file, 'rb') as f:
            tmp = pickle.load(f)
        return tmp

    mus = _load(mus_file)
    sigmas = _load(sigmas_file)
    pis = _load(pis_file)

    return mus, sigmas, pis


def recreate_distribution(_dir):
    mus, sigmas, pis = _load_distribution_params(_dir)
    return GMM(mus, sigmas, pis)


def error_analysis(samples, num_blocks=None):
    """Calc the mean and std error using blocked jackknife resampling. 

    Args:
        samples (array-like): Array of samples on which the error analysis will
            be performed. 

                `samples.shape = (chain_length, batch_size, dim)`,

            where `dim` is the dimensionality of the target distribution.
        num_blocks (int): Number of blocks to use in blocked jackknife
            resampling.

    Returns:
        means (list): List containing the average value of each `component`
            in samples. For example, if the target distribution is
            two-dimensional, means would be `[x_avg, y_avg]`. 
        errs (list): List containing the standard error of each `component`
            in samples.

    NOTE:
        As an example, if `samples.shape = (1e4, 128, 2)` and 
        `num_blocks = None`, then `chain_length = 1e4`, `batch_size = 128` and
        `dim = 2`.

        We want to calculate the average and error for each component in `dim`,
        in this case we want `x_avg, x_err` and `y_avg, y_err`.

        This is done by first calculating the values `x_avg`, and `x_err` and
        then repeating the same calculation for `y_avg`, and `y_err`.

        For simplicity, we describe the calculation of `x_avg` and `x_err`.

        Let `samplesT = samples.transpose((-1, 1, 0))` with 
        `samplesT.shape = (2, 128, 1e4)`.

        Define `X, Y = samplesT` so that `X.shape = Y.shape = (128, 1e4)`.

        Then

        ```
        num_blocks = X.shape[1] // 50   # 1e4 // 50 = 200
        for x in X:
            x_rs = block_resampling(x, num_blocks)  # x_rs.shape = (200, 9950)
            x_rs_mean = x_rs.mean()
            means, errs = [], []
            for block in x_rs:
                block_mean = block.mean()
                denom = (num_blocks - 1) * num_blocks
                err = np.sqrt(np.sum((block_mean - x_rs_mean)**2) / denom)
                means.append(block_mean)
                errs.append(err)
        ```
    """
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)

    if num_blocks is None:
        num_blocks = samples.shape[0] // 50

    denom = (num_blocks - 1) * num_blocks

    def mean_(x):
        return np.mean(x, dtype=np.float64)

    def err_(x, xm):
        return np.sqrt(np.sum((mean_(x) - xm) ** 2) / denom)

    samplesT = samples.transpose((-1, 1, 0))
    # samplesT.shape = (dim, batch_size, chain_length)

    means_arr = []
    errs_arr = []
    for component in samplesT:  # loop over dimensionality (e.g. x, y, z)
        m_arr = []
        e_arr = []
        for x in component:  # loop over samples in batch
            x_rs = block_resampling(x, num_blocks)
            m_arr.extend([mean_(xb) for xb in x_rs])
            e_arr.extend([sem(xb) for xb in x_rs])
            #  m_rs = [mean_(xb) for xb in x_rs]
            #  e_rs = [err_(xb, x_rs_mean) for xb in x_rs]
        means_arr.append(m_arr)
        errs_arr.append(e_arr)

    means = np.array(means_arr).mean(axis=1, dtype=np.float64)
    errs = np.array(errs_arr).mean(axis=1, dtype=np.float64)

    return means, errs


def write_means(samples, num_blocks, means, errs, means_file):
    with open(means_file, 'a') as f:
        _ = f.write(f'samples.shape: {samples.shape}\n')
        _ = f.write(
            f'Using blocked jackknife resampling, with {num_blocks} blocks.\n'
        )
        _ = f.write('Component averages:\n')
        for mean, err in zip(means, errs):
            _ = f.write(f'  {mean:.5g} +/- {err:.5g}\n')

        _ = f.write(80 * '-' + '\n')


def _pickle_dump(data, out_file, name=None):
    if name is None:
        name = 'data'

    io.log(f'Saving {name} to {out_file}')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


def save_inference_data(samples, px, out_dir):
    if not isinstance(px, np.ndarray):
        px = np.array(px)
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)

    samples_out_file = os.path.join(out_dir, f'samples_out.pkl')
    px_out_file = os.path.join(out_dir, 'px_out.pkl')
    _pickle_dump(samples, samples_out_file, name='samples')
    _pickle_dump(px, px_out_file, name='probs')

    num_blocks = samples.shape[0] // 50
    means, errs = error_analysis(samples, num_blocks)
    num_blocks1 = 50
    means1, errs1 = error_analysis(samples, num_blocks1)
    samplesT = samples.transpose((-1, 1, 0))
    m_arr = []
    e_arr = []
    for component in samplesT:
        m_arr.append(np.mean(component, dtype=np.float64))
        e_arr.append(np.mean(sem(component), dtype=np.float64))

    m_arr = np.array(m_arr)
    e_arr = np.array(e_arr)

    means_file = os.path.join(out_dir, 'means.txt')
    write_means(samples, num_blocks, means, errs, means_file)
    write_means(samples, 50, means1, errs1, means_file)
    write_means(samples, 0, m_arr, e_arr, means_file)


def inference(runner, run_logger, **kwargs):
    run_steps = kwargs.get('run_steps', 5000)
    nw = kwargs.get('net_weights', [1., 1., 1.])
    beta = kwargs.get('beta', 1.)
    eps = kwargs.get('eps', None)
    if eps is None:
        eps = runner.eps
        kwargs['eps'] = eps

    run_str = run_logger._get_run_str(**kwargs)
    kwargs['run_str'] = run_str

    args = (nw, run_steps, eps, beta)

    if run_logger.existing_run(run_str):
        _log_inference_header(*args, existing=True)
    else:
        _log_inference_header(*args, existing=False)
        run_logger.reset(**kwargs)  # reset run_logger to prepare for new run

        t0 = time.time()

        runner.run(**kwargs)        # run inference and log time spent

        run_time = time.time() - t0
        io.log(SEP_STR
               + f'\nTook: {run_time:.4g}s to complete run.\n'
               + SEP_STR)

        samples_out = np.array(run_logger.samples_arr)
        px_out = np.array(run_logger.px_arr)
        out_dir = run_logger.run_dir

        save_inference_data(samples_out, px_out, out_dir)

        if HAS_MATPLOTLIB:
            log_dir = os.path.dirname(run_logger.runs_dir)
            distribution = recreate_distribution(log_dir)

            figs_root_dir = os.path.join(log_dir, 'figures')
            basename = os.path.basename(run_logger.run_dir)
            figs_dir = os.path.join(figs_root_dir, basename)

            io.check_else_make_dir(figs_root_dir)
            io.check_else_make_dir(figs_dir)

            plot_kwargs = {
                'out_file': os.path.join(figs_dir, 'single_l2hmc_chain.pdf'),
                'fill': False,
                'ellipse': False,
                'ls': '-',
                'axis_scale': 'scaled',
            }

            _ = _gmm_plot(distribution, samples_out[:, 0], **plot_kwargs)
            #  try:
            #      _gmm_plot3d(distribution, samples_out[:, 0], **plot_kwargs)
            #  except:
            #      import pudb; pudb.set_trace()

            plot_kwargs = {
                'nrows': 3,
                'ncols': 3,
                'num_points': 1000,
                'ellipse': False,
                'out_file': os.path.join(figs_dir, 'inference_plot.pdf'),
                'axis_scale': 'equal',
            }
            _, _ = gmm_plot(distribution, samples_out, **plot_kwargs)

    return runner, run_logger


def main(kwargs):
    params_file = kwargs.get('params_file', None)
    params = load_params(params_file)

    condition1 = not params['using_hvd']
    condition2 = params['using_hvd'] and hvd.rank() == 0
    is_chief = condition1 or condition2

    if not is_chief:
        return

    checkpoint_dir = os.path.join(params['log_dir'], 'checkpoints/')
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    config, params = create_config(params)
    sess = tf.Session(config=config)
    saver = tf.train.import_meta_graph(f'{checkpoint_file}.meta')
    saver.restore(sess, checkpoint_file)

    run_ops = tf.get_collection('run_ops')
    inputs = tf.get_collection('inputs')

    scale_weight = kwargs.get('scale_weight', 1.)
    translation_weight = kwargs.get('translation_weight', 1.)
    transformation_weight = kwargs.get('transformation_weight', 1.)
    net_weights = [scale_weight, translation_weight, transformation_weight]

    beta_inference = kwargs.get('beta_inference', None)
    beta_final = params.get('beta_final', None)
    beta = beta_final if beta_inference is None else beta_inference

    run_logger = RunLogger(params, inputs, run_ops,
                           model_type='gmm_model',
                           save_lf_data=False)

    runner = GaussianMixtureModelRunner(sess, params,
                                        inputs, run_ops,
                                        run_logger)

    inference_kwargs = {
        'run_steps': kwargs.get('run_steps', 5000),
        'net_weights': net_weights,
        'beta': beta,
        'eps': kwargs.get('eps', None),
    }

    runner, run_logger = inference(runner, run_logger, **inference_kwargs)


if __name__ == '__main__':
    args = parse_inference_args()

    t0 = time.time()
    log_file = 'output_dirs.txt'
    FLAGS = args.__dict__

    main(FLAGS)

    io.log('\n\n' + SEP_STR)
    io.log(f'Time to complete: {time.time() - t0:.4g}')





