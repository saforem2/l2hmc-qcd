from __future__ import absolute_import, division, print_function

import os
import pickle

from config import HAS_HOROVOD, HAS_MATPLOTLIB
from update import set_precision
from plotters.plot_utils import plot_histogram, plot_acl_spectrum

import numpy as np
import tensorflow as tf

from scipy.stats import sem

import utils.file_io as io

from utils.data_utils import block_resampling
from utils.distributions import GMM

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt

if HAS_HOROVOD:
    import horovod.tensorflow as hvd

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)


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


def autocovariance(x, tau=0):
    dT, dN, dX = np.shape(x)
    xs = np.mean(x)
    s = 0.
    for t in range(dT - tau):
        x1 = x[t, :, :] - xs
        x2 = x[t+tau, :, :] - xs
        s += np.sum(x1 * x2) / dN
    return s / (dT - tau)


def acl_spectrum(x):
    n = x.shape[0]
    autocovs = np.array([autocovariance(x, tau=t) for t in range(n-1)])
    autocovs /= np.max(autocovs)

    return autocovs


def ESS(A):
    A = A * (A > 0.05)
    return 1. / (1. + 2 * np.sum(A[1:]))


def _bootstrap_replicate_1d(data, func):
    """Generate a bootstrap replicate data."""
    bs_sample = np.random.choice(data, len(data))

    return bs_sample, func(bs_sample)


def bootstrap_replicates_1d(data, func, num_replicates=1000):
    return [_bootstrap_replicate_1d(data, func) for _ in range(num_replicates)]


def bootstrap_resample(x, n=None):
    """Use bootstrap resampling to obtain (re) sampled elements of `x`.

    Args:
        x (array-like): Data to resample.
        n (int, optional): Length of resampled array, equal to len(x) if n is
            None.

    Returns:
        x_rs: The resampled array.
    """
    if n is None:
        n = len(x)

    resample_i = np.floor(np.random.rand(n)*len(x)).astype(int)
    x_rs = x[resample_i]

    return x_rs


def error_analysis(samples, n=None, bs_iters=500):
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)

    samplesT = samples.transpose((-1, 1, 0))
    samples_rs = []
    means_arr = []
    #  errs_arr = []
    for component in samplesT:
        x_arr = []
        for _ in range(bs_iters):
            x_rs = [bootstrap_resample(x, n) for x in component]
            x_arr.append(x_rs)
        samples_rs.append(x_arr)
        x_arr = np.array(x_arr)
        means = np.mean(x_arr, axis=-1, dtype=np.float64)
        means_arr.append(means)
        #  samples_rs.append(x_rs)

    samples_rs = np.array(samples_rs)

    means = np.mean(means_arr, axis=(1, 2), dtype=np.float64)
    errs = np.std(means_arr, axis=(1, 2), dtype=np.float64)
    #  means_ = samples_rs.mean(axis=-1, dtype=np.float64)
    #  errs = np.std(means_, axis=-1, dtype=np.float64)
    #  errs = sem(means_, axis=-1)
    #  means = np.mean(means_, axis=-1, dtype=np.float64)

    return means, errs, means_arr, samples_rs


def alt_error_analysis(samples, num_blocks=None):
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
        '''
        _ = f.write(
            f'Using bootstrap resampling, using {num_blocks} replicates '
            f'for each of the {samples.shape[1]} samples.\n'
        )
        '''
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


def save_inference_data(samples, px, run_dir, fig_dir,
                        acl=True, bs_iters=100):
    if not isinstance(px, np.ndarray):
        px = np.array(px)
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)

    samples_out_file = os.path.join(run_dir, f'samples_out.pkl')
    px_out_file = os.path.join(run_dir, 'px_out.pkl')
    _pickle_dump(samples, samples_out_file, name='samples')
    _pickle_dump(px, px_out_file, name='probs')

    warmup_steps = samples.shape[0] // 20
    io.log(f'INFO: Ignoring first {warmup_steps} steps for thermalization...')
    io.log(f'INFO: Using (naive) `np.mean` and `np.std`:')
    x_mean, y_mean = samples[warmup_steps:].mean(axis=(0, 1))
    x_std, y_std = samples[warmup_steps:].std(axis=(0, 1))
    io.log(f'x_mean: {x_mean:.3g} +/- {x_std:.3g}')
    io.log(f'y_mean: {y_mean:.3g} +/- {y_std:.3g}')
    means, errs, means_, samples_rs = error_analysis(samples[warmup_steps:],
                                                     n=None, bs_iters=bs_iters)

    means_strs = [f'mean: {i:.4g} +/- {j:.4g}' for i, j in zip(means, errs)]
    io.log(means_strs)
    means_file = os.path.join(run_dir, 'means.txt')

    write_means(samples, -1, means, errs, means_file)

    if acl:
        spectrum = acl_spectrum(samples)
        ess = ESS(spectrum)

        spectrum_file = os.path.join(run_dir, 'acl_spectrum.pkl')
        ess_file = os.path.join(run_dir, 'ess.txt')
        with open(spectrum_file, 'wb') as f:
            pickle.dump(spectrum, f)

        with open(ess_file, 'w') as f:
            _ = f.write(f'ESS: {ess}')

    out_files = [os.path.join(fig_dir, 'x_mean_histogram.pdf'),
                 os.path.join(fig_dir, 'y_mean_histogram.pdf')]
    xlabels = ['mean, x', 'mean, y']
    kwargs = {
        'bins': 32,
        'density': True,
        'stacked': True,
        'label': None,
        'out_file': None,
        'xlabel': None
    }
    if HAS_MATPLOTLIB:
        for idx, x in enumerate(means_):
            kwargs.update({
                'label': means_strs[idx],
                'out_file': out_files[idx],
                'xlabel': xlabels[idx]
            })
            fig, ax = plt.subplots()
            _ = plot_histogram(x.flatten(), ax=ax, **kwargs)

        if acl:
            '''
            nx = (spectrum.shape[0] + 1) // 10
            xaxis = 10 * np.arange(nx)
            fig, ax = plt.subplots()
            _ = ax.plot(xaxis, np.abs(spectrum[:nx]))
            _ = ax.set_xlabel('Gradient Computations')
            _ = ax.set_ylabel('Auto-correlation')
            out_file = os.path.join(fig_dir, 'autocorrelation_spectrum.pdf')
            io.log(f'Saving figure to: {out_file}.')
            _ = plt.savefig(out_file, bbox_inches='tight')
            '''
            kwargs = {
                'out_file': os.path.join(fig_dir,
                                         'autocorrelation_spectrum.pdf')
            }
            fig, ax = plot_acl_spectrum(spectrum, **kwargs)
