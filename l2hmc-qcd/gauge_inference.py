"""
gauge_model_inference.py

Runs inference using the trained L2HMC sampler contained in a saved model.

This is done by specifying a `checkpoint_dir` containing the saved `tf.Graph`,
which is then restored and used for running inference.

Author: Sam Foreman (github: @saforem2)
Date: 07/08/2019
"""
from __future__ import absolute_import, division, print_function

import os
import time
import pickle

import numpy as np
import tensorflow as tf

import utils.file_io as io
import inference.utils as utils

from config import HAS_HOROVOD, NetWeights
from seed_dict import seeds
from runners.runner import Runner
from models.gauge_model import GaugeModel
from utils.parse_inference_args import parse_args as parse_inference_args
from inference.gauge_inference_utils import (_log_inference_header,
                                             create_config, load_params,
                                             log_plaq_diffs, parse_flags)
from loggers.run_logger import RunLogger
from loggers.summary_utils import create_summaries
from plotters.energy_plotter import EnergyPlotter
from plotters.plot_observables import plot_autocorrs, plot_charges
from plotters.leapfrog_plotters import LeapfrogPlotter
from plotters.gauge_model_plotter import GaugeModelPlotter
from gauge_inference_np import inference_plots

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if HAS_HOROVOD:
    import horovod.tensorflow as hvd  # pylint: disable=import-error

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)


def load_pkl(pkl_file):
    """Load from `.pkl` file."""
    try:
        with open(pkl_file, 'rb') as f:  # pylint: disable=invalid-name
            obj = pickle.load(f)
        return obj
    except FileNotFoundError:
        io.log(f'Unable to load from {pkl_file}.')


def run_hmc(FLAGS, log_file=None):
    """Run inference using generic HMC."""
    condition1 = not FLAGS.horovod
    condition2 = FLAGS.horovod and hvd.rank() == 0
    is_chief = condition1 or condition2
    if not is_chief:
        return -1

    FLAGS.hmc = True
    FLAGS.log_dir = io.create_log_dir(FLAGS, log_file=log_file)

    params = parse_flags(FLAGS)
    params['hmc'] = True
    params['use_bn'] = False
    params['plot_lf'] = False
    params['log_dir'] = FLAGS.log_dir

    figs_dir = os.path.join(params['log_dir'], 'figures')
    io.check_else_make_dir(figs_dir)

    io.log('\n\nHMC PARAMETERS:\n')
    for key, val in params.items():
        io.log(f'  {key}: {val}')

    config, params = create_config(params)
    tf.reset_default_graph()

    model = GaugeModel(params=params)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    run_summaries_dir = os.path.join(model.log_dir, 'summaries', 'run')
    io.check_else_make_dir(run_summaries_dir)
    _, _ = create_summaries(model, run_summaries_dir, training=False)

    run_logger = RunLogger(params=params,
                           save_lf_data=False,
                           model_type='GaugeModel')

    args = (params, run_logger.figs_dir)
    plotter = GaugeModelPlotter(*args)
    energy_plotter = EnergyPlotter(*args)

    # ----------------------------------------------------------
    # Get keyword arguments to be passed to `inference` method
    # ----------------------------------------------------------
    beta = params.get('beta_inference', None)
    if beta is None:
        beta = model.beta_final

    xsw = params.get('x_scale_weight', 1.)
    xtlw = params.get('x_translation_weight', 1.)
    xtfw = params.get('x_transformation_weight', 1.)
    vsw = params.get('v_scale_weight', 1.)
    vtlw = params.get('v_translation_weight', 1.)
    vtfw = params.get('v_transformation_weight', 1.)
    params['net_weights'] = NetWeights(x_scale=xsw,
                                       x_translation=xtlw,
                                       x_transformation=xtfw,
                                       v_scale=vsw,
                                       v_translation=vtlw,
                                       v_transformation=vtfw)

    # ----------------------------------------------------------
    # Create initial samples to be used at start of inference
    # ----------------------------------------------------------
    init_method = getattr(FLAGS, 'samples_init', 'random')
    #  seed = getattr(FLAGS, 'global_seed', 0)
    tf.random.set_random_seed(seeds['inference_tf'])
    np.random.seed(seeds['inference_np'])
    samples_init = utils.init_gauge_samples(params, init_method)
    inference_kwargs = {
        'samples': samples_init,
    }
    #  inference_kwargs['samples'] = samples_init

    # --------------------------------------
    # Create GaugeModelRunner for inference
    # --------------------------------------
    runner = Runner(sess, params, run_logger, model_type='GaugdeModel')

    # ---------------
    # run inference
    # ---------------
    runner, run_logger = inference(runner,
                                   run_logger,
                                   plotter,
                                   energy_plotter,
                                   **inference_kwargs)

    return -1


def build_run_data(data):
    """Build `run_data` dictionary from `data`."""
    keys = ['actions', 'charges', 'plaqs', 'dx', 'px', 'accept_prob']
    run_data = {k: np.array(data.get(k, None)) for k in keys}

    return run_data


def make_plots(runner, run_logger, plotter, energy_plotter, **kwargs):
    """Make all inference plots from inference run."""
    kwargs['run_str'] = run_logger._run_str
    apd, pkwds = plotter.plot_observables(run_logger.run_data, **kwargs)
    nw = kwargs.get('net_weights', NetWeights(1., 1., 1., 1., 1., 1.))

    log_plaq_diffs(run_logger, nw, apd)

    title = pkwds['title']
    qarr = np.array(run_logger.run_data['charges']).T
    qarr_int = np.around(qarr)

    out_file = os.path.join(plotter.out_dir, 'charges_grid.png')
    fig, ax = plot_charges(qarr, out_file, title=title, nrows=4)

    out_file = os.path.join(plotter.out_dir, 'charges_autocorr_grid.png')
    fig, ax = plot_autocorrs(qarr_int, out_file=out_file, title=title, nrows=4)

    io.save_dict(seeds, run_logger.run_dir, 'seeds')

    tf_data = energy_plotter.plot_energies(run_logger.energy_dict,
                                           out_dir='tf', **kwargs)
    energy_data = {
        'tf_data': tf_data,
    }

    run_logger.save_data(energy_data, 'energy_plots_data.pkl')

    if kwargs.get('plot_lf', False):
        lf_plotter = LeapfrogPlotter(plotter.out_dir, run_logger)
        batch_size = runner.params.get('batch_size', 20)
        lf_plotter.make_plots(run_logger.run_dir,
                              batch_size=batch_size)

    return runner, run_logger


def run_inference(runner, run_logger, **kwargs):
    """Run inference."""
    eps = kwargs.get('eps', None)
    beta = kwargs.get('beta', 5.)
    run_steps = kwargs.get('run_steps', 5000)
    skip_existing = kwargs.get('skip_existing', False)
    net_weights = kwargs.get('net_weights', NetWeights(1., 1., 1., 1., 1., 1.))
    if eps is None:
        eps = runner.eps
        kwargs['eps'] = eps

    existing = run_logger.reset(**kwargs)

    _log_inference_header(net_weights, run_steps, eps, beta, existing=existing)
    for key, val in kwargs.items():  # pylint: disable=redefined-outer-name
        io.log(f'{key}: {val}')

    if existing and skip_existing:
        return runner, run_logger, kwargs

    t0 = time.time()
    runner.run(**kwargs)
    run_time = time.time() - t0
    io.log(80 * '-' + f'\nTook: {run_time}s to complete run.\n' + 80 * '-')
    io.log(80 * '-' + '\n')

    return runner, run_logger, kwargs


def _loop_net_weights(runner, run_logger, plotter, energy_plotter, **kwargs):
    """Perform inference for all 64 possible values of `net_weights`."""
    eps = kwargs.get('eps', None)
    net_weights_arr = [
        tuple(np.array(list(np.binary_repr(i, width=6)), dtype=int))
        for i in range(64)
    ]
    if eps is None:
        eps = runner.eps
        kwargs['eps'] = eps

    for net_weights in net_weights_arr:
        kwargs['net_weights'] = NetWeights(*net_weights)
        runner, run_logger, kwargs = run_inference(runner,
                                                   run_logger,
                                                   **kwargs)
        try:
            runner, run_logger = make_plots(runner, run_logger, plotter,
                                            energy_plotter, **kwargs)
        except (AttributeError, KeyError):  # inference_plots fails if no data
            continue

    return runner, run_logger


def inference(runner, run_logger, plotter, energy_plotter, **kwargs):
    """Perform an inference run, if it hasn't been ran previously.

    Args:
        runner: `Runner` object, responsible for performing inference.
        run_logger: RunLogger object, responsible for running `tf.summary`
            operations and accumulating/saving run statistics.
        plotter: GaugeModelPlotter object responsible for plotting lattice
            observables from inference run.

    NOTE: If inference hasn't been run previously with the param values passed,
        return `avg_plaq_diff`, i.e. the average value of the difference
        between the expected and observed value of the average plaquette.

    Returns:
        avg_plaq_diff: If run hasn't been completed previously, else None
    """
    nw_loop = kwargs.get('loop_net_weights', False)
    if nw_loop:
        _loop_net_weights(runner, run_logger,
                          plotter, energy_plotter, **kwargs)
    else:
        runner, run_logger, kwargs = run_inference(runner,
                                                   run_logger,
                                                   **kwargs)
        runner, run_logger = make_plots(runner, run_logger, plotter,
                                        energy_plotter, **kwargs)

    with open(os.path.join(run_logger.log_dir, 'parameters.pkl'), 'rb') as f:
        params = pickle.load(f)
    with open(os.path.join(run_logger.run_dir, 'run_params.pkl'), 'rb') as f:
        run_params = pickle.load(f)
    run_data = build_run_data(run_logger.run_data)
    energy_data = {k: np.array(v) for k, v in run_logger.energy_dict.items()}
    _, _ = inference_plots(run_data, energy_data,
                           params, run_params, runs_np=False)

    return runner, run_logger

# pylint: disable=too-many-locals
def main(kwargs):
    """Perform inference using saved model.

    NOTE:
        [1.] We want to restrict all communication (file I/O) to only be
             performed on rank 0 (i.e. `is_chief`) so there are two cases:
                1. We're using Horovod, so we have to check hvd.rank()
                    explicitly.  
                2. We're not using Horovod, in which case `is_chief` 
                    is always True.
        [2.] We are only interested in the command line arguments that were
             passed to `inference.py` (i.e. those contained in kwargs).
    """
    log_dir = kwargs.get('log_dir', None)
    if log_dir is not None:
        params = load_pkl(os.path.join(log_dir, 'parameters.pkl'))
    #  params_file = kwargs.get('params_file', None)
    #  params = load_params(params_file)  # load params used during training

    # NOTE: [1.]
    condition1 = not params['using_hvd']
    condition2 = params['using_hvd'] and hvd.rank() == 0
    is_chief = condition1 or condition2
    if not is_chief:
        return

    # --------------------------------------------------------
    # locate `checkpoint_dir` containing `checkpoint_file`
    # --------------------------------------------------------
    checkpoint_dir = os.path.join(params['log_dir'], 'checkpoints/')
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    # --------------------------------------------------------------
    # load meta graph containing saved model from `checkpoint_file`
    # --------------------------------------------------------------
    config, params = create_config(params)
    sess = tf.Session(config=config)
    saver = tf.train.import_meta_graph(f'{checkpoint_file}.meta')
    saver.restore(sess, checkpoint_file)

    # ---------------------------------------------------
    # setup the step size `eps` (if using custom value)
    # ---------------------------------------------------
    eps = kwargs.get('eps', None)
    if eps is not None:
        io.log(f'`eps` is not None: {eps:.4g}')
        utils.set_eps(sess, eps)

    # -------------------
    # setup net_weights
    # -------------------
    xsw = kwargs.get('x_scale_weight', 1.)
    xtlw = kwargs.get('x_translation_weight', 1.)
    xtfw = kwargs.get('x_transformation_weight', 1.)
    vsw = kwargs.get('v_scale_weight', 1.)
    vtlw = kwargs.get('v_translation_weight', 1.)
    vtfw = kwargs.get('v_transformation_weight', 1.)
    net_weights = NetWeights(x_scale=xsw,
                             x_translation=xtlw,
                             x_transformation=xtfw,
                             v_scale=vsw,
                             v_translation=vtlw,
                             v_transformation=vtfw)

    # -------------
    # setup beta
    # -------------
    beta_inference = kwargs.get('beta_inference', None)
    beta_final = params.get('beta_final', None)
    beta = beta_final if beta_inference is None else beta_inference

    # -------------------------------------------------
    # setup initial samples to be used for inference
    # -------------------------------------------------
    init_method = kwargs.get('samples_init', 'random')
    tf.random.set_random_seed(seeds['inference_tf'])
    np.random.seed(seeds['inference_np'])
    samples_init = utils.init_gauge_samples(params, init_method)

    # -----------------------------------------------------------------------
    # Create `RunLogger`, `Runner`, `GaugeModelPlotter` and `EnergyPlotter`
    # -----------------------------------------------------------------------
    run_logger = RunLogger(params, model_type='GaugeModel', save_lf_data=False)
    runner = Runner(sess, params, logger=run_logger, model_type='GaugeModel')
    plotter = GaugeModelPlotter(params, run_logger.figs_dir)
    energy_plotter = EnergyPlotter(params, run_logger.figs_dir)

    inference_kwargs = {
        'eps': eps,
        'beta': beta,
        'init': init_method,
        'samples': samples_init,
        'net_weights': net_weights,
        'run_steps': kwargs.get('run_steps', 5000),
        'save_samples': kwargs.get('save_samples', False),
        'loop_net_weights': kwargs.get('loop_net_weights', False),
        'skip_existing': kwargs.get('skip_existing', False),
    }

    runner, run_logger = inference(runner,
                                   run_logger,
                                   plotter,
                                   energy_plotter,
                                   **inference_kwargs)


if __name__ == '__main__':
    ARGS = parse_inference_args()
    LOG_FILE = 'output_dirs.txt'
    FLAGS = ARGS.__dict__
    io.log(80 * '-' + '\n' + 'INFERENCE FLAGS:\n')
    for key, val in FLAGS.items():
        io.log(f'{key}: {val}\n')

    main(FLAGS)
