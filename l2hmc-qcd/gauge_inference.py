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

from config import HAS_HOROVOD, NetWeights
from seed_dict import seeds

import numpy as np
import tensorflow as tf

import utils.file_io as io
import inference.utils as utils

from runners.runner import Runner
from models.gauge_model import GaugeModel
from utils.parse_inference_args import parse_args as parse_inference_args
from inference.gauge_inference_utils import (_log_inference_header,
                                             create_config, load_params,
                                             log_plaq_diffs, parse_flags)
from loggers.run_logger import RunLogger
from loggers.summary_utils import create_summaries
from plotters.plot_observables import plot_charges, plot_autocorrs
from plotters.leapfrog_plotters import LeapfrogPlotter
from plotters.gauge_model_plotter import GaugeModelPlotter
from plotters.energy_plotter import EnergyPlotter

if HAS_HOROVOD:
    import horovod.tensorflow as hvd

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)


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
    eps = kwargs.get('eps', None)
    beta = kwargs.get('beta', 5.)
    run_steps = kwargs.get('run_steps', 5000)
    #  nw = kwargs.get('net_weights', [1., 1., 1.])
    nw = kwargs.get('net_weights', NetWeights(1., 1., 1., 1., 1., 1.))

    if eps is None:
        eps = runner.eps
        kwargs['eps'] = eps

    _log_inference_header(nw, run_steps, eps, beta, existing=False)

    # -----------------------
    #   RUN INFERENCE
    # -----------------------
    run_logger.reset(**kwargs)
    t0 = time.time()
    runner.run(**kwargs)
    # ------------------------

    run_time = time.time() - t0
    io.log(80 * '-' + f'\nTook: {run_time}s to complete run.\n' + 80 * '-')

    io.log(80 * '-' + '\n')
    for key, val in kwargs.items():
        io.log(f'{key}: {val}')
    # -----------------------------------------------------------
    # PLOT ALL LATTICE OBSERVABLES AND RETURN THE AVG. PLAQ DIFF
    # -----------------------------------------------------------
    kwargs['run_str'] = run_logger._run_str
    avg_plaq_diff, pkwds = plotter.plot_observables(run_logger.run_data,
                                                    **kwargs)
    log_plaq_diffs(run_logger, nw, avg_plaq_diff)

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
    params_file = kwargs.get('params_file', None)
    params = load_params(params_file)  # load params used during training

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
    #  scale_weight = kwargs.get('scale_weight', 1.)
    #  translation_weight = kwargs.get('translation_weight', 1.)
    #  transformation_weight = kwargs.get('transformation_weight', 1.)
    #  net_weights = [scale_weight, translation_weight, transformation_weight]
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
    }

    runner, run_logger = inference(runner,
                                   run_logger,
                                   plotter,
                                   energy_plotter,
                                   **inference_kwargs)


if __name__ == '__main__':
    args = parse_inference_args()

    t0 = time.time()
    log_file = 'output_dirs.txt'
    FLAGS = args.__dict__
    io.log(80 * '-' + '\n' + 'INFERENCE FLAGS:\n')
    for key, val in FLAGS.items():
        io.log(f'{key}: {val}\n')

    main(FLAGS)

    io.log('\n\n' + 80 * '-' + '\n'
           f'Time to complete: {time.time() - t0:.4g}')
