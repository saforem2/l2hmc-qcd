"""
gauge_model_inference.py

Runs inference using the trained L2HMC sampler contained in a saved model.

This is done by reading in the location of the saved model from a .txt file
containing the location of the checkpoint directory.

 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  NOTE:
 ------------------------------------------------------------
   If `--plot_lf` CLI argument passed, create the
   following plots:

     * The metric distance observed between individual
       leapfrog steps and complete molecular dynamics
       updates.

     * The determinant of the Jacobian for each leapfrog
       step and the sum of the determinant of the Jacobian
       (sumlogdet) for each MD update.
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author: Sam Foreman (github: @saforem2)
Date: 07/08/2019
"""
from __future__ import absolute_import, division, print_function

import os
import time
import datetime

from config import HAS_HOROVOD

#  from update import set_precision
from runners.runner import Runner
from plotters.leapfrog_plotters import LeapfrogPlotter
from plotters.gauge_model_plotter import EnergyPlotter, GaugeModelPlotter

import tensorflow as tf
import numpy as np

import utils.file_io as io
import inference.utils as utils

from models.gauge_model import GaugeModel
from loggers.run_logger import RunLogger
from loggers.summary_utils import create_summaries
from inference.gauge_inference_utils import (_log_inference_header,
                                             create_config, inference_setup,
                                             load_params, log_plaq_diffs,
                                             parse_flags, SEP_STR)

from utils.parse_inference_args import parse_args as parse_inference_args

#  from plotters.plot_utils import plot_plaq_diffs_vs_net_weights
#  from tensorflow.core.protobuf import rewriter_config_pb2

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
    params['log_dir'] = FLAGS.log_dir
    params['loop_net_weights'] = False
    params['loop_transl_weights'] = False
    params['plot_lf'] = False

    figs_dir = os.path.join(params['log_dir'], 'figures')
    io.check_else_make_dir(figs_dir)

    io.log(SEP_STR)
    io.log('HMC PARAMETERS:')
    for key, val in params.items():
        io.log(f'  {key}: {val}')
    io.log(SEP_STR)

    config, params = create_config(params)
    tf.reset_default_graph()

    model = GaugeModel(params=params)

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    run_summaries_dir = os.path.join(model.log_dir, 'summaries', 'run')
    io.check_else_make_dir(run_summaries_dir)
    _, _ = create_summaries(model, run_summaries_dir, training=False)
    run_logger = RunLogger(params, model_type='GaugeModel', save_lf_data=False)
    plotter = GaugeModelPlotter(params, run_logger.figs_dir)
    energy_plotter = EnergyPlotter(params, run_logger.figs_dir)

    # ----------------------------------------------------------
    # Get keyword arguments to be passed to `inference` method
    # ----------------------------------------------------------
    inference_kwargs = inference_setup(params)

    # ----------------------------------------------------------
    # Create initial samples to be used at start of inference
    # ----------------------------------------------------------
    init_method = getattr(FLAGS, 'samples_init', 'random')
    seed = getattr(FLAGS, 'global_seed', 0)
    tf.random.set_random_seed(seed)
    np.random.seed(seed)
    samples_init = utils.init_gauge_samples(params, init_method)
    inference_kwargs['samples'] = samples_init

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
    run_steps = kwargs.get('run_steps', 5000)
    nw = kwargs.get('net_weights', [1., 1., 1.])
    beta = kwargs.get('beta', 5.)
    eps = kwargs.get('eps', None)
    if eps is None:
        eps = runner.eps
        kwargs['eps'] = eps

    #  run_str = run_logger._get_run_str(**kwargs)
    #  kwargs['run_str'] = run_str

    args = (nw, run_steps, eps, beta)

    #  if run_logger.existing_run(run_str):
    #      _log_inference_header(*args, existing=True)
    #      now = datetime.datetime.now()
    #      hour_str = now.strftime('%H%M')
    #      run_str += f'_{hour_str}'
    #      kwargs['run_str'] = run_str

    #  else:
    _log_inference_header(*args, existing=False)
    run_logger.reset(**kwargs)
    t0 = time.time()

    # -------------------
    #   RUN INFERENCE
    # -------------------
    runner.run(**kwargs)

    run_time = time.time() - t0
    io.log(SEP_STR + f'\nTook: {run_time}s to complete run.\n' + SEP_STR)

    # -----------------------------------------------------------
    # PLOT ALL LATTICE OBSERVABLES AND RETURN THE AVG. PLAQ DIFF
    # -----------------------------------------------------------
    kwargs['run_str'] = run_logger._run_str
    avg_plaq_diff = plotter.plot_observables(run_logger.run_data, **kwargs)
    log_plaq_diffs(run_logger,
                   kwargs['net_weights'],
                   avg_plaq_diff)

    tf_data = energy_plotter.plot_energies(run_logger.energy_dict,
                                           out_dir='tf', **kwargs)
    np_data = energy_plotter.plot_energies(run_logger.energy_dict_np,
                                           out_dir='np', **kwargs)
    diff_data = energy_plotter.plot_energies(run_logger.energies_diffs_dict,
                                             out_dir='tf-np', **kwargs)

    if kwargs.get('plot_lf', False):
        lf_plotter = LeapfrogPlotter(plotter.out_dir, run_logger)
        batch_size = runner.params.get('batch_size', 20)
        lf_plotter.make_plots(run_logger.run_dir,
                              batch_size=batch_size)

    energy_data = {
        'tf_data': tf_data,
        'np_data': np_data,
        'diff_data': diff_data
    }

    run_logger.save_data(energy_data, 'energy_plots_data.pkl')

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
    scale_weight = kwargs.get('scale_weight', 1.)
    translation_weight = kwargs.get('translation_weight', 1.)
    transformation_weight = kwargs.get('transformation_weight', 1.)
    net_weights = [scale_weight, translation_weight, transformation_weight]

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
    global_seed = kwargs.get('global_seed', 0)
    tf.random.set_random_seed(global_seed)
    np.random.seed(global_seed)
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
        'samples': samples_init,
        'net_weights': net_weights,
        'global_seed': global_seed,
        'run_steps': kwargs.get('run_steps', 5000),
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

    main(FLAGS)

    io.log('\n\n' + SEP_STR)
    io.log(f'Time to complete: {time.time() - t0:.4g}')
