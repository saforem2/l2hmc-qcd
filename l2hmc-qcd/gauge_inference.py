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

from config import HAS_HOROVOD

#  from update import set_precision
from runners.gauge_model_runner import GaugeModelRunner
from models.gauge_model import GaugeModel
from loggers.run_logger import RunLogger
from loggers.summary_utils import create_summaries
from plotters.plot_utils import plot_plaq_diffs_vs_net_weights
from plotters.leapfrog_plotters import LeapfrogPlotter
from plotters.gauge_model_plotter import GaugeModelPlotter

import numpy as np
import tensorflow as tf

#  from tensorflow.core.protobuf import rewriter_config_pb2
import utils.file_io as io

from utils.parse_inference_args import parse_args as parse_inference_args
from inference.gauge_inference_utils import (_log_inference_header,
                                             create_config, inference_setup,
                                             initialize_uninitialized,
                                             load_params, log_plaq_diffs,
                                             parse_flags, SEP_STR)

if HAS_HOROVOD:
    import horovod.tensorflow as hvd

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)


def set_eps(sess, eps, run_ops, inputs, graph=None):
    if graph is None:
        graph = tf.get_default_graph()

    eps_setter = graph.get_operation_by_name('init/eps_setter')
    eps_tensor = [i for i in run_ops if 'eps' in i.name][0]
    eps_ph = [i for i in inputs if 'eps_ph' in i.name][0]

    eps_np = sess.run(eps_tensor)
    io.log(f'INFO: Original value of `eps`: {eps_np}')
    io.log(f'INFO: Setting `eps` to: {eps}.')
    sess.run(eps_setter, feed_dict={eps_ph: eps})
    eps_np = sess.run(eps_tensor)

    io.log(f'INFO: New value of `eps`: {eps_np}')


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

    run_ops = tf.get_collection('run_ops')
    inputs = tf.get_collection('inputs')
    run_summaries_dir = os.path.join(model.log_dir, 'summaries', 'run')
    io.check_else_make_dir(run_summaries_dir)
    _, _ = create_summaries(model, run_summaries_dir, training=False)
    run_logger = RunLogger(params, inputs, run_ops, save_lf_data=False)
    plotter = GaugeModelPlotter(params, run_logger.figs_dir)

    inference_dict = inference_setup(params)

    # --------------------------------------
    # Create GaugeModelRunner for inference
    # --------------------------------------
    runner = GaugeModelRunner(sess, params, inputs, run_ops, run_logger)
    run_inference(inference_dict, runner, run_logger, plotter)


def inference(runner, run_logger, plotter, **kwargs):
    """Perform an inference run, if it hasn't been ran previously.

    Args:
        runner: GaugeModelRunner object, responsible for performing inference.
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

    run_str = run_logger._get_run_str(**kwargs)
    kwargs['run_str'] = run_str

    args = (nw, run_steps, eps, beta)

    if run_logger.existing_run(run_str):
        _log_inference_header(*args, existing=True)

    else:
        _log_inference_header(*args, existing=False)
        run_logger.reset(**kwargs)
        t0 = time.time()

        runner.run(**kwargs)

        run_time = time.time() - t0
        io.log(SEP_STR + f'\nTook: {run_time}s to complete run.\n' + SEP_STR)

        # -----------------------------------------------------------
        # PLOT ALL LATTICE OBSERVABLES AND RETURN THE AVG. PLAQ DIFF
        # -----------------------------------------------------------
        avg_plaq_diff = plotter.plot_observables(run_logger.run_data, **kwargs)
        log_plaq_diffs(run_logger,
                       kwargs['net_weights'],
                       avg_plaq_diff)

        if kwargs.get('plot_lf', False):
            lf_plotter = LeapfrogPlotter(plotter.out_dir, run_logger)
            batch_size = runner.params.get('batch_size', 20)
            lf_plotter.make_plots(run_logger.run_dir,
                                  batch_size=batch_size)

    return runner, run_logger


def run_inference(runner, run_logger=None, plotter=None, **kwargs):
    """Run inference.

    Args:
        inference_dict: Dictionary containing parameters to use for inference.
        runner: GaugeModelRunner object that actually performs the inference.
        run_logger: RunLogger object that logs observables and other data
            generated during inference.
        plotter: GaugeModelPlotter object responsible for plotting observables
            generated during inference.
    """
    if plotter is None or run_logger is None:
        return

    args = (runner, run_logger, plotter)
    src = os.path.join(run_logger.log_dir, 'plaq_diffs_data.txt')
    if os.path.isfile(src):
        dst = os.path.join(run_logger.log_dir, 'plaq_diffs_data_orig.txt')
        os.rename(src, dst)

    #  if not kwargs['loop_net_weights']:
    #      inference(runner, run_logger, plotter, **kwargs)

    #  else:  # looping over different values of net_weights
    #      nw_arr = kwargs.get('net_weights', None)
    #      zero_weights, q_weights, t_weights, s_weights, stq_weights = nw_arr
    #
    #      kwargs.update({'net_weights': zero_weights})
    #      inference(*args, **kwargs)
    #
    #      for net_weights in q_weights:
    #          kwargs.update({'net_weights': net_weights})
    #          inference(*args, **kwargs)
    #
    #      for net_weights in t_weights:
    #          kwargs.update({'net_weights': net_weights})
    #          inference(*args, **kwargs)
    #
    #      for net_weights in s_weights:
    #          kwargs.update({'net_weights': net_weights})
    #          inference(*args, **kwargs)
    #
    #      kwargs.update({'net_weights': stq_weights})
    #      inference(*args, **kwargs)
    #
    #      #  log_mem_usage(run_logger, m_arr)
    #      plot_plaq_diffs_vs_net_weights(run_logger.log_dir)


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

    checkpoint_dir = os.path.join(params['log_dir'], 'checkpoints/')
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    config, params = create_config(params)
    sess = tf.Session(config=config)
    saver = tf.train.import_meta_graph(f'{checkpoint_file}.meta')
    saver.restore(sess, checkpoint_file)

    #  _ = initialize_uninitialized(sess)
    run_ops = tf.get_collection('run_ops')
    inputs = tf.get_collection('inputs')

    eps = kwargs.get('eps', None)
    if eps is not None:
        io.log(f'`eps` is not None: {eps:.4g}')
        graph = tf.get_default_graph()
        set_eps(sess, eps, run_ops, inputs, graph)

    scale_weight = kwargs.get('scale_weight', 1.)
    translation_weight = kwargs.get('translation_weight', 1.)
    transformation_weight = kwargs.get('transformation_weight', 1.)
    net_weights = [scale_weight, translation_weight, transformation_weight]

    beta_inference = kwargs.get('beta_inference', None)
    beta_final = params.get('beta_final', None)
    beta = beta_final if beta_inference is None else beta_inference

    x_dim = params['space_size'] * params['time_size'] * params['dim']
    samples_shape = (params['batch_size'], x_dim)
    init_method = kwargs.get('samples_init', 'random')
    if init_method == 'random':
        io.log(80 * '-' + '\n\n')
        io.log(f'Hit `random init`...')
        io.log(80 * '-' + '\n\n')
        tmp = samples_shape[0] * samples_shape[1]
        samples_init = np.random.uniform(-1, 1, tmp).reshape(*samples_shape)
    elif 'zero' in init_method:
        io.log(80 * '-' + '\n\n')
        io.log(f'Hit `zeros init`...')
        io.log(80 * '-' + '\n\n')
        samples_init = np.zeros(samples_init)
    elif 'ones' in init_method:
        io.log(80 * '-' + '\n\n')
        io.log(f'Hit `ones init`...')
        io.log(80 * '-' + '\n\n')
        samples_init = np.ones(samples_shape)

    run_logger = RunLogger(params, inputs, run_ops,
                           model_type='GaugeModel',
                           save_lf_data=False)

    runner = GaugeModelRunner(sess, params, inputs, run_ops, run_logger)
    plotter = GaugeModelPlotter(params, run_logger.figs_dir)

    inference_kwargs = {
        'run_steps': kwargs.get('run_steps', 5000),
        'net_weights': net_weights,
        'beta': beta,
        'eps': eps,
        'samples': samples_init
    }

    runner, run_logger = inference(runner,
                                   run_logger,
                                   plotter,
                                   **inference_kwargs)

    # NOTE: [2.]
    #  inference_dict = inference_setup(kwargs)
    #  if inference_dict['beta'] is None:
    #      inference_dict['beta'] = params['beta_final']

    #  params['eps'] = inference_dict.get('eps', None)
    #  eps = kwargs.get('eps', None)
    #  if eps is not None:
    #      graph = tf.get_default_graph()
    #      set_eps(sess, eps, run_ops, inputs, graph)
    #
    #  runner = GaugeModelRunner(sess, params, inputs, run_ops, run_logger)

    
    #  run_inference(runner, run_logger, plotter, **inference_kwargs)


if __name__ == '__main__':
    args = parse_inference_args()

    t0 = time.time()
    log_file = 'output_dirs.txt'
    FLAGS = args.__dict__
    #  kwargs = FLAGS.__dict__

    main(FLAGS)

    io.log('\n\n' + SEP_STR)
    io.log(f'Time to complete: {time.time() - t0:.4g}')
