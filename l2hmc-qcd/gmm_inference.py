"""
gmm_inference.py

Run inference using a trained (L2HMC) model.

Author: Sam Foreman (github: @saforem2)
Date: 09/18/2019
"""
from __future__ import absolute_import, division, print_function

import os
import time

from config import HAS_HOROVOD, HAS_MATPLOTLIB
from loggers.run_logger import RunLogger
from runners.gmm_runner import GaussianMixtureModelRunner
from plotters.plot_utils import _gmm_plot, gmm_plot

import numpy as np
import tensorflow as tf

import utils.file_io as io

from utils.parse_inference_args import parse_args as parse_inference_args
from gauge_inference import _log_inference_header
from inference.gmm_inference_utils import (create_config, load_params,
                                           recreate_distribution,
                                           save_inference_data)

if HAS_HOROVOD:
    import horovod.tensorflow as hvd

#  from loggers.summary_utils import create_summaries
#  from scipy.stats import sem
#  from utils.data_utils import block_resampling
#  from utils.distributions import GMM
#  if HAS_MATPLOTLIB:
#      import matplotlib.pyplot as plt
#  if HAS_MEMORY_PROFILER:
#      import memory_profiler

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)


SEP_STR = 80 * '-'


'''
def set_num_steps(sess, num_steps, run_ops, inputs, graph=None):
    """Seet the number of leapfrog steps to `num_steps`. """
    if graph is None:
        graph = tf.get_default_graph()

    num_steps_setter = graph.get_operation_by_name('init/num_steps_setter')
    num_steps_tensor = [i for i in run_
'''


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


def inference(runner, run_logger, **kwargs):
    run_steps = kwargs.get('run_steps', 5000)     # num. of accept/reject steps
    nw = kwargs.get('net_weights', [1., 1., 1.])  # custom net_weights
    bs_iters = kwargs.get('bs_iters', 200)  # num. bootstrap replicates
    beta = kwargs.get('beta', 1.)  # custom value to use for `beta`
    eps = kwargs.get('eps', None)  # custom value to use for the step size
    skip_acl = kwargs.get('skip_acl', False)  # calc autocorrelation or not
    ignore_first = kwargs.get('ignore_first', 0.1)  # % to ignore for therm.
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

        log_dir = os.path.dirname(run_logger.runs_dir)
        run_dir = run_logger.run_dir
        basename = os.path.basename(run_dir)
        figs_dir = os.path.join(log_dir, 'figures')
        fig_dir = os.path.join(figs_dir, basename)
        _ = [io.check_else_make_dir(d) for d in [figs_dir, fig_dir]]

        args = (samples_out, px_out, run_dir, fig_dir)
        kwargs = {
            'skip_acl': skip_acl,
            'bs_iters': bs_iters,
            'ignore_first': ignore_first,
        }
        save_inference_data(*args, **kwargs)

        if HAS_MATPLOTLIB:
            log_dir = os.path.dirname(run_logger.runs_dir)
            distribution = recreate_distribution(log_dir)

            title = (r"""$\varepsilon = $""" + f'{eps:.3g} '
                     + r"""$\langle p_{x} \rangle= $"""
                     + f'{np.mean(px_out[:, -1]):.3g}')

            plot_kwargs = {
                'out_file': os.path.join(fig_dir, 'single_chain.pdf'),
                'fill': False,
                'ellipse': False,
                'ls': '-',
                'axis_scale': 'scaled',
                'title': title,
                'num_contours': 4
            }

            _ = _gmm_plot(distribution, samples_out[:, -1], **plot_kwargs)

            plot_kwargs = {
                'nrows': 2,
                'ncols': 2,
                'ellipse': False,
                'out_file': os.path.join(fig_dir, 'inference_plot.pdf'),
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
    if os.path.isdir(checkpoint_dir):
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    else:
        log_dir = os.path.dirname(params_file)
        checkpoint_dir = os.path.join(log_dir, 'checkpoints/')
        if os.path.isdir(checkpoint_dir):
            checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
            params['log_dir'] = log_dir

    config, params = create_config(params)
    sess = tf.Session(config=config)
    saver = tf.train.import_meta_graph(f'{checkpoint_file}.meta')
    saver.restore(sess, checkpoint_file)

    run_ops = tf.get_collection('run_ops')
    inputs = tf.get_collection('inputs')

    eps = kwargs.get('eps', None)
    if eps is not None:
        graph = tf.get_default_graph()
        set_eps(sess, eps, run_ops, inputs, graph)

    scale_weight = kwargs.get('scale_weight', 1.)
    translation_weight = kwargs.get('translation_weight', 1.)
    transformation_weight = kwargs.get('transformation_weight', 1.)
    net_weights = [scale_weight, translation_weight, transformation_weight]

    beta_inference = kwargs.get('beta_inference', None)
    beta_final = params.get('beta_final', None)
    beta = beta_final if beta_inference is None else beta_inference

    # XXX XXX Try changing how samples are initialized below. XXX XXX
    samples_size = (params['batch_size'], params['x_dim'])
    tmp = samples_size[0] * samples_size[1]
    samples_init = np.random.uniform(-1, 1, tmp).reshape(*samples_size)
    #  samples_init = 2 * np.random.rand(*(params['batch_size'],
    #                                      params['x_dim']))
    kwargs['samples'] = samples_init

    run_logger = RunLogger(params, inputs, run_ops,
                           model_type='gmm_model',
                           save_lf_data=False)

    runner = GaussianMixtureModelRunner(sess, params,
                                        inputs, run_ops,
                                        run_logger)

    # NUMBER OF BOOTSTRAP REPLICATIONS TO USE IN ERROR ANALYSIS
    bs_iters = kwargs.get('bootstrap_iters', 100)
    skip_acl = kwargs.get('skip_acl', False)
    #

    inference_kwargs = {
        'run_steps': kwargs.get('run_steps', 5000),
        'net_weights': net_weights,
        'beta': beta,
        'eps': eps,
        'bs_iters': bs_iters,
        'skip_acl': skip_acl,
    }

    runner, run_logger = inference(runner, run_logger, **inference_kwargs)

    run_hmc = kwargs.get('run_hmc', False)
    if run_hmc:
        io.log(80 * '-' + '\n')
        io.log(f"INFO: Running generic HMC for "
               f" {inference_kwargs['run_steps']} steps at "
               f" `beta = {inference_kwargs['beta']}` with a "
               f" step size `eps = {inference_kwargs['eps']}`.\n")

        hmc_inference_kwargs = {
            'run_steps': kwargs.get('run_steps', 5000),
            'net_weights': [0., 0., 0.],
            'beta': beta,
            'eps': eps,
            'bs_iters': bs_iters,
        }
        runner, run_logger = inference(runner,
                                       run_logger,
                                       **hmc_inference_kwargs)


if __name__ == '__main__':
    args = parse_inference_args()

    t0 = time.time()
    log_file = 'output_dirs.txt'
    FLAGS = args.__dict__

    main(FLAGS)

    io.log('\n\n' + SEP_STR)
    io.log(f'Time to complete: {time.time() - t0:.4g}')
