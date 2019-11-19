"""
gauge_inference_utils.py

Collection of helper functions for running inference using a trained sampler on
the 2D U(1) lattice gauge model.

Author: Sam Foreman (github: @saforem2)
Date: 10/02/2019
"""
from __future__ import absolute_import, division, print_function

import os
import pickle

from config import HAS_HOROVOD, HAS_MATPLOTLIB, HAS_MEMORY_PROFILER, NP_FLOAT
from update import set_precision

import numpy as np
import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2

import utils.file_io as io

#  from runners.runner import GaugeModelRunner
#  from models.gauge_model import GaugeModel
#  from loggers.run_logger import RunLogger
#  from loggers.summary_utils import create_summaries
#  from plotters.gauge_model_plotter import GaugeModelPlotter

#  from utils.parse_inference_args import parse_args as parse_inference_args
#  from plotters.plot_utils import plot_plaq_diffs_vs_net_weights
#  from plotters.leapfrog_plotters import LeapfrogPlotter

if HAS_HOROVOD:
    import horovod.tensorflow as hvd

if HAS_MEMORY_PROFILER:
    import memory_profiler

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)

SEP_STR = 80 * '-'  # + '\n'


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run(
        [tf.is_variable_initialized(var) for var in global_vars]
    )
    not_initialized_vars = [
        v for (v, f) in zip(global_vars, is_not_initialized) if not f
    ]
    io.log([str(i.name) for i in not_initialized_vars])
    if len(not_initialized_vars) > 1:
        sess.run(tf.variables_initializer(not_initialized_vars))

    return not_initialized_vars


def create_config(params):
    """Helper method for creating a tf.ConfigProto object."""
    config = tf.ConfigProto(allow_soft_placement=True)
    if params['time_size'] > 8:
        off = rewriter_config_pb2.RewriterConfig.OFF
        config_attrs = config.graph_options.rewrite_options
        config_attrs.arithmetic_optimization = off

    if params['gpu']:
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
    try:
        kv_pairs = FLAGS.__dict__.items()
    except AttributeError:
        kv_pairs = FLAGS.items()

    params = {}
    for key, val in kv_pairs:
        params[key] = val

    params['log_dir'] = io.create_log_dir(FLAGS, log_file=log_file)
    params['summaries'] = not FLAGS.no_summaries
    if 'no_summaries' in params:
        del params['no_summaries']

    if FLAGS.save_steps is None and FLAGS.train_steps is not None:
        params['save_steps'] = params['train_steps'] // 4

    if FLAGS.float64:
        io.log(f'INFO: Setting floating point precision to `float64`.')
        set_precision('float64')

    return params


def load_params(params_pkl_file=None, log_file=None):
    if params_pkl_file is None:
        params_pkl_file = os.path.join(os.getcwd(), 'params.pkl')

    if os.path.isfile(params_pkl_file):
        with open(params_pkl_file, 'rb') as f:
            params = pickle.load(f)

    return params


def set_model_weights(model, dest='rand'):
    """Randomize model weights."""
    if dest == 'rand':
        io.log('Randomizing model weights...')
    elif 'zero' in dest:
        io.log(f'Zeroing model weights...')

    xnet = model.dynamics.x_fn
    vnet = model.dynamics.v_fn

    for xblock, vblock in zip(xnet.layers, vnet.layers):
        for xlayer, vlayer in zip(xblock.layers, vblock.layers):
            try:
                print(f'xlayer.name: {xlayer.name}')
                print(f'vlayer.name: {vlayer.name}')
                kx, bx = xlayer.get_weights()
                kv, bv = vlayer.get_weights()
                if dest == 'rand':
                    kx_new = np.random.randn(*kx.shape)
                    bx_new = np.random.randn(*bx.shape)
                    kv_new = np.random.randn(*kv.shape)
                    bv_new = np.random.randn(*bv.shape)
                elif 'zero' in dest:
                    kx_new = np.zeros(kx.shape)
                    bx_new = np.zeros(bx.shape)
                    kv_new = np.zeros(kv.shape)
                    bv_new = np.zeros(bv.shape)

                xlayer.set_weights([kx_new, bx_new])
                vlayer.set_weights([kv_new, bv_new])
            except ValueError:
                print(f'Unable to set weights for: {xlayer.name}')
                print(f'Unable to set weights for: {vlayer.name}')

    return model


def log_mem_usage(run_logger, m_arr):
    """Log memory usage."""
    m_pkl_file = os.path.join(run_logger.log_dir, 'memory_usage.pkl')
    m_txt_file = os.path.join(run_logger.log_dir, 'memory_usage.txt')
    with open(m_pkl_file, 'wb') as f:
        pickle.dump(m_arr, f)
    with open(m_txt_file, 'w') as f:
        for i in m_arr:
            _ = [f.write(f'{j}\n') for j in i]


def collect_mem_usage(m_arr=None):
    if HAS_MEMORY_PROFILER:
        usage = memory_profiler.memory_usage()
        if m_arr is not None:
            m_arr.append(usage)

            return m_arr

        return usage


def log_plaq_diffs(run_logger, net_weights, avg_plaq_diff):
    """Log the average values of the plaquette differences.

    NOTE: If inference was performed with either the `--loop_net_weights` 
          or `--loop_transl_weights` flags passed, we want to see how the
          difference between the observed and expected value of the average
          plaquette varies with different values of the net weights, so save
          this data to `.pkl` file and  plot the results.
    """
    try:
        pd_tup = [
            (nw, md) for nw, md in zip(net_weights, avg_plaq_diff)
        ]
        out_dir = run_logger.log_dir
    except TypeError:
        #  pd_tup = [(net_weights, avg_plaq_diff)]
        pd_tup = [*net_weights, avg_plaq_diff]
        out_dir = run_logger.run_dir

    output_arr = np.array(pd_tup)

    #  pd_pkl_file = os.path.join(out_dir, 'plaq_diffs_data.pkl')
    #  with open(pd_pkl_file, 'wb') as f:
    #      pickle.dump(pd_tup, f)
    pd_txt_file = os.path.join(out_dir, 'plaq_diffs_data.txt')
    np.savetxt(pd_txt_file, output_arr, delimiter=',', fmt='%.4g')
    #
    #  with open(pd_txt_file, 'a') as f:
    #      for row in pd_tup:
    #          f.write(f'{row[0][0]}, {row[0][1]}, {row[0][2]}, {row[1]}\n')


def inference_setup(kwargs):
    """Set up relevant (initial) values to use when running inference."""
    if kwargs['loop_net_weights']:  # loop over different values of [S, T, Q]
        #  net_weights_arr = np.zeros((9, 3), dtype=NP_FLOAT)
        #  w = np.random.randn(3) + 1.
        zero_weights = np.array([0.00, 0.00, 0.00])   # set weights to 0.

        q_weights = np.array([[0.00, 0.00, 0.10],   # loop over Q weights
                              [0.00, 0.00, 0.25],
                              [0.00, 0.00, 0.50],
                              [0.00, 0.00, 0.75],
                              [0.00, 0.00, 1.00],
                              [0.00, 0.00, 1.50],
                              [0.00, 0.00, 2.00],
                              [0.00, 0.00, 5.00]])

        t_weights = np.array([[0.00, 0.10, 0.00],
                              [0.00, 0.25, 0.00],
                              [0.00, 0.50, 0.00],
                              [0.00, 0.75, 0.00],
                              [0.00, 1.00, 0.00],
                              [0.00, 1.50, 0.00],
                              [0.00, 2.00, 0.00],
                              [0.00, 5.00, 0.00]])

        s_weights = np.array([[0.10, 0.00, 0.00],
                              [0.25, 0.00, 0.00],
                              [0.50, 0.00, 0.00],
                              [0.75, 0.00, 0.00],
                              [1.00, 0.00, 0.00],
                              [1.50, 0.00, 0.00],
                              [2.00, 0.00, 0.00],
                              [5.00, 0.00, 0.00]])

        stq_weights = np.array([1.00, 1.00, 1.00])

        net_weights_arr = [zero_weights, q_weights,
                           t_weights, s_weights, stq_weights]

    elif kwargs['loop_transl_weights']:
        net_weights_arr = np.array([[1.0, 0.00, 1.0],
                                    [1.0, 0.10, 1.0],
                                    [1.0, 0.25, 1.0],
                                    [1.0, 0.50, 1.0],
                                    [1.0, 0.75, 1.0],
                                    [1.0, 1.00, 1.0]], dtype=NP_FLOAT)

    else:  # set [S, T, Q] = [1, 1, 1]
        scale_weight = kwargs.get('scale_weight', 1.)
        translation_weight = kwargs.get('translation_weight', 1.)
        transformation_weight = kwargs.get('transformation_weight', 1.)
        net_weights_arr = np.array([scale_weight,
                                    translation_weight,
                                    transformation_weight], dtype=NP_FLOAT)

        #  net_weights_arr = np.array([1, 1, 1], dtype=NP_FLOAT)

    # if a value has been passed in `kwargs['beta_inference']` use it
    # otherwise, use `model.beta_final`
    beta_final = kwargs.get('beta_final', None)
    beta_inference = kwargs.get('beta_inference', None)
    beta = beta_final if beta_inference is None else beta_inference
    eps = kwargs.get('eps', None)
    #  betas = [beta_final if beta_inference is None else beta_inference]

    inference_dict = {
        'net_weights': net_weights_arr,
        'beta': beta,
        'eps': eps,
        #  'charge_weight': kwargs.get('charge_weight', 1.),
        'run_steps': kwargs.get('run_steps', 5000),
        'plot_lf': kwargs.get('plot_lf', False),
        'loop_net_weights': kwargs['loop_net_weights'],
        'loop_transl_weights': kwargs['loop_transl_weights']
    }

    return inference_dict


def _log_inference_header(nw, run_steps, eps, beta, existing=False):
    if existing:
        str0 = f'\n Inference has already been completed for:'
    else:
        str0 = f'\n Running inference with:'
    io.log(SEP_STR)
    io.log(f'\n {str0}\n'
           f'\t net_weights: [{nw[0]}, {nw[1]}, {nw[2]}]\n'
           f'\t run_steps: {run_steps}\n'
           f'\t eps: {eps}\n'
           f'\t beta: {beta}\n')
    io.log(SEP_STR)
