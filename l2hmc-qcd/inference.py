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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pickle
import tensorflow as tf
import numpy as np

import utils.file_io as io

from variables import NP_FLOAT
from tensorflow.core.protobuf import rewriter_config_pb2
#  from gauge_model_main import create_config

from utils.parse_args import parse_args
from models.model import GaugeModel
from loggers.run_logger import RunLogger
from plotters.gauge_model_plotter import GaugeModelPlotter
from plotters.leapfrog_plotters import LeapfrogPlotter
from runners.runner import GaugeModelRunner

try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)


SEP_STR = (80 * '-') + '\n'


def create_config(params):
    """Create tensorflow config."""
    config = tf.ConfigProto()
    if params['time_size'] > 8:
        off = rewriter_config_pb2.RewriterConfig.OFF
        config_attrs = config.graph_options.rewrite_options
        config_attrs.arithmetic_optimization = off

    if params['gpu']:
        # Horovod: pin GPU to be used to process local rank (one GPU per
        # process)
        config.gpu_options.allow_growth = True
        #  config.allow_soft_placement = True
        if HAS_HOROVOD and params['using_hvd']:
            #  num_gpus = hvd.size()
            #  io.log(f"Number of GPUs: {num_gpus}")
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


def load_params():
    params_pkl_file = os.path.join(os.getcwd(), 'params.pkl')
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


def run_hmc(params, **kwargs):
    """Run inference using generic HMC."""
    # -----------------------------------------------------------
    #  run HMC following inference if --run_hmc flag was passed
    # -----------------------------------------------------------
    #  if params['run_hmc']:
    #      # Run HMC with the trained step size from L2HMC (not ideal)
    #      params = model.params
    #      params['hmc'] = True
    #      params['log_dir'] = None
    #      #  params['log_dir'] = FLAGS.log_dir = None
    #      if train_logger is not None:
    #          params['eps'] = train_logger._current_state['eps']
    #      else:
    #          params['eps'] = params['eps']
    #
    #      run_hmc(kwargs, params, log_file)
    #
    #      for eps in eps_arr:
    #          params['log_dir'] = FLAGS.log_dir = None
    #          params['eps'] = FLAGS.eps = eps
    #          run_hmc(FLAGS, params, log_file)
    pass


def inference_setup(kwargs):
    """Set up relevant (initial) values to use when running inference."""
    # -------------------------------------------------  
    if kwargs['loop_net_weights']:  # loop over different values of [Q, S, T]
        net_weights_arr = np.zeros((9, 3), dtype=NP_FLOAT)
        mask_arr = np.array([[1, 1, 1],                     # [Q, S, T]
                             [0, 1, 1],                     # [ , S, T]
                             [1, 0, 1],                     # [Q,  , T]
                             [1, 1, 0],                     # [Q, S,  ]
                             [1, 0, 0],                     # [Q,  ,  ]
                             [0, 1, 0],                     # [ , S,  ]
                             [0, 0, 1],                     # [ ,  , T]
                             [0, 0, 0]], dtype=NP_FLOAT)    # [ ,  ,  ]
        net_weights_arr[:mask_arr.shape[0], :] = mask_arr   # [?, ?, ?]
        #  net_weights_arr[-1, :] = np.random.randn(3)

    else:  # set [Q, S, T] = [1, 1, 1]
        net_weights_arr = np.array([[1, 1, 1]], dtype=NP_FLOAT)

    # if a value has been passed in `kwargs['beta_inference']` use it
    # otherwise, use `model.beta_final`
    beta_final = kwargs['beta_final']
    beta_inference = kwargs['beta_inference']
    betas = [beta_final if beta_inference is None else beta_inference]

    inference_dict = {
        'net_weights_arr': net_weights_arr,
        'betas': betas,
        'charge_weight': kwargs.get('charge_weight', 1.),
        'run_steps': kwargs.get('run_steps', 5000),
        'plot_lf': kwargs.get('plot_lf', True)
    }

    return inference_dict


def run_inference(run_dict,
                  runner,
                  run_logger=None,
                  plotter=None,
                  dir_append=None):
    """Run inference.

    Args:
        inference_dict: Dictionary containing parameters to use for inference.
        runner: GaugeModelRunner object that actually performs the inference.
        run_logger: RunLogger object that logs observables and other data
            generated during inference.
        plotter: GaugeModelPlotter object responsible for plotting observables
            generated during inference.
    """
    if plotter is None and run_logger is None:
        return

    net_weights_arr = run_dict['net_weights_arr']
    betas = run_dict['betas']
    charge_weight = run_dict['charge_weight']
    run_steps = run_dict['run_steps']
    plot_lf = run_dict['plot_lf']

    for net_weights in net_weights_arr:
        weights = {
            'charge_weight': charge_weight,
            'net_weights': net_weights
        }
        for beta in betas:
            #  if run_logger is not None:
            run_dir, run_str = run_logger.reset(run_steps,
                                                beta,
                                                weights,
                                                dir_append)

            t0 = time.time()
            runner.run(run_steps, beta, weights['net_weights'], therm_frac=10)
            io.log(SEP_STR)

            # log the total time spent running inference
            run_time = time.time() - t0
            io.log(
                SEP_STR + f'Took: {run_time} s to complete run.\n' + SEP_STR
            )

            # --------------------
            #  Plot observables
            # --------------------
            #  if plotter is not None and run_logger is not None:
            plotter.plot_observables(
                run_logger.run_data, beta, run_str, weights, dir_append
            )
            if plot_lf:
                lf_plotter = LeapfrogPlotter(plotter.out_dir, run_logger)
                num_samples = min((runner.model.num_samples, 20))
                lf_plotter.make_plots(run_dir, num_samples=num_samples)


def main_inference(kwargs):
    """Perform inference using saved model."""
    params = load_params()  # load parameters used during training

    # We want to restrict all communication (file I/O) to only be performed on
    # rank 0 (i.e. `is_chief`) so there are two cases:
    #    1. We're using Horovod, so we have to check hvd.rank() explicitly.
    #    2. We're not using Horovod, in which case `is_chief` is always True.
    condition1 = not params['using_hvd']
    condition2 = params['using_hvd'] and hvd.rank() == 0
    is_chief = condition1 or condition2
    if not is_chief:
        return

    if is_chief:
        checkpoint_dir = os.path.join(params['log_dir'], 'checkpoints/')
        assert os.path.isdir(checkpoint_dir)
    else:
        params['log_dir'] = None
        checkpoint_dir = None

    config, params = create_config(params)
    model = GaugeModel(params=params)

    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------
    sess = tf.Session(config=config)
    if is_chief:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        run_logger = RunLogger(model, params['log_dir'], save_lf_data=False)
        plotter = GaugeModelPlotter(model, run_logger.figs_dir)
    else:
        run_logger = None
        plotter = None

    # -------------------------------------------------------------------
    #  Set up relevant values to use for inference (parsed from kwargs)
    # -------------------------------------------------------------------
    inference_dict = inference_setup(kwargs)

    # --------------------------------------
    # Create GaugeModelRunner for inference
    # --------------------------------------
    runner = GaugeModelRunner(sess, model, run_logger)
    run_inference(inference_dict, runner, run_logger, plotter)

    # set 'net_weights_arr' = [1., 1., 1.] so each Q, S, T contribute
    inference_dict['net_weights_arr'] = np.array([[1, 1, 1]], dtype=NP_FLOAT)

    # set 'betas' to be a single value
    #  inference_dict['betas'] = inference_dict['betas'][-1]

    # randomize the model weights and run inference using these weights
    runner.model = set_model_weights(runner.model, dest='rand')
    run_inference(inference_dict,
                  runner, run_logger,
                  plotter, dir_append='_rand')

    # zero the model weights and run inference using these weights
    runner.model = set_model_weights(runner.model, dest='zero')
    run_inference(inference_dict,
                  runner, run_logger,
                  plotter, dir_append='_zero')


if __name__ == '__main__':
    args = parse_args()
    kwargs = args.__dict__
    main_inference(kwargs)
