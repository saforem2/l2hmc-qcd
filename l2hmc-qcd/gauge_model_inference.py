"""
gauge_model_inference.py

Runs inference using the trained L2HMC sampler contained in a saved model.

This is done by reading in the location of the saved model from a .txt file
containing the location of the checkpoint directory.

Author: Sam Foreman (github: @saforem2)
Date: 07/08/2019
"""
import os
import time
import pickle
import tensorflow as tf
import numpy as np

import utils.file_io as io

from globals import NP_FLOAT
from tensorflow.core.protobuf import rewriter_config_pb2
#  from gauge_model_main import create_config

from utils.parse_args import parse_args
from models.gauge_model import GaugeModel
from loggers.run_logger import RunLogger
from plotters.gauge_model_plotter import GaugeModelPlotter
from plotters.leapfrog_plotters import LeapfrogPlotter
from runners.gauge_model_runner import GaugeModelRunner

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


#  def make_flags(params):
#      FLAGS = AttrDict()
#      for key, val in params.items():
#          setattr(FLAGS, key, val)
#
#      return FLAGS

    #  log_dir = os.path.dirname(checkpoint_dir)

    #  if params is None:
    #      params_pkl_file = os.path.join(os.getcwd(), 'params.pkl')
    #      with open(params_pkl_file, 'rb') as f:
    #          params = pickle.load(f)
    #  else:
    #      if FLAGS is None:
    #          FLAGS = AttrDict()
    #          for key, val in params.items():
    #              setattr(FLAGS, key, val)
    #      else:
    #          params = {}
    #          for key, val in FLAGS.__dict__.items():
    #              params[key] = val


def run_l2hmc(params, **kwargs):
    """Perform inference using saved model."""
    condition1 = not params['using_hvd']
    condition2 = params['using_hvd'] and hvd.rank() == 0
    is_chief = condition1 or condition2

    if is_chief:
        checkpoint_dir = os.path.join(params['log_dir'], 'checkpoints/')
        assert os.path.isdir(checkpoint_dir)
    else:
        params['log_dir'] = None
        checkpoint_dir = None

    config, params = create_config(params)
    model = GaugeModel(params=params)

    if params['using_hvd']:
        bcast_op = hvd.broadcast_global_variables(0)
    else:
        bcast_op = None

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
        plotter =None

    if bcast_op is not None:
        sess.run(bcast_op)
    #  except ValueError:
    #      import pdb
    #      pdb.set_trace()

    #  model.sess = sess
    #  run_logger = RunLogger(model, FLAGS.log_dir, save_lf_data=False)

    # -------------------------------------------------  
    #  Set up relevant parameters to use for inference   
    # -------------------------------------------------  
    if params['loop_net_weights']:
        net_weights_arr = np.array([[1, 1, 1],  # [Q, S, T]
                                    [0, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 0],
                                    [1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1],
                                    [0, 0, 0]], dtype=NP_FLOAT)
    else:
        net_weights_arr = np.array([[1, 1, 1]], dtype=NP_FLOAT)

    beta = kwargs.get('beta', None)
    if beta is None:
        betas = [model.beta_final]
    else:
        betas = [beta]

    charge_weight = kwargs.get('charge_weight', None)
    if charge_weight is None:
        charge_weight = params['charge_weight_init']

    # Create GaugeModelRunner for inference
    runner = GaugeModelRunner(sess, model, run_logger)
    for net_weights in net_weights_arr:
        weights = {
            'charge_weight': charge_weight,
            'net_weights': net_weights
        }
        for beta in betas:
            if run_logger is not None:
                run_dir, run_str = run_logger.reset(model.run_steps,
                                                    beta, **weights)
            #  t0 = time.time()
            runner.run(model.run_steps,
                       beta,
                       weights['net_weights'],
                       therm_frac=10)
            #  run_time = time.time() - t0
            #  io.log(80 * '-')
            #  io.log(f'Took: {run_time} s to complete run.')
            #  io.log(80 * '-')

            if plotter is not None and run_logger is not None:
                plotter.plot_observables(
                    run_logger.run_data, beta, run_str, **weights
                )
                if params['save_lf']:
                    lf_plotter = LeapfrogPlotter(plotter.out_dir, run_logger)
                    lf_plotter.make_plots(run_dir, num_samples=20)


def main_inference(args):
    params = load_params()

    kwargs = {
        'beta': args.beta_inference,
        'charge_weight': args.charge_weight_inference
    }
    #  'beta': args.__dict__.get('beta_inference', None),
    #  'charge_weight': args.__dict__.get('charge_weight_inference', None)

    try:
        run_l2hmc(params, **kwargs)
    except:
        import pdb
        pdb.set_trace()
    #  params, model, run_logger = run_l2hmc(params)
    #  checkpoint_dir = params['checkpoint_dir']
    #  if FLAGS is not None:
    #      for key, val in FLAGS.__dict__.items():

    #  FLAGS = make_flags(params)
    #  FLAGS, params, model, run_logger = run_l2hmc(params)


if __name__ == '__main__':
    args = parse_args()
    main_inference(args)
