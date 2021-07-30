"""
test.py

Test training on 2D U(1) model.
"""
from __future__ import absolute_import, division, print_function, annotations

import os
import sys
import json
import copy
import argparse

import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import Any

if tf.__version__.startswith('1'):
    try:
        tf.compat.v1.enable_v2_behavior()
    except AttributeError:
        print('Unable to call \n'
              '`tf.compat.v1.enable_v2_behavior()`. Continuing...')
    try:
        tf.compat.v1.enable_control_flow_v2()
    except AttributeError:
        print('Unable to call \n'
              '`tf.compat.v1.enable_control_flow_v2()`. Continuing...')
    try:
        tf.compat.v1.enable_v2_tensorshape()
    except AttributeError:
        print('Unable to call \n'
              '`tf.compat.v1.enable_v2_tensorshape()`. Continuing...')
    try:
        tf.compat.v1.enable_eager_execution()
    except AttributeError:
        print('Unable to call \n'
              '`tf.compat.v1.enable_eager_execution()`. Continuing...')
    try:
        tf.compat.v1.enable_resource_variables()
    except AttributeError:
        print('Unable to call \n'
              '`tf.compat.v1.enable_resource_variables()`. Continuing...')


#  try:
#      import horovod.tensorflow as hvd
#  except ImportError:
#      pass

import numpy as np


MODULEPATH = os.path.join(os.path.dirname(__file__), '..')
if MODULEPATH not in sys.path:
    sys.path.append(MODULEPATH)


#  CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#  PARENT_DIR = os.path.dirname(CURRENT_DIR)
#  if PARENT_DIR not in sys.path:
#      sys.path.append(PARENT_DIR)
from functools import wraps

import utils.file_io as io

from config import BIN_DIR, GAUGE_LOGS_DIR
from network.config import ConvolutionConfig
from utils.file_io import timeit
from utils.attr_dict import AttrDict
from utils.training_utils import train
from utils.inference_utils import load_and_run, run, run_hmc
from utils.logger import Logger

logger = Logger()


# pylint:disable=import-outside-toplevel, invalid-name, broad-except
TIMING_FILE = os.path.join(BIN_DIR, 'test_benchmarks.log')
LOG_FILE = os.path.join(BIN_DIR, 'log_dirs.txt')

#  LOG_FILE = os.path.join(
#      os.path.dirname(PROJECT_DIR), 'bin', 'log_dirs.txt'
#  )


def parse_args():
    """Method for parsing CLI flags."""
    description = (
        "Various test functions to make sure everything runs as expected."
    )

    parser = argparse.ArgumentParser(
        description=description,
    )
    parser.add_argument('--horovod', action='store_true',
                        help=("""Running with Horovod."""))

    parser.add_argument('--test_all', action='store_true',
                        help=("""Run all tests."""))

    parser.add_argument('--test_separate_networks', action='store_true',
                        help="Test `--separate_networks` specifically.")

    parser.add_argument('--test_single_network', action='store_true',
                        help="Test `--single_network` specifically.")

    parser.add_argument('--test_conv_net', action='store_true',
                        help="Test conv. nets specifically.")

    parser.add_argument('--test_hmc_run', action='store_true',
                        help="Test HMC inference specifically.")

    parser.add_argument('--test_inference_from_model', action='store_true',
                        help=("Test running inference from saved "
                              "(trained) model specifically."))

    parser.add_argument('--test_resume_training', action='store_true',
                        help=("Test resuming training from checkpoint. "
                              "Must specify `--log_dir`."""))

    parser.add_argument('--log_dir', default=None, type=str,
                        help=("`log_dir` from which to load saved model "
                              "for running inference on."""))
    args = parser.parse_args()

    return args


def parse_test_configs(test_configs_file=None):
    """Parse `test_config.json`."""
    if test_configs_file is None:
        test_configs_file = os.path.join(BIN_DIR, 'test_configs.json')

    with open(test_configs_file, 'rt') as f:
        test_flags = json.load(f)

    test_flags = AttrDict(dict(test_flags))
    for key, val in test_flags.items():
        if isinstance(val, dict):
            test_flags[key] = AttrDict(val)

    return test_flags


def catch_exception(fn):
    """Decorator function for catching a method with `pudb`."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            print(type(e))
            print(e)

    return wrapper


@timeit
def test_hmc_run(flags: AttrDict):
    """Testing generic HMC."""
    flags = AttrDict(**dict(copy.deepcopy(flags)))
    flags['dynamics_config']['hmc'] = True
    #  hmc_dir = os.path.join(os.path.dirname(PROJECT_DIR),
    #                         'gauge_logs_eager', 'test', 'hmc_runs')
    hmc_dir = os.path.join(GAUGE_LOGS_DIR, 'hmc_test_logs')
    dynamics, run_data, x, _ = run_hmc(flags, hmc_dir=hmc_dir,
                                       make_plots=False)

    return {
        'x': x,
        'dynamics': dynamics,
        'run_data': run_data,
    }


@timeit
def test_conv_net(flags: AttrDict):
    """Test convolutional networks."""
    flags = AttrDict(**dict(copy.deepcopy(flags)))
    #  flags.use_conv_net = True
    flags['dynamics_config']['use_conv_net'] = True
    flags.conv_config = ConvolutionConfig(
        sizes=[2, 2],
        filters=[16, 32],
        pool_sizes=[2, 2],
        use_batch_norm=True,
        conv_paddings=['valid', 'valid'],
        conv_activations=['relu', 'relu'],
        input_shape=flags['dynamics_config']['x_shape'][1:],
    )
    dirs = io.setup_directories(flags)
    flags['dirs'] = dirs
    flags['log_dir'] = dirs.get('log_dir', None)
    train_out = train(flags, make_plots=False)
    run_outputs = run(train_out.dynamics, flags, make_plots=False)
    #  x, dynamics, train_data, flags = train(flags, make_plots=False)
    return {'train_out': train_out, 'run_out': run_outputs}



@timeit
def test_single_network(flags: dict[str, Any]):
    """Test training on single network."""
    flags = dict(copy.deepcopy(flags))
    flags['dynamics_config']['separate_networks'] = False
    train_out = train(flags, make_plots=False)
    logdir = train_out.logdir
    runs_dir = os.path.join(logdir, 'inference')
    run_out = run(train_out.dynamics, flags, x=train_out.x,
                  runs_dir=runs_dir, make_plots=False)

    return {'train_out': train_out, 'run_out': run_out}


@timeit
def test_separate_networks(flags: dict[str, Any]):
    """Test training on separate networks."""
    #  flags = AttrDict(**dict(copy.deepcopy(flags)))
    flags = dict(copy.deepcopy(flags))
    flags['hmc_steps'] = 0
    flags['dynamics_config']['separate_networks'] = True
    flags['compile'] = False
    train_out = train(flags, make_plots=False)
    #  flags.log_dir = None
    #  flags.log_dir = io.make_log_dir(flags, 'GaugeModel', LOG_FILE)

    #  flags.dynamics_config['separate_networks'] = True
    #  flags.compile = False
    x = train_out.x
    train_data = train_out.data
    dynamics = train_out.dynamics
    logdir = train_out.logdir
    runs_dir = os.path.join(logdir, 'inference')
    run_out = run(dynamics, flags, x=x, runs_dir=runs_dir, make_plots=True)
    #  beta = flags.get('beta', 1.)

    return {'train_out': train_out, 'run_out': run_out}



@timeit
def test_resume_training(configs: dict[str, Any]):
    """Test restoring a training session from a checkpoint."""
    logdir = configs.get('logdir', configs.get('log_dir', None))
    if logdir is None:
        dirs = configs.get('dirs', None)
        if dirs is not None:
            logdir = dirs.get('log_dir', dirs.get('logdir', None))

    assert logdir is not None

    #  fpath = os.path.join(logdir, 'train_configs.json')
    fpath = os.path.join(logdir, 'train_configs.z')
    if os.path.isfile(fpath):
        train_configs = io.loadz(fpath)

    else:
        try:
            fpath = os.path.join(logdir, 'train_configs.json')
            with open(fpath, 'r') as f:
                train_configs = json.load(f)
        except FileNotFoundError:
            try:
                train_configs = io.loadz(os.path.join(logdir, 'FLAGS.z'))
            except FileNotFoundError as err:
                raise(err)

    logger.info('Restored train_configs successfully.')
    logger.info('Setting `restored_flag` in `train_configs`.')
    train_configs['log_dir'] = logdir
    train_configs['restored'] = True

    logger.info(f'logdir: {train_configs["log_dir"]}')
    logger.info(f'restored: {train_configs["restored"]}')

    train_out = train(train_configs, make_plots=False)
    dynamics = train_out.dynamics
    logdir = train_out.logdir
    x = train_out.x
    runs_dir = os.path.join(logdir, 'inference')
    run_out = run(dynamics, train_configs, x=x, runs_dir=runs_dir)
    return {'train_out': train_out, 'run_out': run_out}


@timeit
def test():
    """Run tests."""
    flags = parse_test_configs()

    _ = test_separate_networks(flags)

    single_net_out = test_single_network(flags)
    flags['log_dir'] = single_net_out['train_out'].logdir

    _ = test_resume_training(flags)
    _ = test_hmc_run(flags)
    _ = test_conv_net(flags)


@timeit
def main(args, flags=None):
    """Main method."""
    fn_map = {
        'test_hmc_run': test_hmc_run,
        'test_separate_networks': test_separate_networks,
        'test_single_network': test_single_network,
        'test_resume_training': test_resume_training,
        'test_conv_net': test_conv_net,
        #  'test_inference_from_model': test_inference_from_model,
    }
    if flags is None:
        flags = parse_test_configs()

    for arg, fn in fn_map.items():
        if args.__dict__.get(arg):
            if arg == 'test_resume_training':
                _ = fn(args.log_dir)
            else:
                _ = fn(flags)


if __name__ == '__main__':
    ARGS = parse_args()
    if ARGS.horovod:
        ARGS.horovod = True

    _ = test() if ARGS.test_all else main(ARGS)
