"""
grain_test.py

Test training on 2D U(1) model using eager execution in tensorflow.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import json
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np

from config import PROJECT_DIR, BIN_DIR
from functools import wraps
from network.gauge_conv_network import ConvolutionConfig
from utils.attr_dict import AttrDict
from utils.training_utils import train
from utils.inference_utils import load_and_run, run, run_hmc
import utils.file_io as io
from utils.file_io import timeit

# pylint:disable=import-outside-toplevel, invalid-name, broad-except
TIMING_FILE = os.path.join(BIN_DIR, 'test_benchmarks.log')

LOG_FILE = os.path.join(
    os.path.dirname(PROJECT_DIR), 'bin', 'log_dirs.txt'
)


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
            import pudb
            pudb.set_trace()
    return wrapper


@timeit(out_file=None)
def test_transition_kernels(dynamics, x, beta, training=None):
    tk_diffs_f = dynamics.test_transition_kernels(x, beta, forward=True,
                                                  training=False)
    tk_diffs_b = dynamics.test_transition_kernels(x, beta, forward=False,
                                                  training=False)
    io.log('\n\n\n')
    io.log('\n'.join([80 * '=', 'transition kernel differences:']))
    io.log('  forward:')
    for key, val in tk_diffs_f.items():
        print(f'    {key}: {val}\n')
    io.log('  backward:')
    for key, val in tk_diffs_b.items():
        print(f'    {key}: {val}\n')

    io.log(80 * '=' + '\n\n\n')

    return AttrDict({'forward': tk_diffs_f, 'backward': tk_diffs_b})


@timeit(out_file=None)
def test_hmc_run(flags: AttrDict):
    """Testing generic HMC."""
    flags.dynamics_config['hmc'] = True
    hmc_dir = os.path.join(os.path.dirname(PROJECT_DIR),
                           'gauge_logs_eager', 'test', 'hmc_runs')
    dynamics, run_data, x = run_hmc(flags, hmc_dir=hmc_dir)

    return {
        'x': x,
        'dynamics': dynamics,
        'flags': flags,
        'run_data': run_data,
        'tk_diffs': None,
    }


@timeit(out_file=None)
def test_conv_net(flags: AttrDict):
    """Test convolutional networks."""
    flags.use_conv_net = True
    flags.conv_config = ConvolutionConfig(
        input_shape=flags.lattice_shape[1:],
        filters=[16, 32],
        sizes=[2, 2],
        pool_sizes=[2, 2],
        conv_activations=['relu', 'relu'],
        conv_paddings=['valid', 'valid'],
        use_batch_norm=True
    )
    x, dynamics, train_data, flags = train(flags, log_file=LOG_FILE)
    dynamics, run_data, x = run(dynamics, flags, x=x)

    return AttrDict({
        'x': x,
        'flags': flags,
        'log_dir': flags.log_dir,
        'dynamics': dynamics,
        'run_data': run_data,
        'train_data': train_data,
    })


@timeit(out_file=None)
def test_single_network(flags: AttrDict):
    """Test training on single network."""
    flags.dynamics_config.separate_networks = False
    x, dynamics, train_data, flags = train(flags, log_file=LOG_FILE)
    beta = flags.get('beta', 1.)
    tk_diffs = test_transition_kernels(dynamics, x, beta, training=False)
    dynamics, run_data, x = run(dynamics, flags, x=x)

    return AttrDict({
        'x': x,
        'flags': flags,
        'log_dir': flags.log_dir,
        'dynamics': dynamics,
        'run_data': run_data,
        'train_data': train_data,
        'tk_diffs': tk_diffs,
    })


@timeit(out_file=None)
def test_separate_networks(flags: AttrDict):
    """Test training on separate networks."""
    flags.hmc_steps = 0
    flags.log_dir = None
    flags.dynamics_config.separate_networks = True
    flags.compile = False
    x, dynamics, train_data, flags = train(flags,
                                           log_file=LOG_FILE)
    beta = flags.get('beta', 1.)
    tk_diffs = test_transition_kernels(dynamics, x, beta, training=False)
    dynamics, run_data, x = run(dynamics, flags, x=x)

    return AttrDict({
        'x': x,
        'flags': flags,
        'log_dir': flags.log_dir,
        'dynamics': dynamics,
        'run_data': run_data,
        'train_data': train_data,
        'tk_diffs': tk_diffs,
    })


@timeit(out_file=None)
def test_resume_training(log_dir: str):
    """Test restoring a training session from a checkpoint."""
    flags = AttrDict(
        dict(io.loadz(os.path.join(log_dir, 'training', 'FLAGS.z')))
    )

    flags.log_dir = log_dir
    flags.train_steps += flags.get('save_steps', 10)
    x, dynamics, train_data, flags = train(flags,
                                           log_file=LOG_FILE)
    beta = flags.get('beta', 1.)
    tk_diffs = test_transition_kernels(dynamics, x, beta, training=False)
    dynamics, run_data, x = run(dynamics, flags, x=x)

    return AttrDict({
        'x': x,
        'flags': flags,
        'log_dir': flags.log_dir,
        'dynamics': dynamics,
        'run_data': run_data,
        'train_data': train_data,
        'tk_diffs': tk_diffs,
    })


@timeit(out_file=None)
def test(args: AttrDict):
    """Run tests."""
    flags = parse_test_configs()
    single_net_out = test_single_network(flags)
    log_dir = single_net_out.log_dir
    _ = test_resume_training(log_dir)
    _ = test_separate_networks(flags)
    _ = test_hmc_run(flags)
    _ = test_conv_net(flags)


@timeit(out_file=None)
def main(args, flags=None):
    """Main method."""
    if flags is None:
        flags = parse_test_configs()

    if args.test_hmc_run:
        _ = test_hmc_run(flags)

    if args.test_separate_networks:
        _ = test_separate_networks(flags)

    if args.test_single_network:
        flags.hmc_steps = 0
        flags.hmc_start = False
        _ = test_single_network(flags)

    if args.test_resume_training:
        _ = test_resume_training(args.log_dir)

    if args.test_inference_from_model:
        if args.log_dir is None:
            raise ValueError('`--log_dir` must be specified.')

        flags.log_dir = args.log_dir
        _, _ = load_and_run(flags)


if __name__ == '__main__':
    ARGS = parse_args()
    if ARGS.horovod:
        ARGS.horovod = True

    _ = test(ARGS) if ARGS.test_all else main(ARGS)
