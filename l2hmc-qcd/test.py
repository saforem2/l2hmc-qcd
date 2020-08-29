"""
train_test.py

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

DESCRIPTION = (
    "Various test functions to make sure everything runs as expected."
)

TEST_FLAGS_FILE = os.path.join(BIN_DIR, 'test_args.json')
with open(TEST_FLAGS_FILE, 'rt') as f:
    TEST_FLAGS = json.load(f)

TEST_FLAGS = AttrDict(TEST_FLAGS)


def parse_args():
    """Method for parsing CLI flags."""
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
    )
    parser.add_argument('--horovod',
                        action='store_true',
                        required=False,
                        help=("""Running with Horovod."""))

    parser.add_argument('--test_all',
                        action='store_true',
                        required=False,
                        help=("""Run all tests."""))

    parser.add_argument('--test_separate_networks',
                        action='store_true',
                        required=False,
                        help=("""Test `--separate_networks` specifically."""))

    parser.add_argument('--test_single_network',
                        action='store_true',
                        required=False,
                        help=("""Test `--single_network` specifically."""))

    parser.add_argument('--test_hmc_run',
                        action='store_true',
                        required=False,
                        help=("""Test HMC inference specifically."""))

    parser.add_argument('--test_inference_from_model',
                        action='store_true',
                        required=False,
                        help=("""Test running inference from saved
                              (trained) model specifically."""))

    parser.add_argument('--test_resume_training',
                        action='store_true',
                        required=False,
                        help=("""Test resuming training from checkpoint. Must
                              specify `--log_dir`."""))

    parser.add_argument('--log_dir',
                        default=None,
                        type=str,
                        required=False,
                        help=("""`log_dir` from which to load saved model for
                              running inference on."""))
    args = parser.parse_args()

    return args


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
def test_hmc_run(args: AttrDict):
    """Testing generic HMC."""
    hmc_args = AttrDict({
        'hmc': True,
        'log_dir': None,
        'horovod': args.get('horovod', False),
        'beta': args.get('beta', 1.),
        'eps': args.get('eps', 0.1),
        'num_steps': args.get('num_steps', 2),
        'run_steps': args.get('run_steps', 1000),
        'plaq_weight': args.get('plaq_weight', 10.),
        'charge_weight': args.get('charge_weight', 0.1),
        'lattice_shape': (128, 16, 16, 2),
    })
    hmc_dir = os.path.join(os.path.dirname(PROJECT_DIR),
                           'gauge_logs_eager', 'test', 'hmc_runs')
    dynamics, run_data, x = run_hmc(hmc_args, hmc_dir=hmc_dir)

    return {
        'x': x,
        'dynamics': dynamics,
        'flags': hmc_args,
        'run_data': run_data,
        'tk_diffs': None,
    }


@timeit(out_file=None)
def test_single_network(flags: AttrDict):
    """Test training on single network."""
    flags.separate_networks = False
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
def test_separate_networks(flags: AttrDict):
    """Test training on separate networks."""
    flags.hmc_steps = 0
    flags.log_dir = None
    flags.separate_networks = True
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
def test(flags: AttrDict):
    """Run tests."""
    flags.compile = True
    single_net_out = test_single_network(flags)
    log_dir = single_net_out.log_dir
    _ = test_resume_training(log_dir)
    #  _ = test_separate_networks(flags)
    _ = test_hmc_run(flags)


@timeit(out_file=None)
def main(args):
    """Main method."""
    if args.test_hmc_run:
        _ = test_hmc_run(TEST_FLAGS)

    #  if args.test_separate_networks:
    #      _ = test_separate_networks(TEST_FLAGS)

    if args.test_single_network:
        TEST_FLAGS.hmc_steps = 0
        TEST_FLAGS.hmc_start = False
        _ = test_single_network(TEST_FLAGS)

    if args.test_resume_training:
        _ = test_resume_training(args.log_dir)

    if args.test_inference_from_model:
        if args.log_dir is None:
            raise ValueError('`--log_dir` must be specified.')

        TEST_FLAGS.log_dir = args.log_dir
        _, _ = load_and_run(TEST_FLAGS)


if __name__ == '__main__':
    FLAGS = parse_args()
    if FLAGS.horovod:
        TEST_FLAGS.horovod = True

    _ = test(TEST_FLAGS) if FLAGS.test_all else main(FLAGS)
