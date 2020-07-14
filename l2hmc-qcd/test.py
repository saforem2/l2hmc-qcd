"""
train_test.py

Test training on 2D U(1) model using eager execution in tensorflow.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import json
import typing
import tensorflow as tf

from config import PROJECT_DIR, BIN_DIR
from functools import wraps
from utils.attr_dict import AttrDict
from utils.training_utils import train
from utils.inference_utils import load_and_run, run, run_hmc
from utils.file_io import timeit

# pylint:disable=import-outside-toplevel, invalid-name, broad-except

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


def test_hmc_run(beta: float = 4.,
                 eps: float = 0.1,
                 num_steps: int = 2,
                 run_steps: int = 500):
    """Testing generic HMC."""
    hmc_args = AttrDict({
        'hmc': True,
        'log_dir': None,
        'beta': beta,
        'eps': eps,
        'num_steps': num_steps,
        'run_steps': run_steps,
        'lattice_shape': (128, 16, 16, 2),
    })
    hmc_dir = os.path.join(os.path.dirname(PROJECT_DIR),
                           'gauge_logs_eager', 'test', 'hmc_runs')
    model, run_data = run_hmc(hmc_args, hmc_dir=hmc_dir)

    return {
        'model': model,
        'flags': hmc_args,
        'run_data': run_data,
    }


def test_single_network(flags: AttrDict):
    """Test training on single network."""
    x, dynamics, train_data, flags = train(flags, log_file=LOG_FILE)
    model, run_data = run(dynamics, flags, x=x)

    return {
        'x': x,
        'flags': flags,
        'model': model,
        'run_data': run_data,
        'train_data': train_data,
    }


def test_separate_networks(flags: AttrDict):
    """Test training on separate networks."""
    flags.log_dir = None
    flags.separate_networks = True
    x, model, train_data, flags = train(flags, log_file=LOG_FILE)
    model, run_data = run(model, flags, x=x)

    return {
        'x': x,
        'flags': flags,
        'model': model,
        'run_data': run_data,
        'train_data': train_data,
    }


def test_resume_training(flags: AttrDict):
    """Test restoring a training session from a checkpoint."""
    assert flags.log_dir is not None
    flags.profiler = False
    #  flags.train_steps *= 2
    #  flags.train_steps += flags.train_steps // 2
    flags.train_steps *= 2

    x, model, train_data, flags = train(flags, log_file=LOG_FILE)
    model, run_data = run(model, flags, x=x)

    return {
        'x': x,
        'flags': flags,
        'model': model,
        'run_data': run_data,
        'train_data': train_data,
    }


def test(flags: AttrDict):
    """Run tests."""
    with tf.name_scope('single_network'):
        flags.compile = True
        single_net_out = test_single_network(flags)
        restored_out = test_resume_training(single_net_out['flags'])
    with tf.name_scope('separate_networks'):
        flags.hmc_start = False
        flags.hmc_steps = 0
        flags.separate_networks = True
        flags.compile = False
        separate_nets_out = test_separate_networks(flags)
    with tf.name_scope('hmc_inference'):
        hmc_out = test_hmc_run()
    return {
        'hmc': hmc_out,
        'single_network': single_net_out,
        'separate_networks': separate_nets_out,
        'restored': restored_out,
    }


def main(args):
    """Main method."""
    if args.test_hmc_run:
        _ = test_hmc_run()

    if args.test_separate_networks:
        TEST_FLAGS.separate_networks = True
        TEST_FLAGS.compile = False
        _ = test_separate_networks(TEST_FLAGS)

    if args.test_single_network:
        TEST_FLAGS.hmc_steps = 0
        TEST_FLAGS.hmc_start = False
        TEST_FLAGS.separate_networks = False
        _ = test_single_network(TEST_FLAGS)

    if args.test_inference_from_model:
        if args.log_dir is None:
            raise ValueError('`--log_dir` must be specified.')

        TEST_FLAGS.log_dir = args.log_dir
        _, _ = load_and_run(TEST_FLAGS)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        _ = test(TEST_FLAGS)
    else:
        FLAGS = parse_args()
        main(FLAGS)
