"""
train_test.py

Test training on 2D U(1) model using eager execution in tensorflow.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse

from config import DEFAULT_FLAGS, PROJECT_DIR
from utils.attr_dict import AttrDict
from utils.training_utils import train
from utils.inference_utils import load_and_run, run, run_hmc

DESCRIPTION = (
    "Various test functions to make sure everything runs as expected."
)


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


def test_hmc_run(beta=4., eps=0.1, num_steps=2, run_steps=500):
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


def test_single_network(flags):
    """Test training on single network."""
    x, model, train_data, flags = train(flags)
    model, run_data = run(model, flags, x=x)

    return {
        'x': x,
        'flags': flags,
        'model': model,
        'run_data': run_data,
        'train_data': train_data,
    }


def test_separate_networks(flags):
    """Test training on separate networks."""
    flags.log_dir = None
    flags.separate_networks = True
    x, model, train_data, flags = train(flags)
    model, run_data = run(model, flags, x=x)

    return {
        'x': x,
        'flags': flags,
        'model': model,
        'run_data': run_data,
        'train_data': train_data,
    }


def test(flags):
    """Run tests."""
    return {
        'single_network': test_single_network(flags),
        'separate_networks': test_separate_networks(flags),
        'hmc': test_hmc_run(),
    }


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        _ = test(DEFAULT_FLAGS)
    else:
        FLAGS = parse_args()
        FLAGS = AttrDict(FLAGS.__dict__)

        if FLAGS.test_hmc_run:
            _ = test_hmc_run()

        if FLAGS.test_separate_networks:
            _ = test_separate_networks(DEFAULT_FLAGS)

        if FLAGS.test_single_network:
            _ = test_single_network(FLAGS)

        if FLAGS.test_inference_from_model:
            if FLAGS.log_dir is None:
                raise ValueError('`--log_dir` must be specified.')

            DEFAULT_FLAGS.log_dir = FLAGS.log_dir
            _, _ = load_and_run(DEFAULT_FLAGS)
