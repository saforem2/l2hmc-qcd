"""
train_test.py

Test training on 2D U(1) model using eager execution in tensorflow.
"""
from __future__ import absolute_import, division, print_function

import os

from utils.attr_dict import AttrDict
from utils.parse_args import parse_args
from utils.training_utils import train
from utils.inference_utils import run, load_and_run, run_hmc

DEFAULT_FLAGS = AttrDict({
    'log_dir': None,
    'eager_execution': True,
    'restore': False,
    'inference': True,
    'run_steps': 50,
    'save_train_data': True,
    'horovod': False,
    'rand': True,
    'eps': 0.1,
    'num_steps': 2,
    'batch_size': 64,
    'time_size': 16,
    'space_size': 16,
    'dim': 2,
    'hmc': False,
    'eps_fixed': False,
    'beta_init': 3.,
    'beta_final': 3.,
    'train_steps': 50,
    'save_steps': 5,
    'print_steps': 1,
    'logging_steps': 1,
    'hmc_start': True,
    'hmc_steps': 20,
    'dropout_prob': 0.1,
    'warmup_lr': True,
    'lr_init': 0.0001,
    'lr_decay_steps': 10,
    'lr_decay_rate': 0.96,
    'plaq_weight': 0.1,
    'charge_weight': 0.1,
    'network_type': 'GaugeNetwork',
    'units': [512, 256, 256, 256, 512],
    'separate_networks': False,
})


def test_hmc_run():
    """Testing generic HMC."""
    hmc_args = AttrDict({
        'hmc': True,
        'log_dir': None,
        'beta': 1.,
        'eps': 0.1,
        'num_steps': 2,
        'run_steps': 500,
        'lattice_shape': (128, 16, 16, 2),
    })
    model, run_data = run_hmc(hmc_args)

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
    _ = test(DEFAULT_FLAGS)
