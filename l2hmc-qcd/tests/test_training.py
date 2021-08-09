"""
test.py

Test training on 2D U(1) model.
"""
from __future__ import absolute_import, annotations, division, print_function

import argparse
import copy
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import Union, Any


warnings.filterwarnings(action='once', category=UserWarning)
warnings.filterwarnings('once', 'keras')

MODULEPATH = os.path.join(os.path.dirname(__file__), '..')
if MODULEPATH not in sys.path:
    sys.path.append(MODULEPATH)


import utils.file_io as io
from config import BIN_DIR, GAUGE_LOGS_DIR
from utils.attr_dict import AttrDict
from utils.hvd_init import RANK
from utils.inference_utils import InferenceResults, run, run_hmc
from utils.logger import Logger
from utils.training_utils import TrainOutputs, train

logger = Logger()


# pylint:disable=import-outside-toplevel, invalid-name, broad-except
TIMING_FILE = os.path.join(BIN_DIR, 'test_benchmarks.log')
LOG_FILE = os.path.join(BIN_DIR, 'log_dirs.txt')


@dataclass
class TestOutputs:
    train: TrainOutputs
    run: Union[InferenceResults, None]


def parse_args():
    """Method for parsing CLI flags."""
    description = (
        "Various test functions to make sure everything runs as expected."
    )

    parser = argparse.ArgumentParser(
        description=description,
    )
    parser.add_argument('--make_plots', action='store_true',
                        help=("""Whether or not to make plots."""))

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


def test_hmc_run(
        configs: dict[str, Any],
        make_plots: bool = True,
) -> TestOutputs:
    """Testing generic HMC."""
    logger.info(f'Testing generic HMC')
    t0 = time.time()
    configs = AttrDict(**dict(copy.deepcopy(configs)))
    configs['dynamics_config']['hmc'] = True
    #  hmc_dir = os.path.join(os.path.dirname(PROJECT_DIR),
    #                         'gauge_logs_eager', 'test', 'hmc_runs')
    hmc_dir = os.path.join(GAUGE_LOGS_DIR, 'hmc_test_logs')
    run_out = run_hmc(configs, hmc_dir=hmc_dir, make_plots=make_plots)

    logger.info(f'Passed! Took: {time.time() - t0:.4f} seconds')
    return TestOutputs(None, run_out)


def test_conv_net(
        configs: dict[str, Any],
        make_plots: bool = True,
) -> TestOutputs:
    """Test convolutional networks."""
    t0 = time.time()
    logger.info(f'Testing convolutional network')
    configs = AttrDict(**dict(copy.deepcopy(configs)))
    #  flags.use_conv_net = True
    configs['dynamics_config']['use_conv_net'] = True
    configs['conv_config'] = dict(
        sizes=[2, 2],
        filters=[16, 32],
        pool_sizes=[2, 2],
        use_batch_norm=True,
        conv_paddings=['valid', 'valid'],
        conv_activations=['relu', 'relu'],
        input_shape=configs['dynamics_config']['x_shape'][1:],
    )
    train_out = train(configs, make_plots=make_plots,
                      num_chains=4, verbose=False)
    runs_dir = os.path.join(train_out.logdir, 'inference')
    run_out = None
    if RANK == 0:
        run_out = run(train_out.dynamics, configs, x=train_out.x,
                      runs_dir=runs_dir, make_plots=make_plots)
    logger.info(f'Passed! Took: {time.time() - t0:.4f} seconds')

    return TestOutputs(train_out, run_out)


def test_single_network(
        configs: dict[str, Any],
        make_plots: bool = True,
) -> TestOutputs:
    """Test training on single network."""
    t0 = time.time()
    logger.info(f'Testing single network')
    configs_ = dict(copy.deepcopy(configs))
    configs_['dynamics_config']['separate_networks'] = False
    train_out = train(configs_, make_plots=make_plots,
                      verbose=False, num_chains=4)
    logdir = train_out.logdir
    runs_dir = os.path.join(logdir, 'inference')
    run_out = None
    if RANK == 0:
        run_out = run(train_out.dynamics, configs_, x=train_out.x,
                      runs_dir=runs_dir, make_plots=make_plots)

    logger.info(f'Passed! Took: {time.time() - t0:.4f} seconds')
    return TestOutputs(train_out, run_out)


def test_separate_networks(
        configs: dict[str, Any],
        make_plots: bool = True,
) -> TestOutputs:
    """Test training on separate networks."""
    t0 = time.time()
    logger.info(f'Testing separate networks')
    configs_ = dict(copy.deepcopy(configs))
    configs_['hmc_steps'] = 0
    configs_['dynamics_config']['separate_networks'] = True
    configs_['compile'] = False
    train_out = train(configs_, make_plots=make_plots,
                      verbose=False, num_chains=4)
    x = train_out.x
    dynamics = train_out.dynamics
    logdir = train_out.logdir
    runs_dir = os.path.join(logdir, 'inference')
    run_out = None
    if RANK == 0:
        run_out = run(dynamics, configs_, x=x,
                      runs_dir=runs_dir, make_plots=make_plots)

    logger.info(f'Passed! Took: {time.time() - t0:.4f} seconds')
    return TestOutputs(train_out, run_out)



def test_resume_training(
        configs: dict[str, Any],
        make_plots: bool = True,
) -> TestOutputs:
    """Test restoring a training session from a checkpoint."""
    t0 = time.time()
    logger.info(f'Testing resuming training')

    configs_ = copy.deepcopy(configs)
    assert configs_.get('restore_from', None) is not None

    train_out = train(configs_, make_plots=make_plots,
                      verbose=False, num_chains=4)
    dynamics = train_out.dynamics
    logdir = train_out.logdir
    x = train_out.x
    runs_dir = os.path.join(logdir, 'inference')
    run_out = None
    if RANK == 0:
        run_out = run(dynamics, configs_, x=x, runs_dir=runs_dir)

    logger.info(f'Passed! Took: {time.time() - t0:.4f} seconds')
    return TestOutputs(train_out, run_out)


def test(make_plots: bool = False):
    """Run tests."""
    t0 = time.time()
    configs = parse_test_configs()

    sep_configs = copy.deepcopy(configs)
    conv_configs = copy.deepcopy(configs)
    single_configs = copy.deepcopy(configs)

    sep_out = test_separate_networks(sep_configs, make_plots=make_plots)

    sep_configs['train_steps'] += 10
    sep_configs['restore_from'] = sep_out.train.logdir
    sep_configs['log_dir'] = None
    _ = test_resume_training(sep_configs, make_plots=make_plots)

    bf = sep_configs.get('beta_final')
    beta_final = bf + 1
    sep_configs['beta_final'] = beta_final
    logger.log(f'Increasing beta: {bf} -> {beta_final}')
    sep_configs['ensure_new'] = True
    sep_configs['beta_final'] = sep_configs['beta_final'] + 1

    _ = test_resume_training(sep_configs, make_plots=make_plots)

    single_net_out = test_single_network(single_configs, make_plots=make_plots)
    single_configs['restore_from'] = single_net_out.train.logdir
    single_configs['dynamics_config']['separate_networks'] = False
    _ = test_resume_training(single_configs, make_plots=False)

    configs['ensure_new'] = True
    configs['log_dir'] = None

    _ = test_separate_networks(configs, make_plots=False)

    configs['ensure_new'] = True
    configs['log_dir'] = None

    _ = test_single_network(configs, make_plots=False)

    _ = test_conv_net(conv_configs, make_plots=make_plots)

    if RANK  == 0:
        _ = test_hmc_run(configs, make_plots=False)

    logger.info(f'All tests passed! Took: {time.time() - t0:.4f} s')
    logger.rule()


def main(args, flags=None):
    """Main method."""
    fn_map = {
        'test_hmc_run': test_hmc_run,
        'test_separate_networks': test_separate_networks,
        'test_single_network': test_single_network,
        'test_resume_training': test_resume_training,
        'test_conv_net': test_conv_net,
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

    _ = test(ARGS.make_plots) if ARGS.test_all else main(ARGS)
