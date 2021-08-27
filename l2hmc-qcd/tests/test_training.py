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
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from functools import wraps
from typing import Union, Any
from pathlib import Path


warnings.filterwarnings(action='once', category=UserWarning)
warnings.filterwarnings('once', 'keras')

MODULEPATH = os.path.join(os.path.dirname(__file__), '..')
if MODULEPATH not in sys.path:
    sys.path.append(MODULEPATH)


from config import BIN_DIR
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
                        help=("""Test running inference from saved
                              (trained) model specifically."""))

    parser.add_argument('--test_resume_training', action='store_true',
                        help=("""Test resuming training from checkpoint.
                              Must specify `--log_dir`."""))

    parser.add_argument('--log_dir', default=None, type=str,
                        help=("""`log_dir` from which to load saved model "
                              "for running inference on."""))

    parser.add_argument('--with_comparisons', action='store_true',
                        help=("""Try switching on and off some variables and
                              comparing."""))
    args = parser.parse_args()

    return args


def parse_test_configs(
    configs: dict[str, Any] = None,
    configs_fpath: str = None
):
    """Parse `test_config.json`."""
    if configs is None:
        configs = get_test_configs()
    if configs_fpath is not None:
        with open(configs_fpath, 'rt') as f:
            configs = json.load(f)

    #  test_flags = AttrDict(dict(test_flags))
    #  for key, val in test_flags.items():
    #      if isinstance(val, dict):
    #          test_flags[key] = AttrDict(val)

    return configs


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
        **kwargs,
) -> TestOutputs:
    """Test convolutional networks."""
    t0 = time.time()
    logger.info(f'Testing convolutional network')
    configs = AttrDict(**dict(copy.deepcopy(configs)))
    #  flags.use_conv_net = True
    configs['dynamics_config']['use_conv_net'] = True
    configs['conv_config'] = dict(
        sizes=[2, 2],
        filters=[4, 8],
        pool_sizes=[2, 2],
        use_batch_norm=True,
        conv_paddings=['valid', 'valid'],
        conv_activations=['relu', 'relu'],
        input_shape=configs['dynamics_config']['x_shape'][1:],
    )
    train_out = train(configs, make_plots=make_plots,
                      num_chains=4, verbose=False, **kwargs)
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
        **kwargs,
) -> TestOutputs:
    """Test training on single network."""
    t0 = time.time()
    logger.info(f'Testing single network')
    configs_ = dict(copy.deepcopy(configs))
    configs_['dynamics_config']['separate_networks'] = False
    train_out = train(configs_, make_plots=make_plots,
                      verbose=False, num_chains=4, **kwargs)
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
        **kwargs,
) -> TestOutputs:
    """Test training on separate networks."""
    t0 = time.time()
    logger.info(f'Testing separate networks')
    configs_ = dict(copy.deepcopy(configs))
    configs_['hmc_steps'] = 0
    configs_['dynamics_config']['separate_networks'] = True
    configs_['compile'] = False
    train_out = train(configs_, make_plots=make_plots,
                      verbose=False, num_chains=4, **kwargs)
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
        **kwargs,
) -> TestOutputs:
    """Test restoring a training session from a checkpoint."""
    t0 = time.time()
    logger.info(f'Testing resuming training')

    configs_ = copy.deepcopy(configs)
    assert configs_.get('restore_from', None) is not None

    train_out = train(configs_, make_plots=make_plots,
                      verbose=False, num_chains=4, **kwargs)
    dynamics = train_out.dynamics
    logdir = train_out.logdir
    x = train_out.x
    runs_dir = os.path.join(logdir, 'inference')
    run_out = None
    if RANK == 0:
        run_out = run(dynamics, configs_, x=x, runs_dir=runs_dir)

    logger.info(f'Passed! Took: {time.time() - t0:.4f} seconds')
    return TestOutputs(train_out, run_out)


def get_test_configs(updates: dict[str, Any] = None) -> dict[str, Any]:
    """Get fresh copy of `bin/test_configs.json` for running tests."""
    fpath = Path(BIN_DIR).joinpath('test_configs.json')
    with open(fpath, 'r') as f:
        configs = json.load(f)

    if updates is not None:
        dconfig = updates.get('dynamics_config', None)
        nconfig = updates.get('network_config', None)
        lconfig = updates.get('lr_config', None)
        cconfig = updates.get('conv_config', None)
        if dconfig is not None:
            configs['dynamics_config'].update(dconfig)
        if nconfig is not None:
            configs['network_config'].update(nconfig)
        if lconfig is not None:
            configs['lr_config'].update(lconfig)
        if cconfig is not None:
            configs['conv_config'].update(cconfig)

    return configs

#
#  @dataclass
#  class Configs:
#      train_config: TrainConfig
#
#
#  @dataclass
#  class TrainConfig:
#      log_dir: str = None
#      ensure_new: bool = False
#      profiler: bool = False
#      beta_init: float = 1.
#      beta_final: float = 1.
#      clip_val: float = 0.0
#      loss_scale: float = 0.1
#      min_lr: float = 1e-5
#      patience: int = 1
#      hmc_steps: int = 10
#      md_steps: int = 100
#      steps_per_epoch: int = 5
#      train_steps: int = 500
#      print_steps: int = 10
#      save_steps: int = 50
#      logging_steps: int = 10
#      run_steps: int = 500
#
#  @dataclass
#  class StepsConfig:
#      train: int          # -- num training steps --------------
#      run: int            # -- num inference steps -------------
#      md: int = 0         # -- Steps prior to training to avoid stuck chains
#      hmc: int = 0        # -- Steps prior to training to `warm up` from HMC
#      print: int = 1      # -- How frequently metrics are printed
#      logging: int = 1    # -- How frequently metrics are logged to TensorBoard
#      per_epoch: int = 1  # -- How long to wait inside a plateau before reducing lr
#
#
#  @dataclass DynamicsConfig:
#      verbose: bool = False
#      eps: float = 0.001
#      num_steps: int = 5
#      hmc: bool = False
#      use_ncp: bool = True
#      eps_fixed: bool = False
#      aux_weight: float = 0.
#      plaq_weight: float = 0.
#      charge_weight: float = 0.01
#      zero_init: bool = False
#      separate_networks: bool = True
#      use_conv_net: bool = False
#      use_mixed_loss: bool = False
#      directional_updates: bool = False
#      combined_updates: bool = False
#      use_scattered_xnet_update: False
#      use_tempered_traj: bool = False
#      gauge_eq_masks: bool = False
#      x_shape: tuple[int] = (None, 16, 16, 2)
#      log_dir: None
#
#
#  @dataclass
#  class NetworkConfig:
#      units: list[int]
#      activation_fn: str = 'relu'
#      dropout_prob: float = 0.05
#      use_batch_norm: bool = True
#
#

def test_aux_weight(configs: dict[str, Any] = None, **kwargs):
    if configs is None:
        configs = get_test_configs()

    configs_ = copy.deepcopy(configs)

    aw = configs['dynamics_config']['aux_weight']  # type: float
    configs_['dynamics_config']['aux_weight'] = float(not bool(aw))
    sep_nets = configs['dynamics_config']['separate_networks']  # type: bool
    test_fn = test_separate_networks if sep_nets else test_single_network

    return test_fn(configs_, **kwargs)

def test_mixed_loss(configs: dict[str, Any] = None, **kwargs):
    if configs is None:
        configs = get_test_configs()

    configs_ = copy.deepcopy(configs)
    mixed_loss = configs['dynamics_config']['use_mixed_loss']  # type: bool
    sep_nets = configs['dynamics_config']['separate_networks']

    configs_['dynamics_config']['use_mixed_loss'] = not mixed_loss
    test_fn = test_separate_networks if sep_nets else test_single_network

    return test_fn(configs_, **kwargs)


def setup_betas(configs):
    b0 = configs.get('beta_init', None)  # type: float
    b1 = configs.get('beta_final', None)  # type: float
    nb = int(configs.get('train_steps', None) // (b1 + 1 - b0))
    betas = []
    for b in range(int(b0), int(b1+1)):
        betas_ = b * np.ones(nb)
        betas.append(betas_)

    betas = np.stack(np.array(betas))
    betas = tf.convert_to_tensor(betas.flatten(),
                                        dtype=tf.keras.backend.floatx())
    return betas


def test(make_plots: bool = False, with_comparisons: bool = False):
    """Run tests."""
    t0 = time.time()
    configs = parse_test_configs()
    betas = None
    if configs.get('discrete_beta', configs.get('discrete_betas', False)):
        betas = setup_betas(configs)


    sep_configs = copy.deepcopy(configs)
    conv_configs = copy.deepcopy(configs)
    single_configs = copy.deepcopy(configs)

    sep_configs = get_test_configs()
    sep_configs['profiler'] = True
    sep_out = test_separate_networks(sep_configs,
                                     make_plots=make_plots,
                                     custom_betas=betas)
    sep_configs['profiler'] = False
    if with_comparisons:
        test_aux_weight(configs, make_plots=make_plots, custom_betas=betas)
        test_mixed_loss(configs, make_plots=make_plots, custom_betas=betas)


    sep_configs['train_steps'] += 10
    sep_configs['restore_from'] = sep_out.train.logdir
    sep_configs['log_dir'] = None
    _ = test_resume_training(sep_configs,
                             make_plots=make_plots,
                             custom_betas=betas)

    bf = sep_configs.get('beta_final')
    beta_final = bf + 1
    sep_configs['beta_final'] = beta_final
    logger.log(f'Increasing beta: {bf} -> {beta_final}')
    sep_configs['ensure_new'] = True
    sep_configs['beta_final'] = sep_configs['beta_final'] + 1

    _ = test_resume_training(sep_configs,
                             make_plots=make_plots,
                             custom_betas=betas)

    single_net_out = test_single_network(single_configs,
                                         make_plots=make_plots,
                                         custom_betas=betas)
    single_configs['restore_from'] = single_net_out.train.logdir
    single_configs['dynamics_config']['separate_networks'] = False
    _ = test_resume_training(single_configs,
                             make_plots=True,
                             custom_betas=betas)

    configs['ensure_new'] = True
    configs['log_dir'] = None
    _ = test_separate_networks(configs,
                               make_plots=True,
                               custom_betas=betas)

    configs['ensure_new'] = True
    configs['log_dir'] = None
    _ = test_single_network(configs,
                            make_plots=True,
                            custom_betas=betas)

    _ = test_conv_net(conv_configs, make_plots=make_plots, custom_betas=betas)

    updates = {
        'dynamics-config': {
            'aux_weight': 1.0,
            'use_mixed_loss': True,
        },
    }
    configs = get_test_configs(updates)
    _ = test_conv_net(configs, make_plots=make_plots, custom_betas=betas)
    _ = test_separate_networks(configs, make_plots=True, custom_betas=betas)

    if RANK  == 0:
        _ = test_hmc_run(configs, make_plots=True)

    logger.info(f'All tests passed! Took: {time.time() - t0:.4f} s')


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

    _ = (
        test(ARGS.make_plots, ARGS.with_comparisons) if ARGS.test_all
        else main(ARGS)
    )
