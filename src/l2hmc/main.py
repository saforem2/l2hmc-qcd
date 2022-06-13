"""
main.py

Contains entry point for training Dynamics.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
import warnings

import hydra
from omegaconf import DictConfig
# import socket

from l2hmc.configs import ExperimentConfig
from l2hmc.utils.rich import print_config


log = logging.getLogger(__name__)


def setup_tensorflow(cfg: DictConfig) -> int:
    import tensorflow as tf
    # tf.config.run_functions_eagerly(True)
    tf.keras.backend.set_floatx(cfg.precision)
    # assert tf.keras.backend.floatx() == tf.float32
    import horovod.tensorflow as hvd
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(
            gpus[hvd.local_rank()],
            'GPU'
        )

    # from l2hmc.scripts.tensorflow.main import main as main_tf
    RANK = hvd.rank()
    LOCAL_RANK = hvd.local_rank()
    SIZE = hvd.size()
    LOCAL_SIZE = hvd.local_size()
    log.warning(f'Global {RANK} / {SIZE}')
    log.warning(f'[{RANK}] local: {LOCAL_RANK} / {LOCAL_SIZE}')
    # if cfg.get('debug_mode', False):
    return RANK


def setup_torch(cfg: DictConfig) -> int:
    import torch
    import horovod.torch as hvd

    hvd.init()

    if cfg.precision == 'float64':
        torch.set_default_dtype(torch.float64)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(cfg.seed)
    # else:
    #     torch.set_default_dtype(torch.float32)
    RANK = hvd.rank()
    LOCAL_RANK = hvd.local_rank()
    SIZE = hvd.size()
    LOCAL_SIZE = hvd.local_size()
    log.info(f'Global Rank: {RANK} / {SIZE}')
    log.info(f'[{RANK}]: Local rank: {LOCAL_RANK} / {LOCAL_SIZE}')
    return RANK


def train_tensorflow(cfg: DictConfig) -> dict:
    RANK = setup_tensorflow(cfg)
    from l2hmc.experiment.tensorflow.experiment import Experiment
    outputs = {}
    ex = Experiment(cfg)
    _ = ex.build(init_wandb=(RANK == 0))
    assert isinstance(ex.config, ExperimentConfig)
    should_train = (
        ex.config.steps.nera > 0
        and ex.config.steps.nepoch > 0
    )
    if should_train:
        outputs['train'] = ex.train()
        # Evaluate trained model
        if RANK == 0 and ex.config.steps.test > 0:
            log.warning('Evaluating trained model')
            outputs['eval'] = ex.evaluate(job_type='eval')

    # Run generic HMC for baseline comparison
    if RANK == 0 and ex.config.steps.test > 0:
        log.warning('Running generic HMC with same traj len')
        outputs['hmc'] = ex.evaluate(job_type='hmc')

    return outputs


def train_pytorch(cfg: DictConfig) -> dict:
    RANK = setup_torch(cfg)

    from l2hmc.experiment.pytorch.experiment import Experiment

    outputs = {}
    ex = Experiment(cfg)
    _ = ex.build(init_wandb=(RANK == 0))
    assert isinstance(ex.config, ExperimentConfig)
    should_train = (
        ex.config.steps.nera > 0
        and ex.config.steps.nepoch > 0
    )
    if should_train:
        outputs['train'] = ex.train()
        # Evaluate trained model
        if RANK == 0 and ex.config.steps.test > 0:
            log.warning('Evaluating trained model')
            outputs['eval'] = ex.evaluate(job_type='eval')

    # Run generic HMC for baseline comparison
    if RANK == 0 and ex.config.steps.test > 0:
        log.warning('Running generic HMC with same traj len')
        outputs['hmc'] = ex.evaluate(job_type='hmc')

    return outputs


def setup(cfg: DictConfig):
    width = cfg.get('width', None)
    if width is not None and os.environ.get('COLUMNS', None) is None:
        os.environ['COLUMNS'] = str(width)
    elif os.environ.get('COLUMNS', None) is not None:
        cfg.update({'width': int(os.environ.get('COLUMNS', 235))})
    if cfg.get('ignore_warnings'):
        warnings.filterwarnings('ignore')


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    setup(cfg)
    framework = cfg.get('framework', None)
    if framework in ['tf', 'tensorflow']:
        RANK = setup_tensorflow(cfg)
        from l2hmc.experiment.tensorflow.experiment import Experiment
    elif framework in ['pt', 'pytorch', 'torch']:
        RANK = setup_torch(cfg)
        from l2hmc.experiment.pytorch.experiment import Experiment
    else:
        raise ValueError(
            'Framework must be specified, one of: [pytorch, tensorflow]'
        )

    if RANK == 0:
        print_config(cfg, resolve=True)

    ex = Experiment(cfg)
    _ = ex.build(init_wandb=(RANK == 0))
    assert isinstance(ex.config, ExperimentConfig)
    should_train = (
        ex.config.steps.nera > 0
        and ex.config.steps.nepoch > 0
    )
    if should_train:
        _ = ex.train()
        # Evaluate trained model
        if RANK == 0 and ex.config.steps.test > 0:
            log.warning('Evaluating trained model')
            _ = ex.evaluate(job_type='eval')

    # Run generic HMC for baseline comparison
    if RANK == 0 and ex.config.steps.test > 0:
        log.warning('Running generic HMC with same traj len')
        _ = ex.evaluate(job_type='hmc')


if __name__ == '__main__':
    import wandb
    wandb.require(experiment='service')
    main()
