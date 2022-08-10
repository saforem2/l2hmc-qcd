"""
main.py

Contains entry point for training Dynamics.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
import random
import warnings
import time

import hydra
from typing import Optional
import numpy as np
from omegaconf import DictConfig

from l2hmc.configs import ExperimentConfig
from l2hmc.utils.rich import print_config

log = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.CRITICAL)
logging.getLogger('graphviz._tools').setLevel(logging.CRITICAL)
logging.getLogger('graphviz').setLevel(logging.CRITICAL)


def seed_everything(seed: int):
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup(cfg: DictConfig):
    if cfg.get('ignore_warnings'):
        warnings.filterwarnings('ignore')


def setup_tensorflow(precision: Optional[str] = None) -> int:
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import horovod.tensorflow as hvd
    hvd.init() if not hvd.is_initialized() else None
    tf.keras.backend.set_floatx(precision)
    TF_FLOAT = tf.keras.backend.floatx()
    # tf.config.run_functions_eagerly(True)
    # assert tf.keras.backend.floatx() == tf.float32
    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(
                gpus[hvd.local_rank()],
                'GPU',
            )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            log.info(
                f'{len(gpus)}, Physical GPUs and '
                f'{len(logical_gpus)} Logical GPUs'
            )
        except RuntimeError as e:
            print(e)
    elif cpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            logical_cpus = tf.config.experimental.list_logical_devices('CPU')
            log.info(
                f'{len(cpus)}, Physical CPUs and '
                f'{len(logical_cpus)} Logical CPUs'
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    RANK = hvd.rank()
    SIZE = hvd.size()
    LOCAL_RANK = hvd.local_rank()
    LOCAL_SIZE = hvd.local_size()

    log.warning(f'Using: {TF_FLOAT} precision')
    log.info(f'Global Rank: {RANK} / {SIZE-1}')
    log.info(f'[{RANK}]: Local rank: {LOCAL_RANK} / {LOCAL_SIZE-1}')
    return RANK


def setup_torch(
        precision: Optional[str] = None,
        seed: Optional[int] = None,
) -> int:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.backends.cudnn.deterministic = True   # type:ignore
    torch.backends.cudnn.benchmark = True       # type:ignore
    torch.use_deterministic_algorithms(True)
    # torch.manual_seed(cfg.seed)
    import horovod.torch as hvd

    hvd.init() if not hvd.is_initialized() else None

    nthreads = os.environ.get('OMP_NUM_THREADS', '1')
    torch.set_num_threads(int(nthreads))

    if precision == 'float64':
        torch.set_default_dtype(torch.float64)

    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())
        # torch.cuda.manual_seed(cfg.seed + hvd.local_rank())
    # else:
    #     torch.set_default_dtype(torch.float32)
    RANK = hvd.rank()
    LOCAL_RANK = hvd.local_rank()
    SIZE = hvd.size()
    LOCAL_SIZE = hvd.local_size()

    log.info(f'Global Rank: {RANK} / {SIZE-1}')
    log.info(f'[{RANK}]: Local rank: {LOCAL_RANK} / {LOCAL_SIZE-1}')
    seed_everything(seed * (RANK + 1) * (LOCAL_RANK + 1))
    return RANK


def get_experiment(
        cfg: DictConfig,
        keep: Optional[str | list[str]] = None,
        skip: Optional[str | list[str]] = None,
):
    framework = cfg.get('framework', None)
    os.environ['RUNDIR'] = str(os.getcwd())
    if framework in ['tf', 'tensorflow']:
        _ = setup_tensorflow(cfg.precision)
        from l2hmc.experiment.tensorflow.experiment import Experiment
        experiment = Experiment(
            cfg,
            keep=keep,
            skip=skip
        )
        return experiment

    if framework in ['pt', 'pytorch', 'torch']:
        _ = setup_torch(
            precision=cfg.precision,
            seed=cfg.seed
        )
        from l2hmc.experiment.pytorch.experiment import Experiment
        # init = (RANK == 0) and not os.environ.get('WANDB_OFFLINE', False)
        # init_wandb = (RANK == 0 and cfg.get('init_wandb', False))
        # init_aim = (RANK == 0 and cfg.get('init_aim', False))
        experiment = Experiment(
            cfg,
            keep=keep,
            skip=skip,
            # init_wandb=init_wandb,
            # init_aim=init_aim,
        )
        # init = (RANK == 0)
        # _ = experiment.build(init_wandb=init, init_aim=init)
        return experiment
        # log.warning('Initializing network weights...')
        # ex.dynamics.networks['xnet'].apply(init_weights)
        # ex.dynamics.networks['vnet'].apply(init_weights)

    raise ValueError(
        'Framework must be specified, one of: [pytorch, tensorflow]'
    )


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # --- [0.] Setup ------------------------------------------------------
    setup(cfg)
    ex = get_experiment(cfg)
    assert isinstance(ex.config, ExperimentConfig)

    # if ex.trainer.rank == 0:
    if ex.trainer._is_chief:
        print_config(ex.cfg, resolve=True)

    should_train: bool = (
        ex.config.steps.nera > 0
        and ex.config.steps.nepoch > 0
    )

    # --- [1.] Train model -------------------------------------------------
    if should_train:
        tstart = time.time()
        _ = ex.train()
        log.info(f'Training took: {time.time() - tstart:.5f}s')
        # --- [2.] Evaluate trained model ----------------------------------
        # if ex.trainer.rank == 0 and ex.config.steps.test > 0:
        if ex.trainer._is_chief and ex.config.steps.test > 0:
            log.info('Evaluating trained model')
            estart = time.time()
            _ = ex.evaluate(job_type='eval')
            log.info(f'Evaluation took: {time.time() - estart:.5f}s')

            try:
                ex.visualize_model()
            except AttributeError as e:
                log.exception(e)

    # --- [3.] Run generic HMC for comparison ------------------------------
    if ex.trainer._is_chief and ex.config.steps.test > 0:
        log.info('Running generic HMC for comparison')
        hstart = time.time()
        _ = ex.evaluate(job_type='hmc')
        log.info(f'HMC took: {time.time() - hstart:.5f}s')
        from l2hmc.utils.plot_helpers import measure_improvement
        measure_improvement(
            experiment=ex,
            title=f'{ex.config.framework}',
        )


if __name__ == '__main__':
    import wandb
    wandb.require(experiment='service')
    main()
