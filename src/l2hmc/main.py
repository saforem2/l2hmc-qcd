"""
main.py

Contains entry point for training Dynamics.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
import sys
import random
import warnings
import time
from pathlib import Path

import torch
import torch.distributed as dist

import hydra
from typing import Optional
import numpy as np
from omegaconf.dictconfig import DictConfig

from l2hmc.configs import ExperimentConfig
from l2hmc.utils.rich import print_config
from l2hmc.utils.plot_helpers import set_plot_style

set_plot_style()

log = logging.getLogger(__name__)

logging.getLogger('filelock').setLevel(logging.CRITICAL)
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


def run_ddp(fn, world_size):
    import torch.multiprocessing as mp
    mp.spawn(  # type:ignore
        fn,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


def cleanup() -> None:
    dist.destroy_process_group()


def setup_tensorflow(
        precision: Optional[str] = None,
        ngpus: Optional[int] = None,
) -> int:
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import horovod.tensorflow as hvd
    hvd.init() if not hvd.is_initialized() else None
    tf.keras.backend.set_floatx(precision)
    TF_FLOAT = tf.keras.backend.floatx()
    eager_mode = os.environ.get('TF_EAGER', None)
    if eager_mode is not None:
        log.warning('Detected `TF_EAGER` from env. Running eagerly.')
        tf.config.run_functions_eagerly(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')
    if gpus:
        try:
            # Currently memory growth needs to be the same across GPUs
            if ngpus is not None:
                gpus = gpus[-ngpus:]

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
        seed: int,
        backend: str = 'horovod',
        precision: str = 'float32',
        port: str = '2345',
) -> int:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True   # type:ignore
    torch.backends.cudnn.benchmark = True       # type:ignore
    torch.use_deterministic_algorithms(True)
    # torch.manual_seed(cfg.seed)
    from l2hmc.common import setup_torch_distributed
    dsetup = setup_torch_distributed(backend=backend, port=port)
    rank = dsetup['rank']
    size = dsetup['size']
    local_rank = dsetup['local_rank']

    nthreads = os.environ.get(
        'OMP_NUM_THREADS',
        None
    )
    if nthreads is not None:
        torch.set_num_threads(int(nthreads))

    if precision == 'float64':
        torch.set_default_dtype(torch.float64)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        # torch.cuda.manual_seed(cfg.seed + hvd.local_rank())
    # else:
    #     torch.set_default_dtype(torch.float32)
    # RANK = hvd.rank()
    # LOCAL_RANK = hvd.local_rank()
    # SIZE = hvd.size()
    # LOCAL_SIZE = hvd.local_size()

    log.info(f'Global Rank: {rank} / {size-1}')
    log.info(f'[{rank}]: Local rank: {local_rank}')
    seed_everything(seed * (rank + 1) * (local_rank + 1))
    return rank


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
            seed=cfg.seed,
            precision=cfg.precision,
            backend=cfg.get('backend', 'horovod'),
            port=cfg.get('port', '2345')
        )
        from l2hmc.experiment.pytorch.experiment import Experiment
        experiment = Experiment(cfg, keep=keep, skip=skip)
        return experiment

    raise ValueError(
        'Framework must be specified, one of: [pytorch, tensorflow]'
    )


def run(cfg: DictConfig) -> str:
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

    nchains_eval = max(2, int(ex.config.dynamics.xshape[0] // 4))


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
            _ = ex.evaluate(job_type='eval', nchains=nchains_eval)
            log.info(f'Evaluation took: {time.time() - estart:.5f}s')

            try:
                ex.visualize_model()
            except AttributeError as e:
                log.exception(e)

    # --- [3.] Run generic HMC for comparison ------------------------------
    if ex.trainer._is_chief and ex.config.steps.test > 0:
        log.info('Running generic HMC for comparison')
        hstart = time.time()
        _ = ex.evaluate(job_type='hmc', nchains=nchains_eval)
        log.info(f'HMC took: {time.time() - hstart:.5f}s')
        from l2hmc.utils.plot_helpers import measure_improvement
        improvement = measure_improvement(
            experiment=ex,
            title=f'{ex.config.framework}',
        )
        log.critical(f'Model improvement: {improvement:.8f}')

    return Path(ex._outdir).as_posix()


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig):
    output = run(cfg)
    if str(cfg.get('backend', '')).lower() == 'ddp':
        cleanup()

    return output


if __name__ == '__main__':
    import wandb
    wandb.require(experiment='service')
    start = time.time()
    outdir = main()
    end = time.time()
    if outdir is not None:
        log.info(f'Run completed in: {end - start:4.4f} s')
        log.info(f'Run located in: {outdir}')
    sys.exit(0)
