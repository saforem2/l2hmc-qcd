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
import socket


log = logging.getLogger(__name__)


def train_tensorflow(cfg: DictConfig) -> dict:
    import tensorflow as tf
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

    from l2hmc.scripts.tensorflow.main import main as main_tf
    output = main_tf(cfg)

    return output


def train_pytorch(cfg: DictConfig) -> dict:
    import torch
    try:
        from mpi4py import MPI
        # WITH_DDP = True
        LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
        SIZE = MPI.COMM_WORLD.Get_size()
        RANK = MPI.COMM_WORLD.Get_rank()
        # WITH_CUDA = torch.cuda.is_available()
        # DEVICE = 'gpu' if WITH_CUDA else 'CPU'
        # pytorch will look for these
        os.environ['RANK'] = str(RANK)
        os.environ['WORLD_SIZE'] = str(SIZE)
        # ----------------------------------------------
        # NOTE: We get the hostname of the master node
        # and broadcast it to all other nodes.
        # It will want the master address too,
        # which we'll also broadcast
        # ----------------------------------------------
        MASTER_ADDR = socket.gethostname() if RANK == 0 else None
        MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = str(9992)
    except (ImportError, ModuleNotFoundError) as e:
        SIZE = 1
        RANK = 0
        # WITH_DDP = False
        # LOCAL_RANK = 0
        MASTER_ADDR = 'localhost'
        log.warning('MPI Initialization Failed!')
        log.warning(e)

    if cfg.precision == 'float64':
        torch.set_default_dtype(torch.float64)
    # else:
    #     torch.set_default_dtype(torch.float32)

    from l2hmc.scripts.pytorch.main import main as main_pt
    return main_pt(cfg)


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    width = cfg.get('width', None)
    if width is not None and os.environ.get('COLUMNS', None) is None:
        os.environ['COLUMNS'] = str(width)
    elif os.environ.get('COLUMNS', None) is not None:
        cfg.update({'width': int(os.environ.get('COLUMNS', 235))})

    framework = cfg.get('framework', None)
    assert framework is not None, (
        'Framework must be specified, one of: [pytorch, tensorflow]'
    )

    if cfg.get('ignore_warnings'):
        warnings.filterwarnings('ignore')

    if framework in ['tf', 'tensorflow']:
        _ = train_tensorflow(cfg)

    elif framework in ['pt', 'pytorch']:
        _ = train_pytorch(cfg)


if __name__ == '__main__':
    main()
