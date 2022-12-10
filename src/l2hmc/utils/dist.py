"""
l2hmc/utils/dist.py

Contains methods for initializing distributed communication.
"""
from __future__ import absolute_import, annotations, division, print_function

import os

from typing import Optional, Callable
from mpi4py import MPI

# from l2hmc.utils.logger import get_pylogger
import logging

log = logging.getLogger(__name__)


BACKENDS = [
    'deepspeed',
    'ds',
    'ddp',
    'horovod',
    'hvd',
]


def setup_tensorflow(
        precision: Optional[str] = None,
        ngpus: Optional[int] = None,
) -> int:
    import tensorflow as tf
    dtypes = {
        'float16': tf.float16,
        'float32': tf.float32,
        'float64': tf.float64,
    }
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
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    os.environ['LOCAL_RANK'] = str(LOCAL_RANK)

    log.warning(f'Using: {TF_FLOAT} precision')
    log.info(f'Global Rank: {RANK} / {SIZE-1}')
    log.info(f'[{RANK}]: Local rank: {LOCAL_RANK} / {LOCAL_SIZE-1}')
    return RANK


def init_deepspeed():
    import deepspeed
    deepspeed.init_distributed()


def init_process_group(
        rank: int | str,
        world_size: int | str,
        backend: Optional[str] = None,
) -> None:
    import torch
    import torch.distributed as dist
    if torch.cuda.is_available():
        backend = 'nccl' if backend is None else str(backend)
    else:
        backend = 'gloo' if backend is None else str(backend)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=int(rank),
            world_size=int(world_size),
            init_method='env://',
        )


def run_ddp(fn: Callable, world_size: int) -> None:
    import torch.multiprocessing as mp
    mp.spawn(  # type:ignore
        fn,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


def get_rank():
    return MPI.COMM_WORLD.Get_rank()


def get_size():
    return MPI.COMM_WORLD.Get_size()


def get_local_rank():
    return os.environ.get(
        'PMI_LOCAL_RANK',
        os.environ.get(
            'OMPI_COMM_WORLD_LOCAL_RANK',
            os.environ.get(
                'LOCAL_RANK',
                '0'
            )
        )
    )


def query_environment() -> dict[str, int]:
    """Query environment variables for info about distributed setup"""
    ws = os.environ.get('WORLD_SIZE', None)
    r = os.environ.get('RANK', None)
    lr = os.environ.get('LOCAL_RANK', None)
    if ws is not None and r is not None and lr is not None:
        return {
            'size': int(ws),
            'rank': int(r),
            'local_rank': int(lr)
        }

    return {
        'size': int(get_size()),
        'rank': int(get_rank()),
        'local_rank': int(get_local_rank()),
    }


def setup_torch_DDP(port: str = '2345') -> dict[str, int]:
    import torch
    rank = os.environ.get('RANK', None)
    size = os.environ.get('WORLD_SIZE', None)
    local_rank = os.environ.get('LOCAL_RANK', None)

    import socket
    # local_rank = int(os.environ.get(
    #     'PMI_LOCAL_RANK',
    #     os.environ.get(
    #         'OMPI_COMM_WORLD_LOCAL_RANK',
    #         '0',
    #     )
    # ))
    size = int(get_size())
    rank = int(get_rank())
    local_rank = int(get_local_rank())
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(size)
    master_addr = (
        socket.gethostname() if rank == 0 else None
    )
    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ['MASTER_ADDR'] = master_addr
    if (eport := os.environ.get('MASTER_PORT', None)) is None:
        os.environ['MASTER_PORT'] = port
    else:
        log.info(f'Caught MASTER_PORT:{eport} from environment!')
        os.environ['MASTER_PORT'] = eport

    init_process_group(
        rank=rank,
        world_size=size,
        backend='nccl' if torch.cuda.is_available() else 'gloo'
    )

    return {'size': size, 'rank': rank, 'local_rank': local_rank}


def setup_torch_distributed(
        backend: str,
        port: str = '2345',
) -> dict:
    rank = os.environ.get('RANK', None)
    size = os.environ.get('WORLD_SIZE', None)
    local_rank = os.environ.get(
        'PMI_LOCAL_RANK',
        os.environ.get(
            'OMPI_COMM_WORLD_LOCAL_RANK',
            None
        )
    )
    be = backend.lower()
    assert be in BACKENDS

    if rank == 0 and local_rank == 0:
        log.info(f'Using {backend} for distributed training')

    if be in ['ddp', 'DDP']:
        dsetup = setup_torch_DDP(port)
        size = dsetup['size']
        rank = dsetup['rank']
        local_rank = dsetup['local_rank']

    elif be in ['deepspeed', 'ds']:
        init_deepspeed()
        size = get_size()
        rank = get_rank

    elif be in ['horovod', 'hvd']:
        import horovod.torch as hvd
        hvd.init() if not hvd.is_initialized() else None
        rank = hvd.rank()
        size = hvd.size()
        local_rank = hvd.local_rank()
    else:
        raise ValueError
        # log.warning(f'Unexpected backend specified: {backend}')
        # log.error('Setting size = 1, rank = 0, local_rank = 0')
        # size = 1
        # rank = 0
        # local_rank = 0

    os.environ['SIZE'] = str(size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)

    return {'size': size, 'rank': rank, 'local_rank': local_rank}


def setup_torch(
        seed: int,
        backend: str = 'horovod',
        precision: str = 'float32',
        port: str = '2345',
) -> int:
    import torch
    from l2hmc.common import seed_everything
    dtypes = {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True   # type:ignore
    torch.backends.cudnn.benchmark = True       # type:ignore
    torch.use_deterministic_algorithms(True)
    dsetup = setup_torch_distributed(backend=backend, port=port)
    rank = dsetup['rank']
    size = dsetup['size']
    local_rank = dsetup['local_rank']
    # size = int(get_size())
    # rank = int(get_rank())
    # local_rank = int(get_local_rank())
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(size)

    nthreads = os.environ.get(
        'OMP_NUM_THREADS',
        None
    )
    if nthreads is not None:
        torch.set_num_threads(int(nthreads))

    # if precision == 'float64':
    torch.set_default_dtype(dtypes.get(precision, torch.float32))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    log.info(f'Global Rank: {rank} / {size-1}')
    log.info(f'[{rank}]: Local rank: {local_rank}')
    seed_everything(seed * (rank + 1) * (local_rank + 1))
    return rank


def cleanup() -> None:
    import torch.distributed as tdist
    tdist.destroy_process_group()
