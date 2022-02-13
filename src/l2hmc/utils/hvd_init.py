"""
utils/hvd_init.py
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
log = logging.getLogger(__name__)

try:
    import tensorflow as tf
    import horovod
    import horovod.tensorflow as hvd
    try:
        RANK = hvd.rank()
    except ValueError:
        hvd.init()

    RANK = hvd.rank()
    SIZE = hvd.size()
    HAS_HOROVOD = True
    IS_CHIEF = (RANK == 0)
    LOCAL_SIZE = hvd.local_size()
    LOCAL_RANK = hvd.local_rank()
    # tf.random.set_seed(int(RANK * 1234))
    #  logging.info(f'using horovod from: {horovod.__file__}')
    #  logging.info(f'using horovod version: {horovod.__version__}')
    prefix = f'{RANK} / {SIZE} ::'
    if IS_CHIEF:
        log.info(f'{prefix} Using tensorflow version: {tf.__version__}')
        log.info(f'{prefix} Using tensorflow from: {tf.__file__}')
        log.info(f'{prefix} Using horovod version: {horovod.__version__}')
        log.info(f'{prefix} Using horovod from: {horovod.__file__}')
    else:
        log.info(f'Hello, im rank: {RANK} of {SIZE} total ranks')

    GPUS = tf.config.experimental.list_physical_devices('GPU')
    if len(GPUS) > 0:
        for gpu in GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)
    if GPUS:
        gpu = GPUS[hvd.local_rank()]
        tf.config.experimental.set_visible_devices(gpu, 'GPU')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


except (ImportError, ModuleNotFoundError):
    RANK = 0
    SIZE = 1
    LOCAL_SIZE = 1
    LOCAL_RANK = 0
    IS_CHIEF = True
    HAS_HOROVOD = False
    log.error('Unable to initialize horovod!!!')


def shutdown():
    if HAS_HOROVOD:
        hvd.shutdown()
    return
