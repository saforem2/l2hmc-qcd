"""
hvd_init.py

Initialize Horovod and check ranks.
"""
from __future__ import absolute_import, division, print_function, annotations

import logging
import tensorflow as tf

RANK = 0
SIZE = 1
LOCAL_SIZE = 1
LOCAL_RANK = 0
IS_CHIEF = True
HAS_HOROVOD = False

try:
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
    #  logging.info(f'using horovod from: {horovod.__file__}')
    #  logging.info(f'using horovod version: {horovod.__version__}')
    prefix = f'{RANK} / {SIZE} ::'
    logging.info(80 * '=')
    logging.info(f'{prefix} Using tensorflow version: {tf.__version__}')
    logging.info(f'{prefix} Using tensorflow from: {tf.__file__}')
    logging.info(f'{prefix} Using horovod version: {horovod.__version__}')
    logging.info(f'{prefix} Using horovod from: {horovod.__file__}')
    logging.info(80 * '=')
    GPUS = tf.config.experimental.list_physical_devices('GPU')
    for gpu in GPUS:
        tf.config.experimental.set_memory_growth(gpu, True)
    if GPUS:
        gpu = GPUS[hvd.local_rank()]
        tf.config.experimental.set_visible_devices(gpu, 'GPU')

except (ImportError, ModuleNotFoundError) as err:
    logging.warning('Unable to initialize horovod!!!')
    logging.warning(err)
