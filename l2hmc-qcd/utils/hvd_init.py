"""
hvd_init.py

Initialize Horovod and check ranks.
"""
from __future__ import absolute_import, division, print_function, annotations

import os
import tensorflow as tf
from utils.logger import Logger

DEFAULT_INTEROP = int(os.cpu_count() / 4)
DEFAULT_INTRAOP = int(os.cpu_count() / 4)

tf.random.set_seed(1234)
try:
    tf.config.threading.set_inter_op_parallelism_threads(DEFAULT_INTEROP)
    tf.config.threading.set_intra_op_parallelism_threads(DEFAULT_INTRAOP)
except RuntimeError:
    pass

logger = Logger()

if tf.__version__.startswith('1'):
    try:
        tf.compat.v1.enable_v2_behavior()
    except AttributeError:
        print('Unable to call \n'
              '`tf.compat.v1.enable_v2_behavior()`. Continuing...')
    try:
        tf.compat.v1.enable_control_flow_v2()
    except AttributeError:
        print('Unable to call \n'
              '`tf.compat.v1.enable_control_flow_v2()`. Continuing...')
    try:
        tf.compat.v1.enable_v2_tensorshape()
    except AttributeError:
        print('Unable to call \n'
              '`tf.compat.v1.enable_v2_tensorshape()`. Continuing...')
    try:
        tf.compat.v1.enable_eager_execution()
    except AttributeError:
        print('Unable to call \n'
              '`tf.compat.v1.enable_eager_execution()`. Continuing...')
    try:
        tf.compat.v1.enable_resource_variables()
    except AttributeError:
        print('Unable to call \n'
              '`tf.compat.v1.enable_resource_variables()`. Continuing...')

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
    if IS_CHIEF:
        logger.info(80 * '=')
        logger.info(f'{prefix} Using tensorflow version: {tf.__version__}')
        logger.info(f'{prefix} Using tensorflow from: {tf.__file__}')
        logger.info(f'{prefix} Using horovod version: {horovod.__version__}')
        logger.info(f'{prefix} Using horovod from: {horovod.__file__}')
        logger.info(80 * '=')
    else:
        logger.info(f'Hello, im rank: {RANK} of {SIZE} total ranks')

    GPUS = tf.config.experimental.list_physical_devices('GPU')
    for gpu in GPUS:
        tf.config.experimental.set_memory_growth(gpu, True)
    if GPUS:
        gpu = GPUS[hvd.local_rank()]
        tf.config.experimental.set_visible_devices(gpu, 'GPU')

except (ImportError, ModuleNotFoundError) as err:
    RANK = 0
    SIZE = 1
    LOCAL_SIZE = 1
    LOCAL_RANK = 0
    IS_CHIEF = True
    HAS_HOROVOD = False
    logger.error('Unable to initialize horovod!!!')
    logger.error(err)
