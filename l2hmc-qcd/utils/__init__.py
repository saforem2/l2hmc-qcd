"""
__init__.py

Initialization module for running training/inference.

Author: Sam Foreman
Date: 08/26/2020
"""
import os
#  os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#  from tqdm.auto import tqdm
import tensorflow as tf
print(f'tf.__version__: {tf.__version__}')
if tf.__version__.startswith('1.'):
    def _try_except(function):
        try:
            function()
        except AttributeError:
            print(f'Unable to call `{function}`. Continuing...')

    _try_except(tf.compat.v1.enable_v2_behavior)
    _try_except(tf.compat.v1.enable_control_flow_v2)
    _try_except(tf.compat.v1.enable_v2_tensorshape)
    _try_except(tf.compat.v1.enable_eager_execution)
    _try_except(tf.compat.v1.enable_resource_variables)


try:
    import horovod
    import horovod.tensorflow as hvd  # pylint:disable=wrong-import-order
    try:
        RANK = hvd.rank()
    except ValueError:
        hvd.init()

    RANK = hvd.rank()
    HAS_HOROVOD = True
    NUM_WORKERS = hvd.size()
    #  hvd.init()
    GPUS = tf.config.experimental.list_physical_devices('GPU')
    for gpu in GPUS:
        tf.config.experimental.set_memory_growth(gpu, True)
    if GPUS:
        gpu = GPUS[hvd.local_rank()]  # pylint:disable=invalid-name
        tf.config.experimental.set_visible_devices(gpu, 'GPU')

except (ImportError, ModuleNotFoundError):
    HAS_HOROVOD = False
    RANK = LOCAL_RANK = 0
    SIZE = LOCAL_SIZE = 1
    IS_CHIEF = (RANK == 0)

_d = ('f', 'b', 'forward', 'backward')
_t = ('start', 'mid', 'end')

bad = ['sumlogdet' 'sumlogdetf', 'sumlogdetf_start', 'sumlogdetf_mid',
       'sumlogdetf_end']


BASE_KEYS = [
    'H', 'ld', 'sld',
    'sinQ', 'intQ', # 'plaqs',
    'xeps', 'veps', # 'accept_prob',
    'logdet', 'logdets', 'sumlogdet',
    'accept_mask', 'xeps', 'veps', 'charges',
    'accept_probf', 'accept_probb',
    'plaqsf', 'plaqsb',
]

DIR_KEYS = [
    'accept_prob', 'plaqs', 'H',  # , 'Hw',
    'xeps', 'veps', 'logdet', 'logdets', 'sumlogdet',
    'accept_mask', 'charges'
]

SKEYS = [
    'forward', 'backward', 'sumlogdet_prop',
]
for bk in BASE_KEYS:
    SKEYS.append(f'{bk}')
    #  for dk in DIR_KEYS:
    #      for sk in SKEYS:
    #          SKEYS.append(f'{dk}{sk}')
    for d in _d:                          # =====
        SKEYS.append(f'{bk}{d}')          # 'Hf', 'Hb', 'Hwf', 'Hwb', etc.
        for t in _t:                      # ====
            SKEYS.append(f'{bk}{d}_{t}')  # 'Hf_start', 'Hf_mid', etc.




# pylint:disable=missing-function-docstring,unused-argument
class Horovod:
    """Dummy object for Horovod."""
    def __init__(self):
        self._size = 1
        self._num_ranks = 1
        self._local_rank = 0
        self._rank = 0
        self._local_size = 1

    @staticmethod
    def broadcast_variables(*args, **kwargs):
        _, _ = args, kwargs
        pass

    @staticmethod
    def init(*args, **kwargs):
        _, _ = args, kwargs
        pass

    @staticmethod
    def rank(*args, **kwargs):
        _, _ = args, kwargs
        return 0

    @staticmethod
    def local_rank(*args, **kwargs):
        _, _ = args, kwargs
        return 0

    @staticmethod
    def size(*args, **kwargs):
        _, _ = args, kwargs
        return 1

    @staticmethod
    def local_size(*args, **kwargs):
        _, _ = args, kwargs
        return 1
