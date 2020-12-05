"""
__init__.py

Initialization module for running training/inference.

Author: Sam Foreman
Date: 08/26/2020
"""
import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tqdm.auto import tqdm
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

_d = ('f', 'b')
_t = ('start', 'mid', 'end')


base_keys = [
    'H', 'ld', 'sld',
    'sinQ', 'intQ', 'accept_prob',
    'sumlogdet', 'logdets', 'logdet',
]

SKEYS = []
for bk in base_keys:
    for d in _d:                          # =====
        SKEYS.append(f'{bk}{d}')          # 'Hf', 'Hb', 'Hwf', 'Hwb', etc.
        for t in _t:                      # ====
            SKEYS.append(f'{bk}{d}_{t}')  # 'Hf_start', 'Hf_mid', etc.


SKIP_KEYS = ['charges',
             'Hf', 'Hb',
             'Hwf', 'Hwb',
             'sldf', 'sldb',
             'forward', 'backward',
             'sinQf', 'sinQb', 'sumlogdet_prop',
             'intQf', 'intQf_start', 'intQf_mid', 'intQf_end',
             'intQb', 'intQb_start', 'intQb_mid', 'intQb_end',
             'sumlogdet_start', 'sumlogdet_mid', 'sumlogdet_end',
             'sumlogdetf_start', 'sumlogdetf_mid', 'sumlogdetf_end',
             'sumlogdetb_start', 'sumlogdetb_mid', 'sumlogdetb_end',
             'logdetf_start', 'logdetf_mid', 'logdetf_end',
             'logdetb_start', 'logdetb_mid', 'logdetb_end',
             'logdetsf_end', 'logdetsb_end',
             'accept_probf', 'accept_probb',
             'accept_probf_start', 'accept_probf_mid', 'accept_probf_end',
             'accept_probb_start', 'accept_probb_mid', 'accept_probb_end',
             'logdetsf', 'logdetsb',
             'logdetsf_start', 'logdetsf_mid', 'logdtsf_end',
             'logdetsb_start', 'logdetsb_mid', 'logdtsb_end',
             'sinQf_start', 'sinQf_mid', 'sinQf_end',
             'sinQb_start', 'sinQb_mid', 'sinQb_end',
             'Hf_start', 'Hf_mid', 'Hf_end',
             'Hb_start', 'Hb_mid', 'Hb_end',
             'ldf_start', 'ldf_mid', 'ldf_end',
             'ldb_start', 'ldb_mid', 'ldb_end']


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
        pass

    @staticmethod
    def init(*args, **kwargs):
        return 0

    @staticmethod
    def rank(*args, **kwargs):
        return 0

    @staticmethod
    def local_rank(*args, **kwargs):
        return 0

    @staticmethod
    def size(*args, **kwargs):
        return 1

    @staticmethod
    def local_size(*args, **kwargs):
        return 1


class DummyTqdmFile:
    """Dummy file-like that will write to tqdm.

    NOTE: This approach is taken from
        https://github.com/tqdm/tqdm/issues/313
    """
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        """Write out using `tqdm.write` wrapper."""
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)  # , end='')

    def flush(self):
        """Flush to file."""
        return getattr(self.file, "flush", lambda: None)()
