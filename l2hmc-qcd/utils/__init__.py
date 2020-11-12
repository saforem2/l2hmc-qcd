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
    #      HAS_HOROVOD = True
    #      RANK = hvd.rank()
    #      LOCAL_RANK = hvd.local_rank()
    #      NUM_WORKERS = hvd.size()
    #      IS_CHIEF = (RANK == 0)
    #
    #  except (ImportError, ModuleNotFoundError):
    #      HAS_HOROVOD = False
    #      RANK = 0
    #      LOCAL_RANK = 0
    #      NUM_WORKERS = 1
    #      IS_CHIEF = (RANK == 0)
    #

#  RANK = hvd.rank()
#  LOCAL_RANK = hvd.local_rank()
#  NUM_WORKERS = hvd.size()
#  IS_CHIEF = (RANK == 0)



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
