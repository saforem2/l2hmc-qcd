"""
__init__.py

Initialization module for running training/inference.

Author: Sam Foreman
Date: 08/26/2020
"""
from tqdm.auto import tqdm
import tensorflow as tf
try:
    import horovod.tensorflow as hvd  # pylint:disable=wrong-import-order
    hvd.init()
    GPUS = tf.config.experimental.list_physical_devices('GPU')
    for gpu in GPUS:
        tf.config.experimental.set_memory_growth(gpu, True)
    if GPUS:
        gpu = GPUS[hvd.local_rank()]  # pylint:disable=invalid-name
        tf.config.experimental.set_visible_devices(gpu, 'GPU')

    HAS_HOROVOD = True
    RANK = hvd.rank()
    LOCAL_RANK = hvd.local_rank()
    NUM_WORKERS = hvd.size()
    IS_CHIEF = (RANK == 0)

except (ImportError, ModuleNotFoundError):
    HAS_HOROVOD = False
    RANK = 0
    LOCAL_RANK = 0
    NUM_WORKERS = 1
    IS_CHIEF = (RANK == 0)


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
