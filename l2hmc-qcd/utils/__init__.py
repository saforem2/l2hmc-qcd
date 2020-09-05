"""
__init__.py

Initialization module for running training/inference.

Author: Sam Foreman
Date: 08/26/2020
"""
from tqdm.auto import tqdm
import tensorflow as tf
import horovod.tensorflow as hvd  # pylint:disable=wrong-import-order
from tensorflow.keras.mixed_precision import experimental as mixed_precision
hvd.init()
GPUS = tf.config.experimental.list_physical_devices('GPU')
for gpu in GPUS:
    tf.config.experimental.set_memory_growth(gpu, True)
if GPUS:
    tf.config.experimental.set_visible_devices(GPUS[hvd.local_rank()], 'GPU')
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)


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
