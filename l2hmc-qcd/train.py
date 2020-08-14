"""
train.py

Train 2D U(1) model using eager execution in tensorflow.
"""
# pylint:disable=invalid-name
from __future__ import absolute_import, division, print_function

import os
import sys
import logging


from utils import DummyTqdmFile
from utils.attr_dict import AttrDict
from utils.parse_args import parse_args
from utils.training_utils import train
from utils.inference_utils import run

import tensorflow as tf
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()
RANK = hvd.rank()
IS_CHIEF = (RANK == 0)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:  # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        tf.config.experimental.set_visible_devices(
            gpus[hvd.local_rank()], 'GPU'
        )
    # Memory growth must be set before GPUs have been initialized
    except RuntimeError as e:
        print(e)

# Set logging configuration
if IS_CHIEF:
    logging.basicConfig(level=logging.INFO, stream=DummyTqdmFile(sys.stdout),
                        format="%(asctime)s:%(levelname)s:%(message)s")
else:
    logging.basicConfig(level=logging.CRITICAL, stream=None,
                        format="%(asctime)s:%(levelname)s:%(message)s")

logging.getLogger('tensorflow').setLevel(logging.ERROR)


def main(args, log_file=None):
    """Main method for training."""
    #  io.log(TRAIN_STR)
    tf.keras.backend.set_floatx('float32')
    x, dynamics, train_data, args = train(args, md_steps=100,
                                          log_file=log_file, rank=RANK)
    if IS_CHIEF and args.run_steps > 0:
        dynamics, run_data, x = run(dynamics, args, x=x)
        # run again with random start
        dynamics, run_data, x = run(dynamics, args)

    return x, dynamics, train_data, run_data, args


if __name__ == '__main__':
    #  io.log(f'Number of devices: {hvd.size()}')
    #  GPUS = tf.config.experimental.list_physical_devices('GPU')
    #  #  for gpu in GPUS:
    #  #      tf.config.experimental.set_memory_growth(gpu, True)
    #  if GPUS:
    #      tf.config.experimental.set_visible_devices(
    #          GPUS[hvd.local_rank()], 'GPU'
    #      )

    FLAGS = parse_args()
    FLAGS = AttrDict(FLAGS.__dict__)
    LOG_FILE = os.path.join(os.getcwd(), 'log_dirs.txt')
    _ = main(FLAGS, LOG_FILE)
