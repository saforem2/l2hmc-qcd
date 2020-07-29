"""
train.py

Train 2D U(1) model using eager execution in tensorflow.
"""
from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf

import utils.file_io as io

from config import TRAIN_STR
from utils.attr_dict import AttrDict
from utils.parse_args import parse_args
from utils.training_utils import train
from utils.inference_utils import run


def main(args, log_file=None):
    """Main method for training."""
    io.log(TRAIN_STR)
    tf.keras.backend.set_floatx('float32')
    x, dynamics, train_data, args = train(args, log_file=log_file, md_steps=10)
    if args.run_steps > 0:
        dynamics, run_data, x = run(dynamics, args, x=x)
        # run again with random start
        dynamics, run_data, x = run(dynamics, args)

    return x, dynamics, train_data, run_data, args


if __name__ == '__main__':
    FLAGS = parse_args()
    FLAGS = AttrDict(FLAGS.__dict__)
    LOG_FILE = os.path.join(os.getcwd(), 'log_dirs.txt')
    _ = main(FLAGS, LOG_FILE)
