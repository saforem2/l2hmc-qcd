"""
train.py

Train 2D U(1) model using eager execution in tensorflow.
"""
from __future__ import absolute_import, division, print_function

import os
import json

from config import NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC
from utils.attr_dict import AttrDict
from utils.parse_args import parse_args
from utils.training_utils import train
from utils.inference_utils import run


def main(args, log_file=None):
    """Main method for training."""
    args.net_weights = NET_WEIGHTS_HMC if args.hmc else NET_WEIGHTS_L2HMC
    x, dynamics, train_data, args = train(args, log_file=log_file)

    if args.inference and args.run_steps > 0:
        _, _ = run(dynamics, args, x=x)

    return x, dynamics, train_data, args


if __name__ == '__main__':
    FLAGS = parse_args()
    FLAGS = AttrDict(FLAGS.__dict__)
    LOG_FILE = os.path.join(os.getcwd(), 'log_dirs.txt')
    _ = main(FLAGS, LOG_FILE)

