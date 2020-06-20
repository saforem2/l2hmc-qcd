"""
train.py

Train 2D U(1) model using eager execution in tensorflow.
"""
from __future__ import absolute_import, division, print_function

import os

from utils.attr_dict import AttrDict
from utils.parse_args import parse_args
from utils.training_utils import train
from config import NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC
from utils.inference_utils import run

if __name__ == '__main__':
    FLAGS = parse_args()
    FLAGS = AttrDict(FLAGS.__dict__)
    LOG_FILE = os.path.join(os.getcwd(), 'log_dirs.txt')
    FLAGS.net_weights = NET_WEIGHTS_HMC if FLAGS.hmc else NET_WEIGHTS_L2HMC
    MODEL, OUTPUTS, FLAGS = train(FLAGS, LOG_FILE)
    if FLAGS.inference and FLAGS.run_steps > 0:
        _, _ = run(MODEL, FLAGS, x=OUTPUTS['x'])
        #  _, _ = run(FLAGS, MODEL, FLAGS.beta_final,
        #             FLAGS.run_steps, x=OUTPUTS['x'])
