"""
train_eager.py

Train 2D U(1) model using eager execution in tensorflow.
"""
from __future__ import absolute_import, division, print_function

import os

from eager.training_utils import train, run
from utils.attr_dict import AttrDict
from utils.parse_args import parse_args

if __name__ == '__main__':
    FLAGS = parse_args()
    FLAGS = AttrDict(FLAGS.__dict__)
    LOG_FILE = os.path.join(os.getcwd(), 'output_dirs.txt')
    MODEL, OUTPUTS, FLAGS = train(FLAGS, LOG_FILE)
    if FLAGS.inference and FLAGS.run_steps > 0:
        _, _ = run(FLAGS, MODEL, FLAGS.beta_final,
                   FLAGS.run_steps, x=OUTPUTS['x'])
