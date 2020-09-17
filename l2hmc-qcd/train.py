"""
train.py

Train 2D U(1) model using eager execution in tensorflow.
"""
from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
import traceback
import contextlib

from utils.attr_dict import AttrDict

#  from utils.parse_args import parse_args
from utils.parse_configs import parse_configs
from utils.training_utils import train
from utils.inference_utils import run, run_hmc


@contextlib.contextmanager
def options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


def main(args):
    """Main method for training."""
    md_steps = args.get('md_steps', 10)
    tf.keras.backend.set_floatx('float32')
    log_file = os.path.join(os.getcwd(), 'log_dirs.txt')

    _, dynamics, _, args = train(args, md_steps=md_steps, log_file=log_file)
    if args.run_steps > 0:
        # run with random start
        dynamics, _, _ = run(dynamics, args)

        # run using chains from training?
        #  dynamics, run_data, x = run(dynamics, args, x=x)

        # run hmc
        args.hmc = True
        hmc_dir = os.path.join(args.log_dir, 'inference_hmc')
        _ = run_hmc(args=args, hmc_dir=hmc_dir)


if __name__ == '__main__':
    #  debug_events_writer = tf.debugging.experimental.enable_dump_debug_info(
    #      debug_dir, circular_buffer_size=-1,
    #      tensor_debug_mode="FULL_HEALTH",
    #  )

    FLAGS = parse_configs()
    FLAGS = AttrDict(FLAGS.__dict__)
    main(FLAGS)
    #
    #  debug_events_writer.FlushExecutionFiles()
    #  debug_events_writer.FlushNonExecutionFiles()
