"""
train.py

Train 2D U(1) model using eager execution in tensorflow.
"""
from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
import traceback
import contextlib
import utils.file_io as io

from utils.attr_dict import AttrDict

#  from utils.parse_args import parse_args
from utils.parse_configs import parse_configs
from utils.training_utils import train, train_hmc
from utils.inference_utils import run, run_hmc


@contextlib.contextmanager
def options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


def restore_flags(flags, train_dir):
    """Update `FLAGS` using restored flags from `log_dir`."""
    rf_file = os.path.join(train_dir, 'FLAGS.z')
    restored = AttrDict(dict(io.loadz(rf_file)))
    io.log(f'Restoring FLAGS from: {rf_file}...')
    flags.update(restored)

    return flags


def main(args):
    """Main method for training."""
    hmc_steps = args.get('hmc_steps', 0)
    tf.keras.backend.set_floatx('float32')
    log_file = os.path.join(os.getcwd(), 'log_dirs.txt')

    x = None
    log_dir = args.get('log_dir', None)
    if log_dir is not None:  # we want to restore from latest checkpoint
        train_steps = args.get('train_steps', None)
        args = restore_flags(args, os.path.join(args.log_dir, 'training'))
        args.restore = True
        if train_steps > args.train_steps:
            args.train_steps = train_steps

    else:  # New training session
        args.log_dir = io.make_log_dir(args, 'GaugeModel', log_file)
        args.restore = False
        if hmc_steps > 0:
            x, _, eps = train_hmc(args)
            args.dynamics_config['eps'] = eps

    _, dynamics, _, args = train(args, x=x)

    # ====
    # Run inference on trained model
    if args.get('run_steps', 5000) > 0:
        # ====
        # Run with random start
        dynamics, _, _ = run(dynamics, args)

        # ====
        # Run HMC
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
