"""
train.py

Train 2D U(1) model using eager execution in tensorflow.
"""
# noqa: E402
# pylint:disable=wrong-import-position,invalid-name
from __future__ import absolute_import, division, print_function

import os
import sys
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import contextlib

import tensorflow as tf
from utils import run_tf_check, RANK, LOCAL_RANK, NUM_WORKERS, IS_CHIEF
run_tf_check()

logger = logging.getLogger(__name__)
logging_datefmt = '%Y-%m-%d %H:%M:%S'
logging_level = logging.INFO
logging_format = (
    '%(asctime)s %(levelname)s:%(process)s:%(thread)s:%(name)s:%(message)s'
)
#  stream = sys.stdout if RANK == 0 else sys.stderr
#
#  logging.basicConfig(level=logging_level,
#                      format=logging_format,
#                      datefmt=logging_datefmt,
#                      stream=sys.stdout if RANK == 0 else sys.stderr)


try:
    import horovod
    import horovod.tensorflow as hvd
    #  hvd.init()
    HAS_HOROVOD = True
    #  gpus = tf.config.experimental.list_physical_devices('GPU')
    #  for gpu in gpus:
    #      tf.config.experimental.set_memory_growth(gpu, True)
    #  if gpus:
    #      tf.config.experimental.set_visible_devices(
    #          gpus[hvd.local_rank()], 'GPU'
    #      )

    logging_format = (
        '%(asctime)s %(levelname)s:%(process)s:%(thread)s:'
        + ('%05d' % hvd.rank()) + ':%(name)s:%(message)s'
    )
    logging.basicConfig(level=logging_level,
                        format=logging_format,
                        datefmt=logging_datefmt,
                        stream=sys.stdout if hvd.rank() == 0 else None)
    logging.warning(' '.join([f'rank: {hvd.rank()}',
                              f'local_rank: {hvd.local_rank()}',
                              f'size: {hvd.size()}',
                              f'local_size: {hvd.local_size()}']))
    logging.info(f'using tensorflow version: {tf.__version__}')
    logging.info(f'using tensorflow from: {tf.__file__}')
    logging.info(f'using horovod version: {horovod.__version__}')
    logging.info(f'using horovod from: {horovod.__file__}')


except ImportError:
    HAS_HOROVOD = False


if RANK > 0:
    logging_level = logging.WARNING

import utils.file_io as io

from utils.attr_dict import AttrDict

from utils.parse_configs import parse_configs
from utils.training_utils import train, train_hmc
from utils.inference_utils import run, run_hmc


@contextlib.contextmanager
def experimental_options(options):
    """Run inside contextmanager with special options."""
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


def restore_flags(flags, train_dir):
    """Update `FLAGS` using restored flags from `log_dir`."""
    rf_file = os.path.join(train_dir, 'FLAGS.z')
    if os.path.isfile(rf_file):
        try:
            restored = io.loadz(rf_file)
            restored = AttrDict(restored)
            io.log(f'Restoring FLAGS from: {rf_file}...')
            flags.update(restored)
        except (FileNotFoundError, EOFError):
            pass
        #  restored = AttrDict(dict(io.loadz(rf_file)))

    return flags


def main(args):
    """Main method for training."""
    hmc_steps = args.get('hmc_steps', 0)
    tf.keras.backend.set_floatx('float32')
    log_file = os.path.join(os.getcwd(), 'log_dirs.txt')

    x = None
    log_dir = args.get('log_dir', None)
    beta_init = args.get('beta_init', None)
    beta_final = args.get('beta_final', None)
    if log_dir is not None:  # we want to restore from latest checkpoint
        args.restore = True
        train_steps = args.get('train_steps', None)
        restored = restore_flags(args, os.path.join(args.log_dir, 'training'))
        for key, val in args.items():
            if key in restored:
                if val != restored[key]:
                    print(f'Restored {key}: {restored[key]}')
                    print(f'Using {key}: {val}')

        args.update({
            'train_steps': train_steps,
        })
        if beta_init != args.get('beta_init', None):
            args.beta_init = beta_init
        if beta_final != args.get('beta_final', None):
            args.beta_final = beta_final

    else:  # New training session
        timestamps = AttrDict({
            'month': io.get_timestamp('%Y_%m'),
            'time': io.get_timestamp('%Y-%M-%d-%H%M%S'),
            'hour': io.get_timestamp('%Y-%m-%d-%H'),
            'minute': io.get_timestamp('%Y-%m-%d-%H%M'),
            'second': io.get_timestamp('%Y-%m-%d-%H%M%S'),
        })
        args.log_dir = io.make_log_dir(args, 'GaugeModel', log_file,
                                       timestamps=timestamps)
        io.write(f'{args.log_dir}', log_file, 'a')
        args.restore = False
        if hmc_steps > 0:
            #  x, _, eps = train_hmc(args)
            x, dynamics_hmc, _, hflags = train_hmc(args)
            dirs_hmc = hflags.get('dirs', None)
            args.dynamics_config['eps'] = dynamics_hmc.eps.numpy()
            _ = run(dynamics_hmc, hflags, dirs=dirs_hmc, save_x=False)

    dynamics_config = args.get('dynamics_config', None)
    if dynamics_config is not None:
        log_dir = dynamics_config.get('log_dir', None)
        if log_dir is not None:
            eps_file = os.path.join(log_dir, 'training', 'models', 'eps.z')
            if os.path.isfile(eps_file):
                io.log(f'Loading eps from: {eps_file}')
                eps = io.loadz(eps_file)
                args.dynamics_config['eps'] = eps

    x, dynamics, train_data, args = train(args, x=x)

    # ====
    # Run inference on trained model
    if args.get('run_steps', 5000) > 0:
        # ====
        # Run with random start
        _ = run(dynamics, args, dirs=args.dirs, save_x=False)
        # ====
        # Run HMC
        args.hmc = True
        args.dynamics_config['eps'] = 0.15
        hmc_dir = os.path.join(args.log_dir, 'inference_hmc')
        _ = run_hmc(args=args, hmc_dir=hmc_dir)


if __name__ == '__main__':
    #  debug_events_writer = tf.debugging.experimental.enable_dump_debug_info(
    #      debug_dir, circular_buffer_size=-1,
    #      tensor_debug_mode="FULL_HEALTH",
    #  )

    CONFIGS = parse_configs()
    CONFIGS = AttrDict(CONFIGS.__dict__)
    if CONFIGS.get('debug', False):
        logging_level = logging.DEBUG
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    io.print_dict(CONFIGS)
    main(CONFIGS)
    #
    #  debug_events_writer.FlushExecutionFiles()
    #  debug_events_writer.FlushNonExecutionFiles()
