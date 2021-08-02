"""
train.py

Train 2D U(1) model using eager execution in tensorflow.
"""
# noqa: E402, F401
# pylint:disable=wrong-import-position,invalid-name, unused-import,
# pylint: disable=ungrouped-imports
from __future__ import absolute_import, division, print_function, annotations
import json
from copy import deepcopy

import os
import contextlib
from pathlib import Path
from typing import Any, Union
import warnings
import tensorflow as tf
#  from tensorflow.python.ops.gen_math_ops import Any


#  try:
#      tf.config.experimental.enable_mlir_bridge()
#      tf.config.experimental.enable_mlir_graph_optimization()
#  except:  # noqa: E722
#      pass

from utils.hvd_init import HAS_HOROVOD, RANK
from utils.logger import Logger
from utils import file_io as io
from utils import attr_dict as AttrDict

from utils.parse_configs import parse_configs
from dynamics.gauge_dynamics import build_dynamics

from utils.training_utils import train, train_hmc
from utils.inference_utils import run


logger = Logger()
#  os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', 'WARNING:matplotlib')
names = ['month', 'time', 'hour', 'minute', 'second']
formats = [
    '%Y_%m',
    '%Y-%m-%d-%H%M%S',
    '%Y-%m-%d-%H',
    '%Y-%m-%d-%H%M',
    '%Y-%m-%d-%H%M%S'
]
TSTAMPS = {
    k: io.get_timestamp(v) for k, v in dict(zip(names, formats)).items()
}

#  logger = logging.getLogger(__name__)
#  logging_datefmt = '%Y-%m-%d %H:%M:%S'
#  logging_level = logging.WARNING
#  logging_format = (
#      '%(asctime)s %(levelname)s:%(process)s:%(thread)s:%(name)s:%(message)s'
#  )

logger.info(f'using tensorflow version: {tf.__version__}')
logger.info(f'using tensorflow from: {tf.__file__}')


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
            logger.info(f'Restoring FLAGS from: {rf_file}...')
            flags.update(restored)
        except (FileNotFoundError, EOFError):
            pass

    return flags


def dict_to_str(d):
    strs = []
    for key, val in d.items():
        if isinstance(val, dict):
            strs_ = dict_to_str(val)
        else:
            strs_ = f'{key}: {val}'

        strs.append(strs_)

    return '\n'.join(strs)


def main(configs: dict[str, Any]):
    """Main method for training."""
    # TODO: Move setup code to separate function and refactor
    #  configs = setup(configs)
    #  tf.keras.backend.set_floatx('float32')
    train_out = train(configs=configs, make_plots=True) # , num_chains=nchains)
    x = train_out.x
    logdir = train_out.logdir
    #  train_data = train_out.data
    dynamics = train_out.dynamics
    configs = train_out.configs

    run_steps = configs.get('run_steps', 20000)

    # ====
    # Run inference on trained model
    if run_steps > 0:
        beta = configs.get('beta_final')
        nchains = configs.get('num_chains', configs.get('nchains', 16))
        if configs.get('small_batch', False):
            batch_size = 256
            old_shape = configs['dynamics_config']['x_shape']
            new_shape = (batch_size, *old_shape[1:])
            configs['dynamics_config']['x_shape'] = new_shape
            dynamics = build_dynamics(configs, log_dir=logdir)
            x = x[:batch_size]

        _ = run(dynamics, configs, x, beta=beta, make_plots=True,
                therm_frac=0.1, num_chains=nchains, save_x=False)
        #  try:
        #      run_data = results.run_data
        #      #  run_dir = run_data.dirs['run_dir']
        #      #  dataset = run_data.save_dataset(run_dir, therm_frac=0.)
        #  except:
        #      # TODO: Properly catch exception (if thrown)
        #      pass
        #

        #  _ = run_inference_from_log_dir(log_dir=log_dir,
        #                                 run_steps=run_steps,
        #                                 beta=beta,
        #                                 num_chains=num_chains,
        #                                 batch_size=batch_size,
        #                                 therm_frac=0.2,
        #                                 make_plots=True,
        #                                 train_steps=0,
        #                                 x=xbatch)
        # Run with random start
        #  _ = run(dynamics, args)
        #  # Run HMC
        #  args.hmc = True
        #  args.dynamics_config['eps'] = 0.15
        #  hmc_dir = os.path.join(args.log_dir, 'inference_hmc')
        #  _ = run_hmc(args=args, hmc_dir=hmc_dir)


if __name__ == '__main__':
    timestamp = io.get_timestamp('%Y-%m-%d-%H%M')
    #  debug_events_writer = tf.debugging.experimental.enable_dump_debug_info(
    #      debug_dir, circular_buffer_size=-1,
    #      tensor_debug_mode="FULL_HEALTH",
    #  )

    configs = parse_configs()
    #  CONFIGS = AttrDict(CONFIGS.__dict__)
    #  if CONFIGS.get('debug', False):
    #  if configs.debug:
    #      logging_level = logging.DEBUG
    #      os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
    #      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    #  else:
    #      logging_level = logging.WARNING
    #  cfgs_str = '\n'.join([f'{k}: {v}' for k, v in dict(**CONFIGS).items()])
    logger.log(configs.__dict__)
    #  logger.log(dict(**CONFIGS))
    #  cstr = dict_to_str(dict(**CONFIGS))
    #  io.print_dict(CONFIGS)
    main(configs)
    #  if RANK == 0:
    #      console.save_text(os.path.join(os.getcwd(), 'train.log'), styles=False)

    #
    #  debug_events_writer.FlushExecutionFiles()
    #  debug_events_writer.FlushNonExecutionFiles()
