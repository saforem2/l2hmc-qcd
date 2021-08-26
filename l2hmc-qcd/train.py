"""
train.py

Train 2D U(1) model using eager execution in tensorflow.
"""
# noqa: E402, F401
# pylint:disable=wrong-import-position,invalid-name, unused-import,
# pylint: disable=ungrouped-imports
from __future__ import absolute_import, division, print_function, annotations
import warnings
#  warnings.filterwarnings('once')
import json
from copy import deepcopy

import os
import contextlib
from pathlib import Path
from typing import Any, Union
import tensorflow as tf
from config import PI


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

from utils.training_utils import train
from utils.inference_utils import run


logger = Logger()
#  os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    #  tf.keras.backend.set_floatx('float32')
    import numpy as np
    custom_betas = None
    if configs.get('discrete_beta', False):
        b0 = configs.get('beta_init', None)  # type: float
        b1 = configs.get('beta_final', None)  # type: float
        db = b1 - b0
        #  per_step = (b1 - b0) // configs.get('train_steps', None)
        per_step = int(configs.get('train_steps', None) // (b1 + 1 - b0))
        custom_betas = []
        for b in range(int(b0), int(b1+1)):
            betas_ = b * np.ones(per_step)
            custom_betas.append(betas_)

        custom_betas = np.stack(np.array(custom_betas))
        custom_betas = tf.convert_to_tensor(custom_betas.flatten(),
                                            dtype=tf.keras.backend.floatx())
        logger.info(f'Using discrete betas!!!')
        logger.info(f'custom_betas: {custom_betas}')

    # -- Train model ----------------------------------------------------
    train_out = train(configs=configs, make_plots=True,
                      custom_betas=custom_betas)
    x = train_out.x
    dynamics = train_out.dynamics
    configs = train_out.configs
    # ------------------------------------------------------------------


    # -- Run inference on trained model ---------------------------------
    run_steps = configs.get('run_steps', 20000)
    if run_steps > 0:
        x = tf.random.uniform(x.shape, *(-PI, PI))
        beta = configs.get('beta_final')
        nchains = configs.get('num_chains', configs.get('nchains', None))
        if nchains is not None:
            x = x[:nchains]
        #  if configs.get('small_batch', False):
        #      batch_size = 256
        #      old_shape = configs['dynamics_config']['x_shape']
        #      new_shape = (batch_size, *old_shape[1:])
        #      configs['dynamics_config']['x_shape'] = new_shape
        #      dynamics = build_dynamics(configs)
        #      x = x[:batch_size]

        _ = run(dynamics, configs, x, beta=beta, make_plots=True,
                therm_frac=0.1, num_chains=nchains, save_x=False)


if __name__ == '__main__':
    timestamp = io.get_timestamp('%Y-%m-%d-%H%M')
    #  debug_events_writer = tf.debugging.experimental.enable_dump_debug_info(
    #      debug_dir, circular_buffer_size=-1,
    #      tensor_debug_mode="FULL_HEALTH",
    #  )

    configs = parse_configs()
    cdict = configs.__dict__
    logger.log(f'configs:\n {json.dumps(cdict)}')
    main(cdict)

    #  debug_events_writer.FlushExecutionFiles()
    #  debug_events_writer.FlushNonExecutionFiles()
