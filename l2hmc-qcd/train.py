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


def load_configs_from_logdir(logdir: Union[str, Path]):
    fpath = os.path.join(logdir, 'train_configs.json')
    with open(fpath, 'r') as f:
        configs = json.load(f)

    return configs


def setup(configs: dict[str, Any]):
    """Setup for training."""
    # Create copy of configs.
    # Try loading configs from `configs.logdir/train_configs.json` and restore
    if hasattr(configs, '__dict__'):
        configs = getattr(configs, '__dict__')

    output = deepcopy(configs)
    logfile = os.path.join(os.getcwd(), 'log_dirs.txt')

    ensure_new = configs.get('ensure_new', False)
    logdir = configs.get('logdir', configs.get('log_dir', None))
    if logdir is not None:
        logdir_exists = os.path.isdir(logdir)
        logdir_nonempty = len(os.listdir(logdir)) > 0
        if logdir_exists and logdir_nonempty and ensure_new:
            raise ValueError(
                f'Nonempty `logdir`, but `ensure_new={ensure_new}'
            )

    dirs = io.setup_directories(configs, timestamps = TSTAMPS)
    if ensure_new:
        dirs = io.setup_directories(configs, timestamps=TSTAMPS)
        output['dirs'] = dirs
        #  logdir = io.make_log_dir(configs, 'GaugeModel', logfile)
        output['log_dir'] = logdir
        output['logdir'] = logdir
        output['restore'] = False
        #  io.write(f'{logdir}', logfile, 'a')

    else:
        dirs = io.setup_directories(configs, timestamps=TSTAMPS)
        logdir = dirs['logdir']
        output['restore'] = True
        output['logdir'] = logdir
        output['log_dir'] = logdir
        output['restored_from'] = logdir
        if logdir is not None:
            output = load_configs_from_logdir(logdir)
            output['restored'] = True
            output['log_dir'] = logdir
            output['logdir'] = logdir
            output['restored_from'] = logdir
            to_overwrite = ['train_steps', 'run_steps', 'beta_final']
            changed = {k: configs.get(k) for k in to_overwrite}
            for key, new in changed.items():
                old = output.get(key)
                if new != old:
                    logger.warning(
                        f'Overwriting {key} from {old} to {new} in configs'
                    )
                    output[key] = new

    num_chains = output.get('num_chains', 16)
    if output.get('hmc_steps', 0) > 0:
        x, hdynamics, _, hflags = train_hmc(output, num_chains=num_chains)
        _ = run(hdynamics, hflags, save_x=False)

    if RANK == 0:
        io.write(f'{logdir}', logfile, 'a')

    return output


def main(configs: dict[str, Any]):
    """Main method for training."""
    # TODO: Move setup code to separate function and refactor
    configs = setup(configs)
    #  tf.keras.backend.set_floatx('float32')

    x = None
    logdir = configs.get('log_dir', None)
    #  beta_init = configs.get('beta_init', None)
    #  beta_final = configs.get('beta_final', None)
    #  ensure_new = configs.get('ensure_new', False)
    #  logfile = os.path.join(os.getcwd(), 'log_dirs.txt')

    #  if logdir is None:
    #      # Check and see if `./log_dirs.txt` exists and if so, try loading from
    #      # there if configs are compatible
    #      # TODO: Check if configs are explicitly compatible
    #      if os.path.isfile(logfile):
    #          logger.rule('Found `log_dirs.txt`!')
    #          logger.log(f'logfile: {logfile}')
    #          logdirs = []
    #          with open(logfile, 'r') as f:
    #              for line in f.readlines():
    #                  logdirs.append(line.rstrip('\n'))
    #
    #          logdir = logdirs[-1]
    #          if os.path.isdir(logdir):
    #              if ensure_new:
    #                  err = (f'configs["ensure_new"] = {ensure_new}, but '
    #                         f'found existing `log_dir` at:\n'
    #                         f'  log_dir: {logdir}. Exiting!!')
    #                  raise ValueError(err)
    #
    #              else:
    #                  logdir = None
    #
    #          # If `./log_dirs.txt` exists and points to a directory containing
    #          # training checkpoints
    #          #  if ensure_new and os.path.isdir(logdir):
    #          #      ckptdir = os.path.join(logdir, 'training', 'checkpoints')
    #          #      exists = (len(os.listdir(ckptdir)) > 0)
    #          #      if exists:
    #          #          err = (f'configs["ensure_new"] = {ensure_new}, but '
    #          #                 f'found checkpoints in `log_dir` from '
    #          #                 f'./`log_dirs.txt. Exiting!\n'
    #          #                 f'  log_dir: {logdir}')
    #          #          raise ValueError(err)
    #          #
    #          #      #  exist = len(os.listdir(ckptdir)) > 0 if os.path.isdir(ckptdir)
    #          #      #  if len(ckpts) > 0:
    #          #      #      raise
    #          #      #  if os.path.isdir(ckptdir):
    #          #      #      ckpts = os.listdir(ckptdir)
    #          #      #      if len(ckpts) > 0:
    #          #      #          err = (f'configs["ensure_new"] = {ensure_new}, but '
    #          #      #                 f'found checkpoints in `log_dir` from '
    #          #      #                 f'./`log_dirs.txt. Exiting!\n'
    #          #      #                 f'  log_dir: {logdir}')
    #          #      #          raise ValueError(err)

    #  if logdir is not None:  # we want to restore from latest checkpoint
    #      #  logdir = configs.get('log_dir', None)
    #      run_steps = configs.get('run_steps', None)
    #      train_steps = configs.get('train_steps', None)
    #      restored = restore_flags(configs, os.path.join(logdir, 'training'))
    #      for key, val in configs.items():
    #          if key in restored:
    #              if val != restored[key]:
    #                  io.log(f'Restored {key}: {restored[key]}')
    #                  io.log(f'Using {key}: {val}')
    #
    #      configs['run_steps'] = run_steps
    #      configs['train_steps'] = train_steps
    #      if beta_init != configs.get('beta_init', None):
    #          configs['beta_init'] = beta_init
    #      if beta_final != configs.get('beta_final', None):
    #          configs['beta_final'] = beta_final
    #
    #      configs['restore'] = True
    #      configs['restored_from'] = logdir

    #  else:  # New training session
    #  x, dynamics, train_data, configs = train(configs=configs,
    train_out = train(configs=configs, x=x, make_plots=True) # , num_chains=nchains)
    x = train_out.x
    logdir = train_out.logdir
    #  train_data = train_out.data
    dynamics = train_out.dynamics
    configs = train_out.configs

    run_steps = configs.get('run_steps', 20000)

    # ====
    # Run inference on trained model
    if run_steps > 0:
        #  run_steps = args.get('run_steps', 125000)
        #  logdir = configs.log_dir
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
