"""
train.py

Train 2D U(1) model using eager execution in tensorflow.
"""
# noqa: E402, F401
# pylint:disable=wrong-import-position,invalid-name, unused-import,
# pylint: disable=ungrouped-imports
from __future__ import absolute_import, division, print_function, annotations

import os
import contextlib
import logging
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import Any


#  try:
#      tf.config.experimental.enable_mlir_bridge()
#      tf.config.experimental.enable_mlir_graph_optimization()
#  except:  # noqa: E722
#      pass


try:
    import horovod
    import horovod.tensorflow as hvd
    try:
        RANK = hvd.rank()
    except ValueError:
        hvd.init()

    RANK = hvd.rank()
    HAS_HOROVOD = True
    logging.info(f'using horovod version: {horovod.__version__}')
    logging.info(f'using horovod from: {horovod.__file__}')
    GPUS = tf.config.experimental.list_physical_devices('GPU')
    for gpu in GPUS:
        tf.config.experimental.set_memory_growth(gpu, True)
    if GPUS:
        gpu = GPUS[hvd.local_rank()]
        tf.config.experimental.set_visible_devices(gpu, 'GPU')

except (ImportError, ModuleNotFoundError):
    HAS_HOROVOD = False


from utils.file_io import console
import utils.file_io as io

from utils.attr_dict import AttrDict

from utils.parse_configs import parse_configs
from dynamics.gauge_dynamics import build_dynamics
from utils.training_utils import train, train_hmc
from utils.inference_utils import run, run_hmc, run_inference_from_log_dir


#  os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = logging.getLogger(__name__)
logging_datefmt = '%Y-%m-%d %H:%M:%S'
logging_level = logging.WARNING
logging_format = (
    '%(asctime)s %(levelname)s:%(process)s:%(thread)s:%(name)s:%(message)s'
)

logging.info(f'using tensorflow version: {tf.__version__}')
logging.info(f'using tensorflow from: {tf.__file__}')

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


def main(
        configs: AttrDict,
        num_chains: int = None,
        run_steps: int = None
):
    """Main method for training."""
    hmc_steps = configs.get('hmc_steps', 0)
    #  tf.keras.backend.set_floatx('float32')
    log_file = os.path.join(os.getcwd(), 'log_dirs.txt')

    x = None
    logdir = configs.get('log_dir', None)
    beta_init = configs.get('beta_init', None)
    beta_final = configs.get('beta_final', None)

    if logdir is not None:  # we want to restore from latest checkpoint
        configs['restore'] = True
        #  logdir = configs.get('log_dir', None)
        run_steps = configs.get('run_steps', None)
        train_steps = configs.get('train_steps', None)
        restored = restore_flags(configs, os.path.join(logdir, 'training'))
        for key, val in configs.items():
            if key in restored:
                if val != restored[key]:
                    io.log(f'Restored {key}: {restored[key]}')
                    io.log(f'Using {key}: {val}')

        configs['run_steps'] = run_steps
        configs['train_steps'] = train_steps
        if beta_init != configs.get('beta_init', None):
            configs['beta_init'] = beta_init
        if beta_final != configs.get('beta_final', None):
            configs['beta_final'] = beta_final

    else:  # New training session
        train_steps = configs.get('train_steps', None)
        run_steps = configs.get('run_steps', None)

        timestamps = AttrDict({
            'month': io.get_timestamp('%Y_%m'),
            'time': io.get_timestamp('%Y-%M-%d-%H%M%S'),
            'hour': io.get_timestamp('%Y-%m-%d-%H'),
            'minute': io.get_timestamp('%Y-%m-%d-%H%M'),
            'second': io.get_timestamp('%Y-%m-%d-%H%M%S'),
        })
        logdir = io.make_log_dir(configs, 'GaugeModel', log_file,
                                 timestamps=timestamps)
        configs['log_dir'] = logdir
        io.write(f'{logdir}', log_file, 'a')
        configs['restore'] = False
        if hmc_steps > 0:
            x, hdynamics, _, hflags = train_hmc(configs, num_chains=num_chains)
            _ = run(hdynamics, hflags, save_x=False)

    if num_chains is None:
        num_chains = configs.get('num_chains', 15)

    #  x, dynamics, train_data, configs = train(configs=configs,
    nchains = int(num_chains)
    out = train(configs=configs, x=x, make_plots=True, num_chains=nchains)
    x = out.x
    train_data = out.data
    dynamics = out.dynamics
    configs = out.configs

    rs = configs.get('run_steps', 50000)
    run_steps = rs if run_steps is None else run_steps

    # ====
    # Run inference on trained model
    if run_steps > 0:
        #  run_steps = args.get('run_steps', 125000)
        #  logdir = configs.log_dir
        beta = configs.get('beta_final')
        if configs.get('small_batch', False):
            batch_size = 256
            old_shape = configs['dynamics_config']['x_shape']
            new_shape = (batch_size, *old_shape[1:])
            configs['dynamics_config']['x_shape'] = new_shape
            dynamics = build_dynamics(configs, log_dir=logdir)
            x = x[:batch_size]

        results = run(dynamics, configs, x, beta=beta, make_plots=True,
                      therm_frac=0.1, num_chains=num_chains, save_x=False)
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

    CONFIGS = parse_configs()
    CONFIGS = AttrDict(CONFIGS.__dict__)
    if CONFIGS.get('debug', False):
        logging_level = logging.DEBUG
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        logging_level = logging.WARNING
    #  cfgs_str = '\n'.join([f'{k}: {v}' for k, v in dict(**CONFIGS).items()])
    cstr = dict_to_str(dict(**CONFIGS))
    console.log(cstr)
    #  console.log(f'CONFIGS: {dict(**CONFIGS)}')
    #  io.print_dict(CONFIGS)
    main(CONFIGS)
    #  if RANK == 0:
    #      console.save_text(os.path.join(os.getcwd(), 'train.log'), styles=False)

    #
    #  debug_events_writer.FlushExecutionFiles()
    #  debug_events_writer.FlushNonExecutionFiles()
