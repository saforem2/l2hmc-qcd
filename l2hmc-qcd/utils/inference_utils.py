"""
inference_utils.py

Collection of helper methods to use for running inference on trained model.
"""
from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np
import tensorflow as tf

import utils.file_io as io

from config import PI, TF_FLOAT, TF_INT, PROJECT_DIR
from utils.attr_dict import AttrDict
from utils.plotting_utils import plot_data
from dynamics.dynamics import Dynamics
from utils.training_utils import build_dynamics
from utils.data_containers import DataContainer

# pylint:disable=no-member
if tf.__version__.startswith('1.'):
    TF_VERSION = '1.x'
elif tf.__version__.startswith('2.'):
    TF_VERSION = '2.x'

try:
    import horovod.tensorflow as hvd

    hvd.init()
    if hvd.rank() == 0:
        print(f'Number of devices: {hvd.size()}')
    if TF_VERSION == '2.x':
        GPUS = tf.config.experimental.list_physical_devices('GPU')
        for gpu in GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)
        if GPUS:
            tf.config.experimental.set_visible_devices(
                GPUS[hvd.local_rank()], 'GPU'
            )
    elif TF_VERSION == '1.x':
        CONFIG = tf.compat.v1.ConfigProto()
        CONFIG.gpu_options.allow_growth = True
        CONFIG.gpu_options.visible_device_list = str(hvd.local_rank())
        tf.compat.v1.enable_eager_execution(config=CONFIG)

except ImportError:
    if TF_VERSION == '1.x':
        tf.compat.v1.enable_eager_execution()

# pylint:disable=too-many-locals,invalid-name

NAMES = [
    'STEP', 'dt', 'LOSS', 'px', 'eps', 'BETA', 'sumlogdet', 'dQ', 'plaq_err',
]
HSTR = ''.join(["{:^12s}".format(name) for name in NAMES])
SEP = '-' * len(HSTR)
HEADER = '\n'.join([SEP, HSTR, SEP])


def check_if_chief(args):
    """Helper function to determine if we're on `rank == 0`."""
    using_hvd = args.get('horovod', False)
    return hvd.rank() == 0 if using_hvd else not using_hvd


def print_args(args):
    """Print out parsed arguments."""
    io.log(80 * '=' + '\n' + 'Parsed args:\n')
    for key, val in args.items():
        io.log(f' {key}: {val}\n')
    io.log(80 * '=')


def run_hmc(
        args: AttrDict,
        hmc_dir: str = None,
        log_file: str = None
) -> (Dynamics, DataContainer):
    """Run HMC using `inference_args` on a model specified by `params`.

    NOTE:
    -----
    args should be a dict with the following keys:
        - 'hmc'
        - 'eps'
        - 'beta'
        - 'num_steps'
        - 'run_steps'
        - 'lattice_shape'
    """
    is_chief = check_if_chief(args)
    if not is_chief:
        return None, None

    print_args(args)
    args.log_dir = io.make_log_dir(args, 'GaugeModel', log_file)

    args.update({
        'hmc': True,
        'units': [],
        'eps_fixed': True,
        'dropout_prob': 0.,
        'horovod': False,
    })

    dynamics, args = build_dynamics(args)
    if hmc_dir is None:
        root_dir = os.path.dirname(PROJECT_DIR)
        hmc_dir = os.path.join(root_dir, 'gauge_logs_eager', 'hmc_runs')
    io.check_else_make_dir(hmc_dir)
    dynamics, run_data = run(dynamics, args, runs_dir=hmc_dir)

    return dynamics, run_data


def load_and_run(args: AttrDict,
                 runs_dir: str = None) -> (Dynamics, AttrDict):
    """Load trained model from checkpoint and run inference."""
    if not check_if_chief(args):
        return None, None

    print_args(args)
    if args.hmc:
        train_dir = os.path.join(args.log_dir, 'training_hmc')
    else:
        train_dir = os.path.join(args.log_dir, 'training')

    ckpt_dir = os.path.join(train_dir, 'checkpoints')
    FLAGS = AttrDict(dict(io.loadz(os.path.join(train_dir, 'FLAGS.z'))))
    FLAGS.horovod = False

    dynamics, FLAGS = build_dynamics(FLAGS)

    #  step_init = tf.Variable(0, dtype=TF_INT)
    ckpt = tf.train.Checkpoint(dynamics=dynamics,
                               optimizer=dynamics.optimizer)
    manager = tf.train.CheckpointManager(ckpt, max_to_keep=5,
                                         directory=ckpt_dir)
    if manager.latest_checkpoint:
        io.log(f'Restored model from: {manager.latest_checkpoint}')
        ckpt.restore(manager.latest_checkpoint)

    if args.eps is None:
        args.eps = dynamics.eps.numpy()
        #  eps_file = os.path.join(args.log_dir, 'eps_final.z')
        #  if os.path.isfile(eps_file):
        #      edict = io.loadz(eps_file)
        #      dynamics.dynamics.eps = edict['eps']
        #      args.eps = edict['eps']
        #  else:
        #      args.eps = ckpt.dynamics.eps

    if args.beta is None:
        train_dir = os.path.join(args.log_dir, 'training')
        l2hmc_flags = AttrDict(
            dict(io.loadz(os.path.join(train_dir, 'FLAGS.z')))
        )
        args.beta = l2hmc_flags.beta_final

    args.update(FLAGS)
    dynamics, run_data = run(dynamics, args, runs_dir=runs_dir)

    return dynamics, run_data


def run(dynamics, args, x=None, runs_dir=None):
    """Run inference.

    Returns:
        model(GaugeModel): Trained model
        ouptuts(dict): Dictionary of outputs from inference run.
    """
    is_chief = check_if_chief(args)
    if not is_chief:
        return None, None

    if runs_dir is None:
        if args.hmc:
            runs_dir = os.path.join(args.log_dir, 'inference_hmc')
        else:
            runs_dir = os.path.join(args.log_dir, 'inference')

    io.check_else_make_dir(runs_dir)

    run_steps = args.get('run_steps', None)
    beta = args.get('beta', None)
    if beta is None:
        beta = args.get('beta_final', None)

    if x is None:
        x = tf.random.uniform(shape=dynamics.config.input_shape,
                              minval=-PI, maxval=PI, dtype=TF_FLOAT)

    run_data = test_dynamics(dynamics, args, x)

    run_dir = io.make_run_dir(args, runs_dir)
    data_dir = os.path.join(run_dir, 'run_data')
    log_file = os.path.join(run_dir, 'run_log.txt')
    io.check_else_make_dir(run_dir, data_dir)

    run_data.flush_data_strs(log_file, mode='a')
    if dynamics.save_run_data:
        run_data.save_data(data_dir)
    io.save_inference(run_dir, run_data)

    eps = dynamics.eps
    if hasattr(eps, 'numpy'):
        eps = eps.numpy()

    run_params = {
        'eps': eps,
        'beta': beta,
        'run_steps': run_steps,
        'lattice_shape': dynamics.lattice_shape,
        'num_steps': dynamics.config.num_steps,
        'net_weights': dynamics.config.net_weights,
        'input_shape': dynamics.config.input_shape,
    }
    io.save_params(run_params, run_dir, name='run_params')

    plot_data(run_data, run_dir, args, thermalize=True, params=run_params)

    return dynamics, run_data


def test_dynamics(dynamics, args, x=None):
    """Run inference on trained dynamics."""
    is_chief = check_if_chief(args)
    if not is_chief:
        return None

    beta = args.get('beta', None)
    run_steps = args.get('run_steps', None)
    if beta is None:
        beta = args.get('beta_final', None)

    if x is None:
        x = tf.random.uniform(shape=dynamics.config.input_shape,
                              minval=-PI, maxval=PI)
        x = tf.cast(x, dtype=TF_FLOAT)

    run_data = DataContainer(run_steps, header=HEADER)

    #  test_step = (
    #      dynamics.test_step if not dynamics._should_compile
    #      else tf.function(dynamics.test_step, experimental_compile=True)
    #  )

    #  eps = model.dynamics.eps
    eps = dynamics.eps
    if hasattr(eps, 'numpy'):
        eps = eps.numpy()

    io.log(SEP)
    io.log(f'Running inference with:')
    io.log(f'  beta: {beta}')
    io.log(f'  dynamics.eps: {eps:.4g}')
    io.log(f'  net_weights: {dynamics.config.net_weights}')
    io.log(SEP)
    io.log(HEADER)
    for step in tf.range(run_steps):
        x, metrics = dynamics.test_step(x, beta)
        run_data.update(step, metrics)

        if step % dynamics.print_steps == 0:
            data_str = run_data.get_fstr(step, metrics)
            io.log(data_str)

        if step % 100 == 0:
            io.log(HEADER)

    return run_data


# pylint:disable=too-many-statements
#  def test_dynamics_old(model, args, x=None):
#      """Run inference on trained `model`.
#
#      Returns:
#          outputs(dict): Dictionary of outputs.
#          data_strs(lsit): List of strings containing inference log.
#      """
#      is_chief = check_if_chief(args)
#      if not is_chief:
#          return None
#
#      #  using_hvd = getattr(model, 'using_hvd', False)
#      #  is_chief = hvd.rank() == 0 if using_hvd else not using_hvd
#      #  if not is_chief:
#      #      return None, None
#
#      run_steps = args.get('run_steps', None)
#      beta = args.get('beta', None)
#      if beta is None:
#          beta = args.get('beta_final', None)
#
#      if x is None:
#          x = tf.random.uniform(shape=model.input_shape,
#                                minval=-PI, maxval=PI)
#          x = tf.cast(x, dtype=TF_FLOAT)
#
#      #  plaqs, q_new = model.calc_observables(x, beta, use_sin=False)
#
#      run_data = RunData(run_steps, header=RUN_HEADER)
#
#      test_step = (
#          model.test_step if not model.compile
#          else tf.function(model.run_step, experimental_compile=True)
#      )
#
#      eps = model.dynamics.eps
#      if hasattr(eps, 'numpy'):
#          eps = eps.numpy()
#
#      io.log(RUN_SEP)
#      io.log(f'Running inference with:')
#      io.log(f'  beta: {beta}')
#      io.log(f'  dynamics.eps: {eps:.4g}')
#      io.log(f'  net_weights: {model.dynamics.config.net_weights}')
#      io.log(RUN_SEP)
#      io.log(RUN_HEADER)
#      for step in np.arange(run_steps):
#          t0 = time.time()
#          x = tf.reshape(x, model.input_shape)
#          #  states, px, sld_states = model.run_step(x, beta)
#          states, px, sld_states = run_step(x, beta)
#          x = states.out.x
#          sld = sld_states.out
#          x = tf.reshape(x, model.lattice_shape)
#          dt = time.time() - t0
#
#          q_old = q_new
#          plaqs_err, q_new = model.calc_observables(x, beta, use_sin=False)
#          dq = tf.math.abs(q_new - q_old)
#
#          outputs = AttrDict({
#              'steps': step,
#              'dt': dt,
#              'px': px.numpy(),
#              'charges': q_new.numpy(),
#              'dq': dq.numpy(),
#              'sumlogdet': sld.numpy(),
#              'plaqs': plaqs_err.numpy(),
#          })
#
#          if step % model.print_steps == 0:
#              data_str = run_data.get_fstr(step, outputs)
#              io.log(data_str)
#
#          if model.save_run_data:
#              run_data.update(outputs)
#
#          if step % 100 == 0:
#              io.log(RUN_HEADER)
#
#      #  outputs = {
#      #      'px': np.array(px_arr),
#      #      'dq': np.array(dq_arr),
#      #      'charges_arr': np.array(charges_arr),
#      #      'x': tf.reshape(x, model.input_shape),
#      #      'plaqs_err': np.array(plaqs_arr),
#      #  }
#      #
#      #  return outputs, data_strs
#
#      return run_data
