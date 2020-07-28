"""
inference_utils.py

Collection of helper methods to use for running inference on trained model.
"""
from __future__ import absolute_import, division, print_function

import os
import time

import tensorflow as tf

import utils.file_io as io

from config import HEADER, PI, PROJECT_DIR, SEP, TF_FLOAT
from dynamics.gauge_dynamics import (build_dynamics, convert_to_angle,
                                     GaugeDynamics)
from utils.attr_dict import AttrDict
from utils.plotting_utils import plot_data
from utils.training_utils import summarize_dict
from utils.data_containers import DataContainer

# pylint:disable=no-member
if tf.__version__.startswith('1.'):
    TF_VERSION = '1.x'
elif tf.__version__.startswith('2.'):
    TF_VERSION = '2.x'

try:
    import horovod.tensorflow as hvd

    hvd.init()
    RANK = hvd.rank()
    io.log(f'Number of devices: {hvd.size()}', RANK)
    GPUS = tf.config.list_physical_devices('GPU')
    for gpu in GPUS:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:  # noqa: E722 pylint:disable=bare-except noqa:E722
            # Invalid device or cannot modify virtual devices once initialized
            pass
    if GPUS:
        tf.config.experimental.set_visible_devices(
            GPUS[hvd.local_rank()], 'GPU'
        )

except ImportError:
    RANK = 0

IS_CHIEF = (RANK == 0)


def print_args(args):
    """Print out parsed arguments."""
    io.log(80 * '=' + '\n' + 'Parsed args:\n')
    for key, val in args.items():
        io.log(f' {key}: {val}\n')
    io.log(80 * '=')


def restore_from_train_flags(args):
    """Populate entries in `args` using the training `FLAGS` from `log_dir`."""
    train_dir = os.path.join(args.log_dir, 'training')
    flags = AttrDict(dict(io.loadz(os.path.join(train_dir, 'FLAGS.z'))))
    flags.horovod = False
    if args.get('lattice_shape', None) is None:
        args.lattice_shape = flags.lattice_shape
    if args.get('beta', None) is None:
        args.beta = flags.beta_final
    if args.get('num_steps', None) is None:
        args.num_steps = flags.num_steps
    #  if args.get('eps', None) is None:
    #      eps = io.loadz(os.path.join(train_dir, 'train_data', 'eps.z'))[-1]
    #      args.eps = eps

    flags.update({
        #  'eps': args.eps,
        'beta': args.beta,
        'num_steps': args.num_steps,
        'lattice_shape': args.lattice_shape,
    })

    return flags


def run_hmc(
        args: AttrDict,
        hmc_dir: str = None,
        skip_existing: bool = False,
) -> (GaugeDynamics, DataContainer):
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
    if not IS_CHIEF:
        return None, None

    if args.log_dir is not None:
        args = restore_from_train_flags(args)

    args.update({
        'hmc': True,
        'units': [],
        'lr_init': 0,
        'restore': False,
        'use_ncp': False,
        'horovod': False,
        'eps_fixed': True,
        'warmup_steps': 0,
        'dropout_prob': 0.,
        #  'plaq_weight': 10.,
        #  'charge_weight': 0.1,
        'lr_decay_steps': None,
        'lr_decay_rate': None,
        'separate_networks': False,
    })

    io.print_flags(args)

    if hmc_dir is None:
        root_dir = os.path.dirname(PROJECT_DIR)
        hmc_dir = os.path.join(root_dir, 'gauge_logs_eager', 'hmc_runs')

    io.check_else_make_dir(hmc_dir)

    def get_run_fstr(run_dir):
        _, tail = os.path.split(run_dir)
        fstr = tail.split('-')[0]
        return fstr

    if skip_existing:
        run_dirs = [os.path.join(hmc_dir, i) for i in os.listdir(hmc_dir)]
        run_fstrs = [get_run_fstr(i) for i in run_dirs]
        run_fstr = io.get_run_dir_fstr(args)
        if run_fstr in run_fstrs:
            io.log('ERROR:Existing run found! Skipping.')
            return None, None, None

    dynamics = build_dynamics(args)
    dynamics, run_data, x = run(dynamics, args,
                                runs_dir=hmc_dir)

    return dynamics, run_data, x


def load_and_run(
        args: AttrDict,
        x: tf.Tensor = None,
        runs_dir: str = None,
) -> (GaugeDynamics, AttrDict):
    """Load trained model from checkpoint and run inference."""
    if not IS_CHIEF:
        return None, None, None

    print_args(args)
    ckpt_dir = os.path.join(args.log_dir, 'training', 'checkpoints')
    flags = restore_from_train_flags(args)
    eps_file = os.path.join(args.log_dir, 'training', 'train_data', 'eps.z')
    flags.eps = io.loadz(eps_file)[-1]
    dynamics = build_dynamics(flags)

    ckpt = tf.train.Checkpoint(dynamics=dynamics,
                               optimizer=dynamics.optimizer)
    manager = tf.train.CheckpointManager(ckpt, max_to_keep=5,
                                         directory=ckpt_dir)
    if manager.latest_checkpoint:
        io.log(f'INFO:Restored model from: {manager.latest_checkpoint}')
        status = ckpt.restore(manager.latest_checkpoint)
        status.assert_existing_objects_matched()
        xfile = os.path.join(args.log_dir, 'training',
                             'train_data', f'x_rank{RANK}.z')
        io.log(f'INFO:Restored x from: {xfile}.')
        x = io.loadz(xfile)

    dynamics, run_data, x = run(dynamics, args, x=x, runs_dir=runs_dir)

    return dynamics, run_data, x


def run(dynamics, args, x=None, runs_dir=None):
    """Run inference.

    Returns:
        model(GaugeModel): Trained model
        ouptuts(dict): Dictionary of outputs from inference run.
    """
    #  is_chief = check_if_chief(args)
    if not IS_CHIEF:
        return None, None, None

    if runs_dir is None:
        if args.hmc:
            runs_dir = os.path.join(args.log_dir, 'inference_hmc')
        else:
            runs_dir = os.path.join(args.log_dir, 'inference')

    io.check_else_make_dir(runs_dir)
    run_dir = io.make_run_dir(args, runs_dir)
    data_dir = os.path.join(run_dir, 'run_data')
    summary_dir = os.path.join(run_dir, 'summaries')
    log_file = os.path.join(run_dir, 'run_log.txt')
    io.check_else_make_dir([run_dir, data_dir, summary_dir])
    writer = tf.summary.create_file_writer(summary_dir)
    writer.set_as_default()

    run_steps = args.get('run_steps', 2000)
    beta = args.get('beta', None)
    if beta is None:
        beta = args.get('beta_final', None)

    if x is None:
        x = convert_to_angle(tf.random.normal(shape=dynamics.x_shape))

    run_data, x, _ = run_dynamics(dynamics, args, x, save_x=False, )

    run_data.flush_data_strs(log_file, mode='a')
    io.save_inference(run_dir, run_data)
    if args.get('save_run_data', True):
        run_data.save_data(data_dir)

    eps = dynamics.eps
    if hasattr(eps, 'numpy'):
        eps = eps.numpy()

    run_params = {
        'eps': eps,
        'beta': beta,
        'run_steps': run_steps,
        'plaq_weight': dynamics.plaq_weight,
        'charge_weight': dynamics.charge_weight,
        'lattice_shape': dynamics.lattice_shape,
        'num_steps': dynamics.config.num_steps,
        'net_weights': dynamics.net_weights,
        'input_shape': dynamics.x_shape,
    }
    run_params.update(dynamics.params)
    io.save_params(run_params, run_dir, name='run_params')

    plot_data(run_data, run_dir, args, thermalize=True, params=run_params)

    return dynamics, run_data, x


def run_dynamics(dynamics, flags, x=None, save_x=False, run_md=False):
    """Run inference on trained dynamics."""
    if not IS_CHIEF:
        return None, None

    # -------------------------------------------------------------
    md_steps = 10
    # Setup
    print_steps = flags.get('print_steps', 5)
    beta = flags.get('beta', flags.get('beta_final', None))

    #  if flags.get('compile', True):
    test_step = tf.function(dynamics.test_step)
    #  else:
    #      test_step = dynamics.test_step

    if x is None:
        x = tf.random.uniform(shape=dynamics.x_shape,
                              minval=-PI, maxval=PI,
                              dtype=TF_FLOAT)

    run_data = DataContainer(flags.run_steps)

    template = '\n'.join([f'beta: {beta}',
                          f'eps: {dynamics.eps.numpy():.4g}',
                          f'net_weights: {dynamics.net_weights}'])
    io.log(f'INFO:Running inference with:\n {template}')

    # Run 50 MD updates (w/o accept/reject) to ensure chains don't get stuck
    if run_md:
        md_steps = 10
        for _ in range(md_steps):
            mc_states, _ = dynamics.md_update(x, beta, training=False)
            x = mc_states.out.x

    try:
        x, metrics = test_step(x, beta)
    except:
        test_step = dynamics.test_step
        x, metrics = test_step(x, beta)

    header = run_data.get_header(metrics,
                                 skip=['charges'],
                                 prepend=['{:^12s}'.format('step')])
    io.log(header)
    # -------------------------------------------------------------

    x_arr = []

    def timed_step(x: tf.Tensor, beta: tf.Tensor):
        start = time.time()
        x, metrics = test_step(x, beta)
        metrics.dt = time.time() - start
        if save_x:
            x_arr.append(x.numpy())

        return x, metrics

    steps = tf.range(flags.run_steps, dtype=tf.int64)
    for step in steps:
        x, metrics = timed_step(x, beta)
        run_data.update(step, metrics)

        if step % print_steps == 0:
            data_str = run_data.get_fstr(step, metrics, skip=['charges'])
            summarize_dict(metrics, step, prefix='testing')
            io.log(data_str)

        #  if (step + 1) % 250 == 0:
        #      io.log(f'Running {md_steps} MD updates (no accept/reject)...')
        #      for _ in range(md_steps):
        #          mc_states, _ = dynamics.md_update(x, beta, training=False)
        #          x = mc_states.out.x

        if step % 100 == 0:
            io.log(header)

    return run_data, x, x_arr
