"""
inference_utils.py

Collection of helper methods to use for running inference on trained model.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import time
import logging
from typing import Optional

from tqdm.autonotebook import tqdm
import tensorflow as tf
try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
    RANK = hvd.rank()
    IS_CHIEF = (RANK == 0)
    NUM_NODES = hvd.size()
except (ImportError, ModuleNotFoundError):
    HAS_HOROVOD = False
    RANK = 0
    IS_CHIEF = (RANK == 0)
    NUM_NODES = 1

import utils.file_io as io

from config import (HEADER, PI, PROJECT_DIR, SEP, TF_FLOAT, CBARS, LOGS_DIR,
                    GAUGE_LOGS_DIR, HMC_LOGS_DIR)
from dynamics.gauge_dynamics import (build_dynamics, convert_to_angle,
                                     GaugeDynamics)
from utils.attr_dict import AttrDict
from utils.plotting_utils import plot_data
from utils.summary_utils import summarize_dict
from utils.data_containers import DataContainer

# pylint:disable=no-member

if IS_CHIEF:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
        stream=sys.stdout
    )
else:
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(asctime)s:%(levelname)s:%(message)s",
        stream=None
    )


SKIP = ['charges', 'sldf', 'sldb', 'Hf', 'Hb', 'Hwf', 'Hwb',
        'ldf_start', 'ldb_start', 'ldf_mid', 'ldf_end',
        'ldb_mid', 'ldb_end', 'Hf_start', 'Hf_mid', 'Hf_end',
        'Hb_start', 'Hb_mid', 'Hb_end']


def restore_from_train_flags(args):
    """Populate entries in `args` using the training `FLAGS` from `log_dir`."""
    train_dir = os.path.join(args.log_dir, 'training')
    flags = AttrDict(dict(io.loadz(os.path.join(train_dir, 'FLAGS.z'))))

    return flags


def run_hmc(
        args: AttrDict,
        hmc_dir: str = None,
        skip_existing: bool = False,
) -> (GaugeDynamics, DataContainer, tf.Tensor):
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
        return None, None, None

    if hmc_dir is None:
        #  root_dir = os.path.join(HMC_LOGS_DIR)
        #  root_dir = os.path.join(GAUGE_LOGS_DIR, 'hmc_logs')
        month_str = io.get_timestamp('%Y_%m')
        hmc_dir = os.path.join(HMC_LOGS_DIR, month_str)

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
    dynamics, run_data, x = run(dynamics, args, runs_dir=hmc_dir)

    return dynamics, run_data, x


def load_and_run(
        args: AttrDict,
        x: tf.Tensor = None,
        runs_dir: str = None,
) -> (GaugeDynamics, DataContainer, tf.Tensor):
    """Load trained model from checkpoint and run inference."""
    if not IS_CHIEF:
        return None, None, None

    io.print_dict(args)
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
        io.log(f'Restored model from: {manager.latest_checkpoint}')
        status = ckpt.restore(manager.latest_checkpoint)
        status.assert_existing_objects_matched()
        xfile = os.path.join(args.log_dir, 'training',
                             'train_data', 'x_rank0.z')
        io.log(f'Restored x from: {xfile}.')
        x = io.loadz(xfile)

    dynamics, run_data, x = run(dynamics, args, x=x, runs_dir=runs_dir)

    return dynamics, run_data, x


def run(
        dynamics: GaugeDynamics,
        args: AttrDict,
        x: tf.Tensor = None,
        runs_dir: str = None,
        make_plots=True,
) -> (GaugeDynamics, DataContainer, tf.Tensor):
    """Run inference."""
    if not IS_CHIEF:
        return None, None, None

    if runs_dir is None:
        if dynamics.config.hmc:
            runs_dir = os.path.join(args.log_dir, 'inference_hmc')
        else:
            runs_dir = os.path.join(args.log_dir, 'inference')

    eps = dynamics.eps
    if hasattr(eps, 'numpy'):
        eps = eps.numpy()

    try:
        args.eps = eps
    except AttributeError:
        args.update({'eps': eps})

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

    run_data, x, _ = run_dynamics(dynamics, args, x, save_x=False)

    run_data.flush_data_strs(log_file, mode='a')
    run_data.write_to_csv(args.log_dir, run_dir, hmc=dynamics.config.hmc)
    io.save_inference(run_dir, run_data)
    if args.get('save_run_data', True):
        run_data.save_data(data_dir)

    run_params = {
        'run_dir': run_dir,
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
    #  run_params.update(dynamics.params)
    io.save_params(run_params, run_dir, name='run_params')

    args.logging_steps = 1
    if make_plots:
        plot_data(run_data, run_dir, args, thermalize=True, params=run_params)

    return dynamics, run_data, x


def run_dynamics(
        dynamics: GaugeDynamics,
        flags: AttrDict,
        x: tf.Tensor = None,
        save_x: bool = False,
        md_steps: int = 0,
) -> (DataContainer, tf.Tensor, list):
    """Run inference on trained dynamics."""
    if not IS_CHIEF:
        return None, None, None

    # Setup
    print_steps = flags.get('print_steps', 5)
    beta = flags.get('beta', flags.get('beta_final', None))

    test_step = dynamics.test_step
    if flags.get('compile', True):
        test_step = tf.function(dynamics.test_step)
        io.log('Compiled `dynamics.test_step` using tf.function!')

    if x is None:
        x = tf.random.uniform(shape=dynamics.x_shape,
                              minval=-PI, maxval=PI,
                              dtype=TF_FLOAT)

    run_data = DataContainer(flags.run_steps)

    template = '\n'.join([f'beta: {beta}',
                          f'eps: {dynamics.eps.numpy():.4g}',
                          f'net_weights: {dynamics.net_weights}'])
    io.log(f'Running inference with:\n {template}')

    # Run 50 MD updates (w/o accept/reject) to ensure chains don't get stuck
    if md_steps > 0:
        for _ in range(md_steps):
            mc_states, _ = dynamics.md_update(x, beta, training=False)
            x = mc_states.out.x

    try:
        x, metrics = test_step((x, tf.constant(beta)))
    except Exception as exception:  # pylint:disable=broad-except
        io.log(f'Exception: {exception}')
        test_step = dynamics.test_step
        x, metrics = test_step((x, tf.constant(beta)))

    header = run_data.get_header(metrics,
                                 skip=SKIP,
                                 prepend=['{:^12s}'.format('step')])
    #  io.log(header)
    io.log(header.split('\n'), should_print=True)
    # -------------------------------------------------------------

    x_arr = []

    def timed_step(x: tf.Tensor, beta: tf.Tensor):
        start = time.time()
        x, metrics = test_step((x, tf.constant(beta)))
        metrics.dt = time.time() - start
        if save_x:
            x_arr.append(x.numpy())

        return x, metrics

    steps = tf.range(flags.run_steps, dtype=tf.int64)
    if NUM_NODES == 1:
        ctup = (CBARS['red'], CBARS['green'], CBARS['red'], CBARS['reset'])
        steps = tqdm(steps, desc='running', unit='step',
                     bar_format=("%s{l_bar}%s{bar}%s{r_bar}%s" % ctup))

    for step in steps:
        x, metrics = timed_step(x, beta)
        run_data.update(step, metrics)

        if step % print_steps == 0:
            summarize_dict(metrics, step, prefix='testing')
            data_str = run_data.get_fstr(step, metrics, skip=SKIP)
            io.log(data_str, should_print=True)

        if (step + 1) % 1000 == 0:
            io.log(header, should_print=True)

    return run_data, x, x_arr
