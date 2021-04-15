"""
inference_utils.py

Collection of helper methods to use for running inference on trained model.
"""
from __future__ import absolute_import, division, print_function

import os
import time

from rich.progress import track
from rich.console import Console
from typing import Optional
from pathlib import Path
from collections import namedtuple

import tensorflow as tf

import utils.file_io as io

from config import HMC_LOGS_DIR, PI, TF_FLOAT
from utils import SKEYS
from utils.file_io import IS_CHIEF, NUM_WORKERS
from utils.attr_dict import AttrDict
from utils.summary_utils import summarize_dict
from utils.plotting_utils import plot_data
from utils.data_containers import DataContainer
from dynamics.config import GaugeDynamicsConfig
from dynamics.gauge_dynamics import (build_dynamics, convert_to_angle,
                                     GaugeDynamics)

SHOULD_TRACK = not os.environ.get('NOTRACK', False)

InferenceResults = namedtuple('InferenceResults',
                              ['dynamics', 'run_data', 'x', 'x_arr'])


def restore_from_train_flags(args):
    """Populate entries in `args` using the training `FLAGS` from `log_dir`."""
    train_dir = os.path.join(args.log_dir, 'training')
    flags = AttrDict(dict(io.loadz(os.path.join(train_dir, 'FLAGS.z'))))

    return flags


def _find_configs(log_dir):
    configs_file = os.path.join(log_dir, 'configs.z')
    if os.path.isfile(configs_file):
        return io.loadz(configs_file)
    configs = [
        x for x in Path(log_dir).rglob('*configs.z*') if x.is_file()
    ]
    if configs != []:
        return io.loadz(configs[0])
    configs = [
        x for x in Path(log_dir).rglob('*FLAGS.z*') if x.is_file()
    ]
    if configs != []:
        return io.loadz(configs[0])

    return None


def short_training(
        train_steps: int,
        beta: float,
        log_dir: str,
        dynamics: GaugeDynamics,
        x: tf.Tensor = None,
):
    """Perform a brief training run prior to running inference."""
    ckpt_dir = os.path.join(log_dir, 'training', 'checkpoints')
    ckpt = tf.train.Checkpoint(dynamics=dynamics, optimizer=dynamics.optimizer)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
    current_step = 0
    if manager.latest_checkpoint:
        io.log(f'Restored model from: {manager.latest_checkpoint}')
        ckpt.restore(manager.latest_checkpoint)
        current_step = dynamics.optimizer.iterations.numpy()

    if x is None:
        x = convert_to_angle(tf.random.normal(dynamics.x_shape))

    train_data = DataContainer(current_step+train_steps, print_steps=1)

    dynamics.compile(loss=dynamics.calc_losses,
                     optimizer=dynamics.optimizer,
                     experimental_run_tf_function=False)

    x, metrics = dynamics.train_step((x, tf.constant(beta)))

    header = train_data.get_header(metrics, skip=SKEYS,
                                   prepend=['{:^12s}'.format('step')])
    io.log(header.split('\n'))
    for step in range(current_step, current_step + train_steps):
        start = time.time()
        x, metrics = dynamics.train_step((x, tf.constant(beta)))
        metrics.dt = time.time() - start
        data_str = train_data.get_fstr(step, metrics, skip=SKEYS)
        io.log(data_str)

    return dynamics, train_data, x


def _get_hmc_log_str(configs):
    dynamics_config = configs.get('dynamics_config', None)

    lf = dynamics_config.get('num_steps', None)
    eps = dynamics_config.get('eps', None)
    ls = dynamics_config.get('x_shape', None)
    bs = ls[0]  # batch size
    nx = ls[1]  # size in 'x' direction

    b = configs.get('beta', None)
    if b is None:
        b = configs.get('beta_final', None)

    log_str = (
        f'HMC_L{nx}_b{bs}_beta{float(b)}_lf{lf}_eps{eps}'.replace('.0', '')
    )

    log_str = log_str.replace('.', '')

    return log_str


def run_hmc(
        args: AttrDict,
        hmc_dir: str = None,
        skip_existing: bool = False,
        save_x: bool = False,
        therm_frac: float = 0.33,
        num_chains: int = 16,
        make_plots: bool = True,
) -> (InferenceResults):
    """Run HMC using `inference_args` on a model specified by `params`.

    NOTE:
    -----
    args should be a dict with the following keys:
        - 'hmc'
        - 'eps'
        - 'beta'
        - 'num_steps'
        - 'run_steps'
        - 'x_shape'
    """
    if not IS_CHIEF:
        return InferenceResults(None, None, None, None)

    if hmc_dir is None:
        month_str = io.get_timestamp('%Y_%m')
        hmc_dir = os.path.join(HMC_LOGS_DIR, month_str)

    io.check_else_make_dir(hmc_dir)

    def get_run_fstr(run_dir):
        # take relevant part of run_dir:
        # ```
        # /path/to/HMC_L16_b512... -> HMC_L16_b512...
        # ```
        _, tail = os.path.split(run_dir)
        # strip off timestamp at the end of `run_dir`
        fstr = tail.split('-')[0]
        return fstr

    if skip_existing:
        fstr = io.get_run_dir_fstr(args)
        base_dir = os.path.dirname(hmc_dir)
        matches = list(
            Path(base_dir).rglob(f'*{fstr}*')
        )
        if len(matches) > 0:
            io.rule('Existing run with current parameters found!')
            io.log(args)
            return InferenceResults(None, None, None, None)

    dynamics = build_dynamics(args)
    try:
        inference_results = run(dynamics=dynamics, args=args, runs_dir=hmc_dir,
                                make_plots=make_plots, save_x=save_x,
                                therm_frac=therm_frac, num_chains=num_chains)
    except FileExistsError:
        inference_results = None
        io.rule('Existing run with current parameters found! Skipping!!')

    return inference_results


def load_and_run(
        args: AttrDict,
        x: tf.Tensor = None,
        runs_dir: str = None,
        save_x: bool = False,
) -> (GaugeDynamics, DataContainer, tf.Tensor):
    """Load trained model from checkpoint and run inference."""
    if not IS_CHIEF:
        return InferenceResults(None, None, None, None)

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

    inference_results = run(dynamics, args, x=x,
                            runs_dir=runs_dir, save_x=save_x)

    return inference_results


def run_inference_from_log_dir(
        log_dir: str,
        run_steps: int = 50000,
        beta: float = None,
        eps: float = None,
        make_plots: bool = True,
        train_steps: int = 10,
        therm_frac: float = 0.33,
        batch_size: int = 16,
        num_chains: int = 16,
        x: tf.Tensor = None,
) -> InferenceResults:      # (type: InferenceResults)
    """Run inference by loading networks in from `log_dir`."""
    configs = _find_configs(log_dir)
    if configs is None:
        raise FileNotFoundError(
            f'Unable to load configs from `log_dir`: {log_dir}. Exiting'
        )

    if eps is not None:
        configs['dynamics_config']['eps'] = eps

    else:
        try:
            eps_file = os.path.join(log_dir, 'training', 'models', 'eps.z')
            eps = io.loadz(eps_file)
        except FileNotFoundError:
            eps = configs.get('dynamics_config', None).get('eps', None)

    if beta is not None:
        configs.update({'beta': beta, 'beta_final': beta})

    if batch_size is not None:
        batch_size = int(batch_size)
        prev_shape = configs['dynamics_config']['x_shape']
        new_shape = (batch_size, *prev_shape[1:])
        configs['dynamics_config']['x_shape'] = new_shape

    configs = AttrDict(configs)
    dynamics = build_dynamics(configs)
    xnet, vnet = dynamics._load_networks(log_dir)
    dynamics.xnet = xnet
    dynamics.vnet = vnet

    if train_steps > 0:
        dynamics, train_data, x = short_training(train_steps,
                                                 configs.beta_final,
                                                 log_dir, dynamics, x=x)
    else:
        dynamics.compile(loss=dynamics.calc_losses,
                         optimizer=dynamics.optimizer,
                         experimental_run_tf_function=False)

    _, log_str = os.path.split(log_dir)

    if x is None:
        x = convert_to_angle(tf.random.normal(dynamics.x_shape))

    configs['run_steps'] = run_steps
    configs['print_steps'] = max((run_steps // 100, 1))
    configs['md_steps'] = 100
    runs_dir = os.path.join(log_dir, 'LOADED', 'inference')
    io.check_else_make_dir(runs_dir)
    io.save_dict(configs, runs_dir, name='inference_configs')
    inference_results = run(dynamics=dynamics,
                            args=configs, x=x, beta=beta,
                            runs_dir=runs_dir, make_plots=make_plots,
                            therm_frac=therm_frac, num_chains=num_chains)

    return inference_results


def run(
        dynamics: GaugeDynamics,
        args: AttrDict,
        x: tf.Tensor = None,
        beta: float = None,
        runs_dir: str = None,
        make_plots: bool = True,
        therm_frac: float = 0.33,
        num_chains: int = 16,
        save_x: bool = False,
        md_steps: int = 50,
        console: Console = None,
        skip_existing: bool = False,
        run_steps: int = None,
) -> (InferenceResults):
    """Run inference. (Note: Higher-level than `run_dynamics`)."""
    if num_chains > 16:
        print(f'Reducing `num_chains` from: {num_chains} to {16}.')
        num_chains = 16

    if not IS_CHIEF:
        return InferenceResults(None, None, None, None)

    if runs_dir is None:
        if dynamics.config.hmc:
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

    args.logging_steps = 1
    #  run_steps = args.get('run_steps', 50000)
    if run_steps is None:
        run_steps = args.get('run_steps', 50000)

    if beta is None:
        beta = args.get('beta_final', args.get('beta', None))

    if x is None:
        x = convert_to_angle(tf.random.normal(shape=dynamics.x_shape))

    results = run_dynamics(dynamics=dynamics,
                           flags=args, x=x, beta=beta, save_x=save_x,
                           md_steps=md_steps, console=console)
    run_data = results.run_data

    run_data.update_dirs({
        'log_dir': args.log_dir,
        'run_dir': run_dir,
    })
    run_data.flush_data_strs(log_file, mode='a')
    run_data.write_to_csv(args.log_dir, run_dir, hmc=dynamics.config.hmc)
    io.save_inference(run_dir, run_data)
    if args.get('save_run_data', True):
        run_data.save_data(data_dir)

    run_params = {
        'hmc': dynamics.config.hmc,
        'run_dir': run_dir,
        'beta': beta,
        'run_steps': run_steps,
        'plaq_weight': dynamics.plaq_weight,
        'charge_weight': dynamics.charge_weight,
        'x_shape': dynamics.x_shape,
        'num_steps': dynamics.config.num_steps,
        'net_weights': dynamics.net_weights,
        'input_shape': dynamics.x_shape,
    }

    traj_len = dynamics.config.num_steps * tf.reduce_mean(dynamics.xeps)

    if hasattr(dynamics, 'xeps') and hasattr(dynamics, 'veps'):
        xeps_avg = tf.reduce_mean(dynamics.xeps)
        veps_avg = tf.reduce_mean(dynamics.veps)
        traj_len = tf.reduce_sum(dynamics.xeps)
        run_params.update({
            'xeps': dynamics.xeps,
            'traj_len': traj_len,
            'veps': dynamics.veps,
            'xeps_avg': xeps_avg,
            'veps_avg': veps_avg,
            'eps_avg': (xeps_avg + veps_avg) / 2.,
        })

    elif hasattr(dynamics, 'eps'):
        run_params.update({
            'eps': dynamics.eps,
        })

    io.save_params(run_params, run_dir, name='run_params')

    if make_plots:
        output = plot_data(data_container=run_data,
                           flags=args,
                           params=run_params,
                           out_dir=run_dir,
                           hmc=dynamics.config.hmc,
                           therm_frac=therm_frac,
                           num_chains=num_chains)
        tint_data = {
            'beta': beta,
            'run_dir': run_dir,
            'traj_len': traj_len,
            'run_params': run_params,
            'eps': run_params['eps_avg'],
            'lf': run_params['num_steps'],
            'narr': output['tint_dict']['narr'],
            'tint': output['tint_dict']['tint'],
        }

        tint_file = os.path.join(run_dir, 'tint_data.z')
        io.savez(tint_data, tint_file, 'tint_data')

    return InferenceResults(dynamics=results.dynamics, run_data=run_data,
                            x=results.x, x_arr=results.x_arr)


def run_dynamics(
        dynamics: GaugeDynamics,
        flags: AttrDict,
        x: tf.Tensor = None,
        beta: float = None,
        save_x: bool = False,
        md_steps: int = 0,
        console: Console = None,
        should_track: bool = False,
) -> (InferenceResults):
    """Run inference on trained dynamics."""
    if not IS_CHIEF:
        return InferenceResults(None, None, None, None)

    # -- Setup -----------------------------
    print_steps = flags.get('print_steps', 5)
    if beta is None:
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
                          f'net_weights: {dynamics.net_weights}'])
    io.log(f'Running inference with:\n {template}')

    # Run `md_steps MD updates (w/o accept/reject)
    # to ensure chains don't get stuck
    if md_steps > 0:
        for _ in range(md_steps):
            mc_states, _ = dynamics.md_update((x, beta), training=False)
            x = mc_states.out.x

    try:
        x, metrics = test_step((x, tf.constant(beta)))
    except Exception as exception:  # pylint:disable=broad-except
        io.log(f'Exception: {exception}')
        test_step = dynamics.test_step
        x, metrics = test_step((x, tf.constant(beta)))

    x_arr = []

    def timed_step(x: tf.Tensor, beta: tf.Tensor):
        start = time.time()
        x, metrics = test_step((x, tf.constant(beta)))
        metrics.dt = time.time() - start
        if save_x:
            x_arr.append(x.numpy())

        return x, metrics

    if flags.run_steps < 1000:
        summary_steps = 5
    else:
        summary_steps = flags.run_steps // 100

    if console is None:
        console = io.console

    steps = tf.range(flags.run_steps, dtype=tf.int64)
    if SHOULD_TRACK and should_track:
        tracked_iter = track(enumerate(steps), total=len(steps),
                             description='Inference', transient=True,
                             console=console)
    else:
        tracked_iter = enumerate(steps)

    for idx, step in tracked_iter:
        x, metrics = timed_step(x, beta)
        run_data.update(step, metrics)  # update data after every accept/reject

        if step % summary_steps == 0:
            summarize_dict(metrics, step, prefix='testing')

        if step % print_steps == 0:
            data_str = run_data.get_fstr(step, metrics, skip=SKEYS)
            io.log(data_str)

    return InferenceResults(dynamics=dynamics,
                            run_data=run_data,
                            x=x, x_arr=x_arr)
