"""
inference_utils.py

Collection of helper methods to use for running inference on trained model.
"""
from __future__ import absolute_import, division, print_function, annotations

import os
import time

from typing import Any
from pathlib import Path
from collections import namedtuple

import tensorflow as tf

import utils.file_io as io

from config import HMC_LOGS_DIR, PI, TF_FLOAT
from utils import SKEYS
from utils.hvd_init import IS_CHIEF
#  from utils.file_io import IS_CHIEF, NUM_WORKERS
from utils.attr_dict import AttrDict
from utils.summary_utils import summarize_dict
from utils.plotting_utils import plot_data
from utils.data_containers import DataContainer
from dynamics.config import GaugeDynamicsConfig
from dynamics.gauge_dynamics import (build_dynamics, convert_to_angle,
                                     GaugeDynamics)
from utils.logger import Logger

SHOULD_TRACK = not os.environ.get('NOTRACK', False)

InferenceResults = namedtuple('InferenceResults',
                              ['dynamics', 'x', 'x_arr',
                               'run_data', 'data_strs'])

logger = Logger()


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
        train_data.update(step, metrics)
        logger.print_metrics(metrics)
        #  data_str = train_data.get_fstr(step, metrics, skip=SKEYS)
        #  io.log(data_str)

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
        configs: dict[str, Any],
        hmc_dir: str = None,
        skip_existing: bool = False,
        save_x: bool = False,
        therm_frac: float = 0.33,
        num_chains: int = 16,
        make_plots: bool = True,
) -> InferenceResults:
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
        return InferenceResults(None, None, None, None, None)

    if hmc_dir is None:
        month_str = io.get_timestamp('%Y_%m')
        hmc_dir = os.path.join(HMC_LOGS_DIR, month_str)

    io.check_else_make_dir(hmc_dir)

    if skip_existing:
        fstr = io.get_run_dir_fstr(configs)
        base_dir = os.path.dirname(hmc_dir)
        matches = list(
            Path(base_dir).rglob(f'*{fstr}*')
        )
        if len(matches) > 0:
            logger.warning('Existing run with current parameters found!')
            logger.print_dict(configs)
            return InferenceResults(None, None, None, None, None)

    dynamics = build_dynamics(configs)
    try:
        inference_results = run(dynamics=dynamics, configs=configs,
                                runs_dir=hmc_dir, make_plots=make_plots,
                                save_x=save_x, therm_frac=therm_frac,
                                num_chains=num_chains)
    except FileExistsError:
        inference_results = InferenceResults(None, None, None, None, None)
        logger.warning('Existing run with current parameters found! Skipping!')

    return inference_results


def load_and_run(
        args: AttrDict,
        x: tf.Tensor = None,
        runs_dir: str = None,
        save_x: bool = False,
) -> InferenceResults:
    """Load trained model from checkpoint and run inference."""
    if not IS_CHIEF:
        return InferenceResults(None, None, None, None, None)

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
                            configs=configs, x=x, beta=beta,
                            runs_dir=runs_dir, make_plots=make_plots,
                            therm_frac=therm_frac, num_chains=num_chains)

    return inference_results


def run(
        dynamics: GaugeDynamics,
        configs: dict[str, Any],
        x: tf.Tensor = None,
        beta: float = None,
        runs_dir: str = None,
        make_plots: bool = True,
        therm_frac: float = 0.33,
        num_chains: int = 16,
        save_x: bool = False,
        md_steps: int = 50,
        skip_existing: bool = False,
        save_dataset: bool = True,
        use_hdf5: bool = False,
        skip_keys: list[str] = None,
        run_steps: int = None,
) -> InferenceResults:
    """Run inference. (Note: Higher-level than `run_dynamics`)."""
    if not IS_CHIEF:
        return InferenceResults(None, None, None, None, None)

    if num_chains > 16:
        logger.warning(f'Reducing `num_chains` from: {num_chains} to {16}.')
        num_chains = 16

    if run_steps is None:
        run_steps = configs.get('run_steps', 50000)

    if beta is None:
        beta = configs.get('beta_final', configs.get('beta', None))

    assert beta is not None

    logdir = configs.get('log_dir', configs.get('logdir', None))
    if runs_dir is None:
        rs = 'inference_hmc' if dynamics.config.hmc else 'inference'
        runs_dir = os.path.join(logdir, rs)

    io.check_else_make_dir(runs_dir)
    run_dir = io.make_run_dir(configs=configs, base_dir=runs_dir,
                              beta=beta, skip_existing=skip_existing)
    logger.info(f'run_dir: {run_dir}')
    data_dir = os.path.join(run_dir, 'run_data')
    summary_dir = os.path.join(run_dir, 'summaries')
    log_file = os.path.join(run_dir, 'run_log.txt')
    io.check_else_make_dir([run_dir, data_dir, summary_dir])
    writer = tf.summary.create_file_writer(summary_dir)
    writer.set_as_default()

    configs['logging_steps'] = 1
    if x is None:
        x = convert_to_angle(tf.random.uniform(shape=dynamics.x_shape,
                                               minval=-PI, maxval=PI))


    # == RUN DYNAMICS =======================================================
    nw = dynamics.net_weights
    inf_type = 'HMC' if dynamics.config.hmc else 'inference'
    logger.info(', '.join([f'Running {inf_type}', f'beta={beta}', f'nw={nw}']))
    t0 = time.time()
    results = run_dynamics(dynamics, flags=configs,
                           x=x, beta=beta, save_x=save_x,
                           md_steps=md_steps)
    logger.info(f'Done running {inf_type}. took: {time.time() - t0:.4f} s')
    #========================================================================

    run_data = results.run_data
    run_data.update_dirs({'log_dir': logdir, 'run_dir': run_dir})
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

    inf_log_fpath = os.path.join(run_dir, 'inference_log.txt')
    with open(inf_log_fpath, 'a') as f:
        f.writelines('\n'.join(results.data_strs))

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

    save_times = {}
    plot_times = {}
    if make_plots:
        output = plot_data(data_container=run_data,
                           configs=configs,
                           params=run_params,
                           out_dir=run_dir,
                           hmc=dynamics.config.hmc,
                           therm_frac=therm_frac,
                           num_chains=num_chains,
                           profile=True,
                           cmap='crest',
                           logging_steps=1)

        save_times = io.SortedDict(**output['save_times'])
        plot_times = io.SortedDict(**output['plot_times'])
        dt1 = io.SortedDict(**plot_times['data_container.plot_dataset'])
        plot_times['data_container.plot_dataset'] = dt1

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

        t0 = time.time()
        tint_file = os.path.join(run_dir, 'tint_data.z')
        io.savez(tint_data, tint_file, 'tint_data')
        save_times['savez_tint_data'] = time.time() - t0

    t0 = time.time()

    logfile = os.path.join(run_dir, 'inference.log')
    run_data.save_and_flush(data_dir=data_dir,
                            log_file=logfile,
                            use_hdf5=use_hdf5,
                            skip_keys=skip_keys,
                            save_dataset=save_dataset)

    save_times['run_data.flush_data_strs'] = time.time() - t0

    t0 = time.time()
    try:
        run_data.write_to_csv(logdir, run_dir, hmc=dynamics.config.hmc)
    except TypeError:
        logger.warning(f'Unable to write to csv. Continuing...')

    save_times['run_data.write_to_csv'] = time.time() - t0

    # t0 = time.time()
    # io.save_inference(run_dir, run_data)
    # save_times['io.save_inference'] = time.time() - t0
    profdir = os.path.join(run_dir, 'profile_info')
    io.check_else_make_dir(profdir)
    io.save_dict(plot_times, profdir, name='plot_times')
    io.save_dict(save_times, profdir, name='save_times')

    return InferenceResults(dynamics=results.dynamics,
                            x=results.x, x_arr=results.x_arr,
                            run_data=results.run_data,
                            data_strs=results.data_strs)


def run_dynamics(
        dynamics: GaugeDynamics,
        flags: dict,
        x: tf.Tensor = None,
        beta: float = None,
        save_x: bool = False,
        md_steps: int = 0,
        #  should_track: bool = False,
) -> (InferenceResults):
    """Run inference on trained dynamics."""
    if not IS_CHIEF:
        return InferenceResults(None, None, None, None, None)

    # -- Setup -----------------------------
    print_steps = flags.get('print_steps', 5)
    if beta is None:
        beta = flags.get('beta', flags.get('beta_final', None))  # type: float
        assert beta is not None

    test_step = dynamics.test_step
    if flags.get('compile', True):
        test_step = tf.function(dynamics.test_step)
        io.log('Compiled `dynamics.test_step` using tf.function!')

    if x is None:
        x = tf.random.uniform(shape=dynamics.x_shape,
                              minval=-PI, maxval=PI,
                              dtype=TF_FLOAT)

    run_steps = flags.get('run_steps', 20000)
    run_data = DataContainer(run_steps)

    template = '\n'.join([f'beta={beta}',
                          f'net_weights={dynamics.net_weights}'])
    logger.info(f'Running inference with {template}')

    # Run `md_steps MD updates (w/o accept/reject)
    # to ensure chains don't get stuck
    if md_steps > 0:
        for _ in range(md_steps):
            mc_states, _ = dynamics.md_update((x, beta), training=False)
            x = mc_states.out.x

    try:
        x, metrics = test_step((x, tf.constant(beta)))
    except Exception as err:  # pylint:disable=broad-except
        logger.warning(err)
        #  io.log(f'Exception: {exception}')
        test_step = dynamics.test_step
        x, metrics = test_step((x, tf.constant(beta)))

    x_arr = []

    def timed_step(x: tf.Tensor, beta: tf.Tensor):
        start = time.time()
        x, metrics = test_step((x, tf.constant(beta)))
        metrics.dt = time.time() - start
        if 'sin_charges' not in metrics:
            charges = dynamics.lattice.calc_both_charges(x=x)
            metrics['charges'] = charges.intQ
            metrics['sin_charges'] = charges.sinQ
        if save_x:
            x_arr.append(x.numpy())

        return x, metrics

    summary_steps = max(run_steps // 100, 50)

    steps = tf.range(run_steps, dtype=tf.int64)
    keep_ = ['step', 'dt', 'loss', 'accept_prob', 'beta',
             'dq_int', 'dq_sin', 'dQint', 'dQsin', 'plaqs', 'p4x4']

    beta = tf.constant(beta, dtype=TF_FLOAT)  # type: tf.Tensor
    data_strs = []
    for idx, step in enumerate(steps):
        x, metrics = timed_step(x, beta)
        run_data.update(step, metrics)  # update data after every accept/reject

        if step % summary_steps == 0:
            summarize_dict(metrics, step, prefix='testing')

        if step % print_steps == 0:
            pre = [f'{step}/{steps[-1]}']
            ms = run_data.print_metrics(metrics, window=50,
                                        pre=pre, keep=keep_)
            #  ms = logger.print_metrics(metrics, window=50, pre=pre, keep=keep_)
            data_strs.append(ms)

    return InferenceResults(dynamics=dynamics, x=x, x_arr=x_arr,
                            run_data=run_data, data_strs=data_strs)
