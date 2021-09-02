# noqa: F401
# pylint: disable=unused-import,invalid-name
# pylint: disable=no-member,too-many-locals,protected-access
"""
training_utils.py

Implements helper functions for training the model.
"""
from __future__ import absolute_import, annotations, division, print_function

import os
import json
import time
import h5py
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.summary_ops_v2 import SummaryWriter

import utils.file_io as io
import utils.live_plots as plotter
from config import PI
from dynamics.config import NET_WEIGHTS_HMC
from dynamics.gauge_dynamics import GaugeDynamics, build_dynamics
from network.config import LearningRateConfig
from utils.annealing_schedules import get_betas
from utils.attr_dict import AttrDict
from utils.data_containers import DataContainer, StepTimer
from utils.hvd_init import IS_CHIEF, LOCAL_RANK, RANK, SIZE
from utils.learning_rate import ReduceLROnPlateau
from utils.logger import Logger, in_notebook
from utils.plotting_utils import plot_data
from utils.summary_utils import update_summaries

if tf.__version__.startswith('1.'):
    TF_VERSION = 1
elif tf.__version__.startswith('2.'):
    TF_VERSION = 2

PLOT_STEPS = 10

TO_KEEP = [
    'H', 'Hf', 'plaqs', 'actions', 'charges', 'sin_charges', 'dqint', 'dqsin',
    'accept_prob', 'accept_mask', 'xeps', 'veps', 'sumlogdet', 'beta', 'loss',
    'dt',
]

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

PlotData = plotter.LivePlotData

logger = Logger()

OPTIONS = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=2,
    python_tracer_level=1,
    device_tracer_level=1,
    delay_ms=None,
)

def update_plots(
        history: dict,
        plots: dict,
        window: int = 1,
        logging_steps: int = 1
):
    lpdata = PlotData(history['loss'], plots['loss']['plot_obj1'])
    bpdata = PlotData(history['beta'], plots['loss']['plot_obj2'])
    fig_loss = plots['loss']['fig']
    id_loss = plots['loss']['display_id']
    plotter.update_joint_plots(lpdata, bpdata,
                               fig=fig_loss,
                               display_id=id_loss,
                               logging_steps=logging_steps)

    for key, val in history.items():
        if key in plots and key != 'loss':
            plotter.update_plot(y=val, window=window,
                                logging_steps=logging_steps, **plots[key])


def check_if_int(x: tf.Tensor) -> tf.Tensor:
    nearest_int = tf.math.round(x)
    return tf.math.abs(x - nearest_int) < 1e-3


def train_hmc(
        configs: AttrDict,
        make_plots: bool = True,
        num_chains: int = 32,
        #  therm_frac: float = 0.33,
):
    """Main method for training HMC model."""
    hconfigs = AttrDict(dict(configs).copy())
    lr_config = AttrDict(hconfigs.pop('lr_config', None))
    config = AttrDict(hconfigs.pop('dynamics_config', None))
    net_config = AttrDict(hconfigs.pop('network_config', None))
    hconfigs.train_steps = hconfigs.pop('hmc_steps', None)
    hconfigs.beta_init = hconfigs.beta_final

    config.update({
        'hmc': True,
        'use_ncp': False,
        'aux_weight': 0.,
        'zero_init': False,
        'separate_networks': False,
        'use_conv_net': False,
        'directional_updates': False,
        'use_scattered_xnet_update': False,
        'use_tempered_traj': False,
        'gauge_eq_masks': False,
    })

    lr_config = LearningRateConfig(
        warmup_steps=0,
        decay_rate=0.9,
        decay_steps=hconfigs.train_steps // 10,
        lr_init=lr_config.get('lr_init', None),
    )
    dirs = io.setup_directories(hconfigs, 'training_hmc')
    hconfigs.update({
        'profiler': False,
        'make_summaries': True,
        'lr_config': lr_config,
        'dirs': dirs,
    })

    dynamics = GaugeDynamics(hconfigs, config, net_config, lr_config)
    dynamics.save_config(dirs['config_dir'])

    x, train_data = train_dynamics(dynamics, hconfigs, dirs=dirs)
    if IS_CHIEF and make_plots:
        output_dir = os.path.join(dirs['train_dir'], 'outputs')
        io.check_else_make_dir(output_dir)
        train_data.save_data(output_dir)
        datafile = Path(output_dir).joinpath(f'data_rank{RANK}.hdf5')
        train_data.restore_data(datafile)

        params = {
            'eps': dynamics.eps.numpy(),
            'num_steps': dynamics.config.num_steps,
            'beta_init': train_data.data.beta[0],
            'beta_final': train_data.data.beta[-1],
            'x_shape': dynamics.config.x_shape,
            'net_weights': NET_WEIGHTS_HMC,
        }
        _ = plot_data(data_container=train_data, flags=hconfigs,
                      params=params, out_dir=dirs['train_dir'],
                      therm_frac=0.0, num_chains=num_chains)
        #  data_container = output['data_container']

    return x, dynamics, train_data, hconfigs


def random_init_from_configs(configs: dict[str, Any]) -> tf.Tensor:
    xshape = configs.get('dynamics_config', {}).get('x_shape', None)
    assert xshape is not None
    return tf.random.uniform(xshape, -PI, PI)


def load_last_training_point(logdir: Union[str, Path]) -> tf.Tensor:
    """Load previous states from `logdir`."""
    xfpath = os.path.join(logdir, 'training', 'train_data',
                          f'x_rank{RANK}-{LOCAL_RANK}.z')
    return io.loadz(xfpath)


def get_starting_point(configs: dict[str, Any]) -> tf.Tensor:
    logdir = configs.get('log_dir', configs.get('logdir', None))
    if logdir is None:
        return random_init_from_configs(configs)
    try:
        return load_last_training_point(
            configs.get('log_dir', configs.get('logdir', None))
        )
    except FileNotFoundError:
        return random_init_from_configs(configs)


def plot_models(dynamics: GaugeDynamics, logdir: Union[str, Path]):
    if dynamics.config.separate_networks:
        networks = {
            'dynamics_vnet': dynamics.vnet[0],
            'dynamics_xnet0': dynamics.xnet[0][0],
            'dynamics_xnet1': dynamics.xnet[0][1],
        }
    else:
        networks = {
            'dynamics_vnet': dynamics.vnet[0],
            'dynamics_xnet': dynamics.xnet[0],
        }

    for key, val in networks.items():
        try:
            fpath = os.path.join(logdir, f'{key}.png')
            tf.keras.utils.plot_model(val, show_shapes=True, to_file=fpath)
        except Exception as exception:
            raise exception


def load_configs_from_logdir(logdir: Union[str, Path]) -> dict[str, Any]:
    try:
        configs_file = os.path.join(logdir, 'train_configs.z')
        configs = io.loadz(configs_file)
    except (EOFError, KeyError, ValueError):
        configs_file = os.path.join(logdir, 'train_configs.json')
        with open(configs_file, 'r') as f:
            configs = json.load(f)

    return configs


def find_conflicts(
        new: dict,
        old: dict,
        name: str = None,
        skips: list[str] = None,
) -> list[str]:
    conflicts = []
    for key in new.keys():
        if skips is not None:
            if key in skips:
                continue

        old_ = old.get(key, None)
        new_ = new.get(key, None)
        if new_ != old_:
            nstr = '' if name is None else name
            logger.error(' '.join([
                'Mismatch encountered',
                f'in {name}\n' if name is not None else '\n',
                '\n'.join([
                    f'{nstr} old[{key}]: {old_}',
                    f'{nstr} new[{key}]: {new_}'
                ]),
            ]))
            conflicts.append(key)

    return conflicts


def check_compatibility(new: dict, old: dict, strict: bool = False) -> dict:
    use_conv_net_old = old['dynamics_config']['use_conv_net']  # type: bool
    use_conv_net_new = new['dynamics_config']['use_conv_net']  # type: bool

    lf_old = old['dynamics_config']['num_steps']  # type: int
    lf_new = new['dynamics_config']['num_steps']  # type: int

    bi_old = old['beta_init']  # type: float
    bi_new = new['beta_init']  # type: float

    bf_old = old['beta_final']  # type: float
    bf_new = new['beta_final']  # type: float

    assert lf_old == lf_new
    assert use_conv_net_old == use_conv_net_new

    bi_conflict = (bi_new != bi_old)
    bf_conflict = (bf_new != bf_old)
    if bi_conflict or bf_conflict:
        new['ensure_new'] = True

    names = ['dynamics_config', 'network_config']
    if new['dynamics_config']['use_conv_net']:
        names.append('conv_config')

    # -- example ----------------------------------------------
    # conflicts = {
    #    'dynamics_config': ['num_steps', 'use_conv_net', ...],
    #    ...,
    # }
    # would imply there was a conflict found in:
    #   - configs['dynamics_config']['num_steps']
    #   - configs['dynamics_config']['use_conv_net']
    skips = ['x_shape', 'input_shape']
    conflicts = {
        name: find_conflicts(new[name], old[name], name=name, skips=skips)
        for name in names
    }
    for name, conflict in conflicts.items():
        if len(conflict) == 0:
            continue

        if strict:
            raise AssertionError('\n'.join([
                'Incompatible configs.', f'{name}: {conflict}',
            ]))

        for c in conflict:
            old_ = old[name][c]
            new_ = new[name][c]
            logger.warning(f'Overwriting {name}[{c}] with restored val')
            logger.warning(f'{name}[{c}]={new_} --> {old_}')
            new[name][c] = old_

    return new

def look_for_previous_logdir(logdir: Union[str, Path]):
    logdir = Path(logdir)
    parent = logdir.parent
    candidates = [
        i for i in sorted(parent.glob('*/'), key=os.path.getmtime)
        if i.is_dir()
    ]
    prev = logdir
    if len(candidates) > 0:
        prev = candidates[-1]
        if len(candidates) > 1:
            prev = candidates[-2]

    return prev


def restore_from(
        configs: dict,
        logdir: Union[str, Path],
        strict: bool = False
) -> GaugeDynamics:
    """Load trained networks and restore model from checkpoint."""

    restored = None
    if logdir is not None:
        restored = load_configs_from_logdir(logdir)
        configs = check_compatibility(configs, restored, strict=strict)
        configs['restored'] = True
        configs['restored_configs'] = restored
        configs['profiler'] = False

    dynamics = build_dynamics(configs)

    networks = dynamics._load_networks(str(logdir))
    dynamics.xnet = networks['xnet']
    dynamics.vnet = networks['vnet']
    logger.info(f'Networks successfully loaded from: {logdir}')

    ckptdir = os.path.join(logdir, 'training', 'checkpoints')
    ckpt = tf.train.Checkpoint(dynamics=dynamics, optimizer=dynamics.optimizer)
    manager = tf.train.CheckpointManager(ckpt, ckptdir, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info(f'Restored from {manager.latest_checkpoint}')

    return dynamics


def setup_directories(configs: dict) -> dict:
    """Setup directories for training."""
    logfile = os.path.join(os.getcwd(), 'log_dirs.txt')
    ensure_new = configs.get('ensure_new', False)
    logdir = configs.get('logdir', configs.get('log_dir', None))
    if logdir is not None:
        logdir_exists = os.path.isdir(logdir)
        contents = os.listdir(logdir)
        logdir_nonempty = False
        if contents is not None and isinstance(contents, list):
            if len(contents) > 0:
                logdir_nonempty = True

        if logdir_exists and logdir_nonempty and ensure_new:
            raise ValueError(
                f'Nonempty `logdir`, but `ensure_new={ensure_new}'
            )

    # Create `logdir`, `logdir/training/...`' etc
    dirs = io.setup_directories(configs, timestamps=TSTAMPS)
    configs['dirs'] = dirs
    logdir = dirs.get('logdir', dirs.get('log_dir', None))  # type: str
    configs['log_dir'] = logdir
    configs['logdir'] = logdir

    restore_dir = configs.get('restore_from', None)
    if restore_dir is None:
        candidate = look_for_previous_logdir(logdir)
        if candidate != logdir:
            if candidate.is_dir():
                nckpts = len(list(candidate.rglob('checkpoint')))
                if nckpts > 0:
                    restore_dir = candidate
                    configs['restore_from'] = restore_dir

    if restore_dir is not None:
        try:
            restored = load_configs_from_logdir(restore_dir)
            if restored is not None:
                io.save_dict(restored, logdir,
                             name='restored_train_configs')
        except FileNotFoundError:
            logger.warning(f'Unable to load configs from {restore_dir}')

    if RANK == 0:
        io.check_else_make_dir(logdir)
        restore_dir = configs.get('restore_dir', None)
        if restore_dir is not None:
            try:
                restored = load_configs_from_logdir(restore_dir)
                if restored is not None:
                    io.save_dict(restored, logdir,
                                 name='restored_train_configs')
            except FileNotFoundError:
                logger.warning(f'Unable to load configs from {restore_dir}')
                pass

    return configs


def setup_betas(
        bi: float,
        bf: float,
        train_steps: int,
        current_step: int,
):
    """Setup array of betas for training."""

    if bi == bf:
        betas = bi * tf.ones(train_steps)
    else:
        betas = get_betas(train_steps, bi, bf)

    if current_step > 0:
        betas = betas[current_step:]

    return tf.convert_to_tensor(betas)


# TODO: Add type annotations
# pylint:disable=too-many-statements, too-many-branches
def setup(
        configs: dict,
        x: tf.Tensor = None,
        betas: list[tf.Tensor]=None,
        strict: bool = False,
        try_restore: bool = False,
):
    """Setup training."""
    train_steps = configs.get('train_steps', None)  # type: int
    save_steps = configs.get('save_steps', None)    # type: int
    print_steps = configs.get('print_steps', None)  # type: int

    beta_init = configs.get('beta_init', None)      # type: float
    beta_final = configs.get('beta_final', None)    # type: float

    dirs = configs.get('dirs', None)  # type: dict[str, Any]
    logdir = dirs.get('logdir', dirs.get('log_dir', None))

    assert dirs is not None
    assert logdir is not None
    assert beta_init is not None and beta_final is not None

    train_data = DataContainer(train_steps, dirs=dirs, print_steps=print_steps)

    # Check if we want to restore from existing directory
    restore_dir = configs.get('restore_from', None)
    if restore_dir is not None:
        dynamics = restore_from(configs, restore_dir, strict=strict)
        datadir = os.path.join(restore_dir, 'training', 'train_data')
    else:
        prev_logdir = look_for_previous_logdir(logdir)
        datadir = os.path.join(logdir, 'training', 'train_data')
        try:
            dynamics = restore_from(configs, prev_logdir, strict=strict)
        except OSError:
            logger.error(f'Unable to restore dynamics from previous logdir!')
            try:
                dynamics = restore_from(configs, logdir, strict=strict)
            except OSError:
                logger.error(f'Unable to restore dynamics! Creating fresh...')
                dynamics = build_dynamics(configs)

    current_step = dynamics.optimizer.iterations.numpy()
    if train_steps <= current_step:
        train_steps = current_step + min(save_steps, print_steps)

    train_data.steps = train_steps

    if os.path.isdir(datadir) and try_restore:
        try:
            x = train_data.restore(datadir, step=current_step,
                                   x_shape=dynamics.x_shape,
                                   rank=RANK, local_rank=LOCAL_RANK)
        except ValueError:
            logger.warning('Unable to restore `x`, re-sampling from [-pi,pi)')
            x = tf.random.uniform(dynamics.x_shape, minval=-PI, maxval=PI)
    else:
        x = tf.random.uniform(dynamics.x_shape, minval=-PI, maxval=PI)


    # Reshape x from [batch_size, Nt, Nx, Nd] --> [batch_size, Nt * Nx * Nd]
    x = tf.reshape(x, (x.shape[0], -1))

    # Create checkpoint and checkpoint manager for saving during training
    ckptdir = os.path.join(logdir, 'training', 'checkpoints')
    ckpt = tf.train.Checkpoint(dynamics=dynamics, optimizer=dynamics.optimizer)
    manager = tf.train.CheckpointManager(ckpt, ckptdir, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info(f'Restored ckpt from: {manager.latest_checkpoint}')

    # Setup summary writer for logging metrics through tensorboard
    summdir = dirs['summary_dir']
    make_summaries = configs.get('make_summaries', True)
    steps = tf.range(current_step, train_steps, dtype=tf.int64)
    betas = setup_betas(beta_init, beta_final, train_steps, current_step)

    dynamics.compile(loss=dynamics.calc_losses,
                     optimizer=dynamics.optimizer,
                     experimental_run_tf_function=False)

    _ = dynamics.apply_transition((x, tf.constant(betas[0])), training=True)

    writer = None
    if IS_CHIEF:
        plot_models(dynamics, dirs['log_dir'])
        io.savez(configs, os.path.join(dirs['log_dir'], 'train_configs.z'))
        if make_summaries and TF_VERSION == 2:
            try:
                writer = tf.summary.create_file_writer(summdir)
            except AttributeError:
                writer = None
        else:
            writer = None

    return {
        'x': x,
        'betas': betas,
        'dynamics': dynamics,
        'dirs': dirs,
        'steps': steps,
        'writer': writer,
        'manager': manager,
        'configs': configs,
        'checkpoint': ckpt,
        'train_data': train_data,
    }


@dataclass
class TrainOutputs:
    x: tf.Tensor
    logdir: str
    configs: dict[str, Any]
    data: DataContainer
    dynamics: GaugeDynamics


def train(
        configs: dict[str, Any],
        x: tf.Tensor = None,
        num_chains: int = 32,
        make_plots: bool = True,
        steps_dict: dict[str, int] = None,
        save_metrics: bool = True,
        save_dataset: bool = False,
        custom_betas: Union[list, np.ndarray] = None,
        **kwargs,
) -> TrainOutputs:
    """Train model.

    Returns:
        train_outputs: Dataclass with attributes:
          - x: tf.Tensor
          - logdir: str
          - configs: dict[str, Any]
          - data: DataContainer
          - dynamics: GaugeDynamics
    """
    start = time.time()
    configs = setup_directories(configs)
    try_restore = kwargs.pop('try_restore', True)
    config = setup(configs, x=x, try_restore=try_restore)
    dynamics = config['dynamics']
    dirs = config['dirs']
    configs = config['configs']
    train_data = config['train_data']

    dynamics.save_config(dirs['config_dir'])
    if RANK == 0:
        logfile = os.path.join(os.getcwd(), 'log_dirs.txt')
        logdir = configs.get('logdir', configs.get('log_dir', None))  # str
        io.save_dict(configs, logdir, name='train_configs')
        io.write(f'{logdir}', logfile, 'a')

        restore_from = configs.get('restore_from', None)
        if restore_from is not None:
            restored = load_configs_from_logdir(restore_from)
            if restored is not None:
                io.save_dict(restored, logdir, name='restored_train_configs')

    # -- Train dynamics -----------------------------------------
    #  logger.rule('TRAINING')
    logger.info(f'Starting training at: {io.get_timestamp("%x %X")}')
    t0 = time.time()
    x, train_data = train_dynamics(dynamics, config, dirs,
                                   x=x, steps_dict=steps_dict,
                                   custom_betas=custom_betas,
                                   save_metrics=save_metrics)
    logger.rule(f'DONE TRAINING. TOOK: {time.time() - t0:.4f}')
    logger.info(f'Training took: {time.time() - t0:.4f}')
    # ------------------------------------

    if IS_CHIEF:
        logdir = dirs['logdir']
        train_dir = dirs['train_dir']
        io.save_dict(dict(configs), dirs['log_dir'], 'configs')
        logger.info(f'Done training model! took: {time.time() - start:.4f}s')
        train_data.save_and_flush(dirs['data_dir'])

        #  outdir = Path(train_dir)
        #  hfile = Path(outdir).joinpath(f'data_rank{RANK}.hdf5')
        #
        #  outdir.mkdir(parents=True, exist_ok=True)
        #  train_data.save_and_flush(outdir)

        #  f = h5py.File(hfile, 'r')
        #  train_data.data = AttrDict({key: f[key] for key in list(f.keys())})

        if make_plots:
            params = {
                'beta_init': train_data.data.beta[0],
                'beta_final': train_data.data.beta[-1],
                'x_shape': dynamics.config.x_shape,
                'num_steps': dynamics.config.num_steps,
                'net_weights': dynamics.net_weights,
            }
            t0 = time.time()
            logging_steps = configs.get('logging_steps', None)  # type: int
            _ = plot_data(data_container=train_data,
                          configs=configs,
                          params=params,
                          out_dir=train_dir,
                          therm_frac=0,
                          cmap='flare',
                          num_chains=num_chains,
                          logging_steps=logging_steps, **kwargs)
            dt = time.time() - t0
            logger.debug(
                f'Time spent plotting: {dt}s = {dt // 60}m {(dt % 60):.4f}s'
            )
        #
        #  if save_metrics:
        #      output_dir = os.path.join(train_dir, 'outputs')
        #      train_data.save_data(output_dir, save_dataset=save_dataset)
        #  f.close()

        if not dynamics.config.hmc:
            dynamics.save_networks(logdir)

    return TrainOutputs(x, dirs['log_dir'], configs, train_data, dynamics)


def run_md(
        dynamics: GaugeDynamics,
        inputs: tuple,
        md_steps: int,
):
    x, beta = inputs
    beta = tf.constant(beta)
    logger.debug(f'Running {md_steps} MD updates!')
    for _ in range(md_steps):
        mc_states, _ = dynamics.md_update((x, beta), training=True)
        x = mc_states.out.x  # type: tf.Tensor
    logger.debug(f'Done!')

    return x


def trace_train_step(
        dynamics: GaugeDynamics,
        writer: SummaryWriter,
        outdir: str,
        x: tf.Tensor = None,
        beta: float = None,
        graph: bool = True,
        profiler: bool = True,
):
    if x is None:
        x = tf.random.uniform(dynamics.x_shape, *(-PI, PI))
    if beta is None:
        beta = 1.

    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export()
    tf.summary.trace_on(graph=graph, profiler=profiler)
    # Call only one tf.function when tracing
    x, metrics = dynamics.train_step((x, beta))
    with writer.as_default():
        tf.summary.trace_export(name='dynamics_train_step', step=0,
                                profiler_outdir=outdir)
    #  for step in range(3):
    #      with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
    #          tf.profiler.experimental.start(logdir=outdir)
    #          x, metrics = dynamics.train_step((x, beta))



def run_profiler(
        dynamics: GaugeDynamics,
        inputs: tuple[tf.Tensor, Union[float, tf.Tensor]],
        logdir: str,
        steps: int = 10
):
    logger.debug(f'Running {steps} profiling steps!')
    x, beta = inputs
    beta = tf.constant(beta)
    metrics = None
    for _ in range(steps):
        x, metrics = dynamics.train_step((x, beta))

    tf.profiler.experimental.start(logdir=logdir, options=OPTIONS)
    for step in range(steps):
        with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            x, metrics = dynamics.train_step((x, beta))

    tf.profiler.experimental.stop(save=True)
    logger.debug(f'Done!')

    return x, metrics


# pylint: disable=broad-except
# pylint: disable=too-many-arguments,too-many-statements, too-many-branches
def train_dynamics(
        dynamics: GaugeDynamics,
        inputs: dict[str, Any],
        dirs: dict[str, str] = None,
        x: tf.Tensor = None,
        steps_dict: dict[str, int] = None,
        save_metrics: bool = True,
        custom_betas: Union[list, np.ndarray] = None,
) -> tuple[tf.Tensor, DataContainer]:
    """Train model."""
    configs = inputs['configs']
    steps = configs.get('steps', [])
    min_lr = configs.get('min_lr', 1e-5)
    patience = configs.get('patience', 10)
    factor = configs.get('reduce_lr_factor', 0.5)

    save_steps = configs.get('save_steps', 10000)           # type: int
    print_steps = configs.get('print_steps', 1000)          # type: int
    logging_steps = configs.get('logging_steps', 500)       # type: int
    steps_per_epoch = configs.get('steps_per_epoch', 1000)  # type: int
    if steps_dict is not None:
        save_steps = steps_dict.get('save', 10000)          # type: int
        print_steps = steps_dict.get('print', 1000)         # type: int
        logging_steps = steps_dict.get('logging_steps', 500)       # type: int
        steps_per_epoch = steps_dict.get('steps_per_epoch', 1000)  # type: int


    # -- Helper functions for training, logging, saving, etc. --------------
    #  step_times = []
    timer = StepTimer(evals_per_step=dynamics.config.num_steps)

    def train_step(x: tf.Tensor, beta: tf.Tensor):
        #  start = time.time()
        timer.start()
        x, metrics = dynamics.train_step((x, tf.constant(beta)))
        dt = timer.stop()
        metrics.dt = dt
        return x, metrics

    def should_print(step: int) -> bool:
        return IS_CHIEF and step % print_steps == 0

    def should_log(step: int) -> bool:
        return IS_CHIEF and step % logging_steps == 0

    def should_save(step: int) -> bool:
        return step % save_steps == 0 and ckpt is not None

    xshape = dynamics._xshape
    xr = tf.random.uniform(xshape, -PI, PI)
    x = inputs.get('x', xr) if x is None else x
    assert x is not None

    if custom_betas is None:
        betas = np.array(inputs.get('betas', None))
        assert betas is not None and betas.shape[0] > 0

        steps = np.array(inputs.get('steps'))
        assert steps is not None and steps.shape[0] > 0
    else:
        betas = np.array(custom_betas)
        start = dynamics.optimizer.iterations
        nsteps = len(betas)
        steps = np.arange(start, start + nsteps)

    dirs = inputs.get('dirs', None) if dirs is None else dirs  # type: dict
    assert dirs is not None

    manager = inputs['manager']  # type: tf.train.CheckpointManager
    ckpt = inputs['checkpoint']  # type: tf.train.Checkpoint
    train_data = inputs['train_data']  # type: DataContainer

    #  tf.compat.v1.autograph.experimental.do_not_convert(dynamics.train_step)

    # -- Setup dynamic learning rate schedule -----------------
    assert dynamics.lr_config is not None
    warmup_steps = dynamics.lr_config.warmup_steps
    reduce_lr = ReduceLROnPlateau(monitor='loss', mode='min',
                                  warmup_steps=warmup_steps,
                                  factor=factor, min_lr=min_lr,
                                  verbose=1, patience=patience)
    reduce_lr.set_model(dynamics)

    # -- Setup summary writer -----------
    writer = inputs.get('writer', None)  # type: tf.summary.SummaryWriter
    if IS_CHIEF and writer is not None:
        writer.set_as_default()

    # -- Run profiler? ----------------------------------------
    if configs.get('profiler', False):
        if RANK == 0:
            sdir = dirs['summary_dir']
            #  trace_train_step(dynamics,
            #                   graph=True,
            #                   profiler=True,
            #                   outdir=sdir,
            #                   writer=writer)
            x, metrics = run_profiler(dynamics, (x, betas[0]),
                                      logdir=sdir, steps=5)
    else:
        x, metrics = dynamics.train_step((x, betas[0]))

    # -- Run MD update to not get stuck ----------------------
    md_steps = configs.get('md_steps', 0)
    if md_steps > 0:
        x = run_md(dynamics, (x, betas[0]), md_steps)

    warmup_steps = dynamics.lr_config.warmup_steps
    total_steps = steps[-1]
    if len(steps) != len(betas):
        betas = betas[steps[0]:]

    #  keep = ['dt', 'loss', 'accept_prob', 'beta', 'Hwb_start', 'Hwf_start',
    #          'Hwb_mid', 'Hwf_mid', 'Hwb_end', 'Hwf_end', 'xeps', 'veps',
    #          'dq', 'dq_sin', 'plaqs', 'p4x4', 'charges', 'sin_charges']

    plots = {}
    if in_notebook():
        plots = plotter.init_plots(configs, figsize=(9, 3), dpi=125)

    # -- Training loop ---------------------------------------------------
    data_strs = []
    logdir = dirs['log_dir']
    data_dir = dirs['data_dir']
    logfile = dirs['log_file']
    logfile = os.path.join(logdir, 'training', 'train_log.txt')

    assert x is not None
    assert manager is not None
    assert len(steps) == len(betas)
    for step, beta in zip(steps, betas):
        x, metrics = train_step(x, beta)

        # ----------------------------------------------------------------
        # TODO: Run inference when beta hits an integer
        # >>> beta_inf = {i: False, for i in np.arange(beta_final)}
        # >>> if any(np.isclose(beta, np.array(list(beta_inf.keys())))):
        # >>>     run_inference(...)
        # ----------------------------------------------------------------

        if (step + 1) > warmup_steps and (step + 1) % steps_per_epoch == 0:
            reduce_lr.on_epoch_end(step+1, {'loss': metrics.loss})

        # -- Save checkpoints and dump configs `x` from each rank --------
        if should_save(step + 1):
            train_data.update(step, metrics)
            train_data.dump_configs(x, data_dir, rank=RANK,
                                    local_rank=LOCAL_RANK)
            if IS_CHIEF:
                _ = timer.save_and_write(logdir, mode='w')
                # -- Save CheckpointManager ------------------------------
                manager.save()
                mstr = f'Checkpoint saved to: {manager.latest_checkpoint}'
                logger.info(mstr)
                with open(logfile, 'w') as f:
                    f.writelines('\n'.join(data_strs))

                # -- Save train_data and free consumed memory ------------
                train_data.save_and_flush(data_dir, logfile,
                                          rank=RANK, mode='a')
                if not dynamics.config.hmc:
                    # -- Save network weights ----------------------------
                    dynamics.save_networks(logdir)
                    logger.info(f'Networks saved to: {logdir}')

        # -- Print current training state and metrics -------------------
        if should_print(step):
            train_data.update(step, metrics)
            keep_ = ['step', 'dt', 'loss', 'accept_prob', 'beta',
                     'dq_int', 'dq_sin', 'dQint', 'dQsin', 'plaqs', 'p4x4']
            pre = [f'{step:>4g}/{total_steps:<4g}']
            #  data_str = logger.print_metrics(metrics, window=50,
            #                                  pre=pre, keep=keep_)
            data_str = train_data.print_metrics(metrics, window=50,
                                                pre=pre, keep=keep_)
            data_strs.append(data_str)

        if in_notebook() and step % PLOT_STEPS == 0 and IS_CHIEF:
            train_data.update(step, metrics)
            if len(train_data.data.keys()) == 0:
                update_plots(metrics, plots,
                             logging_steps=configs['logging_steps'])
            else:
                update_plots(train_data.data, plots,
                             logging_steps=configs['logging_steps'])

        # -- Update summary objects ---------------------
        if should_log(step):
            train_data.update(step, metrics)
            if writer is not None:
                update_summaries(step, metrics, dynamics)
                writer.flush()

    # -- Dump config objects -------------------------------------------------
    train_data.dump_configs(x, data_dir, rank=RANK, local_rank=LOCAL_RANK)
    if IS_CHIEF:
        manager.save()
        logger.log(f'Checkpoint saved to: {manager.latest_checkpoint}')

        with open(logfile, 'w') as f:
            f.writelines('\n'.join(data_strs))

        if save_metrics:
            train_data.save_and_flush(data_dir, logfile, rank=RANK, mode='a')

        if not dynamics.config.hmc:
            dynamics.save_networks(logdir)

        if writer is not None:
            writer.flush()
            writer.close()

        #  ngrad_evals =  SIZE * dynamics.config.num_steps * len(step_times)
        #  eval_rate = ngrad_evals / np.sum(step_times)
        #  outstr = '\n'.join([f'ngrad_evals: {ngrad_evals}',
        #                      f'sum(step_times): {np.sum(step_times)}',
        #                      f'eval rate: {eval_rate}'])
        #  with open(Path(logdir).joinpath('eval_rate.txt'), 'a') as f:
        #      f.write(outstr)
        #
        #  csvfile = Path(logdir).joinpath('dt_train.csv')
        #  pd.DataFrame(step_times).to_csv(csvfile, mode='a')


    return x, train_data
