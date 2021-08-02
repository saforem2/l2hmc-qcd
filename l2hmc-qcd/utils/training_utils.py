# noqa: F401
# pylint: disable=unused-import,invalid-name
# pylint: disable=no-member,too-many-locals,protected-access
"""
training_utils.py

Implements helper functions for training the model.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass

import os
import time
import json
from typing import Any, Optional

import numpy as np
import tensorflow as tf

import utils.file_io as io
from config import PI
from dynamics.config import NET_WEIGHTS_HMC
from dynamics.gauge_dynamics import GaugeDynamics, build_dynamics
from network.config import LearningRateConfig
from utils.annealing_schedules import get_betas
from utils.attr_dict import AttrDict
from utils.data_containers import DataContainer
from utils.learning_rate import ReduceLROnPlateau
import utils.live_plots as plotter
#  import utils.live_plots as plotter
#  from utils.live_plots import (LivePlotData, init_plots, update_joint_plots,
from utils.logger import Logger, in_notebook
#  from utils.logger import Logger, in_notebook
from utils.plotting_utils import plot_data
from utils.summary_utils import update_summaries
from utils.hvd_init import RANK, LOCAL_RANK, IS_CHIEF
#  from utils.logger_config import log
#  from utils.summary_utils import update_summaries
#  try:
#      import horovod
#      import horovod.tensorflow as hvd  # pylint:disable=wrong-import-order
#      try:
#          RANK = hvd.rank()
#      except ValueError:
#          hvd.init()
#
#      RANK = hvd.rank()
#      LOCAL_RANK = hvd.local_rank()
#      IS_CHIEF = (RANK == 0)
#      HAS_HOROVOD = True
#      NUM_WORKERS = hvd.size()
#      #  hvd.init()
#      GPUS = tf.config.experimental.list_physical_devices('GPU')
#      for gpu in GPUS:
#          tf.config.experimental.set_memory_growth(gpu, True)
#      if GPUS:
#          gpu = GPUS[hvd.local_rank()]  # pylint:disable=invalid-name
#          tf.config.experimental.set_visible_devices(gpu, 'GPU')
#
#  except (ImportError, ModuleNotFoundError):
#      HAS_HOROVOD = False
#      RANK = LOCAL_RANK = 0
#      SIZE = LOCAL_SIZE = 1
#      IS_CHIEF = (RANK == 0)
#
#  #      IS_CHIEF = (RANK == 0)
#  #
#


if tf.__version__.startswith('1.'):
    TF_VERSION = 1
elif tf.__version__.startswith('2.'):
    TF_VERSION = 2

#SHOULD_TRACK = os.environ.get('TRACK', True)
#  SHOULD_TRACK = not os.environ.get('NOTRACK', False)
SHOULD_TRACK = False
PLOT_STEPS = 10

TO_KEEP = [
    'H', 'Hf', 'plaqs', 'actions', 'charges', 'sin_charges', 'dqint', 'dqsin',
    'accept_prob', 'accept_mask', 'xeps', 'veps', 'sumlogdet', 'beta', 'loss',
    'dt',
]

#  logger = io.Logger()
logger = Logger()

#  logger = io.Logger()

#  try:
#      tf.config.experimental.enable_mlir_bridge()
#      tf.config.experimental.enable_mlir_graph_optimization()
#  except:  # noqa: E722
#      pass
PlotData = plotter.LivePlotData

def update_plots(history: dict, plots: dict, window: int = 1):
    lpdata = PlotData(history['loss'], plots['loss']['plot_obj1'])
    bpdata = PlotData(history['beta'], plots['loss']['plot_obj2'])
    fig_loss = plots['loss']['fig']
    id_loss = plots['loss']['display_id']
    plotter.update_joint_plots(lpdata, bpdata, fig=fig_loss,
                               display_id=id_loss)

    for key, val in history.items():
        if key in plots and key != 'loss':
            plotter.update_plot(y=val, window=window, **plots[key])




def check_if_int(x):
    nearest_int = tf.math.round(x)
    return tf.math.abs(x - nearest_int) < 1e-3

def train_hmc(
        configs: AttrDict,
        make_plots: bool = True,
        therm_frac: float = 0.33,
        num_chains: int = 32,
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

        params = {
            'eps': dynamics.eps.numpy(),
            'num_steps': dynamics.config.num_steps,
            'beta_init': train_data.data.beta[0],
            'beta_final': train_data.data.beta[-1],
            'x_shape': dynamics.config.x_shape,
            'net_weights': NET_WEIGHTS_HMC,
        }
        t0 = time.time()
        output = plot_data(data_container=train_data, flags=hconfigs,
                           params=params, out_dir=dirs['train_dir'],
                           therm_frac=0.0, num_chains=num_chains)
        data_container = output['data_container']
        logger.rule(f'Time spent plotting: {dt}s = {dt // 60}m {(dt % 60):.3g}s')
        logger.rule(f'Time spent plotting: {dt}s = {dt // 60}m {(dt % 60):.3g}s')

    return x, dynamics, train_data, hconfigs


def random_init_from_configs(configs: dict[str, Any]) -> tf.Tensor:
    xshape = configs.get('dynamics_config', {}).get('x_shape', None)
    assert xshape is not None
    return tf.random.uniform(xshape, -PI, PI)


def load_last_training_point(logdir) -> tf.Tensor:
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


def plot_models(dynamics, logdir: str):
    if dynamics.config.separate_networks:
        networks = {
            'dynamics_vnet': dynamics.vnet[0],
            'dynamics_xnet0': dynamics.xnet[0][0],
            'dynamics_xnet1': dynamics.xnet[0][1],
        }
    else:
        networks = {
            'dynamics_vnet': dynamics.vnet,
            'dynamics_xnet': dynamics.xnet,
        }

    for key, val in networks.items():
        try:
            fpath = os.path.join(logdir, f'{key}.png')
            tf.keras.utils.plot_model(val, show_shapes=True, to_file=fpath)
        except Exception as exception:
            raise exception


# TODO: Add type annotations
# pylint:disable=too-many-statements, too-many-branches
def setup(configs, x=None, betas=None):
    """Setup training."""
    x = get_starting_point(configs)
    # Reshape x from (batch_size, Nt, Nx, 2) --> (batch_size, Nt * Nx * 2)
    x = tf.reshape(x, (x.shape[0], -1))

    logdir = configs.get('logdir', configs.get('log_dir', None))
    ensure_new = configs.get('ensure_new', False)
    #  if logdir is not None:
    #      if os.path.isdir(logdir) and ensure_new:
    #          raise ValueError('logdir exists but `ensure_new` flag is set.')

    #  dirs = io.setup_directories(configs)
    dynamics = build_dynamics(configs)

    dirs = configs.get('dirs', None)
    assert dirs is not None

    logdir = dirs.get('logdir', dirs.get('log_dir', None))
    models_dir = os.path.join(logdir, 'training', 'models')
    nets_exist = os.path.isdir(models_dir) and len(os.listdir(models_dir)) > 0
    if nets_exist and not ensure_new:
        logger.info(f'Loading networks from: {logdir}')
        networks = dynamics._load_networks(str(logdir))
        dynamics.xnet = networks['xnet']
        dynamics.vnet = networks['vnet']


    #  dirs = setup_directories(configs)
    #  configs['dirs'] = dirs
    #  configs['log_dir'] = dirs['log_dir']
    train_steps = configs.get('train_steps')
    print_steps = configs.get('print_steps')


    #  logger.info(f'dynamics.net_weights: {dynamics.net_weights}')
    #  network_dir = dynamics.config.log_dir
    #  if network_dir is not None:
    #      xnet, vnet = dynamics._load_networks(network_dir)
    #      dynamics.xnet = xnet
    #      dynamics.vnet = vnet
    train_data = DataContainer(train_steps, dirs=dirs,
                               print_steps=print_steps)
    #  if dynamics._has_trainable_params:
    ckpt = tf.train.Checkpoint(dynamics=dynamics,
                               optimizer=dynamics.optimizer)
    ckptdir = dirs['ckpt_dir']
    datadir = dirs['data_dir']
    summdir = dirs['summary_dir']
    manager = tf.train.CheckpointManager(ckpt, ckptdir, max_to_keep=5)
    if manager.latest_checkpoint:  # and not configs.get('ensure_new', False):
        logger.rule(f'Restoring model')
        logger.info(f'Restored model from: {manager.latest_checkpoint}')
        ckpt.restore(manager.latest_checkpoint)
        configs['restored'] = True
        configs['restored_from'] = manager.latest_checkpoint
        current_step = dynamics.optimizer.iterations.numpy()
        x = train_data.restore(datadir, step=current_step,
                               rank=RANK, local_rank=LOCAL_RANK,
                               x_shape=dynamics.x_shape)
        if current_step >= train_steps:
            logger.warning(', '.join(['Current step >= train_steps',
                                      f'current_step={current_step}',
                                      f'train_steps={train_steps}']))
            train_steps = current_step + 10
            logger.warning(f'Setting train_steps={train_steps}')
    else:
        configs['restored'] = False
        configs['restored_from'] = None
        logger.warning('Starting new training run')
        logger.rule('NEW TRAINING RUN')

    # Create initial samples if not restoring from ckpt
    if x is None:
        x = tf.random.uniform(shape=dynamics.x_shape, minval=-PI, maxval=PI)

    # Setup summary writer
    make_summaries = configs.get('make_summaries', True)
    if IS_CHIEF and make_summaries and TF_VERSION == 2:
        try:
            writer = tf.summary.create_file_writer(summdir)
        except AttributeError:  # pylint:disable=bare-except
            writer = None
    else:
        writer = None

    current_step = dynamics.optimizer.iterations.numpy()  # get global step
    if current_step > train_steps:
        train_steps = current_step + 10

    steps = tf.range(current_step, train_steps, dtype=tf.int64)
    train_data.steps = steps[-1]

    beta_init = configs.get('beta_init', None)
    beta_final = configs.get('beta_final', None)
    assert beta_init is not None and beta_final is not None
    if beta_init != beta_final:
        betas = get_betas(train_steps, beta_init, beta_final)
        betas = betas[current_step:]
    else:
        betas = beta_init * tf.ones(len(steps))

    #  if current_step > len(betas):
    #      diff = current_step - len(betas)
    #      betas = list(betas) + diff * [tf.constant(beta_final)]
    #      #  betas = list(betas) +  * tf.ones(diff)
    #      #  betas = list(betas) + [betas[-1] for _ in range(diff)]


    remaining_steps = train_steps - current_step
    if len(betas) < remaining_steps:
        diff = remaining_steps - len(betas)
        betas = list(betas) + diff * [tf.constant(beta_final)]


    betas = tf.convert_to_tensor(betas, dtype=x.dtype)
    dynamics.compile(loss=dynamics.calc_losses,
                     optimizer=dynamics.optimizer,
                     experimental_run_tf_function=False)

    _ = dynamics.apply_transition((x, tf.constant(betas[0])), training=True)

    if IS_CHIEF:
        plot_models(dynamics, dirs['log_dir'])
        io.savez(configs, os.path.join(dirs['log_dir'], 'train_configs.z'))
        #  cfgs_fpath = os.path.join(dirs['log_dir'], 'train_configs.json')
        #  cfgs = {str(k): v for k, v in configs.items()}
        #  logger.info(cfgs)
        #  with open(cfgs_fpath, 'w') as f:
        #      json.dump(cfgs, f)


    #  prof_range = (0, 0)
    # If profiling, run for 10 steps in the middle of training
    #  if configs.get('profiler', False):
    #      pstart = len(betas) // 2
    #      prof_range = (pstart, pstart + 10)

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
    configs: dict
    data: DataContainer
    dynamics: GaugeDynamics


def train(
        configs: dict[str, Any],
        x: tf.Tensor = None,
        num_chains: int = 32,
        restore_x: bool = False,
        make_plots: bool = True,
        therm_frac: float = 0.33,
        ensure_new: bool = False,
        #  should_track: bool = True,
) -> TrainOutputs:
    """Train model.

    Returns:
        x (tf.Tensor): Batch of configurations
        dynamics (GaugeDynamics): Dynamics object.
        train_data (DataContainer): Object containing train data.
        configs (AttrDict): AttrDict containing configs used.
    """
    start = time.time()
    config = setup(configs, x=x)
    dynamics = config['dynamics']
    #  manager = config['manager']
    dirs = config['dirs']
    #  betas = config['betas']
    #  steps = config['steps']
    #  writer = config['writer']
    configs = config['configs']
    #  ckpt = config['checkpoint']
    train_data = config['train_data']
    #  dirs = io.setup_directories(configs, ensure_new=ensure_new)
    #  configs.update({'dirs': dirs})


    dynamics.save_config(dirs['config_dir'])
    logger.rule('TRAINING')
    x, train_data = train_dynamics(dynamics, config, dirs, x=x)
    #  x, train_data = train_dynamics(dynamics, configs, dirs, x=x,
    #                                 should_track=SHOULD_TRACK)

    if IS_CHIEF and make_plots:
        output_dir = os.path.join(dirs['train_dir'], 'outputs')
        train_data.save_data(output_dir, save_dataset=True)

        params = {
            'beta_init': train_data.data.beta[0],
            'beta_final': train_data.data.beta[-1],
            'x_shape': dynamics.config.x_shape,
            'num_steps': dynamics.config.num_steps,
            'net_weights': dynamics.net_weights,
        }
        t0 = time.time()
        output = plot_data(data_container=train_data, flags=configs,
                           params=params, out_dir=dirs['train_dir'],
                           therm_frac=therm_frac, num_chains=num_chains)
        #  data_container = output['data_container']
        #  data_container.plot_dataset(output['out_dir'],
        #                              num_chains=num_chains,
        #                              therm_frac=therm_frac,
        #                              ridgeplots=True)
        dt = time.time() - t0
        logger.debug(
            f'Time spent plotting: {dt}s = {dt // 60}m {(dt % 60):.3g}s'
        )

    dt = start - time.time()
    logger.info(f'Done training model! took: {dt:.3g}s')
    io.save_dict(dict(configs), dirs['log_dir'], 'configs')

    #  return x, dynamics, train_data, configs
    return TrainOutputs(x, dirs['log_dir'], configs, train_data, dynamics)


def run_md(
        dynamics: GaugeDynamics,
        inputs: tuple[tf.Tensor, tf.Tensor],
        md_steps: int,
):
    x, beta = inputs
    logger.debug(f'Running {md_steps} MD updates!')
    for _ in range(md_steps):
        mc_states, _ = dynamics.md_update((x, beta), training=True)
        x = mc_states.out.x
    logger.debug(f'Done!')

    return x


def run_profiler(
        dynamics: GaugeDynamics,
        inputs: tuple[tf.Tensor, tf.Tensor],
        logdir: str,
        steps: int = 10
):
    logger.debug(f'Running {steps} profiling steps!')
    x, beta = inputs
    metrics = None
    for step in range(steps):
        with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            tf.profiler.experimental.start(logdir=logdir)
            x, metrics = dynamics.train_step((x, beta))

    logger.debug(f'Done!')

    return x, metrics

# pylint: disable=broad-except
# pylint: disable=too-many-arguments,too-many-statements, too-many-branches,

def train_dynamics(
        dynamics: GaugeDynamics,
        input: dict[str, Any],
        dirs: Optional[dict[str, str]] = None,
        x: Optional[tf.Tensor] = None,
        betas: Optional[tf.Tensor] = None,
        #  should_track: bool = False,
):
    """Train model."""
    # -- Helper functions for training, logging, saving, etc. --------------
    def train_step(x: tf.Tensor, beta: tf.Tensor):
        start = time.time()
        x, metrics = dynamics.train_step((x, tf.constant(beta)))
        metrics.dt = time.time() - start
        return x, metrics

    def should_print(step: int) -> bool:
        return IS_CHIEF and step % ps_ == 0

    def should_log(step: int) -> bool:
        return IS_CHIEF and step % ls_ == 0

    def should_save(step: int) -> bool:
        return step % save_steps == 0 and ckpt is not None

    configs = input['configs']
    min_lr = configs.get('min_lr', 1e-5)
    patience = configs.get('patience', 10)
    save_steps = configs.get('save_steps', None)
    print_steps = configs.get('print_steps', 1000)
    factor = configs.get('reduce_lr_factor', 0.5)
    steps = configs.get('steps', [])

    # -- setup ----------------------------------------------------
    #  config = setup(dynamics, configs, dirs, x, betas)

    betas = input.get('betas')
    if dirs is None:
        dirs = input.get('dirs')

    steps = input['steps']
    writer = input['writer']
    manager = input['manager']
    ckpt = input['checkpoint']
    train_data = input['train_data']

    #  if dirs is None:
    #      dirs = configs.get('dirs', None)
    #      if dirs is None:
    #          dirs = config.get('dirs', None)

    assert dynamics.lr_config is not None
    warmup_steps = dynamics.lr_config.warmup_steps
    reduce_lr = ReduceLROnPlateau(monitor='loss', mode='min',
                                  warmup_steps=warmup_steps,
                                  factor=factor, min_lr=min_lr,
                                  verbose=1, patience=patience)
    reduce_lr.set_model(dynamics)

    #  steps = config.get('steps', [])
    #  train_step = config.train_step

    if IS_CHIEF and writer is not None:
        writer.set_as_default()
        #  writer = config.get('writer', None)
        #  if writer is not None:
        #      writer.set_as_default()

    #  tf.compat.v1.autograph.experimental.do_not_convert(dynamics.train_step)

    # -- Try running compiled `train_step` fn otherwise run imperatively ----
    #  if betas is None:
    #      betas = config.get('betas', [])  # type: list[tf.Tensor]

    #  if steps is None:
    #      steps = np.arange(len(betas))
    #  if x is None:
    #      xshape = dynamics.config.x_shape
    #      x = config.get('x', tf.random.uniform(xshape, -PI, PI))

    #  cbetas = config.get('betas', None)
    #  csteps = config.get('steps', None)
    #  betas = cbetas if cbetas is None else betas
    #  steps = np.arange(len(betas)) if csteps is None else steps
    assert betas is not None and len(betas) > 0
    b0 = tf.constant(betas[0])
    xshape = dynamics._xshape
    #  xshape = tuple(dynamics.config.x_shape)
    xr = tf.random.uniform(xshape, -PI, PI)
    x = input.get('x', xr) if x is None else x

    assert x is not None
    assert b0 is not None
    assert dirs is not None
    if configs.get('profiler', False):
        #  sdir = dirs.get('summary_dir', )
        sdir = dirs['summary_dir']
        x, metrics = run_profiler(dynamics, (x, b0), logdir=sdir, steps=10)
    else:
        x, metrics = dynamics.train_step((x, b0))

    # -- Run MD update to not get stuck -----------------
    md_steps = configs.get('md_steps', 0)
    if md_steps > 0:
        b0 = tf.constant(b0, )
        x = run_md(dynamics, (x, b0), md_steps)


    # -- Final setup; create timing wrapper for `train_step` function -------
    # -- and get formatted header string to display during training. --------
    ps_ = configs.get('print_steps', None)
    ls_ = configs.get('logging_steps', None)
    steps_per_epoch = configs.get('steps_per_epoch', 1000)

    warmup_steps = dynamics.lr_config.warmup_steps
    total_steps = len(betas)
    if len(steps) != len(betas):
        logger.warning(f'len(steps) != len(betas) Restarting step count!')
        logger.warning(f'len(steps): {len(steps)}, len(betas): {len(betas)}')
        steps = np.arange(len(betas))
    #  if should_track:
    #      iterable = track(enumerate(zip(steps, betas)), total=total_steps,
    #                       console=io.console, description='training',
    #                       transient=True)
    #  else:

    keep = ['dt', 'loss', 'accept_prob', 'beta',
            #  'Hf_start', 'Hf_mid', 'Hf_end'
            #  'Hb_start', 'Hb_mid', 'Hb_end',
            'Hwb_start', 'Hwb_mid', 'Hwb_end',
            'Hwf_start', 'Hwf_mid', 'Hwf_end',
            'xeps', 'veps', 'dq', 'dq_sin',
            'plaqs', 'p4x4',
            'charges', 'sin_charges']

    #  plots = init_plots(configs, figsize=(5, 2), dpi=500)
    #  discrete_betas = np.arange(beta, 8, dtype=int)
    plots = {}
    if in_notebook():
        plots = plotter.init_plots(configs, figsize=(9, 3), dpi=125)

    # -- Training loop ----------------------------------------------------
    data_strs = []
    logdir = dirs['log_dir']
    data_dir = dirs['data_dir']
    logfile = dirs['log_file']
    assert manager is not None
    assert x is not None
    #  for idx, (step, beta) in iterable:
    #  for idx, (step, beta) in enumerate(zip(steps, betas)):
    for step, beta in zip(steps, betas):
        x, metrics = train_step(x, beta)

        # TODO: Run inference when beta hits an integer
        # >>> beta_inf = {i: False, for i in np.arange(beta_final)}
        # >>> if any(np.isclose(beta, np.array(list(beta_inf.keys())))):
        # >>>     run_inference(...)

        if (step + 1) > warmup_steps and (step + 1) % steps_per_epoch == 0:
            reduce_lr.on_epoch_end(step+1, {'loss': metrics.loss})

        # -- Save checkpoints and dump configs `x` from each rank ----------
        if should_save(step + 1):
            train_data.update(step, metrics)
            train_data.dump_configs(x, data_dir, rank=RANK,
                                    local_rank=LOCAL_RANK)
            if IS_CHIEF:
                # -- Save CheckpointManager -------------
                manager.save()
                mstr = f'Checkpoint saved to: {manager.latest_checkpoint}'
                logger.log(mstr)
                # -- Save train_data and free consumed memory --------
                train_data.save_and_flush(data_dir, logfile,
                                          rank=RANK, mode='a')
                if not dynamics.config.hmc:
                    # -- Save network weights -------------------------------
                    dynamics.save_networks(logdir)
                    logger.log(f'Networks saved to: {logdir}')

        # -- Print current training state and metrics ---------------
        if should_print(step):
            train_data.update(step, metrics)
            if step % 5000 == 0:
                pre = [f'step={step}/{total_steps}']
                keep_ = keep + ['xeps_start', 'xeps_mid', 'xeps_end',
                                'veps_start', 'veps_mid', 'veps_end']
                data_str = logger.print_metrics(metrics, pre=pre, keep=keep_)
            else:
                keep_ = ['step', 'dt', 'loss', 'accept_prob', 'beta',
                         'dq_int', 'dq_sin', 'dQint', 'dQsin', 'plaqs', 'p4x4']
                pre = [f'step={step}/{total_steps}']
                data_str = logger.print_metrics(metrics, window=50,
                                                pre=pre, keep=keep_)

            data_strs.append(data_str)

        if in_notebook() and step % PLOT_STEPS == 0 and IS_CHIEF:
            train_data.update(step, metrics)
            if len(train_data.data.keys()) == 0:
                update_plots(metrics, plots)
            else:
                update_plots(train_data.data, plots)

        # -- Update summary objects ---------------------
        if should_log(step):
            logger.rule('')
            train_data.update(step, metrics)
            if writer is not None:
                update_summaries(step, metrics, dynamics)
                writer.flush()

    # -- Dump config objects -------------------------------------------------
    train_data.dump_configs(x, data_dir, rank=RANK, local_rank=LOCAL_RANK)
    if IS_CHIEF:
        manager.save()
        logger.log(f'Checkpoint saved to: {manager.latest_checkpoint}')
        train_data.save_and_flush(data_dir, logfile,
                                  rank=RANK, mode='a')
        if not dynamics.config.hmc:
            try:
                dynamics.save_networks(logdir)
            except (AttributeError, TypeError):
                pass

        if writer is not None:
            writer.flush()
            writer.close()

    return x, train_data
