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
from typing import Optional, Union

import numpy as np
import tensorflow as tf

import utils.file_io as io
from config import PI, TF_FLOAT
from dynamics.base_dynamics import BaseDynamics
from dynamics.config import NET_WEIGHTS_HMC
from dynamics.gauge_dynamics import GaugeDynamics, build_dynamics
from network.config import LearningRateConfig
from utils import SKEYS
from utils.annealing_schedules import get_betas
from utils.attr_dict import AttrDict
from utils.data_containers import DataContainer
from utils.inference_utils import run as run_inference
from utils.learning_rate import ReduceLROnPlateau
import utils.live_plots as plotter
#  from utils.live_plots import (LivePlotData, init_plots, update_joint_plots,
#                                update_plot, update_plots)
from utils.logger import Logger, in_notebook
#  from utils.logger_config import logger as logging
from utils.plotting_utils import plot_data
from utils.summary_utils import update_summaries
#  from utils.training_utils import train_dynamics

try:
    import horovod
    import horovod.tensorflow as hvd  # pylint:disable=wrong-import-order
    try:
        RANK = hvd.rank()
    except ValueError:
        hvd.init()

    RANK = hvd.rank()
    LOCAL_RANK = hvd.local_rank()
    IS_CHIEF = (RANK == 0)
    HAS_HOROVOD = True
    NUM_WORKERS = hvd.size()
    #  hvd.init()
    GPUS = tf.config.experimental.list_physical_devices('GPU')
    for gpu in GPUS:
        tf.config.experimental.set_memory_growth(gpu, True)
    if GPUS:
        gpu = GPUS[hvd.local_rank()]  # pylint:disable=invalid-name
        tf.config.experimental.set_visible_devices(gpu, 'GPU')

except (ImportError, ModuleNotFoundError):
    HAS_HOROVOD = False
    RANK = LOCAL_RANK = 0
    SIZE = LOCAL_SIZE = 1
    IS_CHIEF = (RANK == 0)




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

logger = io.Logger()

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
        'dynamics_config': config,
        'dirs': io.setup_directories(hconfigs, 'training_hmc'),
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
        dt = time.time() - t0
        io.rule(f'Time spent plotting: {dt}s = {dt // 60}m {(dt % 60):.3g}s')

    return x, dynamics, train_data, hconfigs


@dataclass
class TrainOutputs:
    x: tf.Tensor
    configs: AttrDict
    data: DataContainer
    dynamics: GaugeDynamics


def train(
        configs: AttrDict,
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
    t0 = time.time()
    ensure_new = configs.get('ensure_new', False)
    dirs = io.setup_directories(configs, ensure_new=ensure_new)
    configs.update({'dirs': dirs})

    if restore_x:
        x, xfile = None, None
        try:
            xfile = os.path.join(dirs['train_dir'], 'train_data',
                                 f'x_rank{RANK}-{LOCAL_RANK}.z')
            x = io.loadz(xfile)
        except Exception as e:  # pylint:disable=broad-except
            logger.log(f'exception: {e}')
            logger.log(f'Unable to restore x from {xfile}. Using random init.')

    if x is None:
        xshape = configs.get('dynamics_config', {}).get('x_shape')
        x = tf.random.uniform(xshape, -PI, PI)

    # Reshape x from (batch_size, Nt, Nx, 2) --> (batch_size, Nt * Nx * 2)
    x = tf.reshape(x, (x.shape[0], -1))

    dynamics = build_dynamics(configs)
    logger.log(f'dynamics.net_weights: {dynamics.net_weights}')
    network_dir = dynamics.config.log_dir
    if network_dir is not None:
        xnet, vnet = dynamics._load_networks(network_dir)
        dynamics.xnet = xnet
        dynamics.vnet = vnet

    dynamics.save_config(dirs['config_dir'])

    io.rule('TRAINING')
    assert x is not None
    x, train_data = train_dynamics(dynamics, configs, dirs, x=x)
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
        io.rule(f'Time spent plotting: {dt}s = {dt // 60}m {(dt % 60):.3g}s')

    dt = time.time() - t0
    io.rule(f'Done training model! took: {dt:.3g}s')
    io.save_dict(dict(configs), dirs['log_dir'], 'configs')

    #  return x, dynamics, train_data, configs
    return TrainOutputs(x, configs, train_data, dynamics)


# pylint:disable=too-many-statements, too-many-branches
def setup(dynamics, configs, dirs=None, x=None, betas=None):
    """Setup training."""
    if dirs is None:
        dirs = io.setup_directories(configs)
        configs['dirs'] = dirs
        configs['log_dir'] = dirs['log_dir']

    train_data = DataContainer(configs.train_steps, dirs=dirs,
                               print_steps=configs.print_steps)
    #  if dynamics._has_trainable_params:
    ckpt = tf.train.Checkpoint(dynamics=dynamics,
                               optimizer=dynamics.optimizer)
    ckptdir = dirs['ckpt_dir']
    datadir = dirs['data_dir']
    summdir = dirs['summary_dir']
    manager = tf.train.CheckpointManager(ckpt, ckptdir, max_to_keep=5)
    if manager.latest_checkpoint and not configs.get('ensure_new', False):
        logger.log(f'Restored model from: {manager.latest_checkpoint}')
        ckpt.restore(manager.latest_checkpoint)
        current_step = dynamics.optimizer.iterations.numpy()
        x = train_data.restore(datadir, step=current_step,
                               rank=RANK, local_rank=LOCAL_RANK,
                               x_shape=dynamics.x_shape)
    else:
        logger.rule('Starting new training run')

    # Create initial samples if not restoring from ckpt
    if x is None:
        x = np.pi * tf.random.normal(shape=dynamics.x_shape)

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
    num_steps = max([configs.train_steps + 1, current_step + 1])
    steps = tf.range(current_step, num_steps, dtype=tf.int64)
    train_data.steps = steps[-1]
    if configs.beta_init == configs.beta_final:
        #  betas = configs.beta_init * np.ones(len(steps))
        ones = tf.ones(len(steps))
        betas = tf.cast(configs.beta_final, ones.dtype) * ones
    else:
        betas = get_betas(num_steps - 1, configs.beta_init, configs.beta_final)

    if current_step > len(betas):
        diff = current_step - len(betas)
        betas = list(betas) + [configs.beta_final for _ in range(diff)]

    betas = betas[current_step:]

    if len(betas) < configs.train_steps:
        diff = configs.train_steps - len(betas)
        betas = (
            [i for i in betas]
            + [configs.beta_final for _ in range(diff)]
        )

    betas = tf.convert_to_tensor(betas, dtype=x.dtype)
    dynamics.compile(loss=dynamics.calc_losses,
                     optimizer=dynamics.optimizer,
                     experimental_run_tf_function=False)
    _ = dynamics.apply_transition((x, tf.constant(betas[0])), training=True)

    # -- Plot computational graph of `dynamics.xnet`, `dynamics.vnet` ------
    if IS_CHIEF:
        xf0 = os.path.join(dirs['log_dir'], 'dynamics_xnet0.png')
        xf1 = os.path.join(dirs['log_dir'], 'dynamics_xnet1.png')
        vf = os.path.join(dirs['log_dir'], 'dynamics_vnet.png')
        try:
            xnet = dynamics.xnet
            vnet = dynamics.vnet
            if dynamics.config.separate_networks:
                xnet = xnet[0]
                vnet = vnet[0]

            tf.keras.utils.plot_model(xnet[0], show_shapes=True, to_file=xf0)
            tf.keras.utils.plot_model(xnet[1], show_shapes=True, to_file=xf1)
            tf.keras.utils.plot_model(vnet, show_shapes=True, to_file=vf)

        except Exception as exception:
            print(exception)

    pstart = 0
    pstop = 0
    if configs.profiler:
        pstart = len(betas) // 2
        pstop = pstart + 10

    output = {
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
        'pstart': pstart,
        'pstop': pstop,
    }

    return output


def run_md(
        dynamics: GaugeDynamics,
        inputs: tuple[tf.Tensor, tf.Tensor],
        md_steps: int
):
    io.rule(f'Running {md_steps} MD updates!')
    for _ in range(md_steps):
        mc_states, _ = dynamics.md_update(inputs, training=True)
        x = mc_states.out.x
    io.rule(f'Done!')

    return x


def run_profiler(
        dynamics: GaugeDynamics,
        inputs: tuple[tf.Tensor, tf.Tensor],
        logdir: str,
        steps: int = 10
):
    io.rule(f'Running {steps} profiling steps!')
    x, beta = inputs
    metrics = None
    for step in range(steps):
        with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            tf.profiler.experimental.start(logdir=logdir)
            x, metrics = dynamics.train_step((x, beta))

    tf.profiler.experimental.stop(save=True)
    io.rule(f'Done!')

    return x, metrics

# pylint: disable=broad-except
# pylint: disable=too-many-arguments,too-many-statements, too-many-branches,
def train_dynamics(
        dynamics: GaugeDynamics,
        configs: AttrDict,
        dirs: Optional[dict[str, str]] = None,
        x: Optional[tf.Tensor] = None,
        betas: Optional[tf.Tensor] = None,
        #  should_track: bool = False,
):
    """Train model."""
    # -- Helper functions for training, logging, saving, etc. --------------
    min_lr = configs.get('min_lr', 1e-5)
    patience = configs.get('patience', 10)
    save_steps = configs.get('save_steps', None)
    print_steps = configs.get('print_steps', 1000)
    factor = configs.get('reduce_lr_factor', 0.5)
    steps = configs.get('steps', [])

    def train_step(x: tf.Tensor, beta: tf.Tensor):
        start = time.time()
        x, metrics = dynamics.train_step((x, tf.constant(beta)))
        metrics.dt = time.time() - start
        return x, metrics

    def should_print(step):
        return IS_CHIEF and step % ps_ == 0

    def should_log(step):
        return IS_CHIEF and step % ls_ == 0

    def should_save(step):
        return step % save_steps == 0 and ckpt is not None

    # -- setup ----------------------------------------------------
    config = setup(dynamics, configs, dirs, x, betas)

    x = config['x']
    dynamics = config['dynamics']
    betas = config['betas']
    dirs = config['dirs']
    configs = config['configs']

    steps = config['steps']
    writer = config['writer']
    manager = config['manager']
    ckpt = config['checkpoint']
    train_data = config['train_data']

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
    x = config.get('x', xr) if x is None else x

    train_data = config.get('train_data', DataContainer(steps))

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
        x = run_md(dynamics, (x, b0), md_steps)


    # -- Final setup; create timing wrapper for `train_step` function -------
    # -- and get formatted header string to display during training. --------
    ps_ = configs.get('print_steps', None)
    ls_ = configs.get('logging_steps', None)

    warmup_steps = dynamics.lr_config.warmup_steps
    steps_per_epoch = configs.get('steps_per_epoch', 1000)
    total_steps = len(betas)
    if len(steps) != len(betas):
        logger.log(f'WARNING: len(steps) != len(betas) Restarting step count!')
        logger.log(f'len(steps): {len(steps)}, len(betas): {len(betas)}')
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
    for idx, (step, beta) in enumerate(zip(steps, betas)):
        # -- Perform a single training step -------------------------------
        x, metrics = train_step(x, beta)

        # TODO: Run inference when beta hits an integer
        #  if (step + 1) > 0 and beta in discrete_betas:
        #      _ = run(dynamics, configs, x, beta=beta,

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
            train_data.update(step, metrics)
            if writer is not None:
                update_summaries(step, metrics, dynamics)
                writer.flush()

        # -- Print header every so often --------------------------
        if IS_CHIEF and (step + 1) % (50 * print_steps) == 0:
            io.rule('')

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
