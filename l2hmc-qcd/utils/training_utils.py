"""
training_utils.py

Implements helper functions for training the model.
"""
# noqa: F401
# pylint:disable=unused-import
from __future__ import absolute_import, division, print_function

import os
import time

from typing import Union

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from tqdm.auto import tqdm

import utils.file_io as io

from config import CBARS, LearningRateConfig, NET_WEIGHTS_HMC, TF_FLOAT
from utils.file_io import timeit
from utils.attr_dict import AttrDict
from utils.summary_utils import update_summaries
from utils.plotting_utils import plot_data
from utils.data_containers import DataContainer
from utils.annealing_schedules import (exp_mult_cooling, get_betas,
                                       linear_additive_cooling,
                                       linear_multiplicative_cooling,
                                       quadratic_additive_cooling)
from dynamics.base_dynamics import BaseDynamics
from dynamics.gauge_dynamics import (build_dynamics, GaugeDynamics,
                                     GaugeDynamicsConfig)

#  try:
#      tf.config.experimental.enable_mlir_bridge()
#      tf.config.experimental.enable_mlir_graph_optimization()
#  except:  # noqa: E722
#      pass

# pylint:disable=no-member
# pylint:disable=too-many-locals
# pylint:disable=protected-access
# pylint:disable=invalid-name

RANK = hvd.rank()
io.log(f'Number of devices: {hvd.size()}')
IS_CHIEF = (RANK == 0)


@timeit(out_file=None)
def train_hmc(flags):
    """Main method for training HMC model."""
    hflags = AttrDict(dict(flags).copy())
    lr_config = AttrDict(hflags.pop('lr_config', None))
    config = AttrDict(hflags.pop('dynamics_config', None))
    net_config = AttrDict(hflags.pop('network_config', None))
    hflags.train_steps = hflags.pop('hmc_steps', None)
    hflags.beta_init = hflags.beta_final

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

    hflags.profiler = False
    hflags.make_summaries = True

    lr_config = LearningRateConfig(
        warmup_steps=0,
        decay_rate=0.9,
        decay_steps=hflags.train_steps // 10,
        lr_init=lr_config.get('lr_init', None),
    )

    train_dirs = io.setup_directories(hflags, 'training_hmc')
    dynamics = GaugeDynamics(hflags, config, net_config, lr_config)
    dynamics.save_config(train_dirs.config_dir)
    x, train_data = train_dynamics(dynamics, hflags, dirs=train_dirs)
    if IS_CHIEF:
        output_dir = os.path.join(train_dirs.train_dir, 'outputs')
        io.check_else_make_dir(output_dir)
        train_data.save_data(output_dir)

        params = {
            'eps': dynamics.eps,
            'num_steps': dynamics.config.num_steps,
            'beta_init': hflags.beta_init,
            'beta_final': hflags.beta_final,
            'lattice_shape': dynamics.config.lattice_shape,
            'net_weights': NET_WEIGHTS_HMC,
        }
        plot_data(train_data, train_dirs.train_dir, hflags,
                  thermalize=True, params=params)

    return x, train_data, dynamics.eps.numpy()


@timeit(out_file=None)
def train(
        flags: AttrDict, x: tf.Tensor = None
):
    """Train model.

    Returns:
        x (tf.Tensor): Batch of configurations
        dynamics (GaugeDynamics): Dynamics object.
        train_data (DataContainer): Object containing train data.
        flags (AttrDict): AttrDict containing flags used.
    """
    dirs = io.setup_directories(flags)
    flags.update({'dirs': dirs})

    if x is None:
        x = tf.random.normal(flags.dynamics_config['lattice_shape'])
        x = tf.reshape(x, (x.shape[0], -1))

    if flags.get('restore', False):
        xfile = os.path.join(dirs.train_dir,
                             'train_data', f'x_rank{RANK}.z')
        x = io.loadz(xfile)

    dynamics = build_dynamics(flags)
    dynamics.save_config(dirs.config_dir)

    io.log('\n'.join([120 * '*', 'Training L2HMC sampler...']))
    x, train_data = train_dynamics(dynamics, flags, dirs, x=x)

    if IS_CHIEF:
        output_dir = os.path.join(dirs.train_dir, 'outputs')
        train_data.save_data(output_dir)

        params = {
            'beta_init': train_data.data.beta[0],
            'beta_final': train_data.data.beta[-1],
            'eps': dynamics.eps.numpy(),
            'lattice_shape': dynamics.config.lattice_shape,
            'num_steps': dynamics.config.num_steps,
            'net_weights': dynamics.net_weights,
        }
        plot_data(train_data, dirs.train_dir, flags,
                  thermalize=True, params=params)

    io.log('\n'.join(['Done training model', 120 * '*']))
    io.save_dict(dict(flags), dirs.log_dir, 'configs')

    return x, dynamics, train_data, flags


def setup(dynamics, flags, dirs=None, x=None, betas=None):
    """Setup training."""
    train_data = DataContainer(flags.train_steps, dirs=dirs)
    ckpt = tf.train.Checkpoint(dynamics=dynamics,
                               optimizer=dynamics.optimizer)
    manager = tf.train.CheckpointManager(ckpt, dirs.ckpt_dir, max_to_keep=5)
    if manager.latest_checkpoint:  # restore from checkpoint
        io.log(f'Restored model from: {manager.latest_checkpoint}')
        ckpt.restore(manager.latest_checkpoint)
        current_step = dynamics.optimizer.iterations.numpy()
        x = train_data.restore(dirs.data_dir, rank=RANK, step=current_step)
        flags.beta_init = train_data.data.beta[-1]

    # Create initial samples if not restoring from ckpt
    if x is None:
        x = np.pi * tf.random.normal(shape=dynamics.x_shape)

    # Setup summary writer
    writer = None
    make_summaries = flags.get('make_summaries', True)
    if IS_CHIEF and make_summaries:
        writer = tf.summary.create_file_writer(dirs.summary_dir)

    current_step = dynamics.optimizer.iterations.numpy()  # get global step
    num_steps = max([flags.train_steps + 1, current_step + 1])
    steps = tf.range(current_step, num_steps, dtype=tf.int64)
    train_data.steps = steps[-1]
    if betas is None:
        if flags.beta_init == flags.beta_final:  # train at fixed beta
            betas = flags.beta_init * np.ones(len(steps))
        else:  # get annealing schedule w/ same length as `steps`
            betas = get_betas(len(steps), flags.beta_init, flags.beta_final)

    betas = tf.constant(betas, dtype=TF_FLOAT)
    dynamics.compile(loss=dynamics.calc_losses,
                     optimizer=dynamics.optimizer,
                     experimental_run_tf_function=False)
    #  x_tspec = tf.TensorSpec(dynamics.x_shape, dtype=x.dtype, name='x')
    #  beta_tspec = tf.TensorSpec([], dtype=TF_FLOAT, name='beta')
    #  input_signature=[x_tspec, beta_tspec])

    #  if dynamics.config['model_type'] == 'GaugeModel':
    inputs = (x, tf.constant(betas[0]))
    _ = dynamics.apply_transition(inputs, training=True)

    train_step = tf.function(dynamics.train_step)

    pstart = 0
    pstop = 0
    if flags.profiler:
        pstart = len(betas) // 2
        pstop = pstart + 10

    output = AttrDict({
        'x': x,
        'betas': betas,
        'steps': steps,
        'writer': writer,
        'manager': manager,
        'checkpoint': ckpt,
        'train_step': train_step,
        'train_data': train_data,
        'pstart': pstart,
        'pstop': pstop,
    })

    return output


# pylint: disable=too-many-arguments,too-many-statements, too-many-branches
@timeit(out_file=None)
def train_dynamics(
        dynamics: Union[BaseDynamics, GaugeDynamics],
        flags: AttrDict,
        dirs: str = None,
        x: tf.Tensor = None,
        betas: tf.Tensor = None,
):
    """Train model."""
    # setup...
    config = setup(dynamics, flags, dirs, x, betas)
    x = config.x
    steps = config.steps
    betas = config.betas
    train_step = config.train_step
    ckpt = config.checkpoint
    manager = config.manager
    train_data = config.train_data
    if IS_CHIEF:
        writer = config.writer
        if writer is not None:
            writer.set_as_default()

    # +-----------------------------------------------------------------+
    # | Try running compiled `train_step` fn otherwise run imperatively |
    # +-----------------------------------------------------------------+
    # pylint:disable=broad-except
    io.log(120 * '*')
    try:
        if flags.profiler:
            tf.summary.trace_on(graph=True, profiler=True)
        x, metrics = train_step((x, tf.constant(betas[0])))
        io.log('Compiled `dynamics.train_step` using tf.function!')
        if IS_CHIEF and flags.profiler:
            tf.summary.trace_export(name='train_step_trace', step=0,
                                    profiler_outdir=dirs.summary_dir)
            tf.summary.trace_off()
    except Exception as exception:  # pylint:disable broad-except
        io.log(exception, level='CRITICAL')
        train_step = dynamics.train_step
        x, metrics = train_step((x, tf.constant(betas[0])))
        lstr = '\n'.join(['`tf.function(dynamics.train_step)` failed!',
                          'Running `dynamics.train_step` imperatively...'])
        io.log(lstr, level='CRITICAL')
    io.log(120*'*')

    # +--------------------------------+
    # | Run MD update to not get stuck |
    # +--------------------------------+
    md_steps = flags.get('md_steps', 0)
    if md_steps > 0:
        io.log(f'Running {md_steps} MD updates...')
        for _ in range(md_steps):
            mc_states, _ = dynamics.md_update((x, tf.constant(betas[0])),
                                              training=True)
            x = mc_states.out.x

    # +--------------------------------------------------------------+
    # | Final setup; create timing wrapper for `train_step` function |
    # | and get formatted header string to display during training.  |
    # +--------------------------------------------------------------+
    def _timed_step(x: tf.Tensor, beta: tf.Tensor):
        start = time.time()
        x, metrics = train_step((x, tf.constant(beta)))
        metrics.dt = time.time() - start
        return x, metrics

    header = train_data.get_header(metrics,
                                   skip=['charges'],
                                   prepend=['{:^12s}'.format('step')])
    if IS_CHIEF:
        ctup = (CBARS['blue'], CBARS['yellow'],
                CBARS['blue'], CBARS['reset'])
        steps = tqdm(steps, desc='training', unit='step',
                     bar_format=("%s{l_bar}%s{bar}%s{r_bar}%s" % ctup))
        io.log_tqdm(header.split('\n'))

    def should_print(step):
        if IS_CHIEF and step % flags.print_steps == 0:
            return True
        return False

    def should_log(step):
        ls_ = flags.get('logging_steps', None)
        if IS_CHIEF and ls_ is not None:
            if step % ls_ == 0 and ls_ > 0:
                return True

        #  if IS_CHIEF and step % ls_ == 0 and ls_ > 0:
        #      return True
        #  return False
        return False

    def should_save(step):
        if step % flags.save_steps == 0 and ckpt is not None:
            return True
        return False

    # +------------------------------------------------+
    # |                 Training loop                  |
    # +------------------------------------------------+
    for step, beta in zip(steps, betas):
        # Perform a single training step
        x, metrics = _timed_step(x, beta)

        # Save checkpoints and dump configs `x` from each rank
        #  if (step + 1) % flags.save_steps == 0 and ckpt is not None:
        if should_save(step + 1):
            train_data.dump_configs(x, dirs.data_dir, rank=RANK)
            if IS_CHIEF:
                manager.save()
                train_data.save_and_flush(dirs.data_dir,
                                          dirs.log_file,
                                          rank=RANK, mode='a')

        # Print current training state and metrics
        #  if IS_CHIEF and step % flags.print_steps == 0:
        if should_print(step):
            data_str = train_data.get_fstr(step, metrics, skip=['charges'])
            io.log_tqdm(data_str)

        # Update summary objects
        #  tf.summary.record_if(IS_CHIEF and step % flags.logging_steps == 0)
        #  if IS_CHIEF and step % flags.logging_steps == 0:
        if should_log(step):
            train_data.update(step, metrics)
            if writer is not None:
                update_summaries(step, metrics, dynamics)
                writer.flush()

        # Print header every hundred steps
        if IS_CHIEF and (step + 1) % (50 * flags.print_steps) == 0:
            io.log_tqdm(header.split('\n'))

        #  if config.pstop > 0 and step == config.pstop:
        #      tf.profiler.experimental.stop()

    train_data.dump_configs(x, dirs.data_dir, rank=RANK)
    if IS_CHIEF:
        manager.save()
        io.log(f'Checkpoint saved to: {manager.latest_checkpoint}')
        train_data.save_and_flush(dirs.data_dir,
                                  dirs.log_file,
                                  rank=RANK, mode='a')
        if writer is not None:
            writer.flush()
            writer.close()

    return x, train_data
