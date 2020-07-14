"""
training_utils.py

Implements helper functions for training the model.
"""
from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np
import tensorflow as tf

import utils.file_io as io

from config import HEADER, NET_WEIGHTS_HMC, PI, TF_FLOAT, TF_INT
from dynamics.gauge_dynamics import build_dynamics, GaugeDynamics
from utils.attr_dict import AttrDict
from utils.plotting_utils import plot_data

#  from dynamics.dynamics import Dynamics
from utils.data_containers import DataContainer

# pylint:disable=no-member
# pylint:disable=too-many-locals
# pylint:disable=protected-access
try:
    import horovod.tensorflow as hvd

    hvd.init()
    if hvd.rank() == 0:
        io.log(f'Number of devices: {hvd.size()}')
    GPUS = tf.config.experimental.list_physical_devices('GPU')
    for gpu in GPUS:
        tf.config.experimental.set_memory_growth(gpu, True)
    if GPUS:
        tf.config.experimental.set_visible_devices(
            GPUS[hvd.local_rank()], 'GPU'
        )

except ImportError:
    pass


def summarize_dict(d, step, prefix=None):
    """Create summaries for all items in d."""
    if prefix is None:
        prefix = ''
    for key, val in d.items():
        name = f'{prefix}/{key}'
        tf.summary.histogram(name, val, step=step)
        tf.summary.scalar(f'{name}_avg', tf.reduce_mean(val), step=step)


def summarize_list(x, step, prefix=None):
    """Create summary objects for all items in `x`."""
    if prefix is None:
        prefix = ''
    for t in x:
        name = f'{prefix}/{t.name}'
        tf.summary.histogram(name, t, step)
        tf.summary.scalar(f'{name}_avg', tf.reduce_mean(t), step=step)


def update_summaries(step: int, metrics: AttrDict, dynamics: GaugeDynamics):
    """Create summary objects.

    NOTE: Explicitly, we create summary objects for all entries in
      - metrics
      - dynamics.variables
      - dynamics.optimizer.variables()

    Returns:
        None
    """
    learning_rate = dynamics.lr(tf.constant(step)).numpy()
    opt_vars = dynamics.optimizer.variables()
    summarize_dict(metrics, step, prefix='training')
    summarize_list(dynamics.variables, step, prefix='dynamics')
    summarize_list(opt_vars, step, prefix='dynamics.optimizer')
    tf.summary.scalar('training/learning_rate', learning_rate, step)


def exp_mult_cooling(step, temp_init, temp_final, num_steps, alpha=None):
    """Annealing function."""
    if alpha is None:
        alpha = tf.exp(
            (tf.math.log(temp_final) - tf.math.log(temp_init)) / num_steps
        )

    temp = temp_init * (alpha ** step)

    return tf.cast(temp, TF_FLOAT)


def get_betas(steps, beta_init, beta_final):
    """Get array of betas to use in annealing schedule."""
    t_init = 1. / beta_init
    t_final = 1. / beta_final
    t_arr = [
        exp_mult_cooling(i, t_init, t_final, steps) for i in range(steps)
    ]

    return 1. / tf.convert_to_tensor(np.array(t_arr))


def restore_flags(flags, train_dir):
    """Update `FLAGS` using restored flags from `log_dir`."""
    restored = AttrDict(dict(io.loadz(os.path.join(train_dir, 'FLAGS.z'))))
    restored_flags = AttrDict({
        'horovod': restored.horovod,
        'plaq_weight': restored.plaq_weight,
        'charge_weight': restored.charge_weight,
        'lattice_shape': restored.lattice_shape,
        'units': restored.units,
    })

    flags.update(restored_flags)

    return flags


def setup_directories(flags, rank, is_chief, train_dir_name='training'):
    """Setup relevant directories for training."""
    train_dir = os.path.join(flags.log_dir, train_dir_name)
    train_paths = AttrDict({
        'log_dir': flags.log_dir,
        'train_dir': train_dir,
        'data_dir': os.path.join(train_dir, 'train_data'),
        'ckpt_dir': os.path.join(train_dir, 'checkpoints'),
        'summary_dir': os.path.join(train_dir, 'summaries'),
        'history_file': os.path.join(train_dir, 'train_log.txt')
    })

    if is_chief:
        io.check_else_make_dir(
            [d for k, d in train_paths.items() if 'file' not in k],
            rank=rank
        )
        io.save_params(dict(flags), train_dir, 'FLAGS', rank=rank)

    return train_paths


def train(flags, log_file=None):
    """Train model."""
    is_chief = hvd.rank() == 0 if flags.horovod else not flags.horovod
    rank = hvd.rank() if flags.horovod else 0

    if flags.log_dir is None:
        flags.log_dir = io.make_log_dir(flags, 'GaugeModel',
                                        log_file, rank=rank)
    else:
        flags = restore_flags(flags, os.path.join(flags.log_dir, 'training'))
        flags.restore = True

    train_dirs = setup_directories(flags, rank, is_chief)

    x = None
    if flags.hmc_start and flags.hmc_steps > 0 and not flags.restore:
        x, train_data, eps_init = train_hmc(flags)
        flags.eps = eps_init

    dynamics = build_dynamics(flags)
    io.log('\n'.join(
        [80 * '=', 'FLAGS:', *[f' {k}: {v}' for k, v in flags.items()]]
    ))

    try:
        x, train_data = train_dynamics(dynamics, flags,
                                       x=x, dirs=train_dirs)
    except:
        import pudb; pudb.set_trace()

    if is_chief:
        output_dir = os.path.join(train_dirs.train_dir, 'outputs')
        train_data.save_data(output_dir)

        params = {
            'beta_init': train_data.data.beta[0],
            'beta_final': train_data.data.beta[-1],
            'eps': dynamics.eps.numpy(),
            'lattice_shape': dynamics.lattice_shape,
            'num_steps': dynamics.config.num_steps,
            'net_weights': dynamics.net_weights,
        }
        plot_data(train_data, train_dirs.train_dir,
                  flags, thermalize=True, params=params)

    io.log('\n'.join(['INFO:Done training model', 80 * '=']), rank=rank)

    return x, dynamics, train_data, flags


def train_hmc(flags):
    """Main method for training HMC model."""
    is_chief = hvd.rank() == 0 if flags.horovod else not flags.horovod
    rank = hvd.rank() if flags.horovod else 0
    hmc_flags = AttrDict(dict(flags))
    hmc_flags.dropout_prob = 0.
    hmc_flags.hmc = True
    hmc_flags.train_steps = hmc_flags.pop('hmc_steps')
    hmc_flags.warmup_steps = 0
    hmc_flags.lr_decay_steps = hmc_flags.train_steps // 4
    hmc_flags.logging_steps = hmc_flags.train_steps // 20
    hmc_flags.beta_final = hmc_flags.beta_init
    hmc_flags.fixed_beta = True
    hmc_flags.no_summaries = True

    train_dirs = setup_directories(hmc_flags, rank, is_chief,
                                   train_dir_name='training_hmc')

    dynamics = build_dynamics(hmc_flags)
    x, train_data = train_dynamics(dynamics, hmc_flags,
                                   dirs=train_dirs)
    if is_chief:
        output_dir = os.path.join(train_dirs.train_dir, 'outputs')
        io.check_else_make_dir(output_dir)
        train_data.save_data(output_dir)

        params = {
            'eps': dynamics.eps,
            'num_steps': dynamics.config.num_steps,
            'beta_init': hmc_flags.beta_init,
            'beta_final': hmc_flags.beta_final,
            'lattice_shape': dynamics.lattice_shape,
            'net_weights': NET_WEIGHTS_HMC,
        }
        plot_data(train_data, train_dirs.train_dir, hmc_flags,
                  thermalize=True, params=params)

    io.log('\n'.join(['Done with HMC start.', 80 * '=']), rank=rank)

    return x, train_data, dynamics.eps.numpy()


def setup(dynamics, flags, dirs=None, x=None, betas=None):
    """Setup training."""
    is_chief = (
        hvd.rank() == 0 if dynamics.using_hvd
        else not dynamics.using_hvd
    )
    rank = hvd.rank() if dynamics.using_hvd else 0

    train_data = DataContainer(flags.train_steps, HEADER, ['charges'])
    ckpt = tf.train.Checkpoint(model=dynamics, optimizer=dynamics.optimizer)
    manager = tf.train.CheckpointManager(ckpt, dirs.ckpt_dir, max_to_keep=5)
    if manager.latest_checkpoint:
        try:
            io.log(f'INFO:Restored model from: {manager.latest_checkpoint}')
            ckpt.restore(manager.latest_checkpoint)
            current_step = dynamics.optimizer.iterations.numpy()
            x = train_data.restore(dirs.data_dir, rank=rank, step=current_step)
            flags.train_steps += 1
            flags.beta_init = train_data.data.beta[-1]
        except:
            import pudb; pudb.set_trace()

    if x is None:
        x = tf.random.uniform(shape=dynamics.x_shape,
                              minval=-PI, maxval=PI,
                              dtype=TF_FLOAT)
    if is_chief:
        writer = tf.summary.create_file_writer(dirs.summary_dir)

    current_step = dynamics.optimizer.iterations.numpy()
    steps = tf.cast(
        tf.range(current_step, flags.train_steps),
        dtype=tf.int64
    )
    if betas is None:
        if flags.beta_init == flags.beta_final:
            betas = tf.convert_to_tensor(flags.beta_init * np.ones(len(steps)))
        else:
            betas = get_betas(len(steps), flags.beta_init, flags.beta_final)
    if flags.get('compile', False):
        io.log('INFO:Compiling dynamics.train_step using tf.function\n', rank)
        dynamics.compile(loss=dynamics.calc_losses,
                         optimizer=dynamics.optimizer,
                         experimental_run_tf_function=False)
        train_step = tf.function(dynamics.train_step)
    else:
        io.log('INFO: Running dynamics.train_step imperatively.\n', rank)
        dynamics.compile(loss=dynamics.calc_losses,
                         optimizer=dynamics.optimizer,
                         experimental_run_tf_function=False)
        train_step = dynamics.train_step

    profiler_start_step = 0
    profiler_stop_step = 0
    if flags.profiler:
        profiler_start_step = len(betas) // 2
        profiler_stop_step = profiler_start_step + 10

    output = AttrDict({
        'x': x,
        'betas': betas,
        'steps': steps,
        'writer': writer,
        'manager': manager,
        'checkpoint': ckpt,
        'train_step': train_step,
        'train_data': train_data,
        'profiler_start_step': profiler_start_step,
        'profiler_stop_step': profiler_stop_step,
    })

    return output


def train_dynamics(dynamics, flags, dirs=None, x=None, betas=None):
    """Train model."""
    is_chief = (
        hvd.rank() == 0 if dynamics.using_hvd
        else not dynamics.using_hvd
    )
    rank = hvd.rank() if dynamics.using_hvd else 0

    '''
    train_data = DataContainer(flags.train_steps, HEADER, ['charges'])
    ckpt = tf.train.Checkpoint(model=dynamics, optimizer=dynamics.optimizer)
    manager = tf.train.CheckpointManager(ckpt, dirs.ckpt_dir, max_to_keep=5)
    if manager.latest_checkpoint:
        try:
            io.log(f'INFO:Restored model from: {manager.latest_checkpoint}')
            ckpt.restore(manager.latest_checkpoint)
            current_step = dynamics.optimizer.iterations.numpy()
            x = train_data.restore(dirs.data_dir, rank=rank, step=current_step)
            flags.train_steps += 1
            flags.beta_init = train_data.data.beta[-1]
        except:
            import pudb; pudb.set_trace()

    if x is None:
        x = tf.random.uniform(shape=dynamics.x_shape,
                              minval=-PI, maxval=PI,
                              dtype=TF_FLOAT)
    if is_chief:
        writer = tf.summary.create_file_writer(dirs.summary_dir)
        writer.set_as_default()

    current_step = dynamics.optimizer.iterations.numpy()
    steps = tf.cast(
        tf.range(current_step, flags.train_steps - current_step),
        dtype=tf.int64
    )
    if betas is None:
        if flags.beta_init == flags.beta_final:
            betas = tf.convert_to_tensor(flags.beta_init * np.ones(len(steps)))

        else:
            betas = get_betas(len(steps), flags.beta_init, flags.beta_final)

    if flags.get('compile', False):
        io.log('INFO:Compiling dynamics.train_step using tf.function\n', rank)
        dynamics.compile(loss=dynamics.calc_losses,
                         optimizer=dynamics.optimizer,
                         experimental_run_tf_function=False)
        train_step = tf.function(dynamics.train_step)
    else:
        io.log('INFO:Running `dynamics.train_step` imperatively\n', rank)
        train_step = dynamics.train_step

    if flags.profiler:
        profiler_start_step = len(betas) // 2
        profiler_stop_step = profiler_start_step + 10
    '''
    outputs = setup(dynamics, flags, dirs, x, betas)
    x = outputs.x
    steps = outputs.steps
    betas = outputs.betas
    train_step = outputs.train_step
    ckpt = outputs.checkpoint
    manager = outputs.manager
    train_data = outputs.train_data
    if is_chief:
        writer = outputs.writer
        writer.set_as_default()

    if dynamics.optimizer.iterations.numpy() == 0:
        x, metrics = train_step(inputs=(x, betas[0]), first_step=True)

    io.log(HEADER, rank=rank)
    betas = tf.cast(betas, dtype=TF_FLOAT)
    for step, beta in zip(steps, betas):
        # Perform a single training step
        start = time.time()
        x, metrics = train_step((x, beta))
        metrics.dt = time.time() - start

        # Start profiler
        #  if flags.profiler and step == profiler_start_step:
        #      tf.profiler.experimental.start(flags.log_dir)

        # Save checkpoints and dump configs `x` from each rank
        if step % flags.save_steps == 0 and ckpt is not None:
            train_data.dump_configs(x, dirs.data_dir, rank=rank)
            if is_chief:
                manager.save()
                io.log(
                    f'INFO:Checkpoint saved to: {manager.latest_checkpoint}'
                )
                train_data.save_and_flush(dirs.data_dir,
                                          dirs.history_file,
                                          rank=rank, mode='a')

        # Print current training state and metrics
        if is_chief and step % flags.print_steps == 0:
            data_str = train_data.get_fstr(step, metrics, rank=rank)
            io.log(data_str, rank=rank)

        # Update summary objects
        #  tf.summary.record_if(is_chief and step % flags.logging_steps == 0)
        if is_chief and step % flags.logging_steps == 0:
            train_data.update(step, metrics)
            update_summaries(step, metrics, dynamics)

        # Print header every hundred steps
        if is_chief and step % 100 == 0:
            io.log(HEADER, rank=rank)

        #  if step % flags.save_steps == 0 and ckpt is not None:
        #      manager.save()
        #      io.log(f'INFO:Checkpoint saved to: '
        #             f'{manager.latest_checkpoint}')
        #      train_data.save_and_flush(x, beta, dirs.data_dir,
        #                                dirs.history_file,
        #                                rank=rank, mode='a')
        #  train_data.save_data(dirs.data_dir)
        #  train_data.flush_data_strs(dirs.history_file,
        #                             rank=rank, mode='a')
        #  if flags.profiler and step == profiler_stop_step:
        #      tf.profiler.experimental.stop()

    #  if flags.profiler:
    #      try:
    #          tf.profiler.experimental.stop()
    #      except AttributeError:
    #          pass

    train_data.dump_configs(x, dirs.data_dir, rank=rank)
    if is_chief:
        manager.save()
        #  ckpt.save(os.path.join(dirs.ckpt_dir, 'ckpt'))
        if not dynamics.config.hmc:
            tf.summary.trace_on(graph=True, profiler=True)
            x, metrics = train_step((x, betas[-1]))
            tf.summary.trace_export(name='train_step_trace', step=steps[-1],
                                    profiler_outdir=dirs.summary_dir)
            tf.summary.trace_off()

        io.log(f'INFO:Checkpoint saved to: {manager.latest_checkpoint}')
        train_data.save_and_flush(dirs.data_dir,
                                  dirs.history_file,
                                  rank=rank, mode='a')
        writer.flush()
        writer.close()
        #  train_data.save_data(dirs.data_dir)
        #  train_data.flush_data_strs(dirs.history_file, rank=rank, mode='a')

    return x, train_data
