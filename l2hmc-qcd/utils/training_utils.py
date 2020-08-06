"""
training_utils.py

Implements helper functions for training the model.
"""
from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
import horovod.tensorflow as hvd
hvd.init()
GPUS = tf.config.experimental.list_physical_devices('GPU')
for gpu in GPUS:
    tf.config.experimental.set_memory_growth(gpu, True)
if GPUS:
    tf.config.experimental.set_visible_devices(GPUS[hvd.local_rank()], 'GPU')

import utils.file_io as io

from config import HEADER, NET_WEIGHTS_HMC, PI, TF_FLOAT, TRAIN_STR, CBARS
from utils.attr_dict import AttrDict
from utils.summary_utils import (summarize_dict, summarize_list,
                                 update_summaries)
from utils.plotting_utils import plot_data
from utils.data_containers import DataContainer
from utils.inference_utils import run
from dynamics.base_dynamics import BaseDynamics
from dynamics.gauge_dynamics import build_dynamics, GaugeDynamics

try:
    tf.config.experimental.enable_mlir_bridge()
    tf.config.experimental.enable_mlir_graph_optimization()
except:  # noqa: E722
    pass

# pylint:disable=no-member
# pylint:disable=too-many-locals
# pylint:disable=protected-access

RANK = hvd.rank()
io.log(f'Number of devices: {hvd.size()}', RANK)
IS_CHIEF = (RANK == 0)

COLOR_TUP = (CBARS['yellow'], CBARS['reset'])


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
    t_arr = tf.convert_to_tensor(np.array([
        exp_mult_cooling(i, t_init, t_final, steps) for i in range(steps)
    ]))

    return tf.constant(1. / t_arr, dtype=TF_FLOAT)


def restore_flags(flags, train_dir):
    """Update `FLAGS` using restored flags from `log_dir`."""
    rf_file = os.path.join(train_dir, 'FLAGS.z')
    restored = AttrDict(dict(io.loadz(rf_file)))
    io.log(f'Restoring FLAGS from: {rf_file}...')
    flags.update(restored)

    return flags


def setup_directories(flags, name='training'):
    """Setup relevant directories for training."""
    train_dir = os.path.join(flags.log_dir, name)
    train_paths = AttrDict({
        'log_dir': flags.log_dir,
        'train_dir': train_dir,
        'data_dir': os.path.join(train_dir, 'train_data'),
        'ckpt_dir': os.path.join(train_dir, 'checkpoints'),
        'summary_dir': os.path.join(train_dir, 'summaries'),
        'log_file': os.path.join(train_dir, 'train_log.txt'),
        'config_dir': os.path.join(train_dir, 'dynamics_configs'),
    })

    if IS_CHIEF:
        io.check_else_make_dir(
            [d for k, d in train_paths.items() if 'file' not in k],
            #  rank=rank
        )
        if not flags.restore:
            io.save_params(dict(flags), train_dir, 'FLAGS')

    return train_paths


def train_hmc(flags):
    """Main method for training HMC model."""
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

    train_dirs = setup_directories(hmc_flags, 'training_hmc')
    dynamics = build_dynamics(hmc_flags)
    dynamics.save_config(train_dirs.config_dir)
    x, train_data = train_dynamics(dynamics, hmc_flags,
                                   test_steps=0, dirs=train_dirs)
    if IS_CHIEF:
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

    io.log('\n'.join(['Done with HMC start.', 80 * '=']), rank=RANK)

    return x, train_data, dynamics.eps.numpy()


def train(flags, log_file=None, md_steps=0, test_steps=1000):
    """Train model."""
    if flags.log_dir is None:
        flags.log_dir = io.make_log_dir(flags, 'GaugeModel',
                                        log_file, rank=RANK)
        flags.restore = False
    else:
        train_steps = flags.train_steps
        flags = restore_flags(flags, os.path.join(flags.log_dir, 'training'))
        if train_steps == flags.train_steps:
            flags.train_steps += flags.logging_steps
        flags.restore = True

    train_dirs = setup_directories(flags)

    x = None
    if flags.hmc_steps > 0 and not flags.restore:
        x, train_data, eps_init = train_hmc(flags)
        flags.eps = eps_init

    if flags.restore:
        xfile = os.path.join(train_dirs.train_dir,
                             'train_data', f'x_rank{RANK}.z')
        x = io.loadz(xfile)

    dynamics = build_dynamics(flags)
    dynamics.save_config(train_dirs.config_dir)
    io.print_flags(flags, rank=RANK)

    x, train_data = train_dynamics(dynamics, flags, x=x, dirs=train_dirs,
                                   test_steps=test_steps, md_steps=md_steps)
    if IS_CHIEF:
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

    io.log('\n'.join(['Done training model', 80 * '=']), rank=RANK)

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
        x = tf.random.uniform(shape=dynamics.x_shape,
                              minval=-PI, maxval=PI,
                              dtype=TF_FLOAT)
    # Setup summary writer
    writer = None
    if IS_CHIEF:
        writer = tf.summary.create_file_writer(dirs.summary_dir)

    current_step = dynamics.optimizer.iterations.numpy()  # get global step
    try:
        steps = tf.range(current_step, flags.train_steps + 1, dtype=tf.int64)
    except tf.errors.InvalidArgumentError:
        flags.train_steps += flags.logging_steps
        steps = tf.range(current_step, flags.train_steps + 1, dtype=tf.int64)

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
    x_tspec = tf.TensorSpec(dynamics.x_shape, dtype=x.dtype, name='x')
    beta_tspec = tf.TensorSpec([], dtype=TF_FLOAT, name='beta')
    train_step = tf.function(dynamics.train_step)
                             #  input_signature=[x_tspec, beta_tspec])

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


# pylint: disable=too-many-arguments,too-many-statements
def train_dynamics(
        dynamics: BaseDynamics,
        flags: AttrDict,
        dirs: str = None,
        x: tf.Tensor = None,
        betas: tf.Tensor = None,
        md_steps: int = 0,
        test_steps: int = 1000,
):
    """Train model."""
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
        writer.set_as_default()
    # +=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=+

    # +---------------------------------------------------------------------+
    # | Try running compiled `train_step` fn otherwise run imperatively     |
    # +---------------------------------------------------------------------+
    try:
        x, metrics = train_step((x, tf.constant(betas[0])))
        io.log('Compiled `dynamics.train_step` using tf.function!')
        # Create tensorboard summary of compiled graph
        #  args = (x, betas[0])
        #  graph = (
        #      dynamics.call.get_concrete_function(*args, training=True).graph
        #  )
        #  summary_ops_v2.graph(graph.as_graph_def())
    except:  # noqa: E722  # pylint:disable=bare-except
        train_step = dynamics.train_step
        x, metrics = train_step((x, tf.constant(betas[0])))
        io.log('Unable to compile `dynamics.train_step`, running imperatively')

    # +----------------------------------------+
    # |     Run MD update to not get stuck     |
    # +----------------------------------------+
    if md_steps > 0:
        io.log(f'Running {md_steps} MD updates...')
        for _ in range(md_steps):
            mc_states, _ = dynamics.md_update(x, tf.constant(betas[0]),
                                              training=True)
            x = mc_states.out.x

    # +--------------------------------------------------------------------+
    # | Final setup; create timing wrapper for `train_step` function       |
    # | and get formatted header string to display during training.        |
    # +--------------------------------------------------------------------+
    def _timed_step(x: tf.Tensor, beta: tf.Tensor):
        start = time.time()
        x, metrics = train_step((x, tf.constant(beta)))
        metrics.dt = time.time() - start
        return x, metrics

    header = train_data.get_header(metrics,
                                   skip=['charges'],
                                   prepend=['{:^12s}'.format('step')],
                                   split=True)
    #  io.log(header, rank=RANK)
    #  _, tail = os.path.split(dirs.log_dir)
    steps_tqdm = tqdm(steps, desc='training',
                      #  ncols=len(header[0]) + 64,
                      bar_format=("{l_bar}%s{bar}%s{r_bar}" % COLOR_TUP))

    io.log(header, rank=RANK, level='INFO')
    #  tqdm.write(header)
    # +------------------------------------------------+
    # |                 Training loop                  |
    # +------------------------------------------------+
    for step, beta in zip(steps_tqdm, betas):
        # Perform a single training step
        x, metrics = _timed_step(x, beta)

        #  Start profiler
        if config.pstart > 0 and step == config.pstart:
            tf.profiler.experimental.start(flags.log_dir)

        # Save checkpoints and dump configs `x` from each rank
        if (step + 1) % flags.save_steps == 0 and ckpt is not None:
            train_data.dump_configs(x, dirs.data_dir, rank=RANK)
            if IS_CHIEF:
                manager.save()
                io.log(
                    f'Checkpoint saved to: {manager.latest_checkpoint}',
                )
                train_data.save_and_flush(dirs.data_dir,
                                          dirs.log_file,
                                          rank=RANK, mode='a')

        # Print current training state and metrics
        if IS_CHIEF and step % flags.print_steps == 0:
            data_str = train_data.get_fstr(step, metrics, skip=['charges'])
            io.log(data_str, rank=RANK, level='INFO')

        # Update summary objects
        #  tf.summary.record_if(IS_CHIEF and step % flags.logging_steps == 0)
        if IS_CHIEF and step % flags.logging_steps == 0:
            train_data.update(step, metrics)
            update_summaries(step, metrics, dynamics)
            writer.flush()

        # Print header every hundred steps
        if IS_CHIEF and step % 100 == 0:
            io.log(header, rank=RANK)

        if config.pstop > 0 and step == config.pstop:
            tf.profiler.experimental.stop()

        # Run inference when halfway done with training to compare against
        if test_steps > 0 and (step + 1) == len(steps) // 2 and IS_CHIEF:
            io.log(header)
            io.log(f'Running inference at beta = {beta:.3g}...')
            args = AttrDict({
                'beta': beta,
                'hmc': False,
                'log_dir': dirs.log_dir,
                'run_steps': test_steps,
            })

            _ = run(dynamics, args, x=x)
            io.log('Done running inference. Resuming training...')
            io.log(header)

    try:  # make sure profiler is shut down
        tf.profiler.experimental.stop()
    except (AttributeError, tf.errors.UnavailableError):
        io.log('No active profiling session!')

    train_data.dump_configs(x, dirs.data_dir, rank=RANK)
    if IS_CHIEF:
        manager.save()
        if not dynamics.config.hmc and flags.profiler:
            tf.summary.trace_on(graph=True, profiler=True)
            x, metrics = train_step((x, tf.constant(betas[-1])))
            tf.summary.trace_export(name='train_step_trace', step=steps[-1],
                                    profiler_outdir=dirs.summary_dir)
            tf.summary.trace_off()

        io.log(f'Checkpoint saved to: {manager.latest_checkpoint}')
        train_data.save_and_flush(dirs.data_dir,
                                  dirs.log_file,
                                  rank=RANK, mode='a')
        writer.flush()
        writer.close()

    return x, train_data
