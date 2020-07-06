"""
training_utils.py

Implements helper functions for training the model.
"""
from __future__ import absolute_import, division, print_function

import os
import time
import utils.file_io as io
from utils.attr_dict import AttrDict
from utils.plotting_utils import plot_data

import tensorflow as tf
import numpy as np
if tf.__version__.startswith('1.'):
    TF_VERSION = '1.x'
    tf.enable_eager_execution()
elif tf.__version__.startswith('2.'):
    TF_VERSION = '2.x'

from config import (NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC, PI, TF_FLOAT, TF_INT,
                    DynamicsConfig, NetworkConfig, HEADER)
from dynamics.gauge_dynamics import GaugeDynamics
#  from dynamics.dynamics import Dynamics
from utils.data_containers import DataContainer

# pylint:disable=no-member
try:
    import horovod.tensorflow as hvd

    hvd.init()
    if hvd.rank() == 0:
        print(f'Number of devices: {hvd.size()}')
    if TF_VERSION == '2.x':
        GPUS = tf.config.experimental.list_physical_devices('GPU')
        for gpu in GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)
        if GPUS:
            tf.config.experimental.set_visible_devices(
                GPUS[hvd.local_rank()], 'GPU'
            )
    elif TF_VERSION == '1.x':
        CONFIG = tf.compat.v1.ConfigProto()
        CONFIG.gpu_options.allow_growth = True
        CONFIG.gpu_options.visible_device_list = str(hvd.local_rank())
        tf.compat.v1.enable_eager_execution(config=CONFIG)

except ImportError:
    if TF_VERSION == '1.x':
        tf.compat.v1.enable_eager_execution()


# pylint:disable=invalid-name, redefined-outer-name
def build_dynamics(FLAGS):
    """Build dynamics using parameters from FLAGS."""
    net_weights = NET_WEIGHTS_HMC if FLAGS.hmc else NET_WEIGHTS_L2HMC
    lattice_shape = FLAGS.get('lattice_shape', None)
    if lattice_shape is None:
        try:
            xdim = FLAGS.time_size * FLAGS.space_size * FLAGS.dim
            input_shape = (FLAGS.batch_size, xdim)
            lattice_shape = (FLAGS.batch_size,
                             FLAGS.time_size,
                             FLAGS.space_size, 2)
        except AttributeError:
            raise('Either `lattice_shape` or '
                  'individual sizes must be specified.')
    else:
        xdim = lattice_shape[1] * lattice_shape[2] * lattice_shape[3]
        input_shape = (lattice_shape[0], xdim)

    FLAGS.net_weights = net_weights
    FLAGS.xdim = xdim
    FLAGS.input_shape = input_shape
    FLAGS.lattice_shape = lattice_shape

    net_config = NetworkConfig(
        units=FLAGS.units,
        type='GaugeNetwork',
        activation_fn=tf.nn.relu,
        dropout_prob=FLAGS.dropout_prob,
    )
    config = DynamicsConfig(
        eps=FLAGS.eps,
        hmc=FLAGS.hmc,
        num_steps=FLAGS.num_steps,
        model_type='GaugeModel',
        eps_trainable=not FLAGS.eps_fixed,
        separate_networks=FLAGS.separate_networks,
    )

    dynamics = GaugeDynamics(FLAGS, config, net_config)

    return dynamics, FLAGS


def train(FLAGS, log_file=None):
    """Train model."""
    is_chief = hvd.rank() == 0 if FLAGS.horovod else not FLAGS.horovod
    rank = hvd.rank() if FLAGS.horovod else 0
    FLAGS.net_weights = NET_WEIGHTS_HMC if FLAGS.hmc else NET_WEIGHTS_L2HMC

    if FLAGS.log_dir is None:
        FLAGS.log_dir = io.make_log_dir(FLAGS, 'GaugeModel',
                                        log_file, rank=rank)
        train_dir = os.path.join(FLAGS.log_dir, 'training')
        io.check_else_make_dir(train_dir, rank=rank)
        io.save_params(dict(FLAGS), train_dir, 'FLAGS', rank=rank)
    else:
        fpath = os.path.join(FLAGS.log_dir, 'training', 'FLAGS.z')
        FLAGS = AttrDict(dict(io.loadz(fpath)))
        train_dir = os.path.join(FLAGS.log_dir, 'training')
        io.check_else_make_dir(train_dir, rank=rank)
        FLAGS.restore = True

    if FLAGS.hmc_start and FLAGS.hmc_steps > 0 and not FLAGS.restore:
        x, train_data, eps_init = train_hmc(FLAGS)
        FLAGS.eps = eps_init
    else:
        x = None

    io.log(f'INFO:Building model...', rank=rank)
    dynamics, FLAGS = build_dynamics(FLAGS)

    io.log(f'INFO:Training model!', rank=rank)
    x, train_data = train_dynamics(dynamics, FLAGS, train_dir=train_dir, x=x)

    if is_chief:
        if not dynamics.config.hmc:
            io.save_network_weights(dynamics, train_dir, rank=rank)
        if FLAGS.save_train_data:
            output_dir = os.path.join(train_dir, 'outputs')
            train_data.save_data(output_dir)

        params = {
            'beta_init': train_data.data.beta[0],
            'beta_final': train_data.data.beta[-1],
            'eps': dynamics.eps.numpy(),
            'lattice_shape': dynamics.lattice_shape,
            'num_steps': dynamics.config.num_steps,
            'net_weights': dynamics.net_weights,
        }
        plot_data(train_data, train_dir, FLAGS,
                  thermalize=True, params=params)

    io.log('\n'.join(['INFO:Done training model', 80 * '=']), rank=rank)

    return x, dynamics, train_data, FLAGS


def train_hmc(FLAGS):
    """Main method for training HMC model."""
    is_chief = hvd.rank() == 0 if FLAGS.horovod else not FLAGS.horovod
    rank = hvd.rank() if FLAGS.horovod else 0
    HFLAGS = AttrDict(dict(FLAGS))
    HFLAGS.dropout_prob = 0.
    HFLAGS.hmc = True
    HFLAGS.save_train_data = True
    HFLAGS.train_steps = HFLAGS.pop('hmc_steps')
    HFLAGS.warmup_lr = False
    HFLAGS.lr_decay_steps = HFLAGS.train_steps // 4
    HFLAGS.logging_steps = HFLAGS.train_steps // 20
    HFLAGS.beta_final = HFLAGS.beta_init
    HFLAGS.fixed_beta = True
    HFLAGS.no_summaries = True
    train_dir = os.path.join(HFLAGS.log_dir, 'training_hmc')

    dynamics, HFLAGS = build_dynamics(HFLAGS)
    x, train_data = train_dynamics(dynamics, HFLAGS, train_dir)
    if is_chief:
        if HFLAGS.save_train_data:
            output_dir = os.path.join(train_dir, 'outputs')
            io.check_else_make_dir(output_dir)
            train_data.save_data(output_dir)

        params = {
            'eps': dynamics.eps,
            'num_steps': dynamics.config.num_steps,
            'beta_init': HFLAGS.beta_init,
            'beta_final': HFLAGS.beta_final,
            'lattice_shape': dynamics.lattice_shape,
            'net_weights': NET_WEIGHTS_HMC,
        }
        plot_data(train_data, train_dir, HFLAGS,
                  thermalize=True, params=params)

    io.log('\n'.join(['Done with HMC start.', 80 * '=']), rank=rank)

    return x, train_data, dynamics.eps.numpy()


def setup_training(dynamics, flags, train_dir=None, x=None, betas=None):
    is_chief = (
        hvd.rank() == 0 if dynamics.using_hvd
        else not dynamics.using_hvd
    )
    rank = hvd.rank() if dynamics.using_hvd else 0
    data_dir = os.path.join(train_dir, 'train_data')
    ckpt_dir = os.path.join(train_dir, 'checkpoints')
    history_file = os.path.join(train_dir, 'train_log.txt')
    if is_chief:
        io.check_else_make_dir([train_dir, ckpt_dir, data_dir])

    if x is None:
        x = tf.random.uniform(shape=dynamics.x_shape, minval=-PI, maxval=PI)
        x = tf.cast(x, dtype=TF_FLOAT)

    train_data = DataContainer(flags.train_steps,
                               header=HEADER,
                               skip_keys=['charges'])

    ckpt = tf.train.Checkpoint(dynamics=dynamics,
                               optimizer=dynamics.optimizer)
    manager = tf.train.CheckpointManager(
        ckpt, directory=ckpt_dir, max_to_keep=5
    )

    if manager.latest_checkpoint:
        io.log(f'INFO:Restored from: {manager.latest_checkpoint}', rank=rank)
        ckpt.restore(manager.latest_checkpoint)
        train_data.restore(data_dir)
        current_step = dynamics.optimizer.iterations.numpy()
    else:
        current_step = tf.convert_to_tensor(0, dtype=TF_INT)
        io.log('\n'.join(['WARNING:No existing checkpoints found.',
                          'Starting from scratch.']), rank=rank)

    train_steps = tf.range(flags.train_steps)
    if betas is None:
        if flags.beta_init == flags.beta_final:
            betas = tf.convert_to_tensor(
                flags.beta_init * np.ones(flags.train_steps)
            )
        else:
            b_arr = np.linspace(flags.beta_init,
                                flags.beta_final,
                                flags.train_steps)
            betas = tf.cast(b_arr, dtype=TF_FLOAT)

    betas = betas[current_step:]
    steps = train_steps[current_step:]
    outputs = {
        'x': x,
        'betas': betas,
        'steps': steps,
        'checkpoint': ckpt,
        'manager': manager,
        'data_dir': data_dir,
        'train_data': train_data,
        'history_file': history_file,
    }

    return outputs


# pylint:disable=too-many-locals
def train_dynamics(dynamics, flags, train_dir=None,
                   x=None, betas=None, should_compile=False):
    """Train model."""
    is_chief = (
        hvd.rank() == 0 if dynamics.using_hvd
        else not dynamics.using_hvd
    )
    rank = hvd.rank() if dynamics.using_hvd else 0

    setup = setup_training(dynamics, flags, train_dir, x, betas)

    x = setup['x']
    betas = setup['betas']
    steps = setup['steps']
    ckpt = setup['checkpoint']
    manager = setup['manager']
    data_dir = setup['data_dir']
    train_data = setup['train_data']
    history_file = setup['history_file']

    if should_compile:
        train_step = tf.function(dynamics.train_step)
    else:
        train_step = dynamics.train_step

    '''
    data_dir = os.path.join(train_dir, 'train_data')
    ckpt_dir = os.path.join(train_dir, 'checkpoints')
    history_file = os.path.join(train_dir, 'train_log.txt')
    if is_chief:
        io.check_else_make_dir([train_dir, ckpt_dir, data_dir])


    train_data = DataContainer(flags.train_steps,
                               skip_keys=['charges'],
                               header=HEADER)
    ckpt = tf.train.Checkpoint(dynamics=dynamics,
                               optimizer=dynamics.optimizer)
    manager = tf.train.CheckpointManager(
        ckpt, directory=ckpt_dir, max_to_keep=5
    )
    if manager.latest_checkpoint:
        io.log(f'INFO:Restored from: {manager.latest_checkpoint}', rank=rank)
        ckpt.restore(manager.latest_checkpoint)
        train_data.restore(data_dir)
        current_step = dynamics.optimizer.iterations.numpy()
    else:
        current_step = tf.convert_to_tensor(0, dtype=TF_INT)
        io.log('\n'.join(['WARNING:No existing checkpoints found.',
                          'Starting from scratch.']), rank=rank)

    train_steps = tf.range(flags.train_steps)
    if betas is None:
        if flags.beta_init == flags.beta_final:
            betas = tf.convert_to_tensor(
                flags.beta_init * np.ones(flags.train_steps)
            )
        else:
            b_arr = np.linspace(flags.beta_init,
                                flags.beta_final,
                                flags.train_steps)
            betas = tf.cast(b_arr, dtype=TF_FLOAT)

    betas = betas[current_step:]
    steps = train_steps[current_step:]
    '''

    # pylint:disable=protected-access
    #  train_step = tf.function(dynamics.train_step)
    # , experimental_compile=True)
    #  train_step = (
    #      dynamics.train_step if FLAGS.eager_execution,
    #      else tf.function(dynamics.train_step, experimental_compile=True)
    #  )
    #
    io.log(HEADER, rank=rank)
    for step, beta in zip(steps, betas):

        start = time.time()
        #  x, metrics = dynamics.train_step((x, beta), step == 0)
        x, metrics = train_step((x, beta), step == 0)
        metrics.dt = time.time() - start

        if step % flags.print_steps == 0:
            data_str = train_data.get_fstr(step, metrics, rank=rank)
            io.log(data_str, rank=rank)

        if flags.save_train_data and step % flags.logging_steps == 0:
            train_data.update(step, metrics)

        if is_chief and step % flags.save_steps == 0 and ckpt is not None:
            manager.save()
            io.log(f'INFO:Checkpoint saved to: {manager.latest_checkpoint}')
            train_data.save_data(data_dir)
            train_data.flush_data_strs(history_file, rank=rank, mode='a')

        if step % 100 == 0:
            io.log(HEADER, rank=rank)

    if is_chief and ckpt is not None and manager is not None:
        manager.save()
        train_data.flush_data_strs(history_file, rank=rank, mode='a')

    return x, train_data
