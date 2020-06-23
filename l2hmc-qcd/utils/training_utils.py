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

from config import NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC, PI, TF_FLOAT, TF_INT
from network import NetworkConfig
from dynamics import DynamicsConfig
from models.gauge_model import GaugeModel
from utils.attr_dict import AttrDict
from utils.plotting_utils import plot_data
from utils.data_containers import TrainData

# pylint:disable=no-member
if tf.__version__.startswith('1.'):
    TF_VERSION = '1.x'
elif tf.__version__.startswith('2.'):
    TF_VERSION = '2.x'

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


NAMES = [
    'STEP', 'dt', 'LOSS', 'px', 'eps', 'BETA', 'sumlogdet', 'dQ', 'plaq_err',
]
HSTR = ''.join(["{:^12s}".format(name) for name in NAMES])
SEP = '-' * len(HSTR)
HEADER = '\n'.join([SEP, HSTR, SEP])


# pylint:disable=invalid-name, redefined-outer-name
def build_model(FLAGS):
    """Build model using parameters from FLAGS."""
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
        net_weights=net_weights,
        input_shape=input_shape,
        eps_trainable=not FLAGS.eps_fixed,
    )

    model = GaugeModel(FLAGS, config, net_config)

    return model, FLAGS


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

    '''
    train_dir = os.path.join(FLAGS.log_dir, 'training')
    ckpt_dir = os.path.join(train_dir, 'checkpoints')
    data_dir = os.path.join(ckpt_dir, 'train_data')
    if is_chief:
        io.check_else_make_dir([train_dir, ckpt_dir, data_dir])
    if not FLAGS.restore:
        io.save_params(dict(FLAGS), train_dir, 'FLAGS', rank=rank)


    step = tf.Variable(0, dtype=TF_INT)
    ckpt = tf.train.Checkpoint(step=step,
                               dynamics=model.dynamics,
                               optimizer=model.optimizer)
    manager = tf.train.CheckpointManager(
        ckpt, directory=ckpt_dir, max_to_keep=5
    )
    if manager.latest_checkpoint:
        io.log(f'Restored from: {manager.latest_checkpoint}', rank=rank)
        ckpt.restore(manager.latest_checkpoint)
        step = ckpt.step

    else:
        io.log('\n'.join(['No existing checkpoints found.',
                          'Starting from scratch.']), rank=rank)
    '''
    io.log(f'Building model...', rank=rank)
    model, FLAGS = build_model(FLAGS)

    io.log(f'Training model!', rank=rank)
    x, train_data = train_model(model, train_dir=train_dir, x=x)

    if is_chief:
        io.save(model, train_data, train_dir, rank=rank)
        io.save_params({'eps': model.dynamics.eps.numpy()},
                       FLAGS.log_dir, name='eps_final')
        params = {
            'beta_init': train_data.data.betas[0],
            'beta_final': train_data.data.betas[-1],
            'eps': model.dynamics.eps.numpy(),
            'lattice_shape': model.lattice_shape,
            'num_steps': model.dynamics.config.num_steps,
            'net_weights': model.dynamics_config.net_weights,
        }
        plot_data(train_data, train_dir, FLAGS, params=params)

    io.log('\n'.join(['Done training model', 80 * '=']), rank=rank)

    return x, model, train_data, FLAGS


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

    #  ckpt_dir = os.path.join(train_dir, 'checkpoints')
    #  if is_chief:
    #      io.check_else_make_dir([train_dir, ckpt_dir], rank=rank)
    #      io.save_params(dict(HFLAGS), train_dir, 'FLAGS', rank=rank)
    #
    #  model, HFLAGS = build_model(HFLAGS)
    #  step = tf.Variable(0, dtype=TF_INT)
    #  ckpt = tf.train.Checkpoint(step=step,
    #                             dynamics=model.dynamics,
    #                             optimizer=model.optimizer)
    #  manager = tf.train.CheckpointManager(
    #      ckpt, directory=ckpt_dir, max_to_keep=3
    #  )
    #  if manager.latest_checkpoint:
    #      io.log(f'Restored from: {manager.latest_checkpoint}', rank=rank)
    #      ckpt.restore(manager.latest_checkpoint)
    #      step = ckpt.step

    model, HFLAGS = build_model(HFLAGS)
    x, train_data = train_model(model, train_dir)
    #  x, train_data = train_model(model, ckpt, manager,
    #                              step=step, train_dir=train_dir)
    if is_chief:
        io.save(model, train_data, train_dir, rank=rank)
        io.save_params({'eps': model.dynamics.eps.numpy()},
                       FLAGS.log_dir, name='eps_final')
        params = {
            'beta_init': train_data.data.betas[0],
            'beta_final': train_data.data.betas[-1],
            'eps': model.dynamics.eps.numpy(),
            'lattice_shape': model.lattice_shape,
            'num_steps': model.dynamics.config.num_steps,
            'net_weights': model.dynamics_config.net_weights,
        }
        plot_data(train_data, train_dir, HFLAGS, params=params)

    #  x_out = outputs['x']
    eps_out = model.dynamics.eps.numpy()
    io.log('\n'.join(['Done with HMC start.', 80 * '=']), rank=rank)

    return x, train_data, eps_out


# pylint:disable=too-many-locals
def train_model(model, train_dir=None, x=None):
    """Train model."""
    is_chief = hvd.rank() == 0 if model.using_hvd else not model.using_hvd
    rank = hvd.rank() if model.using_hvd else 0

    data_dir = os.path.join(train_dir, 'train_data')
    ckpt_dir = os.path.join(train_dir, 'checkpoints')
    history_file = os.path.join(train_dir, 'train_log.txt')
    if is_chief:
        io.check_else_make_dir([train_dir, ckpt_dir, data_dir])

    gstep = tf.Variable(0, dtype=TF_INT)

    if x is None:
        x = tf.random.uniform(shape=model.input_shape,
                              minval=-PI, maxval=PI)
        x = tf.cast(x, dtype=TF_FLOAT)

    train_data = TrainData(model.train_steps, header=HEADER)
    ckpt = tf.train.Checkpoint(step=gstep,
                               dynamics=model.dynamics,
                               optimizer=model.optimizer)
    manager = tf.train.CheckpointManager(
        ckpt, directory=ckpt_dir, max_to_keep=5
    )
    if manager.latest_checkpoint:
        io.log(f'Restored from: {manager.latest_checkpoint}', rank=rank)
        ckpt.restore(manager.latest_checkpoint)
        train_data.restore(data_dir)
        gstep = ckpt.step
        q_new = train_data.data['charges'][-1]
        plaqs = train_data.data['plaqs'][-1]
    else:
        io.log('\n'.join(['No existing checkpoints found.',
                          'Starting from scratch.']), rank=rank)
        plaqs, q_new = model.calc_observables(x, model.betas[0])
        train_data.update({
            'plaqs': plaqs.numpy(),
            'charges': q_new.numpy(),
        })

    train_steps = np.arange(model.train_steps)
    betas = model.betas[int(gstep.numpy()):]
    steps = train_steps[int(gstep.numpy()):]

    train_step = (
        model.train_step if not model.compile
        else tf.function(model.train_step, experimental_compile=True)
    )
    '''
    if model.compile:
        train_step = tf.function(model.train_step, experimental_compile=True)
    else:
        train_step = model.train_step
    '''

    io.log(HEADER, rank=rank)
    for step, beta in zip(steps, betas):
        t0 = time.time()
        x = tf.reshape(x, model.input_shape)
        loss, x, px, sld = train_step(x, beta, step == 0)
        x = tf.reshape(x, model.lattice_shape)
        dt = time.time() - t0

        q_old = q_new
        plaqs_err, q_new = model.calc_observables(x, beta)
        dq = tf.math.abs(q_new - q_old)

        outputs = AttrDict({
            'dt': dt,
            'px': px.numpy(),
            'betas': beta.numpy(),
            'steps': step,
            'loss': loss.numpy(),
            'charges': q_new.numpy(),
            'dq': dq.numpy(),
            'sumlogdet': sld.numpy(),
            'plaqs': plaqs_err.numpy(),
            'eps': model.dynamics.eps.numpy(),
        })

        if step % model.print_steps == 0:
            data_str = train_data.get_fstr(outputs, rank=rank)
            io.log(data_str, rank=rank)

        if model.save_train_data and step % model.logging_steps == 0:
            train_data.update(outputs)

        if is_chief and step % model.save_steps == 0 and ckpt is not None:
            ckpt.step.assign(step)
            manager.save()
            io.log(f'Checkpoint saved to: {manager.latest_checkpoint}')
            train_data.save_data(data_dir)
            train_data.flush_data_strs(history_file, rank=rank, mode='a')
            #  train_data.data_strs = io.flush_data_strs(train_data.data_strs,
            #                                            history_file)

        if step % 100 == 0:
            io.log(HEADER, rank=rank)

    if is_chief and ckpt is not None and manager is not None:
        ckpt.step.assign(step)
        manager.save()
        train_data.data_strs = io.flush_data_strs(train_data.data_strs,
                                                  history_file)

    return x, train_data
