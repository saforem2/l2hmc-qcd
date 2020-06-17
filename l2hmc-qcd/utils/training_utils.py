"""
training_utils.py

Collection of helper methods to use for training model.
"""
from __future__ import absolute_import, division, print_function

from models.gauge_model import (GaugeModel, HEADER, RUN_HEADER,
                                RUN_SEP)
from utils.attr_dict import AttrDict
from utils.plotting_utils import plot_data
from dynamics import DynamicsConfig
from network import NetworkConfig
from config import (NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC, PI, TF_FLOAT,
                    TF_INT)
import utils.file_io as io
import os
import time

import numpy as np
import tensorflow as tf

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


#  import utils.file_io as io


# pylint:disable=invalid-name, redefined-outer-name
def build_model(FLAGS, save_params=True, log_file=None):
    """Build model using parameters from FLAGS."""
    net_weights = NET_WEIGHTS_HMC if FLAGS.hmc else NET_WEIGHTS_L2HMC
    xdim = FLAGS.time_size * FLAGS.space_size * FLAGS.dim
    input_shape = (FLAGS.batch_size, xdim)
    lattice_shape = (FLAGS.batch_size, FLAGS.time_size, FLAGS.space_size, 2)
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

    model = GaugeModel(FLAGS, lattice_shape, config, net_config)

    return model, FLAGS


def train(FLAGS, log_file=None):
    """Train model."""
    is_chief = hvd.rank() == 0 if FLAGS.horovod else not FLAGS.horovod
    rank = hvd.rank() if FLAGS.horovod else 0

    if FLAGS.log_dir is None:
        FLAGS.log_dir = io.make_log_dir(FLAGS, 'GaugeModel', log_file)
    else:
        fpath = os.path.join(FLAGS.log_dir, 'training', 'FLAGS.z')
        FLAGS = AttrDict(dict(io.loadz(fpath)))

    train_dir = os.path.join(FLAGS.log_dir, 'training')
    ckpt_dir = os.path.join(train_dir, 'checkpoints')
    io.check_else_make_dir([train_dir, ckpt_dir])
    io.save_params(dict(FLAGS), train_dir, 'FLAGS', rank=rank)

    if FLAGS.hmc_start and FLAGS.hmc_steps > 0:
        x, eps_init = train_hmc(FLAGS)
        FLAGS.eps = eps_init
    else:
        x = None

    io.log(f'Building model...', rank=rank)
    model, FLAGS = build_model(FLAGS, log_file)
    step_init = tf.Variable(0, dtype=TF_INT)
    ckpt = tf.train.Checkpoint(step=step_init,
                               dynamics=model.dynamics,
                               optimizer=model.optimizer)
    manager = tf.train.CheckpointManager(
        ckpt, directory=ckpt_dir, max_to_keep=5
    )
    if manager.latest_checkpoint:
        io.log(f'Restored from: {manager.latest_checkpoint}', rank=rank)
        ckpt.restore(manager.latest_checkpoint)
        step_init = ckpt.step
    else:
        io.log(f'No existing checkpoints found. Starting from scratch.',
               rank=rank)

    io.log(f'Training model!', rank=rank)
    outputs, data_strs = train_model(model, ckpt, manager,
                                     step_init=step_init, x=x)
    if is_chief:
        io.save(model, train_dir, outputs, data_strs)
        plot_data(outputs, train_dir, FLAGS)

    io.log(f'Done training model.', rank=rank)
    io.log(80 * '=', rank=rank)

    return model, outputs, FLAGS


def run(FLAGS, model, beta, run_steps, x=None):
    """Run inference."""
    is_chief = hvd.rank() == 0 if FLAGS.horovod else not FLAGS.horovod
    if not is_chief:
        return None, None

    runs_dir = os.path.join(FLAGS.log_dir, 'inference')
    io.check_else_make_dir(runs_dir)
    if x is None:
        x = tf.random.uniform(shape=model.input_shape,
                              minval=-PI, maxval=PI)
        x = tf.cast(x, dtype=TF_FLOAT)

    outputs, data_strs = run_model(model, beta, run_steps, x)
    if is_chief:
        run_dir = os.path.join(runs_dir, f'run_{io.get_run_num(runs_dir)}')
        io.save_inference(run_dir, outputs, data_strs)
        plot_data(outputs, run_dir, FLAGS)

    return model, outputs


def train_hmc(FLAGS):
    """Main method for training HMC model."""
    is_chief = hvd.rank() == 0 if FLAGS.horovod else not FLAGS.horovod
    rank = hvd.rank() if FLAGS.horovod else 0
    HFLAGS = AttrDict(dict(FLAGS))
    HFLAGS.dropout_prob = 0.
    HFLAGS.hmc = True
    HFLAGS.save_train_data = True
    HFLAGS.train_steps = HFLAGS.pop('hmc_steps')
    HFLAGS.lr_decay_steps = HFLAGS.train_steps // 4
    HFLAGS.logging_steps = HFLAGS.train_steps // 20
    HFLAGS.beta_final = HFLAGS.beta_init
    HFLAGS.fixed_beta = True
    HFLAGS.no_summaries = True

    train_dir = os.path.join(HFLAGS.log_dir, 'training_hmc')
    ckpt_dir = os.path.join(train_dir, 'checkpoints')
    io.check_else_make_dir([train_dir, ckpt_dir])
    io.save_params(dict(HFLAGS), train_dir, 'FLAGS', rank=rank)

    model, HFLAGS = build_model(HFLAGS)
    step_init = tf.Variable(0, dtype=TF_INT)
    ckpt = tf.train.Checkpoint(step=step_init,
                               dynamics=model.dynamics,
                               optimizer=model.optimizer)
    manager = tf.train.CheckpointManager(
        ckpt, directory=ckpt_dir, max_to_keep=3
    )
    if manager.latest_checkpoint:
        io.log(f'Restored from: {manager.latest_checkpoint}', rank=rank)
        ckpt.restore(manager.latest_checkpoint)
        step_init = ckpt.step

    outputs, data_strs = train_model(model, ckpt, manager,
                                     step_init=step_init, x=None)
    if is_chief:
        io.save(model, train_dir, outputs, data_strs)
        plot_data(outputs, train_dir, HFLAGS)

    x_out = outputs['x']
    eps_out = model.dynamics.eps.numpy()
    io.log(f'Done with HMC start.', rank=rank)
    io.log(80 * '=', rank=rank)

    return x_out, eps_out


def train_model(model, ckpt, manager, step_init=None, x=None):
    """Train model."""
    is_chief = hvd.rank() == 0 if model.using_hvd else not model.using_hvd
    rank = hvd.rank() if model.using_hvd else 0

    if x is None:
        x = tf.random.uniform(shape=model.input_shape,
                              minval=-PI, maxval=PI)
        x = tf.cast(x, dtype=TF_FLOAT)

    _, q_new = model.calc_observables(x, model.beta_init)

    px_arr = []
    dq_arr = []
    loss_arr = []
    data_strs = [HEADER]
    charges_arr = [q_new.numpy()]
    train_steps = np.arange(model.train_steps)
    step = int(step_init.numpy())
    betas = model.betas[step:]
    steps = train_steps[step:]

    if model.compile:
        train_step = tf.function(model.train_step, experimental_compile=True)
    else:
        train_step = model.train_step

    io.log(HEADER, rank=rank)
    for step, beta in zip(steps, betas):
        t0 = time.time()
        x = tf.reshape(x, model.input_shape)
        loss, x, px, sld = train_step(x, beta, step == 0)
        #  loss, x, px, sld = model.train_step(x, beta, step == 0)
        x = tf.reshape(x, model.lattice_shape)
        dt = time.time() - t0

        q_old = q_new
        plaqs_err, q_new = model.calc_observables(x, beta)
        dq = tf.math.abs(q_new - q_old)

        data_str = (
            f"{step:>6g}/{model.train_steps:<6g} "
            f"{dt:^11.4g} "
            f"{loss.numpy():^11.4g} "
            f"{np.mean(px.numpy()):^11.4g} "
            f"{model.dynamics.eps.numpy():^11.4g} "
            f"{beta:^11.4g} "
            f"{np.mean(sld.numpy()):^11.4g} "
            f"{np.mean(dq.numpy()):^11.4g} "
            f"{np.mean(plaqs_err.numpy()):^11.4g} "
        )
        #  if step == 10:
        #      try:
        #          tf.profiler.experimental.start(log_dir)
        #      except AttributeError:
        #          pass

        #  if step == 20:
        #      try:
        #          tf.profiler.experimental.stop(log_dir)
        #      except AttributeError:
        #          pass

        if step % model.print_steps == 0:
            io.log(data_str, rank=rank)
            data_strs.append(data_str)

        if model.save_train_data and step % model.log_steps == 0:
            px_arr.append(px.numpy())
            dq_arr.append(dq.numpy())
            loss_arr.append(loss.numpy())
            charges_arr.append(q_new.numpy())

        if is_chief and step % model.save_steps == 0 and ckpt is not None:
            ckpt.step.assign(step)
            manager.save()

        if step % 100 == 0:
            io.log(HEADER, rank=rank)

    if is_chief and ckpt is not None and manager is not None:
        ckpt.step.assign(step)
        manager.save()

    outputs = {
        'px': np.array(px_arr),
        'dq': np.array(dq_arr),
        'loss_arr': np.array(loss_arr),
        'charges_arr': np.array(charges_arr),
        'x': tf.reshape(x, model.input_shape),
    }

    return outputs, data_strs


def run_model(model, beta, run_steps, x=None):
    """Run inference on trained model."""
    is_chief = hvd.rank() == 0 if model.using_hvd else not model.using_hvd
    if not is_chief:
        return None, None

    if x is None:
        x = tf.random.uniform(shape=model.input_shape,
                              minval=-PI, maxval=PI)
        x = tf.cast(x, dtype=TF_FLOAT)

    _, q_new = model.calc_observables(x, model.beta_init)

    px_arr = []
    dq_arr = []
    data_strs = [RUN_HEADER]
    charges_arr = [q_new.numpy()]

    if model.compile:
        run_step = tf.function(model.run_step, experimental_compile=True)
    else:
        run_step = model.run_step

    io.log(RUN_SEP)
    io.log(f'Running inference on trained model with:')
    io.log(f'  beta: {beta}')
    io.log(f'  dynamics.eps: {model.dynamics.eps.numpy():.4g}')
    io.log(f'  net_weights: {model.dynamics.config.net_weights}')
    io.log(RUN_SEP)
    io.log(RUN_HEADER)
    for step in np.arange(run_steps):
        t0 = time.time()
        x = tf.reshape(x, model.input_shape)
        #  states, px, sld_states = model.run_step(x, beta)
        states, px, sld_states = run_step(x, beta)
        x = states.out.x
        sld = sld_states.out
        x = tf.reshape(x, model.lattice_shape)
        dt = time.time() - t0

        q_old = q_new
        plaqs_err, q_new = model.calc_observables(x, beta)
        dq = tf.math.abs(q_new - q_old)

        data_str = (
            f"{step:>6g}/{run_steps:<6g} "
            f"{dt:^11.4g} "
            f"{np.mean(px.numpy()):^11.4g} "
            f"{np.mean(sld.numpy()):^11.4g} "
            f"{np.mean(dq.numpy()):^11.4g} "
            f"{np.mean(plaqs_err.numpy()):^11.4g} "
        )

        if step % model.print_steps == 0:
            io.log(data_str)
            data_strs.append(data_str)

        if model.save_run_data:
            px_arr.append(px.numpy())
            dq_arr.append(dq.numpy())
            charges_arr.append(q_new.numpy())

        if step % 100 == 0:
            io.log(RUN_HEADER)

    outputs = {
        'px': np.array(px_arr),
        'dq': np.array(dq_arr),
        'charges_arr': np.array(charges_arr),
        'x': tf.reshape(x, model.input_shape),
    }

    data_strs.append(RUN_HEADER)

    return outputs, data_strs
