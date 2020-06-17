"""
inference_utils.py

Collection of helper methods to use for running inference on trained model.
"""
from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np
import tensorflow as tf

import utils.file_io as io

from utils.attr_dict import AttrDict
from utils.data_utils import therm_arr
from utils.plotting_utils import plot_data, get_title_str_from_params
from config import PI, TF_FLOAT, TF_INT
from models.gauge_model import RUN_HEADER, RUN_SEP
from utils.training_utils import build_model

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

# pylint:disable=too-many-locals,invalid-name


# pylint:disable=invalid-name
def load_and_run(args):
    """Load trained model from checkpoint and run inference."""
    if args.hmc:
        train_dir = os.path.join(args.log_dir, 'training_hmc')
    else:
        train_dir = os.path.join(args.log_dir, 'training')

    ckpt_dir = os.path.join(train_dir, 'checkpoints')
    FLAGS = AttrDict(dict(io.loadz(os.path.join(train_dir, 'FLAGS.z'))))

    model, FLAGS = build_model(FLAGS)

    step_init = tf.Variable(0, dtype=TF_INT)
    ckpt = tf.train.Checkpoint(step=step_init,
                               dynamics=model.dynamics,
                               optimizer=model.optimizer)
    manager = tf.train.CheckpointManager(ckpt,
                                         max_to_keep=5,
                                         directory=ckpt_dir)
    if manager.latest_checkpoint:
        io.log(f'Restored model from: {manager.latest_checkpoint}')
        ckpt.restore(manager.latest_checkpoint)
        step_init = ckpt.step

    if args.eps is None:
        eps_file = os.path.join(args.log_dir, 'eps_final.z')
        if os.path.isfile(eps_file):
            edict = io.loadz(eps_file)
            model.dynamics.eps = edict['eps']

    if args.beta is None:
        train_dir = os.path.join(args.log_dir, 'training')
        l2hmc_flags = AttrDict(
            dict(io.loadz(os.path.join(train_dir, 'FLAGS.z')))
        )
        args.beta = l2hmc_flags.beta_final

    model, outputs = run(FLAGS, model, args.beta, args.run_steps)

    return model, outputs


def run(FLAGS, model, beta, run_steps, x=None):
    """Run inference.

    Returns:
        model (GaugeModel): Trained model
        ouptuts (dict): Dictionary of outputs from inference run.
    """
    is_chief = hvd.rank() == 0 if FLAGS.horovod else not FLAGS.horovod
    if not is_chief:
        return None, None

    if FLAGS.hmc:
        runs_dir = os.path.join(FLAGS.log_dir, 'inference_hmc')
    else:
        runs_dir = os.path.join(FLAGS.log_dir, 'inference')

    io.check_else_make_dir(runs_dir)
    if x is None:
        x = tf.random.uniform(shape=model.input_shape,
                              minval=-PI, maxval=PI)
        x = tf.cast(x, dtype=TF_FLOAT)

    outputs, data_strs = run_model(model, beta, run_steps, x)
    if is_chief:
        run_dir = os.path.join(runs_dir, f'run_{io.get_run_num(runs_dir)}')
        io.check_else_make_dir(run_dir)

        history_file = os.path.join(run_dir, 'inference_log.txt')
        io.flush_data_strs(data_strs, history_file)

        run_params = {
            'beta': beta,
            'eps': model.dynamics.eps.numpy(),
            'run_steps': run_steps,
            'net_weights': model.dynamics_config.net_weights,
            'num_steps': model.dynamics_config.num_steps,
            'input_shape': model.dynamics_config.input_shape,
            'lattice_shape': model.lattice_shape,
        }
        io.save_params(run_params, run_dir, name='run_params')
        io.save_inference(run_dir, outputs, data_strs)

        plot_data(outputs, run_dir, FLAGS, thermalize=True, params=run_params)

    return model, outputs


def run_model(model, beta, run_steps, x=None):
    """Run inference on trained `model`.

    Returns:
        outputs (dict): Dictionary of outputs.
        data_strs (lsit): List of strings containing inference log.
    """
    is_chief = hvd.rank() == 0 if model.using_hvd else not model.using_hvd
    if not is_chief:
        return None, None

    if x is None:
        x = tf.random.uniform(shape=model.input_shape,
                              minval=-PI, maxval=PI)
        x = tf.cast(x, dtype=TF_FLOAT)

    _, q_new = model.calc_observables(x, beta, use_sin=False)
    plaqs, q_new = model.calc_observables(x, beta, use_sin=False)
    px_arr = []
    dq_arr = []
    data_strs = [RUN_HEADER]
    charges_arr = [q_new.numpy()]

    plaqs_arr = [plaqs.numpy()]
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
        plaqs_err, q_new = model.calc_observables(x, beta, use_sin=False)
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
            plaqs_arr.append(plaqs_err.numpy())

        if step % 100 == 0:
            io.log(RUN_HEADER)

    outputs = {
        'px': np.array(px_arr),
        'dq': np.array(dq_arr),
        'charges_arr': np.array(charges_arr),
        'x': tf.reshape(x, model.input_shape),
        'plaqs_err': np.array(plaqs_arr),
    }

    return outputs, data_strs
