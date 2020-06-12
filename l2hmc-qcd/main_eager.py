"""
main_eager.py

Train 2D U(1) model using eager execution in tensorflow.
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

try:
    import horovod.tensorflow as hvd

    hvd.init()
    config = tf.ConfigProto()  # pylint: disable=invalid-name
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    try:
        tf.compat.v1.enable_eager_execution(config=config)
    except AttributeError:
        pass

except ImportError:
    try:
        tf.compat.v1.enable_eager_execution()
    except AttributeError:
        pass

import os
import xarray as xr

import numpy as np
import seaborn as sns
import arviz as az
import matplotlib.pyplot as plt
from plotters.data_utils import therm_arr

import utils.file_io as io

from config import (NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC, NetWeights, NP_FLOAT,
                    PI, PROJECT_DIR, TF_FLOAT)
from network import NetworkConfig
from utils.attr_dict import AttrDict
from dynamics.dynamics import DynamicsConfig
from utils.parse_args import parse_args

#  from models.gauge_model_eager import GaugeModel
import datetime

from models.gauge_model_new import GaugeModel

sns.set_palette('bright')


def get_run_str(FLAGS):
    """Parse FLAGS and create unique `run_str` for `log_dir`."""
    run_str = f'L{FLAGS.space_size}'
    run_str += f'_b{FLAGS.batch_size}_lf{FLAGS.num_steps}'
    if FLAGS.network_type != 'GaugeNetwork':
        run_str += f'_{FLAGS.network_type}'

    if FLAGS.charge_weight > 0:
        run_str += f'_qw{FLAGS.charge_weight}'.replace('.', '')

    if FLAGS.plaq_weight > 0:
        run_str += f'_pw{FLAGS.plaq_weight}'.replace('.', '')

    if FLAGS.dropout_prob > 0:
        run_str += f'_dp{FLAGS.dropout_prob}'.replace('.', '')

    if FLAGS.eps_fixed:
        run_str += f'_eps_fixed'

    if FLAGS.clip_value > 0:
        run_str += f'_clip{FLAGS.clip_value}'.replace('.', '')

    return run_str


def make_log_dir(FLAGS, model_type=None, log_file=None):
    """Automatically create and name `log_dir` to save model data to.

    The created directory will be located in `logs/YYYY_M_D/`, and will have
    the format (without `_qw{QW}` if running generic HMC):

        `lattice{LX}_batch{NS}_lf{LF}_eps{SS}_qw{QW}`

    Returns:
        FLAGS, with FLAGS.log_dir being equal to the newly created log_dir.

    NOTE: If log_dir does not already exist, it is created.
    """
    model_type = 'GaugeModel' if model_type is None else model_type
    run_str = get_run_str(FLAGS)
    if FLAGS.train_steps < 5000:
        run_str = f'DEBUG_{run_str}'

    now = datetime.datetime.now()
    month_str = now.strftime('%Y_%m')
    dstr = now.strftime('%Y-%m-%d-%H-%M')
    run_str = f'{run_str}-{dstr}'

    root_dir = os.path.dirname(PROJECT_DIR)
    dirs = [root_dir]
    if tf.executing_eagerly():
        dirs.append('gauge_logs_eager')

    log_dir = os.path.join(*dirs, month_str, run_str)
    io.check_else_make_dir(log_dir)
    if log_file is not None:
        io.write(f'Output saved to: \n\t{log_dir}', log_file, 'a')

    return log_dir


def plot_data(outputs, base_dir, FLAGS, thermalize=False):
    out_dir = os.path.join(base_dir, 'plots')
    io.check_else_make_dir(out_dir)

    data = {}
    for key, val in outputs.items():
        if key == 'x':
            continue
        if key == 'loss_arr':
            fig, ax = plt.subplots()
            steps = FLAGS.logging_steps * np.arange(len(np.array(val)))
            ax.plot(steps, np.array(val), ls='', marker='x', label='loss')
            ax.legend(loc='best')
            ax.set_xlabel('Train step')
            out_file = os.path.join(out_dir, 'loss.png')
            io.log(f'Saving figure to: {out_file}')
            fig.savefig(out_file, dpi=400, bbox_inches='tight')
        else:
            fig, ax = plt.subplots()
            arr = np.array(val)
            chains = np.arange(arr.shape[1])
            steps = FLAGS.logging_steps * np.arange(arr.shape[0])
            if thermalize:
                arr, steps = therm_arr(arr, therm_frac=0.33)
                data[key] = (steps, arr)

            else:
                data[key] = (steps, arr)

            data_arr = xr.DataArray(arr.T, dims=['chain', 'draw'],
                                    coords=[chains, steps])
            az.plot_trace({key: data_arr})
            out_file = os.path.join(out_dir, f'{key}.png')
            io.log(f'Saving figure to: {out_file}.')
            plt.savefig(out_file, dpi=400, bbox_inches='tight')

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    for idx, (key, val) in enumerate(data.items()):
        steps, arr = val
        avg_val = np.mean(arr, axis=1)
        label = r"$\langle$" + f' {key} ' + r"$\rangle$"
        fig, ax = plt.subplots()
        ax.plot(steps, avg_val, color=colors[idx], label=label)
        ax.legend(loc='best')
        ax.set_xlabel('Train step')
        out_file = os.path.join(out_dir, f'{key}_avg.png')
        io.log(f'Saving figure to: {out_file}.')
        fig.savefig(out_file, dpi=400, bbox_inches='tight')


def build_model(FLAGS, save_params=True, log_file=None):
    """Build model using parameters from FLAGS."""
    IS_CHIEF = (  # pylint:disable=invalid-name
        not FLAGS.horovod
        or FLAGS.horovod and hvd.rank() == 0
    )
    if FLAGS.log_dir is None:
        FLAGS.log_dir = make_log_dir(FLAGS, 'GaugeModel', log_file)

    net_weights = NET_WEIGHTS_HMC if FLAGS.hmc else NET_WEIGHTS_L2HMC
    xdim = FLAGS.time_size * FLAGS.space_size * 2
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

    ckpt_dir = None
    training_dir = os.path.join(FLAGS.log_dir, 'training')
    io.check_else_make_dir(training_dir)
    if IS_CHIEF:
        ckpt_dir = os.path.join(training_dir, 'checkpoints')
        io.check_else_make_dir(ckpt_dir)
        if save_params:
            io.save_params(dict(FLAGS), training_dir, 'FLAGS')

    if FLAGS.horovod:
        io.log(f'Number of {hvd.size()} GPUs')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(
                gpus[hvd.local_rank()], 'GPU'
            )

    model = GaugeModel(FLAGS, lattice_shape, config, net_config)

    return model, FLAGS, ckpt_dir


def get_run_num(run_dir):
    dirnames = [i for i in os.listdir(run_dir) if i.startwsith('run_')]
    if len(dirnames) == 0:
        return 1

    return sorted([int(i.split('_')) for i in dirnames])[-1] + 1


def run_inference_hmc(FLAGS):
    IS_CHIEF = (  # pylint:disable=invalid-name
        not FLAGS.horovod
        or FLAGS.horovod and hvd.rank() == 0
    )

    HFLAGS = AttrDict(dict(FLAGS))
    #  HFLAGS.log_dir = os.path.join(FLAGS.log_dir, 'HMC_START')
    HFLAGS.dropout_prob = 0.
    HFLAGS.hmc = True
    HFLAGS.net_weights = NET_WEIGHTS_HMC

    model, HFLAGS, ckpt_dir = build_model(HFLAGS, save_params=False)
    outputs, data_strs = model.run_eager(HFLAGS.run_steps,
                                         beta=HFLAGS.beta_final,
                                         save_run_data=True,
                                         ckpt_dir=None)
    if IS_CHIEF:
        log_dir = HFLAGS.log_dir
        runs_dir = os.path.join(log_dir, 'inference_HMC')
        io.check_else_make_dir(runs_dir)
        run_dir = os.path.join(runs_dir, f'run_{get_run_num(runs_dir)}')
        io.check_else_make_dir(run_dir)

        history_file = os.path.join(run_dir, 'inference_log.txt')
        with open(history_file, 'w') as f:
            f.write('\n'.join(data_strs))

        run_params = {
            'beta': HFLAGS.beta_final,
            'dynamics.eps': model.dynamics.eps.numpy(),
            'net_weights': model.dynamics.config.net_weights
        }
        io.save_dict(run_params, run_dir, 'run_params')

        outputs_dir = os.path.join(run_dir, 'outputs')
        io.check_else_make_dir(outputs_dir)
        for key, val in outputs.items():
            out_file = os.path.join(outputs_dir, f'{key}.z')
            io.savez(np.array(val), out_file, key)

        plot_data(outputs, run_dir, HFLAGS, thermalize=True)

    return model, outputs


def run_inference(FLAGS, model=None):
    IS_CHIEF = (  # pylint:disable=invalid-name
        not FLAGS.horovod
        or FLAGS.horovod and hvd.rank() == 0
    )

    if model is None:
        log_dir = FLAGS.log_dir
        fpath = os.path.join(FLAGS.log_dir, 'training', 'FLAGS.z')
        FLAGS = AttrDict(dict(io.loadz(fpath)))
        FLAGS.log_dir = log_dir
        model, FLAGS, ckpt_dir = build_model(FLAGS, save_params=False)
    else:
        dirname = os.path.join(FLAGS.log_dir, 'training', 'checkpoints')
        ckpt_dir = dirname if IS_CHIEF else None

    outputs, data_strs = model.run_eager(FLAGS.run_steps,
                                         beta=FLAGS.beta_final,
                                         save_run_data=True,
                                         ckpt_dir=ckpt_dir)

    if IS_CHIEF:
        log_dir = FLAGS.log_dir
        runs_dir = os.path.join(log_dir, 'inference')
        io.check_else_make_dir(runs_dir)
        run_dir = os.path.join(runs_dir, f'run_{get_run_num(runs_dir)}')
        io.check_else_make_dir(run_dir)
        history_file = os.path.join(run_dir, 'inference_log.txt')
        with open(history_file, 'w') as f:
            f.write('\n'.join(data_strs))

        run_params = {
            'beta': FLAGS.beta_final,
            'dynamics.eps': model.dynamics.eps.numpy(),
            'net_weights': model.dynamics.config.net_weights
        }
        io.save_dict(run_params, run_dir, 'run_params')

        outputs_dir = os.path.join(run_dir, 'outputs')
        io.check_else_make_dir(outputs_dir)
        for key, val in outputs.items():
            out_file = os.path.join(outputs_dir, f'{key}.z')
            io.savez(np.array(val), out_file, key)

        plot_data(outputs, run_dir, FLAGS, thermalize=True)

    return model, outputs


def train_hmc(FLAGS):
    """Main method for training HMC model."""
    HFLAGS = AttrDict(dict(FLAGS))
    IS_CHIEF = (
        not HFLAGS.horovod
        or HFLAGS.horovod and hvd.rank() == 0
    )

    #  HFLAGS.log_dir = os.path.join(FLAGS.log_dir)
    HFLAGS.log_dir
    HFLAGS.dropout_prob = 0.
    HFLAGS.hmc = True
    HFLAGS.save_train_data = True
    HFLAGS.train_steps = HFLAGS.pop('hmc_steps')
    HFLAGS.lr_decay_steps = HFLAGS.train_steps // 4
    HFLAGS.logging_steps = HFLAGS.train_steps // 20
    HFLAGS.beta_final = HFLAGS.beta_init
    HFLAGS.fixed_beta = True
    HFLAGS.no_summaries = True

    ckpt_dir = None
    training_dir = os.path.join(HFLAGS.log_dir, 'training_hmc')
    io.check_else_make_dir(training_dir)
    if IS_CHIEF:
        ckpt_dir = os.path.join(training_dir, 'checkpoints')
        io.check_else_make_dir(ckpt_dir)
        io.save_params(dict(HFLAGS), training_dir, 'FLAGS')

    model, HFLAGS, ckpt_dir = build_model(HFLAGS)
    outputs, data_strs = model.train_eager(
        save_train_data=HFLAGS.save_train_data,
        ckpt_dir=None
    )
    if IS_CHIEF:
        train_dir = os.path.join(HFLAGS.log_dir, 'training')
        io.check_else_make_dir(train_dir)
        history_file = os.path.join(train_dir, 'training_log.txt')
        with open(history_file, 'w') as f:
            f.write('\n'.join(data_strs))

        if HFLAGS.save_train_data:
            outputs_dir = os.path.join(HFLAGS.log_dir, 'training', 'outputs')
            io.check_else_make_dir(outputs_dir)
            for key, val in outputs.items():
                out_file = os.path.join(outputs_dir, f'{key}.z')
                io.savez(np.array(val), out_file, key)

            plot_data(outputs, train_dir, HFLAGS)

    x_out = outputs['x']
    eps_out = model.dynamics.eps.numpy()

    return x_out, eps_out


# pylint:disable=redefined-outer-name, invalid-name, too-many-locals
def train(FLAGS, log_file=None):
    """Main method for training model."""
    IS_CHIEF = (  # pylint:disable=invalid-name
        not FLAGS.horovod
        or FLAGS.horovod and hvd.rank() == 0
    )

    if FLAGS.log_dir is None:
        FLAGS.log_dir = make_log_dir(FLAGS, 'GaugeModel', log_file)

    else:
        fpath = os.path.join(FLAGS.log_dir, 'training', 'FLAGS.z')
        FLAGS = AttrDict(dict(io.loadz(fpath)))

    x_init = None
    if FLAGS.hmc_start and FLAGS.hmc_steps > 0:
        x_init, eps_init = train_hmc(FLAGS)
        FLAGS.eps = eps_init

    '''
    callbacks = []

    if FLAGS.horovod:
        # Horovod: broadcast initial variable states from rank 0 to all other
        # processes. This is necessary to ensure consistent initialization of
        # all workers when training is started with random weights or restored
        # from a checkpoint.
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        # Horovod: average metric among workers at the end of every epoch.
        # NOTE: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        callbacks.append(hvd.callbacks.MetricAverageCallback())

        # Horovod: Using `lr= 1.0 * hvd.size()` from the very beginning leads
        # to worse final accuracy. Scale the learning rate `lr = 1.0` --> `lr =
        # 1.0 * hvd.size()` during the first three epochs.
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(
            warmup_epochs=3, initial_lr=FLAGS.lr_init, verbose=1
        ))
    '''

    model, FLAGS, ckpt_dir = build_model(FLAGS, log_file)
    outputs, data_strs = model.train_eager(
        x=x_init,
        save_train_data=FLAGS.save_train_data,
        ckpt_dir=ckpt_dir,
    )

    if IS_CHIEF:
        train_dir = os.path.join(FLAGS.log_dir, 'training')
        io.check_else_make_dir(train_dir)
        history_file = os.path.join(train_dir, 'training_log.txt')
        with open(history_file, 'w') as f:
            f.write('\n'.join(data_strs))

        xnets = model.dynamics.xnets
        vnets = model.dynamics.vnets
        wdir = os.path.join(train_dir, 'dynamics_weights')
        io.check_else_make_dir(wdir)
        if FLAGS.separate_networks:
            iterable = enumerate(zip(xnets, vnets))
            xnet_weights = {}
            vnet_weights = {}
            for idx, (xnet, vnet) in iterable:
                xfpath = os.path.join(wdir, f'xnet{idx}_weights.z')
                vfpath = os.path.join(wdir, f'vnet{idx}_weights.z')
                xweights = xnet.save_layer_weights(out_file=xfpath)
                vweights = vnet.save_layer_weights(out_file=vfpath)
                xnet_weights[f'xnet{idx}'] = xweights
                vnet_weights[f'vnet{idx}'] = vweights

            xweights_file = os.path.join(wdir, 'xnet_weights.z')
            vweights_file = os.path.join(wdir, 'vnet_weights.z')
            io.savez(xnet_weights, xweights_file, 'xnet_weights')
            io.savez(vnet_weights, vweights_file, 'vnet_weights')
        else:
            xfpath = os.path.join(wdir, 'xnet_weights.z')
            vfpath = os.path.join(wdir, 'vnet_weights.z')
            xnets.save_layer_weights(out_file=xfpath)
            vnets.save_layer_weights(out_file=vfpath)

        if FLAGS.save_train_data:
            outputs_dir = os.path.join(train_dir, 'outputs')
            io.check_else_make_dir(outputs_dir)
            for key, val in outputs.items():
                out_file = os.path.join(outputs_dir, f'{key}.z')
                io.savez(np.array(val), out_file, key)

            plot_data(outputs, train_dir, FLAGS)

    return model, outputs


if __name__ == '__main__':
    FLAGS = parse_args()
    FLAGS = AttrDict(FLAGS.__dict__)
    LOG_FILE = os.path.join(os.getcwd(), 'output_dirs.txt')
    if FLAGS.inference and FLAGS.log_dir is not None:
        _, _ = run_inference(FLAGS)
        _, _ = run_inference_hmc(FLAGS)
    else:
        MODEL, OUTPUTS = train(FLAGS, LOG_FILE)
        _, _ = run_inference(FLAGS, model=MODEL)
        _, _ = run_inference_hmc(FLAGS)
    #  if FLAGS.inference and FLAGS.log_dir is not None:
    #      _, _, _ = run_inference(FLAGS)
    #
    #  else:
    #      MODEL, OUTPUTS = train(FLAGS, LOG_FILE)
