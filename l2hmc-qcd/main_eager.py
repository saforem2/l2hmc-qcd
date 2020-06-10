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
    tf.compat.v1.enable_eager_execution(config=config)

except ImportError:
    tf.compat.v1.enable_eager_execution()

import os

import numpy as np
import seaborn as sns
import arviz as az
import matplotlib.pyplot as plt

import utils.file_io as io

from config import (NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC, NetWeights, NP_FLOAT,
                    PI, PROJECT_DIR, TF_FLOAT, TWO_PI)
from network import NetworkConfig
from lattice.lattice import GaugeLattice
from utils.attr_dict import AttrDict
from dynamics.dynamics import Dynamics, DynamicsConfig
from utils.parse_args import parse_args

#  from models.gauge_model_eager import GaugeModel
import datetime

from models.gauge_model_new import GaugeModel

sns.set_palette('bright')

def parse_flags(FLAGS):
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
    run_str = parse_flags(FLAGS)
    if FLAGS.train_steps < 5000:
        run_str = f'DEBUG_{run_str}'

    now = datetime.datetime.now()
    month_str = now.strftime('%Y_%m')
    #  day_str = now.strftime('%Y_%m_%d')
    #  hour_str = now.strftime('%H%M')
    dstr = now.strftime('%Y-%m-%d-%H-%M')
    run_str = f'{run_str}-{dstr}'

    root_dir = os.path.dirname(PROJECT_DIR)
    dirs = [root_dir]
    if tf.executing_eagerly():
        dirs.append('eager_logs')
        #  dirs = [root_dir, 'eager_logs', month_str, run_str]
        #  log_dir = os.path.join(root_dir, 'eager_logs', month_str, run_str)
    #  else:
    #      log_dir = os.path.join(root_dir, month_str, run_str)

    #  dirs.append(month_str)
    #  log_dir = os.path.join(*dirs.append(run_str))
    log_dir = os.path.join(*dirs, month_str, run_str)
    #  if os.path.isdir(log_dir):
    #      run_str = f'{run_str}-{hour_str}'
    #      log_dir = os.path.join(os.path.dirname(log_dir), run_str)
    #      #  log_dir_ = os.path.join(*dirs.append(run_str))

    io.check_else_make_dir(log_dir)
    if log_file is not None:
        io.write(f'Output saved to: \n\t{log_dir}', log_file, 'a')

    return log_dir


def plot_train_data(outputs, training_dir):
    out_dir = os.path.join(training_dir, 'train_plots')
    io.check_else_make_dir(out_dir)
    for key, val in outputs.items():
        if key == 'loss_arr':
            fig, ax = plt.subplots()
            ax.plot(np.array(val), ls='', marker='x', label='loss')
            ax.legend(loc='best')
            ax.set_xlabel('Train step')
            fig.savefig(os.path.join(out_dir, 'loss.png'),
                        dpi=400, bbox_inches='tight')
        else:
            fig, ax = plt.subplots()
            val = np.array(val)
            az.plot_trace({key: val.T})
            plt.savefig(os.path.join(out_dir, f'{key}.png'),
                        dpi=400, bbox_inches='tight')


def train(FLAGS, log_file=None):
    """Main method for training model."""
    net_weights = NET_WEIGHTS_HMC if FLAGS.hmc else NET_WEIGHTS_L2HMC

    IS_CHIEF = (
        not FLAGS.horovod
        or FLAGS.horovod and hvd.rank() == 0
    )

    if FLAGS.horovod:
        io.log(f'Number of {hvd.size()} GPUs')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(
                gpus[hvd.local_rank()], 'GPU'
            )

    xdim = FLAGS.time_size * FLAGS.space_size * 2
    input_shape = (FLAGS.batch_size, xdim)
    lattice_shape = (FLAGS.batch_size, FLAGS.time_size, FLAGS.space_size, 2)
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

    #  ckpt_prefix = None
    if IS_CHIEF:
        if FLAGS.log_dir is None:
            log_dir = make_log_dir(FLAGS, 'GaugeModel', log_file)
        else:
            log_dir = FLAGS.log_dir

        training_dir = os.path.join(log_dir, 'training')
        ckpt_dir = os.path.join(training_dir, 'checkpoints')
        #  ckpt_prefix = os.path.join(ckpt_dir, 'ckpt')
        io.check_else_make_dir(ckpt_dir)
        io.save_params(dict(FLAGS), training_dir, 'FLAGS')

        #  xnet_ckpt = tf.train.Checkpoint(model=model.dynamics.xnets,
        #                                  optimizer=model.optimizer)
        #  vnet_ckpt = tf.train.Checkpoint(model=model.dynamics.vnets,
        #                                  optimizer=model.optimizer)
        #  callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_file))

    outputs, data_strs = model.train_eager(FLAGS.save_train_data,
                                           ckpt_dir=ckpt_dir)

    if IS_CHIEF:
        history_file = os.path.join(training_dir, 'training_log.txt')
        with open(history_file, 'w') as f:
            f.write('\n'.join(data_strs))

        if FLAGS.save_train_data:
            outputs_dir = os.path.join(log_dir, 'training', 'outputs')
            io.check_else_make_dir(outputs_dir)
            for key, val in outputs.items():
                out_file = os.path.join(outputs_dir, f'{key}.z')
                io.savez(np.array(val), out_file, key)

            plot_train_data(outputs, training_dir)

    return model, outputs


if __name__ == '__main__':
    FLAGS = parse_args()
    FLAGS = AttrDict(FLAGS.__dict__)
    LOG_FILE = os.path.join(os.getcwd(), 'output_dirs.txt')
    MODEL, OUTPUTS = train(FLAGS, LOG_FILE)
