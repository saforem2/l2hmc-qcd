"""
file_io.py
"""
import os
import numpy as np
import datetime

import utils.file_io as io

from config import (NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC, NetWeights,
                    PI, PROJECT_DIR)


def get_run_num(run_dir):
    dirnames = [i for i in os.listdir(run_dir) if i.startwsith('run_')]
    if len(dirnames) == 0:
        return 1

    return sorted([int(i.split('_')) for i in dirnames])[-1] + 1


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


def make_log_dir(FLAGS, model_type=None, log_file=None, eager=True):
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
    #  if tf.executing_eagerly():
    if eager:
        dirs.append('gauge_logs_eager')

    log_dir = os.path.join(*dirs, month_str, run_str)
    io.check_else_make_dir(log_dir)
    if log_file is not None:
        io.write(f'Output saved to: \n\t{log_dir}', log_file, 'a')

    return log_dir


def save(model, train_dir, outputs, data_strs):
    """Save training results."""
    history_file = os.path.join(train_dir, 'training_log.txt')
    with open(history_file, 'w') as f:
        f.write('\n'.join(data_strs))

    if not model.dynamics_config.hmc:
        xnets = model.dynamics.xnets
        vnets = model.dynamics.vnets
        wdir = os.path.join(train_dir, 'dynamics_weights')
        io.check_else_make_dir(wdir)
        if model.separate_networks:
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

    if model.save_train_data:
        outputs_dir = os.path.join(train_dir, 'outputs')
        io.check_else_make_dir(outputs_dir)
        for key, val in outputs.items():
            out_file = os.path.join(outputs_dir, f'{key}.z')
            io.savez(np.array(val), out_file, key)


def save_inference(model, run_dir, outputs, data_strs):
    """Save inference data."""
    io.check_else_make_dir(run_dir)
    history_file = os.path.join(run_dir, 'inference_log.txt')
    with open(history_file, 'w') as f:
        f.write('\n'.join(data_strs))

    outputs_dir = os.path.join(run_dir, 'outputs')
    io.check_else_make_dir(outputs_dir)
    for key, val in outputs.items():
        out_file = os.path.join(outputs_dir, f'{key}.z')
        io.savez(np.array(val), out_file, key)
