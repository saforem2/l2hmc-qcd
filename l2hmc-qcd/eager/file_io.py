"""
file_io.py
"""
import os
import time
import pickle
import datetime

import joblib
import numpy as np

import utils.file_io as io

from config import (NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC, NetWeights, PI,
                    PROJECT_DIR)

# pylint:disable=invalid-name


def log(s, nl=True, rank=0):
    """Print string `s` to stdout if and only if hvd.rank() == 0."""
    if rank != 0:
        return

    print(s, end='\n' if nl else ' ')


def write(s, f, mode='a', nl=True, rank=0):
    """Write string `s` to file `f` if and only if hvd.rank() == 0."""
    if rank != 0:
        return
    with open(f, mode) as ff:
        ff.write(s + '\n' if nl else ' ')


def log_and_write(s, f, rank=0, mode='a', nl=True):
    """Print string `s` to std out and also write to file `f`."""
    log(s, nl, rank=rank)
    write(s, f, mode=mode, nl=nl, rank=rank)


def check_else_make_dir(d, rank=0):
    """If directory `d` doesn't exist, it is created.

    Args:
        d (str): Location where directory should be created if it doesn't
            already exist.
    """
    if rank != 0:
        return

    if isinstance(d, (list, np.ndarray)):
        for i in d:
            check_else_make_dir(i)
    else:
        if not os.path.isdir(d):
            log(f"Creating directory: {d}", rank=rank)
            os.makedirs(d, exist_ok=True)


def save_params(params, out_dir, name=None, rank=0):
    """save params (dict) to `out_dir`, as both `.z` and `.txt` files."""
    if rank != 0:
        return

    check_else_make_dir(out_dir, rank=rank)
    if name is None:
        name = 'params'
    params_txt_file = os.path.join(out_dir, f'{name}.txt')
    zfile = os.path.join(out_dir, f'{name}.z')
    with open(params_txt_file, 'w') as f:
        for key, val in params.items():
            f.write(f"{key}: {val}\n")
    savez(params, zfile, name=name, rank=rank)


def savez(obj, fpath, name=None, rank=0):
    """Save `obj` to compressed `.z` file at `fpath`."""
    if rank != 0:
        return

    if not fpath.endswith('.z'):
        fpath += '.z'

    if name is not None:
        log(f'Saving {name} to {fpath}.', rank=rank)

    joblib.dump(obj, fpath)


def change_extension(fpath, ext):
    """Change extension of `fpath` to `.ext`."""
    tmp = fpath.split('/')
    out_file = tmp[-1]
    fname, _ = out_file.split('.')
    new_fpath = os.path.join('/'.join(tmp[:-1]), f'{fname}.{ext}')

    return new_fpath


def loadz(fpath):
    """Load from `fpath` using `joblib.load`."""
    try:
        obj = joblib.load(fpath)
    except FileNotFoundError:
        fpath_pkl = change_extension(fpath, 'pkl')
        obj = load_pkl(fpath_pkl)

    return obj


def load_pkl(fpath):
    """Load from `fpath` using `pickle.load`."""
    with open(fpath, 'rb') as f:
        data = pickle.load(f)

    return data


def timeit(method):
    """Timing decorator."""
    def timed(*args, **kwargs):
        """Function to be timed."""
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()

        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = int((end_time - start_time) * 1000)
        else:
            log(80 * '-')
            log(f'`{method.__name__}` took: {(end_time - start_time):.4g}s')
            log(80 * '-')
        return result
    return timed


def get_run_num(run_dir):
    """Get the integer label for naming `run_dir`."""
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
    check_else_make_dir(log_dir)
    if log_file is not None:
        write(f'Output saved to: \n\t{log_dir}', log_file, 'a')

    return log_dir


def save(model, train_dir, outputs, data_strs, rank=0):
    """Save training results."""
    history_file = os.path.join(train_dir, 'training_log.txt')
    with open(history_file, 'w') as f:
        f.write('\n'.join(data_strs))

    if not model.dynamics_config.hmc:
        xnets = model.dynamics.xnets
        vnets = model.dynamics.vnets
        wdir = os.path.join(train_dir, 'dynamics_weights')
        check_else_make_dir(wdir)
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
            savez(xnet_weights, xweights_file, 'xnet_weights')
            savez(vnet_weights, vweights_file, 'vnet_weights')
        else:
            xfpath = os.path.join(wdir, 'xnet_weights.z')
            vfpath = os.path.join(wdir, 'vnet_weights.z')
            xnets.save_layer_weights(out_file=xfpath)
            vnets.save_layer_weights(out_file=vfpath)

    if model.save_train_data:
        outputs_dir = os.path.join(train_dir, 'outputs')
        check_else_make_dir(outputs_dir)
        for key, val in outputs.items():
            out_file = os.path.join(outputs_dir, f'{key}.z')
            savez(np.array(val), out_file, key)


def save_inference(run_dir, outputs, data_strs):
    """Save inference data."""
    check_else_make_dir(run_dir)
    history_file = os.path.join(run_dir, 'inference_log.txt')
    with open(history_file, 'w') as f:
        f.write('\n'.join(data_strs))

    outputs_dir = os.path.join(run_dir, 'outputs')
    check_else_make_dir(outputs_dir)
    for key, val in outputs.items():
        out_file = os.path.join(outputs_dir, f'{key}.z')
        savez(np.array(val), out_file, key)


def get_subdirs(root_dir):
    """Returns all subdirectories in `root_dir`."""
    subdirs = [
        os.path.join(root_dir, i)
        for i in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, i))
    ]
    return subdirs
