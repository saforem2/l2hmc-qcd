"""
file_io.py
"""
import os
import time
import pickle
import datetime

import joblib
import numpy as np

from config import PROJECT_DIR

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


def flush_data_strs(data_strs, out_file, rank=0, mode='a'):
    """Dump `data_strs` to `out_file` and return new, empty list."""
    if rank != 0:
        return []

    with open(out_file, mode) as f:
        for s in data_strs:
            f.write(f'{s}\n')

    return []


def save_params(params, out_dir, name=None, rank=0):
    """save params(dict) to `out_dir`, as both `.z` and `.txt` files."""
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


def print_args(args, rank=0):
    """Print out parsed arguments."""
    log(80 * '=' + '\n' + 'Parsed args:\n', rank=rank)
    for key, val in args.items():
        log(f' {key}: {val}\n', rank=rank)
    log(80 * '=', rank=rank)


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
    dirnames = [
        i for i in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, i))
    ]
    if len(dirnames) == 0:
        return 1

    return sorted([int(i.split('_')[-1]) for i in dirnames])[-1] + 1


def get_run_dir_fstr(FLAGS):
    """Parse FLAGS and create unique fstr for `run_dir`."""
    eps = FLAGS.get('eps', None)
    hmc = FLAGS.get('hmc', False)
    beta = FLAGS.get('beta', None)
    num_steps = FLAGS.get('num_steps', None)
    lattice_shape = FLAGS.get('lattice_shape', None)

    if beta is None:
        beta = FLAGS.get('beta_final', None)

    fstr = ''
    if hmc:
        fstr += f'HMC_'
    if beta is not None:
        fstr += f'beta{beta:.3g}'.replace('.', '')
    if num_steps is not None:
        fstr += f'_lf{num_steps}'
    if eps is not None:
        fstr += f'_eps{eps:.3g}'.replace('.', '')
    if lattice_shape is not None:
        fstr += f'_b{lattice_shape[0]}'

    return fstr


# pylint:disable=too-many-branches, too-many-locals
def get_log_dir_fstr(FLAGS):
    """Parse FLAGS and create unique fstr for `log_dir`."""
    hmc = FLAGS.get('hmc', False)
    batch_size = FLAGS.get('batch_size', None)
    num_steps = FLAGS.get('num_steps', None)
    beta = FLAGS.get('beta', None)
    eps = FLAGS.get('eps', None)
    space_size = FLAGS.get('space_size', None)
    time_size = FLAGS.get('time_size', None)
    lattice_shape = FLAGS.get('lattice_shape', None)
    train_steps = FLAGS.get('train_steps', int(1e3))
    network_type = FLAGS.get('network_type', 'GaugeNetwork')
    charge_weight = FLAGS.get('charge_weight', 0.)
    plaq_weight = FLAGS.get('plaq_weight', 0.)
    eps_fixed = FLAGS.get('eps_fixed', False)
    dropout_prob = FLAGS.get('dropout_prob', 0.)
    clip_value = FLAGS.get('clip_value', 0.)
    separate_networks = FLAGS.get('separate_networks', False)
    using_ncp = FLAGS.get('use_ncp', False)

    fstr = ''

    if hmc:
        fstr += 'HMC_'
        eps_fixed = True
    else:
        if train_steps is not None and train_steps < 5000:
            fstr += 'DEBUG_'

    if lattice_shape is not None:
        fstr += f'L{lattice_shape[1]}_b{lattice_shape[0]}'

    else:
        if space_size is not None:
            fstr += f'L{time_size}'

        if batch_size is not None:
            fstr += f'_b{batch_size}'

    if num_steps is not None:
        fstr += f'_lf{num_steps}'

    if charge_weight > 0:
        fstr += f'_qw{charge_weight}'.replace('.', '')

    if plaq_weight > 0:
        fstr += f'_pw{plaq_weight}'.replace('.', '')

    if dropout_prob > 0:
        fstr += f'_dp{dropout_prob}'.replace('.', '')

    if eps_fixed:
        fstr += f'_eps{eps:.3g}'.replace('.', '')

    if beta is not None:
        fstr += f'_beta{beta:.3g}'.replace('.', '')

    if clip_value > 0:
        fstr += f'_clip{clip_value}'.replace('.', '')

    if network_type != 'GaugeNetwork':
        fstr += f'_{network_type}'

    if separate_networks:
        fstr += f'_sepNets'

    if using_ncp:
        fstr += f'_NCProj'

    return fstr


def make_log_dir(FLAGS, model_type=None, log_file=None,
                 base_dir=None, eager=True, rank=0):
    """Automatically create and name `log_dir` to save model data to.

    The created directory will be located in `logs/YYYY_M_D /`, and will have
    the format(without `_qw{QW}` if running generic HMC):

        `lattice{LX}_batch{NS}_lf{LF}_eps{SS}_qw{QW}`

    Returns:
        FLAGS, with FLAGS.log_dir being equal to the newly created log_dir.

    NOTE: If log_dir does not already exist, it is created.
    """
    model_type = 'GaugeModel' if model_type is None else model_type
    fstr = get_log_dir_fstr(FLAGS)

    now = datetime.datetime.now()
    month_str = now.strftime('%Y_%m')
    dstr = now.strftime('%Y-%m-%d-%H%M%S')
    run_str = f'{fstr}-{dstr}'

    root_dir = os.path.dirname(PROJECT_DIR)
    dirs = [root_dir]
    if eager:
        dirs.append('gauge_logs_eager')

    if fstr.startswith('DEBUG'):
        dirs.append('test')

    log_dir = os.path.join(*dirs, month_str, run_str)
    if os.path.isdir(log_dir):
        log(f'Existing directory found with the same name!')
        log(f'Modifying date string to include seconds.')
        dstr = now.strftime('%Y-%m-%d-%H%M%S')
        run_str = f'{fstr}-{dstr}'
        log_dir = os.path.join(*dirs, month_str, run_str)

    if rank == 0:
        check_else_make_dir(log_dir)
        if log_file is not None:
            write(f'{log_dir}', log_file, 'a')

    return log_dir


def make_run_dir(FLAGS, base_dir):
    """Automatically create `run_dir` for storing inference data."""
    fstr = get_run_dir_fstr(FLAGS)
    now = datetime.datetime.now()
    dstr = now.strftime('%Y-%m-%d-%H%M')
    run_str = f'{fstr}-{dstr}'
    run_dir = os.path.join(base_dir, run_str)
    if os.path.isdir(run_dir):
        log(f'Existing directory found with the same name!')
        log(f'Modifying date string to include seconds.')
        dstr = now.strftime('%Y-%m-%d-%H%M%S')
        run_str = f'{fstr}-{dstr}'
        run_dir = os.path.join(base_dir, run_str)

    check_else_make_dir(run_dir)

    return run_dir


def save_network_weights(dynamics, train_dir, rank=0):
    """Save network weights as dictionary to `.z` files."""
    xnets = dynamics.xnets
    vnets = dynamics.vnets
    wdir = os.path.join(train_dir, 'dynamics_weights')
    check_else_make_dir(wdir)
    if dynamics.config.separate_networks:
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
        savez(xnet_weights, xweights_file, 'xnet_weights', rank=rank)
        savez(vnet_weights, vweights_file, 'vnet_weights', rank=rank)
    else:
        xfpath = os.path.join(wdir, 'xnet_weights.z')
        vfpath = os.path.join(wdir, 'vnet_weights.z')
        xnets.save_layer_weights(out_file=xfpath)
        vnets.save_layer_weights(out_file=vfpath)


def save(dynamics, train_data, train_dir, rank=0):
    """Save training results."""
    if rank != 0:
        return

    check_else_make_dir(train_dir)

    if not dynamics.config.hmc:
        save_network_weights(dynamics, train_dir, rank=rank)

    if dynamics.save_train_data:
        output_dir = os.path.join(train_dir, 'outputs')
        train_data.save_data(output_dir)


def save_inference(run_dir, run_data):
    """Save inference data."""
    data_dir = os.path.join(run_dir, 'run_data')
    log_file = os.path.join(run_dir, 'run_log.txt')
    check_else_make_dir([run_dir, data_dir])
    run_data.save_data(data_dir)
    run_data.flush_data_strs(log_file, mode='a')


def get_subdirs(root_dir):
    """Returns all subdirectories in `root_dir`."""
    subdirs = [
        os.path.join(root_dir, i)
        for i in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, i))
    ]
    return subdirs
