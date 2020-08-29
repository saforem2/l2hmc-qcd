"""
file_io.py
"""
import os
import sys
import time
import pickle
import typing
import logging
import datetime

import joblib
import numpy as np

import horovod.tensorflow as hvd

#  from tqdm import tqdm
from tqdm.auto import tqdm
#  from tqdm.autonotebook import tqdm
from config import PROJECT_DIR
from utils.attr_dict import AttrDict

# pylint:disable=invalid-name
RANK = hvd.rank()
IS_CHIEF = (RANK == 0)

LOG_LEVELS_AS_INTS = {
    'CRITICAL': 50,
    'ERROR': 40,
    'WARNING': 30,
    'INFO': 20,
    'DEBUG': 10,
}

LOG_LEVELS = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
}

logging.getLogger('tensorflow').setLevel(logging.ERROR)
if IS_CHIEF:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
        stream=sys.stdout,
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
        stream=None
    )


def in_notebook():
    """Check if we're currently in a jupyter notebook."""
    try:
        # pylint:disable=import-outside-toplevel
        from IPython import get_ipython
        try:
            cfg = get_ipython().config
            if 'IPKernelApp' not in cfg:
                return False
        except AttributeError:
            return False
    except ImportError:
        return False
    return True


def log(s: str, level: str = 'INFO'):
    """Print string `s` to stdout if and only if hvd.rank() == 0."""
    if RANK != 0:
        return

    level = LOG_LEVELS_AS_INTS[level.upper()]
    if isinstance(s, (list, tuple)):
        _ = [logging.log(level, s_) for s_ in s]
    else:
        logging.log(level, s)


def write(s: str, f: str, mode: str = 'a', nl: bool = True, rank: int = 0):
    """Write string `s` to file `f` if and only if hvd.rank() == 0."""
    if rank != 0:
        return
    with open(f, mode) as f_:
        f_.write(s + '\n' if nl else ' ')


def log_tqdm(s, out=sys.stdout):
    """Write to output using `tqdm`."""
    if isinstance(s, (tuple, list)):
        for i in s:
            tqdm.write(i, file=out)
    else:
        tqdm.write(s, file=out)


def print_flags(flags: AttrDict):
    """Helper method for printing flags."""
    log('\n'.join(
        [80 * '=', 'FLAGS:', *[f' {k}: {v}' for k, v in flags.items()]]
    ))


# pylint:disable=too-many-arguments
def make_header_from_dict(
        data: dict,
        dash: str = '-',
        skip: list = None,
        append: list = None,
        prepend: list = None,
        split: bool = False,
):
    """Build nicely formatted header with names of various metrics."""
    append = [''] if append is None else append
    prepend = [''] if prepend is None else prepend
    skip = [] if skip is None else skip
    keys = ['{:^12s}'.format(k) for k in data.keys() if k not in skip]
    hstr = ''.join(prepend + keys + append)
    sep = dash * len(hstr)
    header = [sep, hstr, sep]
    if split:
        return header

    return '\n'.join(header)


def log_and_write(
        s: str,
        f: str,
        rank: int = 0,
        mode: str = 'a',
        nl: bool = True
):
    """Print string `s` to std out and also write to file `f`."""
    log(s)
    write(s, f, mode=mode, nl=nl, rank=rank)


def check_else_make_dir(d: str, rank: int = 0):
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
            log(f"Creating directory: {d}")
            os.makedirs(d, exist_ok=True)


def flush_data_strs(data_strs: list,
                    out_file: str,
                    rank: int = 0,
                    mode: str = 'a'):
    """Dump `data_strs` to `out_file` and return new, empty list."""
    if rank != 0:
        return []

    with open(out_file, mode) as f:
        for s in data_strs:
            f.write(f'{s}\n')

    return []


def save_params(params: dict, out_dir: str, name: str = None, rank: int = 0):
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


def save_dict(d: dict, out_dir: str, name: str, rank: int = 0):
    """Save dictionary to `out_dir` as both `.z` and `.txt` files."""
    if rank != 0:
        return

    if isinstance(d, AttrDict):
        d = dict(d)

    check_else_make_dir(out_dir, rank=rank)

    zfile = os.path.join(out_dir, f'{name}.z')
    txt_file = os.path.join(out_dir, f'{name}.txt')
    log('\n'.join([f'Saving {name} to:', f'  {zfile}', f'  {txt_file}']))
    savez(d, zfile, name=name, rank=rank)
    with open(txt_file, 'w') as f:
        for key, val in d.items():
            f.write(f'{key}: {val}\n')


def print_args(args: dict, rank: int = 0):
    """Print out parsed arguments."""
    log(80 * '=' + '\n' + 'Parsed args:\n')
    for key, val in args.items():
        log(f' {key}: {val}\n')
    log(80 * '=')


def savez(obj: typing.Any, fpath: str, name: str = None, rank: int = 0):
    """Save `obj` to compressed `.z` file at `fpath`."""
    if rank != 0:
        return

    if not fpath.endswith('.z'):
        fpath += '.z'

    if name is not None:
        log(f'Saving {name} to {fpath}.')

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


def timeit(out_file=None, should_log=True):
    """Timing decorator."""
    def wrap(fn):
        def timed(*args, **kwargs):
            """Function to be timed."""
            start_time = time.time()
            result = fn(*args, **kwargs)
            end_time = time.time()

            if 'log_time' in kwargs:
                name = kwargs.get('log_name', fn.__name__.upper())
                kwargs['log_time'][name] = int((end_time - start_time) * 1000)
            else:
                dt = (end_time - start_time) * 1000
                tstr = f'`{fn.__name__}` took: {dt:.5g}ms'
                if out_file is not None:
                    if should_log:
                        log_and_write(tstr, out_file, mode='a')
                    else:
                        write(tstr, out_file, mode='a')
                if should_log:
                    log(tstr)
            return result
        return timed
    return wrap


def get_run_num(run_dir):
    """Get the integer label for naming `run_dir`."""
    dirnames = [
        i for i in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, i))
    ]
    if len(dirnames) == 0:
        return 1

    return sorted([int(i.split('_')[-1]) for i in dirnames])[-1] + 1


def get_run_dir_fstr(FLAGS: AttrDict):
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
        fstr += 'HMC_'
    if beta is not None:
        fstr += f'beta{beta:.3g}'.replace('.', '')
    if num_steps is not None:
        fstr += f'_lf{num_steps}'
    if eps is not None:
        fstr += f'_eps{eps:.3g}'.replace('.', '')
    if lattice_shape is not None:
        fstr += f'_b{lattice_shape[0]}'

    return fstr


# pylint:disable=too-many-branches, too-many-locals, too-many-statements
def get_log_dir_fstr(flags):
    """Parse FLAGS and create unique fstr for `log_dir`."""
    hmc = flags.get('hmc', False)
    batch_size = flags.get('batch_size', None)
    num_steps = flags.get('num_steps', None)
    beta_init = flags.get('beta_init', None)
    beta_final = flags.get('beta_final', None)
    eps = flags.get('eps', None)
    space_size = flags.get('space_size', None)
    time_size = flags.get('time_size', None)
    lattice_shape = flags.get('lattice_shape', None)
    train_steps = flags.get('train_steps', int(1e3))
    network_type = flags.get('network_type', 'GaugeNetwork')
    charge_weight = flags.get('charge_weight', 0.)
    plaq_weight = flags.get('plaq_weight', 0.)
    eps_fixed = flags.get('eps_fixed', False)
    dropout_prob = flags.get('dropout_prob', 0.)
    clip_val = flags.get('clip_val', 0.)
    aux_weight = flags.get('aux_weight', 0.)
    activation = flags.get('activation', 'relu')
    separate_networks = flags.get('separate_networks', False)
    using_ncp = flags.get('use_ncp', False)
    zero_init = flags.get('zero_init', False)

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

    if aux_weight > 0:
        fstr += f'_aw{aux_weight}'.replace('.', '')

    if activation != 'relu':
        fstr += f'_act{activation}'

    fstr += f'_bi{beta_init:.3g}_bf{beta_final:.3g}'.replace('.', '')

    if dropout_prob > 0:
        fstr += f'_dp{dropout_prob}'.replace('.', '')

    if eps_fixed:
        fstr += f'_eps{eps:.3g}'.replace('.', '')

    if clip_val > 0:
        fstr += f'_clip{clip_val}'.replace('.', '')

    if network_type != 'GaugeNetwork':
        fstr += f'_{network_type}'

    if separate_networks:
        fstr += '_sepNets'

    if using_ncp:
        fstr += '_NCProj'

    if zero_init:
        fstr += '_zero_init'

    return fstr


# pylint:disable=too-many-arguments
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
        log('\n'.join(['Existing directory found with the same name!',
                       'Modifying the date string to include seconds.']))
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
        log('\n'.join(['Existing directory found with the same name!',
                       'Modifying the date string to include seconds.']))
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
