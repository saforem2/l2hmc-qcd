"""
file_io.py
"""
# pylint:disable=too-many-branches, too-many-statements
# pylint:disable=too-many-locals,invalid-name,too-many-locals
import os
import sys
import time
import json
import pickle
import typing
import logging
import datetime

import joblib
import numpy as np

import horovod.tensorflow as hvd

from tqdm.auto import tqdm
from config import PROJECT_DIR
from utils.attr_dict import AttrDict

RANK = hvd.rank()
NUM_NODES = hvd.size()
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
logging.getLogger('arviz').setLevel(logging.ERROR)

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


def write(s: str, f: str, mode: str = 'a', nl: bool = True):
    """Write string `s` to file `f` if and only if hvd.rank() == 0."""
    if RANK != 0:
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


def setup_directories(flags, name='training'):
    """Setup relevant directories for training."""
    train_dir = os.path.join(flags.log_dir, name)
    train_paths = AttrDict({
        'log_dir': flags.log_dir,
        'train_dir': train_dir,
        'data_dir': os.path.join(train_dir, 'train_data'),
        'ckpt_dir': os.path.join(train_dir, 'checkpoints'),
        'summary_dir': os.path.join(train_dir, 'summaries'),
        'log_file': os.path.join(train_dir, 'train_log.txt'),
        'config_dir': os.path.join(train_dir, 'dynamics_configs'),
    })

    if IS_CHIEF:
        check_else_make_dir(
            [d for k, d in train_paths.items() if 'file' not in k],
        )
        if not flags.restore:
            save_params(dict(flags), train_dir, 'FLAGS')

    return train_paths


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
        mode: str = 'a',
        nl: bool = True
):
    """Print string `s` to std out and also write to file `f`."""
    log(s)
    write(s, f, mode=mode, nl=nl)


def check_else_make_dir(d: str):
    """If directory `d` doesn't exist, it is created.

    Args:
        d (str): Location where directory should be created if it doesn't
            already exist.
    """
    if RANK != 0:
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
                    mode: str = 'a'):
    """Dump `data_strs` to `out_file` and return new, empty list."""
    if RANK != 0:
        return []

    with open(out_file, mode) as f:
        for s in data_strs:
            f.write(f'{s}\n')

    return []


def save_params(params: dict, out_dir: str, name: str = None):
    """save params(dict) to `out_dir`, as both `.z` and `.txt` files."""
    if RANK != 0:
        return

    check_else_make_dir(out_dir)
    if name is None:
        name = 'params'
    params_txt_file = os.path.join(out_dir, f'{name}.txt')
    zfile = os.path.join(out_dir, f'{name}.z')
    with open(params_txt_file, 'w') as f:
        for key, val in params.items():
            f.write(f"{key}: {val}\n")
    savez(params, zfile, name=name)


def save_dict(d: dict, out_dir: str, name: str = None):
    """Save dictionary `d` to `out_dir` as both `.z` and `.txt` files."""
    if RANK != 0:
        return

    check_else_make_dir(out_dir)
    name = 'dict' if name is None else name

    zfile = os.path.join(out_dir, f'{name}.z')
    savez(d, zfile, name=name)

    txt_file = os.path.join(out_dir, f'{name}.txt')
    with open(txt_file, 'w') as f:
        for key, val in d.items():
            f.write(f'{key}: {val}\n')


def print_args(args: dict):
    """Print out parsed arguments."""
    log(80 * '=' + '\n' + 'Parsed args:\n')
    for key, val in args.items():
        log(f' {key}: {val}\n')
    log(80 * '=')


def savez(obj: typing.Any, fpath: str, name: str = None):
    """Save `obj` to compressed `.z` file at `fpath`."""
    if RANK != 0:
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


def parse_configs(flags, debug=False):
    """Parse configs to construct unique string for naming `log_dir`."""
    config = AttrDict(flags.get('dynamics_config', None))
    net_config = AttrDict(flags.get('network_config', None))
    #  lr_config = flags.get('lr_config', None)
    #  conv_config = flags.get('conv_config', None)
    fstr = ''
    if config.get('hmc', False):
        fstr += 'HMC_'

    if debug or 0 < flags.get('train_steps', None) < 1e4:
        fstr += 'DEBUG_'

    lattice_shape = config.get('lattice_shape', None)
    if lattice_shape is not None:
        fstr += f'L{lattice_shape[1]}_b{lattice_shape[0]}'

    num_steps = config.get('num_steps', None)
    fstr += f'_lf{num_steps}'

    qw = config.get('charge_weight', None)
    pw = config.get('plaq_weight', None)
    aw = config.get('aux_weight', None)
    act = net_config.get('activation_fn', None)
    if qw > 0:
        fstr += f'_qw{qw}'
    if pw > 0:
        fstr += f'_pw{pw}'
    if aw > 0:
        fstr += f'_aw{aw}'
    if act != 'relu':
        fstr += f'_act{act}'

    bi = flags.get('beta_init', None)
    bf = flags.get('beta_final', None)
    fstr += f'_bi{bi:.3g}_bf{bf:.3g}'

    dp = net_config.get('dropout_prob', None)
    if dp > 0:
        fstr += f'_dp{dp}'

    eps = config.get('eps', None)
    if flags.get('eps_fixed', False):
        fstr += f'_eps{eps:.3g}'

    cv = flags.get('clip_val', None)
    if cv > 0:
        fstr += f'_clip{cv}'

    if config.get('separate_networks', False):
        fstr += '_sepNets'

    if config.get('use_ncp', False):
        fstr += '_NCProj'

    if config.get('use_conv_net', False):
        fstr += '_ConvNets'

        conv_config = flags.get('conv_config', None)
        use_bn = conv_config.get('use_batch_norm', False)
        if use_bn:
            fstr += '_bNorm'

    if config.get('zero_init', False):
        fstr += '_zero_init'

    if config.get('use_mixed_loss', False):
        fstr += '_mixedLoss'

    if config.get('use_scattered_xnet_update', False):
        fstr += '_xScatter'

    if config.get('use_tempered_trajectories', False):
        fstr += '_temperedTraj'

    return fstr.replace('.', '')


# pylint:disable=too-many-arguments
def make_log_dir(FLAGS, model_type=None, log_file=None, root_dir=None):
    """Automatically create and name `log_dir` to save model data to.

    The created directory will be located in `logs/YYYY_M_D /`, and will have
    the format(without `_qw{QW}` if running generic HMC):

        `lattice{LX}_batch{NS}_lf{LF}_eps{SS}_qw{QW}`

    Returns:
        FLAGS, with FLAGS.log_dir being equal to the newly created log_dir.

    NOTE: If log_dir does not already exist, it is created.
    """
    model_type = 'GaugeModel' if model_type is None else model_type
    fstr = parse_configs(FLAGS)

    now = datetime.datetime.now()
    month_str = now.strftime('%Y_%m')
    dstr = now.strftime('%Y-%m-%d-%H%M%S')
    run_str = f'{fstr}-{dstr}'

    #  if root_dir is None:
    if root_dir is None:
        root_dir = os.path.dirname(PROJECT_DIR)

    dirs = [root_dir, 'logs', f'{model_type}_logs']
    if fstr.startswith('DEBUG'):
        dirs.append('test')

    log_dir = os.path.join(*dirs, month_str, run_str)
    if os.path.isdir(log_dir):
        log('\n'.join(['Existing directory found with the same name!',
                       'Modifying the date string to include seconds.']))
        dstr = now.strftime('%Y-%m-%d-%H%M%S')
        run_str = f'{fstr}-{dstr}'
        log_dir = os.path.join(*dirs, month_str, run_str)

    if RANK == 0:
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


def save_network_weights(dynamics, train_dir):
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
        savez(xnet_weights, xweights_file, 'xnet_weights')
        savez(vnet_weights, vweights_file, 'vnet_weights')
    else:
        xfpath = os.path.join(wdir, 'xnet_weights.z')
        vfpath = os.path.join(wdir, 'vnet_weights.z')
        xnets.save_layer_weights(out_file=xfpath)
        vnets.save_layer_weights(out_file=vfpath)


def save(dynamics, train_data, train_dir):
    """Save training results."""
    if RANK != 0:
        return

    check_else_make_dir(train_dir)

    if not dynamics.config.hmc:
        save_network_weights(dynamics, train_dir)

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
