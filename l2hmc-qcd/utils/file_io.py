"""
file_io.py
"""
# pylint:disable=wrong-import-position
# pylint:disable=too-many-branches, too-many-statements
# pylint:disable=too-many-locals,invalid-name,too-many-locals
# pylint:disable=too-many-arguments
from __future__ import absolute_import, annotations, division, print_function

import datetime
import json
import logging
import os
import shutil
import sys
import time
import typing
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Type, Union

import h5py
import joblib
import numpy as np
import tensorflow as tf

from config import BLUE, GREEN, PROJECT_DIR, RED, NetWeights
from utils.attr_dict import AttrDict
from utils.logger import Logger, strformat

#  from tqdm.auto import tqdm

#  from rich.theme import Theme
#  from rich.console import Console as RichConsole
#  from rich.logging import RichHandler
#  from rich.progress import (BarColumn, DownloadColumn, Progress, TaskID,
#                             TextColumn, TimeRemainingColumn)

#  console = Console(record=False,
#                    log_path=False,
#                    width=240,
#                    log_time_format='[%X] ',
#                    theme=Theme({'repr.path': BLUE,
#                                 'repr.number': GREEN}))

# pylint:disable=wrong-import-position
try:
    import horovod
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
    RANK = hvd.rank()
    LOCAL_RANK = hvd.local_rank()
    NUM_WORKERS = hvd.size()
    IS_CHIEF = (RANK == 0)
    print(80 * '=')
    print(f'{RANK} :: Using tensorflow version: {tf.__version__}')
    print(f'{RANK} :: Using tensorflow from: {tf.__file__}')
    print(f'{RANK} :: Using horovod version: {horovod.__version__}')
    print(f'{RANK} :: Using horovod from: {horovod.__file__}')
    print(80 * '=')

except (ImportError, ModuleNotFoundError):
    HAS_HOROVOD = False
    RANK = LOCAL_RANK = 0
    NUM_WORKERS = 1
    IS_CHIEF = True

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

if HAS_HOROVOD:
    print(' '.join([f'rank: {hvd.rank()}',
                    f'local_rank: {hvd.local_rank()}',
                    f'size: {hvd.size()}',
                    f'local_size: {hvd.local_size()}']))

logger = Logger()
console = logger.console

VERBOSE = os.environ.get('VERBOSE', False)

#  if typing.TYPE_CHECKING:
#      from dynamics.base_dynamics import BaseDynamics

class SortedDict(OrderedDict):
    def __init__(self, **kwargs):
        super(SortedDict, self).__init__()
        for key, val in sorted(kwargs.items()):
            if isinstance(val, dict):
                self[key] = SortedDict(**val)
            else:
                self[key] = val


def filter_dict(d: dict, cond: callable, key: str = None):
    """Filter dict using conditionals.

    Explicitly, loop through key, val pairs and accumulate those entries for
    which cond is True.

    If a key is passed, we perform the search on `d[key]`.

    Returns:
        dict: Dictionary containing the matched items.
    """
    if key is not None:
        val = d[key]
        if isinstance(val, dict):
            return {
                k: v for k, v in val.items() if cond
            }
        raise ValueError('If passing a key, d[key] should be a dict.')
    return {
        k: v for k, v in d.items() if cond
    }


def find_matching_files(d, search_str):
    darr = [x for x in Path(d).iterdir() if x.is_dir()]
    matches = []
    for rd in darr:
        results = sorted(rd.rglob(f'*{search_str}*'))
        matches.extend(results)

    return matches


def print_header(header):
    strs = header.split('\n')
    for s in strs:
        console.print(s, style='bold red')


def rule(s: str = ' ', with_time: bool= True, **kwargs: dict):
    day = get_timestamp('%Y-%m-%d')
    t = get_timestamp('%H:%M')
    s = f'[{day} {t}] {s}'
    console.rule(s, **kwargs)


def log(s: str, *args, **kwargs):
    #  def log(s: str, level: str = 'INFO', out=console, style=None):
    """Print string `s` to stdout if and only if hvd.rank() == 0."""
    if RANK != 0:
        return

    #  console.log(s, style=style, markup=True, highlight=True)
    logger.log(s, *args, **kwargs)


def write(s: str, f: str, mode: str = 'a', nl: bool = True):
    """Write string `s` to file `f` if and only if hvd.rank() == 0."""
    if RANK != 0:
        return
    with open(f, mode) as f_:
        f_.write(s + '\n' if nl else ' ')


def print_dict(d: Dict, indent: int = 0, name: str = None, **kwargs):
    """Print nicely-formatted dictionary."""
    console.print(d)


def print_flags(flags: AttrDict):
    """Helper method for printing flags."""
    strs = [80 * '=', 'FLAGS:', *[f' {k}={v}' for k, v in flags.items()]]
    logger.log('\n'.join(strs))


def setup_directories(
        configs: dict[str, Any],
        name: str = 'training',
        save_flags: bool = True,
        ensure_new: bool = False,
):
    """Setup relevant directories for training."""
    if configs.get('log_dir', None) is None:
        logger.rule('Making new log_dir')
        configs['log_dir'] = make_log_dir(configs=configs,
                                          model_type='GaugeModel',
                                          ensure_new=ensure_new)

    train_dir = os.path.join(configs['log_dir'], name)
    train_paths = {
        'log_dir': configs['log_dir'],
        'train_dir': train_dir,
        'data_dir': os.path.join(train_dir, 'train_data'),
        'models_dir': os.path.join(train_dir, 'models'),
        'ckpt_dir': os.path.join(train_dir, 'checkpoints'),
        'summary_dir': os.path.join(train_dir, 'summaries'),
        'log_file': os.path.join(train_dir, 'train_log.txt'),
        'config_dir': os.path.join(train_dir, 'dynamics_configs'),
    }

    if IS_CHIEF:
        for k, v in train_paths.items():
            if 'file' not in k:
                check_else_make_dir(v)
        if save_flags:
            save_params(dict(configs), train_dir, 'FLAGS')

    return train_paths


def make_header_from_dict(
        data: dict,
        dash: str = '-',
        skip: list = None,
        append: list = None,
        prepend: list = None,
        split: bool = False,
        with_sep: bool = True,
):
    """Build nicely formatted header with names of various metrics."""
    append = [] if append is None else append
    prepend = [] if prepend is None else prepend
    skip = [] if skip is None else skip
    keys = ['{:^12s}'.format(k) for k in data.keys() if k not in skip]
    header = ''.join(prepend + keys + append)
    if with_sep:
        sep = dash * len(header)
        header = [sep, header, sep]
        return '\n'.join(header)
    #  if split:
    #      return header
    #  return '\n'.join(header)
    return header


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

    json_file = os.path.join(out_dir, f'{name}.json')
    try:
        with open(json_file, 'w') as f:
            json.dump(d, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    except TypeError:
        log('Unable to save to `.json` file. Continuing...')

    txt_file = os.path.join(out_dir, f'{name}.txt')
    with open(txt_file, 'w') as f:
        for key, val in d.items():
            f.write(f'{key}: {val}\n')


def print_args(args: dict, name: str = None):
    """Print out parsed arguments."""
    rule(f'{name}', False) if name is not None else rule(with_time=False)
    outstr = '\n'.join([
        strformat(k, v) for k, v in args.items()
    ])
    log(outstr)

    #  log(80 * '=' + '\n' + 'Parsed args:\n')
    #  for key, val in args.items():
    #      log(f' {key}={val}\n')
    #  log(80 * '=')
    rule(with_time=False)


def savez(obj: Any, fpath: str, name: str = None):
    """Save `obj` to compressed `.z` file at `fpath`."""
    if RANK != 0:
        return

    head, tail = os.path.split(fpath)
    check_else_make_dir(head)

    if not fpath.endswith('.z'):
        fpath += '.z'

    if VERBOSE:
        if name is not None:
            name = obj.get('__class__', type(name))
        console.log(f'Saving {name} to {os.path.abspath(fpath)}.')

    joblib.dump(obj, fpath)


def change_extension(fpath: str, ext: str):
    """Change extension of `fpath` to `.ext`."""
    tmp = fpath.split('/')
    out_file = tmp[-1]
    fname, _ = out_file.split('.')
    new_fpath = os.path.join('/'.join(tmp[:-1]), f'{fname}.{ext}')

    return new_fpath


def loadz(fpath: str):
    """Load from `fpath` using `joblib.load`."""
    return joblib.load(fpath)


def timeit(fn: callable):
    def timed(*args, **kwargs):
        """Function to be timed."""
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        dt = (end_time - start_time)
        dt_s = (dt % 60)
        dt_min = (dt // 60)
        #  dt_ms = dt * 1000
        #  tstr = f'`fn.__
        tstr = ' '.join([f'`{fn.__name__}` took: {dt_s:.3g}s',
                         f' ({int(dt_min)} min {dt_s:3.8g} sec)'])
        #  if dt_min > 0:

        log(tstr)
        return result
    return timed


def timeit1(out_file: str = None, should_log: bool = True):
    """Timing decorator."""
    def wrap(fn: callable):
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


def get_run_num(run_dir: str):
    """Get the integer label for naming `run_dir`."""
    dirnames = [
        i for i in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, i))
    ]
    if len(dirnames) == 0:
        return 1

    return sorted([int(i.split('_')[-1]) for i in dirnames])[-1] + 1


def get_run_dir_fstr(flags: AttrDict):
    """Parse FLAGS and create unique fstr for `run_dir`."""
    beta = flags.get('beta', None)
    config = flags.get('dynamics_config', None)

    eps = config.get('eps', None)
    hmc = config.get('hmc', False)
    num_steps = config.get('num_steps', None)
    x_shape = config.get('x_shape', None)

    fstr = ''
    if hmc:
        fstr += 'HMC_'
    if x_shape is not None:
        if x_shape[1] == x_shape[2]:
            fstr += f'L{x_shape[1]}_b{x_shape[0]}_'
        else:
            fstr += (
                f'L{x_shape[1]}_T{x_shape[2]}_b{x_shape[0]}_'
            )

    if beta is not None:
        fstr += f'beta{beta:.3g}'.replace('.', '')
    if num_steps is not None:
        fstr += f'_lf{num_steps}'
    if eps is not None:
        fstr += f'_eps{eps:.3g}'.replace('.', '')
    return fstr


def parse_configs(configs: dict[str, Any], debug: bool = False):
    """Parse configs to construct unique string for naming `log_dir`."""
    config = AttrDict(configs.get('dynamics_config', None))
    net_config = AttrDict(configs.get('network_config', None))
    #  lr_config = flags.get('lr_config', None)
    #  conv_config = flags.get('conv_config', None)
    fstr = ''
    if config.get('hmc', False):
        fstr += 'HMC_'

    train_steps = configs.get('train_steps', 1e5)
    if debug or 0 < train_steps < 1e3:
        fstr += 'DEBUG_'

    x_shape = config.get('x_shape', None)
    if x_shape is not None:
        fstr += f'L{x_shape[1]}_b{x_shape[0]}'

    num_steps = config.get('num_steps', None)
    fstr += f'_lf{num_steps}'

    qw = config.get('charge_weight', 0.)
    pw = config.get('plaq_weight', 0.)
    aw = config.get('aux_weight', 0.)
    act = net_config.get('activation_fn', None)
    if qw == 0:
        fstr += '_qw0'
    if pw > 0:
        fstr += f'_pw{pw}'
    if aw > 0:
        fstr += f'_aw{aw}'
    if act != 'relu':
        fstr += f'_act{act}'

    if config.get('hmc', False):
        eps = config.get('eps', None)
        if eps is not None:
            fstr += f'_eps{eps}'.replace('.', '')

    bi = configs.get('beta_init', None)
    bf = configs.get('beta_final', None)
    fstr += f'_bi{bi:.3g}_bf{bf:.3g}'

    dp = net_config.get('dropout_prob', 0.)
    if dp > 0:
        fstr += f'_dp{dp}'

    hstr = ''.join([f'{i}' for i in net_config.get('units', [])])
    if len(hstr) > 0:
        fstr += f'_nh{hstr}'


    eps = config.get('eps', None)
    if configs.get('eps_fixed', False):
        fstr += f'_eps{eps:.3g}'

    cv = configs.get('clip_val', 0.)
    if cv > 0:
        fstr += f'_clip{cv}'

    if config.get('separate_networks', False):
        fstr += '_sepNets'

    if config.get('combined_updates', False):
        fstr += '_combinedUpdates'

    if config.get('use_ncp', False):
        fstr += '_NCProj'

    if config.get('use_conv_net', False):
        fstr += '_ConvNets'

        conv_config = configs.get('conv_config', {})
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

    if config.get('use_batch_norm', False):
        fstr += '_bNorm'

    nw = config.get('net_weights', None)
    #  if config.get('net_weights', None) is not None:
    if nw is not None:
        if isinstance(nw, tuple):
            nw = NetWeights(*nw)

        if nw != NetWeights(1., 1., 1., 1., 1., 1.):
            nwstr = ''.join([str(int(i)) for i in tuple(nw)])
            fstr += f'_nw{nwstr}'

    return fstr.replace('.', '')


def get_timestamp(fstr=None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


def make_log_dir(
        configs: Union[dict[str, Any], AttrDict],
        model_type: str = None,
        log_file: str = None,
        root_dir: str = None,
        timestamps: AttrDict = None,
        skip_existing: bool = False,
        ensure_new: bool = False,
        name: str = None,
):
    """Automatically create and name `log_dir` to save model data to.

    Returns:
        FLAGS, with FLAGS.log_dir being equal to the newly created log_dir.

    NOTE: If log_dir does not already exist, it is created.
    """
    model_type = 'GaugeModel' if model_type is None else model_type
    cfg_str = parse_configs(configs)

    if timestamps is None:
        timestamps = AttrDict({
            'month': get_timestamp('%Y_%m'),
            'time': get_timestamp('%Y-%M-%d-%H%M%S'),
            'hour': get_timestamp('%Y-%m-%d-%H'),
            'minute': get_timestamp('%Y-%m-%d-%H%M'),
            'second': get_timestamp('%Y-%m-%d-%H%M%S'),
        })

    if root_dir is None:
        root_dir = PROJECT_DIR

    dirs = [root_dir, 'logs', f'{model_type}_logs']

    dynamics_config = configs.get('dynamics_config', None)
    if dynamics_config is not None:
        if dynamics_config.get('hmc', False):
            dirs.append('hmc_logs')

    if cfg_str.startswith('DEBUG'):
        dirs.append('test')

    log_dir = os.path.join(*dirs, timestamps.month, cfg_str)
    if os.path.isdir(log_dir):
        if ensure_new or configs.get('ensure_new', False):

            logger.rule('Forcing new directory!')

            log_dir = os.path.join(*dirs, timestamps.month,
                                   f'{cfg_str}-{timestamps.hour}')
            if os.path.isdir(log_dir):
                log_dir = os.path.join(*dirs, timestamps.month,
                                       f'{cfg_str}-{timestamps.minute}')
            if skip_existing:
                raise FileExistsError(f'`log_dir`: {log_dir} already exists! ')

            log('\n'.join(['Existing directory found with the same name!',
                           'Modifying the date string to include seconds.']))
        #      log_dir = os.path.join(*dirs, timestamps.month,
        #                             f'{cfg_str}-{timestamps.second}')
        #  #  if os.path.isdir(log_dir) and NUM_WORKERS == 1:
        #  else:
            #  if NUM_WORKERS > 1:
            #      pass

    if RANK == 0:
        check_else_make_dir(log_dir)
        save_dict(configs, log_dir, name='train_configs')
        if log_file is not None:
            write(f'{log_dir}', log_file, 'a')

    return log_dir


def make_run_dir(
        configs: AttrDict,
        base_dir: str,
        skip_existing: bool = False
):
    """Automatically create `run_dir` for storing inference data."""
    fstr = get_run_dir_fstr(configs)
    now = datetime.datetime.now()
    dstr = now.strftime('%Y-%m-%d-%H%M')
    run_str = f'{fstr}-{dstr}'
    run_dir = os.path.join(base_dir, run_str)
    if os.path.isdir(run_dir):
        if skip_existing:
            raise FileExistsError('Existing run found!')
        log('\n'.join(['Existing directory found with the same name!',
                       'Modifying the date string to include seconds.']))
        dstr = now.strftime('%Y-%m-%d-%H%M%S')
        run_str = f'{fstr}-{dstr}'
        run_dir = os.path.join(base_dir, run_str)

    if RANK == 0:
        check_else_make_dir(run_dir)
        save_dict(configs, run_dir, name='inference_configs')

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


def get_subdirs(root_dir: str):
    """Returns all subdirectories in `root_dir`."""
    subdirs = [
        os.path.join(root_dir, i)
        for i in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, i))
    ]
    return subdirs


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        if isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        if isinstance(obj, (np.bool_)):
            return bool(obj)

        if isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def get_hmc_dirs(extra_paths=None):
    base_dirs = [
        os.path.abspath('/Users/saforem2/grand/projects/DLHMC/l2hmc-qcd/logs/GaugeModel_logs/hmc_logs'),
        os.path.abspath('/Users/saforem2/theta-fs0/projects/DLHMC/thetaGPU/inference'),
        os.path.abspath('/Users/saforem2/thetagpu/inference'),
        os.path.abspath('/lus/grand/projects/DLHMC/l2hmc-qcd/logs/GaugeModel_logs/hmc_logs'),
        os.path.abspath('/lus/theta-fs0/projects/DLHMC/thetaGPU/inference'),
    ]

    if extra_paths is not None:
        if isinstance(extra_paths, (list, tuple)):
            for p in extra_paths:
                base_dirs += p
        else:
            base_dirs += extra_paths

    hmc_dirs = []
    for d in base_dirs:
        if os.path.isdir(d):
            console.log(f'Looking in: {d}...')
            hmc_dirs += [x for x in Path(d).rglob('*HMC_L16*') if x.is_dir()]
            console.log(f'len(hmc_dirs): {len(hmc_dirs)}')

    return list(np.unique(hmc_dirs))


def _look(p, s, conds=None):
    matches = [x for x in Path(p).rglob(f'*{s}*')]
    if conds is not None:
        if isinstance(conds, (list, tuple)):
            for cond in conds:
                matches = [x for x in matches if cond(x)]
        else:
            matches = [x for x in matches if cond(x)]

    return matches


def get_l2hmc_dirs(base_dirs=None, extra_paths=None):
    if base_dirs is None:
        local_path = '/Users/saforem2/'
        thetaGPU_path = '/lus/'
        base_dirs = [
            os.path.abspath(f'{local_path}/thetaGPU/training/'),
            os.path.abspath(f'{local_path}/grand/projects/DLHMC/training/'),
            os.path.abspath(
                f'{local_path}/grand/projects/DLHMC/l2hmc-qcd/logs/GaugeModel_logs/l2hmc_logs/'
            ),
            os.path.abspath(f'{local_path}/grand/projects/DLHMC/training/annealing_schedules/'),
            os.path.abspath('/Users/saforem2/grand/projects/DLHMC/l2hmc-qcd/logs/GaugeModel_logs/2021_02'),
            # ---------------------------------------------------------
            os.path.abspath('/lus/theta-fs0/projects/DLHMC/thetaGPU/training/'),
            os.path.abspath('/lus/grand/projects/DLHMC/training/'),
            os.path.abspath('/lus/grand/projects/DLHMC/l2hmc-qcd/logs/GaugeModel_logs/l2hmc_logs/'),
            os.path.abspath('/lus/grand/projects/DLHMC/training/annealing_schedules/'),
            os.path.abspath('/lus/grand/projects/DLHMC/l2hmc-qcd/logs/GaugeModel_logs/2021_02')
        ]
    #  base_dirs = [ os.path.abspath('/Users/saforem2/thetaGPU/training'),
    #      os.path.abspath('/Users/saforem2/grand/projects/DLHMC/training'),
    #      os.path.abspath('/lus/theta-fs0/projects/DLHMC/thetaGPU/training'),
    #      os.path.abspath('/lus/grand/projects/DLHMC/thetaGPU/training'),
    #      os.path.abspath('/lus/grand/projects/DLHMC/l2hmc-qcd/logs/GaugeModel_logs/l2hmc_logs'),
    #  ]
    l2hmc_dirs = []
    for d in base_dirs:
        console.log(f'Looking in: {d}...')
        conds = (
            lambda x: 'GaugeModel_logs' in (str(x)),
            lambda x: 'HMC_' not in str(x),
            lambda x: Path(x).is_dir(),
            lambda x: os.path.isdir(os.path.join(str(x), 'run_data')),
            lambda x: os.path.isfile(os.path.join(str(x), 'run_params.z')),
        )
        l2hmc_dirs += _look(d, 'L16_b', conds)
        console.log(f'len(l2hmc_dirs): {len(l2hmc_dirs)}')

    return list(np.unique(l2hmc_dirs))
