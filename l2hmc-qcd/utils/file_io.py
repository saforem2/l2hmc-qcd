"""
file_io.py
"""
# pylint:disable=too-many-branches, too-many-statements
# pylint:disable=too-many-locals,invalid-name,too-many-locals
# pylint:disable=too-many-arguments
from __future__ import absolute_import, division, print_function

import os
import sys
import json
import time
import typing
import logging
import datetime

from typing import Any, Dict, Type
from tqdm.auto import tqdm
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    TextColumn,
    TimeRemainingColumn,
    Progress,
    TaskID,
)

console = Console(width=319, record=False,
                  log_time_format='[%X] ')
                  #  log_time_format="[%X] ")
#  FORMAT = "%(levelname)s:%(process)s:%(thread)s:%(name)s:%(message)s"
#  print = console.print
from rich.logging import RichHandler

import joblib
import numpy as np

from config import PROJECT_DIR
from utils.attr_dict import AttrDict

# pylint:disable=wrong-import-position
try:
    import horovod
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
    RANK = hvd.rank()
    LOCAL_RANK = hvd.local_rank()
    NUM_WORKERS = hvd.size()
    IS_CHIEF = (RANK == 0)
    console.log(f'{RANK} :: Using horovod version: {horovod.__version__}')
    console.log(f'{RANK} :: Using horovod from: {horovod.__file__}')
    #  logging.info(f'Using horovod version: {horovod.__version__}')
    #  logging.info(f'Using horovod from: {horovod.__file__}')

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

logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('arviz').setLevel(logging.ERROR)

#  logger = logging.getLogger('rich')
#  logging_level = logging.WARNING
#  FORMAT = "%(level) - %(message)s"
# ----
#  logger = logging.getLogger(__name__)
#  logging_datefmt = '%Y-%m-%d %H:%M:%S'
#  FORMAT = "(levelname)s:(%message)s"
#  FORMAT = "%(asctime)-15s - %(level) - %(message)s"
#  FORMAT = "%(levelname)s:%(process)s:%(thread)s:%(name)s:%(message)s"
#  logging_format = (
#      '%(asctime)s %(levelname)s:%(process)s:%(thread)s:%(name)s:%(message)s'
#  )

#  if IS_CHIEF:
#      logging.basicConfig(
#          level=logging.ERROR,
#          format=FORMAT,
#          datefmt="[%X]",
#          handlers=[RichHandler(rich_tracebacks=True, markup=True)],
#          #  stream=sys.stdout,
#      )
#  else:
#      logging.basicConfig(
#          level=logging.WARNING,
#          format=FORMAT,
#          #  format="%(asctime)s:%(levelname)s:%(message)s",
#          datefmt="[%X]",
#          handlers=None,
#          #  handlers=[RichHandler(rich_tracebacks=True)],
#          #  stream=None
#      )

if HAS_HOROVOD:
    #  FORMAT = (
    #      '%(levelname)s:%(process)s:%(thread)s:'
    #      + ('%04d' % hvd.rank()) + ':%(name)s:%(message)s'
    #  )
    #  #  logging_format = (
    #  #      '%(asctime)s %(levelname)s:%(process)s:%(thread)s:'
    #  #      + ('%05d' % hvd.rank()) + ':%(name)s:%(message)s'
    #  #  )
    #  logging_level = logging.WARNING
    #  if RANK > 0:
    #      logging_level = logging.ERROR
    #
    #  handlers = [RichHandler(rich_tracebacks=True)] if hvd.rank() == 0 else None
    #  logging.basicConfig(level=logging_level,
    #                      format=FORMAT,
    #                      datefmt="[%X]",
    #                      handlers=handlers)
    #  logger = logging.getLogger('rich')
    #  stream=sys.stdout if hvd.rank() == 0 else None)
    #  console.log(' '.join([f'rank: {hvd.rank()}',
    #                        f'local_rank: {hvd.local_rank()}',
    #                        f'size: {hvd.size()}',
    #                        f'local_size: {hvd.local_size()}']))
    console.log(' '.join([f'rank: {hvd.rank()}',
                          f'local_rank: {hvd.local_rank()}',
                          f'size: {hvd.size()}',
                          f'local_size: {hvd.local_size()}']))
    #  logger.warning(' '.join([f'rank: {hvd.rank()}',
    #                           f'local_rank: {hvd.local_rank()}',
    #                           f'size: {hvd.size()}',
    #                           f'local_size: {hvd.local_size()}']))

#  try:
#      from rich.logging import RichHandler
#      handlers = [RichHandler(rich_tracebacks=True)]
#
#  except ImportError:
#      handlers = []


#  if typing.TYPE_CHECKING:
#      from dynamics.base_dynamics import BaseDynamics



class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


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


def _look(p, s, conds=None):
    console.print(f'Looking in {p}...')
    matches = [x for x in Path(p).rglob(f'*{s}*')]
    if conds is not None:
        if isinstance(conds, (list, tuple)):
            for cond in conds:
                matches = [x for x in matches if cond(x)]
        else:
            matches = [x for x in matches if cond(x)]

    return matches


def get_l2hmc_dirs(paths, search_str='L16_b'):
    def _look(p, s, conds=None):
        console.print(f'Looking in {p}...')
        matches = [x for x in Path(p).rglob(f'*{s}*')]
        if conds is not None:
            if isinstance(conds, (list, tuple)):
                for cond in conds:
                    matches = [x for x in matches if cond(x)]
            else:
                matches = [x for x in matches if cond(x)]
        return matches

    dirs = []
    conds = (
        lambda x: 'GaugeModel_logs' in (str(x)),
        lambda x: 'HMC_' not in str(x),
        lambda x: Path(x).is_dir(),
    )
    if isinstance(paths, (list, tuple)):
        for path in paths:
            dirs += _look(path, search_str, conds)
    else:
        dirs = _look(paths, search_str, conds)

    return dirs


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
    #  console.print(header[0], style='bold red')
    #  console.print(header[1], style='bold red')
    #  console.print(header[-1]
    #  console.print(header.split('\n'), style='bold red')


def rule(s: str = ' ', **kwargs: dict):
    console.rule(s, **kwargs)


def log(s: str, level: str = 'INFO', out=console, style=None):
    """Print string `s` to stdout if and only if hvd.rank() == 0."""
    if RANK != 0:
        return

    #  tstr = get_timestamp('%X')
    #  hstr = f'[{tstr}] •'  # , {RANK}••{LOCAL_RANK}]'
    #hstr += '•'
    #  if NUM_WORKERS > 1:
    #      hstr += f'[{RANK}'
    #      if  hvd.local_size() > 1:
    #          hstr += f': {hvd.local_rank()}]'
    #      hstr += '•'
    #  else:
    #  if NUM_WORKERS == 1:
    #      if isinstance(s, (tuple, list)):
    #          for i in s:
    #              tqdm.write(i, file=out)
    #      else:
    #          tqdm.write(s, file=out)
    #  else:
    #  level = LOG_LEVELS_AS_INTS[level.upper()]
    #  if isinstance(s, (list, tuple)):
    #      for ss in s:
    #          console.log(ss, style=style, highlight=True)
    #  else:
    console.log(s, style=style, markup=True, highlight=True)
            #  console.print(' '.join([f'[#505050]{hstr}[/#505050]', ss]),
            #                style=style)
            #  console.print(' '.join([hstr, ss]), style='#505050')
            #  console.log(hstr + ss)
        #  console.log(level, '\n'.join(s))
        #  console.log(level, '
        #  if should_print:
        #      _ = [console.print(s_) for s_ in s]
        #  else:
        #  _ = [console.log(level, s_) for s_ in s]
    #  else:
    #      console.log(s, style=style)
    #      #  console.print(' '.join([f'[#505050]{hstr}[/#505050]', s]),
    #      #                style=style)
    #      #  console.log(' '.join([hstr, s]))
    #      #  console.log(hstr + s)
    #      #  if should_print:
    #      #      console.print(s)
    #      #  else:
    #      #  console.log(level, s)


def write(s: str, f: str, mode: str = 'a', nl: bool = True):
    """Write string `s` to file `f` if and only if hvd.rank() == 0."""
    if RANK != 0:
        return
    with open(f, mode) as f_:
        f_.write(s + '\n' if nl else ' ')


def print_dict(d: Dict, indent: int = 0, name: str = None, **kwargs):
    """Print nicely-formatted dictionary."""
    console.print(dict)
    #  indent_str = indent * ' '
    #  if name is not None:
    #      log(f'{indent_str}{name}:', **kwargs)
    #      sep_str = indent_str + len(name) * '-'
    #      log(sep_str, **kwargs)
    #  for key, val in d.items():
    #      if isinstance(val, (AttrDict, dict)):
    #          print_dict(val, indent=indent, name=str(key), **kwargs)
    #      else:
    #          log(f'  {indent_str}{key}: {val}', **kwargs)


def print_flags(flags: AttrDict):
    """Helper method for printing flags."""
    strs = [80 * '=', 'FLAGS:', *[f' {k}: {v}' for k, v in flags.items()]]
    log('\n'.join(strs))


def setup_directories(
        flags: AttrDict,
        name: str = 'training',
        save_flags: bool = True
):
    """Setup relevant directories for training."""
    if isinstance(flags, dict):
        flags = AttrDict(flags)

    if flags.get('log_dir', None) is None:
        flags.log_dir = make_log_dir(flags, name)

    train_dir = os.path.join(flags.log_dir, name)
    train_paths = AttrDict({
        'log_dir': flags.log_dir,
        'train_dir': train_dir,
        'data_dir': os.path.join(train_dir, 'train_data'),
        'models_dir': os.path.join(train_dir, 'models'),
        'ckpt_dir': os.path.join(train_dir, 'checkpoints'),
        'summary_dir': os.path.join(train_dir, 'summaries'),
        'log_file': os.path.join(train_dir, 'train_log.txt'),
        'config_dir': os.path.join(train_dir, 'dynamics_configs'),
    })

    if IS_CHIEF:
        check_else_make_dir(
            [d for k, d in train_paths.items() if 'file' not in k],
        )
        if save_flags:
            save_params(dict(flags), train_dir, 'FLAGS')

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


def print_args(args: dict):
    """Print out parsed arguments."""
    log(80 * '=' + '\n' + 'Parsed args:\n')
    for key, val in args.items():
        log(f' {key}: {val}\n')
    log(80 * '=')


def savez(obj: Any, fpath: str, name: str = None):
    """Save `obj` to compressed `.z` file at `fpath`."""
    if RANK != 0:
        return

    if not fpath.endswith('.z'):
        fpath += '.z'

    if name is not None:
        console.log(f'Saving {name} to {os.path.abspath(fpath)}.')

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
    return joblib.load(fpath)


def timeit(fn):
    def timed(*args, **kwargs):
        """Function to be timed."""
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        dt = (end_time - start_time)
        dt_s = (dt % 60)
        dt_min = (dt // 60)
        dt_ms = dt * 1000
        tstr = f'`{fn.__name__}` took: {dt_ms:.3g}ms '
        if dt_min > 0:
            tstr += f' ({int(dt_min)} min {dt_s:3.2g} sec)'

        log(tstr)
        return result
    return timed


def timeit1(out_file=None, should_log=True):
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


def get_run_dir_fstr(flags: AttrDict):
    """Parse FLAGS and create unique fstr for `run_dir`."""
    beta = flags.get('beta', None)
    config = flags.get('dynamics_config', None)

    eps = config.get('eps', None)
    hmc = config.get('hmc', False)
    num_steps = config.get('num_steps', None)
    lattice_shape = config.get('lattice_shape', None)

    fstr = ''
    if hmc:
        fstr += 'HMC_'
    if lattice_shape is not None:
        if lattice_shape[1] == lattice_shape[2]:
            fstr += f'L{lattice_shape[1]}_b{lattice_shape[0]}_'
        else:
            fstr += (
                f'L{lattice_shape[1]}_T{lattice_shape[2]}_b{lattice_shape[0]}_'
            )

    if beta is not None:
        fstr += f'beta{beta:.3g}'.replace('.', '')
    if num_steps is not None:
        fstr += f'_lf{num_steps}'
    if eps is not None:
        fstr += f'_eps{eps:.3g}'.replace('.', '')
    return fstr


def parse_configs(flags: AttrDict, debug: bool = False):
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

    if config.get('combined_updates', False):
        fstr += '_combinedUpdates'

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

    if config.get('use_batch_norm', False):
        fstr += '_bNorm'

    return fstr.replace('.', '')


def get_timestamp(fstr=None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


def make_log_dir(
        configs: AttrDict,
        model_type: str = None,
        log_file: str = None,
        root_dir: str = None,
        timestamps: AttrDict = None
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
    if os.path.isdir(log_dir) and NUM_WORKERS == 1:
        log_dir = os.path.join(*dirs, timestamps.month,
                               f'{cfg_str}-{timestamps.hour}')
        if os.path.isdir(log_dir):
            log_dir = os.path.join(*dirs, timestamps.month,
                                   f'{cfg_str}-{timestamps.minute}')

        log('\n'.join(['Existing directory found with the same name!',
                       'Modifying the date string to include seconds.']))

    if RANK == 0:
        check_else_make_dir(log_dir)
        save_dict(configs, log_dir, name='train_configs')
        if log_file is not None:
            write(f'{log_dir}', log_file, 'a')

    return log_dir


def make_run_dir(configs, base_dir):
    """Automatically create `run_dir` for storing inference data."""
    fstr = get_run_dir_fstr(configs)
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


def get_subdirs(root_dir):
    """Returns all subdirectories in `root_dir`."""
    subdirs = [
        os.path.join(root_dir, i)
        for i in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, i))
    ]
    return subdirs
