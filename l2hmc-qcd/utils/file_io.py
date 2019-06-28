"""
Helper methods for performing file IO.

Author: Sam Foreman (github: @saforem2)
Created: 2/27/2019
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import datetime
import pickle
import numpy as np

# pylint:disable=invalid-name

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
    hvd.init()

except ImportError:
    HAS_HOROVOD = False

from globals import FILE_PATH


def log(s, nl=True):
    """Print string `s` to stdout if and only if hvd.rank() == 0."""
    try:
        if HAS_HOROVOD and hvd.rank() != 0:
            return
        print(s, end='\n' if nl else '')
    except NameError:
        print(s, end='\n' if nl else '')


def write(s, f, mode='a', nl=True):
    """Write string `s` to file `f` if and only if hvd.rank() == 0."""
    try:
        if HAS_HOROVOD and hvd.rank() != 0:
            return
        with open(f, mode) as ff:
            ff.write(s + '\n' if nl else '')
    except NameError:
        with open(f, mode) as ff:
            ff.write(s + '\n' if nl else '')


def log_and_write(s, f):
    """Print string `s` to std out and also write to file `f`."""
    log(s)
    write(s, f)


def get_params_from_log_dir(log_dir):
    if '/' in log_dir:
        sep_str = log_dir.split('/')[-1].split('_')
    else:
        sep_str = log_dir.split('_')

    space_size = sep_str[0].lstrip('lattice')
    num_samples = sep_str[1].lstrip('batch')
    lf = sep_str[2].lstrip('lf')
    eps_init = sep_str[3].lstrip('eps')
    qw = sep_str[4].lstrip('qw')
    if len(sep_str) > 4:
        arch = sep_str[5].lstrip('NA')
    else:
        arch = ''

    params = {
        'space_size': space_size,
        'num_samples': num_samples,
        'num_steps': lf,
        'eps': eps_init,
        'charge_weight': qw,
        'arch': arch
    }

    return params


def create_log_dir(FLAGS, root_dir=None, log_file=None):
    """Automatically create and name `log_dir` to save model data to.

    The created directory will be located in `logs/YYYY_M_D/`, and will have
    the format (without `_qw{QW}` if running generic HMC):

        `lattice{LX}_batch{NS}_lf{LF}_eps{SS}_qw{QW}`

    Returns:
        FLAGS, with FLAGS.log_dir being equal to the newly created log_dir.

    NOTE: If log_dir does not already exist, it is created.
    """
    LX = FLAGS.space_size
    NS = FLAGS.num_samples
    LF = FLAGS.num_steps
    #  SS = str(FLAGS.eps).lstrip('0.')
    SS = FLAGS.eps
    QW = FLAGS.charge_weight
    NA = FLAGS.network_arch
    if FLAGS.hmc:
        run_str = f'HMC_lattice{LX}_batch{NS}_lf{LF}_eps{SS:.3g}'
    else:
        run_str = f'lattice{LX}_batch{NS}_lf{LF}_eps{SS:.3g}_qw{QW}_{NA}'

    now = datetime.datetime.now()
    #  print(now.strftime("%b %d %Y %H:%M:%S"))
    day_str = now.strftime('%Y_%m_%d')
    time_str = now.strftime("%Y_%m_%d_%H%M")

    #  day_str = f'{now.year}_{now.month}_{now.day}'
    #  time_str = day_str + f'_{now.hour}{now.minute}'
    project_dir = os.path.abspath(os.path.dirname(FILE_PATH))
    if FLAGS.log_dir is None:
        if root_dir is None:
            _dir = 'logs'
        else:
            _dir = root_dir

    else:
        if root_dir is None:
            _dir = FLAGS.log_dir
        else:
            _dir = os.path.join(FLAGS.log_dir, root_dir)
    root_log_dir = os.path.join(project_dir, _dir, day_str, time_str, run_str)
    check_else_make_dir(root_log_dir)
    run_num = get_run_num(root_log_dir)
    log_dir = os.path.abspath(os.path.join(root_log_dir,
                                           f'run_{run_num}'))
    if log_file is not None:
        write(f'Output saved to: \n\t{log_dir}', log_file, 'a')
        write(80*'-', log_file, 'a')

    return log_dir


def _list_and_join(d):
    """For each dir `dd` in `d`, return a list of paths ['d/dd1', ...]"""
    contents = [os.path.join(d, i) for i in os.listdir(d)]
    paths = [i for i in contents if os.path.isdir(i)]

    return paths


def list_and_join(d):
    """Deal with the case of `d` containing multiple directories."""
    if isinstance(d, (list, np.ndarray)):
        paths = []
        for dd in d:
            _path = _list_and_join(dd)[0]
            paths.append(_path)
    else:
        paths = _list_and_join(d)

    return paths


def get_eps_from_run_history_txt_file(txt_file):
    """Parse `run_history.txt` file and return `eps` (step size)."""
    with open(txt_file, 'r') as f:
        data_line = [f.readline() for _ in range(10)][-1]
    eps = float([i for i in data_line.split(' ') if i != ''][3])

    return eps


def check_else_make_dir(d):
    """If directory `d` doesn't exist, it is created."""
    if not os.path.isdir(d):
        log(f"Creating directory: {d}")
        os.makedirs(d, exist_ok=True)


def make_dirs(dirs):
    """Make directories if and only if hvd.rank == 0."""
    _ = [check_else_make_dir(d) for d in dirs]


def save_data(data, out_file, name=None):
    """Save data to out_file using either pickle.dump or np.save."""
    if os.path.isfile(out_file):
        log(f"WARNING: File {out_file} already exists...")
        tmp = out_file.split('.')
        out_file = tmp[0] + '_1' + f'.{tmp[1]}'

    log(f"Saving {name} to {out_file}...")
    if out_file.endswith('.pkl'):
        with open(out_file, 'wb') as f:
            pickle.dump(data, f)

    elif out_file.endswith('.npy'):
        np.save(out_file, np.array(data))

    else:
        log("Extension not recognized! out_file must end in .pkl or .npy")


def save_params_to_pkl_file(params, out_dir):
    """Save `params` dictionary to `parameters.pkl` in `out_dir.`"""
    check_else_make_dir(out_dir)
    params_file = os.path.join(out_dir, 'parameters.pkl')
    #  print(f"Saving params to: {params_file}.")
    log(f"Saving params to: {params_file}.")
    with open(params_file, 'wb') as f:
        pickle.dump(params, f)


def get_run_num(log_dir):
    """Get integer value for next run directory."""
    check_else_make_dir(log_dir)
    contents = os.listdir(log_dir)
    if contents in ([], ['.DS_Store']):
        return 1
    try:
        run_dirs = [i for i in os.listdir(log_dir) if 'run' in i]
        run_nums = [int(i.split('_')[-1]) for i in run_dirs]
        run_num = sorted(run_nums)[-1] + 1
    except (ValueError, IndexError):
        log(f"No previous runs found in {log_dir}, setting run_num=1.")
        run_num = 1

    return run_num


def _get_run_num(log_dir):
    check_else_make_dir(log_dir)

    contents = os.listdir(log_dir)
    if contents in ([], ['.DS_Store']):
        return 1

    run_nums = []
    for item in contents:
        try:
            run_nums.append(int(item.split('_')[-1]))
        except ValueError:
            continue
    if run_nums == []:
        return 1

    return sorted(run_nums)[-1] + 1
