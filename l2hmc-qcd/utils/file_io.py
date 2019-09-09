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
import shutil
import numpy as np

# pylint:disable=invalid-name

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
    hvd.init()

except ImportError:
    HAS_HOROVOD = False

from config import FILE_PATH


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


def copy(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            log(f'Directory not copied. Error: {e}')


def check_else_make_dir(d):
    """If directory `d` doesn't exist, it is created.

    Args:
        d (str): Location where directory should be created if it doesn't
            already exist.
    """
    if not os.path.isdir(d):
        log(f"Creating directory: {d}")
        os.makedirs(d, exist_ok=True)


def make_dirs(dirs):
    """Make directories if and only if hvd.rank == 0."""
    _ = [check_else_make_dir(d) for d in dirs]


'''
def _parse_gmm_flags(FLAGS):
    """Helper method for parsing flags as both AttrDicts or generic dicts."""
    if isinstance(FLAGS, dict):
        flags_dict = FLAGS
    else:
        try:
            flags_dict = FLAGS.__dict__
        except (NameError, AttributeError):
            pass
'''


def _parse_flags(FLAGS):
    """Helper method for parsing flags as both AttrDicts or generic dicts."""
    if isinstance(FLAGS, dict):
        flags_dict = FLAGS
    else:
        try:
            flags_dict = FLAGS.__dict__
        except (NameError, AttributeError):
            pass
    #  if isinstance(FLAGS, dict):
    try:
        LX = flags_dict['space_size']
        NS = flags_dict['num_samples']
        LF = flags_dict['num_steps']
        SS = flags_dict['eps']
        QW = flags_dict['charge_weight']
        NA = flags_dict['network_arch']
        BN = flags_dict['use_bn']
        DP = flags_dict['dropout_prob']
        AW = flags_dict['aux_weight']
        hmc = flags_dict['hmc']
        try:
            _log_dir = flags_dict['log_dir']
        except KeyError:
            _log_dir = ''

    except (NameError, AttributeError):
        LX = FLAGS.space_size
        NS = FLAGS.num_samples
        LF = FLAGS.num_steps
        SS = FLAGS.eps
        QW = FLAGS.charge_weight
        NA = FLAGS.network_arch
        BN = FLAGS.use_bn
        DP = FLAGS.dropout_prob
        AW = FLAGS.aux_weight
        hmc = FLAGS.hmc
        try:
            _log_dir = FLAGS.log_dir
        except AttributeError:
            _log_dir = ''

    out_dict = {
        'LX': LX,
        'NS': NS,
        'LF': LF,
        'SS': SS,
        'QW': QW,
        'NA': NA,
        'BN': BN,
        'DP': DP,
        'AW': AW,
        'hmc': hmc,
        '_log_dir': _log_dir
    }

    return out_dict


def create_run_str(FLAGS):
    flags_dict = _parse_flags(FLAGS)
    LX = flags_dict['LX']
    NS = flags_dict['NS']
    LF = flags_dict['LF']
    SS = flags_dict['SS']
    QW = flags_dict['QW']
    NA = flags_dict['NA']
    BN = flags_dict['BN']
    DP = flags_dict['DP']
    AW = flags_dict['AW']

    aw = str(AW).replace('.', '')
    qw = str(QW).replace('.', '')
    dp = str(DP).replace('.', '')

    if flags_dict['hmc']:
        run_str = f'HMC_lattice{LX}_batch{NS}_lf{LF}_eps{SS:.3g}'
    else:
        run_str = f'lattice{LX}_batch{NS}_lf{LF}_qw{qw}_aw{aw}_{NA}_dp{dp}'
        if BN:
            run_str += '_bn'

    return run_str, flags_dict


def create_log_dir(FLAGS, root_dir=None, log_file=None, run_str=True):
    """Automatically create and name `log_dir` to save model data to.

    The created directory will be located in `logs/YYYY_M_D/`, and will have
    the format (without `_qw{QW}` if running generic HMC):

        `lattice{LX}_batch{NS}_lf{LF}_eps{SS}_qw{QW}`

    Returns:
        FLAGS, with FLAGS.log_dir being equal to the newly created log_dir.

    NOTE: If log_dir does not already exist, it is created.
    """

    if run_str:
        run_str, flags_dict = create_run_str(FLAGS)
        _log_dir = getattr(flags_dict, '_log_dir', None)
    else:
        run_str = ''
        _log_dir = None

    now = datetime.datetime.now()
    #  print(now.strftime("%b %d %Y %H:%M:%S"))
    day_str = now.strftime('%Y_%m_%d')
    time_str = now.strftime("%Y_%m_%d_%H%M")

    project_dir = os.path.abspath(os.path.dirname(FILE_PATH))
    #  if FLAGS.log_dir is None:
    if _log_dir is None:
        if root_dir is None:
            _dir = 'logs'
        else:
            _dir = root_dir

    else:
        if root_dir is None:
            #  _dir = FLAGS.log_dir
            _dir = _log_dir
        else:
            _dir = os.path.join(_log_dir, root_dir)
    root_log_dir = os.path.join(project_dir, _dir, day_str, time_str, run_str)
    check_else_make_dir(root_log_dir)
    if any('run_' in i for i in os.listdir(root_log_dir)):
        run_num = get_run_num(root_log_dir)
        log_dir = os.path.abspath(os.path.join(root_log_dir,
                                               f'run_{run_num}'))
    else:
        log_dir = root_log_dir
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


def save_params(params, out_dir):
    check_else_make_dir(out_dir)
    params_txt_file = os.path.join(out_dir, 'parameters.txt')
    params_pkl_file = os.path.join(out_dir, 'parameters.pkl')
    with open(params_txt_file, 'w') as f:
        for key, val in params.items():
            f.write(f"{key}: {val}\n")
    with open(params_pkl_file, 'wb') as f:
        pickle.dump(params, f)


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


def get_eps_from_run_history_txt_file(txt_file):
    """Parse `run_history.txt` file and return `eps` (step size)."""
    with open(txt_file, 'r') as f:
        data_line = [f.readline() for _ in range(10)][-1]
    eps = float([i for i in data_line.split(' ') if i != ''][3])

    return eps
