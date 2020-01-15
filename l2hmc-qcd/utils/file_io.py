"""
Helper methods for performing file IO.

Author: Sam Foreman (github: @saforem2)
Created: 2/27/2019
"""
from __future__ import absolute_import, division, print_function

import os
import time
import errno
import pickle
import shutil
import datetime
import config as cfg

import numpy as np

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
    hvd.init()

except ImportError:
    HAS_HOROVOD = False


def copy(src, dst):
    """Copy from src --> dst using `shutil.copytree`."""
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python > 2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else:
            raise

def copy_gauge_figures(root_src_dir=None, root_dst_dir=None):
    """Copy `figures` and `figures_np` from all `log_dirs` to `~`."""
    if root_src_dir is None:
        root_src_dir = os.path.abspath('/home/foremans/DLHMC/'
                                       'l2hmc-qcd/gauge_logs')
    if root_dst_dir is None:
        dirstr = '/home/foremans/gauge_logs_figures'
        if os.path.isdir(os.path.abspath(dirstr)):
            timestr = get_timestr()
            tstr = timestr['timestr']
            dirstr += f'_{tstr}'
            root_dst_dir = os.path.abspath(dirstr)
    for src_dir, _, _ in os.walk(root_src_dir):
        if src_dir.endswith('figures') or src_dir.endswith('figures_np'):
            date_str = src_dir.split('/')[-3]
            log_str = src_dir.split('/')[-2]
            fig_str = src_dir.split('/')[-1]
            dst_dir = os.path.join(root_dst_dir, date_str, log_str, fig_str)
            log(f'Copying {src_dir} --> {dst_dir}')
            copy(src_dir, dst_dir)

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


def get_timestr():
    """Get formatted time string."""
    now = datetime.datetime.now()
    day_str = now.strftime('%Y_%m_%d')
    hour_str = now.strftime('%H%M')
    timestr = f'{day_str}_{hour_str}'

    timestrs = {
        'day_str': day_str,
        'hour_str': hour_str,
        'timestr': timestr,
    }

    return timestrs


def load_params(log_dir):
    """Load params from log_dir."""
    params_file = os.path.join(log_dir, 'parameters.pkl')
    with open(params_file, 'rb') as f:
        params = pickle.load(f)

    return params


# pylint: disable=invalid-name
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


def copy_old(src, dest):
    """Copy from src to dst."""
    try:
        shutil.copytree(src, dest)
    except OSError:
        # If the error was caused because the source wasn't a directory
        #  if e.errno == errno.ENOTDIR:
        try:
            shutil.copy(src, dest)
        except OSError as ee:
            log(f'Directory not copied. Error: {ee}')


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


# pylint: disable=too-many-branches
def _parse_gauge_flags(FLAGS):
    """Parse flags for `GaugeModel` instance."""
    if isinstance(FLAGS, dict):
        flags_dict = FLAGS
    else:
        try:
            flags_dict = FLAGS.__dict__
        except (NameError, AttributeError):
            pass
    d = {
        'LX': flags_dict.get('space_size', None),
        'BS': flags_dict.get('batch_size', None),
        'LF': flags_dict.get('num_steps', None),
        'SS': flags_dict.get('eps', None),
        'EF': flags_dict.get('eps_fixed', False),
        'QW': flags_dict.get('charge_weight', None),
        'NA': flags_dict.get('network_arch', None),
        'BN': flags_dict.get('use_bn', None),
        'DP': flags_dict.get('dropout_prob', None),
        'AW': flags_dict.get('aux_weight', None),
        'XS': flags_dict.get('x_scale_weight', None),
        'XT': flags_dict.get('x_translation_weight', None),
        'XQ': flags_dict.get('x_transformation_weight', None),
        'VS': flags_dict.get('v_scale_weight', None),
        'VT': flags_dict.get('v_translation_weight', None),
        'VQ': flags_dict.get('v_transformation_weight', None),
        'GL': flags_dict.get('use_gaussian_loss', None),
        'NL': flags_dict.get('use_nnehmc_loss', None),
        'CV': flags_dict.get('clip_value', None),
        'hmc': flags_dict.get('hmc', None),
    }
    try:
        _log_dir = flags_dict['log_dir']
    except KeyError:
        _log_dir = ''

    d['_log_dir'] = _log_dir
    aw = str(d['AW']).replace('.', '')
    qw = str(d['QW']).replace('.', '')
    dp = str(d['DP']).replace('.', '')

    d['XS'] = str(int(d['XS'])).replace('.', '')
    d['XT'] = str(int(d['XT'])).replace('.', '')
    d['XQ'] = str(int(d['XQ'])).replace('.', '')
    d['VS'] = str(int(d['VS'])).replace('.', '')
    d['VT'] = str(int(d['VT'])).replace('.', '')
    d['VQ'] = str(int(d['VQ'])).replace('.', '')

    if flags_dict['hmc']:
        run_str = (f"HMC_lattice{d['LX']}_batch{d['BS']}"
                   "_lf{d['LF']}_eps{d['SS']:.3g}")
    else:
        run_str = f"L{d['LX']}_b{d['BS']}_lf{d['LF']}"

        if qw != '00':  # if charge weight != 0
            run_str += f'_qw{qw}'

        if aw != '10':  # if aux_weight != 1
            run_str += f'_aw{aw}'

        if d['NA'] != 'generic':  # if network_arch != generic
            run_str += f"_{d['NA']}"

        if dp != '00':  # if dropout_prob > 0
            run_str += f'_dp{dp}'

        # if x_scale_weight or x_transl_weight or x_transf_weight != 1.
        if d['XS'] != '1' or d['XT'] != '1' or d['XQ'] != '1':
            run_str += f"_x{d['XS']}{d['XT']}{d['XQ']}"

        # if v_scale_weight or v_transl_weight or v_transf_weight != 1.
        #  if d['VS'] != 1. or d['VT'] != 1. or d['VQ'] != 1.:
        if d['VS'] != '1' or d['VT'] != '1' or d['VQ'] != '1':
            run_str += f"_v{d['VS']}{d['VT']}{d['VQ']}"

        # if using a fixed (non-trainable) step size:
        if d['EF']:
            run_str += f'_eps_fixed'

        # if using batch normalization
        if d['BN']:
            run_str += '_bn'

        if d['GL'] and d['NL']:  # if using gaussian_loss and nnehmc_loss
            run_str += '_gnl'  # Gaussian + NNEHMC loss

        # if using gaussian_loss but not nnehmc_loss
        elif d['GL'] and not d['NL']:
            run_str += '_gl'

        # if using nnehmc_loss but not gaussian_loss
        elif d['NL'] and not d['GL']:
            run_str += '_nl'

    return run_str, d


def _parse_gmm_flags(FLAGS):
    """Parse flags for `GaussianMixtureModel` instance."""
    if isinstance(FLAGS, dict):
        flags_dict = FLAGS
    else:
        try:
            flags_dict = FLAGS.__dict__
        except (NameError, AttributeError):
            pass

    d = {'X0': flags_dict.get('center', None),
         'ND': flags_dict.get('num_distributions', None),
         'LF': flags_dict.get('num_steps', None),
         'DG': flags_dict.get('diag', None),
         'S1': flags_dict.get('sigma1', None),
         'S2': flags_dict.get('sigma2', None),
         'P1': flags_dict.get('pi1', None),
         'P2': flags_dict.get('pi2', None),
         'GL': flags_dict.get('use_gaussian_loss', False),
         'NL': flags_dict.get('use_nnehmc_loss', False),
         'BN': flags_dict.get('use_bn', False),
         'AW': flags_dict.get('aux_weight', 1.),
         'AR': flags_dict.get('arrangement', 'xaxis'),
         'XS': flags_dict.get('x_scale_weight', None),
         'XT': flags_dict.get('x_translation_weight', None),
         'XQ': flags_dict.get('x_transformation_weight', None),
         'VS': flags_dict.get('v_scale_weight', None),
         'VT': flags_dict.get('v_translation_weight', None),
         'VQ': flags_dict.get('v_transformation_weight', None)}

    aw = str(d['AW']).replace('.', '')
    s1 = str(d['S1']).replace('.', '')
    s2 = str(d['S2']).replace('.', '')

    d['XS'] = str(int(d['XS'])).replace('.', '')
    d['XT'] = str(int(d['XT'])).replace('.', '')
    d['XQ'] = str(int(d['XQ'])).replace('.', '')
    d['VS'] = str(int(d['VS'])).replace('.', '')
    d['VT'] = str(int(d['VT'])).replace('.', '')
    d['VQ'] = str(int(d['VQ'])).replace('.', '')

    #  run_str = f'GMM_{AR}_lf{LF}_aw{aw}_s1_{s1}_s2_{s2}'
    run_str = f"GMM_{d['AR']}_lf{d['LF']}_s1{s1}_s2{s2}"

    if aw != '10':
        run_str += f'_aw{aw}'

    if d['BN']:
        run_str += '_bn'



    if ['GL'] and d['NL']:
        run_str += '_gnl'  # Gaussian + NNEHMC loss

    elif d['GL'] and not d['NL']:
        run_str += '_gl'

    elif d['NL'] and not d['GL']:
        run_str += '_nl'

    return run_str, d


def _parse_flags(FLAGS, model_type='GaugeModel'):
    """Helper method for parsing flags as both AttrDicts or generic dicts."""
    if model_type == 'GaugeModel':
        run_str, out_dict = _parse_gauge_flags(FLAGS)
    elif model_type == 'GaussianMixtureModel':
        run_str, out_dict = _parse_gmm_flags(FLAGS)

    if cfg.NP_FLOAT == np.float64:
        run_str += '_f64'
    elif cfg.NP_FLOAT == np.float32:
        run_str += '_f32'

    return run_str, out_dict


# pylint: disable=too-many-locals
def create_log_dir(FLAGS, **kwargs):
    """Automatically create and name `log_dir` to save model data to.

    The created directory will be located in `logs/YYYY_M_D/`, and will have
    the format (without `_qw{QW}` if running generic HMC):

        `lattice{LX}_batch{NS}_lf{LF}_eps{SS}_qw{QW}`

    Returns:
        FLAGS, with FLAGS.log_dir being equal to the newly created log_dir.

    NOTE: If log_dir does not already exist, it is created.
    """
    run_str = kwargs.get('run_str', True)
    model_type = kwargs.get('model_type', 'GaugeModel')
    log_file = kwargs.get('log_file', None)
    root_dir = kwargs.get('root_dir', None)
    if run_str:
        run_str, flags_dict = _parse_flags(FLAGS, model_type)
        _log_dir = getattr(flags_dict, '_log_dir', None)
    else:
        run_str = ''
        _log_dir = None

    now = datetime.datetime.now()
    day_str = now.strftime('%Y_%m_%d')
    hour_str = now.strftime('%H%M')

    project_dir = os.path.abspath(os.path.dirname(cfg.FILE_PATH))

    if _log_dir is None:
        _dir = 'gauge_logs' if root_dir is None else root_dir

    else:
        if root_dir is None:
            _dir = _log_dir
        else:
            _dir = os.path.join(_log_dir, root_dir)
    root_log_dir = os.path.join(project_dir, _dir, day_str, run_str)
    # if `root_log_dir` already exists, append '_%H%M' (hour, minute) at end
    #  if os.path.isdir(root_log_dir):
    dirname = run_str + f'_{hour_str}'
    if os.path.isdir(os.path.join(project_dir, _dir, day_str, dirname)):
        dirname += '_1'

    root_log_dir = os.path.join(project_dir, _dir, day_str, dirname)
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
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    elif out_file.endswith('.npy'):
        np.save(out_file, np.array(data))

    else:
        log("Extension not recognized! out_file must end in .pkl or .npy")


def save_params(params, out_dir, name=None):
    """save params (dict) to `out_dir`, as both `.pkl` and `.txt` files."""
    check_else_make_dir(out_dir)
    if name is None:
        name = 'parameters'
    params_txt_file = os.path.join(out_dir, f'{name}.txt')
    params_pkl_file = os.path.join(out_dir, f'{name}.pkl')
    with open(params_txt_file, 'w') as f:
        for key, val in params.items():
            f.write(f"{key}: {val}\n")
    with open(params_pkl_file, 'wb') as f:
        pickle.dump(params, f)


def save_dict(d, out_dir, name):
    """Save generic dict `d` to `out_dir` as both `.pkl` and `.txt` files."""
    check_else_make_dir(out_dir)
    txt_file = os.path.join(out_dir, f'{name}.txt')
    pkl_file = os.path.join(out_dir, f'{name}.pkl')
    with open(txt_file, 'w') as f:
        for key, val in d.items():
            f.write(f"{key}: {val}\n")
    with open(pkl_file, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)


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
