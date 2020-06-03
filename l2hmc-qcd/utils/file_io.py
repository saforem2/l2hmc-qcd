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

from collections import OrderedDict

import joblib
import numpy as np

import config as cfg

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
    hvd.init()

except ImportError:
    HAS_HOROVOD = False


# pylint: disable=invalid-name
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals

def log(s, nl=True):
    """Print string `s` to stdout if and only if hvd.rank() == 0."""
    try:
        if HAS_HOROVOD and hvd.rank() != 0:
            return
        print(s, end='\n' if nl else ' ')
    except NameError:
        print(s, end='\n' if nl else ' ')


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


def log_and_write(s, f, mode='a', nl=True):
    """Print string `s` to std out and also write to file `f`."""
    log(s, nl)
    write(s, f, mode=mode, nl=nl)


def strf(x):
    """Format the number x as a string."""
    if np.allclose(x - np.around(x), 0):
        xstr = f'{int(x)}'
    else:
        xstr = f'{x:.1}'.replace('.', '')
    return xstr


def get_subdirs(root_dir):
    """Returns all subdirectories in `root_dir`."""
    subdirs = [
        os.path.join(root_dir, i)
        for i in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, i))
    ]
    return subdirs


def change_params_log_dir(log_dir):
    """Update params with new log dir when copied to local system."""
    params_file = os.path.join(log_dir, 'params.z')
    if os.path.isfile(params_file):
        log(f'Loading from: {params_file}.')
        copy_file = os.path.join(log_dir, 'params_original.z')
        _ = shutil.copy2(params_file, copy_file)
        params = loadz(os.path.join(log_dir, 'params.z'))
        orig_log_dir = params['log_dir']
        params['orig_log_dir'] = orig_log_dir
        params['log_dir'] = log_dir
        save_dict(params, log_dir, 'params')

    return params


def change_checkpoint_dir(log_dir, orig_str):
    """Change ckpt file in log dir when data copied to local system."""
    if orig_str.endswith('/'):
        orig_str.rstrip('/')
    checkpoint_file = os.path.join(log_dir, 'checkpoints', 'checkpoint')
    new_str = os.path.join(*log_dir.split('/')[:-2])
    new_str = '/'.join(log_dir.split('/')[:-2])
    if not new_str.endswith('/'):
        new_str += '/'
    if os.path.isfile(checkpoint_file):
        log(f'Loading from checkpoint file: {checkpoint_file}.')
        log(f'Replacing:\n \t{orig_str}\n with\n \t{new_str}\n')

        copy_file = os.path.join(log_dir, 'checkpoints', 'checkpoint_original')
        _ = shutil.copy2(checkpoint_file, copy_file)

        lines = []
        with open(checkpoint_file, 'r') as f:
            for line in f:
                lines.append(line.replace(orig_str, new_str))

        with open(checkpoint_file, 'w') as f:
            f.writelines(lines)


def get_run_dirs(log_dir, filter_str=None, runs_str='runs_np'):
    """Get all `run_dirs` in `log_dir/runs_dir/`."""
    run_dirs = None
    runs_dir = os.path.join(log_dir, runs_str)
    run_dirs = get_subdirs(runs_dir)
    if filter_str is not None:
        run_dirs = [i for i in run_dirs if filter_str in i]

    run_dirs = sorted(run_dirs)
    return run_dirs


def write_dict(d, out_file):
    """Recursively write key, val pairs to `out_file`."""
    for key, val in d.items():
        #  if isinstance(val, dict):
        #      write_dict(d, out_file)
        write(f'{key}: {val}\n', out_file)


def savez(obj, fpath, name=None):
    """Save `obj` to compressed `.z` file at `fpath`."""
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


def save_pkl(obj, fpath, name=None, compressed=True):
    """Save `obj` to `fpath`."""
    if compressed:  # force extension type to be '.z' (auto compress)
        zfpath = change_extension(fpath, 'z')
        #  tmp = fpath.split('/')
        #  out_file = tmp[-1]
        #  fname, _ = out_file.split('.')
        #  zfpath = os.path.join('/'.join(tmp[:-1]), f'{fname}.z')

        if name is not None:
            log(f'Saving {name} to {zfpath}.')

        joblib.dump(obj, zfpath)

    else:
        if name is not None:
            log(f'Saving {name} to {fpath}.')
        with open(fpath, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(fpath):
    """Load from `fpath` using `pickle.load`."""
    with open(fpath, 'rb') as f:
        data = pickle.load(f)

    return data


def load_compressed(in_file):
    """Load from `in_file`, and return contents."""
    return joblib.load(in_file)


def make_pngs_from_pdfs(rootdir=None):
    """Use os.walk from `rootdir`, creating pngs from all pdfs encountered."""
    if rootdir is None:
        rootdir = os.path.abspath('/home/foremans/DLHMC/l2hmc-qcd/gauge_logs')
    for root, _, files in os.walk(rootdir):
        if 'old' in root:
            continue
        fnames = [i.rstrip('.pdf') for i in files if i.endswith('.pdf')]
        in_files = [os.path.join(root, i) for i in files if i.endswith('.pdf')]
        if len(in_files) > 1:
            png_dir = os.path.join(root, 'pngs')
            check_else_make_dir(png_dir)
            out_files = [os.path.join(png_dir, f'{i}') for i in fnames]
            for inf, outf in zip(in_files, out_files):
                if not os.path.isfile(outf):
                    log(f'in: {inf} --> out: {outf}\n')
                    try:
                        os.system(f'~/bin/pdftopng {inf} {outf}')
                    except OSError:
                        return


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
        #  if src_dir.endswith('figures') or src_dir.endswith('figures_np'):
        if src_dir == 'figures_np':
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
        'year_str': now.strftime('%Y'),
        'month_str': now.strftime('%m'),
        'date_str': now.strftime('%d'),
    }

    return timestrs


def load_params(log_dir):
    """Load params from log_dir."""
    names = ['parameters.pkl', 'parameters.z', 'params.pkl', 'params.z']
    params = None
    for name in names:
        params_file = os.path.join(log_dir, name)
        if os.path.isfile(params_file):
            params = loadz(params_file)

    return params


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
    if isinstance(d, (list, np.ndarray)):
        for i in d:
            check_else_make_dir(i)
    else:
        if not os.path.isdir(d):
            log(f"Creating directory: {d}")
            os.makedirs(d, exist_ok=True)


def make_dirs(dirs):
    """Make directories if and only if hvd.rank == 0."""
    _ = [check_else_make_dir(d) for d in dirs]


def _parse_gauge_flags(FLAGS):
    """Parse flags for `GaugeModel` instance."""
    if isinstance(FLAGS, dict):
        flags = FLAGS
    else:
        flags = FLAGS.__dict__

    flags_dict = {
        'space_size': flags.get('space_size', 8),
        'time_size': flags.get('time_size', 8),
        'batch_size': flags.get('batch_size', 32),
        'num_steps': flags.get('num_steps', 5),
        'charge_weight': flags.get('charge_weight', 0),
        'plaq_weight': flags.get('plaq_weight', 0.),
        'network_type': flags.get('network_type', None),
        'aux_weight': flags.get('aux_weight', 1.),
        'network_arch': flags.get('network_arch', 'generic'),
        'dropout_prob': flags.get('dropout_prob', 0),
        'eps_fixed': flags.get('eps_fixed', False),
        'batch_norm': flags.get('use_bn', False),
        'use_gaussian_loss': flags.get('use_gaussian_loss', False),
        'use_nnehmc_loss': flags.get('use_nnehmc_loss', False),
        'clip_value': flags.get('clip_value', 0),
        'zero_masks': flags.get('zero_masks', False),
    }

    run_str = f'L{flags_dict["space_size"]}'
    if flags_dict['time_size'] != flags_dict['space_size']:
        run_str += f'T{flags_dict["time_size"]}'

    run_str += f'_b{flags_dict["batch_size"]}_lf{flags_dict["num_steps"]}'

    if flags_dict['network_arch'] != 'generic':
        run_str += f'_{flags_dict["network_arch"]}'

    if flags_dict['network_type'] is not None:
        run_str += f'_{flags_dict["network_type"]}'

    weights = OrderedDict({
        'x_scale_weight': flags.get('x_scale_weight', 1.),
        'x_translation_weight': flags.get('x_translation_weight', 1.),
        'x_transformation_weight': flags.get('x_transformation_weight', 1.),
        'v_scale_weight': flags.get('v_scale_weight', 1.),
        'v_translation_weight': flags.get('v_translation_weight', 1.),
        'v_transformation_weight': flags.get('v_transformation_weight', 1.),
    })

    all_ones = True
    for key, val in weights.items():
        flags_dict[key] = val
        if val != 1.:
            all_ones = False

    if not all_ones:
        run_str += f'_nw'
        for _, val in weights.items():
            wstr = str(int(val)).replace('.', '')
            run_str += wstr
            #  run_str += f"{str(val).replace('.', '').rstrip('0')}"

    def _no_dots(key):
        return str(flags_dict[key]).replace('.', '')

    if flags_dict['aux_weight'] != 1.:
        aw = _no_dots('aux_weight')
        run_str += f'_aw{aw}'

    if flags_dict['charge_weight'] > 0:
        qw = _no_dots('charge_weight')
        run_str += f'_qw{qw}'

    if flags_dict.get('plaq_weight', 0.) > 0.:
        pw = _no_dots('plaq_weight')
        run_str += f'_pw{pw}'

    if flags_dict['dropout_prob'] > 0:
        dp = _no_dots('dropout_prob')
        run_str += f'_dp{dp}'

    if flags_dict['eps_fixed']:
        run_str += f'_eps_fixed'

    if flags_dict['batch_norm']:
        run_str += '_bn'

    if flags_dict['use_gaussian_loss']:
        run_str += '_gaussian_loss'

    if flags_dict['use_nnehmc_loss']:
        run_str += '_nnehmc_loss'

    if flags_dict['clip_value'] > 0:
        run_str += f'_clip{flags_dict["clip_value"]}'

    if flags_dict['zero_masks']:
        run_str += f'_zero_masks'

    return run_str


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

    return run_str


def _parse_flags(FLAGS, model_type='GaugeModel'):
    """Helper method for parsing flags as both AttrDicts or generic dicts."""
    if model_type == 'GaugeModel':
        run_str = _parse_gauge_flags(FLAGS)
    elif model_type == 'GaussianMixtureModel':
        run_str = _parse_gmm_flags(FLAGS)

    if cfg.NP_FLOAT == np.float64:
        run_str += '_f64'
    elif cfg.NP_FLOAT == np.float32:
        run_str += '_f32'

    return run_str


def create_log_dir(FLAGS, model_type=None, log_file=None):
    """Automatically create and name `log_dir` to save model data to.

    The created directory will be located in `logs/YYYY_M_D/`, and will have
    the format (without `_qw{QW}` if running generic HMC):

        `lattice{LX}_batch{NS}_lf{LF}_eps{SS}_qw{QW}`

    Returns:
        FLAGS, with FLAGS.log_dir being equal to the newly created log_dir.

    NOTE: If log_dir does not already exist, it is created.
    """
    model_type = 'GaugeModel' if model_type is None else model_type
    run_str = _parse_flags(FLAGS, model_type)

    if FLAGS.train_steps < 10000:
        run_str = f'DEBUG_{run_str}'

    now = datetime.datetime.now()
    day_str = now.strftime('%Y_%m_%d')
    hour_str = now.strftime('%H%M')
    run_str += f'_{hour_str}'

    project_dir = os.path.abspath(os.path.dirname(cfg.FILE_PATH))
    log_dir = os.path.join(project_dir, FLAGS.root_dir, day_str, run_str)
    if os.path.isdir(log_dir):
        log_dir += '_1'

    #  dirname = '_'.join([run_str, hour_str])
    #  if os.path.isdir(os.path.join(project_dir, logs_dir, day_str, dirname)):
    #      dirname += '_1'

    #  log_dir = os.path.join(project_dir, logs_dir, day_str, dirname)
    check_else_make_dir(log_dir)
    #  if any('run_' in i for i in os.listdir(log_dir)):
    #      run_num = get_run_num(log_dir)
    #      log_dir = os.path.abspath(os.path.join(log_dir, f'run_{run_num}'))

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

    if out_file.endswith('.pkl'):
        out_file = change_extension(out_file, 'z')
        savez(data, out_file, name=name)

    elif out_file.endswith('.npy'):
        np.save(out_file, np.array(data))

    else:
        savez(data, out_file, name=name)


def save_params(params, out_dir, name=None):
    """save params (dict) to `out_dir`, as both `.z` and `.txt` files."""
    check_else_make_dir(out_dir)
    if name is None:
        name = 'params'
    params_txt_file = os.path.join(out_dir, f'{name}.txt')
    zfile = os.path.join(out_dir, f'{name}.z')
    with open(params_txt_file, 'w') as f:
        for key, val in params.items():
            f.write(f"{key}: {val}\n")
    savez(params, zfile, name=name)


def save_dict(d, out_dir, name):
    """Save generic dict `d` to `out_dir` as both `.z` and `.txt` files."""
    check_else_make_dir(out_dir)
    txt_file = os.path.join(out_dir, f'{name}.txt')
    with open(txt_file, 'w') as f:
        for key, val in d.items():
            f.write(f"{key}: {val}\n")

    zfile = os.path.join(out_dir, f'{name}.z')
    savez(d, zfile, name=name)


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
