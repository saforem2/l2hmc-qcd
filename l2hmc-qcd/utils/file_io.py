"""
Helper methods for performing file IO.

Author: Sam Foreman (github: @saforem2)
Created: 2/27/2019
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle

# pylint:disable=invalid-name

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
    hvd.init()

except ImportError:
    HAS_HOROVOD = False

from globals import ROOT_DIR, PROJECT_DIR, FILE_PATH


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


def log(s, nl=True):
    """Print string `s` to stdout if and only if hvd.rank() == 0."""
    try:
        if HAS_HOROVOD and hvd.rank() != 0:
            return
        print(s, end='\n' if nl else '')
    except NameError:
        print(s, end='\n' if nl else '')



def check_else_make_dir(d):
    """If directory `d` doesn't exist, it is created."""
    if not os.path.isdir(d):
        try:
            log(f"Creating directory: {d}")
            os.makedirs(d)
        except OSError:
            pass


def make_dirs(dirs):
    """Make directories if and only if hvd.rank == 0."""
    _ = [check_else_make_dir(d) for d in dirs]


def save_params_to_pkl_file(params, out_dir):
    """Save `params` dictionary to `parameters.pkl` in `out_dir.`"""
    check_else_make_dir(out_dir)
    params_file = os.path.join(out_dir, 'parameters.pkl')
    #  print(f"Saving params to: {params_file}.")
    log(f"Saving params to: {params_file}.")
    with open(params_file, 'wb') as f:
        pickle.dump(params, f)


def create_log_dir(root_dir='gauge_logs_graph'):
    root_log_dir = os.path.join(PROJECT_DIR, root_dir)
    check_else_make_dir(root_log_dir)
    try:
        run_dirs = [i for i in os.listdir(root_log_dir) if 'run' in i]
        run_nums = [int(i.split('_')[-1]) for i in run_dirs]
        run_num = sorted(run_nums)[-1] + 1
    except:
        run_num = 1

    log_dir = os.path.join(root_log_dir, f'run_{run_num}')
    try:
        check_else_make_dir(log_dir)
    except:
        pass

    return log_dir


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
    except ValueError:
        log(f"No previous runs found in {log_dir}, setting run_num=1.")
        run_num = 1

    return run_num





def _get_run_num(log_dir):
    #  if not os.path.isdir(log_dir):
    #      os.makedirs(log_dir)
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
