"""
logger.py

Contains implementation of `Logger` object for printing metrics.
"""
from __future__ import absolute_import, annotations, division, print_function

import datetime
import os
import shutil
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Union
from config import PROJECT_DIR

import joblib
import numpy as np
import tensorflow as tf
from rich import get_console
import logging
import logging.config

from rich.logging import RichHandler
from utils.log_config import logging_config

#  from utils.logger_config import in_notebook
#  from utils.logger_config import logger as log

os.environ['COLUMNS'] = str(shutil.get_terminal_size((120, 24))[0])


#  WIDTH, _ = shutil.get_terminal_size(fallback=(156, 50))
#  logging.config.fileConfig(Path(PROJECT_DIR).joinpath('logging.config'))
logging.config.dictConfig(logging_config)
log = logging.getLogger('root')
log.handlers[0] = RichHandler(markup=True, show_path=False)

def get_timestamp(fstr=None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


def strformat(k, v, window: int = 0):
    if v is None:
        v = 'None'

    outstr = ''
    if isinstance(v, bool):
        v = 'True' if v else 'False'
        return f'{str(k)}={v}'

    if isinstance(v, dict):
        outstr_arr = []
        for key, val in v.items():
            outstr_arr.append(strformat(key, val, window))
        outstr = '\n'.join(outstr_arr)
    else:
        #  if isinstance(v, (tuple, list, np.ndarray)):
        if isinstance(v, (tuple, list, np.ndarray, tf.Tensor)):
            v = np.array(v)
            if window > 0 and len(v.shape) > 0:
                window = min((v.shape[0], window))
                avgd = np.mean(v[-window:])
            else:
                avgd = np.mean(v)
            outstr = f'{str(k)}={avgd:<4.3f}'
        else:
            if isinstance(v, float):
                outstr = f'{str(k)}={v:<4.3f}'
            else:
                try:
                    outstr = f'{str(k)}={v:<4f}'
                except ValueError:
                    outstr = f'{str(k)}={v:<4}'
    return outstr


def in_notebook():
    """Check if we're currently in a jupyter notebook."""
    try:
        # pylint:disable=import-outside-toplevel
        from IPython import get_ipython
        try:
            if 'IPKernelApp' not in get_ipython().config:
                return False
        except AttributeError:
            return False
    except ImportError:
        return False
    return True


# noqa: E999
# pylint:disable=too-few-public-methods,redefined-outer-name
# pylint:disable=missing-function-docstring,missing-class-docstring
class Console:
    """Fallback console object used as in case `rich` isn't installed."""
    def rule(self, s, *args, **kwargs):
        line = len(s) * '-'
        log.info('\n'.join([line, s, line]), *args, **kwargs)
        #  self.log('\n'.join([line, s, line]), *args, **kwargs)

    @staticmethod
    def log(s, *args, **kwargs):
        now = get_timestamp('%X')
        log.info(f'[{now}]  {s}', *args, **kwargs)



class Logger:
    """Logger class for pretty printing metrics during training/testing."""
    def __init__(self, theme: dict = None):
        try:
            # pylint:disable=import-outside-toplevel
            from rich.console import Console as RichConsole
            from rich.theme import Theme

            #  with_jupyter = in_notebook()
            console = get_console()
            width = os.environ.get('COLUMNS', 120)
            console = RichConsole(record=False, log_path=False,
                                  #  force_jupyter=with_jupyter,
                                  #  force_terminal=(not with_jupyter),
                                  log_time_format='[%x %X] ')
                                  #  theme=Theme(theme))#, width=width)

        except (ImportError, ModuleNotFoundError):
            console = Console()

        self.console = console

    def error(self, s: str, *args, **kwargs):
        log.error(s, *args, **kwargs)

    def debug(self, s: str, *args, **kwargs):
        log.debug(s, *args, **kwargs)

    def warning(self, s: str, *args, **kwargs):
        log.warning(s, *args, **kwargs)

    def rule(self, s: str, *args, **kwargs):
        """Print horizontal line."""
        self.console.rule(s, *args, **kwargs)

    def info(self, s: Any, *args, **kwargs):
        log.info(s, *args, **kwargs)

    def load_metrics(self, infile: str = None):
        """Try loading metrics from infile."""
        return joblib.load(infile)

    def log(self, s: Any, *args, **kwargs):
        """Print `s` using `self.console` object."""
        if is_dataclass(s):
            s = asdict(s)
        if isinstance(s, dict):
             _ = self.print_dict(s)
             return

        log.info(s, *args, **kwargs)

    def print_metrics(
            self,
            metrics: dict,
            window: int = 0,
            outfile: str = None,
            skip: list[str] = None,
            keep: list[str] = None,
            pre: Union[str, list, tuple] = None,
    ):
        """Print nicely formatted string of summary of items in `metrics`."""
        if skip is None:
            skip = []
        if keep is None:
            keep = list(metrics.keys())

        fstrs = [
            strformat(k, v, window) for k, v in metrics.items()
            if k not in skip
            and k in keep
        ]
        if pre is not None:
            fstrs = [pre, *fstrs] if isinstance(pre, str) else [*pre] + fstrs

        outstr = ' '.join(fstrs)
        self.log(outstr)
        #  log.info(outstr)
        if outfile is not None:
            with open(outfile, 'a') as f:
                f.write(outstr)

        return outstr

    def dict_to_str(self, d: dict, indent: int = 0, name: str = None):
        kvstrs = []
        pre = indent * ' '
        if name is not None:
            nstr = f'{str(name)}'
            line = len(nstr) * '-'
            kvstrs.extend([pre + nstr, pre + line])
        for key, val in d.items():
            if is_dataclass(val):
                val = asdict(val)

            if isinstance(val, dict):
                strs = self.dict_to_str(val, indent=indent+2, name=key)
            else:
                strs = pre + '='.join([str(key), str(val)])

            kvstrs.append(strs)

        dstr = '\n'.join(kvstrs)

        return dstr

    def print_dict(self, d: dict, indent: int = 0, name: str = None):
        dstr = self.dict_to_str(d, indent, name)
        log.info(dstr)

    def save_metrics(
            self,
            metrics: dict,
            outfile: str = None,
            tstamp: str = None,
    ):
        """Save metrics to compressed `.z.` file."""
        if tstamp is None:
            tstamp = get_timestamp('%Y-%m-%d-%H%M%S')
        if outfile is None:
            outdir = os.path.join(os.getcwd(), tstamp)
            fname = 'metrics.z'
        else:
            outdir, fname = os.path.split(outfile)
        check_else_make_dir(outdir)
        outfile = os.path.join(os.getcwd(), tstamp, 'metrics.z')
        #  self.log(f'Saving metrics to: {os.path.relpath(outdir)}')
        (f'Saving metrics to: {os.path.relpath(outdir)}')
        savez(metrics, outfile, name=fname.split('.')[0])


logger = Logger()


def print_dict(d: dict, indent=0, name: str = None):
    kv_strs = []
    pre = indent * ' '

    if name is not None:
        nstr = f'{str(name)}'
        line = len(nstr) * '-'
        kv_strs.extend([pre + nstr, pre + line])

    for key, val in d.items():
        if is_dataclass(val):
            val = asdict(val)
        if isinstance(val, dict):
            strs = print_dict(val, indent=indent+2, name=key)
        else:
            strs = pre + '='.join([str(key), str(val)])

        kv_strs.append(strs)

    return '\n'.join(kv_strs)


def check_else_make_dir(outdir: Union[str, Path, list, tuple]):
    if isinstance(outdir, (str, Path)) and not os.path.isdir(str(outdir)):
        logger.log(f'Creating directory: {os.path.relpath(outdir)}')
        os.makedirs(str(outdir))

    elif isinstance(outdir, (tuple, list)):
        _ = [check_else_make_dir(str(d)) for d in outdir]


def loadz(infile: str):
    return joblib.load(infile)


def savez(obj: Any, fpath: str, name: str = None):
    """Save `obj` to compressed `.z` file at `fpath`."""
    head, _ = os.path.split(fpath)

    check_else_make_dir(head)

    if not fpath.endswith('.z'):
        fpath += '.z'

    if name is not None:
        logger.log(f'Saving {name} to {os.path.relpath(fpath)}.')
    else:
        logger.log(f'Saving {obj.__class__} to {os.path.relpath(fpath)}.')

    joblib.dump(obj, fpath)
