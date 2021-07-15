"""
logger.py

Contains implementation of `Logger` object for printing metrics.
"""
from __future__ import absolute_import, division, print_function, annotations
from dataclasses import asdict, is_dataclass
import shutil
import datetime
import numpy as np
import tensorflow as tf

try:
    import horovod
    import horovod.tensorflow as hvd
    try:
        RANK = hvd.rank()
    except AttributeError:
        hvd.init()

    RANK = hvd.rank()
except (ImportError, ModuleNotFoundError):
    RANK = 0


WIDTH, _ = shutil.get_terminal_size(fallback=(156, 50))

def get_timestamp(fstr=None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


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
            outstr = f'{str(k)}={avgd:<5.4g}'
        else:
            if isinstance(v, float):
                outstr = f'{str(k)}={v:<5.4g}'
            else:
                try:
                    outstr = f'{str(k)}={v:<5g}'
                except ValueError:
                    outstr = f'{str(k)}={v:<5}'
    return outstr




# noqa:E999
class Console:
    """Fallback console object used in case `rich` isn't installed."""
    @staticmethod
    def log(s, *args, **kwargs):
        if RANK == 0:
            now = get_timestamp('%X')
            print(f'[{now}] {s}', *args, **kwargs)


class Logger:
    """Logger class for pretty printing metrics during training/testing."""
    def __init__(self, width=None):
        self.rank = RANK
        if width is None:
            width = 120

        try:
            # pylint:disable=import-outside-toplevel
            from rich.console import Console as RichConsole
            from rich.theme import Theme
            theme = None
            if in_notebook():
                theme = Theme({
                    'repr.number': 'bold bright_green',
                    'repr.attrib_name': 'bold bright_magenta',
                    'repr.str': '#FFFF00',
                })
            console = RichConsole(record=False, log_path=False,
                                  force_jupyter=in_notebook(),
                                  width=width, theme=theme,
                                  log_time_format='[%X] ')
        except (ImportError, ModuleNotFoundError):
            console = Console()

        self.width = width
        self.console = console

    def rule(self, s: str, *args, **kwargs):
        """Print horizontal line."""
        #  w = self.width - (8 + len(s))
        #  hw = w // 2
        #  rule = ' '.join((hw * '-', f'{s}', hw * '-'))
        #  self.console.log(f'{rule}\n', *args, **kwargs)
        self.console.rule(f'{s}', *args, **kwargs)

    def log(self, s: str, *args, **kwargs):
        self.console.log(s, *args, **kwargs)

    def print_metrics(
        self,
        metrics: dict,
        window: int = 0,
        pre: list = None,
        outfile: str = None,
        keep: list[str] = None,
        skip: list[str] = None,
    ):
        """Print nicely formatted string of summary of items in `metrics`."""
        if self.rank != 0:
            return

        if keep is None:
            keep = list(metrics.keys())

        if skip is None:
            skip = []

        outstr = ' '.join([
            strformat(k, v, window) for k, v in metrics.items()
            if k not in skip
            and k in keep
        ])
        if pre is not None:
            outstr = ' '.join([*pre, outstr])

        self.log(outstr)
        if outfile is not None:
            with open(outfile, 'a') as f:
                f.write(outstr)

        return outstr

    def save_metrics(
        self,
        metrics: dict,
        outfile: str = None,
        tstamp: str = None,
    ):
        """Save `metrics` to compressed `.z.` file."""
        #  if tstamp is None:
        #      tstamp = get_timestamp('%Y-%m-%d-%H%M%S')
        #
        #  if outfile is None:
        #      outdir = os.path.join(os.getcwd(), tstamp)
        #      fname = 'metrics.z'
        #  else:
        #      outdir, fname = os.path.split(outfile)
        #
        #  check_else_make_dir(outdir)
        #  outfile = os.path.join(os.getcwd(), tstamp
        # TODO: rethink this
        pass


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
