"""
l2hmc/__init__.py
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from typing import Optional
import warnings

from mpi4py import MPI
from rich.logging import RichHandler
import tqdm

warnings.filterwarnings('ignore')

# import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

RANK = int(MPI.COMM_WORLD.Get_rank())


class DummyTqdmFile(object):
    """ Dummy file-like that will write to tqdm
    https://github.com/tqdm/tqdm/issues/313
    """
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        # if len(x.rstrip()) > 0:
        tqdm.tqdm.write(x, file=self.file, end='\n')

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


def get_rich_logger(
        name: Optional[str] = None,
        level: str = 'INFO'
) -> logging.Logger:
    # log: logging.Logger = get_logger(name=name, level=level)
    log = logging.getLogger(name)
    log.handlers = []
    from l2hmc.utils.rich import get_console
    console = get_console(
        markup=True,
    )
    handler = RichHandler(
        level,
        rich_tracebacks=True,
        console=console,
        show_path=False,
        enable_link_path=False
    )
    log.handlers = [handler]
    log.setLevel(level)
    return log


def get_file_logger(
        name: Optional[str] = None,
        level: str = 'INFO',
        rank_zero_only: bool = True,
        fname: Optional[str] = None,
        # rich_stdout: bool = True,
) -> logging.Logger:
    # logging.basicConfig(stream=DummyTqdmFile(sys.stderr))
    import logging
    fname = 'l2hmc' if fname is None else fname
    log = logging.getLogger(name)
    if rank_zero_only:
        fh = logging.FileHandler(f"{fname}.log")
        if RANK == 0:
            log.setLevel(level)
            fh.setLevel(level)
        else:
            log.setLevel('CRITICAL')
            fh.setLevel('CRITICAL')
    else:
        fh = logging.FileHandler(f"{fname}-{RANK}.log")
        log.setLevel(level)
        fh.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log


def get_logger(
        name: Optional[str] = None,
        level: str = 'INFO',
        rank_zero_only: bool = True,
        **kwargs,
        # rich_stdout: bool = True,
) -> logging.Logger:
    # logging.basicConfig(stream=DummyTqdmFile(sys.stderr))
    log = logging.getLogger(name)
    # log.handlers = []
    from rich.logging import RichHandler
    from l2hmc.utils.rich import get_console
    # format = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    if rank_zero_only:
        if RANK != 0:
            log.setLevel('CRITICAL')
        else:
            log.setLevel(level)
    if RANK == 0:
        console = get_console(markup=True, **kwargs)
        # log.propagate = True
        # log.handlers = []
        log.addHandler(
            RichHandler(
                omit_repeated_times=False,
                level=level,
                console=console,
                show_path=True,
                enable_link_path=False,
                # tracebacks_width=120,
                markup=True,
                # keywords=['loss=', 'dt=', 'Saving']
            )
        )
        log.setLevel(level)
    return log

def get_logger_old(
        name: Optional[str] = None,
        level: str = 'INFO',
        rank_zero_only: bool = True,
        rich_stdout: bool = True,
) -> logging.Logger:
    # logging.basicConfig(stream=DummyTqdmFile(sys.stderr))
    import logging
    # logging.basicConfig(
    #     format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    # )
    log = logging.getLogger(name)
    log.handlers = []
    from rich.logging import RichHandler
    from l2hmc.utils.rich import get_console
    if rank_zero_only:
        if rank != 0:
            log.setLevel('CRITICAL')
        else:
            log.setLevel(level)
    if rank == 0:
        console = get_console(markup=True)
        # log.propagate = True
        # log.handlers = []
        # formatter = logging.Formatter(
        #     "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
        # )
        rh = RichHandler(
            log_time_format='[%Y-%m-%d][%X]',
            omit_repeated_times=False,
            level=level,
            console=console,
            show_path=False,
            # show_time=False,
            # show_level=False,
            enable_link_path=False,
            markup=True,
        )
        # rh.formatter = None
        # rh.formatter = formatter
        # rh.setFormatter(formatter)
        rh.setLevel(level)
        log.addHandler(rh)
        log.setLevel(level)
    return log
