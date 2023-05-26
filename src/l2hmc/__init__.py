"""
l2hmc/__init__.py
"""
from __future__ import absolute_import, annotations, division, print_function
import logging

# import os
from typing import Optional
from mpi4py import MPI
import tqdm
# from pathlib import Path

# from colorlog import ColoredFormatter


# # formatter = ColoredFormatter(
# # 	"%(log_color)s%(levelname)-8s%(reset)s %(message_log_color)s%(message)s",
# # 	secondary_log_colors={
# # 		'message': {
# # 			'ERROR':    'red',
# # 			'CRITICAL': 'red'
# # 		}
# # 	}
# # )

# formatter = ColoredFormatter(
# 	"%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
# 	datefmt=None,
# 	reset='%X',
# 	log_colors={
# 		'DEBUG':    'cyan',
# 		'INFO':     'green',
# 		'WARNING':  'yellow',
# 		'ERROR':    'red',
# 		'CRITICAL': 'red,bg_white',
# 	},
# 	secondary_log_colors={},
# 	style='%'
# )

# handler = colorlog.StreamHandler()
# handler.setFormatter(colorlog.ColoredFormatter(
# 	'%(log_color)s%(levelname)s:%(name)s:%(message)s'))

# logger = colorlog.getLogger('example')
# logger.addHandler(handler)

# the handler determines where the logs go: stdout/file
# shell_handler = RichHandler()
# file_handler = logging.FileHandler("debug.log")

# logger.setLevel(logging.DEBUG)
# shell_handler.setLevel(logging.DEBUG)
# file_handler.setLevel(logging.DEBUG)

# # the formatter determines what our logs will look like
# fmt_shell = '%(message)s'
# fmt_file = (
#     '%(levelname)s %(asctime)s '
#     '[%(filename)s:%(funcName)s:%(lineno)d] '
#     '%(message)s'
# )

# shell_formatter = logging.Formatter(fmt_shell)
# file_formatter = logging.Formatter(fmt_file)

# # here we hook everything together
# shell_handler.setFormatter(shell_formatter)
# file_handler.setFormatter(file_formatter)

# logger.addHandler(shell_handler)
# logger.addHandler(file_handler)

# from tqdm.contrib import DummyTqdmFile


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
    from rich.logging import RichHandler
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


def get_logger(
        name: Optional[str] = None,
        level: str = 'INFO',
) -> logging.Logger:
    rank = int(MPI.COMM_WORLD.Get_rank())
    # logging.basicConfig(stream=DummyTqdmFile(sys.stderr))
    log = logging.getLogger(name)
    log.handlers = []
    if rank != 0:
        log.setLevel('CRITICAL')
    else:
        # from rich.console import Console
        from rich.logging import RichHandler
        from l2hmc.utils.rich import get_console
        # console = Console(
        #     log_path=False,
        #     width=int(os.environ.get('COLUMNS', 255)),
        #     markup=True,
        # )
        console = get_console(markup=True)
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


# log = get_logger(__name__, level='INFO')
