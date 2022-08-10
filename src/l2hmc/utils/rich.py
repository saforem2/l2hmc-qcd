"""
rich.py

Contains utils for textual layouts using Rich
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Optional
from typing import Any

from omegaconf import DictConfig, OmegaConf
import pandas as pd
import rich
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import rich.syntax
from rich.table import Table
import rich.tree

from l2hmc.configs import Steps
from l2hmc.configs import OUTPUTS_DIR


log = logging.getLogger(__name__)

# from typing import Any, Callable, Optional
# from rich import box
# from rich.live import Live
# import logging
# import time
# import numpy as np


# WIDTH = max(150, int(os.environ.get('COLUMNS', 150)))
size = shutil.get_terminal_size()
WIDTH = size.columns
HEIGHT = size.lines
# os.environ['COLUMNS'] = f'{size.columns}'


def is_interactive():
    from IPython import get_ipython
    return get_ipython() is not None


def get_width():
    width = os.environ.get('COLUMNS', os.environ.get('WIDTH', None))
    if width is not None:
        return int(width)

    size = shutil.get_terminal_size()
    os.environ['COLUMNS'] = str(size.columns)
    return size.columns


def get_console(width: Optional[int] = None, *args, **kwargs) -> Console:
    interactive = is_interactive()
    console = Console(
        force_jupyter=interactive,
        log_path=False,
        # color_system='truecolor',
        *args,
        **kwargs)
    if width is None:
        if is_interactive():
            columns = 100
        else:
            columns = os.environ.get('COLUMNS', os.environ.get('WIDTH', None))
            if columns is None:
                if not interactive:
                    size = shutil.get_terminal_size()
                    columns = size.columns
                else:
                    columns = 120
            else:
                columns = int(columns)

        width = int(max(columns, 120))
        console.width = width
        console._width = width

    return console


def make_layout(ratio: int = 4, visible: bool = True) -> Layout:
    """Define the layout."""
    layout = Layout(name='root', visible=visible)
    layout.split_row(
        Layout(name='main', ratio=ratio, visible=visible),
        Layout(name='footer', visible=visible),
    )
    # Layout(name='left'),
    # Layout(name='main', ratio=3),
    # layout['right'].split_column(
    #     Layout(name='top'),
    #     Layout(name='footer')
    # )
    # layout['footer'].split_column(
    #     Layout(name='top'),
    #     Layout(name='bottom'),
    # )
    return layout


def build_layout(
        steps: Steps,
        visible: bool = True,
        job_type: Optional[str] = 'train',
        # columns: Optional[list[str]] = None
) -> dict:
    job_progress = Progress(
        "{task.description}",
        SpinnerColumn('dots'),
        BarColumn(),
        TimeElapsedColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )
    tasks = {}
    border_style = 'white'
    if job_type == 'train':
        border_style = 'green'
        tasks['step'] = job_progress.add_task(
            "[blue]Total",
            total=(steps.nera * steps.nepoch),
        )
        # tasks['era'] = job_progress.add_task(
        #     "[blue]Era",
        #     total=steps.nera
        # )
        tasks['epoch'] = job_progress.add_task(
            "[cyan]Epoch",
            total=steps.nepoch
        )
    elif job_type == 'eval':
        border_style = 'green'
        tasks['step'] = job_progress.add_task(
            "[green]Eval",
            total=steps.test,
        )
    elif job_type == 'hmc':
        border_style = 'yellow'
        tasks['step'] = job_progress.add_task(
            "[green]HMC",
            total=steps.test,
        )
    else:
        raise ValueError(
            'Expected job_type to be one of train, eval, or HMC,\n'
            f'Received: {job_type}'
        )

    # total = sum(task.total for task in job_progress.tasks)
    # overall_progress = Progress()
    # overall_task = overall_progress.add_task("All jobs", total=int(total))

    progress_table = Table.grid(expand=True)
    progress_table.add_row(
        Panel.fit(
            job_progress,
            title=f'[b]{job_type}',
            border_style=border_style,
            # padding=(1, 1),
        )
    )
    # avgs_table = Table.grid(expand=True)
    # avgs_table.add_row(
    #     Panel.fit(
    #         ' ',
    #         title='Avgs:',
    #         border_style='white',
    #     )
    # )
    layout = make_layout(visible=visible)
    if visible:
        layout['root']['footer'].update(progress_table)
    # layout['root']['right']['top'].update(Panel.fit(' '))
    # if columns is not None:
    #     layout['root']['main'].update(Panel.fit(columns))
    #     # add_row(Panel.fit(columns))
    # layout['root']['footer']['bottom'].update(avgs_table)

    return {
        'layout': layout,
        'tasks': tasks,
        'progress_table': progress_table,
        'job_progress': job_progress,
    }


def add_columns(
    avgs: dict,
    table: Table,
    skip: Optional[str | list[str]] = None,
    keep: Optional[str | list[str]] = None,
) -> Table:
    for key in avgs.keys():
        if skip is not None and key in skip:
            continue
        if keep is not None and key not in keep:
            continue

        if key == 'loss':
            table.add_column(str(key),
                             justify='center',
                             style='green')
        elif key == 'dt':
            table.add_column(str(key),
                             justify='center',
                             style='red')

        elif key == 'acc':
            table.add_column(str(key),
                             justify='center',
                             style='magenta')
        elif key == 'dQint':
            table.add_column(str(key),
                             justify='center',
                             style='cyan')
        elif key == 'dQsin':
            table.add_column(str(key),
                             justify='center',
                             style='yellow')
        else:
            table.add_column(str(key),
                             justify='center')

    return table


def flatten_dict(d) -> dict:
    res = {}
    if isinstance(d, dict):
        for k in d:
            if k == '_target_':
                continue

            dflat = flatten_dict(d[k])
            for key, val in dflat.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = d

    return res


def nested_dict_to_df(d):
    dflat = flatten_dict(d)
    df = pd.DataFrame.from_dict(dflat, orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df


def print_config(
    config: DictConfig,
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config
            components are printed.
        resolve (bool, optional): Whether to resolve reference fields of
            DictConfig.
    """

    # style = "dim"
    tree = rich.tree.Tree("CONFIG")  # , style=style, guide_style=style)

    quee = []
    # yaml_strs = ""

    for f in config:
        if f not in quee:
            quee.append(f)

    dconfig = {}
    for f in quee:

        branch = tree.add(f)  # , style=style, guide_style=style)

        config_group = config[f]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
            cfg = OmegaConf.to_container(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)
            cfg = str(config_group)

        dconfig[f] = cfg
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    outfile = Path(os.getcwd()).joinpath('config_tree.log')
    # with open(outfile, 'wt') as f:
    with outfile.open('wt') as f:
        console = rich.console.Console(file=f)
        console.print(tree)

    with open('config.json', 'w') as f:
        f.write(json.dumps(dconfig))

    cfgfile = Path('config.yaml')
    OmegaConf.save(config, cfgfile, resolve=True)
    cfgdict = OmegaConf.to_object(config)
    logdir = Path(os.getcwd()).resolve().as_posix()
    if not config.get('debug_mode', False):
        dbfpath = Path(OUTPUTS_DIR).joinpath('logdirs.csv')
    else:
        dbfpath = Path(OUTPUTS_DIR).joinpath('logdirs-debug.csv')

    df = pd.DataFrame({logdir: cfgdict})
    df.T.to_csv(dbfpath.resolve().as_posix(), mode='a')


@dataclass
class CustomLogging:
    version: int = 1
    formatters: dict[str, Any] = field(
        default_factory=lambda: {
            'simple': {
                'format': (
                    '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
                )
            }
        }
    )
    handlers: dict[str, Any] = field(
        default_factory=lambda: {
            'console': {
                'class': 'rich.logging.RichHandler',
                'formatter': 'simple',
                'rich_tracebacks': 'true'
            },
            'file': {
                'class': 'logging.FileHander',
                'formatter': 'simple',
                'filename': '${hydra.job.name}.log',
            },
        }
    )
    root: dict[str, Any] = field(
        default_factory=lambda: {
            'level': 'INFO',
            'handlers': ['console', 'file'],
        }
    )
    disable_existing_loggers: bool = False
