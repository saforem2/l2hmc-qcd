"""
rich.py

Contains utils for textual layouts using Rich
"""
from __future__ import absolute_import, annotations, division, print_function
import os
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig
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
    TimeRemainingColumn
)
import rich.syntax
from rich.table import Table
import rich.tree

# from typing import Any, Callable, Optional
# from rich import box
# from rich.live import Live
# import logging
# import time
# import numpy as np

from l2hmc.configs import Steps

WIDTH = max(150, int(os.environ.get('COLUMNS', 150)))


def is_interactive():
    from IPython import get_ipython
    return get_ipython() is not None


console = Console(record=False,
                  color_system='truecolor',
                  log_path=False,
                  width=WIDTH)
console.width = WIDTH


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


def add_columns(avgs: dict, table: Table) -> Table:
    for key in avgs.keys():
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

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field)  # , style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    outfile = Path(os.getcwd()).joinpath('config_tree.log')
    with outfile.open('w') as f:
        rich.print(tree, file=f)
