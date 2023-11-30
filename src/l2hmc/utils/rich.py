"""
rich.py

Contains utils for textual layouts using Rich
"""
from __future__ import absolute_import, annotations, division, print_function
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import time
from typing import Optional
from typing import Any
from typing import Generator

import logging
from enrich.style import STYLES
from enrich.console import Console
from enrich.handler import RichHandler
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import rich
from rich import print
from rich.box import MINIMAL, SIMPLE, SIMPLE_HEAD, SQUARE
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.measure import Measurement
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

# from l2hmc import get_logger

# from l2hmc.configs import Steps


log = logging.getLogger(__name__)

# WIDTH = max(150, int(os.environ.get('COLUMNS', 150)))
size = shutil.get_terminal_size()
WIDTH = size.columns
HEIGHT = size.lines
os.environ['COLUMNS'] = f'{WIDTH}'
# os.environ['COLUMNS'] = f'{size.columns}'

# STYLES = {
#     # 'color': Style(color='#676767'),
#     # 'filepath': Style(color='#50FA7B', bold=True),
#     # 'filename': Style(color='#BD93F9', bold=True),
#     # # 'info': Style(color='#29B6F6'),
#     # # 'warning': Style(color='#FD971F'),
#     # # 'error': Style(color='#FF5252', bold=True),
#     # # 'logging.level.warning': Style(color='#FD971F'),
#     # 'logging.level.info': Style(color='#29B6F6'),
#     # 'logging.level.warning': Style(color='#FD971F'),
#     # 'logging.level.error': Style(color='#FF5252'),
#     # 'yellow': Style(color='#FFFF00'),
#     # "time": Style(color="#676767"),
#     # 'log.time': Style(color='#676767'),
#     # # 'repr.attrib_name': Style(color="#676767"),
#     # "hidden": Style(color="#383b3d", dim=True),
#     # "num": Style(color='#409CDC', bold=True),
#     # # 'repr.number': Style(color='#409CDC', bold=False),
#     # "highlight": Style(color="#111111", bgcolor="#FFFF00", bold=True),
# }
#


def get_console(**kwargs) -> Console:
    interactive = is_interactive()
    from rich.theme import Theme
    theme = Theme(STYLES)
    return Console(
        force_jupyter=interactive,
        log_path=False,
        theme=theme,
        soft_wrap=True,
        # file=outfile,
        # redirect=(get_world_size() > 1),
        # width=int(width),
        **kwargs
    )


def is_interactive() -> bool:
    from IPython.core.getipython import get_ipython
    # from IPython import get_ipython
    eval = os.environ.get('INTERACTIVE', None) is not None
    bval = get_ipython() is not None
    return (eval or bval)


def get_width():
    width = os.environ.get('COLUMNS', os.environ.get('WIDTH', 255))
    if width is not None:
        return int(width)

    size = shutil.get_terminal_size()
    os.environ['COLUMNS'] = str(size.columns)
    return size.columns


def make_layout(ratio: int = 4, visible: bool = True) -> Layout:
    """Define the layout."""
    layout = Layout(name='root', visible=visible)
    layout.split_row(
        Layout(name='main', ratio=ratio, visible=visible),
        Layout(name='footer', visible=visible),
    )
    return layout


def build_layout(
        steps: Any,
        visible: bool = True,
        job_type: Optional[str] = 'train',
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
    for key in avgs:
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
    from l2hmc.configs import OUTPUTS_DIR
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

    if dbfpath.is_file():
        mode = 'a'
        header = False
    else:
        mode = 'w'
        header = True
    df = pd.DataFrame({logdir: cfgdict})
    df.T.to_csv(
        dbfpath.resolve().as_posix(),
        mode=mode,
        header=header
    )
    os.environ['LOGDIR'] = logdir


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


def printarr(*arrs, float_width=6):
    """
    Print a pretty table giving name, shape, dtype, type, and content
    information for input tensors or scalars.

    Call like: printarr(my_arr, some_other_arr, maybe_a_scalar). Accepts a
    variable number of arguments.

    Inputs can be:
        - Numpy tensor arrays
        - Pytorch tensor arrays
        - Jax tensor arrays
        - Python ints / floats
        - None

    It may also work with other array-like types, but they have not been tested

    Use the `float_width` option specify the precision to which floating point
    types are printed.

    Author: Nicholas Sharp (nmwsharp.com)
    Canonical source:
        https://gist.github.com/nmwsharp/54d04af87872a4988809f128e1a1d233
    License: This snippet may be used under an MIT license, and it is also
    released into the public domain. Please retain this docstring as a
    reference.
    """
    import inspect
    frame_ = inspect.currentframe()
    assert frame_ is not None
    frame = frame_.f_back
    # if frame_ is not None:
    #     frame = frame_.f_back
    # else:
    #     frame = inspect.getouterframes()
    default_name = "[temporary]"

    # helpers to gather data about each array

    def name_from_outer_scope(a):
        if a is None:
            return '[None]'
        name = default_name
        if frame_ is not None:
            for k, v in frame_.f_locals.items():
                if v is a:
                    name = k
                    break
        return name

    def dtype_str(a):
        if a is None:
            return 'None'
        if isinstance(a, int):
            return 'int'
        return 'float' if isinstance(a, float) else str(a.dtype)

    def shape_str(a):
        if a is None:
            return 'N/A'
        if isinstance(a, int):
            return 'scalar'
        return 'scalar' if isinstance(a, float) else str(list(a.shape))

    def type_str(a):
        # TODO this is is weird... what's the better way?
        return str(type(a))[8:-2]

    def device_str(a):
        if hasattr(a, 'device'):
            device_str = str(a.device)
            if len(device_str) < 10:
                # heuristic: jax returns some goofy long string we don't want,
                # ignore it
                return device_str
        return ""

    def format_float(x):
        return f"{x:{float_width}g}"

    def minmaxmean_str(a):
        if a is None:
            return ('N/A', 'N/A', 'N/A')
        if isinstance(a, (int, float)):
            return (format_float(a), format_float(a), format_float(a))

        # compute min/max/mean. if anything goes wrong, just print 'N/A'
        min_str = "N/A"
        try:
            min_str = format_float(a.min())
        except Exception:
            pass
        max_str = "N/A"
        try:
            max_str = format_float(a.max())
        except Exception:
            pass
        mean_str = "N/A"
        try:
            mean_str = format_float(a.mean())
        except Exception:
            pass

        return (min_str, max_str, mean_str)

    try:
        props = ['name', 'dtype', 'shape', 'type', 'device', 'min', 'max',
                 'mean']

        # precompute all of the properties for each input
        str_props = []
        for a in arrs:
            minmaxmean = minmaxmean_str(a)
            str_props.append({
                'name': name_from_outer_scope(a),
                'dtype': dtype_str(a),
                'shape': shape_str(a),
                'type': type_str(a),
                'device': device_str(a),
                'min': minmaxmean[0],
                'max': minmaxmean[1],
                'mean': minmaxmean[2],
            })

        # for each property, compute its length
        maxlen = {}
        for p in props:
            maxlen[p] = 0
        for sp in str_props:
            for p in props:
                maxlen[p] = max(maxlen[p], len(sp[p]))

        # if any property got all empty strings,
        # don't bother printing it, remove if from the list
        props = [p for p in props if maxlen[p] > 0]

        # print a header
        header_str = ""
        for p in props:
            prefix = "" if p == 'name' else " | "
            fmt_key = ">" if p == 'name' else "<"
            header_str += f"{prefix}{p:{fmt_key}{maxlen[p]}}"
        print(header_str)
        print("-"*len(header_str))
        # now print the acual arrays
        for strp in str_props:
            for p in props:
                prefix = "" if p == 'name' else " | "
                fmt_key = ">" if p == 'name' else "<"
                print(f"{prefix}{strp[p]:{fmt_key}{maxlen[p]}}", end='')
            print("")

    finally:
        del frame

        import time


# console = Console()

BEAT_TIME = 0.008

COLORS = ["cyan", "magenta", "red", "green", "blue", "purple"]

# log = get_logger(__name__)
log = logging.getLogger(__name__)
handlers = log.handlers
if (
        len(handlers) > 0
        and isinstance(handlers[0], RichHandler)
):
    console = handlers[0].console
else:
    console = get_console(markup=True)


@contextmanager
def beat(length: int = 1) -> Generator:
    with console:
        yield
    time.sleep(length * BEAT_TIME)


class DataFramePrettify:
    """Create animated and pretty Pandas DataFrame.

    Modified from: https://github.com/khuyentran1401/rich-dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The data you want to prettify
    row_limit : int, optional
        Number of rows to show, by default 20
    col_limit : int, optional
        Number of columns to show, by default 10
    first_rows : bool, optional
        Whether to show first n rows or last n rows, by default True.
        If this is set to False, show last n rows.
    first_cols : bool, optional
        Whether to show first n columns or last n columns, by default True.
        If this is set to False, show last n rows.
    delay_time : int, optional
        How fast is the animation, by default 5.
        Increase this to have slower animation.
    clear_console: bool, optional
         Clear the console before printing the table, by default True.
         If this is set to False the previous console
         input/output is maintained
    """

    def __init__(
        self,
        df: pd.DataFrame,
        row_limit: int = 20,
        col_limit: int = 10,
        first_rows: bool = True,
        first_cols: bool = True,
        delay_time: int = 5,
        clear_console: bool = True,
    ) -> None:
        self.df = df.reset_index().rename(columns={"index": ""})
        self.table = Table(show_footer=False)
        self.table_centered = Columns(
            (self.table,), align="center", expand=True
        )
        self.num_colors = len(COLORS)
        self.delay_time = delay_time
        self.row_limit = row_limit
        self.first_rows = first_rows
        self.col_limit = col_limit
        self.first_cols = first_cols
        self.clear_console = clear_console
        if first_cols:
            self.columns = self.df.columns[:col_limit]
        else:
            self.columns = list(self.df.columns[-col_limit:])
            self.columns.insert(0, "index")
        if first_rows:
            self.rows = self.df.values[:row_limit]
        else:
            self.rows = self.df.values[-row_limit:]
        if self.clear_console:
            console.clear()

    def _add_columns(self):
        for col in self.columns:
            with beat(self.delay_time):
                self.table.add_column(str(col))

    def _add_rows(self):
        for row in self.rows:
            with beat(self.delay_time):

                row = row[: self.col_limit] if self.first_cols else row[-self.col_limit:]
                row = [str(item) for item in row]
                self.table.add_row(*list(row))

    def _move_text_to_right(self):
        for i in range(len(self.table.columns)):
            with beat(self.delay_time):
                self.table.columns[i].justify = "right"

    def _add_random_color(self):
        for i in range(len(self.table.columns)):
            with beat(self.delay_time):
                self.table.columns[i].header_style = COLORS[
                    i % self.num_colors
                ]

    def _add_style(self):
        for i in range(len(self.table.columns)):
            with beat(self.delay_time):
                self.table.columns[i].style = (
                    "bold " + COLORS[i % self.num_colors]
                )

    def _adjust_box(self):
        for box in [SIMPLE_HEAD, SIMPLE, MINIMAL, SQUARE]:
            with beat(self.delay_time):
                self.table.box = box

    def _dim_row(self):
        with beat(self.delay_time):
            self.table.row_styles = ["none", "dim"]

    def _adjust_border_color(self):
        with beat(self.delay_time):
            self.table.border_style = "bright_yellow"

    def _change_width(self):
        original_width = Measurement.get(
            console=console,
            options=console.options,
            renderable=self.table
        ).maximum
        width_ranges = [
            [original_width, console.width, 2],
            [console.width, original_width, -2],
            [original_width, 90, -2],
            [90, original_width + 1, 2],
        ]

        for width_range in width_ranges:
            for width in range(*width_range):
                with beat(self.delay_time):
                    self.table.width = width

            with beat(self.delay_time):
                self.table.width = None

    def _add_caption(self):
        row_text = "first" if self.first_rows else "last"
        col_text = "first" if self.first_cols else "last"
        with beat(self.delay_time):
            self.table.caption = (
                f"Only the {row_text} "
                f"{self.row_limit} rows "
                f"and the {col_text} "
                f"{self.col_limit} columns "
                "is shown here."
            )
        with beat(self.delay_time):
            self.table.caption = (
                f"Only the [bold green] {row_text} "
                "{self.row_limit} rows[/bold green] and the "
                "[bold red]{self.col_limit} {col_text} "
                "columns[/bold red] is shown here."
            )
        with beat(self.delay_time):
            self.table.caption = (
                f"Only the [bold magenta not dim] "
                f"{row_text} {self.row_limit} rows "
                f"[/bold magenta not dim] and the "
                f"[bold green not dim]{col_text} "
                f"{self.col_limit} columns "
                f"[/bold green not dim] "
                f"are shown here."
            )

    def prettify(self):
        with Live(
            self.table_centered,
            console=console,
            refresh_per_second=self.delay_time,
            vertical_overflow="ellipsis",
        ):
            self._add_columns()
            self._add_rows()
            self._move_text_to_right()
            self._add_random_color()
            self._add_style()
            self._adjust_border_color()
            self._add_caption()

        return self.table


def prettify(
    df: pd.DataFrame,
    row_limit: int = 20,
    col_limit: int = 10,
    first_rows: bool = True,
    first_cols: bool = True,
    delay_time: int = 5,
    clear_console: bool = True,
):
    """Create animated and pretty Pandas DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        The data you want to prettify
    row_limit : int, optional
        Number of rows to show, by default 20
    col_limit : int, optional
        Number of columns to show, by default 10
    first_rows : bool, optional
        Whether to show first n rows or last n rows, by default True. If this is set to False, show last n rows.
    first_cols : bool, optional
        Whether to show first n columns or last n columns, by default True. If this is set to False, show last n rows.
    delay_time : int, optional
        How fast is the animation, by default 5. Increase this to have slower animation.
    clear_console: bool, optional
        Clear the console before printing the table, by default True. If this is set to false the previous console input/output is maintained
    """
    if isinstance(df, pd.DataFrame):
        DataFramePrettify(
            df, row_limit, col_limit, first_rows, first_cols, delay_time,clear_console
        ).prettify()

    else:
        # In case users accidentally pass a non-datafame input, use rich's print instead
        print(df)
