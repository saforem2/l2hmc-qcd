"""
l2hmc/utils/live_plots.py

Methods for plotting data.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass
import logging
from typing import Optional, Sequence
from typing import Any, Union

from IPython.display import DisplayHandle, display
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np

from l2hmc import get_logger
from l2hmc.utils.plot_helpers import set_plot_style
from l2hmc.utils.rich import is_interactive

_ = mplstyle.use('fast')


log = logging.getLogger(__name__)
# log = get_logger(__name__)

mplog = logging.getLogger('matplotlib')
mplog.setLevel('CRITICAL')


@dataclass
class PlotObject:
    ax: plt.Axes
    line: list[plt.Line2D]


@dataclass
class LivePlotData:
    data: Any
    plot_obj: PlotObject


# xType = Union[Any, np.ndarray]
ArrayLike = Union[np.ndarray, Sequence, list]


def moving_average(x: ArrayLike, window: int = 10):
    xarr = np.array(x)
    if len(xarr.shape) > 0 and xarr.shape[0] < window:
        return np.mean(xarr, keepdims=True)

    return np.convolve(xarr, np.ones(window), 'valid') / window


def init_plots(
        title: Optional[str] = None,
        ylabels: Optional[Sequence[str]] = None,
        keys: Optional[Sequence[str]] = None,
        xlabel: str = 'Step',
        **kwargs
):
    set_plot_style()
    plots = {}
    if plt.interactive:
        if keys is not None and len(keys) > 0:
            for key in keys:
                plots[key] = init_live_plot(
                    title=title,
                    ylabel=key,
                    xlabel=xlabel,
                    **kwargs
                )

        c0 = ['C0', 'C1']
        if ylabels is None:
            ylabels = ['loss', 'dQint']
        plots['loss'] = init_live_joint_plots(
            ylabels=ylabels,
            xlabel=xlabel,
            colors=c0,
            **kwargs
        )

    return plots


def update_plot(
        y: np.ndarray | list,
        ax: plt.Axes,
        line: list[plt.Line2D],
        display_id: DisplayHandle,
        logging_steps: int = 1,
        fig: Optional[plt.Figure] = None,
) -> None:
    if not is_interactive():
        return

    if isinstance(y, list):
        yarr: np.ndarray = np.stack(y) if isinstance(y[0], np.ndarray) else np.array(y)
    else:
        yarr = y

    if len(yarr.shape) == 2:
        yarr = yarr.mean(-1)

    assert isinstance(y, np.ndarray)
    x = np.arange(yarr.shape[0])
    line[0].set_ydata(yarr)
    line[0].set_xdata(logging_steps * x)
    ax.relim()
    ax.autoscale_view()
    if fig is not None:
        fig.canvas.draw()
        display_id.update(fig)

    plt.show()
    # return fig, ax


def update_joint_plots(
        plot_data1: LivePlotData,
        plot_data2: LivePlotData,
        display_id: DisplayHandle,
        logging_steps: int = 1,
        fig: Optional[plt.Figure | plt.FigureBase] = None,
):
    if not is_interactive():
        return

    plot_obj1 = plot_data1.plot_obj
    plot_obj2 = plot_data2.plot_obj

    y1 = np.array(plot_data1.data)
    y2 = np.array(plot_data2.data)

    #  x1avg = x1.mean(-1) if len(x1.shape) == 2 else x1  # type: np.ndarray
    #  x2avg = x2.mean(-1) if len(x2.shape) == 2 else x2  # type: np.ndarray
    if len(y1.shape) == 2:
        y1 = np.mean(y1, -1)

    if len(y2.shape) == 2:
        y2 = np.mean(y2, -1)

    x1 = np.arange(y1.shape[0])
    x2 = np.arange(y1.shape[0])
    # y1 = moving_average(x1avg, window=window)
    # y2 = moving_average(x2avg, window=window)
    # yavg = moving_average(y.squeeze(), window=window)
    # _ = line[0].set_ydata(y)
    # _ = line[0].set_xdata(logging_steps * np.arange(y.shape[0]))
    # line[0].set_xdata(logging_steps * np.arange(yavg.shape[0]))

    plot_obj1.line[0].set_ydata(y1)
    plot_obj1.line[0].set_xdata(logging_steps * x1)

    plot_obj2.line[0].set_ydata(y2)
    plot_obj2.line[0].set_xdata(logging_steps * x2)

    plot_obj1.ax.relim()
    plot_obj2.ax.relim()

    plot_obj1.ax.autoscale_view()
    plot_obj2.ax.autoscale_view()

    # if fig is None:
    #     fig = plt.gcf()

    if fig is not None:
        fig.canvas.draw()  # type:ignore
        # if isinstance(fig, plt.Figure):
        # elif isinstance(fig, plt.FigureBase):
        #     fig1 = fig.get_figure()
        #     fig1.canvas.draw()

        display_id.update(fig)  # need to force colab to update plot

    # return fig


def init_live_plot(
        dpi: int = 400,
        figsize: Optional[tuple[int, int]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        # ls: Optional[str] = None,
        # marker: Optional[str] = None,
        **kwargs
):
    color = kwargs.pop('color', '#0096FF')
    xlabel = 'Step' if xlabel is None else xlabel
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        dpi=dpi,
        figsize=figsize,
        constrained_layout=True
    )
    # ax = ax.squeeze()
    # ax = axs[0]
    # fig, ax = canvas
    assert isinstance(ax, plt.Axes)
    line, = ax.plot([0], [0], c=color, animated=True, **kwargs)

    # title = None
    # if configs is not None:
    #     title = get_title_str_from_params(configs)

    if figsize is None:
        dpi = 125
        figsize = (9, 3)

    if title is not None and len(title) > 0:
        if isinstance(title, list):
            fig.suptitle('\n'.join(title))
        else:
            fig.suptitle(title)

    if ylabel is not None:
        ax.set_ylabel(ylabel, color=color)

    ax.tick_params(axis='y', labelcolor=color)

    ax.autoscale(True, axis='y')
    #  plt.Axes.autoscale(True, axis='y')
    # plt.show()
    display_id = display(fig, display_id=True)
    return {
        # 'fig': fig,
        'ax': ax,
        'line': line,
        'display_id': display_id,
    }


#
#  def init_live_joint_plots(
#          train_steps: int,
#          #  n_epoch: int,
#          dpi: int = 400,
#          figsize: tuple = (5, 2),
#          params: dict = None,
#          #  param: Param = None,
#          #  config: TrainConfig = None,
#          xlabel: str = None,
#          ylabel: list[str] = None,
#          colors: list[str] = None,
#  ):
def init_live_joint_plots(
        ylabels: Sequence[str],
        dpi: int = 120,
        figsize: Optional[tuple[int, int]] = None,
        xlabel: Optional[str] = None,
        colors: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        fig: Optional[plt.Figure | plt.FigureBase] = None,
        ax: Optional[plt.Axes] = None,
):
    #  assert configs is not None if (use_title or set_xlim)
    # if use_title or set_xlim:
    #     assert configs is not None

    if colors is None:
        n = np.random.randint(10, size=2)
        colors = [f'C{n[0]}', f'C{n[1]}']

    if figsize is None:
        dpi = 125
        figsize = (9, 3)
    #  if colors is None:
    #      colors = ['#0096ff', '#f92672']

    #  sns.set_style('ticks')
    if fig is None:
        fig, ax0 = plt.subplots(
            1,
            1,
            dpi=dpi,
            figsize=figsize,
            constrained_layout=True
        )
        ax = ax0 if isinstance(ax0, plt.Axes) else ax0[0]
    else:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # else:
    #     fig = plt.gcf()
    #     ax = plt.gca()

    assert ax is not None and isinstance(ax, plt.Axes)
    ax1 = ax.twinx()
    line0 = ax.plot([0], [0], alpha=0.9, c=colors[0], animated=True)
    line1 = ax1.plot([0], [0], alpha=0.9, c=colors[1], animated=True)  # dummy

    ax.set_ylabel(ylabels[0], color=colors[0])
    ax1.set_ylabel(ylabels[1], color=colors[1])

    ax.tick_params(axis='y', labelcolor=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[1])

    ax.grid(False)
    ax1.grid(False)

    ax.set_xlabel('Step' if xlabel is None else xlabel)

    # if set_xlim:
    #     assert configs is not None
    #     train_steps = configs.get('train_steps', None)
    #     if train_steps is not None:
    #         plt.xlim(0, train_steps)

    # if use_title:
    #     assert configs is not None
    #     title = get_title_str_from_params(configs)
    #     if isinstance(title, list):
    #         title = '\n'.join(title)

    if title is not None:
        if fig is not None:
            fig.suptitle(title, font_size='small')

    #  title = get_title(param, config)
    #  if len(title) > 0:
    #      fig.suptitle('\n'.join(title))
    #  title = ''
    #  if param is not None:
    #      title += param.uniquestr()
    #  if config is not None:
    #      title += config.uniquestr()

    #  fig.suptitle(title)

    display_id = display(fig, display_id=True)
    plot_obj1 = PlotObject(ax, line0)  # type:ignore
    plot_obj2 = PlotObject(ax1, line1)
    # plt.show()
    return {
        'fig': fig,
        'ax0': ax,
        'ax1': ax1,
        'plot_obj1': plot_obj1,
        'plot_obj2': plot_obj2,
        'display_id': display_id
    }


PlotData = LivePlotData


def update_plots(
        history: dict,
        plots: dict,
        # window: int = 1,
        logging_steps: int = 1
):
    lpdata = PlotData(history['loss'], plots['loss']['plot_obj1'])
    bpdata = PlotData(history['dQint'], plots['loss']['plot_obj2'])
    fig_loss = plots['loss']['fig']
    id_loss = plots['loss']['display_id']
    update_joint_plots(
        lpdata,
        bpdata,
        fig=fig_loss,
        display_id=id_loss,
        logging_steps=logging_steps
    )

    # for key, plot in plots.items():
    # for key in plots.keys():
    #     if key == 'loss':
    #         continue

    #     val = history.get(key, None)
    #     if val is not None:
    #         update_plot(
    #             y=val,
    #             window=window,
    #             logging=steps=logging_steps,
    #             **plots[key]
    #         )
    for key, val in history.items():
        if key in plots and key != 'loss':
            _ = update_plot(
                y=val,
                # window=window,
                logging_steps=logging_steps,
                **plots[key]
            )
