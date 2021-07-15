"""
plotting.py

Methods for plotting data.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass
from typing import Any, Union

import matplotlib.style as mplstyle

mplstyle.use('fast')
import itertools as it
import os
import time
from copy import deepcopy
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import xarray as xr
from IPython.display import DisplayHandle, display

import utils.file_io as io
from config import NP_FLOAT, TF_FLOAT
from dynamics.config import NetWeights
from utils import SKEYS
from utils.attr_dict import AttrDict
from utils.autocorr import calc_tau_int_vs_draws
from utils.file_io import timeit
from utils.plotting_utils import get_title_str_from_params, savefig
from utils.logger import Logger, in_notebook

#  TF_FLOAT = FLOATS[tf.keras.backend.floatx()]
#  NP_FLOAT = NP_FLOATS[tf.keras.backend.floatx()]

@dataclass
class PlotObject:
    ax: plt.Axes
    line: list[plt.Line2D]


@dataclass
class LivePlotData:
    data: Any
    plot_obj: PlotObject


def moving_average(x: np.ndarray, window: int = 10):
    #  if len(x) < window:
    if len(x.shape) > 0 and x.shape[0] < window:
        return np.mean(x, keepdims=True)
    #  if x.shape[0] < window:
    #      return np.mean(x, keepdims=True)

    return np.convolve(x, np.ones(window), 'valid') / window


Metric = Union[list, np.ndarray, tf.Tensor]


#def init_plots(configs: dict, dpi: int = 400,  figsize: tuple = None):
def init_plots(configs: dict = None, ylabels: list[str] = None, **kwargs):
    loss = None
    dq_int = None
    dq_sin = None
    if in_notebook():
        dq_int = init_live_plot(configs=configs, ylabel='dq_int',
                                xlabel='Training Step', **kwargs)
        dq_sin = init_live_plot(configs=configs, ylabel='dq_sin',
                                xlabel='Training Step', **kwargs)

        c0 = ['C0', 'C1']
        if ylabels is None:
            ylabels = ['loss', 'beta']
        loss = init_live_joint_plots(ylabels=ylabels, xlabel='Train Step',
                                     colors=c0, use_title=False, **kwargs)
    return {
        'dq_int': dq_int,
        'dq_sin': dq_sin,
        'loss': loss,
    }


def update_plot(
        y: Metric,
        fig: plt.Figure,
        ax: plt.Axes,
        line: list[plt.Line2D],
        display_id: DisplayHandle,
        window: int = 15,
        plot_freq: int = 1,
):
    if not in_notebook():
        return

    y = np.array(y)
    if len(y.shape) == 2:
        y = y.mean(-1)

    yavg = moving_average(y.squeeze(), window=window)
    line[0].set_ydata(yavg)
    line[0].set_xdata(np.arange(yavg.shape[0]))
    #  line[0].set_ydata(y)
    #  line[0].set_xdata(plot_freq * np.arange(y.shape[0]))
    #  line[0].set_xdata(np.arange(len(yavg)))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    display_id.update(fig)


def update_joint_plots(
        plot_data1: LivePlotData,
        plot_data2: LivePlotData,
        display_id: DisplayHandle,
        window: int = 15,
        plot_freq: int = 1,
        fig: plt.Figure = None,
):
    if not in_notebook():
        return

    if fig is None:
        fig = plt.gcf()

    x1 = plot_data1.data
    x2 = plot_data2.data
    plot_obj1 = plot_data1.plot_obj
    plot_obj2 = plot_data2.plot_obj

    x1 = np.array(x1).squeeze()
    x2 = np.array(x2).squeeze()
    if len(x1.shape) == 2:
        x1 = x1.mean(-1)
    if len(x2.shape) == 2:
        x2 = x2.mean(-1)

    y1 = moving_average(x1, window=window)
    y2 = moving_average(x2, window=window)

    plot_obj1.line[0].set_ydata(np.array(y1))
    plot_obj1.line[0].set_xdata(plot_freq * np.arange(y1.shape[0]))

    plot_obj2.line[0].set_ydata(y2)
    plot_obj2.line[0].set_xdata(plot_freq * np.arange(y2.shape[0]))

    plot_obj1.ax.relim()
    plot_obj2.ax.relim()

    plot_obj1.ax.autoscale_view()
    plot_obj2.ax.autoscale_view()

    fig.canvas.draw()
    display_id.update(fig)  # need to force colab to update plot


def init_live_plot(
    dpi: int = 400,
    figsize: tuple[int] = None,
    xlabel: str = None,
    ylabel: str = None,
    configs: dict = None,
    **kwargs
):
    color = kwargs.pop('color', '#0096FF')
    xlabel = 'Step' if xlabel is None else xlabel
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize, constrained_layout=True)
    line = ax.plot([0], [0], c=color, **kwargs)

    title = None
    if configs is not None:
        title = get_title_str_from_params(configs)

    if figsize is None:
        dpi = 125
        figsize = (9, 3)

    if title is not None and len(title) > 0:
        if isinstance(title, list):
            _ = fig.suptitle('\n'.join(title))
        else:
            _ = fig.suptitle(title)

    if ylabel is not None:
        _ = ax.set_ylabel(ylabel, color=color)

    ax.tick_params(axis='y', labelcolor=color)

    _ = ax.autoscale(True, axis='y')
    #  plt.Axes.autoscale(True, axis='y')
    display_id = display(fig, display_id=True)
    return {
        'fig': fig, 'ax': ax, 'line': line, 'display_id': display_id,
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
        ylabels: list[str],
        dpi: int = 120,
        figsize: tuple = None,
        configs: dict = None,
        xlabel: str = None,
        colors: list[str] = None,
        use_title: bool = False,
        set_xlim: bool = False,
):
    #  assert configs is not None if (use_title or set_xlim)
    if use_title or set_xlim:
        assert configs is not None

    if colors is None:
        n = np.random.randint(10, size=2)
        colors = [f'C{n[0]}', f'C{n[1]}']

    if figsize is None:
        dpi = 125
        figsize = (9, 3)
    #  if colors is None:
    #      colors = ['#0096ff', '#f92672']

    #  sns.set_style('ticks')
    fig, ax0 = plt.subplots(1, 1, dpi=dpi, figsize=figsize,
                            constrained_layout=True)

    ax1 = ax0.twinx()

    line0 = ax0.plot([0], [0], alpha=0.9, c=colors[0])
    line1 = ax1.plot([0], [0], alpha=0.9, c=colors[1])  # dummy

    ax0.set_ylabel(ylabels[0], color=colors[0])
    ax1.set_ylabel(ylabels[1], color=colors[1])

    ax0.tick_params(axis='y', labelcolor=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[1])

    ax0.grid(False)
    ax1.grid(False)

    ax0.set_xlabel('Step' if xlabel is None else xlabel)

    if set_xlim:
        assert configs is not None
        train_steps = configs.get('train_steps', None)
        if train_steps is not None:
            plt.xlim(0, train_steps)

    if use_title:
        assert configs is not None
        title = get_title_str_from_params(configs)
        if isinstance(title, list):
            title = '\n'.join(title)

        _ = fig.suptitle(title, font_size='small')


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
    plot_obj1 = PlotObject(ax0, line0)
    plot_obj2 = PlotObject(ax1, line1)
    return {
        'fig': fig,
        'ax0': ax0,
        'ax1': ax1,
        'plot_obj1': plot_obj1,
        'plot_obj2': plot_obj2,
        'display_id': display_id
    }
