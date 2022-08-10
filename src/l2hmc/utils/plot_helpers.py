"""
plot_helpers.py

Contains helpers for plotting.
"""
from __future__ import absolute_import, annotations, division, print_function
import datetime
import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import matplotx
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

# from l2hmc.experiment.pytorch.experiment import Experiment as ptExperiment
# from l2hmc.experiment.tensorflow.experiment import Experiment as tfExperiment

warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)

xplt = xr.plot  # type: ignore

LW = plt.rcParams.get('axes.linewidth', 1.75)

COLORS = {
    'blue':     '#007DFF',
    'red':      '#FF5252',
    'green':    '#63FF5B',
    'yellow':   '#FFFF00',
    'orange':   '#FD971F',
    'purple':   '#AE81FF',
    'pink':     '#F92672',
    'teal':     '#00CC99',
    'white':    '#CFCFCF',
}

plt.style.use('default')
plt.rcParams.update({
    'image.cmap': 'viridis',
    'savefig.transparent': True,
    'text.color': '#666666',
    'xtick.color': '#66666604',
    'ytick.color': '#66666604',
    'ytick.labelcolor': '#666666',
    'xtick.labelcolor': '#666666',
    'axes.edgecolor': '#66666600',
    'axes.labelcolor': '#666666',
    # 'axes.labelcolor': (189, 189, 189, 1.0),
    'grid.color': (0.434, 0.434, 0.434, 0.2),  # #66666602
    'axes.facecolor': (1.0, 1.0, 1.0, 0.0),
    'figure.facecolor': (1.0, 1.0, 1.0, 0.0),
})

plt.rcParams['axes.labelcolor'] = '#bdbdbd'
# sns.set_palette(list(colors.values()))
# sns.set_context('notebook', font_scale=0.8)
# plt.rcParams.update({
#     'image.cmap': 'viridis',
#     'figure.facecolor': (1.0, 1.0, 1.0, 0.),
#     'axes.facecolor': (1.0, 1.0, 1.0, 0.),
#     'axes.grid': False,
#     # 'grid.color': '#cfcfcf',
#     'figure.dpi': plt.rcParamsDefault['figure.dpi'],
#     'figure.figsize': plt.rcParamsDefault['figure.figsize'],
# })


def set_plot_style():
    plt.style.use('default')
    plt.rcParams.update({
        'image.cmap': 'viridis',
        'savefig.transparent': True,
        'text.color': '#666666',
        'xtick.color': '#66666604',
        'ytick.color': '#66666604',
        'ytick.labelcolor': '#666666',
        'xtick.labelcolor': '#666666',
        'axes.edgecolor': '#66666600',
        'axes.labelcolor': '#666666',
        # 'axes.labelcolor': (189, 189, 189, 1.0),
        'grid.color': (0.434, 0.434, 0.434, 0.2),  # #66666602
        'axes.facecolor': (1.0, 1.0, 1.0, 0.0),
        'figure.facecolor': (1.0, 1.0, 1.0, 0.0),
    })

    plt.rcParams['axes.labelcolor'] = '#bdbdbd'


def get_timestamp(fstr=None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


FigAxes = Tuple[plt.Figure, plt.Axes]


def save_figure(fig: plt.Figure, fname: str, outdir: os.PathLike):
    pngdir = Path(outdir).joinpath('pngs')
    pngdir.mkdir(exist_ok=True, parents=True)
    pngfile = pngdir.joinpath(f'{fname}.png')
    svgfile = Path(outdir).joinpath(f'{fname}.svg')
    fig.savefig(pngfile, dpi=400, bbox_inches='tight')
    fig.savefig(svgfile, dpi=400, bbox_inches='tight')


def savefig(fig: plt.Figure, outfile: os.PathLike):
    fout = Path(outfile)
    parent = fout.parent
    parent.mkdir(exist_ok=True, parents=True)
    log.info(f'Saving figure to: {fout.as_posix()}')
    fig.savefig(fout.as_posix(), dpi=400, bbox_inches='tight')


def measure_improvement(
        experiment: Any,
        title: Optional[str] = None,
) -> None:
    ehist = experiment.trainer.histories.get('eval', None)
    hhist = experiment.trainer.histories.get('hmc', None)
    if ehist is not None and hhist is not None:
        edset = ehist.get_dataset()
        hdset = hhist.get_dataset()
        dQint_eval = edset.dQint.mean('chain')[1:]
        dQint_hmc = hdset.dQint.mean('chain')[1:]
        fig, ax = plt.subplots()
        _ = ax.plot(
            dQint_eval,
            label='Trained',
            lw=2.,
            color=COLORS['blue'],
        )
        _ = ax.plot(
            dQint_hmc,
            label='HMC',
            ls=':',
            lw=1.5,
            color=COLORS['blue'],
        )
        _ = ax.grid(True, alpha=0.2)
        xticks = ax.get_xticks()
        # xticklabels = ax.get_xticklabels()
        _ = ax.set_xticklabels([
            f'{experiment.config.steps.log * int(i)}' for i in xticks
        ])
        _ = ax.set_xlabel('MD Step')
        _ = ax.set_ylabel('dQint')
        _ = ax.legend(
            loc='best',
            framealpha=0.1,
            ncol=2,
            labelcolor='#FFF',
            shadow=True
        )
        if title is not None:
            _ = ax.set_title(title)

        outdir = experiment._outdir
        improvement = np.mean(dQint_eval.values / dQint_hmc.values)
        txtfile = Path(outdir).joinpath('model_improvement.txt').as_posix()
        log.warning(f'Writing model improvement to: {txtfile}')
        with open(txtfile, 'w') as f:
            f.write(f'{improvement:.8f}')

        save_figure(fig, fname='model_improvement', outdir=outdir)


def plot_scalar(
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        fig_axes: Optional[FigAxes] = None,
        outfile: Optional[os.PathLike] = None,
        **kwargs,
) -> FigAxes:
    assert len(y.shape) == 1
    if x is None:
        x = np.arange(len(y))

    if fig_axes is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_axes

    _ = ax.plot(x, y, label=label, **kwargs)
    if xlabel is not None:
        _ = ax.set_xlabel(xlabel)
    if ylabel is not None:
        _ = ax.set_ylabel(ylabel)
    if label is not None:
        _ = matplotx.line_labels()

    if outfile is not None:
        savefig(fig, outfile)

    return fig, ax


def plot_chains(
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        num_chains: Optional[int] = 8,
        fig_axes: Optional[FigAxes] = None,
        label: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        outfile: Optional[os.PathLike] = None,
        **kwargs,
) -> FigAxes:
    assert len(y.shape) == 2
    # y.shape = [ndraws, nchains]
    num_chains = 8 if num_chains is None else num_chains
    nchains = min(num_chains, y.shape[1])
    if x is None:
        x = np.arange(y.shape[0])

    if fig_axes is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_axes

    label = f'{label}, avg: {y.mean():4.3g}'
    _ = kwargs.pop('color', None)
    color = f'C{np.random.randint(8)}'
    _ = ax.plot(x, y.mean(-1), label=label, color=color, lw=2.0, **kwargs)

    for idx in range(nchains):
        _ = ax.plot(x, y[:, idx], lw=1.0, color=color, alpha=0.7, **kwargs)

    if xlabel is not None:
        _ = ax.set_xlabel(xlabel)

    if ylabel is not None:
        _ = ax.set_ylabel(ylabel)

    if label is not None:
        _ = matplotx.line_labels()

    if outfile is not None:
        savefig(fig, outfile)

    return fig, ax


def plot_leapfrogs(
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        fig_axes: Optional[FigAxes] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        outfile: Optional[os.PathLike] = None,
        line_labels: Optional[bool] = False,
) -> FigAxes:
    assert len(y.shape) == 3

    if fig_axes is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_axes

    if x is None:
        x = np.arange(y.shape[0])

    # y.shape = [ndraws, nleapfrog, nchains]
    nlf = y.shape[1]
    yavg = y.mean(-1)
    cmap = plt.get_cmap('viridis')
    colors = {n: cmap(n / nlf) for n in range(nlf)}
    for lf in range(nlf):
        _ = ax.plot(x, yavg[:, lf], color=colors[lf], label=f'{lf}')

    if line_labels:
        _ = matplotx.line_labels(font_kwargs={'size': 'small'})

    if xlabel is not None:
        _ = ax.set_xlabel(xlabel)
    if ylabel is not None:
        _ = ax.set_ylabel(ylabel)

    if outfile is not None:
        savefig(fig, outfile)

    return fig, ax


def plot_combined(
        val: xr.DataArray,
        key: Optional[str] = None,
        num_chains: Optional[int] = 10,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
) -> tuple:
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
    figsize = subplots_kwargs.get('figsize', set_size())
    subplots_kwargs.update({'figsize': figsize})
    subfigs = None
    num_chains = 10 if num_chains is None else num_chains

    _ = subplots_kwargs.pop('constrained_layout', True)
    figsize = (3 * figsize[0], 1.5 * figsize[1])
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(1, 2)
    gs_kw = {'width_ratios': [1.33, 0.33]}
    vmin = np.min(val)
    vmax = np.max(val)
    if vmin < 0 < vmax:
        color = '#FF5252' if val.mean() > 0 else '#2979FF'
    elif 0 < vmin < vmax:
        color = '#3FB5AD'
    else:
        color = plot_kwargs.get('color', f'C{np.random.randint(5)}')

    (ax1, ax2) = subfigs[1].subplots(1, 2, sharey=True, gridspec_kw=gs_kw)
    ax1.grid(alpha=0.4)
    ax2.grid(False)
    sns.kdeplot(y=val.values.flatten(), ax=ax2, color=color, shade=True)
    axes = (ax1, ax2)
    ax0 = subfigs[0].subplots(1, 1)
    if 'chain' in val.dims:
        val = val.dropna('chain')
        _ = xplt.imshow(val, 'draw', 'chain', ax=ax0,
                        robust=True, add_colorbar=True)
    # _ = xplt.pcolormesh(val, 'draw', 'chain', ax=ax0,
    #                     robust=True, add_colorbar=True)
    # sns.despine(ax0)
    nchains = min(num_chains, len(val.coords['chain']))
    label = f'{key}_avg'
    # label = r'$\langle$' + f'{key} ' + r'$\rangle$'
    steps = np.arange(len(val.coords['draw']))
    chain_axis = val.get_axis_num('chain')
    if chain_axis == 0:
        for idx in range(nchains):
            ax1.plot(steps, val.values[idx, :],
                     color=color, lw=LW/2., alpha=0.6)

    ax1.plot(steps, val.mean('chain'), color=color, label=label, lw=1.5*LW)
    if key is not None and 'eps' in key:
        _ = ax0.set_ylabel('leapfrog')

    ax2.set_xticks([])
    ax2.set_xticklabels([])
    # sns.despine(ax=ax0, top=True, right=True, left=True, bottom=True)
    sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
    sns.despine(ax=ax2, top=True, right=True, left=True, bottom=True)
    ax2.set_xlabel('')
    ax1.set_xlabel('draw')
    sns.despine(subfigs[0])
    plt.autoscale(enable=True, axis=ax0)

    return (fig, axes)


def plot_dataArray(
        val: xr.DataArray,
        key: Optional[str] = None,
        therm_frac: Optional[float] = 0.,
        num_chains: Optional[int] = 10,
        title: Optional[str] = None,
        outdir: Optional[str | Path] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
        line_labels: Optional[bool] = False,
) -> tuple:
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
    set_plot_style()
    plt.rcParams['axes.labelcolor'] = '#bdbdbd'
    figsize = subplots_kwargs.get('figsize', set_size())
    subplots_kwargs.update({'figsize': figsize})
    subfigs = None

    if key == 'dt':
        therm_frac = 0.2

    arr = val.values  # shape: [nchains, ndraws]
    steps = np.arange(len(val.coords['draw']))

    if therm_frac is not None and therm_frac > 0.0:
        drop = int(therm_frac * arr.shape[0])
        arr = arr[drop:]
        steps = steps[drop:]

    if len(arr.shape) == 2:
        fig, axes = plot_combined(val, key=key,
                                  num_chains=num_chains,
                                  plot_kwargs=plot_kwargs,
                                  subplots_kwargs=subplots_kwargs)
    else:
        if len(arr.shape) == 1:
            fig, ax = plt.subplots(**subplots_kwargs)
            try:
                ax.plot(steps, arr, **plot_kwargs)
            except ValueError:
                try:
                    ax.plot(steps, arr[~np.isnan(arr)], **plot_kwargs)
                except Exception:
                    log.error(f'Unable to plot {key}! Continuing')
            axes = ax
        elif len(arr.shape) == 3:
            fig, ax = plt.subplots(**subplots_kwargs)
            cmap = plt.get_cmap('viridis')
            y = val.mean('chain')
            for idx in range(len(val.coords['leapfrog'])):
                pkwargs = {
                    'color': cmap(idx / len(val.coords['leapfrog'])),
                    'label': f'{idx}',
                }
                ax.plot(steps, y[idx], **pkwargs)
            axes = ax
        else:
            raise ValueError('Unexpected shape encountered')

        ax = plt.gca()
        ax.set_ylabel(key)
        # matplotx.line_labels()
        if line_labels:
            matplotx.line_labels()

        # if num_chains > 0 and len(arr.shape) > 1:
        #     lw = LW / 2.
        #     #for idx in range(min(num_chains, arr.shape[1])):
        #     nchains = len(val.coords['chains'])
        #     for idx in range(min(nchains, num_chains)):
        #         # ax = subfigs[0].subplots(1, 1)
        #         # plot values of invidual chains, arr[:, idx]
        #         # where arr[:, idx].shape = [ndraws, 1]
        #         ax.plot(steps, val
        #                 alpha=0.5, lw=lw/2., **plot_kwargs)

    if title is not None:
        fig = plt.gcf()
        fig.suptitle(title)

    if outdir is not None:
        outfile = Path(outdir).joinpath(f'{key}.svg')
        if outfile.is_file():
            tstamp = get_timestamp('%Y-%m-%d-%H%M%S')
            pngdir = Path(outdir).joinpath('pngs')
            pngdir.mkdir(exist_ok=True, parents=True)
            pngfile = pngdir.joinpath(f'{key}-{tstamp}.png')
            svgfile = Path(outdir).joinpath(f'{key}-{tstamp}.svg')
            plt.savefig(pngfile, dpi=400, bbox_inches='tight')
            plt.savefig(svgfile, dpi=400, bbox_inches='tight')
            # plt.savefig(Path(outdir).joinpath(f'{key}.svg'),
            #             dpi=400, bbox_inches='tight')

    return (fig, subfigs, axes)


def plot_array(
        val: list | np.ndarray,
        key: Optional[str] = None,
        xlabel: Optional[str] = None,
        title: Optional[str] = None,
        num_chains: Optional[int] = 10,
        outdir: Optional[str | Path] = None,
        **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(constrained_layout=True)
    arr = np.array(val)
    if num_chains is None:
        num_chains = 10

    # arr.shape = [ndraws, nleapfrog, nchains]
    if len(arr.shape) == 3:
        ndraws, nlf, _ = arr.shape
        lfarr = np.arange(nlf)
        cmap = plt.get_cmap('viridis')
        colors = {lf: cmap(lf / nlf) for lf in lfarr}
        yarr = arr.transpose((1, 0, 2))  # shape: [nleapfrog, ndraws, nchains]

        for idx, ylf in enumerate(yarr):
            y = ylf.mean(-1)  # average over chains, shape = [ndraws]
            x = np.arange(len(y))
            _ = ax.plot(x, y, label=f'{idx}', color=colors[idx], **kwargs)

        x = np.arange(ndraws)
        _ = ax.plot(x, yarr.mean((0, 1)), **kwargs)
        # arr = arr.mean()

    # arr.shape = [ndraws, nchains]
    elif len(arr.shape) == 2:
        # ndraws, nchains = arr.shape
        for idx in range(min((arr.shape[1], num_chains))):
            y = arr[:, idx]
            x = np.arange(len(y))
            _ = ax.plot(x, y, lw=1., alpha=0.7, **kwargs)
        y = arr.mean(-1)
        x = np.arange(len(y))
        _ = ax.plot(x, y, label=key, **kwargs)

    elif len(arr.shape) == 1:
        y = arr
        x = np.arange(y.shape[0])
        _ = ax.plot(x, y, label=key, **kwargs)

    else:
        raise ValueError(f'Unexpected shape encountered: {arr.shape}')

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if title is not None:
        ax.set_title(title)

    _ = ax.legend(loc='best')

    if outdir is not None:
        outfile = Path(outdir).joinpath(f'{key}.svg')
        if outfile.is_file():
            tstamp = get_timestamp('%Y-%m-%d-%H%M%S')
            pngdir = Path(outdir).joinpath('pngs')
            pngdir.mkdir(exist_ok=True, parents=True)
            pngfile = pngdir.joinpath(f'{key}-{tstamp}.png')
            svgfile = Path(outdir).joinpath(f'{key}-{tstamp}.svg')
            plt.savefig(pngfile, dpi=400, bbox_inches='tight')
            plt.savefig(svgfile, dpi=400, bbox_inches='tight')

    return fig, ax


def set_size(
        width: Optional[str] = None,
        fraction: Optional[float] = None,
        subplots: Optional[tuple] = None,
) -> tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX."""
    width_pt = 345.0
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987

    fraction = 1.0 if fraction is None else fraction
    subplots = (1, 1) if subplots is None else subplots

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set asethetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def plot_metric(
        val: np.ndarray | xr.DataArray,
        key: Optional[str] = None,
        therm_frac: Optional[float] = 0.,
        num_chains: Optional[int] = 0,
        title: Optional[str] = None,
        outdir: Optional[os.PathLike] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
        ext: Optional[str] = 'png',
        line_labels: Optional[bool] = False,
) -> tuple:
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
    figsize = subplots_kwargs.get('figsize', set_size())
    subplots_kwargs.update({'figsize': figsize})
    therm_frac = 0.0 if therm_frac is None else therm_frac
    num_chains = 16 if num_chains is None else num_chains

    # tmp = val[0]
    arr = np.array(val)

    subfigs = None
    steps = np.arange(arr.shape[0])
    if therm_frac > 0:
        drop = int(therm_frac * arr.shape[0])
        arr = arr[drop:]
        steps = steps[drop:]

    # arr.shape = [draws, chains]
    if len(arr.shape) == 2:
        _ = subplots_kwargs.pop('constrained_layout', True)
        figsize = (3 * figsize[0], 1.5 * figsize[1])

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        subfigs = fig.subfigures(1, 2)

        gs_kw = {'width_ratios': [1.33, 0.33]}
        (ax, ax1) = subfigs[1].subplots(1, 2, sharey=True,
                                        gridspec_kw=gs_kw)
        ax.grid(alpha=0.4)
        ax1.grid(False)
        color = plot_kwargs.get('color', 'C0')
        label = plot_kwargs.pop('label', None)
        # label = r'$\langle$' + f' {key} ' + r'$\rangle$'
        label = f'{key}_avg'
        ax.plot(steps, arr.mean(-1), lw=1.5*LW, label=label, **plot_kwargs)
        if num_chains > 0:
            for chain in range(min((num_chains, arr.shape[1]))):
                plot_kwargs.update({'label': None})
                ax.plot(steps, arr[:, chain], lw=LW/2., **plot_kwargs)
        sns.kdeplot(y=arr.flatten(), ax=ax1, color=color, shade=True)
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        sns.despine(ax=ax, top=True, right=True)
        sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
        ax1.set_xlabel('')
        axes = (ax, ax1)
    else:
        # arr.shape = [draws]
        if len(arr.shape) == 1:
            fig, ax = plt.subplots(**subplots_kwargs)
            ax.plot(steps, arr, **plot_kwargs)
            axes = ax
        # arr.shape = [draws, nleapfrog, chains]
        elif len(arr.shape) == 3:
            fig, ax = plt.subplots(**subplots_kwargs)
            cmap = plt.get_cmap('viridis', lut=arr.shape[1])
            _ = plot_kwargs.pop('color', None)
            for idx in range(arr.shape[1]):
                label = plot_kwargs.pop('label', None)
                if label is not None:
                    label = f'{label}-{idx}'
                y = arr[:, idx]
                color = cmap(idx / y.shape[1])
                if len(y.shape) == 2:
                    # TOO: Plot chains
                    if num_chains > 0:
                        for idx in range(min((num_chains, y.shape[1]))):
                            ax.plot(steps, y[:, idx], color=color,
                                    lw=LW/4., alpha=0.7, **plot_kwargs)

                    ax.plot(steps, y.mean(-1), color=color,
                            label=label, **plot_kwargs)
                else:

                    ax.plot(steps, y, color=color,
                            label=label, **plot_kwargs)
            axes = ax
        else:
            raise ValueError('Unexpected shape encountered')

        ax.set_ylabel(key)
    if num_chains > 0 and len(arr.shape) > 1:
        lw = LW / 2.
        for idx in range(min(num_chains, arr.shape[1])):
            # plot values of invidual chains, arr[:, idx]
            # where arr[:, idx].shape = [ndraws, 1]
            ax.plot(steps, arr[:, idx], alpha=0.5, lw=lw/2., **plot_kwargs)

    if line_labels:
        matplotx.line_labels(font_kwargs={'size': 'small'})

    ax.set_xlabel('draw')
    if title is not None:
        fig.suptitle(title)

    if outdir is not None:
        outfile = Path(outdir).joinpath(f'{key}.{ext}')
        if not outfile.is_file():
            plt.savefig(Path(outdir).joinpath(f'{key}.{ext}'),
                        dpi=400, bbox_inches='tight')

    return fig, subfigs, axes


def plot_history(
    data: dict[str, np.ndarray],
    num_chains: Optional[int] = 0,
    therm_frac: Optional[float] = 0.,
    title: Optional[str] = None,
    outdir: Optional[os.PathLike] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
):
    for key, val in data.items():
        _ = plot_metric(
            val=val,
            key=str(key),
            title=title,
            outdir=outdir,
            therm_frac=therm_frac,
            num_chains=num_chains,
            plot_kwargs=plot_kwargs,
        )


def plot_dataset(
        dataset: xr.Dataset,
        num_chains: Optional[int] = 8,
        therm_frac: Optional[float] = 0.,
        title: Optional[str] = None,
        outdir: Optional[os.PathLike] = None,
        subplots_kwargs: Optional[dict[str, Any]] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
        ext: Optional[str] = 'png',
):
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs

    if outdir is None:
        import os
        tstamp = get_timestamp('%Y-%m-%d-%H%M%S')
        outdir = Path(os.getcwd()).joinpath('plots', f'plots-{tstamp}')
        outdir.mkdir(exist_ok=True, parents=True)

    for idx, (key, val) in enumerate(dataset.data_vars.items()):
        color = f'C{idx%9}'
        plot_kwargs['color'] = color

        fig, subfigs, ax = plot_metric(
            val=val.values,
            key=str(key),
            title=title,
            outdir=None,
            therm_frac=therm_frac,
            num_chains=num_chains,
            plot_kwargs=plot_kwargs,
            subplots_kwargs=subplots_kwargs,
        )
        if outdir is not None:
            outfile = Path(outdir).joinpath(f'{key}.{ext}')
            Path(outfile.parent).mkdir(exist_ok=True, parents=True)
            outfile = outfile.as_posix()

        if subfigs is not None:
            # edgecolor = plt.rcParams['axes.edgecolor']
            plt.rcParams['axes.edgecolor'] = plt.rcParams['axes.facecolor']
            ax = subfigs[0].subplots(1, 1)
            # ax = fig[1].subplots(constrained_layout=True)
            cbar_kwargs = {
                # 'location': 'top',
                # 'orientation': 'horizontal',
            }
            im = val.plot(ax=ax, cbar_kwargs=cbar_kwargs)  # type:ignore
            # ax.set_ylim(0, )
            im.colorbar.set_label(f'{key}')                # type:ignore
            sns.despine(subfigs[0], top=True, right=True,
                        left=True, bottom=True)
            if outdir is not None:
                print(f'Saving figure to: {outfile}')
                plt.savefig(outfile, dpi=400, bbox_inches='tight')
        else:
            fig.savefig(outfile, dpi=400, bbox_inches='tight')

    return dataset


def make_ridgeplots(
        dataset: xr.Dataset,
        num_chains: Optional[int] = None,
        outdir: Optional[os.PathLike] = None,
        drop_zeros: Optional[bool] = False,
        drop_nans: Optional[bool] = True,
        cmap: Optional[str] = 'rainbow',
):
    """Make ridgeplots."""
    data = {}
    # with sns.axes_style('white', rc={'axes.facecolor': (0, 0, 0, 0)}):
    # sns.set(style='white', palette='bright', context='paper')
    # with sns.set_style(style='white'):
    outdir = Path(os.getcwd()) if outdir is None else Path(outdir)
    outdir = outdir.joinpath('ridgeplots')
    with sns.plotting_context(
            context='paper',
    ):
        sns.set(
            style='white',
            palette='bright',
        )
        plt.rcParams['axes.facecolor'] = (0, 0, 0, 0.0)
        plt.rcParams['figure.facecolor'] = (0, 0, 0, 0.0)
        for key, val in dataset.data_vars.items():
            if 'leapfrog' in val.coords.dims:
                lf_data = {
                    key: [],
                    'lf': [],
                }
                for lf in val.leapfrog.values:
                    # val.shape = (chain, leapfrog, draw)
                    # x.shape = (chain, draw);  selects data for a single lf
                    x = val[{'leapfrog': lf}].values
                    # if num_chains is not None, keep `num_chains` for plotting
                    if num_chains is not None:
                        x = x[:num_chains, :]

                    x = x.flatten()
                    if drop_zeros:
                        x = x[x != 0]
                    #  x = val[{'leapfrog': lf}].values.flatten()
                    if drop_nans:
                        x = x[np.isfinite(x)]

                    lf_arr = np.array(len(x) * [f'{lf}'])
                    lf_data[key].extend(x)
                    lf_data['lf'].extend(lf_arr)

                lfdf = pd.DataFrame(lf_data)
                data[key] = lfdf

                # Initialize the FacetGrid object
                ncolors = len(val.leapfrog.values)
                pal = sns.color_palette(cmap, n_colors=ncolors)
                g = sns.FacetGrid(
                    lfdf,
                    row='lf', hue='lf',
                    aspect=15, height=0.25, palette=pal  # type:ignore
                )

                # Draw the densities in a few steps
                _ = g.map(sns.kdeplot, key, cut=1,
                          shade=True, alpha=0.7, linewidth=1.25)
                _ = g.map(plt.axhline, y=0, lw=1.5, alpha=0.9, clip_on=False)

                # Define and use a simple function to
                # label the plot in axes coords:
                def label(_, color, label):  # type:ignore #noqa
                    ax = plt.gca()
                    ax.set_ylabel('')
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                    ax.text(0, 0.10, label, fontweight='bold', color=color,
                            ha='left', va='center', transform=ax.transAxes)

                _ = g.map(label, key)
                # Set the subplots to overlap
                _ = g.fig.subplots_adjust(hspace=-0.75)
                # Remove the axes details that don't play well with overlap
                _ = g.set_titles('')
                _ = g.set(yticks=[])
                _ = g.set(yticklabels=[])
                _ = g.set(xlabel=f'{key}')
                _ = g.despine(bottom=True, left=True)
                if outdir is not None:
                    outdir = Path(outdir)
                    pngdir = outdir.joinpath('pngs')
                    fsvg = Path(outdir).joinpath(f'{key}_ridgeplot.svg')
                    fpng = Path(pngdir).joinpath(f'{key}_ridgeplot.png')

                    outdir.mkdir(exist_ok=True, parents=True)
                    pngdir.mkdir(exist_ok=True, parents=True)

                    log.warning(f'Saving figure to: {fsvg.as_posix()}')
                    plt.savefig(fsvg.as_posix(), dpi=500, bbox_inches='tight')
                    plt.savefig(fpng.as_posix(), dpi=500, bbox_inches='tight')

        # plt.close('all')

    #  sns.set(style='whitegrid', palette='bright', context='paper')
    fig = plt.gcf()
    ax = plt.gca()

    return fig, ax, data


def plot_plaqs(
        plaqs: np.ndarray,
        nchains: Optional[int] = 10,
        outdir: Optional[os.PathLike] = None,
):
    assert len(plaqs.shape) == 2
    ndraws, nchains = plaqs.shape
    xplot = np.arange(ndraws)
    fig, ax = plt.subplots(constrained_layout=True)
    plaq_avg = plaqs.mean()
    label = f'avg: {plaq_avg:.4g}'
    _ = ax.plot(xplot, plaqs.mean(-1), label=label, lw=2.0, color='C0')
    for idx in range(min(nchains, plaqs.shape[1])):
        _ = ax.plot(xplot, plaqs[:, idx], lw=1.0, alpha=0.5, color='C0')

    _ = ax.set_ylabel('dxp')
    _ = ax.set_xlabel('Train Epoch')
    _ = ax.grid(True, alpha=0.4)
    _ = matplotx.line_labels()
    if outdir is not None:
        outfile = Path(outdir).joinpath('plaqs_diffs.svg')
        log.info(f'Saving figure to: {outfile}')
        fig.savefig(outfile.as_posix(), dpi=500, bbox_inches='tight')

    return fig, ax
