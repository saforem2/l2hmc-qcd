"""
history.py

Contains implementation of History object for tracking / aggregating metrics.
"""
from __future__ import absolute_import, annotations, division, print_function

import os
import time
from contextlib import ContextDecorator
import opinionated
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import matplotx
import numpy as np
import seaborn as sns
import tensorflow as tf
import torch
import wandb
import xarray as xr

import l2hmc.utils.plot_helpers as hplt
import logging
# from l2hmc import get_logger
from l2hmc.common import grab_tensor
from l2hmc.configs import MonteCarloStates, Steps
from l2hmc.configs import PROJECT_DIR
# from l2hmc.common import get_timestamp
from l2hmc.common import plot_dataset
from l2hmc.utils.plot_helpers import (
    make_ridgeplots,
    set_plot_style,
    set_size,
    plot_combined,
    subplots
)

from l2hmc.utils.rich import is_interactive

# TensorLike = Union[tf.Tensor, torch.Tensor, np.ndarray]
TensorLike = Union[tf.Tensor, torch.Tensor, np.ndarray, list]
ScalarLike = Union[float, int, bool, np.floating, np.integer]

PT_FLOAT = torch.get_default_dtype()
TF_FLOAT = tf.dtypes.as_dtype(tf.keras.backend.floatx())
Scalar = Union[float, int, np.floating, bool]
# Scalar = TF_FLOAT | PT_FLOAT | np.floating | int | bool

log = logging.getLogger(__name__)

# log = get_logger(__name__)

xplt = xr.plot  # type:ignore
LW = plt.rcParams.get('axes.linewidth', 1.75)


def format_pair(k: str, v: ScalarLike) -> str:
    if isinstance(v, (int, bool, np.integer)):
        # return f'{k}={v:<3}'
        return f'{k}={v}'
    # return f'{k}={v:<3.4f}'
    return f'{k}={v:<.3f}'


def summarize_dict(d: dict) -> str:
    return ' '.join([format_pair(k, v) for k, v in d.items()])


# def subsample_dict(d: dict) -> dict:
#     for key, val in d.items():
#         pass

# def timeit(func):
#     @functools.wraps(func)
#     def time_closure(*args, **kwargs):
#         start = time.perf_counter()
#         result = func(*args, **kwargs)
#         end = time.perf_counter() - start
#         log.info(f')


class StopWatch(ContextDecorator):
    def __init__(
            self,
            msg: str,
            wbtag: Optional[str] = None,
            iter: Optional[int] = None,
            commit: Optional[bool] = False,
            prefix: str = 'StopWatch/',
            log_output: bool = True,
    ) -> None:
        self.msg = msg
        self.data = {}
        self.iter = iter if iter is not None else None
        self.prefix = prefix
        self.wbtag = wbtag if wbtag is not None else None
        self.log_output = log_output
        self.commit = commit
        if wbtag is not None:
            self.data = {
                f'{self.wbtag}/dt': None,
            }
            if iter is not None:
                self.data |= {
                    f'{self.wbtag}/iter': self.iter,
                }

    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, t, v, traceback):
        dt = time.perf_counter() - self.time
        # if self.wbtag is not None and wandb.run is not None:
        if len(self.data) > 0 and wandb.run is not None:
            self.data |= {f'{self.wbtag}/dt': dt}
            wandb.run.log({self.prefix: self.data}, commit=self.commit)
        if self.log_output:
            log.info(
                f"{self.msg} took "
                f"{dt:.3f} seconds"
            )


@dataclass
class StateHistory:
    def __post_init__(self):
        self.init = {'x': [], 'v': []}
        self.proposed = {'x': [], 'v': []}
        self.out = {'x': [], 'v': []}

    def update(self, mc_states: MonteCarloStates) -> None:
        self.init['x'].append(mc_states.init.x.numpy())
        self.init['v'].append(mc_states.init.v.numpy())
        self.proposed['x'].append(mc_states.proposed.x.numpy())
        self.proposed['v'].append(mc_states.proposed.v.numpy())
        self.out['x'].append(mc_states.out.x.numpy())
        self.out['v'].append(mc_states.out.v.numpy())


class History:
    def __init__(self, keys: Optional[list[str]] = None) -> None:
        self.keys = [] if keys is None else keys
        self.history = {}

    def update(self, metrics: dict):
        for key, val in metrics.items():
            try:
                self.history[key].append(val)
            except KeyError:
                self.history[key] = [val]


class BaseHistory:
    def __init__(self, steps: Optional[Steps] = None):
        self.steps = steps
        self.history = {}
        self.era_metrics = {}
        if steps is not None:
            self.era_metrics = {
                str(era): {} for era in range(steps.nera)
            }
        # nera = 1 if steps is None else steps.nera
        # self.era_metrics = {str(era): {} for era in range(nera)}

    def era_summary(self, era) -> str:
        emetrics = self.era_metrics.get(str(era), None)
        if emetrics is None:
            return ''
        return ', '.join([
            f'{k}={np.mean(v):<5.4f}' for k, v in emetrics.items()
            if k not in ['era', 'epoch']
            and v is not None
        ])

    def metric_to_numpy(
            self,
            metric: Any,
    ) -> np.ndarray | Scalar:
        if isinstance(metric, (Scalar, np.ndarray)):
            return metric

        if isinstance(metric, (tf.Tensor, torch.Tensor)):
            return grab_tensor(metric)

        if isinstance(metric, list):
            if isinstance(metric[0], np.ndarray):
                return np.stack(metric)
            if isinstance(metric[0], tf.Tensor):
                return grab_tensor(tf.stack(metric))
            if isinstance(metric[0], torch.Tensor):
                return grab_tensor(torch.stack(metric))

        return np.array(metric)

    def _update(
            self,
            key: str,
            val: Any
    ) -> float | int | bool | np.floating | np.integer:
        if isinstance(val, (list, tuple)):
            if isinstance(val[0], torch.Tensor):
                val = grab_tensor(torch.stack(val))
            elif isinstance(val, np.ndarray):
                val = np.stack(val)
            else:
                val = val
        if isinstance(val, (tf.Tensor, torch.Tensor)):
            val = grab_tensor(val)
        try:
            self.history[key].append(val)
        except KeyError:
            self.history[key] = [val]

        # ScalarLike = Union[float, int, bool, np.floating]
        if isinstance(val, (float, int, bool, np.floating, np.integer)):
            return val
        # if (
        #         isinstance(
        #             val,
        #             (float, int, bool, np.floating, np.integer)
        #         )
        #         # or key in ['era', 'epoch']
        #         # or 'step' in key
        # ):
        #     # assert val is not None and isinstance(val, ScalarLike)
        #     return val
        avg = np.mean(val).real
        assert isinstance(avg, np.floating)
        return avg

    def update(
            self,
            metrics: dict
    ) -> dict[str, Any]:
        avgs = {}
        era = metrics.get('era', None)
        avg = 0.0
        for key, val in metrics.items():
            if val is None:
                continue
            if isinstance(val, dict):
                for k, v in val.items():
                    kk = f'{key}/{k}'
                    avg = self._update(kk, v)
                    avgs[kk] = avg
            else:
                avg = self._update(key, val)
                avgs[key] = avg

            if era is not None:
                if str(era) not in self.era_metrics.keys():
                    self.era_metrics[str(era)] = {}

                try:
                    self.era_metrics[str(era)][key].append(avg)
                except KeyError:
                    self.era_metrics[str(era)][key] = [avg]

        return avgs

    def plot_dataArray1(
            self,
            val: xr.DataArray,
            key: Optional[str] = None,
            therm_frac: float = 0.,
            num_chains: int = 16,
            title: Optional[str] = None,
            outdir: Optional[os.PathLike] = None,
            subplots_kwargs: Optional[dict[str, Any]] = None,
            plot_kwargs: Optional[dict[str, Any]] = None,

    ) -> tuple:
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        figsize = subplots_kwargs.get('figsize', hplt.set_size())
        subplots_kwargs.update({'figsize': figsize})
        subfigs = None

        # tmp = val[0]
        # arr = val.values  # shape: [nchains, ndraws]
        # steps = np.arange(arr.shape[0])
        steps = val.draw.values
        nchains, ndraws = val.shape
        if therm_frac is not None and therm_frac > 0:
            drop = int(therm_frac * val.shape[0])
            arr = val[drop:]
            steps = steps[drop:]

        if len(val.shape) == 2:
            _ = subplots_kwargs.pop('constrained_layout', True)
            figsize = (3 * figsize[0], 1.5 * figsize[1])

            fig = plt.figure(figsize=figsize, constrained_layout=True)
            subfigs = fig.subfigures(1, 2)

            gs_kw = {'width_ratios': [1.33, 0.33]}
            (ax, ax1) = subfigs[1].subplots(1, 2, sharey=True,
                                            gridspec_kw=gs_kw)
            ax.grid(alpha=0.2)
            ax1.grid(False)
            color = plot_kwargs.get('color', None)
            label = r'$\langle$' + f' {key} ' + r'$\rangle$'
            ax.plot(steps, val.mean('chain'), lw=1.5 *
                    LW, label=label, **plot_kwargs)
            sns.kdeplot(y=val.values.flatten(), ax=ax1,
                        color=color, shade=True)
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            sns.despine(ax=ax, top=True, right=True)
            sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
            ax1.set_xlabel('')
            # _ = subfigs[1].subplots_adjust(wspace=-0.75)
            axes = (ax, ax1)

            ax0 = subfigs[0].subplots(1, 1)
            im = val.plot(ax=ax0)           # type:ignore
            im.colorbar.set_label(key)      # type:ignore
            sns.despine(subfigs[0])
            ax0.plot(steps, val.mean('chain'), lw=2., color=color)
            for idx in range(min(num_chains, len(val.chain))):
                ax0.plot(steps, val.values[idx, :],
                         lw=1., alpha=0.7, color=color)

        elif len(val.shape) == 1:
            fig, ax = plt.subplots(**subplots_kwargs)
            assert isinstance(ax, plt.Axes)
            ax.plot(steps, val.values, **plot_kwargs)
            axes = ax
        elif len(val.shape) == 3:
            fig, ax = plt.subplots(**subplots_kwargs)
            assert isinstance(ax, plt.Axes)
            cmap = plt.get_cmap('viridis')
            # nlf = arr.shape[1]
            nlf = val.leapfrog
            for idx in range(nlf):
                y = val[:, idx, :].mean('chain')
                pkwargs = {
                    'color': cmap(idx / nlf),
                    'label': f'{idx}',
                }
                ax.plot(steps, y, **pkwargs)
            axes = ax
        else:
            raise ValueError('Unexpected shape encountered')

        matplotx.line_labels()
        ax.set_xlabel('draw')
        if title is not None:
            fig.suptitle(title)

        if outdir is not None:
            plt.savefig(Path(outdir).joinpath(f'{key}.svg'),
                        dpi=400, bbox_inches='tight')

        return (fig, subfigs, axes)

    def plot(
            self,
            val: np.ndarray,
            key: Optional[str] = None,
            therm_frac: Optional[float] = 0.,
            num_chains: Optional[int] = None,
            title: Optional[str] = None,
            outdir: Optional[os.PathLike] = None,
            subplots_kwargs: Optional[dict[str, Any]] = None,
            plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Dependent on `val.shape`. If:

        1. `len(val.shape) == 1`: `[steps]`
        2. `len(val.shape) == 2`:  `[steps, chains]`
        3. `len(val.shape) == 3`:  `[steps, nleapfrog, chains]`
        """
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        figsize = subplots_kwargs.get('figsize', hplt.set_size())
        subplots_kwargs.update({'figsize': figsize})
        num_chains = 32 if num_chains is None else num_chains

        # tmp = val[0]
        arr = np.array(val)

        subfigs = None
        # Assume len(arr.shape) >= 1, with first dimension `steps`
        steps = np.arange(arr.shape[0])
        if therm_frac is not None and therm_frac > 0:
            # Drop for thermalization
            drop = int(therm_frac * arr.shape[0])
            arr = arr[drop:]
            steps = steps[drop:]

        # `arr.shape = [steps, chains]`
        if len(arr.shape) == 2:
            _ = subplots_kwargs.pop('constrained_layout', True)
            figsize = (3 * figsize[0], 2 * figsize[1])
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            subfigs = fig.subfigures(1, 2)
            gs_kw = {'width_ratios': [1.33, 0.33]}
            (ax, ax1) = subfigs[1].subplots(
                1,
                2,
                sharey=True,
                gridspec_kw=gs_kw
            )
            ax.grid(alpha=0.2)
            ax1.grid(False)
            color = plot_kwargs.get('color', None)
            label = r'$\langle$' + f' {key} ' + r'$\rangle$'
            ax.plot(steps, arr.mean(-1), lw=1.25*LW, label=label, **plot_kwargs)
            sns.kdeplot(y=arr.flatten(), ax=ax1, color=color, shade=True)
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            # ax1.set_yticks([])
            # ax1.set_yticklabels([])
            sns.despine(ax=ax, top=True, right=True)
            sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
            ax1.set_xlabel('')
            if key is not None:
                plt.legend(loc='best', frameon=False)
            # ax1.set_ylabel('')
            # ax.set_yticks(ax.get_yticks())
            # ax.set_yticklabels(ax.get_yticklabels())
            # ax.set_ylabel(key)
            # _ = subfigs[1].subplots_adjust(wspace=-0.75)
            axes = (ax, ax1)
        else:
            if len(arr.shape) == 1:
                fig, ax = plt.subplots(**subplots_kwargs)
                assert isinstance(ax, plt.Axes)
                ax.plot(steps, arr, **plot_kwargs)
                axes = ax
            elif len(arr.shape) == 3:
                fig, ax = plt.subplots(**subplots_kwargs)
                assert isinstance(ax, plt.Axes)
                cmap = plt.get_cmap('viridis')
                nlf = arr.shape[1]
                for idx in range(nlf):
                    # y = arr[:, idx, :].mean(-1)
                    # pkwargs = {
                    #     'color': cmap(idx / nlf),
                    #     'label': f'{idx}',
                    # }
                    # ax.plot(steps, y, **pkwargs)
                    label = plot_kwargs.pop('label', None)
                    if label is not None:
                        label = f'{label}-{idx}'
                    y = arr[:, idx, :num_chains]
                    color = cmap(idx / y.shape[1])
                    plot_kwargs['color'] = cmap(idx / y.shape[1])
                    if len(y.shape) == 2:
                        # TOO: Plot chains
                        if num_chains > 0:
                            for idx in range(min((num_chains, y.shape[1]))):
                                _ = ax.plot(
                                    steps,
                                    y[:, idx],
                                    # color,
                                    lw=LW/2.,
                                    alpha=0.8,
                                    **plot_kwargs
                                )

                        _ = ax.plot(
                            steps,
                            y.mean(-1),
                            # color=color,
                            label=label,
                            **plot_kwargs
                        )
                    else:

                        _ = ax.plot(
                            steps,
                            y,
                            # color=color,
                            label=label,
                            **plot_kwargs
                        )
                axes = ax
            else:
                raise ValueError('Unexpected shape encountered')

            ax.set_ylabel(key)

        if num_chains > 0 and len(arr.shape) > 1:
            # lw = LW / 2.
            for idx in range(min(num_chains, arr.shape[1])):
                # ax = subfigs[0].subplots(1, 1)
                # plot values of invidual chains, arr[:, idx]
                # where arr[:, idx].shape = [ndraws, 1]
                ax.plot(steps, arr[:, idx], alpha=0.5, lw=LW/2., **plot_kwargs)

        matplotx.line_labels()
        ax.set_xlabel('draw')
        if title is not None:
            fig.suptitle(title)

        if outdir is not None:
            # plt.savefig(Path(outdir).joinpath(f'{key}.svg'),
            #             dpi=400, bbox_inches='tight')
            outfile = Path(outdir).joinpath(f'{key}.svg')
            if outfile.is_file():
                tstamp = hplt.get_timestamp('%Y-%m-%d-%H%M%S')
                pngdir = Path(outdir).joinpath('pngs')
                pngdir.mkdir(exist_ok=True, parents=True)
                pngfile = pngdir.joinpath(f'{key}-{tstamp}.png')
                svgfile = Path(outdir).joinpath(f'{key}-{tstamp}.svg')
                plt.savefig(pngfile, dpi=400, bbox_inches='tight')
                plt.savefig(svgfile, dpi=400, bbox_inches='tight')

        return fig, subfigs, axes

    def plot_dataArray(
            self,
            val: xr.DataArray,
            key: Optional[str] = None,
            therm_frac: Optional[float] = 0.,
            num_chains: Optional[int] = 0,
            title: Optional[str] = None,
            outdir: Optional[str] = None,
            subplots_kwargs: Optional[dict[str, Any]] = None,
            plot_kwargs: Optional[dict[str, Any]] = None,
            line_labels: bool = False,
            logfreq: Optional[int] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        set_plot_style()
        # plt.rcParams['axes.labelcolor'] = '#bdbdbd'
        figsize = subplots_kwargs.get('figsize', set_size())
        subplots_kwargs.update({'figsize': figsize})
        subfigs = None
        # if key == 'dt':
        #     therm_frac = 0.2
        arr = val.values  # shape: [nchains, ndraws]
        # steps = np.arange(len(val.coords['draw']))
        steps = val.coords['draw']
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
                fig, ax = subplots(**subplots_kwargs)
                try:
                    ax.plot(steps, arr, **plot_kwargs)
                except ValueError:
                    try:
                        ax.plot(steps, arr[~np.isnan(arr)], **plot_kwargs)
                    except Exception:
                        log.error(f'Unable to plot {key}! Continuing')
                _ = ax.grid(True, alpha=0.2)
                axes = ax
            elif len(arr.shape) == 3:
                fig, ax = subplots(**subplots_kwargs)
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
            assert isinstance(ax, plt.Axes)
            _ = ax.set_ylabel(key)
            _ = ax.set_xlabel('draw')
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
            _ = fig.suptitle(title)
        if logfreq is not None:
            ax = plt.gca()
            xticks = ax.get_xticks()
            _ = ax.set_xticklabels([
                f'{logfreq * int(i)}' for i in xticks
            ])
        if outdir is not None:
            dirs = {
                'png': Path(outdir).joinpath("pngs/"),
                'svg': Path(outdir).joinpath("svgs/"),
            }
            _ = [i.mkdir(exist_ok=True, parents=True) for i in dirs.values()]
            from l2hmc.configs import PROJECT_DIR
            log.info(
                f"Saving {key} plot to: "
                f"{Path(outdir).resolve().relative_to(PROJECT_DIR)}"
            )
            for ext, d in dirs.items():
                outfile = d.joinpath(f"{key}.{ext}")
                plt.savefig(outfile, dpi=400, bbox_inches='tight')
        return (fig, subfigs, axes)

    def plot_dataset(
        self,
            therm_frac: float = 0.,
            title: Optional[str] = None,
            outdir: Optional[os.PathLike] = None,
            subplots_kwargs: Optional[dict[str, Any]] = None,
            plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        dataset = self.get_dataset()
        return plot_dataset(
            dataset=dataset,
            therm_frac=therm_frac,
            title=title,
            outdir=outdir,
            subplots_kwargs=subplots_kwargs,
            plot_kwargs=plot_kwargs
        )

    def plot_2d_xarr(
            self,
            xarr: xr.DataArray,
            label: Optional[str] = None,
            num_chains: Optional[int] = None,
            title: Optional[str] = None,
            outdir: Optional[os.PathLike] = None,
            subplots_kwargs: Optional[dict[str, Any]] = None,
            plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        assert len(xarr.shape) == 2
        assert 'draw' in xarr.coords and 'chain' in xarr.coords
        num_chains = len(xarr.chain) if num_chains is None else num_chains
        # _ = subplots_kwargs.pop('constrained_layout', True)
        figsize = plt.rcParams.get('figure.figsize', (8, 6))
        figsize = (3 * figsize[0], 1.5 * figsize[1])
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        subfigs = fig.subfigures(1, 2)
        gs_kw = {'width_ratios': [1.33, 0.33]}
        (ax, ax1) = subfigs[1].subplots(
            1,
            2,
            sharey=True,
            gridspec_kw=gs_kw
        )
        ax.grid(alpha=0.2)
        ax1.grid(False)
        color = plot_kwargs.get('color', f'C{np.random.randint(6)}')
        label = r'$\langle$' + f' {label} ' + r'$\rangle$'
        ax.plot(
            xarr.draw.values,
            xarr.mean('chain'),
            color=color,
            lw=1.5*LW,
            label=label,
            **plot_kwargs
        )
        for idx in range(num_chains):
            # ax = subfigs[0].subplots(1, 1)
            # plot values of invidual chains, arr[:, idx]
            # where arr[:, idx].shape = [ndraws, 1]
            # ax0.plot(
            #     xarr.draw.values,
            #     xarr[xarr.chain == idx][0],
            #     lw=1.,
            #     alpha=0.7,
            #     color=color
            # )
            ax.plot(
                xarr.draw.values,
                xarr[xarr.chain == idx][0],
                color=color,
                alpha=0.5,
                lw=LW/2.,
                **plot_kwargs
            )

        axes = (ax, ax1)
        sns.kdeplot(
            y=xarr.values.flatten(),
            ax=ax1,
            color=color,
            shade=True
        )
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        # ax1.set_yticks([])
        # ax1.set_yticklabels([])
        sns.despine(ax=ax, top=True, right=True)
        sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
        # ax.legend(loc='best', frameon=False)
        ax1.set_xlabel('')
        # ax1.set_ylabel('')
        # ax.set_yticks(ax.get_yticks())
        # ax.set_yticklabels(ax.get_yticklabels())
        # ax.set_ylabel(key)
        # _ = subfigs[1].subplots_adjust(wspace=-0.75)
        # if num_chains > 0 and len(arr.shape) > 1:
        # lw = LW / 2.
        # num_chains = np.min([
        #     16,
        #     len(xarr.coords['chain']),
        # ])
        sns.despine(subfigs[0])
        ax0 = subfigs[0].subplots(1, 1)
        im = xarr.plot(ax=ax0)           # type:ignore
        im.colorbar.set_label(label)      # type:ignore
        # ax0.plot(
        #     xarr.draw.values,
        #     xarr.mean('chain'),
        #     lw=2.,
        #     color=color
        # )
        # for idx in range(min(num_chains, i.shape[1])):
        # matplotx.line_labels()
        ax.set_xlabel('draw')
        if title is not None:
            fig.suptitle(title)

        if outdir is not None:
            assert label is not None
            # plt.savefig(Path(outdir).joinpath(f'{key}.svg'),
            #             dpi=400, bbox_inches='tight')
            outfile = Path(outdir).joinpath(f'{label}.svg')
            if outfile.is_file():
                tstamp = hplt.get_timestamp('%Y-%m-%d-%H%M%S')
                pngdir = Path(outdir).joinpath('pngs')
                pngdir.mkdir(exist_ok=True, parents=True)
                pngfile = pngdir.joinpath(f'{label}-{tstamp}.png')
                svgfile = Path(outdir).joinpath(f'{label}-{tstamp}.svg')
                plt.savefig(pngfile, dpi=400, bbox_inches='tight')
                plt.savefig(svgfile, dpi=400, bbox_inches='tight')

    def plot_all(
            self,
            num_chains: Optional[int] = None,
            therm_frac: float = 0.,
            title: Optional[str] = None,
            outdir: Optional[os.PathLike] = None,
            subplots_kwargs: Optional[dict[str, Any]] = None,
            plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        dataset = self.get_dataset()

        plt.style.use(opinionated.STYLES['opinionated_min'])
        _ = make_ridgeplots(
            dataset,
            outdir=outdir,
            drop_nans=True,
            drop_zeros=False,
            num_chains=num_chains,
            cmap='viridis',
            save_plot=(outdir is not None),
        )

        plt.style.use(opinionated.STYLES['opinionated_min'])
        max_chains = 64
        for idx, (key, val) in enumerate(dataset.data_vars.items()):
            # val = val
            color = f'C{idx%9}'
            plot_kwargs['color'] = color
            if 'chain' in val.dims:
                nchains = max(max_chains, len(val.coords['chain']))
                val = val.sel(chain=slice(0, nchains))
            fig, subfigs, ax = self.plot(
                val=val.values.T.real,
                key=str(key),
                title=title,
                outdir=outdir,
                therm_frac=therm_frac,
                num_chains=num_chains,
                plot_kwargs=plot_kwargs,
                subplots_kwargs=subplots_kwargs,
            )
            if fig is not None:
                _ = sns.despine(
                    fig,
                    top=True,
                    right=True,
                    bottom=True,
                    left=True
                )

            # _ = plt.grid(True, alpha=0.4)
            if subfigs is not None:
                # edgecolor = plt.rcParams['axes.edgecolor']
                plt.rcParams['axes.edgecolor'] = plt.rcParams['axes.facecolor']
                ax = subfigs[0].subplots(1, 1)
                # ax = fig[1].subplots(constrained_layout=True)
                _ = xplt.pcolormesh(
                    val,
                    'draw',
                    'chain',
                    ax=ax,
                    robust=True,
                    add_colorbar=True,
                    rasterized=True,
                )
                # im = val.plot(ax=ax, cbar_kwargs=cbar_kwargs)
                # im.colorbar.set_label(f'{key}')  # , labelpad=1.25)
                sns.despine(
                    subfigs[0],
                    top=True,
                    right=True,
                    left=True,
                    bottom=True
                )
            if outdir is not None:
                dirs = {
                    'png': Path(outdir).joinpath("pngs/"),
                    'svg': Path(outdir).joinpath("svgs/"),
                }
                _ = [
                    i.mkdir(exist_ok=True, parents=True) for i in dirs.values()
                ]
                log.info(
                    f"Saving {key} plot to: "
                    f"{Path(outdir).resolve().relative_to(PROJECT_DIR)}"
                )
                for ext, d in dirs.items():
                    outfile = d.joinpath(f"{key}.{ext}")
                    if outfile.is_file():
                        log.info(
                            f"Saving {key} plot to: "
                            f"{outfile.resolve().relative_to(PROJECT_DIR)}"
                        )
                        outfile = d.joinpath(f"{key}-subfig.{ext}")
                    # log.info(f"Saving {key}.ext to: {outfile}")
                    plt.savefig(outfile, dpi=400, bbox_inches='tight')
            if is_interactive():
                plt.show()

        return dataset

    def to_DataArray(
            self,
            x: Union[list, np.ndarray],
            therm_frac: Optional[float] = 0.0,
    ) -> xr.DataArray:
        try:
            arr = np.array(x).real
        except ValueError:
            arr = np.array(x)
            log.info(f'len(x): {len(x)}')
            log.info(f'x[0].shape: {x[0].shape}')
            log.info(f'arr.shape: {arr.shape}')
        if therm_frac is not None and therm_frac > 0:
            drop = int(therm_frac * arr.shape[0])
            arr = arr[drop:]
        # steps = np.arange(len(arr))
        if len(arr.shape) == 1:                     # [ndraws]
            ndraws = arr.shape[0]
            dims = ['draw']
            coords = [np.arange(len(arr))]
            return xr.DataArray(arr, dims=dims, coords=coords)  # type:ignore

        if len(arr.shape) == 2:                   # [nchains, ndraws]
            arr = arr.T
            nchains, ndraws = arr.shape
            dims = ('chain', 'draw')
            coords = [np.arange(nchains), np.arange(ndraws)]
            return xr.DataArray(arr, dims=dims, coords=coords)  # type:ignore

        if len(arr.shape) == 3:                   # [nchains, nlf, ndraws]
            arr = arr.T
            nchains, nlf, ndraws = arr.shape
            dims = ('chain', 'leapfrog', 'draw')
            coords = [np.arange(nchains), np.arange(nlf), np.arange(ndraws)]
            return xr.DataArray(arr, dims=dims, coords=coords)  # type:ignore

        else:
            print(f'arr.shape: {arr.shape}')
            raise ValueError('Invalid shape encountered')

    def get_dataset(
            self,
            data: Optional[dict[str, Union[list, np.ndarray]]] = None,
            therm_frac: Optional[float] = 0.0,
    ):
        data = self.history if data is None else data
        data_vars = {}
        for key, val in data.items():
            name = key.replace('/', '_')
            try:
                data_vars[name] = self.to_DataArray(val, therm_frac)
            except ValueError:
                log.error(f'Unable to create DataArray for {key}! Skipping!')
                log.error(f'{key}.shape= {np.stack(val).shape}')  # type:ignore

        return xr.Dataset(data_vars)
