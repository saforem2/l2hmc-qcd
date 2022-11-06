"""
history.py

Contains implementation of History object for tracking / aggregating metrics.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import matplotx
import numpy as np
import seaborn as sns
import tensorflow as tf
# from tensorflow.python.framework.ops import EagerTensor
import torch
import xarray as xr
from l2hmc.common import grab_tensor

from l2hmc.configs import MonteCarloStates, Steps
import l2hmc.utils.plot_helpers as hplt

# TensorLike = Union[tf.Tensor, torch.Tensor, np.ndarray]
TensorLike = Union[tf.Tensor, torch.Tensor, np.ndarray, list]
ScalarLike = Union[float, int, bool, np.floating, np.integer]

PT_FLOAT = torch.get_default_dtype()
TF_FLOAT = tf.dtypes.as_dtype(tf.keras.backend.floatx())
Scalar = Union[float, int, np.floating, bool]
# Scalar = TF_FLOAT | PT_FLOAT | np.floating | int | bool

log = logging.getLogger(__name__)

xplt = xr.plot  # type:ignore
LW = plt.rcParams.get('axes.linewidth', 1.75)


def format_pair(k: str, v: ScalarLike) -> str:
    if isinstance(v, (int, bool, np.integer)):
        return f'{k}={v:<3}'
    return f'{k}={v:<3.4f}'


def summarize_dict(d: dict) -> str:
    return ' '.join([format_pair(k, v) for k, v in d.items()])


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

        if (
            isinstance(metric, tf.Tensor)
            or isinstance(metric, torch.Tensor)
        ):
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

    def plot_dataArray(
            self,
            val: xr.DataArray,
            key: Optional[str] = None,
            therm_frac: float = 0.,
            num_chains: int = 16,
            title: Optional[str] = None,
            outdir: Optional[str] = None,
            subplots_kwargs: Optional[dict[str, Any]] = None,
            plot_kwargs: Optional[dict[str, Any]] = None,

    ) -> tuple:
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        figsize = subplots_kwargs.get('figsize', hplt.set_size())
        subplots_kwargs.update({'figsize': figsize})
        subfigs = None

        # tmp = val[0]
        arr = val.values  # shape: [nchains, ndraws]
        steps = np.arange(arr.shape[0])

        if therm_frac is not None and therm_frac > 0:
            drop = int(therm_frac * arr.shape[0])
            arr = arr[drop:]
            steps = steps[drop:]

        if len(arr.shape) == 2:
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
            ax.plot(steps, arr.mean(-1), lw=1.5*LW, label=label, **plot_kwargs)
            sns.kdeplot(y=arr.flatten(), ax=ax1, color=color, shade=True)
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
            ax0.plot(steps, arr.mean(0), lw=2., color=color)
            for idx in range(min(num_chains, arr.shape[0])):
                ax0.plot(steps, arr[idx, :], lw=1., alpha=0.7, color=color)

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
                    y = arr[:, idx, :].mean(-1)
                    pkwargs = {
                        'color': cmap(idx / nlf),
                        'label': f'{idx}',
                    }
                    ax.plot(steps, y, **pkwargs)
                axes = ax
            else:
                raise ValueError('Unexpected shape encountered')

            ax.set_ylabel(key)

            if num_chains > 0 and len(arr.shape) > 1:
                lw = LW / 2.
                for idx in range(min(num_chains, arr.shape[1])):
                    # ax = subfigs[0].subplots(1, 1)
                    # plot values of invidual chains, arr[:, idx]
                    # where arr[:, idx].shape = [ndraws, 1]
                    ax.plot(steps, arr[:, idx],
                            alpha=0.5, lw=lw/2., **plot_kwargs)

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
            num_chains: Optional[int] = 0,
            title: Optional[str] = None,
            outdir: Optional[str] = None,
            subplots_kwargs: Optional[dict[str, Any]] = None,
            plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        figsize = subplots_kwargs.get('figsize', hplt.set_size())
        subplots_kwargs.update({'figsize': figsize})
        num_chains = 16 if num_chains is None else num_chains

        # tmp = val[0]
        arr = np.array(val)

        subfigs = None
        steps = np.arange(arr.shape[0])
        if therm_frac is not None and therm_frac > 0:
            drop = int(therm_frac * arr.shape[0])
            arr = arr[drop:]
            steps = steps[drop:]

        if len(arr.shape) == 2:
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
            ax.plot(steps, arr.mean(-1), lw=1.5*LW, label=label, **plot_kwargs)
            sns.kdeplot(y=arr.flatten(), ax=ax1, color=color, shade=True)
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
                    y = arr[:, idx, :].mean(-1)
                    pkwargs = {
                        'color': cmap(idx / nlf),
                        'label': f'{idx}',
                    }
                    ax.plot(steps, y, **pkwargs)
                axes = ax
            else:
                raise ValueError('Unexpected shape encountered')

            ax.set_ylabel(key)

        if num_chains > 0 and len(arr.shape) > 1:
            lw = LW / 2.
            for idx in range(min(num_chains, arr.shape[1])):
                # ax = subfigs[0].subplots(1, 1)
                # plot values of invidual chains, arr[:, idx]
                # where arr[:, idx].shape = [ndraws, 1]
                ax.plot(steps, arr[:, idx], alpha=0.5, lw=lw/2., **plot_kwargs)

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

    def plot_all(
            self,
            num_chains: int = 0,
            therm_frac: float = 0.,
            title: Optional[str] = None,
            outdir: Optional[str] = None,
            subplots_kwargs: Optional[dict[str, Any]] = None,
            plot_kwargs: Optional[dict[str, Any]] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs

        dataset = self.get_dataset()
        for idx, (key, val) in enumerate(dataset.data_vars.items()):
            color = f'C{idx%9}'
            plot_kwargs['color'] = color

            _, subfigs, ax = self.plot(
                val=val.values.T.real,
                key=str(key),
                title=title,
                outdir=outdir,
                therm_frac=therm_frac,
                num_chains=num_chains,
                plot_kwargs=plot_kwargs,
                subplots_kwargs=subplots_kwargs,
            )
            if subfigs is not None:
                # edgecolor = plt.rcParams['axes.edgecolor']
                plt.rcParams['axes.edgecolor'] = plt.rcParams['axes.facecolor']
                ax = subfigs[0].subplots(1, 1)
                # ax = fig[1].subplots(constrained_layout=True)
                _ = xplt.pcolormesh(val, 'draw', 'chain', ax=ax,
                                    robust=True, add_colorbar=True)
                # im = val.plot(ax=ax, cbar_kwargs=cbar_kwargs)
                # im.colorbar.set_label(f'{key}')  # , labelpad=1.25)
                sns.despine(subfigs[0], top=True, right=True,
                            left=True, bottom=True)
                if outdir is not None:
                    Path(outdir).mkdir(exist_ok=True)
                    outfile = Path(outdir).joinpath(f'{key}.svg').as_posix()
                    print(f'Saving figure to: {outfile}')
                    plt.savefig(outfile, dpi=400, bbox_inches='tight')

        return dataset

    def to_DataArray(
            self,
            x: Union[list, np.ndarray],
            therm_frac: Optional[float] = 0.0,
    ) -> xr.DataArray:
        arr = np.array(x).real
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
