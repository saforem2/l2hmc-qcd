"""
history.py

Implements the History class for keeping a history of data.
"""
from __future__ import absolute_import, division, print_function, annotations
import numpy as np
from typing import Union, Any
import matplotlib.pyplot as plt
from utils.step_timer import StepTimer
from pathlib import Path
import matplotx
import tensorflow as tf
import xarray as xr
from utils.data_containers import invert_list_of_dicts
import seaborn as sns
from utils.plotting_utils import set_size

LW = plt.rcParams.get('axes.linewidth', 1.75)

import logging
logger = logging.getLogger('l2hmc')


class Metrics:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def to_dict(self):
        return self.__dict__


class innerHistory:
    def __init__(self, names=None):
        self.__dict__ = {}
        #  self.metrics = {}
        if names is not None:
            for name in names:
                self.__dict__[name] = []

    def update(self, metrics: Union[dict[str, list], Metrics]):
        if isinstance(metrics, Metrics):
            metrics = metrics.to_dict()

        for key, val in metrics.items():
            try:
                self.__dict__[key].append(val)
            except KeyError:

                self.__dict__[key] = [val]


class History:
    def __init__(self, names: list[str] = None, timer: StepTimer = None):  #, data: dict[str, list] = None):
        self.steps = []
        self.running_avgs = {}
        self._names = names
        if timer is None:
            self.step = 0
            self.timer = StepTimer()
        else:
            self.step = len(timer.data)
        self.timer = timer if timer is not None else StepTimer()
        self.data = {name: [] for name in names} if names is not None else {}

    def _reset(self):
        self.steps = []
        self.running_avgs = {}
        self.data = ({name: [] for name in self._names}
                     if self._names is not None else {})

    @staticmethod
    def _running_avg(x: Union[list, np.ndarray], window: int = 10):
        arr = np.array(x)
        win = min((window, arr.shape[0]))
        if len(arr.shape) == 1:               # x.shape = [nsteps,]
            return arr[-win:].mean()

        return arr[-win:].mean(0)

    def _running_avgs(self, window: int = 10):
        return {k: self._running_avg(v, window) for k, v in self.data.items()}

    def update_running_avgs(self, window: int = 10):
        ravgs = self._running_avgs(window)
        for key, val in ravgs.items():
            try:
                self.running_avgs[key].append(val)
            except KeyError:
                self.running_avgs[key] = val

        return ravgs

    def strformat(
            self,
            k: str,
            v: Union[dict, tf.Tensor, list, np.ndarray],
            window: int = 0,
    ):
        outstr = ''
        if isinstance(v, dict):
            outstr_arr = []
            for key, val in v.items():
                outstr_arr.append(self.strformat(key, val, window))

            outstr = '\n'.join(outstr_arr)

        elif tf.is_tensor(v):
            assert v is not None and isinstance(v, tf.Tensor)
            v = v.numpy().squeeze()

        elif isinstance(v, (list, np.ndarray)):
            v = np.array(v).squeeze()
            # v = torch.Tensor(v).numpy()
            if window > 0 and len(v.shape) > 0:
                window = min((v.shape[0], window))
                avgd = np.mean(v[-window:])
            else:
                avgd = np.mean(v)

            outstr = f'{str(k)}={avgd:<3.2g}'

        elif isinstance(v, float):
            outstr = f'{str(k)}={v:<3.2f}'

        else:
            outstr = f'{str(k)}={v:<3}'

        return outstr

    def generic_summary(
            self,
            data: dict,
            window: int = 0,
            pre: Union[str, list[str]] = None,
            skip: Union[str, list[str]] = None,
            keep: Union[str, list[str]] = None,
    ) -> str:
        skip = [] if skip is None else skip
        keep = list(data.keys()) if keep is None else keep
        fstrs = [
            self.strformat(k, v, window) for k, v in data.items()
            if k not in skip
            and k in keep
        ]
        if pre is not None:
            fstrs = [pre, *fstrs] if isinstance(pre, str) else [*pre] + fstrs

        outstr = ' '.join(fstrs)
        return outstr

    def metrics_summary(self, **kwargs) -> str:
        """A wrapper around generic summary, using self.data"""
        return self.generic_summary(self.data, **kwargs)

    def summary(
            self,
            data: dict = None,
            window: int = 0,
            pre: Union[str, list[str]] = None,
            skip: Union[str, list[str]] = None,
            keep: Union[str, list[str]] = None,
    ) -> str:
        """Returns a summary of the items in `data` over the last `window`"""
        data = self.data if data is None else data
        mstr = self.generic_summary(data=data, window=window,
                                    skip=skip, keep=keep, pre=pre)

        return mstr

    def print_summary(
            self,
            mstr: str = None,
            **kwargs,
    ) -> str:
        if mstr is None:
            mstr = self.generic_summary(self.data, **kwargs)

        logger.log(f'{mstr}')

        return mstr

    def update(
        self,
        metrics: dict,
        step: int = None,
        # update_avgs: bool = False,
        # window: int = 0,
    ):
        if step is not None:
            self.steps.append(step)

        for k, v in metrics.items():
            if isinstance(v, tf.Tensor):
                # if torch.cuda.is_available():
                #     v = v.cpu()

                v = v.numpy().squeeze()
            try:
                self.data[k].append(v)
            except KeyError:
                self.data[k] = [v]

        # if update_avgs:
        #     _ = self.update_running_avgs(window)

    def finalize(self):
        data = {}
        for key, val in self.data.items():
            try:
                arr = tf.Tensor(val).numpy()
            except:
                arr = tf.Tensor([i.numpy() for i in val]).numpy()

            data[key] = arr

        self.data = data
        return data

    def plot(
            self,
            val: list | np.ndarray | tf.Tensor,
            key: str = None,
            therm_frac: float = 0.,
            num_chains: int = 0,
            title: str = None,
            outdir: str = None,
            subplots_kwargs: dict[str, Any] = None,
            plot_kwargs: dict[str, Any] = None,
            ext: str = 'svg',
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
        figsize = subplots_kwargs.get('figsize', set_size())
        subplots_kwargs.update({'figsize': figsize})
        key = '' if key is None else key
        label = ' '.join([r'$\langle$', f'{key}', r'$\rangle$'])

        if isinstance(val[0], tf.Tensor):
            arr = val.numpy()
        elif isinstance(val[0], float):
            arr = np.array(val)
        else:
            try:
                arr = np.array([np.array(i) for i in val])
            except (AttributeError, ValueError) as exc:
                raise exc

        subfigs = None
        steps = np.arange(arr.shape[0])
        if therm_frac > 0:
            drop = int(therm_frac * arr.shape[0])
            arr = arr[drop:]
            steps = steps[drop:]

        if len(arr.shape) == 2:
            _ = subplots_kwargs.pop('constrained_layout', True)
            figsize = (3 * figsize[0], 1.5 * figsize[1])

            fig = plt.figure(figsize=figsize, constrained_layout=True)
            subfigs = fig.subfigures(1, 2)
            # , wspace=0.1)#, width_ratios=[1., 1.5])

            gs_kw = {'width_ratios': [1.33, 0.33]}
            (ax, ax1) = subfigs[1].subplots(1, 2, sharey=True,
                                            gridspec_kw=gs_kw)
            ax.grid(alpha=0.2)
            ax1.grid(False)
            color = plot_kwargs.get('color', None)
            ax.plot(steps, arr.mean(-1), lw=1.5*LW, label=label, **plot_kwargs)
            sns.kdeplot(y=arr.flatten(), ax=ax1, color=color, shade=True)
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            sns.despine(ax=ax, top=True, right=True)
            sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
            ax1.set_xlabel('')
            ax1.set_ylabel('')
            axes = (ax, ax1)
            matplotx.line_labels(ax=ax)
        else:
            if len(arr.shape) == 1:
                fig, ax = plt.subplots(**subplots_kwargs)
                ax.plot(steps, arr, label=label, **plot_kwargs)
                axes = ax
                matplotx.line_labels(ax=ax)
            elif len(arr.shape) == 3:
                fig, ax = plt.subplots(**subplots_kwargs)
                for idx in range(arr.shape[1]):
                    if idx == 0:
                        ax.plot(steps, arr[:, idx, :].mean(-1),
                                label=label, **plot_kwargs)
                    else:
                        ax.plot(steps, arr[:, idx, :].mean(-1), **plot_kwargs)
                axes = ax
                matplotx.line_labels(ax=ax)
            else:
                raise ValueError('Unexpected shape encountered')

            ax.set_ylabel(key)

        if num_chains > 0 and len(arr.shape) > 1:
            for idx in range(min(num_chains, arr.shape[1])):
                # plot values of invidual chains, arr[:, idx]
                # where arr[:, idx].shape = [ndraws, 1]
                ax.plot(steps, arr[:, idx], alpha=0.4, lw=LW/5., **plot_kwargs)

        ax.set_xlabel('draw')
        if title is not None:
            fig.suptitle(title)

        if outdir is not None:
            fig.savefig(Path(outdir).joinpath(f'{key}.{ext}'))

        return fig, subfigs, axes

    def plot_all(
            self,
            num_chains: int = 0,
            therm_frac: float = 0.,
            title: str = None,
            outdir: str = None,
            subplots_kwargs: dict[str, Any] = None,
            plot_kwargs: dict[str, Any] = None,
            skip: list[str] = None,
            ext: str = 'svg',
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs

        dataset = self.get_dataset()
        for idx, (key, val) in enumerate(dataset.data_vars.items()):
            if skip is not None and key in skip:
                continue

            color = f'C{idx%9}'
            plot_kwargs['color'] = color

            _, subfigs, ax = self.plot(
                val=val.values.T,
                key=str(key),
                title=title,
                outdir=outdir,
                therm_frac=therm_frac,
                num_chains=num_chains,
                plot_kwargs=plot_kwargs,
                subplots_kwargs=subplots_kwargs,
                ext=ext,
            )
            if subfigs is not None:
                edgecolor = plt.rcParams['axes.edgecolor']
                plt.rcParams['axes.edgecolor'] = plt.rcParams['axes.facecolor']
                ax = subfigs[0].subplots(1, 1)
                im = val.plot(ax=ax, robust=True)  # , rasterized=True
                im.colorbar.set_label(f'{key}')
                sns.despine(subfigs[0], top=True,
                            right=True, left=True, bottom=True)

                if 'eps' in key:
                    ax.set_ylabel('leapfrog step')

                if outdir is not None:
                    Path(outdir).mkdir(exist_ok=True)
                    outfile = Path(outdir).joinpath(f'{key}.{ext}').as_posix()
                    logger.debug(f'Saving figure to: {outfile}')
                    plt.savefig(outfile)

        return dataset

    def finalize_data(self):
        for key, val in self.data.items():
            if isinstance(val, list):
                if isinstance(val[0], dict):
                    self.data[key] = invert_list_of_dicts(val)

    def to_DataArray(self, x: Union[list, np.ndarray]) -> xr.DataArray:
        arr = np.array(x)
        # steps = np.arange(len(arr))
        if len(arr.shape) == 1:                     # [ndraws]
            ndraws = arr.shape[0]
            dims = ['draw']
            coords = [np.arange(len(arr))]
            return xr.DataArray(arr, dims=dims, coords=coords)

        if len(arr.shape) == 2:                   # [nchains, ndraws]
            arr = arr.T
            nchains, ndraws = arr.shape
            dims = ('chain', 'draw')
            coords = [np.arange(nchains), np.arange(ndraws)]
            return xr.DataArray(arr, dims=dims, coords=coords)

        if len(arr.shape) == 3:                   # [nchains, nlf, ndraws]
            arr = arr.T
            nchains, nlf, ndraws = arr.shape
            dims = ('chain', 'leapfrog', 'draw')
            coords = [np.arange(nchains), np.arange(nlf), np.arange(ndraws)]
            return xr.DataArray(arr, dims=dims, coords=coords)

        else:
            raise ValueError('Invalid shape encountered')


    def get_dataset(self, data: dict[str, Union[list, np.ndarray]] = None):
        data = self.data if data is None else data
        data_vars = {}
        for key, val in data.items():
            # TODO: FIX ME
            # if isinstance(val, list):
            #     if isinstance(val[0], (dict, AttrDict)):
            #         tmp = invert
            #      data_vars[key] = dataset = self.get_dataset(val)

            data_vars[key] = self.to_DataArray(val)

        return xr.Dataset(data_vars)


class TimedHistory:
    def __init__(self, history: History = None, timer: StepTimer = None):
        self.history = history
        self.timer = timer
