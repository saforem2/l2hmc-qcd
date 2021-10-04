"""
data_containers.py

Implements `TrainData` class, for working with training data.
"""
from __future__ import absolute_import, division, print_function, annotations

import os
from pathlib import Path
import time

from collections import defaultdict
from typing import Any, Union

import joblib
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
from rich.console import Console

from utils import SKEYS
import utils.file_io as io

from config import BASE_DIR
from utils.attr_dict import AttrDict
from utils.data_utils import therm_arr
from utils.hvd_init import RANK, SIZE
from utils.plotting_utils import (set_size, make_ridgeplots, mcmc_lineplot,
                                  mcmc_traceplot, get_title_str_from_params,
                                  plot_data)

from utils.logger import Logger, in_notebook

logger = Logger()
LW = int(plt.rcParams['axes.linewidth'])

plt.style.use('default')
sns.set_context('paper')
sns.set_style('whitegrid')
sns.set_palette('bright')


VERBOSE = os.environ.get('VERBOSE', False)

import torch
import tensorflow as tf


class Metrics:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def to_dict(self):
        return self.__dict__


class StepTimer:
    def __init__(self, evals_per_step: int = 1):
        self.data = []
        self.t = time.time()
        self.evals_per_step = evals_per_step

    def start(self):
        self.t = time.time()

    def stop(self):
        dt = time.time() - self.t
        self.data.append(dt)

        return dt

    def reset(self):
        self.data = []

    def get_eval_rate(self, evals_per_step: int = None):
        if evals_per_step is None:
            evals_per_step = self.evals_per_step

        elapsed = np.sum(self.data)
        num_evals = SIZE * evals_per_step * len(self.data)
        eval_rate = num_evals / elapsed
        output = {
            'eval_rate': eval_rate,
            'total_time': elapsed,
            'num_evals': num_evals,
            'size': SIZE,
            'num_steps': len(self.data),
            'evals_per_step': evals_per_step,
        }

        return output

    def write_eval_rate(
            self,
            outdir: Union[str, Path],
            mode: str = 'w',
            **kwargs,
    ):
        eval_rate = self.get_eval_rate(**kwargs)
        outfile = Path(outdir).joinpath('eval_rate.txt')
        logger.debug(f'Writing eval rate to: {outfile}')
        with open(outfile, mode) as f:
            f.write('\n'.join([f'{k}: {v}' for k, v  in eval_rate.items()]))

        return eval_rate

    def save_and_write(
            self,
            outdir: Union[str, Path],
            mode: str = 'w',
            **kwargs,
    ):
        eval_rate = self.write_eval_rate(outdir, mode=mode, **kwargs)
        outfile = str(Path(outdir).joinpath('step_times.csv'))
        data = self.save_data(outfile, mode=mode)
        return {'eval_rate': eval_rate, 'step_times': data}

    def save_data(self, outfile: str, mode='w'):
        df = pd.DataFrame(self.data)

        fpath = Path(outfile).resolve()
        fpath.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f'Saving step times to: {outfile}')
        df.to_csv(str(fpath), mode=mode)

        return df



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


class TimedHistory:
    def __init__(self, history: History = None, timer: StepTimer = None):
        self.history = history
        self.timer = timer


def grab(x: torch.Tensor):
    if x.device == torch.device('cuda'):
        return x.cpu().detach().numpy().squeeze()
    return x.detach().numpy().squeeze()

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
        if isinstance(x, list):
            x = np.array(x)

        win = min((window, x.shape[0]))
        if len(x.shape) == 1:               # x.shape = [nsteps,]
            return x[-win:].mean()

        return x[-win:].mean(0)

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

    def strformat(self, k, v, window = None):
        if v is None:
            v = 'None'

        outstr = ''
        if isinstance(v, dict):
            outstr_arr = []
            for key, val in v.items():
                outstr_arr.append(self.strformat(key, val, window))

            outstr = '\n'.join(outstr_arr)

        else:
            if isinstance(v, torch.Tensor):
                v = grab(v)
                # v = v.detach().numpy().squeeze()

            if isinstance(v, tf.Tensor):
                v = v.numpy().squeeze()

            if isinstance(v, (list, np.ndarray)):
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
            skip: list[str] = None,
            keep: list[str] = None,
            pre: Union[str, list, tuple] = None,
    ):
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
        should_print = kwargs.pop('should_print', False) 
        mstr = self.generic_summary(self.data, **kwargs)
        # if should_print:
        #     self.logger.log(mstr)
        # logger.console.log(mstr) if in_notebook() else logger.info(mstr)

        return mstr

    def print_summary(
            self,
            mstr: str = None,
            logger: Union[Logger, Console] = None,
            **kwargs,
    ) -> str:
        if mstr is None:
            mstr = self.generic_summary(self.data, **kwargs)

        logger = logger if logger is not None else Logger()
        logger.log(f'{mstr}')

        return mstr

    def running_avgs_summary(self, **kwargs) -> str:
        # if len(list(self.running_avgs.keys())) == 0:
        #     self.update_running_avgs()

        # return self.generic_summary(self.running_avgs, **kwargs)
        pass

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
            if isinstance(v, torch.Tensor):
                v = grab(v.to('cpu').detach())
                # if torch.cuda.is_available():
                #     v = v.cpu()

                # v = v.detach().numpy().squeeze()
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
                arr = torch.Tensor(val).numpy()
            except:
                arr = torch.Tensor([i.numpy() for i in val]).numpy()

            data[key] = arr

        self.data = data
        return data

    def plot(
            self,
            val: torch.Tensor,
            key: str = None,
            therm_frac: float = 0.,
            num_chains: int = 0,
            title: str = None,
            outdir: str = None,
            subplots_kwargs: dict[str, Any] = None,
            plot_kwargs: dict[str, Any] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {'figsize': (4, 3)} if subplots_kwargs is None else subplots_kwargs
        figsize = subplots_kwargs.get('figsize', (4, 3))
        # assert key in self.data

        # val = self.data[key]
        # color = f'C{idx%9}'
        # tmp = val[0]
        # -----
        if isinstance(val, torch.Tensor):
            if isinstance(val[0], torch.Tensor):
                try:
                    val = np.array([grab(i) for i in val])
                except (AttributeError, ValueError) as exc:
                    raise exc
            else:
                val = grab(val)
        else:
            try:
                val = np.array([np.array(grab(i)) for i in val])
            except (AttributeError, ValueError) as exc:
                raise exc
        # if isinstance(tmp, torch.Tensor):
        #     arr = grab(val.detach().numpy())
        # elif isinstance(tmp, tf.Tensor):
        #     arr = val.numpy()
        # elif isinstance(tmp, float):
        #     arr = np.array(val)
        # else:
        #     try:
        #         arr = np.array([np.array(i) for i in val])
        #     except (AttributeError, ValueError) as exc:
        #         raise exc

        # -----------------------
        subfigs = None
        # steps = np.arange(arr.shape[0])
        if len(val.shape) == 0:
            steps = np.arange(len(val))
        else:
            steps = np.arange(val.shape[0])

        if therm_frac > 0:
            drop = int(therm_frac * val.shape[0])
            val = val[drop:]
            steps = steps[drop:]

        if len(val.shape) == 2:
            _ = subplots_kwargs.pop('constrained_layout', True)

            figsize = (2 * figsize[0], figsize[1])

            fig = plt.figure(figsize=figsize, constrained_layout=True, dpi=200)
            subfigs = fig.subfigures(1, 2, wspace=0.1, width_ratios=[1., 1.5])

            gs_kw = {'width_ratios': [1.25, 0.5]}
            (ax, ax1) = subfigs[1].subplots(1, 2, gridspec_kw=gs_kw, sharey=True)
            # (ax, ax1) = fig.subfigures(1, 1).subplots(1, 2)
            # gs = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[1.5, 1., 1.5])
            color = plot_kwargs.get('color', None)
            label = r'$\langle$' + f' {key} ' + r'$\rangle$'
            ax.plot(steps, val.mean(-1), lw=2.*LW, label=label, **plot_kwargs)

            sns.kdeplot(y=val.flatten(), ax=ax1, color=color, shade=True)
            #ax1.set_yticks([])
            #ax1.set_yticklabels([])
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            sns.despine(ax=ax, top=True, right=True)
            sns.despine(ax=ax1, top=True, right=True, left=True, bottom=True)
            ax.legend(loc='best', frameon=False)
            ax1.grid(False)
            ax1.set_xlabel('')
            ax1.set_ylabel('')
            ax.set_ylabel(key, fontsize='large')
            axes = (ax, ax1)
        else:
            if len(val.shape) == 1:
                fig, ax = plt.subplots(**subplots_kwargs)
                ax.plot(steps, val, **plot_kwargs)
                axes = ax
            elif len(val.shape) == 3:
                fig, ax = plt.subplots(**subplots_kwargs)
                for idx in range(val.shape[1]):
                    ax.plot(steps, val[:, idx, :].mean(-1), label='idx', **plot_kwargs)
                axes = ax
            else:
                raise ValueError('Unexpected shape encountered')

            ax.set_ylabel(key)

        if num_chains > 0 and len(val.shape) > 1:
            num_chains = val.shape[1] if (num_chains > val.shape[1]) else num_chains
            for idx in range(num_chains):
                ax.plot(steps, val[:, idx], alpha=0.5, lw=LW/4, **plot_kwargs)

        ax.set_xlabel('draw', fontsize='large')
        if title is not None:
            fig.suptitle(title)

        if outdir is not None:
            fig.savefig(Path(outdir).joinpath(f'{key}.pdf'))

        return fig, subfigs, axes

    def plot_all(
            self,
            num_chains: int = 0,
            therm_frac: float = 0.,
            title: str = None,
            outdir: str = None,
            subplots_kwargs: dict[str, Any] = None,
            plot_kwargs: dict[str, Any] = None,
    ):
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs

        dataset = self.get_dataset()
        for idx, (key, val) in enumerate(dataset.data_vars.items()):
            color = f'C{idx%9}'
            plot_kwargs['color'] = color

            _, subfigs, ax = self.plot(
                val=torch.from_numpy(val.values).T,
                key=str(key),
                title=title,
                outdir=outdir,
                therm_frac=therm_frac,
                num_chains=num_chains,
                plot_kwargs=plot_kwargs,
                subplots_kwargs=subplots_kwargs,
            )
            # if isinstance(subfigs, tuple):
            if subfigs is not None:
                # _, subfigs = fig
                # ax1 = subfigs[1].subplots(1, 1)

                edgecolor = plt.rcParams['axes.edgecolor']
                plt.rcParams['axes.edgecolor'] = plt.rcParams['axes.facecolor']
                ax = subfigs[0].subplots(1, 1)
                # ax = fig[1].subplots(constrained_layout=True)
                cbar_kwargs = {
                    # 'location': 'top',
                    # 'orientation': 'horizontal',
                }
                im = val.plot(ax=ax, cbar_kwargs=cbar_kwargs)
                im.colorbar.set_label(f'{key}', fontsize='large') #, labelpad=1.25)
                sns.despine(subfigs[0], top=True, right=True, left=True, bottom=True)
                # sns.despine(im.axes, top=True, right=True, left=True, bottom=True)
                plt.rcParams['axes.edgecolor'] = edgecolor

            # else:
            #     ax1 = fig.add_subplot(1, 2, 2)
            #     val.plot(ax=ax1)

        return dataset

    def get_dataset(self):
        data_vars = {}
        for key, val in self.data.items():
            arr = np.array(val)
            steps = np.arange(len(arr))
            if len(arr.shape) == 1:
                dims = ['draw']
                coords = [steps]
            elif len(arr.shape) == 2:
                nchains = arr.shape[1]
                dims = ['chain', 'draw']
                coords = [np.arange(nchains), steps]
            elif len(arr.shape) == 3:
                arr = arr.T
                nchains, nlf, _ = arr.shape
                dims = ['chain', 'leapfrog', 'draw']
                coords = [np.arange(nchains), np.arange(nlf), steps]
            else:
                raise ValueError('Invalid shape encountered')

            data_vars[key] = xr.DataArray(arr.T, dims=dims, coords=coords)

        return xr.Dataset(data_vars)




class DataContainer:
    """Base class for dealing with data."""

    def __init__(self, steps, header=None, dirs=None, print_steps=100):
        if dirs is not None:
            dirs = AttrDict(**dirs)
            names = [v for k, v in dirs.items() if 'file' not in k]
            io.check_else_make_dir(names)

        self.dirs = dirs
        self.steps = steps
        self.steps_arr = []
        self.data_strs = []
        self.print_steps = print_steps
        self.data = AttrDict(defaultdict(list))

    def update_dirs(self, dirs: dict):
        """Update `self.dirs` with key, val pairs from `dirs`."""
        if self.dirs is None:
            self.dirs = AttrDict(**dirs)
        else:
            for key, val in dirs.items():
                self.dirs.update({key: val})

    def plot_data(
            self,
            out_dir: str = None,
            flags: AttrDict = None,
            therm_frac: float = 0.,
            params: AttrDict = None,
    ):
        """Make plots from `self.data`."""
        plot_data(self.data, out_dir, flags,
                  therm_frac=therm_frac, params=params)

    def get_dataset(self, therm_frac=0., make_plots=False) -> (xr.Dataset):
        """Create `xr.Dataset` from `self.data`."""
        data_vars = {}
        for key, val in self.data.items():
            arr = np.array(val)
            steps = np.arange(len(arr))
            if therm_frac > 0:
                arr, steps = therm_arr(arr, therm_frac=therm_frac)
            if len(arr.shape) == 1:
                data_vars[key] = xr.DataArray(arr, dims=['draw'],
                                              coords=[steps])
            elif len(arr.shape) == 3:
                arr = arr.T
                num_chains, num_lf, _ = arr.shape
                dims = ['chain', 'leapfrog', 'draw']
                coords = [np.arange(num_chains), np.arange(num_lf), steps]
                data_vars[key] = xr.DataArray(arr, dims=dims, coords=coords)
            else:
                chains = np.arange(arr.shape[1])
                data_vars[key] = xr.DataArray(arr.T, dims=['chain', 'draw'],
                                              coords=[chains, steps])

        return xr.Dataset(data_vars)

    def save_dataset(self, out_dir, therm_frac=0.):
        """Save `self.data` as `xr.Dataset` to `out_dir/dataset.nc`."""
        dataset = self.get_dataset(therm_frac)
        out_file = os.path.join(out_dir, 'dataset.nc')
        logger.debug(f'Saving dataset to: {out_file}.')
        try:
            dataset.to_netcdf(os.path.join(out_dir, 'dataset.nc'))
        except ValueError:
            logger.warning('Unable to save dataset! Continuing...')

        return dataset

    def to_h5pyfile(self, hfile: Union[str, Path], **kwargs):
        logger.debug(f'Saving self.data to {hfile}')
        f = h5py.File(hfile, 'a')
        for key, val in self.data.items():
            arr = np.array(val)
            if len(arr) == 0:
                continue

            if key in list(f.keys()):
                f[key].resize((f[key].shape[0] + arr.shape[0]), axis=0)
                f[key][-arr.shape[0]:] = arr
            else:
                maxshape = (None,)
                if len(arr.shape) > 1:
                    maxshape = (None, *arr.shape[1:])
                f.create_dataset(key, data=arr, maxshape=maxshape, **kwargs)

            #  self.data[key] = []

        f.close()

    def restore_data(self, hfile):
        self.data = self.from_h5pyfile(hfile)

    def from_h5pyfile(self, hfile: Union[str, Path]):
        logger.debug(f'Loading from {hfile}')
        f = h5py.File(hfile, 'r')
        data = {key: f[key] for key in list(f.keys())}
        f.close()

        return data

    def print_metrics(self, metrics: dict, **kwargs):
        data_str = logger.print_metrics(metrics, **kwargs)
        self.data_strs.append(data_str)
        return data_str

    def update(self, step, metrics):
        """Update `self.data` with new values from `data`."""
        self.steps_arr.append(step)
        for key, val in metrics.items():
            if key not in self.data.keys():
                self.data[key] = []

            assert isinstance(self.data[key], list)
            if isinstance(val, np.ndarray):
                self.data[key].append(val)
            else:
                try:
                    val = grab(val)
                    # val.detach()
                except AttributeError:
                    try:
                        self.data[key].append(val.numpy())
                    except KeyError:
                        try:
                            val.detach()
                        except AttributeError:
                            self.data[key] = [val.numpy()]
            # try:
            #     self.data[key].append(val))


    # pylint:disable=too-many-arguments
    def get_header(self, metrics=None, prepend=None,
                   append=None, skip=None, split=False, with_sep=True):
        """Get nicely formatted header of variable names for printing."""
        if metrics is None:
            metrics = self.data

        header = io.make_header_from_dict(metrics, skip=skip, split=split,
                                          prepend=prepend, append=append,
                                          with_sep=with_sep)

        if self.data_strs[0] != header:
            self.data_strs.insert(0, header)

        return header


    def get_fstr(
            self,
            step: int,
            metrics: dict,
            skip: list = None,
            keep: list = None,
            #  skip_endpts: bool = True
    ):
        """Get formatted data string from `data`."""
        skip = [] if skip is None else skip

        #  data = {
        #      k: tf.reduce_mean(v) for k, v in metrics.items() if (
        #          k not in skip
        #          and not isinstance(v, dict)
        #          and k not in SKEYS
        #      )
        #  }

        #  if keep is not None:
        #      metrics = {k: v for k, v in metrics.items() if k in keep}

        #  if skip_endpts:
        #      data = {k: v for k, v in metrics.items() if (
        #          '_start' not in str(k)
        #          and '_mid' not in str(k)
        #          and '_end' not in str(k)
        #      )}
        #
        conds = lambda k: (
            k not in skip
            and k not in SKEYS
            and (k in keep if keep is not None else True)
            and '_start' not in str(k)
            and '_mid' not in str(k)
            and '_end' not in str(k)
        )

        sstr = 'step'
        fstr = (
            f'{sstr:s}: {step:5g}/{self.steps:<5g} ' + ' '.join([
                io.strformat(k, v, window=10) for k, v in metrics.items()
                if conds(k)
                #  if k not in skip
                #  and not isinstance(v, dict)
                #  and k not in SKEYS
                #  and k in keep if keep is not None
                #  and '_start' not in str(k)
                #  and '_mid' not in str(k)
                #  and '_end' not in str(k)
                #  f'{k:s}={v:5.3g}' for k, v in data.items()
            ])
        )

        self.data_strs.append(fstr)

        return fstr

    def restore(self, data_dir, rank=0, local_rank=0, step=None, x_shape=None):
        """Restore `self.data` from `data_dir`."""
        if step is not None:
            self.steps += step

        x_file = os.path.join(data_dir, f'x_rank{rank}-{local_rank}.z')
        try:
            x = io.loadz(x_file)
            logger.info(f'Restored `x` from: {x_file}.')
        except FileNotFoundError:
            logger.warning(f'Unable to load `x` from {x_file}.')
            x = np.random.uniform(-np.pi, np.pi, size=x_shape)
            # x = tf.random.uniform(x_shape, minval=np.pi, maxval=np.pi)

        data = self.load_data(data_dir)
        for key, val in data.items():
            arr = np.array(val)
            shape = arr.shape
            self.data[key] = arr.tolist()
            logger.debug(f'Restored: train_data.data[{key}].shape={shape}')

        return x

    @staticmethod
    def load_data(data_dir: Union[str, Path]):
        """Load data from `data_dir` and populate `self.data`."""
        # TODO: Update to use h5py for `.hdf5` files
        contents = os.listdir(data_dir)
        fnames = [i for i in contents if i.endswith('.z')]
        keys = [i.rstrip('.z') for i in fnames]
        data_files = [os.path.join(data_dir, i) for i in fnames]
        data = {}
        for key, val in zip(keys, data_files):
            if 'x_rank' in key:
                continue

            data[key] = io.loadz(val)
            if VERBOSE:
                logger.info(f'Restored {key} from {val}.')


        return AttrDict(data)

    def save_data(self, data_dir: Union[str, Path], **kwargs):
        """Save `self.data` entries to individual files in `output_dir`."""
        data_dir = Path(data_dir)
        io.check_else_make_dir(str(data_dir))
        hfile = data_dir.joinpath(f'data_rank{RANK}.hdf5')
        self.to_h5pyfile(hfile, compression='gzip')
        #  for key, val in self.data.items():
        #      if skip_keys is not None:
        #          if key in skip_keys:
        #              continue
        #
        #      out_file = os.path.join(data_dir, f'{key}.z')
        #      head, tail = os.path.split(out_file)
        #      io.check_else_make_dir(head)
        #      io.savez(np.array(val), out_file)
        #
        #  if save_dataset:
        #      self.save_dataset(data_dir)

    def plot_dataset(
            self,
            out_dir: str = None,
            therm_frac: float = 0.,
            num_chains: int = None,
            ridgeplots: bool = True,
            profile: bool = False,
            cmap: str = 'viridis_r',
    ):
        """Create trace plot + histogram for each entry in self.data."""
        tdict = {}
        dataset = self.get_dataset(therm_frac)
        for key, val in dataset.data_vars.items():
            t0 = time.time()
            if np.std(val.values.flatten()) < 1e-2:
                continue

            if len(val.shape) == 2:  # shape: (chain, draw)
                val = val[:num_chains, :]

            if len(val.shape) == 3:  # shape: (chain, leapfrogs, draw)
                val = val[:num_chains, :, :]

            fig, ax = plt.subplots(constrained_layout=True, figsize=set_size())
            _ = val.plot(ax=ax)
            if out_dir is not None:
                io.check_else_make_dir(out_dir)
                out_file = os.path.join(out_dir, f'{key}_xrPlot.png')
                fig.savefig(out_file, dpi=400, bbox_inches='tight')
                plt.close('all')
                plt.clf()

            if profile:
                tdict[key] = time.time() - t0

        if out_dir is not None:
            out_dir = os.path.join(out_dir, 'ridgeplots')
            io.check_else_make_dir(out_dir)

        if ridgeplots:
            make_ridgeplots(dataset, num_chains=num_chains,
                            out_dir=out_dir, cmap=cmap)

        return dataset, tdict

    def flush_data_strs(self, out_file, rank=0, mode='a'):
        """Dump `data_strs` to `out_file` and return new, empty list."""
        if rank == 0:
            with open(out_file, mode) as f:
                for s in self.data_strs:
                    f.write(f'{s}\n')

        self.data_strs = []

    def write_to_csv(self, log_dir, run_dir, hmc=False):
        """Write data averages to bulk csv file for comparing runs."""
        _, run_str = os.path.split(run_dir)
        avg_data = {
            'log_dir': log_dir,
            'run_dir': run_str,
            'hmc': hmc,
        }

        for key, val in dict(sorted(self.data.items())).items():
            # tensor = tf.convert_to_tensor(val)
            try:
                arr = val.numpy()
            except AttributeError:
                try:
                    arr = np.array(val)
                except ValueError:
                    arr = val

            arr, steps = therm_arr(arr, therm_frac=0.2)
            if 'steps' not in avg_data:
                avg_data['steps'] = len(steps)
            avg_data[key] = np.mean(arr)

        avg_df = pd.DataFrame(avg_data, index=[0])
        outdir = os.path.join(BASE_DIR, 'logs', 'GaugeModel_logs')
        csv_file = os.path.join(outdir, 'inference.csv')
        head, tail = os.path.split(csv_file)
        io.check_else_make_dir(head)
        logger.info(f'Appending inference results to {csv_file}.')
        if not os.path.isfile(csv_file):
            avg_df.to_csv(csv_file, header=True, index=False, mode='w')
        else:
            avg_df.to_csv(csv_file, header=False, index=False, mode='a')

    @staticmethod
    def dump_configs(x, data_dir, rank=0, local_rank=0):
        """Save configs `x` separately for each rank."""
        xfile = os.path.join(data_dir, f'x_rank{rank}-{local_rank}.z')
        head, _ = os.path.split(xfile)
        io.check_else_make_dir(head)
        joblib.dump(x, xfile)

    # pylint:disable=too-many-arguments
    def save_and_flush(
            self, data_dir=None, log_file=None, rank=0, mode='a', **kwargs
    ):
        """Call `self.save_data` and `self.flush_data_strs`."""
        if data_dir is None:
            data_dir = self.dirs.get('data_dir', None)
        if log_file is None:
            log_file = self.dirs.get('log_file', None)

        if log_file is None:
            log_file = Path(data_dir).joinpath('output.log')

        self.save_data(data_dir, **kwargs)
        self.flush_data_strs(log_file, rank=rank, mode=mode)
