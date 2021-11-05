"""
step_timer.py

Implements `TrainData` class, for working with training data.
"""
from __future__ import absolute_import, division, print_function, annotations

import os
from pathlib import Path
import time

from typing import Union

import joblib
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py

from utils import SKEYS
import utils.file_io as io

from config import BASE_DIR
from utils.attr_dict import AttrDict
from utils.data_utils import therm_arr
from utils.hvd_init import RANK
from utils.plotting_utils import (set_size, make_ridgeplots,
                                  # get_title_str_from_params,
                                  plot_data)

from utils.logger import logger  # , in_notebook

# logger = Logger()
LW = plt.rcParams['axes.linewidth']

plt.style.use('default')
sns.set_context('paper')
sns.set_style('whitegrid')
sns.set_palette('bright')


VERBOSE = os.environ.get('VERBOSE', False)

import torch
import tensorflow as tf


def invert_list_of_dicts(input: list[dict]) -> dict[str, np.ndarray]:
    output = {
        k: np.zeros((len(input), *np.array(v).shape))
        for k, v in input[0].items()
    }
    for idx, x in enumerate(input):
        for k, v in x.items():
            output[k][idx] = np.array(v)

    return output



class DataContainer:
    """Base class for dealing with data."""
    def __init__(
            self,
            steps: int = None,
            # header: str = None,
            dirs: dict[str, Union[str, Path]] = None,
            print_steps: int = 100,
    ):
        self.steps = steps
        self.data_strs = []
        self.steps_arr = []
        self.print_steps = print_steps
        self.data = {}
        # self.data = AttrDict(defaultdict(list))

        if dirs is not None:
            names = [v for k, v in dirs.items() if 'file' not in k]
            io.check_else_make_dir(names)
        else:
            dirs = {}

        self.dirs = {k: Path(v).resolve() for k, v in dirs.items()}

    def update_dirs(self, dirs: dict):
        """Update `self.dirs` with key, val pairs from `dirs`."""
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

    def to_DataArray(self, x: list | np.ndarray) -> xr.DataArray:
        arr = np.array(x)
        steps = np.arange(len(arr))
        if len(arr.shape) == 1:
            # [draws]
            dims = ['draw']
            coords = [steps]
            return xr.DataArray(arr, dims=dims, coords=coords)

        if len(arr.shape) == 2:
            # [draws, chains]
            nchains = arr.shape[1]
            # nchains, nsteps = arr.shape
            dims = ['chain', 'draw']
            coords = [np.arange(nchains), steps]
            return xr.DataArray(arr.T, dims=dims, coords=coords)

        if len(arr.shape) == 3:
            # [draws, lf, chains] --> [chains, lf, draws]
            arr = arr.T
            nchains, nlf, nsteps = arr.shape
            dims = ['chain', 'leapfrog', 'draw']
            coords = [np.arange(nchains), np.arange(nlf), np.arange(nsteps)]
            return xr.DataArray(arr.T, dims=dims, coords=coords)

        raise ValueError('Invalid shape encountered')

    def _get_dataset(self, data: dict = None, skip: list[str] = None):
        """Create `xr.Dataset` from `self.data`."""
        data = self.data if data is None else data
        data_vars = {}
        skip = [] if skip is None else skip
        for key, val in data.items():
            if key in skip:
                continue

            if isinstance(val[0], (dict, AttrDict)):
                tmp = invert_list_of_dicts(val)
                for k, v in tmp.items():
                    data_vars[f'{key}/{k}'] = self.to_DataArray(v)
            else:
                data_vars[key] = self.to_DataArray(val)

        return xr.Dataset(data_vars)

    def get_dataset(self, therm_frac=0., make_plots=False) -> (xr.Dataset):
        data_vars = {}
        for key, val in self.data.items():
            # ---------------------------------------------------------------
            # TODO:
            # Currently, self.data['forward'] is aggregated as a list of
            # dicts where self.data['forward'][0] = {
            #    'plaqs': TensorShape([nleapfrog, nchains])
            # }
            # so self.data['forward'] = [
            #     {'sumlogdet': [...], 'accept_prob': [...], ...},
            #     {'sumlogdet': [...], 'accept_prob': [...], ...},
            #     {'sumlogdet': [...], 'accept_prob': [...], ...},
            # ]
            # what we want is self.data['forward'] to be a dict of lists
            # so self.data['forward'] = {
            #     'sumlogdet': [[...], [...], ...],
            #     'accept_prob': [[...], [...], ...],
            #     ...,
            # }
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
        io.check_else_make_dir(out_dir)
        outfile = Path(out_dir).joinpath('dataset.nc')
        mode = 'a' if outfile.is_file() else 'w'
        logger.debug(f'Saving dataset to: {outfile}.')
        dataset.to_netcdf(str(outfile), mode=mode)

    def _to_h5pyfile(
            self,
            key: str,
            arr: np.ndarray,
            hfile: Union[str, Path, h5py.File],
    ):
        if isinstance(hfile, (str, Path)):
            hfile = h5py.File(str(hfile), 'a')

        if key in list(hfile.keys()):
            hfile[key].resize((f[key].shape[0] + arr.shape[0]), axis=0)
            hfile[key][-arr.shape[0]:] = arr
        else:
            maxshape = (None,)
            if len(arr.shape) > 1:
                maxshape = (None, *arr.shape[1:])
            hfile.create_dataset(key, data=arr, maxshape=maxshape)


    def to_h5pyfile(
            self,
            hfile: Union[str, Path],
            skip_keys: list[str] = None
    ):
        f = h5py.File(hfile, 'a')
        logger.debug(f'Saving self.data to {hfile}')
        skip_keys = [] if skip_keys is None else skip_keys
        for key, val in self.data.items():
            if key in skip_keys:
                continue

            arr = np.array(val)
            if len(arr) == 0:
                continue

            # If arr is an array of dicts, convert to dict of lists
            if isinstance(arr[0], (dict, AttrDict)):
                dlist = invert_list_of_dicts(arr)
                for k, v in dlist.items():
                    self._to_h5pyfile(key=f'{key}/{k}',
                                      arr=np.array(v),
                                      hfile=f)
            else:
                self._to_h5pyfile(key=key, arr=arr, hfile=f)

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

    def update(self, step: int, metrics: dict):
        """Update `self.data` with new values from `data`."""
        self.steps_arr.append(step)
        for key, val in metrics.items():
            if isinstance(val, np.ndarray):
                self.data[key].append(val)
            else:
                if isinstance(val, torch.Tensor):
                    val.detach().numpy()

                if isinstance(val, tf.Tensor):
                    val = val.numpy()

                try:
                    self.data[key].append(val)
                except KeyError:
                    self.data[key] = [val]

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

    def save_data(
            self,
            data_dir: Union[str, Path],
            skip_keys: list[str] = None,
            use_hdf5: bool = False,
            save_dataset: bool = True,
            # compression: str = 'gzip',
    ):
        """Save `self.data` entries to individual files in `output_dir`."""
        success = 0
        data_dir = Path(data_dir)
        io.check_else_make_dir(str(data_dir))
        if use_hdf5:
            # ------------ Save using h5py -------------------
            hfile = data_dir.joinpath(f'data_rank{RANK}.hdf5')
            self.to_h5pyfile(hfile, skip_keys=skip_keys)
            success = 1
        if save_dataset:
            # ----- Save to `netCDF4` file using hierarchical grouping ------
            self.save_dataset(data_dir)
            success = 1
        if success == 0:
            # -------------------------------------------------
            # Save each `{key: val}` entry in `self.data` as a 
            # compressed file, named `f'{key}.z'` .
            # -------------------------------------------------
            skip_keys = [] if skip_keys is None else skip_keys
            logger.info(f'Saving individual entries to {data_dir}.')
            for key, val in self.data.items():
                if key in skip_keys:
                    continue
                out_file = os.path.join(data_dir, f'{key}.z')
                io.savez(np.array(val), out_file)


    def plot_dataset(
            self,
            out_dir: str = None,
            therm_frac: float = 0.,
            num_chains: int = None,
            ridgeplots: bool = True,
            profile: bool = False,
            skip: Union[list[str], str] = None,
            cmap: str = 'viridis_r',
    ):
        """Create trace plot + histogram for each entry in self.data."""
        tdict = {}
        dataset = self.get_dataset(therm_frac)
        for key, val in dataset.data_vars.items():
            if skip is not None:
                if key in skip:
                    continue

            t0 = time.time()
            try:
                std = np.std(val.values.flatten())
            except TypeError:
                continue
            # if np.std(val.values.flatten()) < 1e-2:
            if std < 1e-2:
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
            # try:
            #     arr = val.numpy()
            # except AttributeError:
            #     try:
            #         arr = np.array(val)
            #     except ValueError:
            #         arr = val
            try:
                avg_data[key] = np.mean(val)
                # avg_data[key] = val.mean()
            except AttributeError:
                raise AttributeError(f'Unable to call `val.mean()` on {val}')

            # if 'steps' not in avg_data:
            #     avg_data['steps'] = len(steps)

            # avg_data[key] = np.mean(arr)

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
            self,
            data_dir: Union[str, Path] =None,
            log_file: Union[str, Path] = None,
            rank: int = 0,
            mode: str = 'a',
            skip_keys: list = None,
            save_dataset: bool = False,
            use_hdf5: bool = False,
    ):
        """Call `self.save_data` and `self.flush_data_strs`."""
        if data_dir is None:
            data_dir = self.dirs.get('data_dir', None)
        if log_file is None:
            log_file = self.dirs.get('log_file', None)

        if log_file is None:
            log_file = Path(data_dir).joinpath('output.log')

        self.save_data(data_dir, skip_keys,
                       use_hdf5=use_hdf5,
                       save_dataset=save_dataset)
        self.flush_data_strs(log_file, rank=rank, mode=mode)
