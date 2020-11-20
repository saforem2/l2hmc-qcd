"""
data_containers.py

Implements `TrainData` class, for working with training data.
"""
from __future__ import absolute_import, division, print_function

import os

from collections import defaultdict

import joblib
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import utils.file_io as io

from config import BASE_DIR
from utils.attr_dict import AttrDict
from utils.data_utils import therm_arr
from utils.plotting_utils import (set_size, make_ridgeplots, mcmc_lineplot,
                                  mcmc_traceplot, get_title_str_from_params,
                                  plot_data)


plt.style.use('default')
sns.set_context('paper')
sns.set_style('whitegrid')
sns.set_palette('bright')


class DataContainer:
    """Base class for dealing with data."""

    def __init__(self, steps, header=None, dirs=None, print_steps=100):
        self.steps = steps
        self.print_steps = print_steps
        if dirs is not None:
            dirs = AttrDict(**dirs)
        self.dirs = dirs
        self.data_strs = [header]
        self.steps_arr = []
        self.data = AttrDict(defaultdict(list))
        if dirs is not None:
            io.check_else_make_dir(
                [v for k, v in dirs.items() if 'file' not in k]
            )

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
        io.log(f'Saving dataset to: {out_file}.')
        dataset.to_netcdf(os.path.join(out_dir, 'dataset.nc'))

    def update(self, step, metrics):
        """Update `self.data` with new values from `data`."""
        self.steps_arr.append(step)
        for key, val in metrics.items():
            try:
                self.data[key].append(tf.convert_to_tensor(val).numpy())
            except KeyError:
                self.data[key] = [tf.convert_to_tensor(val).numpy()]

    # pylint:disable=too-many-arguments
    def get_header(self, metrics=None, prepend=None,
                   append=None, skip=None, split=False):
        """Get nicely formatted header of variable names for printing."""
        if metrics is None:
            metrics = self.data

        header = io.make_header_from_dict(metrics, skip=skip, split=split,
                                          prepend=prepend, append=append)

        if self.data_strs[0] != header:
            self.data_strs.insert(0, header)

        return header

    def get_fstr(self, step, metrics, skip=None):
        """Get formatted data string from `data`."""
        skip = [] if skip is None else skip

        data = {
            k: tf.reduce_mean(v) for k, v in metrics.items() if k not in skip
        }

        fstr = (
            f'{step:>5g}/{self.steps:<5g} '
            + ''.join([f'{v:^12.4g}' for _, v in data.items()])
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
            io.log(f'Restored `x` from: {x_file}.', should_print=True)
        except FileNotFoundError:
            io.log(f'Unable to load `x` from {x_file}.', level='WARNING')
            io.log('Using random normal init.', level='WARNING')
            x = tf.random.normal(x_shape)

        data = self.load_data(data_dir)
        for key, val in data.items():
            self.data[key] = np.array(val).tolist()

        return x

    @staticmethod
    def load_data(data_dir):
        """Load data from `data_dir` and populate `self.data`."""
        contents = os.listdir(data_dir)
        fnames = [i for i in contents if i.endswith('.z')]
        keys = [i.rstrip('.z') for i in fnames]
        data_files = [os.path.join(data_dir, i) for i in fnames]
        data = {}
        for key, val in zip(keys, data_files):
            if 'x_rank' in key:
                continue
            io.log(f'Restored {key} from {val}.')
            data[key] = io.loadz(val)

        return AttrDict(data)

    def save_data(self, data_dir, rank=0, save_dataset=False):
        """Save `self.data` entries to individual files in `output_dir`."""
        if rank != 0:
            return

        io.check_else_make_dir(data_dir)
        for key, val in self.data.items():
            out_file = os.path.join(data_dir, f'{key}.z')
            io.savez(np.array(val), out_file)

        if save_dataset:
            self.save_dataset(data_dir)

    def plot_data(self, out_dir=None, therm_frac=0.):
        """Create trace plot + histogram for each entry in self.data."""
        dataset = self.get_dataset(therm_frac)
        for key, val in dataset.data_vars.items():
            if np.std(val.values.flatten()) < 1e-2:
                continue
            fig, ax = plt.subplots(constrained_layout=True,
                                   figsize=set_size())
            _ = val.plot(ax=ax)
            #  _ = sns.kdeplot(val.values.flatten(), ax=axes[1], shade=True)
            #  _ = axes[1].set_ylabel('')
            #  _ = fig.suptitle(key)
            if out_dir is not None:
                io.check_else_make_dir(out_dir)
                out_file = os.path.join(out_dir, f'{key}_xrPlot.png')
                fig.savefig(out_file, dpi=400, bbox_inches='tight')

        if out_dir is not None:
            out_dir = os.path.join(out_dir, 'ridgeplots')
            io.check_else_make_dir(out_dir)

        make_ridgeplots(dataset, out_dir)


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
        for key, val in self.data.items():
            tensor = tf.convert_to_tensor(val)
            arr, steps = therm_arr(tensor.numpy(), therm_frac=0.2)
            if 'steps' not in avg_data:
                avg_data['steps'] = len(steps)
            avg_data[key] = np.mean(arr)
            #  avg_data[key] = tf.reduce_mean(arr)

        avg_df = pd.DataFrame(avg_data, index=[0])
        csv_file = os.path.join(BASE_DIR, 'logs', 'GaugeModel_logs',
                                'inference_results.csv')
        io.log(f'Appending inference results to {csv_file}.')
        if not os.path.isfile(csv_file):
            avg_df.to_csv(csv_file, header=True, index=False, mode='w')
        else:
            avg_df.to_csv(csv_file, header=False, index=False, mode='a')

    @staticmethod
    def dump_configs(x, data_dir, rank=0, local_rank=0):
        """Save configs `x` separately for each rank."""
        xfile = os.path.join(data_dir, f'x_rank{rank}-{local_rank}.z')
        io.log('Saving configs from rank '
               f'{rank}-{local_rank} to: {xfile}.')
        head, _ = os.path.split(xfile)
        io.check_else_make_dir(head)
        joblib.dump(x, xfile)

    # pylint:disable=too-many-arguments
    def save_and_flush(
            self, data_dir=None, log_file=None, rank=0, mode='a'
    ):
        """Call `self.save_data` and `self.flush_data_strs`."""
        if data_dir is None:
            data_dir = self.dirs.get('data_dir', None)
        if log_file is None:
            log_file = self.dirs.get('log_file', None)

        self.save_data(data_dir, rank=rank, save_dataset=False)
        self.flush_data_strs(log_file, rank=rank, mode=mode)
