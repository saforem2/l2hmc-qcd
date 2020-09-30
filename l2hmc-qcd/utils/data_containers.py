"""
data_containers.py

Implements `TrainData` class, for working with training data.
"""
from __future__ import absolute_import, division, print_function

import os

from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

import utils.file_io as io

from config import BASE_DIR
from utils.attr_dict import AttrDict
from utils.data_utils import therm_arr



class DataContainer:
    """Base class for dealing with data."""

    def __init__(self, steps, header=None, dirs=None):
        self.steps = steps

        self.dirs = dirs
        self.data_strs = [header]
        self.steps_arr = []
        self.data = AttrDict(defaultdict(list))
        if dirs is not None:
            io.check_else_make_dir(
                [v for k, v in dirs.items() if 'file' not in k]
            )

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

        header = io.make_header_from_dict(metrics,
                                          skip=skip,
                                          prepend=prepend,
                                          append=append,
                                          split=split)

        if self.data_strs[0] != header:
            self.data_strs.insert(0, header)

        return header

    def get_fstr(self, step, metrics, skip=None):
        """Get formatted data string from `data`."""
        skip = [] if skip is None else skip

        data = {
            k: tf.reduce_mean(v) for k, v in metrics.items() if k not in skip
        }
        fstr = (f'{step:>5g}/{self.steps:<5g} '
                + ''.join([f'{v:^12.4g}' for _, v in data.items()]))

        self.data_strs.append(fstr)

        return fstr

    def restore(self, data_dir, rank=0, step=None):
        """Restore `self.data` from `data_dir`."""
        if step is not None:
            self.steps += step

        x_file = os.path.join(data_dir, f'x_rank{rank}.z')
        try:
            x = io.loadz(x_file)
            io.log_tqdm(f'Restored `x` from: {x_file}.')
        except FileNotFoundError as err:
            io.log_tqdm(f'Unable to load `x` from {x_file}.')
            raise err

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
            io.log_tqdm(f'Restored {key} from {val}.')
            data[key] = io.loadz(val)

        return AttrDict(data)

    def save_data(self, data_dir, rank=0):
        """Save `self.data` entries to individual files in `output_dir`."""
        if rank != 0:
            return

        io.check_else_make_dir(data_dir)
        for key, val in self.data.items():
            out_file = os.path.join(data_dir, f'{key}.z')
            io.savez(np.array(val), out_file)

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
        io.log_tqdm(f'Appending inference results to {csv_file}.')
        if not os.path.isfile(csv_file):
            avg_df.to_csv(csv_file, header=True, index=False, mode='w')
        else:
            avg_df.to_csv(csv_file, header=False, index=False, mode='a')

    @staticmethod
    def dump_configs(x, data_dir, rank=0):
        """Save configs `x` separately for each rank."""
        xfile = os.path.join(data_dir, f'x_rank{rank}.z')
        io.log_tqdm(f'Saving configs from rank {rank} to: {xfile}.')
        head, _ = os.path.split(xfile)
        io.check_else_make_dir(head)
        joblib.dump(x, xfile)

    # pylint:disable=too-many-arguments
    def save_and_flush(self, data_dir=None, log_file=None, rank=0, mode='a'):
        """Call `self.save_data` and `self.flush_data_strs`."""
        if data_dir is None:
            data_dir = self.dirs.get('data_dir', None)
        if log_file is None:
            log_file = self.dirs.get('log_file', None)

        self.save_data(data_dir, rank=rank)
        self.flush_data_strs(log_file, rank=rank, mode=mode)
