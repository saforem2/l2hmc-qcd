"""
data_containers.py

Implements `TrainData` class, for working with training data.
"""
from __future__ import absolute_import, division, print_function

import os

from collections import defaultdict

import numpy as np
import tensorflow as tf

import utils.file_io as io

from utils.attr_dict import AttrDict


class DataContainer:
    """Base class for dealing with data."""

    def __init__(self, steps, header=None, skip_keys=None):
        self.steps = steps
        if skip_keys is None:
            skip_keys = ['']
        self.skip_keys = skip_keys
        self.data_strs = [header]
        self.steps_arr = []
        self.data = AttrDict(defaultdict(list))

    def update(self, step, metrics):
        """Update `self.data` with new values from `data`."""
        self.steps_arr.append(step)
        for key, val in metrics.items():
            try:
                self.data[key].append(tf.convert_to_tensor(val).numpy())
            except KeyError:
                self.data[key] = [tf.convert_to_tensor(val).numpy()]

    def get_fstr(self, step, metrics, rank=0):
        """Get formatted data string from `data`."""
        if rank != 0:
            return ''

        fstr = (
            f"{step:>6g}/{self.steps:<6g} "
        )

        fstr = f"{step:>6g}/{self.steps:<6g} "
        if self.skip_keys is not None:
            data = {
                k: tf.reduce_mean(v) for k, v in metrics.items()
                if k not in self.skip_keys
            }
        fstr += ' '.join([f'{v:^11.4g}' for _, v in data.items()])

        #  fstr = " ".join(f'{vals:^11.4g}')
        #
        #  fstr = " ".join(f'{tf.reduce_mean(x):^11.4g} ')
        #
        #  fstr = (
        #      f"{step:>6g}/{self.steps:<6g} "
        #      f"{metrics.dt:^11.4g} "
        #      f"{metrics.loss:^11.4g} "
        #      f"{metrics.ploss:^11.4g} "
        #      f"{metrics.qloss:^11.4g} "
        #      f"{np.mean(metrics.accept_prob):^11.4g} "
        #      f"{tf.reduce_mean(metrics.eps):^11.4} "
        #      f"{tf.reduce_mean(metrics.beta):^11.4g} "
        #      f"{tf.reduce_mean(metrics.sumlogdet):^11.4g} "
        #      f"{tf.reduce_mean(metrics.dq):^11.4g} "
        #      f"{tf.reduce_mean(metrics.plaqs):^11.4g} "
        #  )

        self.data_strs.append(fstr)
        return fstr

    def restore(self, data_dir):
        """Restore `self.data` from `data_dir`."""
        data = self.load_data(data_dir)
        for key, val in data.items():
            self.data[key] = [val]

    @staticmethod
    def load_data(data_dir):
        """Load data from `data_dir` and populate `self.data`."""
        contents = os.listdir(data_dir)
        fnames = [i for i in contents if i.endswith('.z')]
        keys = [i.rstrip('.z') for i in fnames]
        data_files = [os.path.join(data_dir, i) for i in fnames]
        data = {}
        for key, val in zip(keys, data_files):
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
