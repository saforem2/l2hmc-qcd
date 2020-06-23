"""
data_containers.py

Implements `TrainData` class, for working with training data.
"""
from __future__ import absolute_import, division, print_function

import os
import numpy as np
import utils.file_io as io


from utils.attr_dict import AttrDict


class DataContainer:
    """Base class for dealing with data."""

    def __init__(self, steps, header=None):
        self.steps = steps
        self.data_strs = [header]
        self.data = AttrDict({})

    def update(self, data):
        """Update `self.data` with new values from `data`."""
        for key, val in data.items():
            try:
                self.data[key].append(val)
            except KeyError:
                self.data[key] = [val]

    def get_fstr(self, data, rank=0):
        """Get formatted data string."""
        raise NotImplementedError

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


class TrainData(DataContainer):
    """Implements a container for dealing with training data."""

    def get_fstr(self, data, rank=0):
        """Get formatted data string from `data`."""
        if rank != 0:
            return ''

        fstr = (
            f"{data.steps: > 6g}/{self.steps:<6g} "
            f"{data.dt:^11.4g} "
            f"{data.loss:^11.4g} "
            f"{np.mean(data.px):^11.4g} "
            f"{data.eps:^11.4g} "
            f"{data.betas:^11.4g} "
            f"{np.mean(data.sumlogdet):^11.4g} "
            f"{np.mean(data.dq):^11.4g} "
            f"{np.mean(data.plaqs):^11.4g} "
        )

        self.data_strs.append(fstr)
        return fstr


class RunData(DataContainer):
    """Implements a container for dealing with inference data."""

    def get_fstr(self, data, rank=0):
        if rank != 0:
            return ''

        fstr = (
            f"{data.steps}/{self.steps:<6g} "
            f"{data.dt:^11.4g} "
            f"{np.mean(data.px):^11.4g} "
            f"{np.mean(data.sumlogdet):^11.4g} "
            f"{np.mean(data.dq):^11.4g} "
            f"{np.mean(data.plaqs):^11.4g} "
        )

        self.data_strs.append(fstr)
        return fstr
