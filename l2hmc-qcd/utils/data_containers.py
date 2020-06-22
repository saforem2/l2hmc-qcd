"""
data_containers.py

Implements `TrainData` class, for working with training data.
"""
from __future__ import absolute_import, division, print_function

import os
import numpy as np
import utils.file_io as io
from config import NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC, PI, TF_FLOAT, TF_INT


from utils.attr_dict import AttrDict


class DataContainer:
    """Base class for dealing with data."""

    def __init__(self, steps, data=None, header=None):
        self.steps = steps
        self.data_strs = [header]
        self.data = data if data is not None else AttrDict({})

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

    def save_data(self, output_dir):
        """Save `self.data` entries to individual files in `output_dir`."""
        io.check_else_make_dir(output_dir)
        for key, val in self.data.items():
            out_file = os.path.join(output_dir, f'{key}.z')
            io.savez(np.array(val), out_file, key)


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
