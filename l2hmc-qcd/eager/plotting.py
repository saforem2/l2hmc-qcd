"""
plotting.py

Methods for plotting data.
"""
import os

import arviz as az
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

import utils.file_io as io

from config import (NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC, NetWeights, NP_FLOAT,
                    PI, PROJECT_DIR, TF_FLOAT)
from plotters.data_utils import therm_arr


def plot_data(outputs, base_dir, FLAGS, thermalize=False):
    out_dir = os.path.join(base_dir, 'plots')
    io.check_else_make_dir(out_dir)

    data = {}
    for key, val in outputs.items():
        if key == 'x':
            continue
        if key == 'loss_arr':
            fig, ax = plt.subplots()
            steps = FLAGS.logging_steps * np.arange(len(np.array(val)))
            ax.plot(steps, np.array(val), ls='', marker='x', label='loss')
            ax.legend(loc='best')
            ax.set_xlabel('Train step')
            out_file = os.path.join(out_dir, 'loss.png')
            io.log(f'Saving figure to: {out_file}')
            fig.savefig(out_file, dpi=400, bbox_inches='tight')
        else:
            fig, ax = plt.subplots()
            arr = np.array(val)
            chains = np.arange(arr.shape[1])
            steps = FLAGS.logging_steps * np.arange(arr.shape[0])
            if thermalize:
                arr, steps = therm_arr(arr, therm_frac=0.33)
                data[key] = (steps, arr)

            else:
                data[key] = (steps, arr)

            data_arr = xr.DataArray(arr.T, dims=['chain', 'draw'],
                                    coords=[chains, steps])
            az.plot_trace({key: data_arr})
            out_file = os.path.join(out_dir, f'{key}.png')
            io.log(f'Saving figure to: {out_file}.')
            plt.savefig(out_file, dpi=400, bbox_inches='tight')

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    for idx, (key, val) in enumerate(data.items()):
        steps, arr = val
        avg_val = np.mean(arr, axis=1)
        label = r"$\langle$" + f' {key} ' + r"$\rangle$"
        fig, ax = plt.subplots()
        ax.plot(steps, avg_val, color=colors[idx], label=label)
        ax.legend(loc='best')
        ax.set_xlabel('Train step')
        out_file = os.path.join(out_dir, f'{key}_avg.png')
        io.log(f'Saving figure to: {out_file}.')
        fig.savefig(out_file, dpi=400, bbox_inches='tight')


