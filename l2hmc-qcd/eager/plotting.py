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


def plot_charges(steps, charges, title_str=None, out_dir=None):
    charges = charges.T
    if charges.shape[0] > 4:
        charges = charges[:4, :]
    fig, ax = plt.subplots()
    for idx, q in enumerate(charges):
        ax.plot(steps, np.around(q) + 5 * idx, marker='', ls='-')
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.xmargin: 0
    ax.yaxis.set_label_coords(-0.01, 1.02)
    ax.set_ylabel(r"$\mathcal{Q}$", fontsize='x-large',
                  rotation='horizontal')
    ax.set_xlabel('MC Step', fontsize='x-large')
    if title_str is not None:
        ax.set_title(title_str, fontsize='x-large')
    plt.tight_layout()
    if out_dir is not None:
        out_file = os.path.join(out_dir, 'charges_traceplot.png')
        io.log(f'Saving figure to: {out_file}.')
        plt.savefig(out_file, dpi=400, bbox_inches='tight')


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
            plt.close('all')

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    for idx, (key, val) in enumerate(data.items()):
        steps, arr = val
        avg_val = np.mean(arr, axis=1)
        label = r"$\langle$" + f' {key} ' + r"$\rangle$"
        fig, ax = plt.subplots()
        ax.plot(steps, avg_val, color=colors[idx], label=label)
        ax.legend(loc='best')
        ax.set_xlabel('Step')
        #  ax.set_xlabel('Train step')
        out_file = os.path.join(out_dir, f'{key}_avg.png')
        io.log(f'Saving figure to: {out_file}.')
        fig.savefig(out_file, dpi=400, bbox_inches='tight')
        plt.close('all')

    steps, charges = data['charges_arr']
    plot_charges(steps, charges, out_dir=out_dir)
    plt.close('all')
