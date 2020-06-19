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

from config import (NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC, NetWeights,
                    NP_FLOAT, PI, PROJECT_DIR, TF_FLOAT)

sns.set_palette('bright')


def therm_arr(arr, therm_frac=0.2, ret_steps=True):
    """Drop first `therm_frac` steps of `arr` to account for thermalization."""
    #  step_axis = np.argmax(arr.shape)
    step_axis = 0
    num_steps = arr.shape[step_axis]
    therm_steps = int(therm_frac * num_steps)
    arr = np.delete(arr, np.s_[:therm_steps], axis=step_axis)
    steps = np.arange(therm_steps, num_steps)

    if ret_steps:
        return arr, steps

    return arr


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
    ax.yaxis.set_label_coords(-0.03, 0.5)
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


def get_title_str_from_params(params):
    """Create a formatted string with relevant params from `params`."""
    eps = params.get('eps', None)
    net_weights = params.get('net_weights', None)
    num_steps = params.get('num_steps', None)
    lattice_shape = params.get('lattice_shape', None)

    title_str = (r"$N_{\mathrm{LF}} = $" + f'{num_steps}, '
                 r"$\varepsilon = $" + f'{eps:.4g}, ')

    if 'beta_init' in params and 'beta_final' in params:
        beta_init = params.get('beta_init', None)
        beta_final = params.get('beta_final', None)
        #  title_str = r"$\beta_{mathrm{init}} = $" + f'{beta_init}'
        title_str += (r"$\beta: $" + f'{beta_init:.3g}'
                      + r"$\rightarrow$" f'{beta_final:.3g}, ')
    elif 'beta' in params:
        beta = params.get('beta', None)
        title_str += r"$\beta = $" + f'{beta:.3g}, '

    title_str += f'shape: {tuple(lattice_shape)}'

    if net_weights == NET_WEIGHTS_HMC:
        title_str += f', (HMC)'

    return title_str


# pylint:disable=unsubscriptable-object
def plot_data(outputs, base_dir, FLAGS, thermalize=False, params=None):
    out_dir = os.path.join(base_dir, 'plots')
    io.check_else_make_dir(out_dir)

    title_str = None if params is None else get_title_str_from_params(params)

    data = {}
    for key, val in outputs.items():
        if key == 'x':
            continue

        if key == 'betas':
            if 'training' in base_dir:
                steps = FLAGS.logging_steps * np.arange(len(np.array(val)))
            else:
                steps = np.arange(len(np.array(val)))
            fig, ax = plt.subplots()
            ax.plot(steps, np.array(val))
            ax.set_xlabel('MC Step', fontsize='large')
            ax.yaxis.set_label_coords(-0.02, 0.5)
            ax.set_ylabel(r"$\beta$", fontsize='large', rotation='horizontal')
            if title_str is not None:
                ax.set_title(title_str, fontsize='x-large')
            out_file = os.path.join(out_dir, f'betas.png')
            io.log(f'Saving figure to: {out_file}.')
            fig.savefig(out_file, dpi=400, bbox_inches='tight')

        if key == 'loss_arr':
            if 'training' in base_dir:
                steps = FLAGS.logging_steps * np.arange(len(np.array(val)))
            else:
                steps = np.arange(len(np.array(val)))
            fig, ax = plt.subplots()
            ax.plot(steps, np.array(val), ls='',
                    marker='x', label='loss')
            ax.legend(loc='best')
            ax.set_xlabel('Train step')
            if title_str is not None:
                ax.set_title(title_str, fontsize='x-large')
            out_file = os.path.join(out_dir, 'loss.png')
            io.log(f'Saving figure to: {out_file}')
            fig.savefig(out_file, dpi=400, bbox_inches='tight')

        else:
            if key in ['beta', 'betas', 'loss_arr']:
                continue
            fig, ax = plt.subplots()
            arr = np.array(val)
            chains = np.arange(arr.shape[1])
            if 'training' in base_dir:
                steps = FLAGS.logging_steps * np.arange(arr.shape[0])
            else:
                steps = np.arange(arr.shape[0])
            if thermalize:
                arr, steps = therm_arr(arr, therm_frac=0.33)
                data[key] = (steps, arr)

            else:
                data[key] = (steps, arr)

            data_arr = xr.DataArray(arr.T, dims=['chain', 'draw'],
                                    coords=[chains, steps])
            az.plot_trace({key: data_arr})
            if title_str is not None:
                fig = plt.gcf()
                fig.suptitle(title_str, fontsize='x-large', y=1.04)

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
        if title_str is not None:
            ax.set_title(title_str, fontsize='x-large')
        #  ax.set_xlabel('Train step')
        out_file = os.path.join(out_dir, f'{key}_avg.png')
        io.log(f'Saving figure to: {out_file}.')
        fig.savefig(out_file, dpi=400, bbox_inches='tight')
        plt.close('all')

    steps, charges = data['charges_arr']
    plot_charges(steps, charges, out_dir=out_dir, title_str=title_str)
    plt.close('all')
