"""
train_plots.py

Contains helper method for plotting training data.

Author: Sam Foreman
Date: 04/20/2020
"""
import os

import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az

import utils.file_io as io

from .data_utils import therm_arr
from .inference_plots import traceplot_posterior
from lattice.lattice import u1_plaq_exact

sns.set_palette('bright')

MARKERS = 10 * ['.', 'x', '+', '^', 'v', '<', '>', 'D']
COLORS = 10 * ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
sns.set_palette('bright')


def get_title_str(params):
    """Parse `params` to create title string for plots."""
    num_steps = params.get('num_steps', None)
    batch_size = params.get('batch_size', None)
    train_steps = params.get('train_steps', None)
    time_size = params.get('time_size', None)
    space_size = params.get('space_size', None)
    plaq_weight = params.get('plaq_weight', 0.)
    charge_weight = params.get('charge_weight', 0.)
    std_weight = params.get('std_weight', 0.)

    title_str = (f'{time_size}' + r"$\times$" + f'{space_size}, '
                 + r"$N_{\mathrm{B}} = $" + f'{batch_size}, '
                 + r"$N_{\mathrm{LF}} = $" + f'{num_steps}, '
                 + r"$N_{\mathrm{train}} = $" + f'{train_steps}, ')

    if std_weight > 0:
        title_str += r"$\alpha_{\mathrm{STD}} = $" + f'{std_weight:.2g}, '
    if plaq_weight > 0:
        title_str += r"$\alpha_{\mathrm{plaq}} = $" + f'{plaq_weight:.2g}, '
    if charge_weight > 0:
        title_str += r"$\alpha_{\mathrm{Q}} = $" + f'{charge_weight:.2g}, '

    return title_str


def build_dataset(data, filter_str=None, steps=None):
    """Build `xarray.Dataset` from `data`."""
    _dict = {}
    for key, val in data.items():
        cond1 = (filter_str is not None and filter_str in key)
        cond2 = (val == [])
        if cond1 or cond2:
            continue

        if steps is None:
            arr, steps = therm_arr(np.array(val))
        else:
            arr = np.array(val)

        arr = arr.T
        _dict[key] = xr.DataArray(arr, dims=['chain', 'draw'],
                                  coords=[np.arange(arr.shape[0]), steps])
    dataset = xr.Dataset(_dict)

    return dataset


def charges_trace_plot(charges, out_dir, title_str, therm_frac=0.2):
    """Create separate trace plot of topological charges."""
    arr, steps = therm_arr(np.array(charges), therm_frac=therm_frac)
    arr = arr.T
    _dict = {
        'charges': xr.DataArray(arr, dims=['chain', 'draw'],
                                coords=[np.arange(arr.shape[0]), steps]),
    }
    dataset = xr.Dataset(_dict)
    traceplot_posterior(dataset,
                        name='charges',
                        fname='train',
                        fig_dir=out_dir,
                        filter_str=None,
                        title_str=title_str)
    return dataset


def plot_train_data(train_data, params):
    """Plot all training data and save figures to `log_dir/train_plots`.

    Args:
        train_data (dict): Dictionary of training data.
        params (dict): Dictionary of parameters used for training.
    """
    title_str = get_title_str(params)
    log_dir = params.get('log_dir', None)
    out_dir = os.path.join(log_dir, 'train_plots')
    io.check_else_make_dir(out_dir)

    data = {}
    for idx, (key, val) in enumerate(train_data.items()):
        if key == 'train_op':
            continue

        label = key

        y, t = therm_arr(np.array(val), therm_frac=0.1)
        x = np.array(train_data['beta'])[int(t[0]):]

        if len(np.unique(x)) == 1:
            x = t
            #  x = np.arange(len(x))

        kwargs = {
            'ls': '',
            'color': COLORS[idx],
            'marker': MARKERS[idx],
        }
        fig, ax = plt.subplots()

        if len(y.shape) == 1:
            ax.plot(x, y, label=label, **kwargs)

        if len(y.shape) == 2:
            label = r'$\langle$' + f'{key}' + r'$\rangle$'

            data[key] = y

            ax.plot(x, y.mean(axis=-1),
                    label=label, **kwargs)

        if len(y.shape) == 3:
            label = r'$\langle$' + f'{key}' + r'$\rangle$'

            z = y.mean(axis=(-1))
            data[key] = z

            try:
                ax.plot(x, z.mean(axis=-1),
                        label=label, **kwargs)
            except:
                import pudb; pudb.set_trace()


        if key == 'plaqs':
            beta_arr = np.array(train_data['beta'])[int(t[0]):]
            ax.plot(beta_arr, u1_plaq_exact(beta_arr),
                    color='k', marker='', ls='-', label='exact')

        ax.legend(loc='best')
        ax.set_xlabel(r"$\beta$", fontsize='large')
        ax.set_title(title_str, fontsize='x-large')
        plt.tight_layout()
        out_file = os.path.join(out_dir, f'{key}.png')
        io.log(f'Saving figure to: {out_file}.')
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close('all')

    obs_strs = ['actions', 'plaqs', 'charges', 'dq']
    x_strs = ['x_in', 'x_out', 'dx', 'dx_proposed']

    #  observables = {
    #      'data['
    dataset = build_dataset(data, steps=t)
    traceplot_posterior(dataset=dataset,
                        name='data',
                        fname='train',
                        fig_dir=out_dir,
                        filter_str=None,
                        title_str=title_str)
    plt.close('all')

    charges_trace_plot(np.array(data['charges']),
                       out_dir=out_dir,
                       title_str=title_str)

    return dataset
