"""
inference_plots.py

Contains helper methods for plotting inference results.

Author: Sam Foreman (github: @saforem2)
Date: 01/15/2020
"""
import os

import arviz as az
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import utils.file_io as io

from lattice.lattice import u1_plaq_exact
from plotters.seaborn_plots import plot_setup

sns.set_palette('bright')

HEADER = 80 * '-'
SEPERATOR = 80 * '-'

mpl.rcParams['axes.formatter.limits'] = -4, 4

# pylint:disable=too-many-locals,too-many-arguments,invalid-name,
# pylint:disable=too-many-arguments,too-many-statements


def savefig(fig, out_file):
    io.log(HEADER)
    io.log(f'Saving figure to: {out_file}.')
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    io.log(HEADER)


def autocorr(x):
    """Return the autocorrelation of a signal."""
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]


def plot_autocorr(x, params, run_params, **kwargs):
    """Plot autocorrelation of `x`."""
    name = kwargs.get('name', '')
    run_str = run_params['run_str']
    log_dir = params['log_dir']
    figs_dir = os.path.join(log_dir, 'figures_np')
    fig_dir = os.path.join(figs_dir, run_str)
    io.check_else_make_dir(fig_dir)

    fname, title_str, _ = plot_setup(log_dir, run_params)
    fig, ax = plt.subplots()
    if len(x.shape) > 1:
        for idx, sample in enumerate(x):
            x_acl = autocorr(sample)
            if idx == 0:
                ax.plot(x_acl / float(x_acl.max()), '-', label=name)
            else:
                ax.plot(x_acl / float(x_acl.max()), '-')
    else:
        x_acl = autocorr(x)
        ax.plot(x_acl / float(x_acl.max()), '-')

    ax.legend(loc='best')
    ax.set_title(title_str)
    ax.set_ylabel('Autocorrelation')
    ax.set_xlabel('Step')
    out_file = os.path.join(fig_dir, f'{name}_autocorrelation.pdf')
    savefig(fig, out_file)

    return fig, ax


def therm_arr(arr, therm_frac=0.25):
    """Returns thermalized arr, by dropping the first therm_frac of data.%"""
    num_steps = arr.shape[0]
    therm_steps = int(therm_frac * num_steps)
    arr = arr[therm_steps:]
    steps = np.arange(therm_steps, num_steps)
    return arr, steps


def calc_tunneling_rate(charges):
    """Calculate the tunneling rate as the difference in charge b/t steps."""
    step_axis = np.argmax(charges.shape)
    num_steps = charges.shape[step_axis]

    charges = np.around(charges)
    charges = np.insert(charges, 0, 0, axis=0)
    dq = np.abs(charges[1:] - charges[-1])
    tunneling_rate = dq / num_steps
    return tunneling_rate


def build_energy_dataset(energy_data):
    """Build `xarray.Datset` containing `energy_data` for plotting."""
    ed_dict = {}
    for key, val in energy_data.items():
        arr, steps = therm_arr(np.array(val))
        arr = arr.T
        ed_dict[key] = xr.DataArray(arr, dims=['chain', 'draw'],
                                    coords=[np.arange(arr.shape[0]), steps])
    dataset = xr.Dataset(ed_dict)

    return dataset


def build_energy_diffs_dataset(energy_data):
    """Build xarray.Dataset containing energy difference data for plotting."""
    def _ediff(key1, key2):
        return np.array(energy_data[key1]) - np.array(energy_data[key2])

    denergy_data = {
        'dpotential_prop_init': _ediff('potential_proposed', 'potential_init'),
        'dpotential_out_init': _ediff('potential_out', 'potential_init'),
        'dpotential_out_prop': _ediff('potential_out', 'potential_proposed'),
        'dkinetic_prop_init': _ediff('kinetic_proposed', 'kinetic_init'),
        'dkinetic_out_init': _ediff('kinetic_out', 'kinetic_init'),
        'dkinetic_out_prop': _ediff('kinetic_out', 'kinetic_proposed'),
        'dhamiltonian_prop_init': _ediff('hamiltonian_proposed',
                                         'hamiltonian_init'),
        'dhamiltonian_out_init': _ediff('hamiltonian_out', 'hamiltonian_init'),
        'dhamiltonian_out_prop': _ediff('hamiltonian_out',
                                        'hamiltonian_proposed')
    }

    de_dict = {}
    for key, val in denergy_data.items():
        arr, steps = therm_arr(np.array(val))
        arr = arr.T
        de_dict[key] = xr.DataArray(arr, dims=['chain', 'draw'],
                                    coords=[np.arange(arr.shape[0]), steps])

    dataset = xr.Dataset(de_dict)

    return dataset


def build_energy_transition_dataset(energy_data):
    """Build xarray.Dataset containing `Ei - <Ei>` for plotting."""
    data_dict = {}
    for key, val in energy_data.items():
        arr, steps = therm_arr(np.array(val))
        arr -= np.mean(arr, axis=0)
        arr = arr.T
        key_ = f'{key}_minus_avg'
        data_dict[key_] = xr.DataArray(arr, dims=['chain', 'draw'],
                                       coords=[np.arange(arr.shape[0]), steps])

    dataset = xr.Dataset(data_dict)

    return dataset


def build_dataset(run_data, run_params):
    """Build dataset."""
    rd_dict = {}
    for key, val in run_data.items():
        if 'mask' in key:
            continue
        if 'forward' in key:
            continue

        if len(np.array(val).shape) < 2:
            continue

        arr, draws = therm_arr(np.array(val))
        arr = arr.T
        chains = np.arange(arr.shape[0])

        if 'plaqs' in key:
            key = 'plaqs_diffs'
            arr = u1_plaq_exact(run_params['beta']) - arr

        #  if 'charges' in key:
        #      arr = np.around(arr)

        rd_dict[key] = xr.DataArray(arr,
                                    dims=['chain', 'draw'],
                                    coords=[chains, draws])

    #  rd_dict['charges_squared'] = rd_dict['charges'] ** 2

    charges = rd_dict['charges'].values.T
    tunneling_rate = calc_tunneling_rate(charges).T
    rd_dict['tunneling_rate'] = xr.DataArray(tunneling_rate,
                                             dims=['chain', 'draw'],
                                             coords=[chains, draws])
    dataset = xr.Dataset(rd_dict)

    return dataset


def _check_existing(out_dir, fname):
    """Check if `fname` exists in `out_dir`. If so, append time to `fname`."""
    if os.path.isfile(os.path.join(out_dir, f'{fname}.pdf')):
        timestr = io.get_timestr()
        hour_str = timestr['hour_str']
        fname += f'_{hour_str}'

    return fname


def plot_reverse_data(reverse_data, params, run_params, **kwargs):
    """Plot reversibility results."""
    run_str = run_params['run_str']
    log_dir = params['log_dir']
    runs_np = kwargs.get('runs_np', True)
    if runs_np:
        figs_dir = os.path.join(log_dir, 'figures_np')
    else:
        figs_dir = os.path.join(log_dir, 'figures_tf')

    fig_dir = os.path.join(figs_dir, run_str)
    io.check_else_make_dir(fig_dir)
    out_dir = kwargs.get('out_dir', None)
    try:
        fname, title_str, _ = plot_setup(log_dir, run_params)
    except FileNotFoundError:
        return None, None

    fname = f'{fname}_reversibility_hist'
    out_file = os.path.join(fig_dir, f'{fname}.pdf')
    fig, ax = plt.subplots()
    for key, val in reverse_data.items():
        sns.kdeplot(np.array(val).flatten(), shade=True, label=key, ax=ax)
    ax.legend(loc='best')
    ax.set_title(title_str)
    plt.tight_layout()
    io.log(f'Saving figure to: {out_file}...')
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    if out_dir is not None:
        fout = os.path.join(out_dir, f'{fname}.pdf')
        io.log(f'Saving figure to: {fout}...')
        fig.savefig(fout, dpi=200, bbox_inches='tight')

    return fig, ax


def plot_trace(data, fname, title_str=None, filter_str=None):
    """Create traceplot of `data`.

    Args:
        data (xr.Dataset): Dataset object containing data to be plotted.
        fname (str): Where to save plot.
        filter_str (list): List of strings to filter variable names. Only those
            variables contained in this list will be included in the plot.

    Returns:
        None
    """
    var_names = None
    if filter_str is not None:
        if isinstance(filter_str, list):
            var_names = []
            for s in filter_str:
                var_names.extend(
                    [var for var in data.data_vars if s in var]
                )
        else:
            var_names = [
                var for var in data.data_vars if filter_str in var
            ]

    _ = az.plot_trace(data, var_names=var_names,
                      compact=False, combined=False)
    fig = plt.gcf()
    if title_str is not None:
        fig.suptitle(title_str, fontsize=24, y=1.05)
    savefig(fig, fname)


def plot_posterior(data, fname, title_str=None, filter_str=None):
    """Create distribution plot of posterior distribution. Includes stats.

    Args:
        data (xr.Dataset): Dataset object containing data to be plotted.
        fname (str): Where to save plot.
        filter_str (list): List of strings to filter variable names. Only those
            variables contained in this list will be included in the plot.

    Returns:
        None
    """
    var_names = None
    if filter_str is not None:
        var_names = []
        if isinstance(filter_str, list):
            var_names = []
            for s in filter_str:
                var_names.extend(
                    [var for var in data.data_vars if s in var]
                )
        else:
            var_names = [
                var for var in data.data_vars if filter_str in var
            ]
        var_names = list(set(var_names))
    _ = az.plot_posterior(data, var_names=var_names)
    fig = plt.gcf()
    if title_str is not None:
        fig.suptitle(title_str, fontsize=24, y=1.05)
    savefig(fig, fname)


def traceplot_posterior(dataset, name, fname, fig_dir,
                        title_str=None, filter_str=None):
    """Create traceplot of the posterior distribution.

    Args:
        data (xr.Dataset): Dataset object containing data to be plotted.
        fname (str): Where to save plot.
        filter_str (list): List of strings to filter variable names. Only those
            variables contained in this list will be included in the plot.

    Returns:
        None
    """
    tp_fname = _check_existing(fig_dir, f'{fname}_{name}_traceplot')
    pp_fname = _check_existing(fig_dir, f'{fname}_{name}_posterior')
    tp_fout = os.path.join(fig_dir, f'{tp_fname}.pdf')
    pp_fout = os.path.join(fig_dir, f'{pp_fname}.pdf')
    plot_trace(dataset, tp_fout,
               title_str=title_str,
               filter_str=filter_str)
    plot_posterior(dataset, pp_fout,
                   title_str=title_str,
                   filter_str=filter_str)


def inference_plots(data_dict, params, run_params, **kwargs):
    """Create trace plots of lattice observables and energy data."""
    run_data = data_dict.get('run_data', None)
    energy_data = data_dict.get('energy_data', None)
    reverse_data = data_dict.get('reverse_data', None)
    run_str = run_params['run_str']
    log_dir = params['log_dir']
    runs_np = kwargs.get('runs_np', True)
    if runs_np:
        figs_dir = os.path.join(log_dir, 'figures_np')
    else:
        figs_dir = os.path.join(log_dir, 'figures_tf')

    fig_dir = os.path.join(figs_dir, run_str)
    io.check_else_make_dir(fig_dir)
    out_dir = kwargs.get('out_dir', None)

    dataset = None
    energy_dataset = None
    try:
        fname, title_str, _ = plot_setup(log_dir, run_params)
    except FileNotFoundError:
        return dataset, energy_dataset

    ####################################################
    # Create traceplot + possterior plot of energy data
    ####################################################
    if energy_data is not None:
        energy_dataset = build_energy_dataset(energy_data)
        pe_dir = os.path.join(fig_dir, 'potential_plots')
        io.check_else_make_dir(pe_dir)
        traceplot_posterior(energy_dataset, name='potential',
                            fname=fname, fig_dir=pe_dir,
                            title_str=title_str,
                            filter_str='potential')
        ke_dir = os.path.join(fig_dir, 'kinetic_plots')
        io.check_else_make_dir(ke_dir)
        traceplot_posterior(energy_dataset, name='kinetic',
                            fname=fname, fig_dir=ke_dir,
                            filter_str='kinetic')
        h_dir = os.path.join(fig_dir, 'hamiltonian_plots')
        io.check_else_make_dir(h_dir)
        traceplot_posterior(energy_dataset, name='hamiltonian',
                            fname=fname, fig_dir=h_dir,
                            title_str=title_str,
                            filter_str='hamiltonian')

        denergy_dataset = build_energy_diffs_dataset(energy_data)
        traceplot_posterior(denergy_dataset, name='potential_diffs',
                            fname=fname, fig_dir=pe_dir, title_str=title_str,
                            filter_str='potential')
        traceplot_posterior(denergy_dataset, name='kinetic_diffs',
                            fname=fname, fig_dir=ke_dir, title_str=title_str,
                            filter_str='kinetic')
        traceplot_posterior(denergy_dataset, name='hamiltonian_diffs',
                            fname=fname, fig_dir=h_dir, title_str=title_str,
                            filter_str='hamiltonian')

        energy_transitions = build_energy_transition_dataset(energy_data)
        traceplot_posterior(energy_transitions, name='potential_transitions',
                            fname=fname, fig_dir=pe_dir, title_str=title_str,
                            filter_str='potential')
        traceplot_posterior(energy_transitions, name='kinetic_transitions',
                            fname=fname, fig_dir=ke_dir, title_str=title_str,
                            filter_str='kinetic')
        traceplot_posterior(energy_transitions, name='hamiltonian_transitions',
                            fname=fname, fig_dir=h_dir, title_str=title_str,
                            filter_str='hamiltonian')

    dataset = build_dataset(run_data, run_params)
    #################################
    # Create ridgeplot of plaq diffs
    #################################
    rp_fname = _check_existing(fig_dir, f'{fname}_ridgeplot')
    rp_out_file = os.path.join(fig_dir, f'{rp_fname}.pdf')
    _ = az.plot_forest(dataset,
                       kind='ridgeplot',
                       var_names=['plaqs_diffs'],
                       ridgeplot_alpha=0.4,
                       ridgeplot_overlap=0.1,
                       combined=False)
    fig = plt.gcf()
    fig.suptitle(title_str, fontsize='x-large', y=1.025)
    savefig(fig, rp_out_file)
    if out_dir is not None:
        rp_out_file_ = os.path.join(out_dir, f'{rp_fname}.pdf')
        savefig(fig, rp_out_file_)

    ####################################################
    # Create histogram plots of the reversibility data.
    ####################################################
    if reverse_data is not None:
        _, _ = plot_reverse_data(reverse_data,
                                 params, run_params,
                                 runs_np=runs_np)

    ############################################
    # Create autocorrelation plot of plaq_diffs
    ############################################
    plaqs = np.array(run_data['plaqs'])
    plaqs_therm = plaqs.T
    fig, _ = plot_autocorr(plaqs_therm, params, run_params, name='plaqs')

    ####################################################
    # Create traceplot + posterior plot of observables
    ####################################################
    var_names = ['plaqs_diffs', 'accept_prob',
                 'charges', 'tunneling_rate',
                 'dx_out', 'dx_proposed']
    if hasattr(dataset, 'dx'):
        var_names.append('dx')

    dx_dir = os.path.join(fig_dir, 'dx_plots')
    io.check_else_make_dir(dx_dir)
    traceplot_posterior(dataset, name='dx', fname=fname, fig_dir=dx_dir,
                        title_str=title_str, filter_str='dx')

    sld_dir = os.path.join(fig_dir, 'sumlogdet_plots')
    io.check_else_make_dir(sld_dir)
    traceplot_posterior(dataset, name='sumlogdet',
                        fname=fname, fig_dir=sld_dir,
                        title_str=title_str,
                        filter_str='sumlogdet')

    traceplot_posterior(dataset, '', fname=fname, fig_dir=fig_dir,
                        title_str=title_str, filter_str=var_names)

    return dataset, energy_dataset
