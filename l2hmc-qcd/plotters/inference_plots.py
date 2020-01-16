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
import matplotlib.pyplot as plt

import utils.file_io as io

from .seaborn_plots import plot_setup
from lattice.lattice import u1_plaq_exact

HEADER = 80 * '-'


def therm_arr(arr, therm_frac=0.25):
    num_steps = arr.shape[0]
    therm_steps = int(therm_frac * num_steps)
    arr = arr[therm_steps:, :]
    steps = np.arange(therm_steps, num_steps)
    return arr, steps


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


def calc_tunneling_rate(charges):
    """Calculate the tunneling rate as the difference in charge b/t steps."""
    charges = np.around(charges)
    charges = np.insert(charges, 0, 0, axis=0)
    dq = np.abs(charges[1:] - charges[-1])
    return dq


def build_dataset(run_data, run_params):
    rd_dict = {}
    for key, val in run_data.items():
        if 'mask' in key:
            continue

        arr, steps = therm_arr(np.array(val))
        arr = arr.T

        if 'plaqs' in key:
            key = 'plaqs_diffs'
            arr = u1_plaq_exact(run_params['beta']) - arr

        if 'charges' in key:
            arr = np.around(arr)

        rd_dict[key] = xr.DataArray(arr,
                                    dims=['chain', 'draw'],
                                    coords=[np.arange(arr.shape[0]), steps])

    rd_dict['charges_squared'] = rd_dict['charges'] ** 2

    charges = rd_dict['charges'].values.T
    tunneling_rate = calc_tunneling_rate(charges).T
    tmp = tunneling_rate.shape[0]
    rd_dict['tunneling_rate'] = xr.DataArray(tunneling_rate,
                                             dims=['chain', 'draw'],
                                             coords=[np.arange(tmp), steps])
    dataset = xr.Dataset(rd_dict)

    return dataset


def _check_existing(out_dir, fname):
    if os.path.isfile(os.path.join(out_dir, f'{fname}.pdf')):
        timestr = io.get_timestr()
        hour_str = timestr['hour_str']
        fname += f'_{hour_str}'

    return fname


def inference_plots(run_data, energy_data, params, run_params):
    """Create trace plots of lattice observables and energy data."""
    run_str = run_params['run_str']
    log_dir = params['log_dir']
    figs_dir = os.path.join(log_dir, 'figures_np')
    fig_dir = os.path.join(figs_dir, run_str)
    io.check_else_make_dir(fig_dir)

    fname, title_str, _ = plot_setup(log_dir, run_params)
    tp_fname = f'{fname}_traceplot'
    etp_fname = f'{fname}_energy_traceplot'
    pp_fname = f'{fname}_posterior'
    epp_fname = f'{fname}_posterior'
    rp_fname = f'{fname}_ridgeplot'

    dataset = build_dataset(run_data, run_params)
    energy_dataset = build_energy_dataset(energy_data)

    def _savefig(fig, out_file):
        io.log(HEADER)
        io.log(f'Saving figure to: {out_file}.')
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
        io.log(HEADER)

    def _plot_posterior(data, out_file, var_names=None):
        _ = az.plot_posterior(data, var_names=var_names)
        fig = plt.gcf()
        fig.suptitle(title_str, fontsize=24, y=1.05)
        _savefig(fig, out_file)

    def _plot_trace(data, out_file, var_names=None):
        _ = az.plot_trace(data, compact=True,
                          combined=True,
                          var_names=var_names)
        fig = plt.gcf()
        fig.suptitle(title_str, fontsize=24, y=1.05)
        _savefig(fig, out_file)

    ####################################################
    # Create traceplot + posterior plot of observables 
    ####################################################
    tp_fname = _check_existing(fig_dir, tp_fname)
    pp_fname = _check_existing(fig_dir, pp_fname)
    tp_out_file = os.path.join(fig_dir, f'{tp_fname}.pdf')
    pp_out_file = os.path.join(fig_dir, f'{pp_fname}.pdf')
    var_names = ['plaqs_diffs', 'dx', 'accept_prob',
                 'tunneling_rate', 'charges', 'charges_squared']
    _plot_trace(dataset, tp_out_file, var_names=var_names)
    _plot_posterior(dataset, pp_out_file, var_names=var_names)

    ####################################################
    # Create traceplot + possterior plot of energy data
    ####################################################
    etp_fname = _check_existing(fig_dir, etp_fname)
    epp_fname = _check_existing(fig_dir, epp_fname)
    etp_out_file = os.path.join(fig_dir, f'{etp_fname}.pdf')
    epp_out_file = os.path.join(fig_dir, f'{epp_fname}.pdf')
    _plot_trace(energy_dataset, etp_out_file)
    _plot_posterior(energy_dataset, epp_out_file)

    #################################
    # Create ridgeplot of plaq diffs
    #################################
    rp_fname = _check_existing(fig_dir, rp_fname)
    rp_out_file = os.path.join(fig_dir, f'{rp_fname}.pdf')
    _ = az.plot_forest(dataset,
                       kind='ridgeplot',
                       var_names=['plaqs_diffs'],
                       ridgeplot_alpha=0.4,
                       ridgeplot_overlap=0.1,
                       combined=False)
    fig = plt.gcf()
    fig.suptitle(title_str, fontsize='x-large', y=1.025)
    _savefig(fig, rp_out_file)

    return dataset, energy_dataset
