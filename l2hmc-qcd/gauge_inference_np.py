"""
gauge_inference_np.py

Runs tensorflow independent inference on a trained model.

Author: Sam Foreman (github: @saforem2)
Date: 01/09/2020
"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import arviz as az
import xarray as xr
import utils.file_io as io

from config import NetWeights
from lattice.lattice import u1_plaq_exact
from runners.runner_np import (create_dynamics, load_pkl, run_inference_np,
                               create_lattice, _update_params)
from utils.file_io import timeit
from utils.parse_inference_args_np import parse_args as parse_inference_args
from plotters.data_utils import InferenceData
from plotters.seaborn_plots import plot_setup
from plotters.energy_plotter import EnergyPlotter
from plotters.plot_observables import plot_autocorrs, plot_charges
from plotters.gauge_model_plotter import GaugeModelPlotter

HEADER = 80 * '-'

mpl.rcParams['axes.formatter.limits'] = -4, 4


def _get_title(params, run_params):
    lf = params['num_steps']
    beta = run_params['beta']
    eps = run_params['eps']
    title_str = (r"$N_{\mathrm{LF}} = $" + f'{lf}, '
                 r"$\beta = $" + f'{beta:.1g}, '
                 r"$\varepsilon = $" + f'{eps:.3g}')

    if params['eps_fixed']:
        title_str += ' (fixed)'

    if params['clip_value'] > 0:
        clip_value = params['clip_value']
        title_str += f', clip: {clip_value}'

    return title_str


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

        if 'charges' in key:
            arr = np.around(arr)

        rd_dict[key] = xr.DataArray(arr,
                                    dims=['chain', 'draw'],
                                    coords=[chains, draws])

    rd_dict['charges_squared'] = rd_dict['charges'] ** 2

    charges = rd_dict['charges'].values.T
    tunneling_rate = calc_tunneling_rate(charges).T
    rd_dict['tunneling_rate'] = xr.DataArray(tunneling_rate,
                                             dims=['chain', 'draw'],
                                             coords=[chains, draws])
    dataset = xr.Dataset(rd_dict)

    return dataset


def _check_existing(out_dir, fname):
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
    plt.tight_layout()
    io.log(f'Saving figure to: {out_file}...')
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    if out_dir is not None:
        fout = os.path.join(out_dir, f'{fname}.pdf')
        io.log(f'Saving figure to: {fout}...')
        fig.savefig(fout, dpi=200, bbox_inches='tight')

    return fig, ax


# pylint: disable=too-many-locals
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

    tp_fname = f'{fname}_traceplot'
    etp_fname = f'{fname}_energy_traceplot'
    pp_fname = f'{fname}_posterior'
    epp_fname = f'{fname}_energy_posterior'
    rp_fname = f'{fname}_ridgeplot'

    dataset = build_dataset(run_data, run_params)

    def _savefig(fig, out_file):
        io.log(HEADER)
        io.log(f'Saving figure to: {out_file}.')
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
        io.log(HEADER)

    def _plot_posterior(data, out_file, var_names=None, out_file1=None):
        _ = az.plot_posterior(data, var_names=var_names)
        fig = plt.gcf()
        fig.suptitle(title_str, fontsize=24, y=1.05)
        _savefig(fig, out_file)
        if out_file1 is not None:
            _savefig(fig, out_file1)

    def _plot_trace(data, out_file, var_names=None, out_file1=None):
        _ = az.plot_trace(data, compact=True,
                          combined=True,
                          var_names=var_names)
        fig = plt.gcf()
        fig.suptitle(title_str, fontsize=24, y=1.05)
        _savefig(fig, out_file)
        if out_file1 is not None:
            _savefig(fig, out_file1)

    ####################################################
    # Create traceplot + posterior plot of observables
    ####################################################
    tp_fname = _check_existing(fig_dir, tp_fname)
    pp_fname = _check_existing(fig_dir, pp_fname)
    tp_out_file = os.path.join(fig_dir, f'{tp_fname}.pdf')
    pp_out_file = os.path.join(fig_dir, f'{pp_fname}.pdf')

    var_names = ['tunneling_rate', 'plaqs_diffs',
                 'accept_prob', 'charges_squared', 'charges']
    if hasattr(dataset, 'dx'):
        var_names.append('dx')

    tp_out_file_ = None
    pp_out_file_ = None
    if out_dir is not None:
        io.check_else_make_dir(out_dir)
        tp_out_file_ = os.path.join(out_dir, f'{tp_fname}.pdf')
        pp_out_file_ = os.path.join(out_dir, f'{pp_fname}.pdf')

    _plot_trace(dataset, tp_out_file,
                var_names=var_names,
                out_file1=tp_out_file_)
    _plot_posterior(dataset, pp_out_file,
                    out_file1=pp_out_file_)

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
    if out_dir is not None:
        rp_out_file_ = os.path.join(out_dir, f'{rp_fname}.pdf')
        _savefig(fig, rp_out_file_)

    ####################################################
    # Create traceplot + possterior plot of energy data
    ####################################################
    if energy_data is not None:
        energy_dataset = build_energy_dataset(energy_data)
        etp_fname = _check_existing(fig_dir, etp_fname)
        epp_fname = _check_existing(fig_dir, epp_fname)
        etp_out_file = os.path.join(fig_dir, f'{etp_fname}.pdf')
        epp_out_file = os.path.join(fig_dir, f'{epp_fname}.pdf')
        _plot_trace(energy_dataset, etp_out_file)
        _plot_posterior(energy_dataset, epp_out_file)

    ####################################################
    # Create histogram plots of the reversibility data.
    ####################################################
    if reverse_data is not None:
        _, _ = plot_reverse_data(reverse_data, params, run_params,
                                 runs_np=runs_np)

    return dataset, energy_dataset


def make_csv(run_data, energy_data, run_params):
    """Make .csv file containing relevant inference data."""
    plaq_exact = u1_plaq_exact(run_params['beta'])
    csv_dict = {}
    for e_key, e_val in energy_data.items():
        arr = np.squeeze(np.array(e_val)).flatten()
        csv_dict[e_key] = arr
        _shape = arr.shape

    for r_key, r_val in run_data.items():
        arr = np.squeeze(np.array(r_val)).flatten()
        if len(arr.shape) == 1:
            arr = np.squeeze(np.array([arr for _ in _shape]))
        if r_key == 'plaqs':
            csv_dict['plaqs_diffs'] = plaq_exact - np.squeeze(np.array(r_val))
        else:
            csv_dict[r_key] = np.squeeze(np.array(r_val))
    csv_df = pd.DataFrame(csv_dict)
    csv_file = os.path.join(run_params['run_dir'], 'inference_data.csv')
    io.log(f'Saving inference data to {csv_file}.')
    csv_df.to_csv(csv_file, mode='a')
    return csv_dict


@timeit
def main(args):
    """Perform tensorflow-independent inference on a trained model."""
    log_dir = getattr(args, 'log_dir', None)
    if log_dir is None:
        params_file = os.path.join(os.getcwd(), 'params.pkl')
    else:
        log_dir = os.path.abspath(log_dir)
        params_file = os.path.join(log_dir, 'parameters.pkl')

    params = load_pkl(params_file)
    params = _update_params(params, args.eps, args.num_steps, args.batch_size)
    lattice = create_lattice(params)
    _fn = lattice.calc_actions_np

    log_dir = params['log_dir']
    dynamics = create_dynamics(log_dir,
                               potential_fn=_fn,
                               x_dim=lattice.x_dim,
                               hmc=args.hmc,
                               eps=args.eps,
                               num_steps=args.num_steps,
                               batch_size=args.batch_size,
                               model_type='GaugeModel')
    if args.hmc:
        net_weights = NetWeights(0, 0, 0, 0, 0, 0)
    else:
        net_weights = NetWeights(x_scale=args.x_scale_weight,
                                 x_translation=args.x_translation_weight,
                                 x_transformation=args.x_transformation_weight,
                                 v_scale=args.v_scale_weight,
                                 v_translation=args.v_translation_weight,
                                 v_transformation=args.v_transformation_weight)
    run_params = {
        'beta': args.beta,
        'eps': dynamics.eps,
        'net_weights': net_weights,
        'run_steps': args.run_steps,
        'num_steps': dynamics.num_steps,
        'batch_size': lattice.batch_size,
    }

    outputs = run_inference_np(log_dir, dynamics, lattice,
                               run_params, init=args.init, skip=False)
    run_data = outputs['data']['run_data']
    energy_data = outputs['data']['energy_data']
    #  reverse_data = outputs['data']['reverse_data']
    run_params = outputs['run_params']
    #  beta = run_params['beta']
    #  plaq_exact = u1_plaq_exact(beta)

    make_csv(run_data, energy_data, run_params)

    #  reverse_csv_file = os.path.join(run_params['run_dir'], 'reverse_data.csv')
    #  reverse_df = pd.DataFrame(reverse_data)
    #  io.log(f'Saving reverse data to `.csv` file: {reverse_csv_file}.')
    #  reverse_df.to_csv(reverse_csv_file)

    run_params = outputs['run_params']
    params = load_pkl(os.path.join(log_dir, 'parameters.pkl'))
    dataset, energy_dataset = inference_plots(outputs['data'], params,
                                              outputs['run_params'],
                                              runs_np=True)

    return run_params, outputs['data'], dataset


if __name__ == '__main__':
    FLAGS = parse_inference_args()
    io.log(HEADER)
    io.log('FLAGS: ')
    for key, val in FLAGS.__dict__.items():
        io.log(f'  - {key}: {val}\n')

    io.log(HEADER)

    _ = main(FLAGS)
