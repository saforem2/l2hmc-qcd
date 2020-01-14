"""
gauge_inference_np.py

Runs tensorflow independent inference on a trained model.

Author: Sam Foreman (github: @saforem2)
Date: 01/09/2020
"""
import os

import xarray as xr
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

import utils.file_io as io

from config import NetWeights
from runners.runner_np import create_dynamics, run_inference_np
from utils.file_io import timeit
from utils.parse_inference_args_np import parse_args as parse_inference_args
from plotters.energy_plotter import EnergyPlotter
from plotters.plot_observables import plot_autocorrs, plot_charges
from plotters.gauge_model_plotter import GaugeModelPlotter
from lattice.lattice import u1_plaq_exact
from plotters.seaborn_plots import plot_setup

HEADER = 80 * '-'

def inference_plots_old(run_data, energy_data, params, run_params, **kwargs):
    """Make all inference plots from inference run."""
    run_str = run_params['run_str']
    log_dir = params['log_dir']
    figs_dir = os.path.join(log_dir, 'figures_np')
    fig_dir = os.path.join(figs_dir, run_str)
    plotter = GaugeModelPlotter(params, figs_dir)
    energy_plotter = EnergyPlotter(params, fig_dir)

    apd, pkwds = plotter.plot_observables(run_data, **run_params)
    title = pkwds['title']
    qarr = np.array(run_data['charges']).T
    qarr_int = np.around(qarr)

    out_file = os.path.join(fig_dir, 'charges_grid.png')
    fig, ax = plot_charges(qarr, out_file, title=title, nrows=4)

    out_file = os.path.join(fig_dir, 'chargs_autocorr_grid.png')
    fig, ax = plot_autocorrs(qarr_int, out_file=out_file, title=title, nrows=4)

    fig, ax = plt.subplots()
    ax.hist(qarr_int.flatten(), density=True, label=r"""$\mathcal{Q}$""")
    ax.legend(loc='best')
    ax.set_title(title, fontsize='x-large')
    plt.tight_layout()
    out_file = os.path.join(fig_dir, 'charges_histogram.pdf')
    plt.savefig(out_file, dpi=200, bbox_inches='tight')

    np_data = energy_plotter.plot_energies(energy_data,
                                           out_dir='np',
                                           **run_params)

    io.save_dict(np_data, log_dir, 'energy_data')


def _get_title(params, run_params):
    lf = params['num_steps']
    beta = run_params['beta']
    eps = run_params['eps']
    title_str = (r"$N_{\mathrm{LF}} = $" + f'{lf}, '
                 r"$\beta = $" + f'{beta:.1g}, '
                 r"$\varepsilon = $" + f'{eps:.3g}')
    eps_str = f'{eps:.4g}'.replace('.', '')

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
    dataset = xr.Dataset(rd_dict)

    return dataset


def _check_existing(out_dir, fname):
    if os.path.isfile(os.path.join(out_dir, f'{fname}.pdf')):
        timestr = io.get_timestr()
        hour_str = timestr['hour_str']
        fname += f'_{hour_str}'

    return fname


def inference_plots(run_data, params, run_params):
    run_str = run_params['run_str']
    log_dir = params['log_dir']
    figs_dir = os.path.join(log_dir, 'figures_np')
    fig_dir = os.path.join(figs_dir, run_str)
    io.check_else_make_dir(fig_dir)

    base_dir = os.path.abspath('/home/foremans/inference_numpy_figs')
    io.check_else_make_dir(base_dir)
    log_str = log_dir.split('/')[-1]

    out_dir = os.path.join(base_dir, log_str)
    io.check_else_make_dir(out_dir)

    #  title_str = _get_title(params, run_params)
    fname, title_str, _ = plot_setup(log_dir, run_params)
    tp_fname = f'{fname}_traceplot'
    rp_fname = f'{fname}_ridgeplot'

    dataset = build_dataset(run_data, run_params)

    #################################
    # Create traceplot of plaq diffs
    #################################
    _ = az.plot_trace(dataset, compact=True, combined=True,
                      var_names=['plaqs_diffs', 'actions',
                                 'charges', 'dx', 'accept_prob'])
    fig = plt.gcf()
    fig.suptitle(title_str, fontsize=24, y=1.05)

    # Save figure in `log_dir`
    tp_fname = _check_existing(fig_dir, tp_fname)
    tp_out_file = os.path.join(fig_dir, f'{tp_fname}.pdf')
    io.log(80 * '-')
    io.log(f'Saving figure to: {tp_out_file}.')
    fig.savefig(tp_out_file, dpi=200, bbox_inches='tight')
    io.log(80 * '-')

    # Save figure in `~/inference_numpy_figs/`
    tp_fname = _check_existing(out_dir, tp_fname)
    tp_out_file = os.path.join(out_dir, f'{tp_fname}.pdf')
    io.log(80 * '-')
    io.log(f'Saving figure to: {tp_out_file}.')
    fig.savefig(tp_out_file, dpi=200, bbox_inches='tight')
    io.log(80 * '-')

    #################################
    # Create ridgeplot of plaq diffs
    #################################
    axes = az.plot_forest(dataset,
                          kind='ridgeplot',
                          var_names=['plaqs_diffs'],
                          ridgeplot_alpha=0.4,
                          ridgeplot_overlap=0.1,
                          combined=False)
    #  axes[0].set_title(title_str, fontsize='large')

    # Save figure in `log_dir` 
    rp_fname = _check_existing(fig_dir, rp_fname)
    rp_out_file = os.path.join(fig_dir, f'{rp_fname}.pdf')
    io.log(80 * '-')
    io.log(f'Saving figure to: {rp_out_file}.')
    plt.savefig(rp_out_file, dpi=200, bbox_inches='tight')
    io.log(80 * '-')

    # Save figure in `~/inference_numpy_figs/`
    rp_fname = _check_existing(out_dir, rp_fname)
    rp_out_file = os.path.join(out_dir, f'{rp_fname}.pdf')
    io.log(80 * '-')
    io.log(f'Saving figure to: {rp_out_file}.')
    plt.savefig(rp_out_file, dpi=200, bbox_inches='tight')
    io.log(80 * '-')

    return dataset


@timeit
def main(args):
    """Perform tensorflow-independent inference on a trained model."""
    log_dir = os.path.abspath(args.log_dir)
    dynamics, lattice = create_dynamics(log_dir, args.eps)

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
    }

    run_params, run_data, energy_data = run_inference_np(log_dir, dynamics,
                                                         lattice, run_params,
                                                         init=args.init)
    params = load_pkl(os.path.join(log_dir, 'parameters.pkl'))
    datset = inference_plots(run_data, params, run_params)

    return run_params, run_data, energy_data, dataset


if __name__ == '__main__':
    FLAGS = parse_inference_args()
    io.log(HEADER)
    io.log('FLAGS: ')
    for key, val in FLAGS.__dict__.items():
        io.log(f'  - {key}: {val}\n')

    io.log(HEADER)

    _ = main(FLAGS)
