"""
inference_plots.py

Contains helper methods for plotting inference results.

Author: Sam Foreman (github: @saforem2)
Date: 01/15/2020
"""
# pylint:disable=too-many-locals,too-many-arguments,invalid-name,
# pylint:disable=too-many-arguments,too-many-statements
import os

import arviz as az
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import utils.file_io as io

from plotters.seaborn_plots import plot_setup

sns.set_palette('bright')

HEADER = 80 * '-'
SEPERATOR = 80 * '-'
MARKERS = 10 * ['o', 'v', '^', '<', '>', 's', 'd', '*', '+', 'x']

mpl.rcParams['axes.formatter.limits'] = -4, 4

def savefig(fig, out_file):
    """Save `fig` to `out_file`."""
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


def _check_existing(out_dir, fname):
    """Check if `fname` exists in `out_dir`. If so, append time to `fname`."""
    if os.path.isfile(os.path.join(out_dir, f'{fname}.pdf')):
        timestr = io.get_timestr()
        hour_str = timestr['hour_str']
        fname += f'_{hour_str}'

    return fname


def plot_reverse_data(run_data, params, **kwargs):
    """Plot reversibility results."""
    run_str = run_data.run_params['run_str']
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
        fname, title_str, _ = plot_setup(log_dir, run_data.run_params)
    except FileNotFoundError:
        return None, None

    fname = f'{fname}_reversibility_hist'
    out_file = os.path.join(fig_dir, f'{fname}.pdf')
    fig, ax = plt.subplots()
    for key, val in run_data.reverse_data.items():
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


def _plot_volume_diff(drms_in, drms_out, name, out_file, title_str=None):
    """Plot `rms_diff_out` vs. `rms_diff_in`."""
    batch_size = drms_in.shape[0]
    fig, ax = plt.subplots()
    if batch_size < 10:
        dalpha = 0.1
    if 10 < batch_size < 20:
        dalpha = 0.05
    if batch_size > 20:
        dalpha = 0.01
    for idx in range(batch_size):
        ax.plot(drms_in[idx], drms_out[idx],
                ls='', marker=MARKERS[idx], markersize=4.,
                alpha=(1. - idx * dalpha),
                fillstyle='none', label=f'sample {idx}')
    #  if name == 'x':
    #      #  ax.axhline(y=3 * np.pi / 4, label=r"$3\pi/4$")
    #      ax.axhline(y=2*np.pi/3, label=r"$\frac{2\pi}{3}$")
    #      ax.axhline(y=np.pi, label=r"$\pi$")
    ax.set_xlabel(f'd{name}_rms_in', fontsize='large')
    ax.set_ylabel(f'd{name}_rms_out', fontsize='large')
    ax.legend(loc='best')
    if title_str is not None:
        ax.set_title(title_str, fontsize='x-large')
    io.log(f'Saving {name} volume diffs to: {out_file}.')
    plt.savefig(out_file, dpi=400, bbox_inches='tight')

    return fig, ax


def plot_volume_diffs(volume_diffs, fig_dir, title_str=None):
    """Plot RMS diff of the output pert. vs. the RMS diff of the input pert."""
    def rms(x):
        x = np.array(x).transpose((1, 0, -1))
        return np.sqrt(np.mean(x ** 2, axis=-1))

    dx_in_rms = rms(volume_diffs['dx_in'])
    dv_in_rms = rms(volume_diffs['dv_in'])
    dx_out_rms = rms(volume_diffs['dx_out'])
    dv_out_rms = rms(volume_diffs['dv_out'])

    out_dir = os.path.join(fig_dir, 'volume_diffs')
    io.check_else_make_dir(out_dir)

    xfile = os.path.join(out_dir, 'x_volume_diffs.pdf')
    _, _ = _plot_volume_diff(dx_in_rms, dx_out_rms,
                             name='x', out_file=xfile,
                             title_str=title_str)
    vfile = os.path.join(out_dir, 'v_volume_diffs.pdf')
    _, _ = _plot_volume_diff(dv_in_rms, dv_out_rms,
                             name='v', out_file=vfile,
                             title_str=title_str)


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
        fig.suptitle(title_str, fontsize='x-large', y=1.1)
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
    #  if title_str is not None:
    #      fig.suptitle(title_str, fontsize='xx-large', y=1.1)
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


def inference_plots(run_data, params, **kwargs):
    """Create trace plots of lattice observables and energy data."""
    run_str = run_data.run_params['run_str']
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
        fname, title_str, _ = plot_setup(log_dir, run_data.run_params)
    except FileNotFoundError:
        return dataset, energy_dataset

    ##############################################################
    # Symplectic check:
    # -----------------
    # Look at how the sampler transforms regions of phase space.
    ##############################################################
    if run_data.run_params['symplectic_check']:
        plot_volume_diffs(run_data.volume_diffs, fig_dir, title_str=title_str)

    ####################################################
    # Create traceplot + possterior plot of energy data
    ####################################################
    dataset = run_data.build_dataset()
    energy_dataset = run_data.build_energy_dataset()
    denergy_dataset = run_data.build_energy_diffs_dataset()
    energy_transitions = run_data.build_energy_transition_dataset()
    #  dataset = build_dataset(run_data)
    #  energy_dataset = build_energy_dataset(run_data.energy_data)
    #  denergy_dataset = build_energy_diffs_dataset(run_data.energy_data)

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

    traceplot_posterior(denergy_dataset, name='potential_diffs',
                        fname=fname, fig_dir=pe_dir, title_str=title_str,
                        filter_str='potential')
    traceplot_posterior(denergy_dataset, name='kinetic_diffs',
                        fname=fname, fig_dir=ke_dir, title_str=title_str,
                        filter_str='kinetic')
    traceplot_posterior(denergy_dataset, name='hamiltonian_diffs',
                        fname=fname, fig_dir=h_dir, title_str=title_str,
                        filter_str='hamiltonian')

    traceplot_posterior(energy_transitions, name='potential_transitions',
                        fname=fname, fig_dir=pe_dir, title_str=title_str,
                        filter_str='potential')
    traceplot_posterior(energy_transitions, name='kinetic_transitions',
                        fname=fname, fig_dir=ke_dir, title_str=title_str,
                        filter_str='kinetic')
    traceplot_posterior(energy_transitions, name='hamiltonian_transitions',
                        fname=fname, fig_dir=h_dir, title_str=title_str,
                        filter_str='hamiltonian')

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
    fig.suptitle(title_str, fontsize='x-large', y=1.1)
    savefig(fig, rp_out_file)
    if out_dir is not None:
        rp_out_file_ = os.path.join(out_dir, f'{rp_fname}.pdf')
        savefig(fig, rp_out_file_)

    ####################################################
    # Create histogram plots of the reversibility data.
    ####################################################
    _, _ = plot_reverse_data(run_data, params, runs_np=runs_np)

    ############################################
    # Create autocorrelation plot of plaq_diffs
    ############################################
    plaqs = np.array(run_data.run_data['plaqs_diffs']).T
    fig, _ = plot_autocorr(plaqs, params,
                           run_data.run_params, name='plaqs')

    ###############################################
    # Create plots for `dx_out` and `dx_proposed`
    ###############################################
    dx_dir = os.path.join(fig_dir, 'dx_plots')
    io.check_else_make_dir(dx_dir)
    traceplot_posterior(dataset, name='dx', fname=fname, fig_dir=dx_dir,
                        title_str=title_str, filter_str='dx')

    #########################################
    # Create plots for dynamics reverse data
    #########################################
    reverse_dir = os.path.join(fig_dir, 'reverse_plots')
    io.check_else_make_dir(reverse_dir)
    traceplot_posterior(dataset, name='reverse_diffs',
                        fname=fname, fig_dir=reverse_dir,
                        title_str=title_str,
                        filter_str=['xdiff_r', 'vdiff_r'])

    #############################################################
    # Create plots for `sumlogdet_out` and `sumlogdet_proposed`
    #############################################################
    sld_dir = os.path.join(fig_dir, 'sumlogdet_plots')
    io.check_else_make_dir(sld_dir)
    traceplot_posterior(dataset, name='sumlogdet',
                        fname=fname, fig_dir=sld_dir,
                        title_str=title_str,
                        filter_str='sumlogdet')

    ####################################################
    # Create traceplot + posterior plot of observables
    ####################################################
    var_names = ['plaqs_diffs', 'accept_prob',
                 'charges', 'tunneling_rate',
                 'dx_out', 'dx_proposed']

    traceplot_posterior(dataset, '', fname=fname, fig_dir=fig_dir,
                        title_str=title_str, filter_str=var_names)

    return dataset, energy_dataset
