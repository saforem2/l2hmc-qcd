"""
inference_plots.py

Contains helper methods for plotting inference results.

Author: Sam Foreman (github: @saforem2)
Date: 01/15/2020
"""
import os

import arviz as az
import numpy as np
import seaborn as sns
import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.colors import (BoundaryNorm, LinearSegmentedColormap,
                               ListedColormap)
from matplotlib.pyplot import cycler
from matplotlib.collections import LineCollection

import utils.file_io as io

from plotters.plot_utils import plot_setup

#  from plotters.seaborn_plots import plot_setup
#  import matplotlib.cm

# pylint:disable=too-many-locals,too-many-arguments,invalid-name,
# pylint:disable=too-many-arguments,too-many-statements

sns.set_palette('bright')

HEADER = 80 * '-'
SEPERATOR = 80 * '-'
#  MARKERS = 10 * ['o', 'v', '^', '<', '>', 's', 'd', '*', '+', 'x']
MARKERS = 10 * ['o', 's', 'x', 'v', 'h', '^', 'p', '<', 'd', '>', 'o']

mpl.rcParams['axes.formatter.limits'] = -4, 4


def savefig(fig, out_file):
    """Save `fig` to `out_file`."""
    io.log(HEADER)
    io.log(f'Saving figure to: {out_file}.')
    #  plt.tight_layout()
    try:
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
    except:
        fig.savefig(out_file, bbox_inches='tight')
    io.log(HEADER)


def _plaq_sums(x):
    """Calculate the sum of the link variables around each plaquette."""
    return (x[..., 0]
            - x[..., 1]
            - np.roll(x[..., 0], shift=-1, axis=-2)
            + np.roll(x[..., 1], shift=-1, axis=-3))


def _plot_angle_timeseries(chain, num_steps=500,
                           out_file=None, title_str=None):
    """Plot an angular repr. of the phase's timeseries."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(211, polar=True)
    chain, steps = therm_arr(chain)
    num_steps = min([num_steps, chain.shape[0]])
    r = np.linspace(0, 1, num_steps)
    theta = chain[:num_steps]
    points = np.array([theta, r]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #  norm = plt.Normalize(theta.min(), theta.max())
    norm = plt.Normalize(theta.min(), theta.max())
    lc = LineCollection(segments, cmap='hsv', norm=norm, zorder=1, alpha=0.6)

    # Set the values used for colormapping
    lc.set_array(theta)
    lc.set_linewidth(0.9)
    _ = ax.add_collection(lc)
    ax.set_yticklabels([])

    _ = ax.scatter(theta, r, marker='o', c=theta, s=5.,
                   cmap='hsv', alpha=0.85, zorder=2)

    ax = fig.add_subplot(212)

    points = np.array([steps[:num_steps], theta]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(theta.min(), theta.max())
    lc = LineCollection(segments, cmap='hsv', norm=norm, zorder=1, alpha=0.6)
    #
    #   # Set the values used for colormapping
    lc.set_array(theta)
    lc.set_linewidth(1.)
    _ = ax.add_collection(lc)
    _ = ax.scatter(steps[:num_steps], theta, c=theta,
                   cmap='hsv', alpha=1., zorder=2)
    ylabels = [r'$0$', r'$\pi / 2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    yticks = [0, np.pi/2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.grid(True)
    ax.set_ylabel(r'$\phi_{\mu}(x)$', fontsize='large')
    ax.set_xlabel(r'Step', fontsize='large')

    if title_str is not None:
        fig.suptitle(title_str, fontsize='x-large')
    if out_file is not None:
        savefig(fig, out_file)

    return fig, ax


def plot_angle_timeseries(chains, num_plots=20, num_steps=500,
                          out_dir=None, title_str=None):
    """Create `num_plots` angular timeseries from `chains`."""
    chains = np.array(chains)
    out_file = None
    # chains.shape = (num_steps, batch_size, num_links)
    chain = chains[:, 0, :]  # look at first sample in batch
    for idx in range(num_plots):
        if out_dir is not None:
            out_file = os.path.join(out_dir, f'phase_timeseries{idx}.png')

        _, _ = _plot_angle_timeseries(chain[:, idx],
                                      num_steps=num_steps,
                                      out_file=out_file,
                                      title_str=title_str)
        plt.close('all')


def plot_plaq_timeseries(chains, num_plots, lattice_shape,
                         num_steps=500, out_dir=None, title_str=None):
    """Create `num_plots` angular timeseries of plaquettes."""
    chains = np.array(chains)
    out_file = None
    num_plots = min([num_plots, chains.shape[1]])
    #  num_plots = min([num_plots, chain_ps.shape[-1]])
    for idx in range(num_plots):
        if out_dir is not None:
            out_file = os.path.join(out_dir, f'plaqs_timeseries{idx}.png')

        chain = chains[:, idx, :]
        chain = np.reshape(chain, (-1, *lattice_shape))
        plaqs = (chain[..., 0]
                 - chain[..., 1]
                 - np.roll(chain[..., 0], shift=-1, axis=2)
                 + np.roll(chain[..., 1], shift=-1, axis=1))

        _, _ = _plot_angle_timeseries(plaqs[:, 0, 0],
                                      num_steps=num_steps,
                                      out_file=out_file,
                                      title_str=title_str)
        plt.close('all')


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
    out_file = os.path.join(fig_dir, f'{name}_autocorrelation.png')
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
    if os.path.isfile(os.path.join(out_dir, f'{fname}.png')):
        timestr = io.get_timestr()
        hour_str = timestr['hour_str']
        fname += f'_{hour_str}'

    return fname


def plot_reverse_data(run_data, params, **kwargs):
    """Plot reversibility results.

    Args:
        run_data (RunData object): Container of data from inference run.
        params (dict): Dictionary of parameters.
    """
    run_str = run_data.run_params['run_str']
    log_dir = params['log_dir']
    runs_np = kwargs.get('runs_np', True)
    if runs_np:
        figs_dir = os.path.join(log_dir, 'figures_np')
    else:
        figs_dir = os.path.join(log_dir, 'figures_tf')

    fig_dir = os.path.join(figs_dir, run_str)
    io.check_else_make_dir(fig_dir)
    out_dir = os.path.join(fig_dir, 'reverse_plots')
    io.check_else_make_dir(out_dir)
    #  out_dir = kwargs.get('out_dir', None)
    try:
        fname, title_str, _ = plot_setup(log_dir, run_data.run_params)
    except FileNotFoundError:
        return None, None

    #  fname = f'{fname}_reversibility_hist'
    #  out_file = os.path.join(fig_dir, f'{fname}.png')
    #  fig, ax = plt.subplots()
    reverse_data = {
        'xdiff_r': run_data.run_data['xdiff_r'],
        'vdiff_r': run_data.run_data['vdiff_r'],
    }
    for key, val in reverse_data.items():
        #  fname = f'{key}_reversibility.png'
        out_file = os.path.join(out_dir, f'{key}.png')
        fig, ax = plt.subplots()
        sns.kdeplot(np.array(val).flatten(), shade=True, label=key, ax=ax)
        ax.legend(loc='best')
        ax.set_title(title_str)
        plt.tight_layout()
        savefig(fig, out_file)
        #  if out_dir is not None:
        #      fout = os.path.join(out_dir, f'{fname}.png')
        #      io.log(f'Saving figure to: {fout}...')
        #      fig.savefig(fout, dpi=200, bbox_inches='tight')

    return fig, ax


def _plot_volume_diff(drms_data, fit_data, out_dir, title_str=None):
    """Plot `rms_diff_out` vs. `rms_diff_in`."""
    xfit, yfit, polyfits = fit_data

    fig, axes = plt.subplots(nrows=2, ncols=1)
    lines = []
    labels = []
    for idx, key in enumerate(polyfits.keys()):
        ax = axes[idx]

        k = (f'd{key}_in', f'd{key}_out')
        batch_size = drms_data[k[0]].shape[0]
        for jdx in range(batch_size):
            if idx == 0:
                label = f'sample {jdx}'
            else:
                label = None

            line, = ax.plot(drms_data[k[0]][jdx],
                            drms_data[k[1]][jdx],
                            markersize=4., alpha=0.75,
                            ls='', marker=MARKERS[jdx],
                            fillstyle='none', label=label)
            if idx == 0:
                lines.append(line)
                labels.append(label)

        fitstr = (r"$y = $" + f' {polyfits[key][0]:.5g} '
                  + r"$x + $" + f'{polyfits[key][1]:.5g}')
        fit_line, = ax.plot(xfit[key], yfit[key], 'k-', label=fitstr)
        ax.legend([fit_line], [fitstr], loc='upper left')

        ax.set_xlabel(f'd{key}_rms_in', fontsize='large')
        ax.set_ylabel(f'd{key}_rms_out', fontsize='large')

    fig.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    if title_str is not None:
        axes[0].set_title(title_str, fontsize='x-large')
        #  fig.suptitle(title_str, fontsize='x-large')

    plt.tight_layout()
    out_file = os.path.join(out_dir, f'volume_diffs.png')
    savefig(fig, out_file)
    #  io.log(f'Saving volume diffs plot to: {out_file}.')
    #  plt.savefig(out_file, dpi=200, bbox_inches='tight')

    return fig, ax


def fit_volume_diffs(drms):
    """Create callable fit functions for the volume diffs."""
    drms = {key: val.flatten() for key, val in drms.items()}
    polyfits = {
        'x': np.polyfit(drms['dx_in'], drms['dx_out'], 1),
        'v': np.polyfit(drms['dv_in'], drms['dv_out'], 1),
    }
    x_arrs = {
        'x': np.arange(np.min(drms['dx_in']), np.max(drms['dx_in']), 0.05),
        'v': np.arange(np.min(drms['dv_in']), np.max(drms['dv_in']), 0.05),
    }
    fit_fns = {
        key: np.poly1d(val) for key, val in polyfits.items()
    }
    y_arrs = {
        key: fit_fn(x_arrs[key]) for key, fit_fn in fit_fns.items()
    }
    return x_arrs, y_arrs, polyfits


def plot_volume_diffs(volume_diffs, fig_dir, title_str=None):
    """Plot RMS diff of the output pert. vs. the RMS diff of the input pert."""
    out_dir = os.path.join(fig_dir, 'volume_diffs')
    io.check_else_make_dir(out_dir)

    def rms(x):
        x = np.array(x).transpose((1, 0, -1))
        return np.sqrt(np.mean(x ** 2, axis=-1))

    drms = {
        key: rms(val) for key, val in volume_diffs.items()
    }
    #  for key, val in volume_diffs.items():
    #      drms[key] = rms(val)

    fit_data = fit_volume_diffs(drms)


    _plot_volume_diff(drms, fit_data, out_dir, title_str=title_str)

    #  for key in polyfits.keys():
    #      k = (f'd{key}_in', f'd{key}_out')
    #      drms_data = (drms[k[0]], drms[k[1]])
    #      fit_data = (x_arrs[key], y_arrs[key], polyfits[key])
    #      out_file = os.path.join(out_dir, f'{key}_volume_diffs.png')
    #      #  fig, ax = _plot_volume_diff(drms_data, fit_data,
    #      #                           name=key, out_file=out_file,
    #      #                           title_str=title_str)
    #
    #  xfile = os.path.join(out_dir, 'x_volume_diffs.png')
    #  drms_data = (drms['dx_in'], drms['dx_out'])
    #  fit_data = (x_arrs['x'], y_arrs['x'], polyfits['x'])
    #  _, _ = _plot_volume_diff(drms_data, fit_data,
    #                           name='x', out_file=xfile,
    #                           title_str=title_str)
    #
    #  vfile = os.path.join(out_dir, 'v_volume_diffs.png')
    #  drms_data = (drms['dv_in'], drms['dv_out'])
    #  fit_data = (x_arrs['v'], y_arrs['v'], polyfits['v'])
    #  _, _ = _plot_volume_diff(drms_data, fit_data,
    #                           name='v', out_file=vfile,
    #                           title_str=title_str)


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
        fig.suptitle(title_str, fontsize='x-large')
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


def plot_losses(plaq_loss, charge_loss, title_str=None, out_dir=None):
    """Plot losses from inference run."""
    plaq_loss = np.array(plaq_loss)
    charge_loss = np.array(charge_loss)
    steps = np.arange(plaq_loss.shape[0])
    fig, axes = plt.subplots(nrows=2, sharex=True)
    axes[0].plot(steps, plaq_loss.mean(axis=1),
                 marker=',', ls='', label='plaq_loss')
    axes[0].legend(loc='best')
    axes[1].plot(steps, charge_loss.mean(axis=1),
                 marker=',', ls='', label='charge_loss')
    axes[1].legend(loc='best')
    axes[1].set_xlabel(f'Step', fontsize='large')
    if title_str is not None:
        fig.suptitle(title_str, fontsize='x-large')
    if out_dir is not None:
        out_file = os.path.join(out_dir, 'charge_loss.png')
        savefig(fig, out_file)


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
    tp_fout = os.path.join(fig_dir, f'{tp_fname}.png')
    pp_fout = os.path.join(fig_dir, f'{pp_fname}.png')
    plot_trace(dataset, tp_fout,
               title_str=title_str,
               filter_str=filter_str)
    plot_posterior(dataset, pp_fout,
                   title_str=title_str,
                   filter_str=filter_str)


def inference_plots(run_data, params, **kwargs):
    """Create trace plots of lattice observables and energy data.

    Args:
        run_data (RunData object): Container of inference data.
        params (dict): Dictionary of parameters.
    """
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

    ##############################################################
    # Symplectic check:
    # -----------------
    # Look at how the sampler transforms regions of phase space.
    ##############################################################
    if run_data.run_params['symplectic_check']:
        try:
            plot_volume_diffs(run_data.volume_diffs,
                              fig_dir, title_str=title_str)
        except:
            import pudb; pudb.set_trace()

    plot_losses(run_data.observables['plaq_loss'],
                run_data.observables['charge_loss'],
                title_str=title_str, out_dir=fig_dir)

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
    rp_out_file = os.path.join(fig_dir, f'{rp_fname}.png')
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
        rp_out_file_ = os.path.join(out_dir, f'{rp_fname}.png')
        savefig(fig, rp_out_file_)

    ####################################################
    # Create histogram plots of the reversibility data.
    ####################################################
    try:
        _, _ = plot_reverse_data(run_data, params, runs_np=runs_np)
    except:
        pass

    ############################################
    # Create autocorrelation plot of plaq_diffs
    ############################################
    plaqs = np.array(run_data.observables['plaqs_diffs']).T
    fig, _ = plot_autocorr(plaqs, params,
                           run_data.run_params, name='plaqs')
    plt.close('all')

    ###############################################
    # Create plots for `dx_out` and `dx_proposed`
    ###############################################
    #  dx_dir = os.path.join(fig_dir, 'dx_plots')
    #  io.check_else_make_dir(dx_dir)
    #  traceplot_posterior(dataset, name='dx', fname=fname, fig_dir=dx_dir,
    #                      title_str=title_str, filter_str='dx')

    ##################################
    # Create plots for `plaqs_diffs`
    ##################################
    pd_dir = os.path.join(fig_dir, 'plaqs_diffs_plots')
    io.check_else_make_dir(pd_dir)
    traceplot_posterior(dataset, name='plaqs_diffs', fname=fname,
                        fig_dir=pd_dir, title_str=title_str,
                        filter_str='plaqs_diffs')
    plt.close('all')

    #########################################
    # Create plots for dynamics reverse data
    #########################################
    reverse_dir = os.path.join(fig_dir, 'reverse_plots')
    io.check_else_make_dir(reverse_dir)
    traceplot_posterior(dataset, name='reverse_diffs',
                        fname=fname, fig_dir=reverse_dir,
                        title_str=title_str,
                        filter_str=['xdiff_r', 'vdiff_r'])
    plt.close('all')

    #############################################################
    # Create plots for `sumlogdet_out` and `sumlogdet_proposed`
    #############################################################
    sld_dir = os.path.join(fig_dir, 'sumlogdet_plots')
    io.check_else_make_dir(sld_dir)
    traceplot_posterior(dataset, name='sumlogdet',
                        fname=fname, fig_dir=sld_dir,
                        title_str=title_str,
                        filter_str='sumlogdet')
    plt.close('all')

    ####################################################
    # Create traceplot + posterior plot of observables
    ####################################################
    var_names = ['plaqs_diffs', 'accept_prob',
                 'charges', 'tunneling_rate',
                 'dx_out', 'dx_proposed']

    traceplot_posterior(dataset, '', fname=fname, fig_dir=fig_dir,
                        title_str=title_str, filter_str=var_names)
    plt.close('all')

    #  run_data.samples_arr

    out_dir = os.path.join(fig_dir, 'angular_timeseries')
    io.check_else_make_dir(out_dir)
    plot_angle_timeseries(run_data.samples_arr,
                          out_dir=out_dir,
                          num_plots=5,
                          num_steps=1000,
                          title_str=title_str)
    plt.close('all')

    out_dir = os.path.join(fig_dir, 'plaq_sums_timeseries')
    io.check_else_make_dir(out_dir)
    lattice_shape = (params['time_size'],
                     params['space_size'], params['dim'])
    plot_plaq_timeseries(run_data.samples_arr,
                         out_dir=out_dir,
                         num_plots=5,
                         num_steps=1000,
                         title_str=title_str,
                         lattice_shape=lattice_shape)
    plt.close('all')

    #  out_file = os.path.join(fig_dir, 'run_summary.txt')
    #  run_data.log_summary(n_boot=10000,  out_file=out_file)

    return dataset, energy_dataset, fig_dir
