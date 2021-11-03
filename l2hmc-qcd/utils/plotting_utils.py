"""
plotting.py

Methods for plotting data.
"""
from __future__ import absolute_import, print_function, division, annotations
from typing import Union
import matplotlib.style as mplstyle

mplstyle.use('fast')
import itertools as it
import os
import time
import warnings
from copy import deepcopy
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import xarray as xr

import utils.file_io as io
#  from config import NP_FLOAT, TF_FLOAT
from dynamics.config import NetWeights
from utils import SKEYS
from utils.attr_dict import AttrDict
from utils.autocorr import calc_tau_int_vs_draws
#  from utils.file_io import timeit
from utils.logger import Logger, in_notebook
#  from dynamics.gauge_dynamics import GaugeDynamics

#  TF_FLOAT = FLOATS[tf.keras.backend.floatx()]
#  NP_FLOAT = NP_FLOATS[tf.keras.backend.floatx()]
logger = Logger()

COLORS = 100 * ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

plt.style.use('default')
sns.set_context('paper')
sns.set_style('whitegrid')
sns.set_palette('bright')
warnings.filterwarnings('once', 'UserWarning:')
warnings.filterwarnings('once', 'seaborn')

#  if TYPE_CHECKING:
#      from utils.data_containers import DataContainer

#  plt.ticklabel_format(scilimits=None)
#  plt.rc('text', usetex=True)
#  plt.rc('text.latex', preamble=(
#      r"""
#      \usepackage{amsmath}
#      \usepackage[sups]{XCharter}
#      \usepackage[scaled=1.04,varqu,varl]{inconsolata}
#      \usepackage[type1]{cabin}
#      \usepackage[charter,vvarbb,scaled=1.07]{newtxmath}
#      \usepackage[cal=boondoxo]{mathalfa}
#      """
#  ))
#

def truncate_colormap(
        cmap: str,
        minval:float = 0.0,
        maxval:float = 1.0,
        n:int = 100,
):
    import matplotlib as mpl
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def make_ridgeplots(
        dataset: xr.Dataset,
        num_chains: int = None,
        out_dir: str = None,
        drop_zeros: bool = False,
        cmap: str = 'viridis_r',
        default_style: dict = None,
):
    data = {}
    with sns.axes_style('white', rc={'axes.facecolor': (0, 0, 0, 0)}):
        for key, val in dataset.data_vars.items():
            if 'leapfrog' in val.coords.dims:
                lf_data = {
                    key: [],
                    'lf': [],
                }
                for lf in val.leapfrog.values:
                    # val.shape = (chain, leapfrog, draw)
                    # x.shape = (chain, draw);  selects data for a single lf
                    x = val[{'leapfrog': lf}].values
                    # if num_chains is not None, keep `num_chains` for plotting
                    if num_chains is not None:
                        x = x[:num_chains, :]

                    x = x.flatten()
                    if drop_zeros:
                        x = x[x != 0]
                    #  x = val[{'leapfrog': lf}].values.flatten()
                    lf_arr = np.array(len(x) * [f'{lf}'])
                    lf_data[key].extend(x)
                    lf_data['lf'].extend(lf_arr)

                lfdf = pd.DataFrame(lf_data)
                data[key] = lfdf

                # Initialize the FacetGrid object
                pal = sns.color_palette(cmap, n_colors=len(val.leapfrog.values))
                g = sns.FacetGrid(lfdf, row='lf', hue='lf',
                                  aspect=15, height=0.25, palette=pal)

                # Draw the densities in a few steps
                _ = g.map(sns.kdeplot, key, cut=1,
                          shade=True, alpha=0.7, linewidth=1.25)
                _ = g.map(plt.axhline, y=0, lw=1.5, alpha=0.7, clip_on=False)

                # Define and use a simple function to
                # label the plot in axes coords:
                def label(x, color, label):
                    ax = plt.gca()
                    ax.text(0, 0.10, label, fontweight='bold', color=color,
                            ha='left', va='center', transform=ax.transAxes,
                            fontsize='small')

                _ = g.map(label, key)
                # Set the subplots to overlap
                _ = g.fig.subplots_adjust(hspace=-0.75)
                # Remove the axes details that don't play well with overlap
                _ = g.set_titles('')
                _ = g.set(yticks=[])
                _ = g.set(yticklabels=[])
                _ = g.despine(bottom=True, left=True)
                if out_dir is not None:
                    io.check_else_make_dir(out_dir)
                    out_file = os.path.join(out_dir, f'{key}_ridgeplot.pdf')
                    #  logger.log(f'Saving figure to: {out_file}.')
                    plt.savefig(out_file, dpi=400, bbox_inches='tight')

            #plt.close('all')

    #  sns.set(style='whitegrid', palette='bright', context='paper')
    fig = plt.gcf()
    ax = plt.gca()

    return fig, ax, data


def set_size(
        width: float = None,
        fraction: float = 1,
        subplots: tuple = (1, 1)
):
    """Set figure dimensions to avoid scaling in LaTeX."""
    if width is None:
        width_pt = 345
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set asethetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def drop_sequential_duplicates(chain):
    if tf.is_tensor(chain):
        return tf.convert_to_tensor([i[0] for i in it.groupby(chain)])
    return np.array([i[0] for i in it.groupby(chain)])


def savefig(fig, fpath):
    io.check_else_make_dir(os.path.dirname(fpath))
    #  logger.log(f'Saving figure to: {fpath}.')
    fig.savefig(fpath, dpi=400, bbox_inches='tight')
    fig.clf()
    plt.close('all')


def therm_arr(arr, therm_frac=0., ret_steps=True):
    """Drop first `therm_frac` steps of `arr` to account for thermalization."""
    if therm_frac == 0:
        if ret_steps:
            return arr, np.arange(len(arr))
        return arr

    #  step_axis = np.argmax(arr.shape)
    step_axis = 0
    num_steps = arr.shape[step_axis]
    therm_steps = int(therm_frac * num_steps)
    arr = np.delete(arr, np.s_[:therm_steps], axis=step_axis)
    steps = np.arange(therm_steps, num_steps)

    if ret_steps:
        return arr, steps

    return arr


def plot_energy_distributions(data, out_dir=None, title=None):
    """Plot energy distributions at beginning, middle, and end of trajectory.

    NOTE: This creates side-by-side plots of the above quantities for the
    forward and backward directions.

    Returns:
        fig (plt.Figure): `plt.Figure` object
        axes (list): List of `plt.Axes` objects, flattened
    """
    try:
        chains, leapfrogs, draws = data['Hwf'].shape
    except KeyError:
        logger.warning('WARNING: `Hwf` not in `data.keys()`.', style='#ffff00')
        logger.warning('Not creating energy distribution plots, returning!')

    midpt = leapfrogs // 2
    energies_combined = {
        'forward': {
            'start': data['Hwf'][:, 0, :],
            'mid': data['Hwf'][:, midpt, :],
            'end': data['Hwf'][:, -1, :],
        },
        'backward': {
            'start': data['Hwb'][:, 0, :],
            'mid': data['Hwb'][:, midpt, :],
            'end': data['Hwb'][:, -1, :],
        }
    }

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex='col', figsize=(8, 4),
                             constrained_layout=True)
    for idx, (key, val) in enumerate(energies_combined.items()):
        ax = axes[idx]
        for k, v in val.items():
            label = f'{key}/{k}'
            _ = sns.kdeplot(v.values.flatten(), label=label, ax=ax, shade=True)

    for ax in axes:
        ax.set_ylabel('')

    _ = axes[0].set_title('forward')
    _ = axes[1].set_title('backward')
    _ = axes[0].legend(loc='best')
    _ = axes[0].set_xlabel(r"$\mathcal{H} - \sum\log\|\mathcal{J}\|$")
    _ = axes[1].set_xlabel(r"$\mathcal{H} - \sum\log\|\mathcal{J}\|$")
    if title is not None:
        _ = fig.suptitle(title)
    if out_dir is not None:
        out_file = os.path.join(out_dir, 'energy_dists_traj.pdf')
        savefig(fig, out_file)

    return fig, axes.flatten()


def energy_traceplot(key, arr, out_dir=None, title=None):
    if out_dir is not None:
        out_dir = os.path.join(out_dir, 'energy_traceplots')
        io.check_else_make_dir(out_dir)

    for idx in range(arr.shape[1]):
        arr_ = arr[:, idx, :]
        steps = np.arange(arr_.shape[0])
        chains = np.arange(arr_.shape[1])
        data_arr = xr.DataArray(arr_.T,
                                dims=['chain', 'draw'],
                                coords=[chains, steps])
        new_key = f'{key}_lf{idx}'
        if out_dir is not None:
            tplot_fname = os.path.join(out_dir,
                                       f'{new_key}_traceplot.pdf')

        _ = mcmc_traceplot(new_key, data_arr, title, tplot_fname)


def plot_charges(steps, charges, title=None, out_dir=None):
    charges = charges.T
    if charges.shape[0] > 4:
        charges = charges[:4, :]
    fig, ax = plt.subplots(constrained_layout=True)
    for idx, q in enumerate(charges):
        ax.plot(steps, np.around(q) + 5 * idx, marker='', ls='-')
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.xmargin: 0
    ax.yaxis.set_label_coords(-0.03, 0.5)
    ax.set_ylabel(r"$\mathcal{Q}$",
                  rotation='horizontal')
    ax.set_xlabel('MC Step')
    if title is not None:
        ax.set_title(title)

    if out_dir is not None:
        fpath = os.path.join(out_dir, 'charge_chains.pdf')
        savefig(fig, fpath)

    return fig, ax


def get_title_str_from_params(params):
    """Create a formatted string with relevant params from `params`."""
    nw = params.get('net_weights', None)  # type: NetWeights
    dconfig = params.get('dynamics_config', None)
    xshape, num_steps = '?', '?'
    if dconfig is not None:
        num_steps = dconfig.get('num_steps', None)
        xshape = dconfig.get('x_shape', dconfig.get('xshape', None))

    #  if num_steps is None:
    #      import pdb; pdb.set_trace()
    #  x_shape = params.get('x_shape', None)

    title_str = (r"$N_{\mathrm{LF}} = $" + f'{num_steps}, ')

    if 'beta_init' in params and 'beta_final' in params:
        beta_init = params.get('beta_init', None)
        beta_final = params.get('beta_final', None)
        title_str += (r"$\beta: $" + f'{beta_init:.3g}'
                      + r"$\rightarrow$" f'{beta_final:.3g}, ')
    elif 'beta' in params:
        beta = params.get('beta', None)
        title_str += r"$\beta = $" + f'{beta:.3g}, '

    if xshape is not None:
        title_str += f'shape: {tuple(xshape)}'

    if nw is not None:
        if nw != NetWeights(*np.ones(6)):       # if not l2hmc default
            if nw == NetWeights(*np.zeros(6)):  #   hmc?
                title_str += ', (HMC)'
            else:
                title_str += f'nw: ({", ".join([str(i) for i in nw])})'

    return title_str


def mcmc_avg_lineplots(data, title=None, out_dir=None):
    """Plot trace of avg."""
    fig, axes = None, None
    for idx, (key, val) in enumerate(data.items()):
        fig, axes = plt.subplots(ncols=2, figsize=(8, 4),
                                 constrained_layout=True)
        axes = axes.flatten()
        if len(val) == 2:
            if len(val[0].shape) > len(val[1].shape):
                arr, steps = val
            else:
                steps, arr = val
        else:
            arr = val
            steps = np.arange(arr.shape[0])

        if isinstance(arr, xr.DataArray):
            arr = arr.values

        if len(arr.shape) == 3:
            # ====
            # TODO: Create separate plots for each leapfrog?
            arr = np.mean(arr, axis=1)

        xlabel = 'MC Step'
        if len(val.shape) == 1:
            avg = arr
            ylabel = ' '.join(key.split('_'))

        else:
            avg = np.mean(arr, axis=1)
            ylabel = ' '.join(key.split('_')) + r" avg"

        _ = axes[0].plot(steps, avg, color=COLORS[idx])
        _ = axes[0].set_xlabel(xlabel)
        _ = axes[0].set_ylabel(ylabel)
        _ = sns.kdeplot(arr.flatten(), ax=axes[1],
                        color=COLORS[idx], fill=True)
        _ = axes[1].set_xlabel(ylabel)
        _ = axes[1].set_ylabel('')
        if title is not None:
            _ = fig.suptitle(title)

        if out_dir is not None:
            dir_ = os.path.join(out_dir, 'avg_lineplots')
            io.check_else_make_dir(dir_)
            fpath = os.path.join(dir_, f'{key}_avg.pdf')
            savefig(fig, fpath)

    return fig, axes


def mcmc_lineplot(data, labels, title=None,
                  fpath=None, show_avg=False, **kwargs):
    """Make a simple lineplot."""
    fig, ax = plt.subplots(constrained_layout=True)

    if show_avg:
        avg = np.mean(data[1])
        ax.axhline(y=avg, color='gray',
                   label=f'avg: {avg:.3g}',
                   ls='-', marker='')
        ax.legend(loc='best')

    ax.plot(*data, **kwargs)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if title is not None:
        ax.set_title(title)

    if fpath is not None:
        savefig(fig, fpath)

    return fig, ax


def mcmc_traceplot(key, val, title=None, fpath=None, **kwargs):
    if '_' in key:
        key = ' '.join(key.split('_'))

    try:
        az.plot_trace({key: val}, **kwargs)
        fig = plt.gcf()
        if title is not None:
            fig.suptitle(title)

        if fpath is not None:
            savefig(fig, fpath)

        return fig

    except ValueError:
        return None


def plot_autocorrs_vs_draws(
        qarr: np.ndarray,
        num_pts: int = 20,
        nstart: int = 1000,
        therm_frac: float = 0.2,
        out_dir: str = None,
        lf: int = None,
        title: str = None,
):
    tint_dict = calc_tau_int_vs_draws(qarr, num_pts, nstart, therm_frac)
    fig, ax = plt.subplots()
    y = np.mean(tint_dict['tint'], axis=-1)

    xlabel = f'MC step'
    if lf is not None:
        y *= lf
        ylabel = r'$N_{\mathrm{LF}}\cdot \tau_{\mathrm{int}}$'
    else:
        ylabel = r'$\tau_{\mathrm{int}}$'

    yerr = np.std(tint_dict['tint'], axis=-1)
    _ = ax.errorbar(tint_dict['narr'], y, yerr, marker='.', ls='')
    _ = ax.set_ylabel(ylabel)
    _ = ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)

    if out_dir is not None:
        out_file = os.path.join(out_dir, 'tint_vs_draws.pdf')
        logger.log(f'Saving figure to: {out_file}')
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

    return tint_dict, (fig, ax)


def plot_autocorrs1(
    beta: float,
    data: dict,
    data_compare: dict = None,
    ax: plt.Axes = None,
    #  out_dir: str = None,
    #  labels: tuple = None,
    cmap: str = 'Blues',
):
    if ax is None:
        fig, ax = plt.subplots()

    num_items = len(list(data.keys()))
    cmap = plt.get_cmap(cmap, num_items + 10)
    good_idxs = [0, num_items - 1, (num_items - 1) // 2]
    if data_compare is not None:
        for idx, (key, val) in enumerate(data_compare):
            if not np.isfinite(val['tint'][-1]) or max(val['tint']) > 500:
                continue

            label = None
            if beta == 7.:
                label = ('HMC, '
                         + r"$N_{\mathrm{LF}} \cdot$"
                         + r"$\varepsilon = {{{key:.2g}}}$")

            _ = ax.loglog(val['N'], val['tint'], marker='.', ls='',
                          lw=0.8,
                          alpha=0.8, color=cmap(idx+10), label=label)

    for idx, (key, val) in enumerate(data.items()):
        cond1 = max(val['N']) < 1e3
        cond2 = not np.isfinite(val['tint'][-1])
        cond3 = max(val['tint']) > 500
        if cond1 or cond2 or cond3:
            continue

        if beta == 7. and idx == 0:
            # TODO: FINISH
            pass


def get_params_from_configs(configs: dict):
    return {
        'beta_init': configs['beta_init'],
        'beta_final': configs['beta_final'],
        'x_shape': configs['dynamics_config']['x_shape'],
        'num_steps': configs['dynamics_config']['num_steps'],
        'net_weights': configs['dynamics_config']['net_weights'],
    }



def plot_data(
        data_container: DataContainer, #  "DataContainer",  # noqa:F821
        configs: dict = None,
        out_dir: str = None,
        therm_frac: float = 0,
        params: Union[dict, AttrDict] = None,
        hmc: bool = None,
        num_chains: int = 32,
        profile: bool = False,
        cmap: str = 'crest',
        verbose: bool = False,
        logging_steps: int = 1,
) -> dict:
    """Plot data from `data_container.data`."""
    if verbose:
        keep_strs = list(data_container.data.keys())

    else:
        keep_strs = [
            'charges', 'plaqs', 'accept_prob',
            'Hf_start', 'Hf_mid', 'Hf_end',
            'Hb_start', 'Hb_mid', 'Hb_end',
            'Hwf_start', 'Hwf_mid', 'Hwf_end',
            'Hwb_start', 'Hwb_mid', 'Hwb_end',
            'xeps_start', 'xeps_mid', 'xeps_end',
            'veps_start', 'veps_mid', 'veps_end'
        ]

    with_jupyter = in_notebook()
    # -- TODO: --------------------------------------
    #  * Get rid of unnecessary `params` argument,
    #    all of the entries exist in `configs`.
    # ----------------------------------------------
    if num_chains > 16:
        logger.warning(
            f'Reducing `num_chains` from {num_chains} to 16 for plotting.'
        )
        num_chains = 16

    plot_times = {}
    save_times = {}

    title = None
    if params is not None:
        try:
            title = get_title_str_from_params(params)
        except:
            title = None

    else:
        if configs is not None:
            params = {
                'beta_init': configs['beta_init'],
                'beta_final': configs['beta_final'],
                'x_shape': configs['dynamics_config']['x_shape'],
                'num_steps': configs['dynamics_config']['num_steps'],
                'net_weights': configs['dynamics_config']['net_weights'],
            }
        else:
            params = {}

    tstamp = io.get_timestamp('%Y-%m-%d-%H%M%S')
    plotdir = None
    if out_dir is not None:
        plotdir = os.path.join(out_dir, f'plots_{tstamp}')
        io.check_else_make_dir(plotdir)

    tint_data = {}
    output = {}
    if 'charges' in data_container.data:
        if configs is not None:
            lf = configs['dynamics_config']['num_steps']  # type: int
        else:
            lf = 0
        qarr = np.array(data_container.data['charges'])
        t0 = time.time()
        tint_dict, _ = plot_autocorrs_vs_draws(qarr, num_pts=20,
                                               nstart=1000, therm_frac=0.2,
                                               out_dir=plotdir, lf=lf)
        plot_times['plot_autocorrs_vs_draws'] = time.time() - t0

        tint_data = deepcopy(params)
        tint_data.update({
            'narr': tint_dict['narr'],
            'tint': tint_dict['tint'],
            'run_params': params,
        })

        run_dir = params.get('run_dir', None)
        if run_dir is not None:
            if os.path.isdir(str(Path(run_dir))):
                tint_file = os.path.join(run_dir, 'tint_data.z')
                t0 = time.time()
                io.savez(tint_data, tint_file, 'tint_data')
                save_times['tint_data'] = time.time() - t0

        t0 = time.time()
        qsteps = logging_steps * np.arange(qarr.shape[0])
        _ = plot_charges(qsteps, qarr, out_dir=plotdir, title=title)
        plot_times['plot_charges'] = time.time() - t0

        output.update({
            'tint_dict': tint_dict,
            'charges_steps': qsteps,
            'charges_arr': qarr,
        })

    hmc = params.get('hmc', False) if hmc is None else hmc

    data_dict = {}
    data_vars = {}
    #  charges_steps = []
    #  charges_arr = []
    for key, val in data_container.data.items():
        if key in SKEYS and key not in keep_strs:
            continue
        #  if key == 'x':
        #      continue
        #
        # ====
        # Conditional to skip logdet-related data
        # from being plotted if data generated from HMC run
        if hmc:
            for skip_str in ['Hw', 'ld', 'sld', 'sumlogdet']:
                if skip_str in key:
                    continue

        arr = np.array(val)
        steps = logging_steps * np.arange(len(arr))

        if therm_frac > 0:
            if logging_steps == 1:
                arr, steps = therm_arr(arr, therm_frac=therm_frac)
            else:
                drop = int(therm_frac * arr.shape[0])
                arr = arr[drop:]
                steps = steps[drop:]

        if logging_steps == 1 and therm_frac > 0:
            arr, steps = therm_arr(arr, therm_frac=therm_frac)

        #  if arr.flatten().std() < 1e-2:
        #      logger.warning(f'Skipping plot for: {key}')
        #      logger.warning(f'std({key}) = {arr.flatten().std()} < 1e-2')

        labels = ('MC Step', key)
        data = (steps, arr)

        # -- NOTE: arr.shape: (draws,) = (D,) -------------------------------
        if len(arr.shape) == 1:
            data_dict[key] = xr.DataArray(arr, dims=['draw'], coords=[steps])
            #  if verbose:
            #      plotdir_ = os.path.join(plotdir, f'mcmc_lineplots')
            #      io.check_else_make_dir(plotdir_)
            #      lplot_fname = os.path.join(plotdir_, f'{key}.pdf')
            #      _, _ = mcmc_lineplot(data, labels, title,
            #                           lplot_fname, show_avg=True)
            #      plt.close('all')

        # -- NOTE: arr.shape: (draws, chains) = (D, C) ----------------------
        elif len(arr.shape) == 2:
            data_dict[key] = data
            chains = np.arange(arr.shape[1])
            data_arr = xr.DataArray(arr.T,
                                    dims=['chain', 'draw'],
                                    coords=[chains, steps])
            data_dict[key] = data_arr
            data_vars[key] = data_arr
            #  if verbose:
            #      plotdir_ = os.path.join(plotdir, 'traceplots')
            #      tplot_fname = os.path.join(plotdir_, f'{key}_traceplot.pdf')
            #      _ = mcmc_traceplot(key, data_arr, title, tplot_fname)
            #      plt.close('all')

        # -- NOTE: arr.shape: (draws, leapfrogs, chains) = (D, L, C) ---------
        elif len(arr.shape) == 3:
            _, leapfrogs_, chains_ = arr.shape
            chains = np.arange(chains_)
            leapfrogs = np.arange(leapfrogs_)
            data_dict[key] = xr.DataArray(arr.T,  # NOTE: [arr.T] = (C, L, D)
                                          dims=['chain', 'leapfrog', 'draw'],
                                          coords=[chains, leapfrogs, steps])

    #  plotdir_xr = None
    #  if plotdir is not None:
    #      plotdir_xr = os.path.join(plotdir, 'xarr_plots')

    plotdir_xr = None
    if plotdir is not None:
        plotdir_xr = os.path.join(plotdir, 'xarr_plots')

    t0 = time.time()
    dataset, dtplot = data_container.plot_dataset(plotdir_xr,
                                                  num_chains=num_chains,
                                                  therm_frac=therm_frac,
                                                  ridgeplots=True,
                                                  cmap=cmap,
                                                  profile=profile)

    if not with_jupyter:
        plt.close('all')

    plot_times['data_container.plot_dataset'] = {'total': time.time() - t0}
    for key, val in dtplot.items():
        plot_times['data_container.plot_dataset'][key] = val

    if not hmc and 'Hwf' in data_dict.keys():
        t0 = time.time()
        _ = plot_energy_distributions(data_dict, out_dir=plotdir, title=title)
        plot_times['plot_energy_distributions'] = time.time() - t0

    t0 = time.time()
    _ = mcmc_avg_lineplots(data_dict, title, plotdir)
    plot_times['mcmc_avg_lineplots'] = time.time() - t0

    if not with_jupyter:
        plt.close('all')

    output.update({
        'data_container': data_container,
        'data_dict': data_dict,
        'data_vars': data_vars,
        'out_dir': plotdir,
        'save_times': save_times,
        'plot_times': plot_times,
    })

    return output
