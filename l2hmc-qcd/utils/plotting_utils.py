"""
plotting.py

Methods for plotting data.
"""
import matplotlib.style as mplstyle
mplstyle.use('fast')

import os

import arviz as az
import numpy as np
import xarray as xr
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as it

from utils import SKEYS
import utils.file_io as io
from utils.file_io import timeit
from utils.attr_dict import AttrDict
from utils.autocorr import calc_tau_int_vs_draws

from config import TF_FLOAT, NP_FLOAT
from dynamics.config import NetWeights

#  TF_FLOAT = FLOATS[tf.keras.backend.floatx()]
#  NP_FLOAT = NP_FLOATS[tf.keras.backend.floatx()]

COLORS = 100 * ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

plt.style.use('default')
sns.set_context('paper')
sns.set_style('whitegrid')
sns.set_palette('bright')

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


@timeit
def make_ridgeplots(dataset, num_chains=None, out_dir=None, drop_zeros=False):
    sns.set(style='white', rc={"axes.facecolor": (0, 0, 0, 0)})
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

            # Initialize the FacetGrid object
            pal = sns.color_palette('flare', n_colors=len(val.leapfrog.values))
            #  n_colors=len(val.leapfrog.values))
            #pal = sns.cubehelix_palette(len(val.leapfrog.values),
            #                            rot=-2.25, light=0.7)
            g = sns.FacetGrid(lfdf, row='lf', hue='lf', # hue_kws={'cmap': pal},
                              aspect=15, height=0.25, palette=pal)

            # Draw the densities in a few steps
            _ = g.map(sns.kdeplot, key, cut=1,
                      shade=True, alpha=0.7, linewidth=1.25)
            #  _ = g.map(sns.kdeplot, key, color='w', cut=1, lw=1.5)
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
            _ = g.despine(bottom=True, left=True)
            if out_dir is not None:
                io.check_else_make_dir(out_dir)
                out_file = os.path.join(out_dir, f'{key}_ridgeplot.pdf')
                io.log(f'Saving figure to: {out_file}.')
                plt.savefig(out_file, dpi=400, bbox_inches='tight')

            #plt.close('all')

    plt.style.use('default')
    sns.set(style='whitegrid', palette='bright', context='paper')
    fig = plt.gcf()
    ax = plt.gca()

    return fig, ax


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


@timeit
def savefig(fig, fpath):
    io.check_else_make_dir(os.path.dirname(fpath))
    io.log(f'Saving figure to: {fpath}.')
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


@timeit
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
        io.log('WARNING: `Hwf` not in `data.keys()`.', style='#ffff00')
        io.log('Not creating energy distribution plots, returning!')

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


@timeit
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


@timeit
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


@timeit
def get_title_str_from_params(params):
    """Create a formatted string with relevant params from `params`."""
    net_weights = params.get('net_weights', None)
    num_steps = params.get('num_steps', None)

    x_shape = None
    dynamics_config = params.get('dynamics_config', None)
    if dynamics_config is not None:
        x_shape = dynamics_config.get('x_shape', None)

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

    if x_shape is not None:
        title_str += f'shape: {tuple(x_shape)}'

    if net_weights == NetWeights(0., 0., 0., 0., 0., 0.):
        title_str += ', (HMC)'

    return title_str


@timeit
def mcmc_avg_lineplots(data, title=None, out_dir=None):
    """Plot trace of avg."""
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


@timeit
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


@timeit
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
    if out_dir is not None:
        out_file = os.path.join(out_dir, 'tint_vs_draws.pdf')
        io.log(f'Saving figure to: {out_file}')
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

    return tint_dict, (fig, ax)


@timeit
def plot_autocorrs1(
    beta: float,
    data: dict,
    data_compare: dict = None,
    out_dir: str = None,
    ax: plt.Axes = None,
    labels: tuple = None,
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



@timeit
def plot_data(
        data_container: "DataContainer",  # noqa:F821
        out_dir: str,
        flags: AttrDict = None,
        therm_frac: float = 0,
        params: AttrDict = None,
        hmc: bool = None,
        num_chains: int = 32,
):
    """Plot data from `data_container.data`."""
    keep_strs = [
        'charges', 'plaqs', 'accept_prob',
        'Hf_start', 'Hf_mid', 'Hf_end',
        'Hb_start', 'Hb_mid', 'Hb_end',
        'Hwf_start', 'Hwf_mid', 'Hwf_end',
        'Hwb_start', 'Hwb_mid', 'Hwb_end',
        'xeps_start', 'xeps_mid', 'xeps_end',
        'veps_start', 'veps_mid', 'veps_end'
    ]
    if num_chains > 16:
        io.log(f'Reducing `num_chains` from {num_chains} to 16 for plotting.')
        num_chains = 16

    out_dir = os.path.join(out_dir, 'plots')
    io.check_else_make_dir(out_dir)
    if hmc is None:
        if params is not None:
            hmc = params.get('hmc', False)
        hmc = False

    if hmc:
        skip_strs = ['Hw', 'ld', 'sld', 'sumlogdet']

    title = None if params is None else get_title_str_from_params(params)

    if flags is not None:
        logging_steps = flags.get('logging_steps', 1)
        flags_file = os.path.join(out_dir, 'FLAGS.z')
        if os.path.isfile(flags_file):
            train_flags = io.loadz(flags_file)
            logging_steps = train_flags.get('logging_steps', 1)
    else:
        logging_steps = 1

    data_dict = {}
    data_vars = {}
    charges_steps = []
    charges_arr = []
    plots_dir = out_dir
    #  out_dir_ = out_dir
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
            for skip_str in skip_strs:
                if skip_str in key:
                    continue

        arr = np.array(val)
        steps = logging_steps * np.arange(len(arr))

        try:
            if np.std(arr.flatten()) < 1e-2:
                continue
        except ValueError:
            pass

        arr, steps = therm_arr(arr, therm_frac=therm_frac)
        #  steps = steps[::logging_setps]
        #  steps *= logging_steps

        labels = ('MC Step', key)
        data = (steps, arr)
        if key == 'charges':
            charges_steps = steps
            charges_arr = arr

        if len(arr.shape) == 1:  # shape: (draws,)
            out_dir_ = os.path.join(plots_dir, f'mcmc_lineplots')
            io.check_else_make_dir(out_dir_)
            lplot_fname = os.path.join(out_dir_, f'{key}.pdf')
            data_dict[key] = xr.DataArray(arr, dims=['draw'], coords=[steps])
            #  _, _ = mcmc_lineplot(data, labels, title,
            #                       lplot_fname, show_avg=True)

        elif len(arr.shape) == 2:  # shape: (draws, chains)
            data_dict[key] = data
            out_dir_ = os.path.join(plots_dir, 'traceplots')
            chains = np.arange(arr.shape[1])
            data_arr = xr.DataArray(arr.T,
                                    dims=['chain', 'draw'],
                                    coords=[chains, steps])
            data_dict[key] = data_arr
            data_vars[key] = data_arr

            #  tplot_fname = os.path.join(out_dir_, f'{key}_traceplot.png')
            #  _ = mcmc_traceplot(key, data_arr, title, tplot_fname)

        #
        elif len(arr.shape) == 3:  # shape: (draws, leapfrogs, chains)
            steps_, leapfrogs_, chains_ = arr.shape
            chains = np.arange(chains_)
            leapfrogs = np.arange(leapfrogs_)
            data_dict[key] = xr.DataArray(arr.T,
                                          dims=['chain', 'leapfrog', 'draw'],
                                          coords=[chains, leapfrogs, steps])

        #  else:
        #      raise ValueError('Unexpected shape encountered in data.')

        plt.close('all')

    #  out_dir_xr = None
    #  if out_dir is not None:
    #      out_dir_xr = os.path.join(out_dir, 'xarr_plots')

    out_dir_xr = os.path.join(plots_dir, 'xarr_plots')
    data_container.plot_dataset(out_dir_xr,
                                num_chains=num_chains,
                                therm_frac=therm_frac,
                                ridgeplots=True)
    plt.close('all')
    charges = np.array(data_container.data['charges'])
    lf = flags['dynamics_config']['num_steps']
    tint_dict, _ = plot_autocorrs_vs_draws(charges, num_pts=20,
                                           nstart=1000, therm_frac=0.2,
                                           out_dir=out_dir, lf=lf)
    #  try:
    if not hmc and 'Hwf' in data_dict.keys():
        _ = plot_energy_distributions(data_dict, out_dir=out_dir, title=title)
    #  except KeyError:
    #      import pudb; pudb.set_trace()
    #      pass

    _ = mcmc_avg_lineplots(data_dict, title, out_dir)
    _ = plot_charges(charges_steps, charges_arr, out_dir=out_dir, title=title)
    plt.close('all')

    output = {
        'tint_dict': tint_dict,
        'data_container': data_container,
        'data_dict': data_dict,
        'data_vars': data_vars,
        'charges_steps': charges_steps,
        'charges_arr': charges_arr,
        'out_dir': out_dir_,
    }

    return output
