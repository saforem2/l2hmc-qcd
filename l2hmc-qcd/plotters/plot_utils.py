"""
plot_utils.py

Collection of helper methods used for creating plots.

Author: Sam Foreman (github: @saforem2)
Date: 08/21/2019
"""
import os

from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from scipy.stats import multivariate_normal

import utils.file_io as io

from config import COLORS, HAS_MATPLOTLIB, MARKERS, Weights
from lattice.lattice import u1_plaq_exact
from plotters.plot_observables import calc_stats, get_obs_dict
from plotters.data_utils import calc_var_explained

if HAS_MATPLOTLIB:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from matplotlib.patches import Ellipse

    #  mpl.rcParams.update(MPL_PARAMS)
    import matplotlib.style as mplstyle

    mplstyle.use('fast')

try:
    import seaborn as sns

    sns.set_palette('bright')
    sns.set_style('ticks', {'xtick.major.size': 8,
                            'ytick.major.size': 8})
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# pylint: disable=too-many-statements, too-many-branches, too-many-arguments
# pylint: disable=invalid-name, too-many-nested-blocks, too-many-locals
def get_train_weights(params):
    """Extract the `net_weights` used for training from `params`."""
    xsw = int(params['x_scale_weight'])
    xtw = int(params['x_translation_weight'])
    xqw = int(params['x_transformation_weight'])
    vsw = int(params['v_scale_weight'])
    vtw = int(params['v_translation_weight'])
    vqw = int(params['v_transformation_weight'])
    return (xsw, xtw, xqw, vsw, vtw, vqw)


def plot_setup(log_dir, run_params, idx=None, nw_run=True):
    """Setup for plotting. Creates `filename` and `title_str`.

    Args:
        log_dir (str): Path to `log_dir`.
        run_params (RunParams): RunParams object.
        idx (int): Index.
        nw_run (bool): Whether to include net_weights used for inference in
            title_str.
    """
    params = io.loadz(os.path.join(log_dir, 'parameters.z'))
    clip_value = params.get('clip_value', 0)
    eps_fixed = params.get('eps_fixed', False)
    time_size = params.get('time_size', 0)
    space_size = params.get('space_size', 0)
    train_weights = get_train_weights(params)

    estr = f'{run_params.eps:.4g}'.replace('.', '')
    train_weights_str = ''.join((io.strf(i) for i in train_weights))
    net_weights_str = ''.join((io.strf(i) for i in run_params.net_weights))

    fname = (f'lf{run_params.num_steps}_steps{run_params.run_steps}_e{estr}')

    title_str = (f'{time_size}' + r"$\times$" + f'{space_size}, '
                 r"$N_{\mathrm{LF}} = $" + f'{run_params.num_steps}, '
                 r"$N_{\mathrm{B}} = $" + f'{run_params.batch_size}, '
                 r"$\beta = $" + f'{run_params.beta:.2g}, '
                 r"$\varepsilon = $" + f'{run_params.eps:.3g}')

    if eps_fixed:
        title_str += ' (fixed)'
        fname += '_fixed'

    if params['clip_value'] > 0:
        title_str += f', clip: {clip_value}'
        fname += f'_clip{clip_value}'.replace('.', '')

    if any([tw == 0 for tw in train_weights]):
        tws = '(' + ', '.join((str(i) for i in train_weights_str)) + ')'
        title_str += (', '
                      + r"$\mathrm{nw}_{\mathrm{train}}=$"
                      + f' {tws}')
        fname += f'_train{train_weights_str}'

    if nw_run:
        nws = '(' + ', '.join((str(i) for i in net_weights_str)) + ')'
        title_str += (', '
                      + r"$\mathrm{nw}_{\mathrm{run}}=$"
                      + f' {nws}')
        fname += f'_{net_weights_str}'

    if idx is not None:
        fname += f'_{idx}'

    return fname, title_str


def load_weights(log_dir):
    """Load weights dict from `log_dir`."""
    #  weights = io.loadz(os.path.join(log_dir, 'weights.z'))
    xweights = io.loadz(os.path.join(log_dir, 'xnet_weights.z'))
    vweights = io.loadz(os.path.join(log_dir, 'vnet_weights.z'))
    #  xweights = weights['xnet']
    #  vweights = weights['vnet']
    #  xweights = weights['xnet']['GenericNet']
    #  vweights = weights['vnet']['GenericNet']
    weights_dict = {
        'xnet': {},
        'vnet': {},
    }
    for (xk, xv), (vk, vv) in zip(xweights.items(), vweights.items()):
        if 'layer' in xk:
            W, _ = xv
            weights_dict['xnet'][xk] = W
        if 'layer' in vk:
            W, _ = vv
            weights_dict['vnet'][vk] = W

    return weights_dict


def plot_singular_values(log_dir):
    """Plot the % var explained by the singular values of `weights_dict`."""
    weights_dict = load_weights(log_dir)
    var_explained = calc_var_explained(weights_dict)
    for key, val in var_explained.items():
        x = np.arange(1, len(val)+1)
        _, ax = plt.subplots()
        ax.plot(x, val, marker='+', ls='')
        ax.set_xlabel('Singular values', fontsize=14)
        ax.set_ylabel('% Variance Explained', fontsize=14)
        ax.set_title(f'{key}', fontsize=16)
        out_dir = os.path.join(log_dir, 'svd_plots')
        io.check_else_make_dir(out_dir)
        out_file = os.path.join(out_dir, f'{key}_svd.pdf')
        io.log(f'Saving figure to: {out_file}.')
        plt.savefig(out_file, dpi=200, bbox_inches='tight')


def get_matching_log_dirs(string, root_dir):
    """Get `log_dirs` whose name contains `string` from `rot_dir`."""
    contents = os.listdir(root_dir)
    matches = [os.path.join(root_dir, i) for i in contents if string in i]
    log_dirs = []

    def check(log_dir):
        if not os.path.isdir(log_dir):
            return False
        figs_dir = os.path.join(log_dir, 'figures')
        runs_dir = os.path.join(log_dir, 'runs')
        if os.path.isdir(figs_dir) and os.path.isdir(runs_dir):
            return True
        return False

    for match in matches:
        if os.path.isdir(match):
            contents = os.listdir(match)
            log_dirs.extend([os.path.join(match, i) for i in contents
                             if check(os.path.join(match, i))])

    return log_dirs


def weights_hist(log_dir, weights=None, init=False):
    """Create histogram of entries of weight matrices."""
    if HAS_SEABORN:
        sns.set_palette('bright', 100)

    if weights is None:
        weights = io.loadz(os.path.join(log_dir, 'weights.z'))

    figs_dir = os.path.join(log_dir, 'figures', 'weights')
    io.check_else_make_dir(figs_dir)

    idx = 0
    for key, val in weights.items():
        for k1, v1 in val.items():
            for k2, v2 in v1.items():
                fig, ax = plt.subplots()
                hist_kws = {
                    'density': True,
                    'histtype': 'step',
                    'color': 'C6',
                }
                try:
                    w = v2.w.flatten()
                    b = v2.b.flatten()
                    inc = 2
                    bavg = np.mean(b)
                    berr = np.std(b)
                    blabel = (r"""$\langle b\rangle = $"""
                              + f' {bavg:.3g} +/- {berr:.3g}')
                    #  b = v2.b.flatten()
                except AttributeError:
                    w = v2.flatten()
                    b = None
                    inc = 1
                avg = np.mean(w)
                err = np.std(w)
                label = (r"""$\langle W\rangle = $"""
                         + f' {avg:.3g} +/- {err:.3g}')
                if HAS_SEABORN:
                    try:
                        sns.kdeplot(w, ax=ax, shade=True,
                                    color='C6', label=label)
                    except np.linalg.LinAlgError:
                        io.log(f'LinAlgError raised. Returning.')
                        continue
                else:
                    _ = ax.hist(w, **hist_kws)
                if b is not None:
                    label = (r"""$\langle b\rangle$"""
                             + f' {avg:.3g} +/- {err:.3g}')
                    if HAS_SEABORN:
                        try:
                            sns.kdeplot(b, ax=ax, shade=True,
                                        color='C7', label=blabel)
                        except np.linalg.LinAlgError:
                            continue
                    else:
                        _ = ax.hist(b, density=True,
                                    color='C7', histtype='step')
                _ = ax.set_title(f'{key}/{k1}/{k2}', fontsize='x-large')
                _ = ax.legend(loc='best')
                fname = f'{key}_{k1}_{k2}_weights_hist'
                if init:
                    fname += f'_init'
                fname += '.png'
                out_file = os.path.join(figs_dir, fname)
                fig.tight_layout()
                io.log(f'Saving figure to: {out_file}.')
                fig.savefig(out_file, dpi=200, bbox_inches='tight')
                idx += inc
            plt.close('all')
            plt.clf()


def reset_plots():
    """Reset (close and clear) all plots."""
    plt.close('all')
    plt.clf()


def get_params(dirname, fname=None):
    """Get `params` from `dirname`."""
    if fname is None:
        params_file = os.path.join(dirname, 'parameters.z')
    else:
        params_file = os.path.join(dirname, fname)

    return io.loadz(params_file)


def get_title_str(params, run_params=None):
    """Get descriptive title string for plot."""
    ss = params['space_size']
    ts = params['time_size']
    lf_steps = params['num_steps']
    batch_size = params['batch_size']
    clip_value = params.get('clip_value', 0)
    nw_desc = (r'($\alpha_{\mathrm{S_{x}}}$, '
               r'$\alpha_{\mathrm{T_{x}}}$, '
               r'$\alpha_{\mathrm{Q_{x}}}$, '
               r'$\alpha_{\mathrm{S_{v}}}$, '
               r'$\alpha_{\mathrm{T_{v}}}$, '
               r'$\alpha_{\mathrm{Q_{v}}}$)')

    title_str = (f"{ss} x {ts}, "
                 r"$N_{\mathrm{LF}} = $" + f"{lf_steps}, "
                 r"$N_{\mathrm{B}} = $" + f"{batch_size}, ")

    if params.get('clip_value', 0) > 0:
        title_str += f'clip: {clip_value}'

    if run_params is not None:
        title_str += get_run_title_str(run_params)

    #  if nw_legend:
        #  title_str += f"nw: {nw_desc}"

    return title_str


def get_run_title_str(run_params):
    """Parses `run_params` and returns string detailing parameters."""
    beta = run_params['beta']
    eps = run_params['eps']
    nw = tuple(run_params['net_weights'])  # pylint:disable=invalid-name
    #  run_str = run_params['run_str']

    title_str = (r"""$\beta = $""" + f'{beta}'
                 + r"""$\varepsilon = $""" + f'{eps:.3g}'
                 + f'nw: {nw}')

    zero_masks = run_params.get('zero_masks', False)
    if zero_masks:
        mask_str = r'$m^{t} = \vec{1}$'
        maskb_str = r'$\bar{m}^{t} = \vec{0}$'
        title_str += mask_str
        title_str += ', ' + maskb_str

    direction = run_params.get('direction', 'rand')
    if direction == 'forward':
        title_str += ', (forward)'
    elif direction == 'backward':
        title_str += ', (backward)'

    return title_str


def _get_title(lf_steps, eps, batch_size, beta, nw):
    """Parse various parameters to make figure title when creating plots."""
    try:
        nw_str = '[' + ', '.join([f'{i:.3g}' for i in nw]) + ']'
        title_str = (r"$N_{\mathrm{LF}} = $" + f"{lf_steps}, "
                     r"$\varepsilon = $" + f"{eps:.3g}, "
                     r"$N_{\mathrm{B}} = $" + f"{batch_size}, "
                     r"$\beta =$" + f"{beta:.2g}, "
                     r"$\mathrm{nw} = $" + nw_str)
    except ValueError:
        title_str = ''
    return title_str


def _load_obs(run_dir, obs_name):
    """Load observable."""
    run_params = io.loadz(os.path.join(run_dir, 'run_params.z'))
    obs = None
    if 'plaq' in obs_name:
        exact = u1_plaq_exact(run_params['beta'])
        try:
            pf = os.path.join(run_dir, 'observables', 'plaqs.z')
            plaqs = io.loadz(pf)
            obs = exact - np.array(plaqs)
        except FileNotFoundError:
            io.log(f'Unable to load plaquettes from {run_dir}. Returning.')
    else:
        z_file = os.path.join(run_dir, 'observables', f'{obs_name}.z')
        try:
            obs = np.array(io.loadz(z_file))
        except FileNotFoundError:
            io.log(f'Unable to load observable from {run_dir}. Returning.')

    return obs


def trace_plot(data, ax, color, **kwargs):
    """Create tracplot from data."""
    if not isinstance(data, dict):
        data = calc_stats(data, **kwargs)

    x = data.get('x', None)
    y = data.get('data', None)
    avg = data.get('avg', np.mean(y))
    err = data.get('err', np.std(y))
    avgs = data.get('avgs', np.mean(y, axis=-1))
    errs = data.get('errs', np.std(y, axis=-1))

    yp_ = avg + err
    ym_ = avg - err
    #  yp = avgs + err
    yps = avgs + errs
    #  ym = avgs - err
    yms = avgs - errs

    label = kwargs.get('label', '')
    avg_label = kwargs.get('avg_label', False)

    avg_label_ = f'{avg:.3g} +/- {err:.3g}' if avg_label else ''
    _ = ax.axhline(y=avg, color=color, label=avg_label_)

    # represent errors using semi-transparent `fill_between`
    _ = ax.plot(x, avgs, color=color, label=label)
    _ = ax.fill_between(x, y1=yps, y2=yms, color=color, alpha=0.3)

    # plot horizontal lines to show avg errors
    _ = ax.axhline(y=yp_, color=color, ls=':', alpha=0.5)
    _ = ax.axhline(y=ym_, color=color, ls=':', alpha=0.5)

    if kwargs.get('zeroline', False):  # draw horizontal line at y = 0
        _ = ax.axhline(y=0, color='gray', ls='--', zorder=-1)

    if label != '' and avg_label:
        _ = ax.legend(loc='best', fontsize='small')

    strip_yticks = kwargs.get('strip_yticks', False)
    if strip_yticks:
        #  yticks = list(ax.get_yticks())
        #  yticks.append(avg)
        #  _ = ax.set_yticks(sorted(yticks))

        _ = ax.label_outer()
        _ = ax.set_yticks([avg])
        _ = ax.set_yticklabels([f'{avg:3.2g}'])

    return ax


def kde_hist(x, **kwargs):
    """Make histogram of x using `seaborn.kdeplot` if available.

    Args:
        x (array-like): Data to be plotted

    Kwargs (optional):
        histtype (str): Histogram type. Defaults to `'stepfilled'`
        label (str): Label to use for plot in legend. Defaults to ''.
        color (str): Color to use, Defaults to gray.
        ax (matplotlib.pyplot.Axes object): Axes to use for plot. If not
            specified, uses `matplotlib.pyplot.gca()`.
        therm_frac (float): Fraction of chain length to drop to account for
            thermalization. Defaults to `0.2`.  NOTE: If `therm_percent = 0.`,
            the chain is assumed to already be thermalized.
    """
    histtype = kwargs.get('histtype', 'stepfilled')
    label = kwargs.get('key', '')
    color = kwargs.get('color', 'k')
    ax = kwargs.get('ax', None)
    therm_frac = kwargs.get('therm_frac', 0.2)

    if ax is None:
        ax = plt.gca()

    #  if not isinstance(x, dict):
    if isinstance(x, dict):
        #  xkeys = sorted(list(x.keys()))
        #  keys = sorted(['x', 'avg' 'err', 'avgs', 'errs' 'data'])
        #  if xkeys != keys:
        x = calc_stats(x, therm_frac=therm_frac)

        data = x['data']
        avg = x.get('avg', np.mean(data))
        err = x.get('err', np.std(data))

    else:
        data = np.array(x)
        avg = np.mean(data)
        #  avgs = np.mean(data.reshape((data.shape[0] -1)), axis=-1)

    if HAS_SEABORN:
        _ = sns.kdeplot(x['data'].flatten(), ax=ax, color=color,
                        label=f'{avg:.3g} +/- {err:.3g}')

    hist_kws = dict(color=color,
                    alpha=0.3,
                    bins=50,
                    density=True,
                    label=str(label),
                    histtype=histtype)
    _ = ax.hist(data.flatten(), **hist_kws)
    if kwargs.get('zeroline', False):
        _ = ax.axvline(x=0, color='gray', ls='--', zorder=-1)

    _ = ax.axvline(x=avg, color=color, ls='-')
    _ = ax.legend(loc='best', fontsize='small')
    _ = ax.set_ylabel('')
    _ = ax.set_yticklabels([])
    _ = ax.set_yticks([])
    _ = ax.label_outer()

    return ax


def _plot_obs(run_dir, obs=None, obs_name=None, **kwargs):
    """Create a traceplot + histogram for an observable."""
    filter_str = kwargs.get('filter_str', '')
    therm_frac = kwargs.get('therm_frac', 0.2)
    #  make_vline = kwargs.get('make_vline', False)
    if obs is None and obs_name is None:
        raise ValueError(f'One of `obs` or `obs_name` must be provided.')

    if obs is None:
        obs = get_obs_dict(run_dir, obs_name)

    params = io.loadz(os.path.join(run_dir, 'params.z'))
    run_params = io.loadz(os.path.join(run_dir, 'run_params.z'))
    title_str = get_title_str(params, run_params)

    fig, axes = plt.subplots(ncols=2, figsize=(12.8, 4.8))
    ps = calc_stats(obs, therm_frac=therm_frac)
    axes[0] = trace_plot(ps, axes[0], color='k')
    axes[1] = kde_hist(ps, ax=axes[1], key=params['net_weights'], color='gray')
    _ = plt.suptitle(title_str, fontsize='xx-large', y=1.04)
    plt.tight_layout()
    if obs_name == 'plaqs':
        fname = f'plaqs_diffs'
    else:
        fname = obs_name

    if filter_str != '':
        fname += f'_{filter_str}'

    log_dir = os.path.dirname(os.path.dirname(run_dir))
    run_str = params.get('run_str', None)
    fig_dir = os.path.join(log_dir, 'figures', run_str)
    io.check_else_make_dir(fig_dir)
    out_file = os.path.join(fig_dir, f'{fname}_{therm_frac}.png')
    print(f'saving figure to: {out_file}...')
    plt.savefig(out_file, dpi=200, bbox_inches='tight')

    return fig, axes


def plot_obs(log_dir, obs_dict, run_params=None,
             obs_name=None, filter_str=None, therm_frac=0.2):
    """Plot `plaqs_diffs` w/ hists for all run dirs in `log_dir`."""
    zeroline = (obs_name == 'plaqs')
    nrows = len(obs_dict.keys())
    ncols = 2

    if HAS_SEABORN:
        sns.set_style('ticks', {'xtick.major.size': 8, 'ytick.major.size': 8})
        sns.set_palette('bright', len(obs_dict.keys()))
        colors = sns.color_palette()
    else:
        colors = COLORS

    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols,
                             sharex='col',
                             figsize=(12.8, 9.6),
                             gridspec_kw={'wspace': 0.,
                                          'hspace': 0.})

    for idx, (k, v) in enumerate(obs_dict.items()):
        ps = calc_stats(v, therm_frac=therm_frac)
        axes[idx, 0] = trace_plot(ps, axes[idx, 0],
                                  colors[idx],
                                  zeroline=zeroline,
                                  strip_yticks=True)
        axes[idx, 1] = kde_hist(ps, ax=axes[idx, 1],
                                color=colors[idx],
                                key=k, zeroline=zeroline)

        if idx == int(nrows // 2):
            if obs_name == 'plaqs':
                ylabel = (r"$\langle\phi_{\mathrm{P}}\rangle "
                          r"- \phi_{\mathrm{P}}^{*}$")
            else:
                ylabel = obs_name

            _ = axes[idx, 0].set_ylabel(ylabel, fontsize='x-large')

    params_file = os.path.join(log_dir, 'parameters.z')
    params = io.loadz(params_file)
    title_str = get_title_str(params, run_params=run_params)
    _ = plt.suptitle(title_str, fontsize='xx-large', y=1.04)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.)

    out_dir = os.path.join(log_dir, 'figures')
    if obs_name == 'plaqs':
        fname = f'plaqs_diffs'
    else:
        fname = obs_name

    out_dir = os.path.join(out_dir, fname)
    io.check_else_make_dir(out_dir)

    if filter_str != '':
        fname += f'_{filter_str}'
        out_dir = os.path.join(out_dir, filter_str)

    out_file = os.path.join(out_dir, f'{fname}_{therm_frac}.png')
    print(f'saving figure to: {out_file}...')
    plt.savefig(out_file, dpi=200, bbox_inches='tight')

    return fig, axes


def tsplotboot(ax, data, **kwargs):
    """Create timeseries plot of bootstrapped data."""
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    cis = bootstrap(data)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kwargs)
    ax.plot(x, est, **kwargs)


def get_colors(batch_size=10, cmaps=None):
    """Get colors from `cmaps`."""
    if cmaps is None:
        cmap0 = mpl.cm.get_cmap('Greys', batch_size + 1)
        cmap1 = mpl.cm.get_cmap('Reds', batch_size + 1)
        cmap2 = mpl.cm.get_cmap('Blues', batch_size + 1)

        cmaps = (cmap0, cmap1, cmap2)

    idxs = np.linspace(0.1, 0.75, batch_size + 1)
    colors_arr = []
    for cmap in cmaps:
        colors_arr.append([cmap(i) for i in idxs])

    return colors_arr


def get_cmap(N=10, cmap=None):
    """Get `N` colors from `cmap`."""
    cmap_name = 'viridis' if cmap is None else cmap
    cmap_ = mpl.cm.get_cmap(cmap_name, N)
    idxs = np.linspace(0., 1., N)
    colors_arr = [cmap_(i) for i in idxs]

    return colors_arr


def _get_plaq_diff_data(log_dir):
    """Extract plaq. diff data to use in `plot_plaq_diffs_vs_net_weights`."""
    f1 = os.path.join(log_dir, 'plaq_diffs_data.txt')
    f2 = os.path.join(log_dir, 'plaq_diffs_data_orig.txt')

    if os.path.isfile(f1):
        txt_file = f1
    elif os.path.isfile(f2):
        txt_file = f2
    else:
        raise FileNotFoundError(f'Unable to locate either {f1} or {f2}.')

    data = pd.read_csv(txt_file, header=None).values

    zero_data = data[0]
    q_data = data[data[:, 2] > 0][:-1]
    t_data = data[data[:, 1] > 0][:-1]
    s_data = data[data[:, 0] > 0][:-1]
    stq_data = data[-1]

    qx, qy = q_data[:, 2], q_data[:, -1]
    tx, ty = t_data[:, 1], t_data[:, -1]
    sx, sy = s_data[:, 0], s_data[:, -1]
    x0, y0 = 0, zero_data[-1]
    stqx, stqy = 1, stq_data[-1]

    Pair = namedtuple('Pair', ['x', 'y'])
    Data = namedtuple('Data', ['q_pair', 't_pair', 's_pair',
                               'zero_pair', 'stq_pair'])

    data = Data(Pair(qx, qy), Pair(tx, ty),
                Pair(sx, sy), Pair(x0, y0),
                Pair(stqx, stqy))

    return data


def plot_acl_spectrum(acl_spectrum, **kwargs):
    """Make plot of autocorrelation spectrum.

    Args:
        acl_spectrum (array-like): Autocorrelation spectrum data.

    Returns:
        fig, ax (tuple of matplotlib `Figure`, `Axes` objects)
    """
    nx = kwargs.get('nx', None)
    label = kwargs.get('label', None)
    out_file = kwargs.get('out_file', None)

    if nx is None:
        nx = (acl_spectrum.shape[0] + 1) // 10

    xaxis = 10 * np.arange(nx)

    fig, ax = plt.subplots()
    _ = ax.plot(xaxis, np.abs(acl_spectrum[:nx]), label=label)
    _ = ax.set_xlabel('Gradient computations')
    _ = ax.set_ylabel('Auto-correlation')

    if out_file is not None:
        io.log(f'Saving figure to: {out_file}.')
        _ = plt.savefig(out_file, bbox_inches='tight')

    return fig, ax


def plot_histograms(data, labels, is_mixed=False, **kwargs):
    """Plot multiple histograms in a single figure."""
    n_bins = kwargs.get('num_bins', 50)
    title = kwargs.get('title', None)
    out_file = kwargs.get('out_file', None)
    histtype = kwargs.get('histtype', 'step')

    fig, ax = plt.subplots()
    means = []
    errs = []
    for idx, (label, x) in enumerate(zip(labels, data)):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if not is_mixed:
            steps = x.shape[1]
            therm_steps = int(0.1 * steps)
            x = x[therm_steps:, :]
        mean = x.mean()
        err = x.std()
        data_flat = x.flatten()
        label += f' avg: {mean:.4g} +/- {err:.3g}'
        kwargs = dict(ec='gray',
                      bins=n_bins,
                      density=True,
                      histtype=histtype)
        ax.hist(data_flat, **kwargs)
        means.append(mean)
        errs.append(err)

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    if out_file is not None:
        io.log(f'Saving figure to: {out_file}.')
        fig.savefig(out_file, bbox_inches='tight')

    return fig, ax, np.array(means), np.array(errs)


def plot_histogram(data, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    #  ax = ax or plt.gca()

    bins = kwargs.get('bins', 100)
    density = kwargs.get('density', True)
    stacked = kwargs.get('stacked', True)
    label = kwargs.get('label', None)
    out_file = kwargs.get('out_file', None)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)

    _ = ax.hist(data, bins=bins, density=density,
                stacked=stacked, label=label)

    if label is not None:
        _ = ax.legend(loc='best')

    if xlabel is not None:
        _ = ax.set_xlabel(xlabel)

    if ylabel is not None:
        _ = ax.set_ylabel(ylabel)

    if out_file is not None:
        io.log(f'Saving histogram plot to: {out_file}')
        _ = plt.savefig(out_file, bbox_inches='tight')

    return ax


def plot_gaussian_contours(mus, covs, ax=None, **kwargs):
    """Plot contour lines for Gaussian Mixture Model w/ `mus` and `covs`.

    Args:
        mus (array-like): Array containing the means of each component (mode).
        covs (array-like): Array of covariances describing the GMM.

    Kwargs:
        passed to `plt.plot` method.

    Returns:
        plt (???)
    """
    #  ax = ax or plt.gca()
    if ax is None:
        ax = plt.gca()

    xlims = kwargs.get('xlims', None)
    ylims = kwargs.get('ylims', None)
    res = kwargs.get('res', 100)
    cmap = kwargs.get('cmap', 'vidiris')
    #  spacing = kwargs.get('spacing', 5)
    fill = kwargs.get('fill', False)
    #  ax = kwargs.get('ax', None)

    X = np.linspace(xlims[0], xlims[1], res)
    Y = np.linspace(ylims[0], ylims[1], res)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    for idx, (mu, cov) in enumerate(zip(mus, covs)):
        F = multivariate_normal(mu, cov)
        Z = F.pdf(pos)
        #  plt.contour(X, Y, Z, spacing, colors=colors[0])
        if fill:
            _ = ax.contourf(X, Y, Z, cmap=cmap)
        else:
            _ = ax.contour(X, Y, Z, cmap=cmap)

    return ax


def draw_ellipse(position, covariance, ax=None, cmap=None, N=4, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    #  ax = ax or plt.gca()
    if ax is None:
        ax = plt.gca()

    if cmap is not None:
        color_arr = get_cmap(N=N, cmap=cmap)
    else:
        color_arr = N * ['C0']

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for idx, nsig in enumerate(range(1, N)):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle,
                             color=color_arr[idx], **kwargs))

    return ax


def get_lims(samples):
    x = samples[:, 0]
    y = samples[:, 1]
    # Define the borders
    deltaX = (max(x) - min(x))/50
    deltaY = (max(y) - min(y))/50
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    xlims = [xmin, xmax]
    ylims = [ymin, ymax]

    return xlims, ylims


def _gaussian_kde(distribution, samples):
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)

    dim = samples.shape[-1]

    samples.reshape((-1, dim))

    target_samples = distribution.get_samples(5000)
    xlims, ylims = get_lims(target_samples)
    xmin, xmax = xlims
    ymin, ymax = ylims

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    x, y = samples.T

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    z = np.reshape(kernel(positions).T, xx.shape)

    return z, (xx, yy), (xlims, ylims)


def _gmm_plot3d(distribution, samples, **kwargs):
    z, coords, lims = _gaussian_kde(distribution, samples)
    xx, yy = coords
    xlims, ylims = lims
    xmin, xmax = xlims
    ymin, ymax = ylims

    cmap = kwargs.get('cmap', 'coolwarm')
    out_file = kwargs.get('out_file', None)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    _ = ax.plot_surface(xx, yy, z, zdir='z', offset=-1., cmap=cmap)
    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit
    # on the 'walls' of the graph
    _ = ax.contour(xx, yy, z, zdir='z', offset=-0.1, cmap=cmap)
    _ = ax.contour(xx, yy, z, zdir='x', offset=xmin, cmap=cmap)
    _ = ax.contour(xx, yy, z, zdir='y', offset=ymax, cmap=cmap)
    xlim = [xmin, xmax]
    ylim = [ymin-0.05, ymax+0.05]
    _ = ax.set_xlim(xlim)
    _ = ax.set_ylim(ylim)
    _ = ax.set_xlabel('x')
    _ = ax.set_ylabel('y')

    zlim = ax.get_zlim()
    _ = ax.set_zlim((-0.1, zlim[1]))

    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')

    return fig, ax


def _gmm_plot(distribution, samples, ax=None, **kwargs):
    """
    Plot contours of target distribution overlaid with scatter plot of samples.

    Args:
        distribution (`GMM` object): Gaussian Mixture Model distribution.
            Defined in `utils/distributions.py`.'
        samples (array-like): Collection of `samples` (np.ndarray of 2-D points
            [x, y]) for scatter plot.
        Returns:
            fig, axes (output from plt.subplots(..))
    """
    #  ax = ax or plt.gca()
    cmap = kwargs.get('cmap', None)
    num_points = kwargs.get('num_points', 1000)
    ellipse = kwargs.get('ellipse', True)
    num_contours = kwargs.get('num_contours', 4)
    fill = kwargs.get('fill', False)
    title = kwargs.get('title', None)
    out_file = kwargs.get('out_file', None)
    ls = kwargs.get('ls', '-')
    line_color = kwargs.get('line_color', 'gray')
    axis_scale = kwargs.get('axis_scale', 'equal')

    #  if ellipse:
    #      lc = 'C0'
    #  else:
    #      lc = 'k'

    if ax is None:
        fig, ax = plt.subplots()

    mus = distribution.mus
    sigmas = distribution.sigmas
    pis = distribution.pis

    target_samples = distribution.get_samples(500)
    xlims, ylims = get_lims(target_samples)

    if ellipse:
        w_factor = 0.2 / np.max(pis)
        for pos, covar, w in zip(mus, sigmas, pis):
            ax = draw_ellipse(pos, covar,
                              ax=ax,
                              cmap=cmap,
                              N=num_contours,
                              alpha=w * w_factor,
                              fill=fill)
    else:
        ax = plot_gaussian_contours(mus, sigmas,
                                    xlims=xlims,
                                    ylims=ylims,
                                    ax=ax,
                                    cmap=cmap,
                                    fill=fill)

    _ = ax.plot(samples[:num_points, 0], samples[:num_points, 1],
                marker=',', ls=ls, lw=0.6, color=line_color, alpha=0.4)
    _ = ax.plot(samples[:num_points, 0], samples[:num_points, 1],
                marker=',', ls='', color='k', alpha=0.6)  # , zorder=3)
    _ = ax.plot(samples[0, 0], samples[0, 1],
                marker='X', ls='', color='r', alpha=1.,
                markersize=1.5, zorder=10)

    _ = ax.axis(axis_scale)
    _ = ax.set_xlim(xlims)
    _ = ax.set_ylim(ylims)

    if title is not None:
        _ = ax.set_title(title, fontsize=16)

    if out_file is not None:
        io.log(f'Saving figure to: {out_file}.')
        plt.savefig(out_file, bbox_inches='tight')

    return ax


def _get_ticks_labels(ax):
    xticks = ax.get_xticks()
    xticklabels = ax.get_xticklabels()
    yticks = ax.get_yticks()
    yticklabels = ax.get_yticklabels()

    return (xticks, xticklabels), (yticks, yticklabels)


def gmm_plot(distribution, samples, **kwargs):
    """
    Plot contours of target distribution overlaid with scatter plot of samples.

    Args:
        distribution (`GMM` object): Gaussian Mixture Model distribution.
            Defined in `utils/distributions.py`.'
        samples (array-like): Collection of `samples` (np.ndarray of 2-D points
            [x, y]) for scatter plot.
        Returns:
            fig, axes (output from plt.subplots(..))
    """
    nrows = kwargs.get('nrows', 3)
    ncols = kwargs.get('ncols', 3)
    out_file = kwargs.get('out_file', None)
    title = kwargs.get('title', None)
    cmap = kwargs.get('cmap', None)
    num_points = kwargs.get('num_points', 2000)
    ellipse = kwargs.get('ellipse', True)
    num_contours = kwargs.get('num_contours', 4)
    axis_scale = kwargs.get('axis_scale', 'equal')

    if ellipse:
        lc = 'C0'
    else:
        lc = 'gray'

    mus = distribution.mus
    sigmas = distribution.sigmas
    pis = distribution.pis
    target_samples = distribution.get_samples(5000)
    xlims, ylims = get_lims(target_samples)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    idx = 0
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            #  xlims, ylims = get_lims(samples[:, idx])
            if ellipse:
                w_factor = 0.2 / np.max(pis)
                for pos, covar, w in zip(mus, sigmas, pis):
                    _ = draw_ellipse(pos, covar,
                                     ax=ax,
                                     cmap=cmap,
                                     N=num_contours,
                                     alpha=w * w_factor,
                                     fill=kwargs.get('fill', False))
            else:
                _ = plot_gaussian_contours(mus, sigmas,
                                           xlims=xlims,
                                           ylims=ylims,
                                           ax=ax, cmap=cmap)

            _ = ax.plot(samples[:num_points, idx, 0],
                        samples[:num_points, idx, 1],
                        marker=',', ls='-', lw=0.6,
                        color=lc, alpha=0.4, zorder=2)
            _ = ax.plot(samples[:num_points, idx, 0],
                        samples[:num_points, idx, 1],
                        marker=',', ls='', color='k', alpha=0.6, zorder=2)
            _ = ax.plot(samples[0, idx, 0], samples[0, idx, 1],
                        marker='X', ls='', color='r',
                        alpha=1., markersize=1.5, zorder=10)
            _ = ax.axis(axis_scale)

            #  xtl, ytl = _get_ticks_labels(ax)
            if ax != axes[-1, 0]:     # Keep ticks/labels on lower left subplot
                _ = ax.set_xticks([])
                _ = ax.set_yticks([])
                _ = ax.set_xlim(xlims)
                _ = ax.set_ylim(ylims)

            idx += 1

    #  xticks, xticklabels = xtl
    #  yticks, yticklabels = ytl

    #  _ = axes[-1, 0].set_yticks(yticks)
    #  _ = axes[-1, 0].set_yticklabels(yticklabels)
    #  _ = axes[-1, 0].set_xticks(xticks)
    #  _ = axes[-1, 0].set_xticklabels(xticklabels)
    #  _ = axes[-1, 0].axis(axis_scale)
    #
    # _ = fig.tight_layout()

    if title is not None:
        _ = fig.suptitle(title)

    if out_file is not None:
        print(f'Saving figure to: {out_file}.')
        plt.savefig(out_file, bbox_inches='tight')

    return fig, axes


def plot_plaq_diffs_vs_net_weights(log_dir, **kwargs):
    """Plot avg. plaq diff. vs net_weights.

    Args:
        log_dir (str): Location of `log_dir` containing `plaq_diffs_data.txt`
            file containing avg. plaq_diff data.

    Kwargs:
        passed to ax.plot

    Returns:
        fig, ax: matplotlib Figure and Axes instances.
    """
    plaq_diff_data = _get_plaq_diff_data(log_dir)
    qx, qy = plaq_diff_data.q_pair
    tx, ty = plaq_diff_data.t_pair
    sx, sy = plaq_diff_data.s_pair
    x0, y0 = plaq_diff_data.zero_pair
    stqx, stqy = plaq_diff_data.stq_pair

    fig, ax = plt.subplots()
    ax.plot(qx, qy, label='Transformation (Q) fn', marker='.')
    ax.plot(tx, ty, label='Translation (T) fn', marker='.')
    ax.plot(sx, sy, label='Scale (S) fn', marker='.')
    ax.plot(0, y0, label='S, T, Q = 0', marker='s')
    ax.plot(1, stqy, label='S, T, Q = 1', marker='v')
    ax.set_xlabel('Net weight', fontsize=14)
    ax.set_ylabel('Avg. plaq. difference', fontsize=14)
    ax.legend(loc='best')
    plt.tight_layout()

    figs_dir = os.path.join(log_dir, 'figures')
    io.check_else_make_dir(figs_dir)

    ext = kwargs.get('ext', 'png')
    out_file = os.path.join(figs_dir, f'plaq_diff_vs_net_weights.{ext}')
    io.log(f'Saving figure to: {out_file}.')
    plt.savefig(out_file, bbox_inches='tight')

    return fig, ax


def plot_multiple_lines(data, xy_labels, **kwargs):
    """Plot multiple lines along with their average."""
    out_file = kwargs.get('out_file', None)
    markers = kwargs.get('markers', False)
    lines = kwargs.get('lines', True)
    alpha = kwargs.get('alpha', 1.)
    legend = kwargs.get('legend', False)
    title = kwargs.get('title', None)
    ret = kwargs.get('ret', False)
    batch_size = kwargs.get('batch_size', 10)
    colors_arr = get_colors(batch_size)
    greys = colors_arr[0]
    #  reds = colors_arr[1]
    #  blues = colors_arr[2]
    if isinstance(data, list):
        data = np.array(data)

    try:
        x_data, y_data = data
    except (IndexError, ValueError):
        x_data = np.arange(data.shape[0])
        y_data = data

    x_label, y_label = xy_labels

    if y_data.shape[0] > batch_size:
        y_sample = y_data[:batch_size, :]
    else:
        y_sample = y_data

    fig, ax = plt.subplots()

    marker = None
    ls = '-'
    fillstyle = 'full'
    for idx, row in enumerate(y_sample):
        if markers:
            marker = MARKERS[idx]
            fillstyle = 'none'
            ls = '-'
        if not lines:
            ls = ''
        _ = ax.plot(x_data, row, label=f'sample {idx}', fillstyle=fillstyle,
                    marker=marker, ls=ls, alpha=alpha, lw=0.5,
                    color=greys[idx])

    _ = ax.plot(
        x_data, y_data.mean(axis=0), label='average',
        alpha=1., lw=1.0, color='k'
    )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    plt.tight_layout()
    if legend:
        ax.legend(loc='best')
    if title is not None:
        ax.set_title(title)
    if out_file is not None:
        out_dir = os.path.dirname(out_file)
        io.check_else_make_dir(out_dir)
        io.log(f'Saving figure to {out_file}.')
        fig.savefig(out_file, bbox_inches='tight')

    if ret:
        return fig, ax

    return 1


def plot_with_inset(data, labels=None, **kwargs):
    """Make plot with zoomed inset."""
    out_file = kwargs.get('out_file', None)
    markers = kwargs.get('markers', False)
    lines = kwargs.get('lines', True)
    alpha = kwargs.get('alpha', 7.)
    legend = kwargs.get('legend', False)
    title = kwargs.get('title', None)
    lw = kwargs.get('lw', 1.)
    ret = kwargs.get('ret', False)
    #  data_lims = kwargs.get('data_lims', None)

    color = kwargs.get('color', 'C0')
    bounds = kwargs.get('bounds', [0.2, 0.2, 0.7, 0.3])
    dx = kwargs.get('dx', 100)

    plt_label = labels.get('plt_label', None)
    x_label = labels.get('x_label', None)
    y_label = labels.get('y_label', None)
    #  batch_size = kwargs.get('batch_size', 10)
    #  greys, reds, blues = get_colors(batch_size)
    if isinstance(data, list):
        data = np.array(data)

    if isinstance(data, (tuple, np.ndarray)):
        if len(data) == 1:
            x = np.arange(data.shape[0])
            y = data
        if len(data) == 2:
            x, y = data
        elif len(data) == 3:
            x, y, yerr = data

    marker = None
    ls = '-'
    fillstyle = 'full'
    if markers:
        marker = 'o'
        fillstyle = 'none'

    if not lines:
        ls = ''

    fig, ax = plt.subplots()
    if yerr is None:
        _ = ax.plot(x, y, label=plt_label,
                    marker=marker, fillstyle=fillstyle,
                    ls=ls, alpha=alpha, lw=lw, color=color)
    else:
        _ = ax.errorbar(x, y, yerr=yerr, label=plt_label,
                        marker=marker, fillstyle=fillstyle,
                        ls=ls, alpha=alpha, lw=lw, color=color)

    axins = ax.inset_axes(bounds)

    mid_idx = len(x) // 2
    idx0 = mid_idx - dx
    idx1 = mid_idx + dx
    skip = 10

    _x = x[idx0:idx1:skip]
    _y = y[idx0:idx1:skip]
    if yerr is not None:
        _yerr = yerr[idx0:idx1:skip]
        _ymax = max(_y + abs(_yerr))
        _ymax += 0.1 * _ymax
        _ymin = min(_y - abs(_yerr))
        _ymin -= 0.1 * _y
        axins.errorbar(_x, _y, yerr=_yerr, label='',
                       marker=marker, fillstyle=fillstyle,
                       ls=ls, alpha=alpha, lw=lw, color=color)
    else:
        _ymax = max(_y)
        _ymax += 0.1 * _ymax
        _ymin = min(_y)
        _ymin -= 0.1 * _ymin
        axins.plot(_x, _y, label='',
                   marker=marker, fillstyle=fillstyle,
                   ls=ls, alpha=alpha, lw=lw, color=color)

    axins.indicate_inset_zoom(axins, label='')
    axins.xaxis.get_major_locator().set_params(nbins=3)

    _ = ax.set_xlabel(x_label, fontsize=14)
    _ = ax.set_ylabel(y_label, fontsize=14)
    _ = plt.tight_layout()
    if legend:
        _ = ax.legend(loc='best')
    if title is not None:
        _ = ax.set_title(title)
    if out_file is not None:
        out_dir = os.path.dirname(out_file)
        io.check_else_make_dir(out_dir)
        io.log(f'Saving figure to {out_file}.')
        fig.savefig(out_file, bbox_inches='tight')

    if ret:
        return fig, ax, axins

    return None
