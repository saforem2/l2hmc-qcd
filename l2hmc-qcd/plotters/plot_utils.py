"""
plot_utils.py

Collection of helper methods used for creating plots.

Author: Sam Foreman (github: @saforem2)
Date: 08/21/2019
"""
import os
import pickle

from config import HAS_MATPLOTLIB, MARKERS, COLORS
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from scipy.stats import multivariate_normal

import utils.file_io as io
from lattice.lattice import u1_plaq_exact


MPL_PARAMS = {
    #  'backend': 'ps',
    #  'text.latex.preamble': [r'\usepackage{gensymb}'],
    'axes.labelsize': 14,   # fontsize for x and y labels (was 10)
    'axes.titlesize': 10,
    'legend.fontsize': 10,  # was 10
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    #  'text.usetex': True,
    #  'figure.figsize': [fig_width, fig_height],
    #  'font.family': 'serif',
}

if HAS_MATPLOTLIB:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from matplotlib.patches import Ellipse
    mpl.rcParams.update(MPL_PARAMS)

try:
    import seaborn as sns
    sns.set_palette('bright')
    sns.set_style('ticks', {'xtick.major.size': 8,
                            'ytick.major.size': 8})
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def reset_plots():
    plt.close('all')
    plt.clf()


def get_run_dirs(log_dir):
    runs_dir = os.path.join(log_dir, 'runs')
    run_dirs = [
        os.path.join(runs_dir, i) for i in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, i))
    ]
    return run_dirs


def load_pkl(pkl_file, arr=False):
    io.log(f'Loading from: {pkl_file}...')
    with open(pkl_file, 'rb') as f:
        tmp = pickle.load(f)
    if arr:
        return np.array(tmp)
    return tmp


def get_params(dirname, fname=None):
    if fname is None:
        params_file = os.path.join(dirname, 'parameters.pkl')
    else:
        params_file = os.path.join(dirname, fname)

    return load_pkl(params_file)


def load_plaqs(run_dir):
    plaqs_file = os.path.join(run_dir, 'observables', 'plaqs.pkl')
    return load_pkl(plaqs_file, arr=True)


def get_title_str(params, beta):
    ss = params['space_size']
    ts = params['time_size']
    lf_steps = params['num_steps']
    batch_size = params['batch_size']
    nw_desc = (r'nw: ($\alpha_{\mathrm{S_{x}}}$, '
               r'$\alpha_{\mathrm{T_{x}}}$, '
               r'$\alpha_{\mathrm{Q_{x}}}$, '
               r'$\alpha_{\mathrm{S_{v}}}$, '
               r'$\alpha_{\mathrm{T_{v}}}$, '
               r'$\alpha_{\mathrm{Q_{v}}}$)')

    title_str = (f"{ss} x {ts}, "
                 r"$N_{\mathrm{LF}} = $" + f"{lf_steps}, "
                 r"$N_{\mathrm{B}} = $" + f"{batch_size}, "
                 r"$\beta =$" + f"{beta:.2g}, " + f"{nw_desc}")

    return title_str


def _get_title(lf_steps, eps, batch_size, beta, nw):
    """Parse vaious parameters to make figure title when creating plots."""
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


def get_plaqs_dict(log_dir, run_dirs=None):
    """Construct plaqs_dict by averaging over all samples in batch.

    Args:
        log_dir (str): Path to log_dir containing multiple runs.

    Returns:
        plaqs_dict (dict): Dictionary containing the `plaqs_diffs` for
            each run in `log_dir`, with values computed by averaging over all
            samples in the batch for each inference step.
    """
    if run_dirs is None:
        run_dirs = get_run_dirs(log_dir)

    plaqs_dict = {}
    for rd in sorted(run_dirs):
        rp_file = os.path.join(rd, 'run_params.pkl')
        run_params = load_pkl(rp_file)
        nw = run_params['net_weights']
        exact = u1_plaq_exact(run_params['beta'])
        try:
            plaqs = load_plaqs(rd)
        except FileNotFoundError:
            continue

        plaqs = exact - load_plaqs(rd)
        try:
            nw = np.array(list(nw._asdict().values()), dtype=int)
        except AttributeError:
            nw = np.array(nw, dtype=int)

        key = tuple(list(nw))
        #  num_steps = plaqs.shape[0]
        #  therm_steps = int(0.1 * num_steps)
        #  batch_avg = np.mean(plaqs[therm_steps:, :], axis=1)
        try:
            plaqs_dict[key].append(plaqs)
        except KeyError:
            plaqs_dict[key] = [plaqs]

    return plaqs_dict


def plot_plaqs_diffs(log_dir, plaqs_dict=None):
    """Plot `plaqs_diffs` w/ hists for all run dirs in `log_dir`."""
    run_dirs = get_run_dirs(log_dir)
    if plaqs_dict is None:
        plaqs_dict = get_plaqs_dict(log_dir, run_dirs)

    nrows = len(plaqs_dict.keys())
    ncols = 2

    if HAS_SEABORN:
        sns.set_style('ticks', {'xtick.major.size': 8, 'ytick.major.size': 8})
        sns.set_palette('bright', len(plaqs_dict.keys()))
        colors = sns.color_palette()
    else:
        colors = COLORS

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(12.8, 9.6),
                             sharex='col',
                             gridspec_kw={'wspace': 0, 'hspace': 0})
    #  for idx, k in enumerate(sorted_keys):
    for idx, (k, v) in enumerate(plaqs_dict.items()):
        v = np.array(v)
        num_steps = v.shape[1]
        therm_steps = int(0.1 * num_steps)
        plaqs = v[:, therm_steps:, :]
        #  batch_avg = np.mean(plaqs, axis=-1)
        runs_avg = np.mean(plaqs, axis=(0, -1))
        runs_err = np.std(plaqs, axis=(0, -1))
        x = np.arange(therm_steps, num_steps)
        #  runs_avg = np.mean(batch_avg, axis=0)
        avg = np.mean(plaqs)
        err = np.std(plaqs)
        _ = axes[idx, 0].plot(x, runs_avg, color=colors[idx],
                              label=f'avg: {avg:.3g} +/- {err:.3g}')
        _ = axes[idx, 0].fill_between(x,
                                      y1=runs_avg + runs_err,
                                      y2=runs_avg - runs_err,
                                      color=colors[idx],
                                      alpha=0.3, ls='-', lw=0.2)

        if idx == int(nrows // 2):
            ylabel = (r"$\langle\phi_{\mathrm{P}}\rangle "
                      r"- \phi_{\mathrm{P}}^{*}$")
            _ = axes[idx, 0].set_ylabel(ylabel, fontsize='x-large')

        if HAS_SEABORN:
            _ = sns.kdeplot(plaqs.flatten(), ax=axes[idx, 1],
                            color=colors[idx], label=str(k))
        hist_kws = dict(color=colors[idx],
                        #  ec=colors[idx],
                        #  label=str(k),
                        alpha=0.3,
                        bins=50,
                        density=True,
                        histtype='stepfilled')
        _ = axes[idx, 1].hist(plaqs.flatten(), **hist_kws)

        _ = axes[idx, 0].axhline(y=0, color='gray',
                                 ls='--', zorder=-1)
        _ = axes[idx, 0].axhline(y=avg, color=colors[idx])
        _ = axes[idx, 1].axvline(x=0, color='gray', ls='--', zorder=-1)
        _ = axes[idx, 1].axvline(x=avg, color=colors[idx])
        _ = axes[idx, 0].legend(loc='best', fontsize='small')
        _ = axes[idx, 1].legend(loc='best', fontsize='small')
        _ = axes[idx, 1].set_ylabel('')
        _ = axes[idx, 1].set_yticklabels([])
        _ = axes[idx, 1].set_yticks([])
        _ = axes[idx, 0].label_outer()
        _ = axes[idx, 1].label_outer()
        _ = axes[idx, 0].set_yticks([avg])
        _ = axes[idx, 0].set_yticklabels([f'{avg:.2g}'])
        #  yticklabels = axes[idx, 0].get_yticklabels()
        #  _ = axes[idx, 0].set_yticklabels(yticklabels, fontsize='small')

    params_file = os.path.join(log_dir, 'parameters.pkl')
    params = load_pkl(params_file)
    title_str = get_title_str(params, beta=5.)
    _ = plt.suptitle(title_str, fontsize='x-large', y=1.01)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    out_file = os.path.join(log_dir, 'figures', 'plaqs_diffs.png')
    print(f'saving figure to: {out_file}...')
    plt.savefig(out_file, dpi=200, bbox_inches='tight')

    return fig, axes


def bootstrap(data, n_boot=10000, ci=68):
    boot_dist = []
    for i in range(int(n_boot)):
        resampler = np.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(np.mean(sample, axis=0))
    b = np.array(boot_dist)
    s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50. - ci / 2.)
    s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50. + ci / 2.)

    mean = np.mean(b)
    err = max(mean - s1.mean(), s2.mean() - mean)

    return mean, err, b


def tsplotboot(ax, data, **kwargs):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    cis = bootstrap(data)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kwargs)
    ax.plot(x, est, **kwargs)


def get_colors(batch_size=10, cmaps=None):
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
    spacing = kwargs.get('spacing', 5)
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
            _ = ax.contourf(X, Y, Z, spacing, cmap=cmap)
        else:
            _ = ax.contour(X, Y, Z, spacing, cmap=cmap)

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
