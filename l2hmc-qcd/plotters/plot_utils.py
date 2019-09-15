"""
plot_utils.py

Collection of helper methods used for creating plots.

Author: Sam Foreman (github: @saforem2)
Date: 08/21/2019
"""
import os

import numpy as np
import pandas as pd

import utils.file_io as io

from collections import namedtuple
from scipy.stats import multivariate_normal
from config import MARKERS, HAS_MATPLOTLIB

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

    mpl.rcParams.update(MPL_PARAMS)


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


def plot_gaussian_contours(mus, covs, **kwargs):
    """Plot contour lines for Gaussian Mixture Model w/ `mus` and `covs`.

    Args:
        mus (array-like): Array containing the means of each component (mode).
        covs (array-like): Array of covariances describing the GMM.

    Kwargs:
        passed to `plt.plot` method.

    Returns:
        plt (???)
    """
    spacing = kwargs.get('spacing', 5)
    xlims = kwargs.get('x_lims', [-4, 4])
    ylims = kwargs.get('y_lims', [-3, 3])
    res = kwargs.get('res', 100)
    cmap = kwargs.get('cmap', None)
    ax = kwargs.get('ax', None)

    X = np.linspace(xlims[0], xlims[1], res)
    Y = np.linspace(ylims[0], ylims[1], res)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    for idx, mu in enumerate(mus):
        cov = covs[idx]
        F = multivariate_normal(mu, cov)
        Z = F.pdf(pos)
        #  plt.contour(X, Y, Z, spacing, colors=colors[0])
        if cmap is None:
            if ax is None:
                plt.contour(X, Y, Z, spacing)
            else:
                ax.contour(X, Y, Z, spacing)
        else:
            if ax is None:
                plt.contour(X, Y, Z, spacing, cmap=cmap)
            else:
                ax.contour(X, Y, Z, spacing, cmap=cmap)

    return ax if ax is not None else plt


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
    num_points = kwargs.get('num_points', 2000)
    xlims = kwargs.get('xlims', None)
    ylims = kwargs.get('ylims', None)

    mus = distribution.mus
    sigmas = distribution.sigmas

    if xlims is None:
        xmin = - np.min(mus) - 5 * np.max(sigmas)
        xmax = np.max(mus) + 5 * np.max(sigmas)

        xlims = [xmin, xmax]
    if ylims is None:
        ymin = xmin
        ymax = xmax
        ylims = [ymin, ymax]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    idx = 0
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            _ = plot_gaussian_contours(mus, sigmas,
                                       xlims=xlims,
                                       ylims=ylims,
                                       res=400,
                                       ax=ax)
            _ = ax.plot(samples[:num_points, idx, 0],
                        samples[:num_points, idx, 1],
                        marker=',', ls='-',  color='gray', alpha=0.4)
            _ = ax.plot(samples[:num_points, idx, 0],
                        samples[:num_points, idx, 1],
                        marker=',', ls='', color='k', alpha=0.6)
            _ = axes[row, col].axis('equal')
            _ = axes[row, col].set_xticks([])
            _ = axes[row, col].set_yticks([])
            idx += 1

    _ = axes[0, 0].set_yticks(mus[0])
    _ = axes[0, 0].set_yticklabels([str(i) for i in mus[0]])
    _ = axes[-1, -1].set_xticks(mus[1])
    _ = axes[-1, -1].set_xticklabels([str(i) for i in mus[1]])

    if title is not None:
        _ = fig.suptitle(title)

    _ = ax.axis('equal')
    _ = fig.tight_layout()

    if out_file is not None:
        print(f'Saving figure to: {out_file}.')
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

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

    ext = kwargs.get('ext', 'pdf')
    out_file = os.path.join(figs_dir, f'plaq_diff_vs_net_weights.{ext}')
    io.log(f'Saving figure to: {out_file}.')
    plt.savefig(out_file, dpi=400, bbox_inches='tight')

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
        fig.savefig(out_file, dpi=400, bbox_inches='tight')

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
        fig.savefig(out_file, dpi=400, bbox_inches='tight')

    if ret:
        return fig, ax, axins

    return None
