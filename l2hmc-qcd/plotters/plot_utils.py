"""
plot_utils.py

Collection of helper methods used for creating plots in Matplotlib.

Author: Sam Foreman (github: @saforem2)
Date: 06/16/2019
"""
import os

import numpy as np
import matplotlib as mpl
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from globals import MARKERS
import utils.file_io as io


params = {
    #  'backend': 'ps',
    #  'text.latex.preamble': [r'\usepackage{gensymb}'],
    'axes.labelsize': 14,   # fontsize for x and y labels (was 10)
    'axes.titlesize': 16,
    'legend.fontsize': 10,  # was 10
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    #  'text.usetex': True,
    #  'figure.figsize': [fig_width, fig_height],
    #  'font.family': 'serif',
}

mpl.rcParams.update(params)


def get_colors(num_samples=10, cmaps=None):
    if cmaps is None:
        cmap0 = mpl.cm.get_cmap('Greys', num_samples + 1)
        cmap1 = mpl.cm.get_cmap('Reds', num_samples + 1)
        cmap2 = mpl.cm.get_cmap('Blues', num_samples + 1)
        cmaps = (cmap0, cmap1, cmap2)

    idxs = np.linspace(0.1, 0.75, num_samples + 1)
    colors_arr = []
    for cmap in cmaps:
        colors_arr.append([cmap(i) for i in idxs])

        #  colors0 = [cmap0(i) for i in idxs]
        #  colors1 = [cmap1(i) for i in idxs]
        #  cmap0 = mpl.cm.get_cmap(cmaps[0], num_samples + 1)
        #  cmap1 = mpl.cm.get_cmap(cmaps[1], num_samples + 1)
    #  reds_cmap = mpl.cm.get_cmap('Reds', num_samples + 1)
    #  blues_cmap = mpl.cm.get_cmap('Blues', num_samples + 1)

    #  return colors0, colors1
    return colors_arr


def plot_multiple_lines(data, xy_labels, **kwargs):
    """Plot multiple lines along with their average."""
    out_file = kwargs.get('out_file', None)
    markers = kwargs.get('markers', False)
    lines = kwargs.get('lines', True)
    alpha = kwargs.get('alpha', 1.)
    legend = kwargs.get('legend', False)
    title = kwargs.get('title', None)
    ret = kwargs.get('ret', False)
    num_samples = kwargs.get('num_samples', 10)
    greys, reds, blues = get_colors(num_samples)
    if isinstance(data, list):
        data = np.array(data)

    try:
        x_data, y_data = data
    except (IndexError, ValueError):
        x_data = np.arange(data.shape[0])
        y_data = data

    x_label, y_label = xy_labels

    if y_data.shape[0] > num_samples:
        y_sample = y_data[:num_samples, :]
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
        _ = ax.plot(x_data, row,  # label=f'sample {idx}',
                    fillstyle=fillstyle, marker=marker,
                    ls=ls, lw=0.5, alpha=alpha, color=greys[idx])

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
