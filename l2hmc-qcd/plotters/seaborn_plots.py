"""
seaborn_plots.py

Collection of functions for making figures with `seaborn`.

Author: Sam Foreman
Date: 01/04/2020
"""
import os
import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.style as mplstyle

import utils.file_io as io

from config import COLORS, PROJECT_DIR
from utils.file_io import timeit
from lattice.lattice import u1_plaq_exact
from plotters.data_utils import bootstrap

#  sns.set_palette('bright')
plt.style.use('seaborn-bright')
mplstyle.use('fast')
#  ticklabelsize = 14
DEFAULT_TICKLABELSIZE = mpl.rcParams['xtick.labelsize']

# pylint:disable=invalid-name
# pylint: disable=too-many-locals


def infer_cmap(color, palette='bright'):
    """Infer cmap from `color`."""
    hues = sns.color_palette(palette)
    if color == hues[0]:
        return sns.light_palette(hues[0], 12, as_cmap=True)
    if color == hues[1]:
        return sns.light_palette(hues[1], 12, as_cmap=True)
    if color == hues[2]:
        return sns.light_palette(hues[2], 12, as_cmap=True)
    if color == hues[3]:
        return sns.light_palette(hues[3], 12, as_cmap=True)
    if color == hues[4]:
        return sns.light_palette(hues[4], 12, as_cmap=True)
    if color == hues[5]:
        return sns.light_palette(hues[5], 12, as_cmap=True)


def kde_color_plot(x, y, **kwargs):
    """Create `kde_color_plot`. Returns `plt.Axes` object."""
    palette = kwargs.pop('palette', 'bright')
    cmap = infer_cmap(kwargs['color'], palette=palette)
    ax = sns.kdeplot(x, y, cmap=cmap, **kwargs)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    #  sns.despine(ax=ax, bottom=True, left=True)
    return ax


def _kdeplot(x, y, **kwargs):
    ax = sns.kdeplot(x, y, **kwargs)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))

    return ax


def kde_diag_plot(x, **kwargs):
    """Create KDE histogram along diagonal for `sns.GridPlot`."""
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    _ = sns.kdeplot(x, **kwargs)
    plt.xticks(ticks=ax.get_xticks(), rotation=30, ha='center', va='top')
    return ax


def plot_pts(x, y, **kwargs):
    """Make scatter plot of points."""
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    #  if len(x) > 5000:
    #      x = x[:5000]
    #  if len(y) > 5000:
    #      y = y[:5000]
    _ = ax.plot(x, y, **kwargs)
    plt.xticks(ticks=ax.get_xticks(), rotation=30, ha='center', va='top')
    return ax


def calc_tunneling_rate(charges):
    """Calculate the tunneling rate as Q_{i+1} - Q_{i}."""
    if not isinstance(charges, np.ndarray):
        charges = np.array(charges)

    if charges.shape[0] > charges.shape[1]:
        charges = charges.T

    charges = np.around(charges)
    dq = np.abs(charges[:, 1:] - charges[:, :-1])

    return dq

def plot_setup(log_dir, run_params, idx=None, nw_run=True):
    """Setup for plotting. Creates `filename` and `title_str`."""
    params = io.loadz(os.path.join(log_dir, 'parameters.z'))
    lf = params['num_steps']
    clip_value = params.get('clip_value', 0)
    eps_fixed = params.get('eps_fixed', False)
    train_weights = get_train_weights(params)
    train_weights_str = ''.join((io.strf(i) for i in train_weights))
    net_weights = run_params['net_weights']
    net_weights_str = ''.join((io.strf(i) for i in net_weights))
    #  net_weights_str = ''.join(
    #      (io.strf(i).replace('.', '') for i in net_weights)
    #  )

    date_str = log_dir.split('/')[-2]
    y, m, d = date_str.split('_')
    y = int(y)
    m = int(m)
    d = int(d)
    old_dx = True
    if y == 2020 and m >= 1 and d >= 4:
        old_dx = False

    beta = run_params['beta']
    eps = run_params['eps']
    fname = f'lf{lf}'
    run_steps = run_params['run_steps']
    fname += f'_steps{run_steps}'
    title_str = (r"$N_{\mathrm{LF}} = $" + f'{lf}, '
                 r"$\beta = $" + f'{beta:.1g}, '
                 r"$\varepsilon = $" + f'{eps:.3g}')
    eps_str = f'{eps:.4g}'.replace('.', '')
    fname += f'_e{eps_str}'

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

    return fname, title_str, old_dx


def get_lf(log_dir):
    params = io.loadz(os.path.join(log_dir, 'parameters.z'))
    lf = params['num_steps']
    return lf


def get_observables(run_dir,
                    log_dir=None,
                    n_boot=5000,
                    therm_frac=0.25,
                    nw_include=None,
                    calc_stats=True):
    """Build `pd.DataFrame` containing inference data in `run_dir`.

    Args:
        run_dir: Path to directory containing inference data.
        log_dir (optional): Path to directory containing `run_dir` to load
            from. If included, this will be included as an additional column in
            the resulting dataframe which is useful for keeping track of where
            the inference data came from.
        n_boot (int): Number of bootstrap iterations to use when calculating
            statistics.
        therm_frac (float): Percent of steps to drop for thermalization.
        nw_include (array-like, optional): List or array of `net_weights` to
            include when loading inference data. If `None`, all values of
            `net_weights` will be included. Otherwise, only those included in
            `nw_include` will be counted.
        calc_stats (bool, optional): If True, statistics will be computed using
            bootstrap resampling.

    Returns:
        data (pd.DataFrame): DataFrame containing (flattened) inference data.
        data_bs (pd.DataFrame): DataFrame containing (bootstrapped) inference
            data.
        run_params (dict): Dictionary containing the parameters used for
            inference run.
    """
    run_params = io.loadz(os.path.join(run_dir, 'run_params.z'))
    net_weights = tuple([io.strf(i) for i in run_params['net_weights']])
    #  eps = run_params['eps']
    beta = run_params['beta']
    observables_dir = os.path.join(run_dir, 'observables')
    px = io.loadz(os.path.join(observables_dir, 'accept_prob.z'))
    px = np.squeeze(np.array(px))
    avg_px = np.mean(px)

    if nw_include is not None:
        keep_data = net_weights in nw_include
    else:
        keep_data = True

    if avg_px < 0.1: #or not keep_data:
        print(f'INFO:Skipping! nw: {net_weights}, avg_px: {avg_px:.3g}')
        return None, None, run_params

    io.log(f'Loading data for net_weights: {net_weights}...')
    io.log(f'  run_dir: {run_dir}')

    def load_sqz(fname):
        data = io.loadz(os.path.join(observables_dir, fname))
        return np.squeeze(np.array(data))

    charges = load_sqz('charges.z')
    dx_out = load_sqz('dx_out.z')
    dx_prop = load_sqz('dx_proposed.z')
    plaqs = load_sqz('plaqs.z')
    dplq = u1_plaq_exact(beta) - plaqs

    num_steps = px.shape[0]
    therm_steps = int(therm_frac * num_steps)
    #  steps = np.arange(therm_steps, num_steps)
    # NOTE: Since the number of tunneling events is computed as
    # `dq = charges[1:] - charges[:-1]`,
    # we have that
    # `dq.shape[0] = num_steps - 1`.
    # Because of this, we drop the first step of `px` and `dplq`
    # to enforce that they all have the same shape.
    px = px[therm_steps:]
    dplq = dplq[therm_steps:]
    charges = np.insert(charges, 0, 0, axis=0)
    charges = charges[therm_steps:]
    dq = calc_tunneling_rate(charges)
    dq = dq.T

    def get_stats(arr, axis=0):
        avg, err, arr_ = bootstrap(arr, n_boot=n_boot)
        return arr_.mean(axis=axis).flatten(), arr_.std(axis=axis).flatten()

    if calc_stats:
        px_, px_err_ = get_stats(px, axis=0)
        dplq_, dplq_err_ = get_stats(dplq, axis=0)
        dq_, dq_err_ = get_stats(dq, axis=0)
        dx_out_, dx_out_err_ = get_stats(dx_out, axis=0)
        dx_prop_, dx_prop_err_ = get_stats(dx_prop, axis=0)

        entries = len(dq_.flatten())
        data_bs = pd.DataFrame({
            'plaqs_diffs': dplq_,
            'plaqs_diffs_err': dplq_err_,
            'accept_prob': px_,
            'accept_prob_err': px_err_,
            'dx_out': dx_out_,
            'dx_out_err': dx_out_err_,
            'dx_prop': dx_prop_,
            'dx_prop_err': dx_prop_err_,
            'tunneling_rate': dq_,
            'tunneling_rate_err': dq_err_,
            'net_weights': tuple([net_weights for _ in range(entries)]),
            'run_dir': np.array([run_dir for _ in range(entries)]),
            'log_dir': np.array([log_dir for _ in range(entries)]),
        })

    else:
        data_bs = None

    entries = len(dq.flatten())
    data = pd.DataFrame({
        'plaqs_diffs': dplq.flatten(),
        'accept_prob': px.flatten(),
        'tunneling_rate': dq.flatten(),
        'dx_out': dx_out.flatten(),
        'dx_proposed': dx_prop.flatten(),
        'net_weights': tuple([net_weights for _ in range(entries)]),
        'run_dir': np.array([run_dir for _ in range(entries)]),
        'log_dir': np.array([log_dir for _ in range(entries)]),
    })

    return data, data_bs, run_params


def _build_dataframes(run_dirs, data=None, data_bs=None, **kwargs):
    """Build `pd.DataFrames` containing all inference data from `run_dirs`.

    Args:
        run_dirs (array-like): List of run_dirs in which to look for inference
            data.
        data (pd.DataFrame): DataFrame containing inference data. If `data is
            not None`, the new `pd.DataFrame` will be appended to `data`.
        data_bs (pd.DataFrame): DataFrame containing bootstrapped inference
            data. If `data_bs is not None`, the new `pd.DataFrame` will be
            appended to `data_bs`.
    Kwargs:
        Passed to `get_observables`.

    Returns:
        data (pd.DataFrame): DataFrame containing (flattened) inference data.
        data_bs (pd.DataFrame): DataFrame containing (bootstrapped) inference
            data.
        run_params (dict): Dictionary of parameters used to generate inference
            data.
    """
    run_params = None
    for run_dir in run_dirs:
        if data is not None and hasattr(data, 'run_dir'):
            if not data[data.run_dir == run_dir].empty:
                continue
        run_params_file = os.path.join(run_dir, 'run_params.z')
        if os.path.isfile(run_params_file):
            new_df, new_df_bs, run_params = get_observables(run_dir, **kwargs)
            if data is None:
                data = new_df
            else:
                data = pd.concat(
                    [data, new_df], axis=0
                ).reset_index(drop=True)

            if data_bs is None:
                data_bs = new_df_bs
            else:
                data_bs = pd.concat(
                    [data_bs, new_df_bs], axis=0
                ).reset_index(drop=True)
        else:
            run_params = None
            continue

    return data, data_bs, run_params


def build_dataframes(log_dirs,
                     df_dict=None,
                     df_bs_dict=None,
                     rp_dict=None,
                     filter_str=None,
                     runs_np=True):
    if df_dict is None:
        df_dict = {}
    if df_bs_dict is None:
        df_bs_dict = {}
    if rp_dict is None:
        rp_dict = {}

    runs_str = 'runs_np' if runs_np else 'runs'

    #  log_dirs = sorted(log_dirs, key=get_lf, reverse=True)
    for idx, log_dir in enumerate(log_dirs):
        if log_dir in df_dict.keys():
            continue
        if log_dir in df_bs_dict.keys():
            continue
        if log_dir in rp_dict.keys():
            continue
        else:
            data = None
            data_bs = None
            n_boot = 5000
            frac = 0.25
            try:
                run_dirs = io.get_run_dirs(log_dir,
                                           filter_str=filter_str,
                                           runs_str=runs_str)
            except FileNotFoundError:
                continue


            if run_dirs is not None:
                run_dirs = sorted(run_dirs)[::-1]

                nw_inc = [
                    (0, 0, 0, 0, 0, 0),
                    (1, 0, 1, 1, 0, 1),
                    (1, 0, 1, 1, 1, 1),
                    (1, 1, 1, 1, 0, 1),
                    (1, 1, 1, 1, 1, 1),
                ]

                outputs = _build_dataframes(run_dirs,
                                            data=data,
                                            log_dir=log_dir,
                                            data_bs=data_bs,
                                            n_boot=n_boot,
                                            calc_stats=True,
                                            therm_frac=frac,
                                            nw_include=nw_inc)
                data, data_bs, run_params = outputs

                df_dict[log_dir] = data
                df_bs_dict[log_dir] = data_bs
                rp_dict[log_dir] = run_params

    return df_dict, df_bs_dict, rp_dict


@timeit
def _gridplots_combined(g, **kwargs):
    """Makde combined gridplots."""
    g = g.map_diag(kde_diag_plot, shade=True)
    upper_map = kwargs.get('upper_map', None)
    if upper_map == 'scatter':
        g = g.map_upper(plot_pts, ls='', marker='+',
                        markeredgewidth=0.4, rasterized=True)
        marker = 'o'
    else:
        gridsize = kwargs.get('gridsize', 50)
        g = g.map_upper(kde_color_plot, shade=False, gridsize=gridsize)

    marker = kwargs.get('marker', 'o')
    alpha = kwargs.get('alpha', 0.4)
    markeredgewidth = kwargs.get('markeredgewidth', 1.)
    g = g.map_lower(plot_pts, ls='', marker=marker,
                    markeredgewidth=markeredgewidth,
                    alpha=alpha, rasterized=True)

    return g

@timeit
def _gridplots_separated(g, **kwargs):
    """Create separated gridplots."""
    color = kwargs.get('color', 'C0')
    cmap = sns.light_palette(color, as_cmap=True)
    g = g.map_diag(kde_diag_plot, shade=True, color=color)
    upper_map = kwargs.get('upper_map', None)
    if upper_map == 'scatter':
        g = g.map_upper(plot_pts, ls='', marker='+',
                        markeredgewidth=0.4, rasterized=True)
        marker = 'o'
    else:
        gridsize = kwargs.get('gridsize', 100)
        g = g.map_upper(sns.kdeplot, cmap=cmap, shade=True,
                        gridsize=gridsize, shade_lowest=False)

    alpha = kwargs.get('alpha', 0.4)
    marker = kwargs.get('marker', 'o')
    g = g.map_lower(plot_pts, color=color, ls='',
                    marker=marker, rasterized=True, alpha=alpha)

    return g


@timeit
def _gridplots(log_dir, data, title_str, fname,
               combined=False, out_dir=None, **kwargs):
    """Make gridplot using `sns.PairGrid`."""
    _vars = ['plaqs_diffs', 'accept_prob', 'tunneling_rate']
    if hasattr(data, 'dx'):
        _vars += ['dx']

    upper_map = kwargs.get('upper_map', None)

    g = sns.PairGrid(data,
                     hue='net_weights',
                     palette='bright',
                     diag_sharey=False,
                     vars=_vars)

    if combined:
        g = _gridplots_combined(g, **kwargs)
    else:
        g = _gridplots_separated(g, **kwargs)

    g.add_legend()
    g.fig.suptitle(title_str, y=1.02, fontsize='x-large')
    if out_dir is None:
        out_dir = os.path.abspath('../../gridplots')
        io.check_else_make_dir(out_dir)

    out_file = os.path.join(out_dir, f'{fname}.pdf')
    logdir_id = log_dir.split('/')[-1].split('_')[-1]
    #  if os.path.isfile(out_file):

        #  timestr = io.get_timestr()
        #  hour_str = timestr['hour_str']
        #  id_str = f'{logdir_id}'
        #  out_file = os.path.join(out_dir, f'{fname}_{id_str}_{hour_str}.pdf')

    io.log(f'INFO: Saving figure to: {out_file}')
    plt.savefig(out_file, bbox_inches='tight')
    if not os.path.isfile(out_file):
        plt.savefig(out_file, bbox_inches='tight')

    return g


@timeit
def gridplots(log_dirs,
              df_dict=None,
              df_bs_dict=None,
              rp_dict=None,
              rootdir=None):
    """Make gridplots for each run_dir in log_dirs."""
    #  now = datetime.datetime.now()
    #  day_str = now.strftime('%Y_%m_%d')
    #  hour_str = now.strftime('%H%M')
    #  time_str = f'{day_str}_{hour_str}'
    if rootdir is None:
        #  proj_dir = os.path.dirname(PROJECT_DIR)
        rootdir = os.path.abspath(
            '/Users/saforem2/ANL/l2hmc-qcd/gauge_logs/gridplots'
        )
        #  rootdir = os.path.join(proj_dir, 'cooley_figures')
        #  rootdir = os.path.abspath(f'/home/foremans/cooley_figures/')

    io.check_else_make_dir(rootdir)

    #  ticklabelsize = DEFAULT_TICKLABELSIZE
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['axes.formatter.min_exponent'] = 2

    if df_dict is None and df_bs_dict is None and rp_dict is None:
        df_dict, df_bs_dict, rp_dict = build_dataframes(log_dirs)

    colors = 100 * [*COLORS]
    keys = list(rp_dict.keys())[::-1]
    for idx, log_dir in enumerate(keys):
        io.log(f'log_dir: {log_dir}')
        run_params = rp_dict[log_dir]
        if run_params is None:
            continue

        data = df_dict[log_dir] if df_dict is not None else None
        data_bs = df_bs_dict[log_dir] if df_bs_dict is not None else None
        io.check_else_make_dir(rootdir)

        color = colors[idx]

        fname, title_str, old_dx = plot_setup(log_dir, run_params, idx=idx)

        if data is not None:
            out_dir = os.path.join(rootdir,
                                   f'combined_pairplots')
            io.check_else_make_dir(out_dir)
            #  try:
            g_combined = _gridplots(log_dir, data,
                                    title_str, fname,
                                    color=None, combined=True,
                                    marker='x', markeredgewidth=0.4,
                                    upper_map='scatter', gridsize=50,
                                    out_dir=out_dir)
            #  except UnboundLocalError:
            #      io.log(f'Unable to create _gridplots for {log_dir}.')
            #      continue
        if data_bs is not None:
            out_dir = os.path.join(rootdir,
                                   f'combined_pairplots_boostrap')
            io.check_else_make_dir(out_dir)
            #  try:
            g_combined = _gridplots(log_dir, data_bs,
                                    title_str, fname,
                                    color=None,
                                    combined=True,
                                    out_dir=out_dir)
            #  except UnboundLocalError:
            #      io.log(f'Unable to create _gridplots for {log_dir}.')
            #      continue

        run_dirs = []
        if data is not None:
            run_dirs += list(np.unique(data['run_dir']))
        if data_bs is not None:
            run_dirs += list(np.unique(data_bs['run_dir']))
        run_dirs = np.unique(run_dirs)
        for run_dir in run_dirs:
            run_params = io.loadz(os.path.join(run_dir, 'run_params.z'))
            fname, title_str, old_dx = plot_setup(log_dir, run_params)

            if data is not None:
                data_ = data[data.run_dir == run_dir]
                out_dir = os.path.join(rootdir, f'pairplots')
                io.check_else_make_dir(out_dir)
                #  try:
                _ = _gridplots(log_dir, data_, title_str, fname,
                               color=color, marker='x',
                               upper_map='scatter', markeredgewidth=0.4,
                               gridsize=50, out_dir=out_dir)
                #  except UnboundLocalError:
                #      io.log(f'Unable to create _gridplots for {run_dir}.')
                #      continue

            if data_bs is not None:
                out_dir = os.path.join(rootdir,
                                       f'pairplots_bootstrap')
                io.check_else_make_dir(out_dir)
                data_bs_ = data_bs[data_bs.run_dir == run_dir]
                #  try:
                _ = _gridplots(log_dir, data_bs_, title_str, fname,
                               color=color, out_dir=out_dir)
                #  except UnboundLocalError:
                #      io.log(f'Unable to create _gridplots for {run_dir}.')
                #      continue
            plt.close('all')
        plt.close('all')


def _violinplots(data, axes, has_dx, has_dxf,
                 has_dxb, old_dx, bs=False):
    nw_include = [
        (0, 0, 0, 0, 0, 0),
        (1, 0, 1, 1, 0, 1),
        (1, 0, 1, 1, 1, 1),
        (1, 1, 1, 1, 0, 1),
        (1, 1, 1, 1, 1, 1),
    ]
    colors = sns.color_palette(palette='muted',
                               n_colors=len(nw_include))[::-1]
    palette = dict(zip(nw_include, colors))
    axes = axes.flatten()
    axes[0].axvline(x=0, ls=':', color='k')
    axes[1].axvline(x=0, ls=':', color='k')
    x_arr = ['plaqs_diffs', 'tunneling_rate', 'accept_prob']
    cuts = [1., 0., 1.]
    for idx, var in enumerate(x_arr):
        axes[idx] = sns.violinplot(x=var, y='net_weights',
                                   data=data, ax=axes[idx],
                                   cut=cuts[idx], palette=palette)
    if has_dx:
        axes[3] = sns.violinplot(x='dx', y='net_weights',
                                 data=data[data.dx < 10],
                                 palette=palette, ax=axes[3])
    if has_dxf:
        axes[4] = sns.violinplot(x='dxf', y='net_weights',
                                 data=data[data.dxf < 10],
                                 palette=palette, ax=axes[4])
    if has_dxb:
        axes[5] = sns.violinplot(x='dxb', y='net_weights',
                                 data=data[data.dxb < 10],
                                 palette=palette, ax=axes[5])

    for ax in axes:
        ax.set_ylabel('')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))

    for ax in axes[1:]:
        ax.set_yticks([])
        ax.set_yticklabels([])


    if bs:
        labels = [r"""$\langle\delta \phi_{P}\rangle$""",
                  r"""$\langle\gamma\rangle$""",
                  r"""$\langle A(\xi^{\prime}|\xi)\rangle$"""]
        l2_str = r"""$\langle\|x^{(i+1)} - x^{(i)}\|^{2}_{2}\rangle$"""
        cos_str = r"""$\langle 1 - \cos(x^{(i+1)} - x^{(i)})\rangle$"""
        l2_strf = (
            r"""$\langle\|x_{f}^{(i+1)} - x_{f}^{(i)}\|^{2}_{2}\rangle$"""
        )
        cos_strf = (
            r"""$\langle 1 - \cos(x_{f}^{(i+1)} - x_{f}^{(i)})\rangle$"""
        )
        l2_strb = (
            r"""$\langle \|x_{b}^{(i+1)} - x_{b}^{(i)}\|^{2}_{2}\rangle$"""
        )
        cos_strb = (
            r"""$\langle 1 - \cos(x_{b}^{(i+1)} - x_{b}^{(i)})\rangle$"""
        )
    else:
        labels = [r"""$\delta \phi_{P}$""",
                  r"""$\gamma$""",
                  r"""$A(\xi^{\prime}|\xi)$"""]
        l2_str = r"""$\|x^{(i+1)} - x^{(i)}\|^{2}_{2}$"""
        cos_str = r"""$1 - \cos(x^{(i+1)} - x^{(i)})$"""
        l2_strf = r"""$\|x_{f}^{(i+1)} - x_{f}^{(i)}\|^{2}_{2}$"""
        cos_strf = r"""$1 - \cos(x_{f}^{(i+1)} - x_{f}^{(i)})$"""
        l2_strb = r"""$\|x_{b}^{(i+1)} - x_{b}^{(i)}\|^{2}_{2}$"""
        cos_strb = r"""$1 - \cos(x_{b}^{(i+1)} - x_{b}^{(i)})$"""

    if has_dx:
        if old_dx:
            labels.append(l2_str)
        else:
            labels.append(cos_str)
    if has_dxf:
        if old_dx:
            labels.append(l2_strf)
        else:
            labels.append(cos_strf)
    if has_dxb:
        if old_dx:
            labels.append(l2_strb)
        else:
            labels.append(cos_strb)

    labelsize = 16
    for ax, label in zip(axes, labels):
        ax.set_xlabel(label, fontsize=labelsize)

    return axes


def violinplots(log_dirs,
                df_dict=None,
                df_bs_dict=None,
                rp_dict=None,
                rootdir=None):
    """Make violinplots for each log_dir in log_dirs."""
    now = datetime.datetime.now()
    day_str = now.strftime('%Y_%m_%d')
    hour_str = now.strftime('%H%M')
    time_str = f'{day_str}_{hour_str}'
    if rootdir is None:
        proj_dir = os.path.dirname(PROJECT_DIR)
        rootdir = os.path.join(proj_dir, 'cooley_figures')

    out_dir = os.path.join(rootdir, f'violinplots_{time_str}')
    io.check_else_make_dir(out_dir)
    #  fontsize=14

    ticklabelsize = 14
    mpl.rcParams['xtick.labelsize'] = ticklabelsize
    mpl.rcParams['ytick.labelsize'] = ticklabelsize
    titlesize = 22
    #  axeslabelsize = 18
    #  labelsize = 16
    if df_dict is None and df_bs_dict is None and rp_dict is None:
        df_dict, df_bs_dict, rp_dict = build_dataframes(log_dirs)

    now = datetime.datetime.now()
    day_str = now.strftime('%Y_%m_%d')
    hour_str = now.strftime('%H%M')
    time_str = f'{day_str}_{hour_str}'

    keys = list(df_dict.keys())[::-1]
    for idx, log_dir in enumerate(keys):
        io.log(f'log_dir: {log_dir}')
        data = df_dict[log_dir] if df_dict is not None else None
        data_bs = df_bs_dict[log_dir] if df_bs_dict is not None else None
        run_params = rp_dict[log_dir]

        fname, title_str, old_dx = plot_setup(log_dir, run_params)

        has_dx = hasattr(data, 'dx')
        has_dxf = hasattr(data, 'dxf')
        has_dxb = hasattr(data, 'dxb')

        ncols = 3
        figsize = [20, 10]
        if has_dx:
            ncols += 1
            figsize[0] += 4
        if has_dxf:
            ncols += 1
            figsize[0] += 4
        if has_dxb:
            ncols += 1
            figsize[0] += 4

        figsize = tuple(figsize)
        fig, (axes0, axes1) = plt.subplots(nrows=2,
                                           ncols=ncols,
                                           figsize=figsize)

        axes0 = axes0.flatten()
        if data is not None:
            axes0 = _violinplots(data, axes0, has_dx, has_dxf,
                                 has_dxb, old_dx, bs=False)

        axes1 = axes1.flatten()
        axes1 = _violinplots(data_bs, axes1, has_dx, has_dxf,
                             has_dxb, old_dx, bs=True)


        fig.suptitle(title_str, fontsize=titlesize, y=1.025)

        plt.tight_layout()
        out_file = os.path.join(out_dir, f'{fname}.pdf')
        if os.path.isfile(out_file):
            id_str = f'{idx}'
            out_file = os.path.join(out_dir, f'{fname}_{id_str}.pdf')
        io.log(f'INFO: Saving figure to: {out_file}')
        plt.savefig(out_file, bbox_inches='tight')
        if not os.path.isfile(out_file):
            plt.savefig(out_file, bbox_inches='tight')

    return fig, axes0, axes1
