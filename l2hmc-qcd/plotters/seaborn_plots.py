"""
seaborn_plots.py

Collection of functions for making figures with `seaborn`.

Author: Sam Foreman
Date: 01/04/2020
"""
import os
import sys
import time
import pickle
import datetime

from plot_script import get_matching_log_dirs
from plotters.plot_utils import bootstrap, load_pkl
from plotters.plot_observables import (get_obs_dict, get_run_dirs,
                                       get_title_str, grid_plot)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.style as mplstyle

import utils.file_io as io

from lattice.lattice import u1_plaq_exact

sns.set_palette('bright')
mplstyle.use('fast')
ticklabelsize = 14
mpl.rcParams['xtick.labelsize'] = ticklabelsize
mpl.rcParams['ytick.labelsize'] = ticklabelsize


def calc_tunneling_rate(charges):
    """Calculate the tunneling rate as Q_{i+1} - Q_{i}."""
    if not isinstance(charges, np.ndarray):
        charges = np.array(charges)

    if charges.shape[0] > charges.shape[1]:
        charges = charges.T

    charges = np.around(charges)
    dq = np.abs(charges[:, 1:] - charges[:, :-1])
    tunneling_rate = np.mean(dq, axis=1)

    return dq, tunneling_rate


def get_train_weights(params):
    """Extract the `net_weights` used for training from `params`."""
    xsw = int(params['x_scale_weight'])
    xtw = int(params['x_translation_weight'])
    xqw = int(params['x_transformation_weight'])
    vsw = int(params['v_scale_weight'])
    vtw = int(params['v_translation_weight'])
    vqw = int(params['v_transformation_weight'])
    return (xsw, xtw, xqw, vsw, vtw, vqw)


def get_lf(log_dir):
    params = load_pkl(os.path.join(log_dir, 'parameters.pkl'))
    lf = params['num_steps']
    return lf


def get_observables(run_dir,
                    log_dir=None,
                    n_boot=500,
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
    run_params = load_pkl(os.path.join(run_dir, 'run_params.pkl'))
    net_weights = tuple([int(i) for i in run_params['net_weights']])
    #  eps = run_params['eps']
    beta = run_params['beta']
    observables_dir = os.path.join(run_dir, 'observables')
    px = load_pkl(os.path.join(observables_dir, 'px.pkl'))
    px = np.squeeze(np.array(px))
    avg_px = np.mean(px)

    if nw_include is not None:
        keep_data = net_weights in nw_include
    else:
        keep_data = True

    if avg_px < 0.1 or not keep_data:
        print(f'INFO:Skipping! nw: {net_weights}, avg_px: {avg_px:.3g}')
        return None, None, run_params

    io.log(f'Loading data for net_weights: {net_weights}')

    def load_sqz(fname):
        data = load_pkl(os.path.join(observables_dir, fname))
        return np.squeeze(np.array(data))

    charges = load_sqz('charges.pkl')
    plaqs = load_sqz('plaqs.pkl')
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
    dq, _ = calc_tunneling_rate(charges)
    dq = dq.T

    def get_dx(fname):
        dx_file = os.path.join(observables_dir, fname)
        if os.path.isfile(dx_file):
            dx = load_sqz(fname)
            dx = dx[therm_steps:]
            if dx.shape != px.shape:
                dx = None
        else:
            dx = None

        return dx

    dx = get_dx('dx.pkl')
    dxf = get_dx('dxf.pkl')
    dxb = get_dx('dxb.pkl')

    if calc_stats:
        px_avg, px_err, px_ = bootstrap(px, n_boot=n_boot)
        dplq_avg, dplq_err, dplq_ = bootstrap(dplq, n_boot=n_boot)
        dq_avg, dq_err, dq_ = bootstrap(dq, n_boot=n_boot)
        px_ = px_.mean(axis=0)
        dplq_ = dplq_.mean(axis=0)
        dq_ = dq_.mean(axis=0)
        entries = len(dq_.flatten())
        data_bs = pd.DataFrame({
            'plaqs_diffs': dplq_.flatten(),
            'accept_prob': px_.flatten(),
            'tunneling_rate': dq_.flatten(),
            'net_weights': tuple([net_weights for _ in range(entries)]),
            'log_dir': np.array([log_dir for _ in range(entries)]),
        })

        if dx is not None:
            dx_avg, dx_err, dx_ = bootstrap(dx, n_boot=n_boot)
            dx_ = dx_.mean(axis=0)
            data_bs['dx'] = dx_.flatten()
        if dxf is not None:
            dxf_avg, dxf_err, dxf_ = bootstrap(dxf, n_boot=n_boot)
            dxf_ = dxf_.mean(axis=0)
            data_bs['dxf'] = dxf_.flatten()
        if dxb is not None:
            dxb_avg, dxb_err, dxb_ = bootstrap(dxb, n_boot=n_boot)
            dxb_ = dxb_.mean(axis=0)
            data_bs['dxb'] = dxb_.flatten()
    else:
        data_bs = None

    entries = len(dq.flatten())
    data = pd.DataFrame({
        'plaqs_diffs': dplq.flatten(),
        'accept_prob': px.flatten(),
        'tunneling_rate': dq.flatten(),
        'net_weights': tuple([net_weights for _ in range(entries)]),
        'log_dir': np.array([log_dir for _ in range(entries)]),
    })

    if dx is not None:
        data['dx'] = dx.flatten(),
    if dxf is not None:
        data['dxf'] = dxf.flatten(),
    if dxb is not None:
        data['dxb'] = dxb.flatten()

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
    for run_dir in run_dirs:
        try:
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
        except FileNotFoundError:
            continue

    return data, data_bs, run_params


def build_dataframes(log_dirs, df_dict=None, df_bs_dict=None, rp_dict=None):
    if df_dict is None:
        df_dict = {}
    if df_bs_dict is None:
        df_bs_dict = {}
    if rp_dict is None:
        rp_dict = {}

    log_dirs = sorted(log_dirs, key=get_lf, reverse=True)
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
            n_boot = 1000
            frac = 0.25

            try:
                run_dirs = sorted(get_run_dirs(log_dir))[::-1]
            except FileNotFoundError:
                continue

            nw_inc = [
                (0, 0, 0, 0, 0, 0),
                (1, 0, 1, 1, 0, 1),
                (1, 0, 1, 1, 1, 1),
                (1, 1, 1, 1, 0, 1),
                (1, 1, 1, 1, 1, 1),
            ]

            data, data_bs, run_params = _build_dataframes(run_dirs,
                                                          data=data,
                                                          data_bs=data_bs,
                                                          n_boot=n_boot,
                                                          calc_stats=True,
                                                          therm_frac=frac,
                                                          nw_include=nw_inc)
            df_dict[log_dir] = data
            df_bs_dict[log_dir] = data_bs
            rp_dict[log_dir] = run_params

    return df_dict, df_bs_dict, rp_dict


def _violinplots(data, axes, has_dx, has_dxf, has_dxb, old_dx, bs=False):
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
    for idx, var in enumerate(x_arr):
        axes[idx] = sns.violinplot(x=var, y='net_weights', data=data,
                                   palette=palette, ax=axes[idx])
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

    labels = [r"""$\delta \phi_{P}$""",
              r"""$\gamma$""",
              r"""$A(\xi^{\prime}|\xi)$"""]

    if bs:
        l2_str = r"""$\langle\|x^{(i+1)} - x^{(i)}\|^{2}_{2}\rangle$"""
        cos_str = r"""$\langle 1 - \cos(x^{(i+1)} - x^{(i)})\rangle$"""
        l2_strf = (r"""$\langle\|x_{f}^{(i+1)}
                   - x_{f}^{(i)}\|^{2}_{2}\rangle$""")
        cos_strf = (r"""$\langle 1 - \cos(x_{f}^{(i+1)}
                    - x_{f}^{(i)})\rangle$""")
        l2_strb = (r"""$\langle \|x_{b}^{(i+1)}
                   - x_{b}^{(i)}\|^{2}_{2}\rangle$""")
        cos_strb = (r"""$\langle 1 - \cos(x_{b}^{(i+1)}
                    - x_{b}^{(i)})\rangle$""")
    else:
        l2_str = r"""$\|x^{(i+1)} - x^{(i)}\|^{2}_{2}$"""
        cos_str = r"""$1 - \cos(x^{(i+1)} - x^{(i)})$"""

        l2_strf = r"""$\|x_{f}^{(i+1)} - x_{f}^{(i)}\|^{2}_{2}$"""
        cos_strf = r"""$1 - \cos(x_{f}^{(i+1)} - x_{f}^{(i)})$"""

        l2_strb = r"""$\|x_{b}^{(i+1)} - x_{b}^{(i)}\|^{2}_{2}$"""
        cos_strb = r"""$1 - \cos(x_{b}^{(i+1)} - x_{b}^{(i)})$"""

    if has_dx:
        if old_dx:
            labels += l2_str
        else:
            labels += cos_str

    if has_dxf:
        if old_dx:
            labels += l2_strf
        else:
            labels += cos_strf
    if has_dxb:
        if old_dx:
            labels += l2_strb
        else:
            labels += cos_strb

    labelsize = 16
    for ax, label in zip(axes, labels):
        ax.set_xlabel(label, fontsize=labelsize)

    return axes


def violinplots(log_dirs, df_dict=None, df_bs_dict=None, rp_dict=None):
    """Make violinplots for each log_dir in log_dirs."""
    #  fontsize=14
    #  xticklabelsize = 14
    #  yticklabelsize = 14
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
        params = load_pkl(os.path.join(log_dir, 'parameters.pkl'))
        lf = params['num_steps']
        clip_value = params.get('clip_value', 0)

        train_weights = get_train_weights(params)
        train_weights_str = ''.join((str(i) for i in train_weights))
        eps_fixed = params.get('eps_fixed', False)

        data = df_dict[log_dir]
        data_bs = df_bs_dict[log_dir]
        run_params = rp_dict[log_dir]

        has_dx = hasattr(data, 'dx')
        has_dxf = hasattr(data, 'dxf')
        has_dxb = hasattr(data, 'dxb')

        date_str = log_dir.split('/')[-2]
        y, m, d = date_str.split('_')
        y = int(y)
        m = int(m)
        d = int(d)

        old_dx = True
        if y == 2020 and m == 1 and d >= 4:
            old_dx = False

        eps = run_params['eps']

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
        axes0 = _violinplots(data, axes0, has_dx, has_dxf,
                             has_dxb, old_dx, bs=False)

        axes1 = axes1.flatten()
        axes1 = _violinplots(data_bs, axes1, has_dx, has_dxf,
                             has_dxb, old_dx, bs=True)

        fname = f'lf{lf}'
        title_str = (r"$N_{\mathrm{LF}} = $" + f'{lf}, '
                     r"$\varepsilon = $" + f'{eps:.3g}')
        eps_str = f'{eps:.3g}'.replace('.', '')
        fname += f'_e{eps_str}'

        if eps_fixed:
            title_str += ' (fixed) '
            fname += '_fixed'

        if any([tw == 0 for tw in train_weights]):
            tws = '(' + ', '.join((str(i) for i in train_weights_str)) + ')'
            title_str += (', '
                          + r"$\mathrm{nw}_{\mathrm{train}}=$"
                          + f' {tws}')
            fname += f'_{train_weights_str}'

        if params['clip_value'] > 0:
            title_str += f', clip: {clip_value}'
            fname += f'_clip{clip_value}'

        fig.suptitle(title_str, fontsize=titlesize, y=1.025)
        out_dir = os.path.abspath(
            f'/home/foremans/cooley_figures/violinplots_{time_str}'
        )
        io.check_else_make_dir(out_dir)

    return fig, axes0, axes1
