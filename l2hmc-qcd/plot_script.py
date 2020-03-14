"""
plot_script.py

Recreates all necessary plots.

Author: Sam Foreman
Date: 02/29/2020
"""
import os
import time
import argparse
import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.style as mplstyle

import utils.file_io as io

from lattice.lattice import u1_plaq_exact
from plotters.data_utils import bootstrap
from plotters.plot_utils import get_matching_log_dirs, load_pkl
from plotters.seaborn_plots import get_lf, get_observables, get_train_weights
from plotters.plot_observables import get_run_dirs

sns.set_palette('bright')

LABEL_SIZE = 9
mpl.rcParams['xtick.labelsize'] = LABEL_SIZE
mpl.rcParams['ytick.labelsize'] = LABEL_SIZE

mplstyle.use('fast')


# pylint:disable=invalid-name

def parse_args():
    """Parse command line arguments."""
    description = ('Loop over log_dirs and create '
                   '`sns.pairplots` from inference data.')
    parser = argparse.ArgumentParser(
        description=description,
    )
    parser.add_argument('--skip_existing',
                        dest='skip_existing',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will prevent the plotter
                              from unnecessarily re-creating existing
                              plots."""))
    args = parser.parse_args()

    return args


def build_dataframes1(run_dirs, data=None, **kwargs):
    """Build dataframes containing inference data from `run_dirs`."""
    for run_dir in run_dirs:
        try:
            new_df, run_params = get_observables(run_dir, **kwargs)
            if data is None:
                data = new_df
            else:
                data = pd.concat([data, new_df], axis=0).reset_index(drop=True)
        except FileNotFoundError:
            continue
    return data, run_params


def calc_tunneling_rate(charges):
    """Calculate the tunneling rate from `charges`."""
    if charges.shape[0] > charges.shape[1]:
        charges = charges.T

    charges = np.around(charges)
    dq = np.abs(charges[:, 1:] - charges[:, :-1])
    tunneling_rate = np.mean(dq, axis=1)

    return dq, tunneling_rate


def get_observables2(run_dir, log_dir=None, n_boot=500,
                     therm_frac=0.25, nw_include=None, calc_stats=True):
    run_params_file = os.path.join(run_dir, 'run_params.pkl')
    if os.path.isfile(run_params_file):
        run_params = load_pkl(run_params_file)
    else:
        raise FileNotFoundError("Unable to locate `run_params.pkl`, returning.")
    net_weights = tuple([int(i) for i in run_params['net_weights']])
    eps = run_params['eps']
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
        io.log(f'INFO: Skipping nw: {net_weights}, avg_px: {avg_px:.3g}')
        return None, run_params

    io.log(f'Loading data for net_weights: {net_weights}')

    def load_sqz(fname):
        data = load_pkl(os.path.join(observables_dir, fname))
        return np.squeeze(np.array(data))

    charges = load_sqz('charges.pkl')
    plaqs = load_sqz('plaqs.pkl')
    dplq = u1_plaq_exact(beta) - plaqs
    num_steps = px.shape[0]
    therm_steps = int(therm_frac * num_steps)
    steps = np.arange(therm_steps, num_steps)

    def therm_arr(arr):
        arr = arr[1:]
        return arr[therm_steps:]

    px = therm_arr(px)
    dplq = therm_arr(dplq)
    charges = therm_arr(charges)
    dq, _ = calc_tunneling_rate(charges)
    dq = dq.T
    dxf_file = os.path.join(observables_dir, 'dxf.pkl')
    dxb_file = os.path.join(observables_dir, 'dxb.pkl')
    if os.path.isfile(dxf_file) and os.path.isfile(dxb_file):
        dxf = load_sqz('dxf.pkl')
        dxb = load_sqz('dxb.pkl')
        dx = (dxf + dxb) / 2
        dx = dx.mean(axis=-1)
        dx = therm_arr(dx)
    else:
        dx = None

    if calc_stats:
        px_avg, px_err, px = bootstrap(px, n_boot=n_boot)
        dplq_avg, dplq_err, dplq = bootstrap(dplq, n_boot=n_boot)
        dq_avg, dq_err, dq = bootstrap(dq, n_boot=n_boot)
        px = px.mean(axis=0)
        dplq = dplq.mean(axis=0)
        dq = dq.mean(axis=0)
        if dx is not None:
            dx_avg, dx_err, dx = bootstrap(dx, n_boot=n_boot)
            dx = dx.mean(axis=0)

    num_entries = len(dq.flatten())
    if dx is not None:
        data = pd.DataFrame({
            'plaqs_diffs': dplq.flatten(),
            'accept_prob': px.flatten(),
            'tunneling_rate': dq.flatten(),
            'dx': dx.flatten(),
            'net_weights': tuple([net_weights for _ in range(num_entries)]),
            'log_dir': np.array([log_dir for _ in range(num_entries)]),
        })
    else:
        data = pd.DataFrame({
            'plaqs_diffs': dplq.flatten(),
            'accept_prob': px.flatten(),
            'tunneling_rate': dq.flatten(),
            #  'dx': dx.flatten(),
            'net_weights': tuple([net_weights for _ in range(num_entries)]),
            'log_dir': np.array([log_dir for _ in range(num_entries)]),
        })

    #  data = data.dropna()

    return data, run_params


def get_observables1(run_dir, n_boot=1000, therm_frac=0.2, nw_include=None):
    run_params = load_pkl(os.path.join(run_dir, 'run_params.pkl'))
    net_weights = tuple([int(i) for i in run_params['net_weights']])
    #  eps = run_params['eps']
    observables_dir = os.path.join(run_dir, 'observables')
    px = load_pkl(os.path.join(observables_dir, 'px.pkl'))
    px = np.squeeze(np.array(px))
    avg_px = np.mean(px)

    if nw_include is not None:
        keep_data = net_weights in nw_include
    else:
        keep_data = True

    if avg_px < 0.1 or not keep_data:
        print(f'INFO: Skipping! nw: {net_weights}, avg_px: {avg_px:.3g}')
        return None, run_params

    io.log(f'Loading data for net_weights: {net_weights}')
    plaqs = load_pkl(os.path.join(observables_dir, 'plaqs.pkl'))
    dplq = u1_plaq_exact(run_params['beta']) - np.squeeze(np.array(plaqs))
    charges = load_pkl(os.path.join(observables_dir, 'charges.pkl'))
    charges = np.squeeze(np.array(charges))

    # Since the number of tunneling events is computed as
    # `dq = charges[1:] - charges[:-1]`, `dq.shape[0] = num_steps - 1`.
    # Because of this, we drop the first step of `px` and `dplq`
    # to enforce that they all have the same shape.
    px = px[1:]
    dplq = dplq[1:]

    num_steps = px.shape[0]
    therm_steps = int(therm_frac * num_steps)

    px = px[therm_steps:]
    dplq = dplq[therm_steps:]
    charges = charges[therm_steps:]

    dq, _ = calc_tunneling_rate(charges)
    dq = dq.T

    px_avg, px_err, px = bootstrap(px, n_boot=n_boot)
    dplq_avg, dplq_err, dplq = bootstrap(dplq, n_boot=n_boot)
    dq_avg, dq_err, dq = bootstrap(dq, n_boot=n_boot)

    px = px.mean(axis=0)
    dplq = dplq.mean(axis=0)
    dq = dq.mean(axis=0)

    data = pd.DataFrame({
        'plaqs_diffs': dplq.flatten(),
        'accept_prob': px.flatten(),
        'tunneling_rate': dq.flatten(),
        'net_weights': tuple([net_weights for _ in range(len(dq.flatten()))]),
    })

    return data, run_params


def get_previous_dir(root_dir):
    """Get previous dir."""
    dirs = sorted(filter(os.path.isdir,
                         os.listdir(root_dir)),
                  key=os.path.getmtime)
    dirs = [i for i in dirs if 'pairplots' in i
            and 'combined' not in i]
    previous_dir = dirs[0]

    return previous_dir


def infer_cmap(color, palette='bright'):
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
    palette = kwargs.pop('palette', 'bright')
    cmap = infer_cmap(kwargs['color'], palette=palette)
    ax = sns.kdeplot(x, y, cmap=cmap, **kwargs)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    #  sns.despine(ax=ax, bottom=True, left=True)
    return ax


def kde_diag_plot(x, **kwargs):
    ax = sns.kdeplot(x, **kwargs)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    return ax


def plot_pts(x, y, **kwargs):
    ax = plt.gca()
    _ = ax.plot(x, y, **kwargs)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    return ax


def combined_pair_plotter(log_dirs, therm_frac=0.2, calc_stats=True,
                          n_boot=1000, nw_include=None):
    now = datetime.datetime.now()
    day_str = now.strftime('%Y_%m_%d')
    hour_str = now.strftime('%H%M')
    time_str = f'{day_str}_{hour_str}'

    m1 = ['x', 'v']
    m2 = ['scale', 'translation', 'transformation']
    matches = [f'{i}_{j}' for i in m1 for j in m2]

    t0 = time.time()
    for log_dir in log_dirs:
        io.log(f'log_dir: {log_dir}')
        params = load_pkl(os.path.join(log_dir, 'parameters.pkl'))
        lf = params['num_steps']
        clip_value = params.get('clip_value', 0)
        train_weights = get_train_weights(params)
        #  train_weights = tuple([
        #      int(params[key]) for key in params
        #      if any([m in key for m in matches])
        #  ])
        train_weights_str = ''.join((str(i) for i in train_weights))
        eps_fixed = params.get('eps_fixed', False)

        data = None
        data_bs = None
        n_boot = 5000
        therm_frac = 0.25

        try:
            run_dirs = sorted(get_run_dirs(log_dir))[::-1]
        except FileNotFoundError:
            continue

        if nw_include is None:
            nw_include = [
                #  (0, 0, 0, 0, 0, 0),
                (1, 0, 1, 1, 0, 1),
                (1, 0, 1, 1, 1, 1),
                (1, 1, 1, 1, 0, 1),
                (1, 1, 1, 1, 1, 1),
            ]

        for run_dir in run_dirs:
            try:
                kwargs = {
                    'n_boot': n_boot,
                    'calc_stats': calc_stats,
                    'therm_frac': therm_frac,
                    'nw_include': nw_include,
                }
                new_df, new_df_bs, run_params = get_observables(run_dir,
                                                                **kwargs)
                eps = run_params['eps']
                if data is None:
                    data = new_df
                else:
                    data = pd.concat([data, new_df], axis=0)
                    data = data.reset_index(drop=True)

                if data_bs is None:
                    data_bs = new_df_bs
                else:
                    data_bs = pd.concat(
                        [data_bs, new_df_bs], axis=0
                    ).reset_index(drop=True)
            except FileNotFoundError:
                continue

        if calc_stats:
            plot_data = data_bs
        else:
            plot_data = data

        g = sns.PairGrid(plot_data, hue='net_weights',
                         palette='bright', diag_sharey=False,
                         #  hue_kws={"cmap": list_of_cmaps},
                         vars=['plaqs_diffs', 'accept_prob', 'tunneling_rate'])
        g = g.map_diag(sns.kdeplot, shade=True)
        g = g.map_lower(plot_pts, ls='', marker='o',
                        rasterized=True, alpha=0.4)
        g = g.map_upper(kde_color_plot, shade=False, gridsize=100)

        g.add_legend()
        # Create title for plot
        title_str = (r"$N_{\mathrm{LF}} = $" + f'{lf}, '
                     r"$\varepsilon = $" + f'{eps:.3g}')
        if eps_fixed:
            title_str += ' (fixed) '
        if any([tw == 0 for tw in train_weights]):
            tws = '(' + ', '.join((str(i) for i in train_weights_str)) + ')'
            title_str += (', '
                          + r"$\mathrm{nw}_{\mathrm{train}}=$"
                          + f' {tws}')

        #  title_str += f'; nw: {key}'
        #  if any([tw == 0 for tw in train_weights]):
        #      tws = '(' + ', '.join((str(i) for i in train_weights_str)) + ')'
        #      title_str += (
        #          ', ' + r"$\vec{\alpha}_{\mathrm{train}}=$" + f' {tws}'
        #      )

        if params['clip_value'] > 0:
            title_str += f', clip: {clip_value}'

        g.fig.suptitle(title_str, y=1.02, fontsize='x-large')

        if calc_stats:
            outpath = (f'/home/foremans/cooley_figures'
                       f'/combined_pairplots_boostrap_{time_str}')
        else:
            outpath = (f'/home/foremans/cooley_figures'
                       f'/combined_pairplots_{time_str}')

        out_dir = os.path.abspath(outpath)
        io.check_else_make_dir(out_dir)

        fname = f'lf{lf}'
        if clip_value > 0:
            fname += f'_clip{int(clip_value)}'

        if eps_fixed:
            fname += f'_eps_fixed_'

        if any([tw == 0 for tw in train_weights]):
            fname += f'_train{train_weights_str}'

        id_str = log_dir.split('/')[-1].split('_')[-1]
        out_file = os.path.join(out_dir, f'{fname}_{id_str}.png')
        if os.path.isfile(out_file):
            out_file = os.path.join(out_dir, f'{fname}_{id_str}_1.png')
        io.log(f'INFO:Saving figure to: {out_file}')
        g.savefig(out_file, dpi=150, bbox_inches='tight')

        if not os.path.isfile(out_file):
            plt.savefig(out_file, dpi=150, bbox_inches='tight')
        print(f'Time spent plotting: {time.time() - t0:.3g}')
        io.log(80 * '-' + '\n')


def pair_plotter(log_dirs, therm_frac=0.2, n_boot=1000,
                 nw_include=None, skip_existing=False, calc_stats=True):
    now = datetime.datetime.now()
    day_str = now.strftime('%Y_%m_%d')
    hour_str = now.strftime('%H%M')
    time_str = f'{day_str}_{hour_str}'

    from config import COLORS

    colors = 100 * [*COLORS]

    m1 = ['x', 'v']
    m2 = ['scale', 'translation', 'transformation']
    matches = [f'{i}_{j}' for i in m1 for j in m2]

    t0 = time.time()
    for idx, log_dir in enumerate(log_dirs):
        io.log(f'log_dir: {log_dir}')
        run_dirs = sorted(get_run_dirs(log_dir))
        params = load_pkl(os.path.join(log_dir, 'parameters.pkl'))
        lf = params['num_steps']
        clip_value = params.get('clip_value', 0)
        train_weights = get_train_weights(params)
        #  train_weights = tuple([
        #      int(params[key]) for key in params
        #      if any([m in key for m in matches])
        #  ])
        train_weights_str = ''.join((str(i) for i in train_weights))
        cmap = sns.light_palette(colors[idx], as_cmap=True)

        if skip_existing:
            root_dir = os.path.abspath(f'/home/foremans/cooley_figures/')
            previous_dir = get_previous_dir(root_dir)
            out_dir = os.path.join(root_dir, previous_dir)

        else:
            if calc_stats:
                outpath = (f'/home/foremans/cooley_figures/'
                           f'pairplots_bootstrap_{time_str}')
            else:
                outpath = (f'/home/foremans/cooley_figures/'
                           f'pairplots_{time_str}')

            out_dir = os.path.abspath(outpath)

        io.check_else_make_dir(out_dir)
        for run_dir in run_dirs:
            t1 = time.time()
            px_file = os.path.join(run_dir, 'observables', 'px.pkl')
            if os.path.isfile(px_file):
                kwargs = {
                    'n_boot': n_boot,
                    'calc_stats': calc_stats,
                    'therm_frac': therm_frac,
                    'nw_include': nw_include,
                }
                data, data_bs, run_params = get_observables(run_dir, **kwargs)
            else:
                continue

            if calc_stats:
                plot_data = data_bs
            else:
                plot_data = data

            key = tuple([int(i) for i in run_params['net_weights']])
            eps = run_params['eps']
            if data is None:
                continue

            fname = f'lf{lf}_'
            if any([tw == 0 for tw in train_weights]):
                fname += f'_train{train_weights_str}_'
                nw_train_str = ('('
                                + ', '.join((str(i) for i in train_weights))
                                + ')')


            if params['eps_fixed']:
                fname += f'_eps_fixed_'

            if clip_value > 0:
                fname += f'clip{int(clip_value)}_'

            nw_str = ''.join((str(int(i)) for i in key))
            fname += f'_{nw_str}'

            id_str = log_dir.split('/')[-1].split('_')[-1]
            out_file = os.path.join(out_dir, f'{fname}_{id_str}.png')
            if os.path.isfile(out_file):
                if skip_existing:
                    continue
                else:
                    out_file = os.path.join(out_dir, f'{fname}_{id_str}_1.png')

            if os.path.isfile(out_file):
                if skip_existing:
                    continue
                else:
                    out_file = os.path.join(out_dir, fname + '_1.png')

            g = sns.PairGrid(plot_data,
                             hue='net_weights',
                             palette='bright',
                             diag_sharey=False,
                             vars=['plaqs_diffs',
                                   'accept_prob',
                                   'tunneling_rate'])
            g = g.map_lower(plot_pts, color=colors[idx],
                            ls='', marker='o', rasterized=True, alpha=0.4)
            try:
                g = g.map_diag(sns.kdeplot, shade=True,
                               color=colors[idx], gridsize=100)
                g = g.map_upper(sns.kdeplot,
                                cmap=cmap,
                                shade=True,
                                gridsize=50,
                                shade_lowest=False)
                #  g = g.map_upper(kde_color_plot, shade=False, gridsize=50)
            except np.linalg.LinAlgError:
                g = g.map_upper(plt.hist, histtype='step',
                                color=colors[idx], alpha=0.6,
                                ec=colors[idx], density=True)
                g = g.map_diag(plt.hist, histtype='step',
                               color=colors[idx], alpha=0.6,
                               ec=colors[idx], density=True)

            # Create title for plot
            title_str = (r"$N_{\mathrm{LF}} = $" + f'{lf}, '
                         r"$\varepsilon = $" + f'{eps:.3g}')
            if params['clip_value'] > 0:
                title_str += f', clip: {clip_value}'

            if any([tw == 0 for tw in train_weights]):
                tws = '(' + ', '.join((str(i) for i in train_weights_str)) + ')'
                title_str += (', '
                              + r"$\mathrm{nw}_{\mathrm{train}}=$"
                              + f' {tws}')

            title_str += f'; nw: {key}'
            g.fig.suptitle(title_str, y=1.02, fontsize='x-large')

            io.log(f'  Saving figure to: {out_file}')
            g.savefig(out_file, dpi=150, bbox_inches='tight')

            if not os.path.isfile(out_file):
                plt.savefig(out_file, dpi=150, bbox_inches='tight')

            print(f'Time spent plotting: {time.time() - t1:.3g}s.\n')
        print(80*'-' + '\n')

    print(80 * '-' + '\n\n')
    print(f'Time to complete: {time.time() - t0:.4g}s.')


def main():
    therm_frac = 0.25  # percent of steps to skip for thermalization
    n_boot = 5000     # number of bootstrap iterations to run for statistics

    nw_include = [
        (0, 0, 0, 0, 0, 0),
        # --------------------
        #  (0, 0, 0, 0, 0, 1),
        #  (0, 0, 0, 0, 1, 0),
        #  (0, 0, 0, 1, 0, 0),
        #  (0, 0, 1, 0, 0, 0),
        #  (0, 1, 0, 0, 0, 0),
        #  (1, 0, 0, 0, 0, 0),
        # --------------------
        #  (0, 1, 1, 1, 1, 1),
        (1, 0, 1, 1, 1, 1),
        #  (1, 1, 0, 1, 1, 1),
        #  (1, 1, 1, 0, 1, 1),
        (1, 1, 1, 1, 0, 1),
        #  (1, 1, 1, 1, 1, 0),
        # --------------------
        #  (0, 0, 1, 1, 1, 1),
        #  (0, 1, 0, 1, 1, 1),
        #  (1, 0, 0, 1, 1, 1),
        #  (1, 1, 1, 0, 0, 1),
        #  (1, 1, 1, 0, 1, 0),
        #  (1, 1, 1, 1, 0, 0),
        # --------------------
        #  (0, 1, 0, 0, 1, 0),
        (1, 0, 1, 1, 0, 1),
        # --------------------
        #  (0, 0, 0, 1, 1, 1),
        #  (0, 0, 1, 1, 1, 0),
        #  (0, 1, 1, 1, 0, 0),
        #  (1, 1, 1, 0, 0, 0),
        # --------------------
        (1, 1, 1, 1, 1, 1),
    ]
    root_dir = os.path.abspath('/home/foremans/DLHMC/l2hmc-qcd/gauge_logs')
    dates = ['2019_12_15',
             '2019_12_16',
             '2019_12_22',
             '2019_12_24',
             '2019_12_25',
             '2019_12_26',
             '2019_12_28',
             '2019_12_29',
             '2019_12_30',
             '2019_12_31',
             '2020_01_02',
             '2020_01_03',
             '2020_01_04',
             '2020_01_05',
             '2020_01_06',
             '2020_01_07']
    log_dirs = []
    for date in dates:
        ld = get_matching_log_dirs(date, root_dir=root_dir)
        log_dirs += [*ld]

    #  with sns.axes_style('darkgrid'):
    log_dirs = sorted(log_dirs, key=get_lf, reverse=True)
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    #  pair_plotter(log_dirs=log_dirs, n_boot=n_boot, calc_stats=True,
    #               therm_frac=therm_frac, nw_include=nw_include)
    combined_pair_plotter(log_dirs=log_dirs, n_boot=n_boot, calc_stats=True,
                          therm_frac=therm_frac, nw_include=None)
    #  pair_plotter(log_dirs=log_dirs, n_boot=n_boot, calc_stats=True,
    #               therm_frac=therm_frac, nw_include=nw_include)
    #  combined_pair_plotter(log_dirs=log_dirs, n_boot=n_boot, calc_stats=True,
    #                        therm_frac=therm_frac, nw_include=None)


if __name__ == '__main__':
    main()
