import os
import sys
import time
import datetime
import shutil
import pickle
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette('bright')

#  label_size = 10
#  import matplotlib as mpl
#  mpl.rcParams['xtick.labelsize'] = label_size
#  mpl.rcParams['xtick.labelsize'] = label_size

import matplotlib.style as mplstyle
mplstyle.use('fast')

import utils.file_io as io
from lattice.lattice import u1_plaq_exact
from plotters.plot_observables import get_run_dirs, get_title_str
from plotters.plot_utils import bootstrap, load_pkl


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


def calc_tunneling_rate(charges):
    if charges.shape[0] > charges.shape[1]:
        charges = charges.T

    charges = np.around(charges)
    dq = np.abs(charges[:, 1:] - charges[:, :-1])
    tunneling_rate = np.mean(dq, axis=1)

    return dq, tunneling_rate


def get_observables(run_dir, n_boot=1000, therm_frac=0.2, nw_include=None):
    run_params = load_pkl(os.path.join(run_dir, 'run_params.pkl'))
    net_weights = tuple([int(i) for i in run_params['net_weights']])
    eps = run_params['eps']
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

    data = pd.DataFrame({
        'plaqs_diffs': dplq.flatten(),
        'accept_prob': px.flatten(),
        'tunneling_rate': dq.flatten(),
        'net_weights': tuple([net_weights for _ in range(len(dq.flatten()))]),
    })

    return data, run_params


def get_previous_dir(root_dir):
    dirs = sorted(filter(os.path.isdir,
                          os.listdir(root_dir)),
                          key=os.path.getmtime)
    dirs = [i for i in dirs if 'pairplots' in i
            and 'combined' not in i]
    previous_dir = dirs[0]

    return previous_dir


def plotter(log_dirs, therm_frac=0.2, n_boot=1000,
            nw_include=None, skip_existing=False):
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
        train_weights = tuple([
            int(params[key]) for key in params
            if any([m in key for m in matches])
        ])
        train_weights_str = ''.join((str(i) for i in train_weights))
        cmap = sns.light_palette(colors[idx], as_cmap=True)

        if skip_existing:
            root_dir = os.path.abspath(f'/home/foremans/cooley_figures/')
            previous_dir = get_previous_dir(root_dir)
            out_dir = os.path.join(root_dir, previous_dir)

        else:
            out_dir = os.path.abspath(f'/home/foremans/'
                                      f'cooley_figures/pairplots_{time_str}')
            io.check_else_make_dir(out_dir)

        if any([tw == 0 for tw in train_weights]):
            out_dir = os.path.join(out_dir, f'train_{train_weights_str}')

        for run_dir in run_dirs:
            t1 = time.time()
            data, run_params = get_observables(run_dir,
                                               n_boot=n_boot,
                                               therm_frac=therm_frac,
                                               nw_include=nw_include)
            key = tuple([int(i) for i in run_params['net_weights']])
            eps = run_params['eps']
            if data is None:
                continue

            nw_str = ''.join((str(int(i)) for i in key))
            fname = f'lf{lf}_'
            if clip_value > 0:
                fname += f'clip{int(clip_value)}_'
            fname += f'_{nw_str}'

            out_file = os.path.join(out_dir, fname + '.png')
            if os.path.isfile(out_file):
                if skip_existing:
                    continue
                else:
                    out_file = os.path.join(out_dir, fname + '_1.png')

            g = sns.PairGrid(data, diag_sharey=False)
            g = g.map_lower(plt.plot, color=colors[idx],
                            ls='', marker='+', rasterized=True,
                            markeredgewidth=0.1)
            try:
                g = g.map_diag(sns.kdeplot, shade=True,
                               color=colors[idx], gridsize=100)
                g = g.map_upper(sns.kdeplot, shade=False,
                                cmap=cmap, gridsize=50)
            except:
                g = g.map_upper(plt.hist, histtype='step',
                                color=colors[idx], alpha=0.6,
                                ec=colors[idx], density=True)
                g = g.map_diag(plt.hist, histtype='step',
                               color=colors[idx], alpha=0.6,
                               ec=colors[idx], density=True)

            # Create title for plot
            title_str = (r"$N_{\mathrm{LF}} = $" + f'{lf}, '
                         r"$\varepsilon = $"  + f'{eps:.3g}')
            if params['clip_value'] > 0:
                title_str += f', clip: {clip_value}'
            title_str += f', nw: {key}'
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
    therm_frac = 0.2  # percent of steps to skip for thermalization
    n_boot = 1000     # number of bootstrap iterations to run for statistics

    nw_include = [
    (0, 0, 0, 0, 0, 0),
    # --------------------
    (0, 0, 0, 0, 0, 1),
    (0, 0, 0, 0, 1, 0),
    (0, 0, 0, 1, 0, 0),
    (0, 0, 1, 0, 0, 0),
    (0, 1, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0),
    # --------------------
    (0, 1, 1, 1, 1, 1),
    (1, 0, 1, 1, 1, 1),
    (1, 1, 0, 1, 1, 1),
    (1, 1, 1, 0, 1, 1),
    (1, 1, 1, 1, 0, 1),
    (1, 1, 1, 1, 1, 0),
    # --------------------
    (0, 0, 1, 1, 1, 1),
    (0, 1, 0, 1, 1, 1),
    (1, 0, 0, 1, 1, 1),
    (1, 1, 1, 0, 0, 1),
    (1, 1, 1, 0, 1, 0),
    (1, 1, 1, 1, 0, 0),
    # --------------------
    (0, 0, 0, 1, 1, 1),
    (0, 0, 1, 1, 1, 0),
    (0, 1, 1, 1, 0, 0),
    (1, 1, 1, 0, 0, 0),
    # --------------------
    (1, 1, 1, 1, 1, 1),
    ]

    log_dirs = [
        os.path.abspath('../gauge_logs/2019_12_15/L8_b64_lf1_f32/'),
        os.path.abspath('../gauge_logs/2019_12_15/L8_b64_lf1_f32_0929/'),
        # lf2
        os.path.abspath('../gauge_logs/2019_12_15/L8_b64_lf2_f32_0408/'),
        os.path.abspath('../gauge_logs/2019_12_15/L8_b64_lf2_f32_2157/'),
        # lf3
        os.path.abspath('../gauge_logs/2019_12_15/L8_b64_lf3_f32_0413/'),
        os.path.abspath('../gauge_logs/2019_12_15/L8_b64_lf3_f32_2206/'),
        # lf4
        os.path.abspath('../gauge_logs/2019_12_15/L8_b64_lf4_f32_0431/'),
        os.path.abspath('../gauge_logs/2019_12_15/L8_b64_lf4_f32_2224/'),
        # lf5
        os.path.abspath('../gauge_logs/2019_12_15/L8_b64_lf5_f32_0843/'),
        os.path.abspath('../gauge_logs/2019_12_16/L8_b64_lf5_f32_0328/'),
    ]


    plotter(log_dirs=log_dirs, n_boot=n_boot,
            therm_frac=therm_frac, nw_include=nw_include)

if __name__ == '__main__':
    main()
