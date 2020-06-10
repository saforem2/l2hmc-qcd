"""
tunneling_rate_plots.py

Implements methods for plotting the tunneling rate results.
"""
import os
import shutil

from typing import NoReturn
from collections import namedtuple

import arviz as az
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from config import GAUGE_LOGS_DIR

from .data_utils import therm_arr

sns.set_palette('bright')
mplstyle.use('fast')

import utils.file_io as io

Losses = namedtuple('Losses', ['steps', 'plaq', 'charge'])


def _update_run_dir(run_dir):
    obs_dir = os.path.join(run_dir, 'observables')
    if os.path.isdir(obs_dir):
        return obs_dir
    return run_dir


def get_losses(run_dir):
    """Get losses from `run_dir`."""
    run_dir = _update_run_dir(run_dir)
    plaq_loss = io.loadz(os.path.join(run_dir, 'plaq_loss.z'))
    charge_loss = io.loadz(os.path.join(run_dir, 'charge_loss.z'))

    return plaq_loss, charge_loss


def load_charges(run_dir):
    run_dir = _update_run_dir(run_dir)
    return io.loadz(os.path.join(run_dir, 'charges.z'))


def _get_from_fname(run_dir, fname):
    run_dir = _update_run_dir(run_dir)
    return io.loadz(os.path.join(run_dir, f'{fname}.z'))


def get_from_fname(log_dir, fname, filter_strs=None):
    run_dirs = io.get_run_dirs(log_dir)
    if filter_strs is not None:
        for filter_str in filter_strs:
            run_dirs = [i for i in run_dirs if filter_str in i]
    obs_arr = [_get_from_fname(rd, fname) for rd in run_dirs]
    return np.squeeze(obs_arr)


def calc_dq(q):
    """Calculate the change in topological charge."""
    q = np.insert(q, 0, q[0], axis=0)
    return np.abs(np.around(q[1:]) - np.around(q[:-1]))


def calc_tunneling_stats(charges):
    """Calculate the tunneling rate and number of tunneling events."""
    step_ax = 0
    num_steps = charges.shape[step_ax]
    charges = np.insert(charges, 0, charges[0], axis=step_ax)
    dq = np.abs(np.around(charges[1:]) - np.around(charges[:-1]))
    tunneling_events = np.sum(dq, axis=step_ax)
    tunn_stats = {
        'tunneling_events': tunneling_events,
        'tunneling_rate': tunneling_events / num_steps,
    }

    return tunn_stats


def find_run_dirs(log_dir, filter_str, runs_str='np'):
    runs_dir = os.path.join(log_dir, f'runs_{runs_str}')
    run_dirs = []
    if os.path.isdir(runs_dir):
        run_dirs = [
            os.path.join(runs_dir, i) for i in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, i))
        ]
        run_dirs = [rd for rd in run_dirs if filter_str in rd]
    return run_dirs


def get_run_params(run_dir):
    try:
        run_params = io.loadz(os.path.join(run_dir, 'run_params.z'))
    except FileNotFoundError:
        run_str = run_dir.split('/')[-1].split('_')
        eps_str = str([
            i for i in run_str if 'eps' in i and 'steps' not in i
        ][0]).lstrip('eps')
        beta_str = str([
            i for i in run_str if 'beta' in i
        ][0]).lstrip('beta')
        ns_str = [i for i in run_str if 'lf' in i][0]
        bs_str = [i for i in run_str if 'bs' in i][0]
        rs_str = [i for i in run_str if 'steps' in i][0]
        nw_str = str([i for i in run_str if 'nw' in i][0]).lstrip('nw')

        try:
            run_params = {
                'num_steps': int(ns_str.lstrip('lf')),
                'batch_size': int(bs_str.lstrip('bs')),
                'run_steps': int(rs_str.lstrip('steps')),
                'beta': float(beta_str) / (10 ** (len(beta_str) - 1)),
                'eps': float(eps_str) / (10 ** (len(eps_str) - 1)),
                'net_weights': [float(i) for i in nw_str],
                'hmc': 'nw000000' in run_str,
            }
        except TypeError:
            run_params = {}

    return run_params


def get_title_str(params, filter_str):
    batch_size = params.get('batch_size', None)
    beta_str = filter_str.lstrip('beta')
    beta = float(beta_str) / (10 ** (len(beta_str) - 1))
    title_str = (f"{params['time_size']}"
                 r"$\times$" + f"{params['space_size']}, "
                 r"$\beta = $" + f'{beta:.3g}, '
                 r"$N_{\mathrm{LF}} = $" + f"{params['num_steps']}, "
                 r"$N_{\mathrm{B}}^{\mathrm{train}} = $" + f"{batch_size}")

    return title_str


def get_unique_betas(log_dir):
    def _get_beta(rd):
        rs = rd.split('/')[-1].split('_')
        return rs[3]

    runs_dir = os.path.join(log_dir, 'runs_np')
    if os.path.isdir(runs_dir):
        run_dirs = [
            os.path.join(runs_dir, i) for i in os.listdir(runs_dir)
        ]
        run_dirs = [rd for rd in run_dirs if os.path.isdir(rd)]
        betas = [_get_beta(rd) for rd in run_dirs]
    else:
        betas = []

    return np.unique(betas)


def get_log_dirs(base_dir):
    """Find all `log_dirs` in `base_dir` and its subdirectories."""
    log_dirs = []
    for root, _, _ in os.walk(base_dir):
        if check_if_log_dir(root):
            log_dirs.append(root)

    return log_dirs


def check_if_log_dir(log_dir):
    """Check if `log_dir` is a proper `log_dir` by examining its contents."""
    train_str = log_dir.split('/')[-1]
    cond1 = 'L16' in train_str
    cond2 = 'L8' in train_str

    if not (cond1 or cond2):
        return False

    fnames = ['params.z', 'seeds.z', 'seeds.txt']
    dirs = ['runs_np', 'training', 'checkpoints', ]
    file_exists = [
        os.path.isfile(os.path.join(log_dir, fname)) for fname in fnames
    ]
    dir_exists = [
        os.path.isdir(os.path.join(log_dir, d)) for d in dirs
    ]
    file_check = (np.sum(file_exists) == len(fnames))
    dir_check = (np.sum(dir_exists) == len(dirs))
    #  if file_check and dir_check:
    #      return True
    #  else:
    #      return False
    return file_check and dir_check


def get_matching_hmc_dir(run_params):
    """Find HMC data with matching `run_params` to compare against."""
    base_dir = os.path.join(GAUGE_LOGS_DIR, 'HMC_LOGS_NP')
    lf = run_params.get('num_steps', None)
    beta = run_params.get('beta', None)
    hmc_dirs = []
    if lf is not None and beta is not None:
        lf_str = f'lf{lf}'
        beta_str = f'beta{beta}'.replace('.', '')
        for root, dirs, _ in os.walk(base_dir):
            head, tail = os.path.split(root)
            cond1 = tail.startswith(lf_str)
            cond2 = beta_str in tail
            cond3 = 'runs_np' in head
            if cond1 and cond2 and cond3:
                hmc_dirs.append(root)

    return hmc_dirs


def update_data(idx, run_dir, run_params,
                data, tunn_rates, fnames, therm_frac=0.33):
    """Update `data` and `tunn_rates` with new charges from `run_dirs`."""
    qfile = os.path.join(run_dir, 'charges.z')
    if not os.path.isfile(qfile) or run_params == {}:
        return data, tunn_rates, fnames
    q = io.loadz(qfile)
    q, steps = therm_arr(q, therm_frac=therm_frac)
    dq = calc_dq(q)
    tunn_rate = np.sum(dq, axis=0) / dq.shape[0]
    eps = run_params.get('eps', None)
    bs = run_params.get('batch_size', None)
    if 'nw000000' in run_dir:
        key = f'HMC, ' + r"$\varepsilon = $" + f'{eps:.3g}'
        fname = f'hmc_eps{eps:.3g}'

    elif 'nw111111' in run_dir:
        key = f'L2HMC, ' + r"$\varepsilon = $" + f'{eps:.3g}'
        fname = f'l2hmc_eps{eps:.3g}'
        #  key = f'L2HMC, eps: {eps:.3g}'
    key += r", $N_{\mathrm{B}} = $" + f'{bs}'
    fname += f'_bs{bs}'
    if key in tunn_rates:
        arr = tunn_rates[key]
        tunn_rates[key] = np.concatenate((tunn_rate.flatten(), arr.flatten()))
        key += f', run: {idx}'
        fname += f'_run{idx}'
    else:
        tunn_rates[key] = tunn_rate.flatten()

    q = np.array(q)
    fnames[key] = fname
    q = q.T
    data[key] = {
        'steps': steps,
        'charges': q,
    }

    return data, tunn_rates, fnames


def get_charges(run_dirs, data, tunn_rates, fnames, therm_frac=0.33):
    """Load all topological charges from `run_dirs`."""
    for idx, run_dir in enumerate(run_dirs):
        try:
            run_params = get_run_params(run_dir)
        except ValueError:
            continue

        data, tunn_rates, fnames = update_data(idx, run_dir, run_params,
                                               data, tunn_rates, fnames,
                                               therm_frac=therm_frac)
        hmc_dirs = get_matching_hmc_dir(run_params)
        if len(hmc_dirs) > 0:
            for jdx, hmc_dir in enumerate(hmc_dirs):
                try:
                    rpf = os.path.join(hmc_dir, 'run_params.z')
                    rp_hmc = io.loadz(rpf)
                except FileNotFoundError:
                    rp_hmc = run_params
                data, tunn_rates, fnames = update_data(jdx, hmc_dir,
                                                       rp_hmc, data,
                                                       tunn_rates, fnames,
                                                       therm_frac=therm_frac)

    return data, tunn_rates, fnames


def make_density_plots(tunn_rates, title_str=None,
                       out_file=None, copy_dst=None):
    """Make density plots of the tunneling rate."""
    fig, ax = plt.subplots()
    for idx, (key, tunn_rate) in enumerate(tunn_rates.items()):
        #  dq = calc_dq(q)
        #  tunn_rate = np.sum(dq, axis=0) / dq.shape[0]
        kwargs = {
            'shade': True,
            'ax': ax,
            'label': key,
        }
        if 'L2HMC' in key:
            kwargs['color'] = 'k'

        sns.kdeplot(tunn_rate.flatten(), **kwargs)
        #  try:
        #  except:
        #      ax.hist(tunn_rate.flatten(), density=True, label=key)
        #      #  ax.legend(loc='best')

    _ = ax.legend(loc='best', fontsize='small')
    #  _ = plt.legend(ncol=1, bbox_to_anchor=(1.04, 0.5),
    #                 loc="center left", borderaxespad=0)
    _ = ax.set_xlabel('tunneling rate, ' + r"$\gamma$", fontsize='large')
    if title_str is not None:
        _ = fig.suptitle(title_str, fontsize='x-large')
    #  plt.tight_layout()
    if out_file is not None:
        io.log(f'Saving figure to: {out_file}.')
        plt.savefig(out_file, dpi=400, bbox_inches='tight')
        if copy_dst is not None:
            io.check_else_make_dir(copy_dst)
            io.log(f'Copying figure to: {copy_dst}.')
            _ = shutil.copy2(out_file, copy_dst, follow_symlinks=True)


def make_traceplot1(data, title_str=None, out_file=None):
    az.plot_trace(xr.Dataset(data))
    fig = plt.gcf()
    if title_str is not None:
        fig.suptitle(title_str, fontsize=20)
    if out_file is not None:
        plt.savefig(out_file, dpi=400, bbox_inches='tight')
        plt.show()


def make_traceplot(data, fnames, title_str=None, out_dir=None, copy_dst=None):
    sns.set_style('white')
    for idx, (key, d) in enumerate(data.items()):
        fig, ax = plt.subplots()
        steps = d['steps']
        charges = d['charges']
        fname = fnames[key]
        if charges.shape[0] > 4:
            charges = charges[:4, :]
        for jdx, q in enumerate(charges):
            ax.plot(steps, np.around(q) + 5 * jdx,
                    marker='', ls='-')  # , alpha=0.6)
        ax.set_title(key, fontsize='large')
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.xmargin: 0
        #  axes[idx].set_ylabel(r"$\mathcal{Q}$", fontsize='large')
        #  axPres.yaxis.set_label_coords(-0.1,1.02)
        ax.yaxis.set_label_coords(-0.01, 1.02)
        ax.set_ylabel(r"$\mathcal{Q}$", fontsize='x-large',
                      rotation='horizontal')
        ax.set_xlabel('MC Step', fontsize='x-large')
        if title_str is not None:
            fig.suptitle(title_str, fontsize='x-large', y=1.02)
        plt.tight_layout()
        #  sns.despine(left=False, top=True, right=True, trim=True)
        #  plt.grid(False)
        if out_dir is not None:
            out_file = os.path.join(out_dir, fname.replace('.', ''))
            out_file = f'{out_file}.png'
            io.log(f'Saving figure to: {out_file}.')
            plt.savefig(out_file, dpi=400, bbox_inches='tight')
            if copy_dst is not None:
                io.check_else_make_dir(copy_dst)
                io.log(f'Copying figure to: {copy_dst}.')
                _ = shutil.copy2(out_file, copy_dst, follow_symlinks=True)
        #  plt.show()


def make_charge_plots(base_dirs):
    for base_dir in base_dirs:
        io.log(80 * '*')
        io.log(f'base_dir: {base_dir}')
        log_dirs = get_log_dirs(base_dir)
        for log_dir in log_dirs:
            io.log(80 * '=')
            io.log(f'log_dir: {log_dir}')
            params = io.loadz(os.path.join(log_dir, 'params.z'))
            filter_strs = get_unique_betas(log_dir)
            lf = params.get('num_steps', None)
            for filter_str in filter_strs:
                io.log(80 * '-')
                io.log(f'filter_str: {filter_str}')
                run_dirs = find_run_dirs(log_dir, filter_str)
                data, tunn_rates, fnames = get_charges(run_dirs, {}, {}, {})
                title_str = get_title_str(params, filter_str)
                if tunn_rates != {}:
                    figs_dir = os.path.join(log_dir, 'figures_np',
                                            'tunneling_rate_plots', filter_str)
                    io.check_else_make_dir(figs_dir)
                    dfile = os.path.join(figs_dir,
                                         f'{filter_str}_tunn_rates.png')
                    copy_dst_dp = copy_dst_tp = None
                    if lf is not None:
                        if 'DLHMC' in base_dir:
                            src_str = 'cooley_logs'
                        elif 'gce-project' in base_dir:
                            src_str = 'gce_logs'
                        elif 'gauge_logs/cooley_logs' in base_dir:
                            src_str = 'cooley_logs_local'
                        elif 'gauge_logs/gce_logs' in base_dir:
                            src_str = 'gce_logs_local'
                        else:
                            src_str = 'local'

                        _, tail = os.path.split(log_dir)
                        dname = f'lf{lf}_{filter_str}'.replace('.', '')
                        copy_dst_dp = os.path.join(GAUGE_LOGS_DIR,
                                                   'tunneling_rate_plots',
                                                   src_str, dname, tail)

                        copy_dst_tp = os.path.join(copy_dst_dp, 'traceplots')
                        io.check_else_make_dir(copy_dst_dp)
                        io.check_else_make_dir(copy_dst_tp)
                        io.save_dict(params, copy_dst_dp, name='params')
                    make_density_plots(tunn_rates, title_str,
                                       dfile, copy_dst_dp)
                    figs_dir_tp = os.path.join(figs_dir, 'traceplots')
                    io.check_else_make_dir(figs_dir_tp)
                    make_traceplot(data, fnames, title_str,
                                   figs_dir_tp, copy_dst_tp)
                io.log(80 * '=')
                io.log('\n')
            io.log(80 * '*')
            io.log('\n')
