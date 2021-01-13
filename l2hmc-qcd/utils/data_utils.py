"""
data_utils.py

Collection of functions for dealing with data for plotting.

Author: Sam Foreman
Date: 01/27/2020
"""
from __future__ import absolute_import, division, print_function

import os

import arviz as az
import numpy as np
import tensorflow as tf
import pandas as pd
import xarray as xr
from dataclasses import dataclass
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import Dict
from pathlib import Path

import utils.file_io as io
from utils.file_io import timeit
from lattice.utils import u1_plaq_exact
from utils.attr_dict import AttrDict

#  from plotters.plot_utils import get_matching_log_dirs
#  from plotters.plot_observables import get_obs_dict, grid_plot

mpl.style.use('fast')
sns.set_palette('bright')
TLS_DEFAULT = mpl.rcParams['xtick.labelsize']

@dataclass
class RunParams:
    hmc: bool
    run_dir: str
    eps: float
    beta: float
    run_steps: int
    plaq_weight: float
    charge_weight: float
    num_steps: int
    x_shape: tuple
    input_shape: tuple

    def __post__init__(self):
        self.traj_len = self.num_steps * self.eps


@dataclass
class ChargeData:
    q: tf.Tensor
    dq: tf.Tensor
    params: RunParams


# pylint:disable=invalid-name

def filter_dict(d, cond, key=None):
    if key is not None:
        val = d[key]
        if isinstance(val, dict):
            return {
                k: v for k, v in val.items() if cond
            }
        raise ValueError('If passing a key, d[key] must be a dict.')
    return {
        k: v for k, v in d.items() if cond
    }


def _look(p, s, conds=None):
    print(f'Looking in {p}...')
    matches = [x for x in Path(p).rglob(f'*{s}*')]
    if conds is not None:
        if isinstance(conds, (list, tuple)):
            for cond in conds:
                matches = [x for x in matches if cond(x)]
        else:
            matches = [x for x in matches if cond(x)]

    return matches


def _get_dirs(paths, hmc=False):
    def _look(p, s, conds=None):
        print(f'Looking in {p}...')
        matches = [x for x in Path(p).rglob(f'*{s}*')]
        if conds is not None:
            if isinstance(conds, (list, tuple)):
                for cond in conds:
                    matches = [x for x in matches if cond(x)]
            else:
                matches = [x for x in matches if cond(x)]
        return matches

    dirs = []
    if hmc:
        search_str = 'HMC_L16_b'
        conds = (
            lambda x: 'hmc_logs' in str(x),
            lambda x: 'hmc' in str(x).lower()
        )
    else:
        search_str = 'L16_b2048'
        conds = (
            lambda x: 'GaugeModel_logs' in (str(x)),
            lambda x: 'HMC_' not in str(x),
            lambda x: Path(x).is_dir(),
        )

    if isinstance(paths, (list, tuple)):
        for path in paths:
            dirs += _look(path, search_str, conds)

    else:
        dirs = _look(paths, search_str, conds)

    return dirs



def load_from_dir(d, fnames=None):
    if fnames is None:
        fnames = {
            'dq': 'dq.z',
            'charges': 'charges.z',
            'run_params': 'run_params.z'
        }

    darr = [x for x in Path(d).iterdir() if x.is_dir()]
    for rd in darr:
        files = {k: sorted(rd.glob(f'*{v}*')) for k, v in fnames.items()}
        data = {k: io.loadz(v) for k, v in files.items()}

    return data


def load_charge_data(dirs, hmc=False):
    data = {}
    for d in dirs:
        print(f'Looking in dir: {d}...')
        if 'inference_hmc' in str(d):
            print(f'Skipping {str(d)}...')
            continue

        dqfile = sorted(d.rglob('dq.z'))
        qfile = sorted(d.rglob('charges.z'))
        rpfile = sorted(d.rglob('run_params.z'))
        num_runs = len(dqfile)
        if num_runs > 0:
            for dqf, qf, rpf in zip(dqfile, qfile, rpfile):
                params = io.loadz(rpf)

                if 'xeps' and 'veps' in params.keys():
                    xeps = np.array([i.numpy() for i in params['xeps']])
                    veps = np.array([i.numpy() for i in params['veps']])
                    eps = (np.mean(xeps) + np.mean(veps)) / 2.
                elif 'eps' in params.keys():
                    eps = params['eps']

                params['eps'] = eps
                params = RunParams(**params)
                qarr = io.loadz(qf)
                dqarr = io.loadz(dqf)

                print(
                    '...loading data for (beta, num_steps, eps): '
                    f'({params.beta}, {params.num_steps}, {params.eps:.3g})'
                )

                def _update_dict(z, qdata):
                    try:
                        z[params.beta].update({params.traj_len: qdata})
                    except KeyError:
                        z[params.beta] = {params.traj_len: qdata}

                    return z

                charge_data = ChargeData(q=qarr, dq=dqarr, params=params)
                data = _update_dict(data, charge_data)

    return data


def calc_tau_int(data, therm_frac=0.2):
    """Calculate the integrated autocorrelation time."""
    tau_int = {}
    for key, val in data.items():
        tau_int[key] = {}
        for k, v in val.items():
            arr, _ = therm_arr(v, therm_frac=therm_frac)
            arr = arr.T

            chains, draws = arr.shape
            n, tint = calc_autocorr



# reference:
# https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_solutions.md
# ----------
# Problem:
#   100. Compute bootstrapped 95% confidence intervals for the mean of a 1D
#   array X (i.e., resample the elements of an array with replacement N times,
#   compute the mean of each sample, and then compute percentiles over the
#   means).
def bootstrapped_confidence_interval(x: np.ndarray, N: int = 1000):
    idx = np.random.randint(0, x.size, (N, x.size))
    means = x[idx].mean(axis=1)
    confint = np.percentile(means, [2.5, 97.5])

    return confint


# Reference: https://dfm.io/posts/autocorr/
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1

    return i


def autocorr_func_1d(x, norm=True):
    """Compute the autocorrelation function of a 1D chain."""
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError('Invalid dimensions for 1D autocorrelation function.')

    n = next_pow_two(len(x))
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


def auto_window(taus, c):
    """Automated windowing procedure following Sokal (1989)."""
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)

    return len(taus) - 1


def autocorr_gw2010(y, c=5.0):
    """Following the suggestion from Goodman & Weare (2010)."""
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)

    return taus[window]


def autocorr_new(y, c=5.0):
    """New implementation of autocorrelation function."""
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)

    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)

    return taus[window]


def calc_autocorr(x):
    N = np.exp(np.linspace(np.log(100), np.log(y.shape[1]), 20)).astype(int)
    new = np.empty(len(N))
    for i, n in enumerate(N):
        new[i] = autocorr_new(y[:, :n])

    return N, new


def flatten_dict(d):
    """Recursively convert all entries of `d` to be `AttrDict`."""
    if not isinstance(d, AttrDict):
        d = AttrDict(**d)

    for key, val in d.items():
        if isinstance(val, dict):
            if not isinstance(val, AttrDict):
                d[key] = flatten_dict(val)
            else:
                d[key] = AttrDict(**val)

    return d


def _load_inference_data(log_dir, fnames, inference_str='inference'):
    """Helper function for loading inference data from `log_dir`."""
    run_dir = os.path.join(log_dir, inference_str)
    if os.path.isdir(run_dir):
        data_dir = os.path.join(run_dir, 'run_data')
        rp_file = os.path.join(run_dir, 'run_params.z')
        if os.path.isfile(rp_file) and os.path.isdir(data_dir):
            run_params = io.loadz(rp_file)
            key = (run_params['beta'],
                   run_params['eps'],
                   run_params['num_steps'])
            data = [
                io.loadz(os.path.join(data_dir, f'{fname}.z'))
                for fname in fnames
            ]

    return key, data


def load_inference_data(dirs, search_strs, inference_str='inference'):
    data = {
        s: {} for s in search_strs
    }

    for d in dirs:
        print(f'Looking in dir: {d}...')
        run_dir = Path(os.path.join(d, inference_str))
        if run_dir.is_dir():
            run_dirs = [x for x in run_dir.iterdir() if x.is_dir()]
            for rd in run_dirs:
                print(f'...looking in run_dir: {rd}...')
                rp_file = os.path.join(str(rd), 'run_params.z')
                if os.path.isfile(rp_file):
                    params = io.loadz(rp_file)
                    beta = params['beta']
                    eps = params['eps']
                    num_steps = params['num_steps']
                    data_dir = os.path.join(str(rd), 'run_data')
                    if os.path.isdir(data_dir):
                        for search_str in search_strs:
                            dfile = os.path.join(data_dir, f'{search_str}.z')
                            if os.path.isfile(dfile):
                                _data = io.loadz(dfile)
                                try:
                                    data[search_str].update({
                                        (beta, num_steps, eps): _data
                                    })
                                except KeyError:
                                    data[search_str] = {
                                        (beta, num_steps, eps): _data
                                    }

        return data


def _get_l2hmc_dirs(paths, search_str=None):
    """Look for `log_dirs` containing a training/inference run for L2HMC."""
    if search_str is None:
        search_str = '*L16_b*'

    dirs = []
    for path in paths:
        if not isinstance(path, Path):
            path = Path(os.path.abspath(path))

        print(f'Looking in {path}...')
        dirs += [
            x for x in path.rglob(search_str)
            if 'GaugeModel_logs' in str(x)
            and 'HMC_' not in str(x)
            and x.is_dir()
        ]

    return dirs


def get_l2hmc_dirs():
    bd_local = os.path.abspath(
        '/Users/saforem2/thetaGPU/training'
    )
    bd_theta = os.path.abspath(
        '/lus/theta-fs0/projects/DLHMC/thetaGPU/training'
    )

    l2hmc_dirs = []
    if os.path.isdir(bd_local):
        l2hmc_dirs += _get_l2hmc_dirs(bd_local)

    if os.path.isdir(bd_theta):
        l2hmc_dirs += _get_l2hmc_dirs(bd_theta)

    return l2hmc_dirs


def get_hmc_dirs(base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(
            '/lus/theta-fs0/projects/DLHMC/thetaGPU/inference/'
        )
        if not os.path.isdir(base_dir):
            base_dir = os.path.abspath(
                '/Users/saforem2/thetaGPU/inference'
            )
            if not os.path.isdir(base_dir):
                raise FileNotFoundError(f'Unable to locate {base_dir}')
    base_dir = Path(base_dir)
    hmc_dirs = [x for x in hmc_dir.rglob('*HMC_L16*') if x.is_dir()]

    return hmc_dirs


def bootstrap(x, reps=10000):
    n = len(x)
    xb = np.random.choice(x, (n, reps), replace=True)
    yb = xb.mean(axis=0)
    upper, lower = np.percentile(yb, [2.5, 97.5])

    return yb, (lower, upper)


def dq_stats(dq, reps=10000, therm_frac=0.2):
    stats = {}
    for key, val in dq.items():
        for k, v in val.items():
            data = therm_arr(v, therm_frac=therm_frac, ret_steps=False)
            avgs = []
            errs = []
            for chain in data.T:
                avg, (lower, upper) = bootstrap(chain, reps)
                err = np.max([np.abs(avg - lower), np.abs(upper - avg)])
                avgs.append(avg)
                errs.append(err)

            try:
                stats[key].update({
                    k: {
                        'avg': np.mean(avgs),
                        'avg_std': np.std(avgs),
                        'err': np.mean(errs),
                        'err_std': np.std(errs),
                        'min': np.min(data),
                        'max': np.max(data),
                    }
                })
            except KeyError:
                stats[key] = {
                    k: {
                        'avg': np.mean(avgs),
                        'avg_std': np.std(avgs),
                        'err': np.mean(errs),
                        'err_std': np.std(errs),
                        'min': np.min(data),
                        'max': np.max(data),
                    },
                }

    return stats



def autocorrelation_time(x, s, mu, var):
    """Compute the autocorrelation time."""
    b, t, d = x.shape
    act_ = np.zeros([d])
    for i in range(b):
        y = x[i] - mu
        p, n = y[:-s], y[s:]
        act_ += np.mean(p * n, axis=0) / var
    act_ /= b

    return act_


def effective_sample_size(x, mu, var):
    # batch_size, time, dimension
    b, t, d = x.shape
    ess_ = np.ones([d])
    for s in range(1, t):
        p = autocorrelation_time(x, s, mu, var)
        if np.sum(p > 0.05) == 0:
            break
        else:
            for j in range(d):
                if p[j] > 0.05:
                    ess_[j] += 2. * p[j] * (1. - float(s) / t)
    return t / ess_


def batch_means_ess(x):
    """Estimate the ESS.

    We estimate the ESS as the ratio of the variance of the batch means to the
    variance of the chain, [ref](https://arxiv.org/pdf/1011.0175.pdf).

    NOTE: `x` should be a chain with shape: [time_steps, num_chains, dim].
    """
    x = np.transpose(x, [1, 0, 2])
    T, M, D = x.shape
    num_batches = int(np.floor(T ** (1 / 3)))
    batch_size = int(np.floor(num_batches ** 2))
    batch_means = []
    for i in range(num_batches):
        batch = x[batch_size * i:batch_size * i + batch_size]
        batch_means.append(np.mean(batch, axis=0))
    batch_variance = np.var(np.array(batch_means), axis=0)
    chain_variance = np.var(x, axis=0)
    act = batch_size * batch_variance / (chain_variance + 1e-20)

    return 1. / act


def _calc_var_explained(x):
    """Calculate the % variance explained by the singular values of `x`."""
    _, s, _ = np.linalg.svd(x, full_matrices=True)
    return s ** 2 / np.sum(s ** 2)


def calc_var_explained(weights_dict):
    """Calculate the % variance explained by the sv's for each weight mtx."""
    xweights = weights_dict['xnet']
    vweights = weights_dict['vnet']
    var_explained = {}
    for ((xk, xv), (vk, vv)) in zip(xweights.items(), vweights.items()):
        xk_ = f'xnet_{xk}'
        vk_ = f'vnet_{vk}'
        var_explained[xk_] = _calc_var_explained(xv)
        var_explained[vk_] = _calc_var_explained(vv)

    return var_explained


def bootstrap_old(data, n_boot=10000, ci=68):
    """Bootstrap resampling.

    Returns:
        mean (float): Mean of the (bootstrap) resampled data.
        err (float): Standard deviation of the (bootstrap) resampled data.
        data_rs (np.ndarray): Boostrap resampled data.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    step_axis = np.argmax(data.shape)

    samples = []
    for _ in range(int(n_boot)):
        resampler = np.random.randint(0,
                                      data.shape[step_axis],
                                      data.shape[step_axis])
        sample = data.take(resampler, axis=step_axis)
        samples.append(np.mean(sample, axis=step_axis))

    data_rs = np.array(samples)
    mean = np.mean(data_rs)
    err = np.std(data_rs)

    return mean, err, data_rs


def calc_tunneling_rate(charges):
    """Calculate the tunneling rate as Q_{i+1} - Q_{i}."""
    if not isinstance(charges, np.ndarray):
        charges = np.array(charges)

    charges = charges.T if charges.shape[0] > charges.shape[1] else charges
    charges = np.around(charges)
    #  dq = np.abs(charges[:, 1:] - charges[:, :-1])
    dq = np.abs(charges[1:] - charges[:-1])
    tunneling_rate = np.mean(dq, axis=1)

    return dq, tunneling_rate


def therm_arr(arr, therm_frac=0.2, ret_steps=True):
    """Drop first `therm_frac` steps of `arr` to account for thermalization."""
    #  step_axis = np.argmax(arr.shape)
    step_axis = 0
    num_steps = arr.shape[step_axis]
    therm_steps = int(therm_frac * num_steps)
    arr = np.delete(arr, np.s_[:therm_steps], axis=step_axis)
    steps = np.arange(therm_steps, num_steps)

    if ret_steps:
        return arr, steps

    return arr


# pylint:disable = invalid-name
class InferenceData:
    """InferenceData object."""

    def __init__(self, params, run_params, run_data, energy_data):
        self._params = params
        self._run_params = run_params

        self._run_data = run_data
        self._energy_data = self._sort_energy_data(energy_data)

        self._log_dir = params.get('log_dir', None)
        self._params = io.loadz(os.path.join(self._log_dir,
                                             'parameters.pkl'))
        self._train_weights = (
            self._params['x_scale_weight'],
            self._params['x_translation_weight'],
            self._params['x_transformation_weight'],
            self._params['v_scale_weight'],
            self._params['v_translation_weight'],
            self._params['v_transformation_weight'],
        )
        _tws_title = ', '.join((str(i) for i in self._train_weights))
        self._tws_title = f'({_tws_title})'
        self._tws_fname = ''.join((io._strf(i) for i in self._train_weights))

    @staticmethod
    def _sort_energy_data(energy_data):
        ekeys = ['potential_init', 'potential_proposed', 'potential_out',
                 'kinetic_init', 'kinetic_proposed', 'kinetic_out',
                 'hamiltonian_init', 'hamiltonian_proposed', 'hamiltonian_out']
        energy_data = {k: energy_data[k] for k in ekeys}
        return energy_data

    @staticmethod
    def build_energy_dataset(energy_data):
        """Build energy dataset."""
        ed_dict = {}
        for key, val in energy_data.items():
            arr, steps = therm_arr(np.array(val))
            arr = arr.T
            chains = np.arange(arr.shape[0])
            ed_dict[key] = xr.DataArray(arr,
                                        dims=['chain', 'draw'],
                                        coords=[chains, steps])
        dataset = xr.Dataset(ed_dict)

        return dataset

    @staticmethod
    def build_dataset(run_data, run_params):
        """Build dataset from `run_data."""
        rd_dict = {}
        for key, val in run_data.items():
            if 'mask' in key:
                continue

            x, draws = therm_arr(np.array(val))
            x = x.T

            if 'plaqs' in key:
                key = 'plaqs_diffs'
                x = u1_plaq_exact(run_params['beta']) - x

            if 'charges' in key:
                x = np.around(x)

            chains = x.shape[0]
            rd_dict[key] = xr.DataArray(x,
                                        dims=['chain', 'draw'],
                                        coords=[chains, draws])

        rd_dict['charges_squared'] = rd_dict['charges'] ** 2

        charges = rd_dict['charges'].values.T
        _, dq = calc_tunneling_rate(charges)
        dq = dq.T

        chains = np.arange(dq.shape[0])
        rd_dict['tunneling_rate'] = xr.DataArray(dq,
                                                 dims=['chain', 'draw'],
                                                 coords=[chains, draws])

        dataset = xr.Dataset(rd_dict)

        return dataset

    def _plot_setup(self, run_params, idx=None, nw_run=True):
        """Setup for creating plots.

        Returns:
            fname (str): String containing the filename containing info about
                data.
            title_str (str): Title string to set as title of figure.
        """
        eps = run_params['eps']
        beta = run_params['beta']
        net_weights = run_params['net_weights']

        nw_str = ''.join((io.strf(i).replace('.', '') for i in net_weights))
        nws = '(' + ', '.join((str(i) for i in net_weights)) + ')'

        lf = self._params['num_steps']
        fname = f'lf{lf}'
        run_steps = run_params['run_steps']
        fname += f'_steps{run_steps}'
        title_str = (r"$N_{\mathrm{LF}} = $" + f'{lf}, '
                     r"$\beta = $" + f'{beta:.1g}, '
                     r"$\varepsilon = $" + f'{eps:.3g}')
        eps_str = f'{eps:4g}'.replace('.', '')
        fname += f'_e{eps_str}'

        if self._params.get('eps_fixed', False):
            title_str += ' (fixed)'
            fname += '_fixed'

        if any([tw == 0 for tw in self._train_weights]):
            title_str += (', '
                          + r"$\mathrm{nw}_{\mathrm{train}} = $"
                          + f' {self._tws_title}')
            fname += f'_train{self._tws_fname}'

        clip_value = self._params.get('clip_value', 0)
        if clip_value > 0:
            title_str += f', clip: {clip_value}'
            fname += f'_clip{clip_value}'.replace('.', '')

        if nw_run:
            title_str += ', ' + r"$\mathrm{nw}_{\mathrm{run}}=$" + f' {nws}'
            fname += f'_{nw_str}'
            #  fname += f'_{net_weights_str}'

        if idx is not None:
            fname += f'_{idx}'

        return fname, title_str

    @staticmethod
    def _savefig(fig, out_file):
        """Save `fig` to `out_file`."""
        io.log(f'Saving figure to: {out_file}.')
        fig.savefig(out_file, dpi=200, bbox_inches='tight')

    def _plot_posterior(self, data, out_file, title_str,
                        var_names=None, out_file1=None):
        """Plot posterior using `arviz.plot_posterior`."""
        _ = az.plot_posterior(data, var_names=var_names)
        fig = plt.gcf()
        fig.suptitle(title_str, fontsize=14, y=1.05)
        self._savefig(fig, out_file)
        if out_file1 is not None:
            self._savefig(fig, out_file1)

    def _plot_trace(self, data, out_file, title_str,
                    var_names=None, out_file1=None):
        _ = az.plot_trace(data,
                          compact=True,
                          combined=True,
                          var_names=var_names)
        fig = plt.gcf()
        fig.suptitle(title_str, fontsize=14, y=1.05)
        self._savefig(fig, out_file)
        if out_file1 is not None:
            self._savefig(fig, out_file)

    def make_plots(self,
                   run_params,
                   run_data=None,
                   energy_data=None,
                   runs_np=True, out_dir=None):
        """Create trace + KDE plots of lattice observables and energy data."""
        type_str = 'figures_np' if runs_np else 'figures_tf'
        figs_dir = os.path.join(self._log_dir, type_str)
        fig_dir = os.path.join(figs_dir, run_params['run_str'])
        io.check_else_make_dir(fig_dir)

        dataset = None
        energy_dataset = None
        try:
            fname, title_str = self._plot_setup(run_params)
        except FileNotFoundError:
            return dataset, energy_dataset

        tp_fname = f'{fname}_traceplot'
        pp_fname = f'{fname}_posterior'
        rp_fname = f'{fname}_ridgeplot'

        dataset = self.build_dataset(run_data, run_params)

        tp_out_file = os.path.join(fig_dir, f'{tp_fname}.pdf')
        pp_out_file = os.path.join(fig_dir, f'{pp_fname}.pdf')

        var_names = ['tunneling_rate', 'plaqs_diffs']
        if hasattr(dataset, 'dx'):
            var_names.append('dx')
        var_names.extend(['accept_prob', 'charges_squared', 'charges'])

        tp_out_file_ = None
        pp_out_file_ = None
        if out_dir is not None:
            io.check_else_make_dir(out_dir)
            tp_out_file1 = os.path.join(out_dir, f'{tp_fname}.pdf')
            pp_out_file1 = os.path.join(out_dir, f'{pp_fname}.pdf')

        ###################################################
        # Create traceplot + posterior plot of observables
        ###################################################
        self._plot_trace(dataset, tp_out_file,
                         var_names=var_names,
                         out_file1=tp_out_file1)

        self._plot_posterior(dataset, pp_out_file,
                             var_names=var_names,
                             out_file1=pp_out_file1)

        # * * * * * * * * * * * * * * * * *
        # Create ridgeplot of plaq diffs  *
        # * * * * * * * * * * * * * * * * *
        rp_out_file = os.path.join(fig_dir, f'{rp_fname}.pdf')
        _ = az.plot_forest(dataset,
                           kind='ridgeplot',
                           var_names=['plaqs_diffs'],
                           ridgeplot_alpha=0.4,
                           ridgeplot_overlap=0.1,
                           combined=False)
        fig = plt.gcf()
        fig.suptitle(title_str, fontsize='x-large', y=1.025)
        self._savefig(fig, rp_out_file)
        if out_dir is not None:
            rp_out_file1 = os.path.join(out_dir, f'{rp_fname}.pdf')
            self._savefig(fig, rp_out_file1)

        # * * * * * * * * * * * * * * * * * * * * * * * * * *
        # Create traceplot + posterior plot of energy data  *
        # * * * * * * * * * * * * * * * * * * * * * * * * * *
        if energy_data is not None:
            energy_dataset = self.energy_plots(energy_data, run_params,
                                               fname, out_dir=out_dir)

        return dataset, energy_dataset

    def energy_plots(self, energy_data, fname, out_dir=None):
        """Plot energy data from inference run."""
        energy_dataset = self.build_dataset(energy_data, )
        epp_fname = f'{fname}_energy_posterior'
        etp_fname = f'{fname}_energy_traceplot'
        etp_out_file = os.path.join(fig_dir, f'{etp_fname}.pdf')
        epp_out_file = os.path.join(fig_dir, f'{epp_fname}.pdf')

        etp_out_file1 = None
        etp_out_file2 = None
        if out_dir is not None:
            etp_out_file1 = os.path.join(out_dir, f'{etp_fname}.pdf')
            epp_out_file1 = os.path.join(out_dir, f'{epp_fname}.pdf')

        self._plot_trace(energy_dataset, etp_out_file,
                         out_file1=etp_out_file1)
        self._plot_posterior(energy_dataset, epp_out_file,
                             out_file1=epp_out_file1)

        return energy_dataset


# pylint: disable=invalid-name
class DataLoader:
    """DataLoader object."""

    def __init__(self,
                 log_dir=None,
                 n_boot=5000,
                 therm_frac=0.25,
                 nw_include=None,
                 calc_stats=True,
                 filter_str=None,
                 runs_np=False):
        """Initialization method."""
        self._log_dir = log_dir
        self._n_boot = n_boot
        self._therm_frac = therm_frac
        self._nw_include = nw_include
        self._calc_stats = calc_stats
        self.run_dirs = io.get_run_dirs(log_dir, filter_str, runs_np)
        self._params = io.loadz(os.path.join(self._log_dir,
                                             'parameters.pkl'))
        self._train_weights = (
            self._params['x_scale_weight'],
            self._params['x_translation_weight'],
            self._params['x_transformation_weight'],
            self._params['v_scale_weight'],
            self._params['v_translation_weight'],
            self._params['v_transformation_weight'],
        )
        _tws_title = ', '.join((str(i) for i in self._train_weights))
        self._tws_title = f'({_tws_title})'
        self._tws_fname = ''.join((io.strf(i) for i in self._train_weights))

    def _load_sqz(self, fname):
        data = io.loadz(os.path.join(self._obs_dir, fname))
        return np.squeeze(np.array(data))

    def _get_dx(self, fname):
        dx_file = os.path.join(self._obs_dir, fname)
        if os.path.isfile(dx_file):
            dx = self._load_sqz(fname)

    def _stats(self, arr, axis=0):
        """Calculate statistics using `bootstrap` resampling along `axis`."""
        _, _, arr = bootstrap(arr, n_boot=self._n_boot)
        return arr.mean(axis=axis).flatten(), arr.std(axis=axis).flatten()

    def _get_observables_bs(self, data, run_params):
        data_bs = {}
        for key, val in data.items():
            if val is None:
                continue
            avg, err = self._stats(val, axis=0)
            err_key = f'{key}_err'
            data_bs[key] = avg
            data_bs[err_key] = err

        entries = len(avg.flatten())
        net_weights = tuple(run_params['net_weights'])
        data_bs['run_dir'] = np.array([run_dir for _ in range(entries)])
        data_bs['net_weights'] = tuple([net_weights for _ in range(entries)])
        data_bs['log_dir'] = np.array([self._log_dir for _ in range(entries)])

        return pd.DataFrame(data_bs)

    def get_observables(self, run_dir=None):
        """Get all observables from inference_data in `run_dir`."""
        run_params = io.loadz(os.path.join(run_dir, 'run_params.pkl'))
        beta = run_params['beta']
        net_weights = tuple([int(i) for i in run_params['net_weights']])

        keep = True
        if self._nw_include is not None:
            keep = net_weights in self._nw_include

        # If none (< 10 %) of the proposed configs are rejected,
        # don't bother loading data and calculating statistics.
        px = self._load_sqz('px.pkl')
        avg_px = np.mean(px)
        if avg_px < 0.1 or not keep:
            io.log(f'Skipping! nw: {net_weights}, avg_px: {avg_px:.3g}')
            return None, run_params

        io.log(f'Loading data for net_weights: {net_weights}...')
        io.log(f'  run_dir: {run_dir}')

        # load chages, plaqs data
        charges = self._load_sqz('charges.pkl')
        plaqs = self._load_sqz('plaqs.pkl')
        dplq = u1_plaq_exact(beta) - plaqs

        # thermalize configs
        px, _ = therm_arr(px, self._therm_frac)
        dplq, _ = therm_arr(dplq, self._therm_frac)
        charges, _ = np.insert(charges, 0, 0, axis=0)
        charges, _ = therm_arr(charges)
        dq, _ = calc_tunneling_rate(charges)
        dq = dq.T

        dx = self._get_dx('dx.pkl')
        dxf = self.get_dx('dxf.pkl')
        dxb = self._get_dx('dxb.pkl')
        observables = {
            'plaqs_diffs': dplq,
            'accept_prob': px,
            'tunneling_rate': dq,
        }
        _names = ['dx', 'dxf', 'dxb']
        _vals = [dx, dxf, dxb]
        for name, val in zip(_names, _vals):
            if val is not None:
                observables[name] = val

        return observables

    def _build_dataframes(self, observables, run_params):
        """Build dataframes from `observables`."""
        data_bs = None
        if self._calc_stats:
            data_bs = self._get_data_bs(observables, run_params)

        data = {}
        for key, val in observables.items():
            data[key] = val.flatten()

        entries = len(dq.flatten())
        data = {key: val.flatten() for (key, val) in observables.items()}
        data.update({
            'run_dir': np.array([run_dir for _ in range(entries)]),
            'net_weights': tuple([net_weights for _ in range(entries)]),
            'log_dir': np.array([self._log_dir for _ in range(entries)]),
        })

        data = pd.DataFrame(data)

        return data, data_bs, run_params

    def build_dataframes(self, data=None, data_bs=None, **kwargs):
        """Build `pd.DataFrames` containing all inference data from `run_dirs`.

        Args:
            run_dirs (array-like): List of run_dirs in which to look for
                inference data.
            data (pd.DataFrame): DataFrame containing inference data. If `data
                is not None`, the new `pd.DataFrame` will be appended to
                `data`.
            data_bs (pd.DataFrame): DataFrame containing bootstrapped inference
                data. If `data_bs is not None`, the new `pd.DataFrame` will be
                appended to `data_bs`.

        Kwargs:
            Passed to `get_observables`.

        Returns:
            data (pd.DataFrame): DataFrame containing (flattened) inference
                data.
            data_bs (pd.DataFrame): DataFrame containing (bootstrapped)
                inference data.
            run_params (dict): Dictionary of parameters used to generate
                inference data.
        """
        run_params = None
        for run_dir in run_dirs:
            if data is not None and hasattr(data, 'run_dir'):
                if not data[data.run_dir == run_dir].empty:
                    continue

            run_params_file = os.path.join(run_dir, 'run_params.pkl')
            if not os.path.isfile(run_params_file):
                run_params = None
                continue

            _data, _data_bs, run_params = self.get_observables(run_dir)

            if data is None:
                data = _data
            else:
                data = pd.concat(
                    [data, _data], axis=0
                ).reset_index(drop=True)

            if data_bs is None:
                data_bs = _data_bs
            else:
                data_bs = pd.concat(
                    [data_bs, _data_bs], axis=0
                ).reset_index(drop=True)

        return data, data_bs, run_params

    def _plot_setup(self, run_params, idx=None, nw_run=True):
        """Setup for creating plots.

        Returns:
            fname (str): String containing the filename containing info about
                data.
            title_str (str): Title string to set as title of figure.
        """
        eps = run_params['eps']
        beta = run_params['beta']
        net_weights = run_params['net_weights']

        nw_str = ''.join((io.strf(i).replace('.', '') for i in net_weights))
        nws = '(' + ', '.join((str(i) for i in net_weights)) + ')'

        lf = self._params['num_steps']
        fname = f'lf{lf}'
        run_steps = run_params['run_steps']
        fname += f'_steps{run_steps}'
        title_str = (r"$N_{\mathrm{LF}} = $" + f'{lf}, '
                     r"$\beta = $" + f'{beta:.1g}, '
                     r"$\varepsilon = $" + f'{eps:.3g}')
        eps_str = f'{eps:4g}'.replace('.', '')
        fname += f'_e{eps_str}'

        if self._params.get('eps_fixed', False):
            title_str += ' (fixed)'
            fname += '_fixed'

        if any([tw == 0 for tw in self._train_weights]):
            title_str += (', '
                          + r"$\mathrm{nw}_{\mathrm{train}} = $"
                          + f' {self._tws_title}')
            fname += f'_train{self._tws_fname}'

        clip_value = self._params.get('clip_value', 0.)
        if clip_value > 0:
            title_str += f', clip: {clip_value}'
            fname += f'_clip{clip_value}'.replace('.', '')

        if nw_run:
            title_str += ', ' + r"$\mathrm{nw}_{\mathrm{run}}=$" + f' {nws}'
            fname += f'_{net_weights_str}'

        if idx is not None:
            fname += f'_{idx}'

        return fname, title_str

    def _savefig(self, fig, out_file):
        """Save `fig` to `out_file`."""
        io.log(f'Saving figure to: {out_file}.')
        fig.savefig(out_file, dpi=200, bbox_inches='tight')

    def _plot_posterior(self, data, out_file, title_str,
                        var_names=None, out_file1=None):
        """Plot posterior using `arviz.plot_posterior`."""
        _ = az.plot_posterior(data, var_names=var_names)
        fig = plt.gcf()
        fig.suptitle(title_str, fontsize=14, y=1.05)
        self._savefig(fig, out_file)
        if out_file1 is not None:
            self._savefig(fig, out_file1)

    def _plot_trace(self, data, out_file, title_str,
                    var_names=None, out_file1=None):
        _ = az.plot_trace(data,
                          compact=True,
                          combined=True,
                          var_names=var_names)
        fig = plt.gcf()
        fig.suptitle(title_str, fontsize=14, y=1.05)
        self._savefig(fig, out_file)
        if out_file1 is not None:
            self._savefig(fig, out_file)

    def inference_plots(self,
                        run_data,
                        run_params,
                        runs_np=True,
                        out_dir=None,
                        energy_data=None):
        """Create trace + KDE plots of lattice observables and energy data."""
        type_str = 'figures_np' if runs_np else 'figures_tf'
        figs_dir = os.path.join(self._log_dir, type_str)
        fig_dir = os.path.join(figs_dir, run_params['run_str'])
        io.check_else_make_dir(fig_dir)

        dataset = None
        energy_dataset = None
        try:
            fname, title_str, _ = self._plot_setup(run_params)
        except FileNotFoundError:
            return dataset, energy_dataset

        tp_fname = f'{fname}_traceplot'
        pp_fname = f'{fname}_posterior'
        rp_fname = f'{fname}_ridgeplot'

        dataset = self.build_dataset(run_data, run_params)

        tp_out_file = os.path.join(fig_dir, f'{tp_fname}.pdf')
        pp_out_file = os.path.join(fig_dir, f'{pp_fname}.pdf')

        var_names = ['tunneling_rate', 'plaqs_diffs']
        if hasattr(dataset, 'dx'):
            var_names.append('dx')
        var_names.extend(['accept_prob', 'charges_squared', 'charges'])

        tp_out_file_ = None
        pp_out_file_ = None
        if out_dir is not None:
            io.check_else_make_dir(out_dir)
            tp_out_file1 = os.path.join(out_dir, f'{tp_fname}.pdf')
            pp_out_file1 = os.path.join(out_dir, f'{pp_fname}.pdf')

        # -- Create traceplot + posterior plot of observables -----------
        self._plot_trace(dataset, tp_out_file,
                         var_names=var_names,
                         out_file1=tp_out_file1)

        self._plot_posterior(dataset, pp_out_file,
                             var_names=var_names,
                             out_file1=pp_out_file1)

        # * * * * * * * * * * * * * * * * *
        # Create ridgeplot of plaq diffs  *
        # * * * * * * * * * * * * * * * * *
        rp_out_file = os.path.join(fig_dir, f'{rp_fname}.pdf')
        _ = az.plot_forest(dataset,
                           kind='ridgeplot',
                           var_names=['plaqs_diffs'],
                           ridgeplot_alpha=0.4,
                           ridgeplot_overlap=0.1,
                           combined=False)
        fig = plt.gcf()
        fig.suptitle(title_str, fontsize='x-large', y=1.025)
        self._savefig(fig, rp_out_file)
        if out_dir is not None:
            rp_out_file1 = os.path.join(out_dir, f'{rp_fname}.pdf')
            self._savefig(fig, rp_out_file1)

        # -- Create traceplot + posterior plot of energy data --------------
        if energy_data is not None:
            energy_dataset = self.energy_plots(energy_data,
                                               fname, out_dir=out_dir)

    def energy_plots(self, energy_data, fname, out_dir=None):
        """Plot energy data from inference run."""
        energy_dataset = self.build_dataset(energy_data)
        epp_fname = f'{fname}_energy_posterior'
        etp_fname = f'{fname}_energy_traceplot'
        etp_out_file = os.path.join(fig_dir, f'{etp_fname}.pdf')
        epp_out_file = os.path.join(fig_dir, f'{epp_fname}.pdf')
        if out_dir is not None:
            etp_out_file1 = os.path.join(out_dir, f'{etp_fname}.pdf')
            epp_out_file1 = os.path.join(out_dir, f'{epp_fname}.pdf')
        self._plot_trace(energy_dataset, etp_out_file)
        self._plot_posterior(energy_dataset, epp_out_file)

        return energy_dataset
