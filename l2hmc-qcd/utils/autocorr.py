import os
import sys
import time
import datetime

from pathlib import Path
from dataclasses import dataclass

import autograd
import celerite
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from celerite import terms
from scipy.optimize import minimize

import utils.file_io as io

from utils.data_utils import therm_arr

sns.set_palette('bright')

THETA_LOCAL = os.path.abspath('/Users/saforem2/thetaGPU')
THETA_REMOTE = os.path.abspath('/lus/theta-fs0/projects/DLHMC/thetaGPU')

@dataclass
class ChargeData:
    data_dir: str
    params: dict
    beta: float
    eps: float
    lf: int
    traj_len: float
    accept_prob: np.ndarray

    def __post_init__(self):
        self.traj_len = self.eps * self.lf

@dataclass
class IntAutocorrTimeData:
    charge_data: ChargeData
    num_samples: np.ndarray
    tau_int: np.ndarray



def filter_dict(d: dict, cond: bool, key: str = None):
    """Filter dictionary by applying `cond` to `d`.

    Returns:
        d (dict): New dictionary obtained after applying `cond`.
    """
    if key is not None:
        val = d[key]
        if isinstance(val, dict):
            return {
                k: v for k, v in val.items() if cond
            }
        raise ValueError(f'type(d): {type(d[key])}; expected `dict`')

    return {
        k: v for k, v in d.items() if cond
    }


def look(p: str, s: str, conds: [list, callable] = None):
    """Look in path `p` (recursively) for `matches` containing `s`.

    If `conds` is passed as a list, will only return those matches for which
    `cond(match)` is True.

    Returns:
        matches (list): List of matches.
    """
    io.log(f'Looking in {p}...')
    matches = [x for x in Path(p).rglob(f'*{s}*')]
    if conds is not None:
        if isinstance(conds, (list, tuple)):
            for cond in conds:
                matches = [x for x in matches if cond(x)]
        else:
            matches = [x for x in matches if cond(x)]

    return matches


def get_hmc_dirs(dirs: list = None, glob='HMC_L16'):
    if dirs is None:
        dirs = [
            os.path.join(THETA_LOCAL, 'inference', 'hmc_2020_12_20'),
            os.path.join(THETA_LOCAL, 'hmc', 'l2hmc-qcd', 'logs',
                         'GaugeModel_logs', 'hmc_logs'),
            os.path.join(THETA_REMOTE, 'inference', 'hmc',
                         'l2hmc-qcd', 'logs', 'GaugeModel_logs')
        ]

    hdirs = []
    for d in dirs:
        io.log(f'len(hdirs): {len(hdirs)}')
        if os.path.isdir(d):
            hdirs += [x for x in Path(d).rglob(f'*{glob}*') if x.is_dir()]

    return hdirs


def get_l2hmc_dirs(dirs: list = None, glob: str = None):
    if dirs is None:
        dirs = [
            os.path.join(THETA_LOCAL, 'training'),
            os.path.join(THETA_REMOTE, 'training'),
        ]

    ldirs = []
    for d in dirs:
        io.log(f'len(ldirs): {len(ldirs)}')
        if os.path.isdir(d):
            ldirs += [x for x in Path(d).rglob(f'*{glob}*') if x.is_dir()]

    return ldirs


def load_from_dir(d, fnames=None):
    if fnames is None:
        fnames = ['dq', 'charges']

    darr = [x for x in Path(d).iterdir() if x.is_dir()]

    data = {}
    for d in darr:
        for fname in fnames:
            data[fname] = {}
            files = d.rglob(f'*{fname}*')
            if len(files) > 0:
                for f in files:
                    x = io.loadz(f)
                    data[fname] = x

    return data


def load_charge_data(dirs, hmc=False):
    """Load in charge data from `dirs`."""
    q = {}
    for d in dirs:
        if not os.path.isdir(d):
            io.log('\n'.join([
                'WARNING: Skipping entry!',
                f'\t {d} is not a directory.',
            ]), style='blue')

            continue

        io.log(f'Looking in {d}...')
        if 'inference_hmc' in str(d) and not hmc:
            continue

        qfiles = [x for x in d.rglob('charges.z') if x.is_file()]
        pxfiles = [x for x in d.rglob('accept_prob.z') if x.is_file()]
        rpfiles = [x for x in d.rglob('run_params.z') if x.is_file()]
        #  qfile = d.rglob('charges.z')
        #  pxfile = d.rglob('accept_prob.z')
        #  rpfile = d.rglob('run_params.z')
        num_runs = len(qfiles)
        io.log(f'num_runs: {num_runs}')
        if num_runs > 0:
            for qf, pxf, rpf in zip(qfiles, pxfiles, rpfiles):
                params = io.loadz(rpf)

                # -- Ignore those runs which have poor acceptance
                px = io.loadz(pxf)  # px.shape = (draws, chains)
                midpt = px.shape[0] //2
                px_avg = np.mean(px[midpt:])
                if px_avg < 0.1:
                    io.log('\n'.join([
                        'WARNING: Skipping entry!',
                        f'\t px_avg: {px_avg:.3g} < 0.1',
                    ]), style='blue')

                    continue

                if 'xeps' and 'veps' in params.keys():
                    xeps = tf.reduce_sum(params['xeps'])
                    veps = tf.reduce_mean(params['veps'])
                    eps = tf.reduce_mean([xeps, veps])

                elif 'eps' in params.keys():
                    eps = params['eps']

                else:
                    head, tail = os.path.split(rpf)
                    cfgs_file = os.path.join(head, 'inference_configs.z')
                    if os.path.isfile(cfgs_file):
                        configs = io.loadz(cfgs_file)
                        eps = configs['dynamics_config']['eps']
                    else:
                        raise ValueError('Unable to determine step size eps.')

                beta = params['beta']
                lf = params['num_steps']
                eps = tf.reduce_mean(eps).numpy()
                template = ', '.join([
                    f'beta: {str(beta)}',
                    f'lf: {str(lf)}',
                    f'eps: {eps:.3g}',
                ])
                io.log(f'Loading data for: {template}')
                if beta not in q.keys():
                    q[beta] = {}

                else:
                    qarr = io.loadz(qf)
                    qarr = np.array(qarr, dtype=int)

                    if (lf, eps) not in q[beta].keys():
                        q[beta][(lf, eps)] = qarr
                    else:
                        small_delta = 1e-6 * np.random.randn()
                        q[beta][(lf, eps + small_delta)] = qarr
    return q


def integrated_autocorr(x: np.ndarray, size: str = 'cbrt'):
    """Estimate the integrated autocorrelation time, of a time series.

    Args:
      x (np.ndarray): shape = (chains, draws), with the time series
          along axis 0.
      size (str: {'sqrt', 'cbrt'}): The batch size. The default value is
          "sqroot", which uses the square root of the sample size. "cuberoot"
          will choose the function to use for the cube root of the sample size.
          A numeric value may be provided if neither "sqrt" nor "cbrt" is
          satisfactory.

    Returns:
        tau_int (np.ndarray, shape=(num_dims)): The estimated integrated
            autocorrelation time of each dimension in `x`, considered
            independently.

    References:
    .. [1] Flegal, J. M., Haran, M. and Jones, G. L. (2008) Markov chain Monte
       Carlo: Can we trust the third significant figure? Statistical Science,
       23, 250-260.

    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if size == 'sqrt':
        batch_size = int(np.sqrt(x.shape[0]))
    elif size == 'cbrt':
        batch_size = int(np.sqrt(x.shape[0]))

    elif np.isscalar(size):
        batch_size = size

    bigvar = np.var(x, axis=0, ddof=1)
    # leave off the extra bit at the end that's not a clean multiple
    x = x[:batch_size*(len(x)//batch_size)]
    bigmean = np.mean(x, axis=0)
    sigma2_bm = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        # reshape into the batches, and then compute the batch means
        bm = x[:, j].reshape(-1, batch_size).mean(axis=1)
        sigma2_bm[j] = (
            batch_size / (len(bm) - 1) * np.sum((bm - bigmean[j]) ** 2)
        )

    return sigma2_bm / bigvar


def autocorr_ml(y, thin=1, c=5.0):
    """Compute the autocorrelation using a GP model."""
    # Compute the initial estimate of tau using the standard method
    init = autocorr_new(y, c=c)
    z = y[:, ::thin]
    N = z.shape[1]

    # Build the GP model
    tau = max(1.0, init / thin)
    bounds = [(-5.0, 5.0), (-np.log(N), 0.0)]
    kernel = terms.RealTerm(
        np.log(0.9 * np.var(z)),
        -np.log(tau),
        bounds=bounds
    )
    kernel += terms.RealTerm(
        np.log(0.1 * np.var(z)),
        -np.log(0.5 * tau),
        bounds=bounds,
    )

    gp = celerite.GP(kernel, mean=np.mean(z))
    gp.compute(np.arange(z.shape[1]))

    # Define the objective
    def nll(p):
        # Update the GP model
        gp.set_parameter_vector(p)
        # Loop over the chains and compute the likelihoods
        v, g = zip(*(gp.grad_log_likelihood(z0, quiet=True) for zo in z))
        # Combine the datasets
        return -np.sum(v), -np.sum(g, axis=0)

    # Optimize the model
    p0 = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    soln = minimize(nll, p0, jac=True, bounds=bounds)
    gp.set_parameter_vector(soln.x)

    # Compute the maximum likelihood tau
    a, c = kernel.coefficients[:2]
    tau = thin * 2 * np.sum(a / c) / np.sum(a)

    return tau


def autocorr_gw2010(y, c=5.0):
    """Compute tau_int from Goodman & Weare (2010)."""
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)

    return taus[window]


def next_power_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError('Invalid dimensions for 1D autocorrelation fn.')

    n = next_power_two(len(x))
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4 * n

    if norm:
        acf /= acf[0]

    return acf


def auto_window(taus, c):
    """Automated windowing procedure following Sokal (1989)."""
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)

    return len(taus) - 1


def autocorr_new(y, c=5.0):
    """Compute tau_int using newer technique.

    NOTE: Samples `y` should be aligned along the first dimension, so that
    ```python
    num_samples = y.shape[1]
    ```
    """
    f = np.zeros(y.shape[1])
    for idx, yy in enumerate(y):
        out = autocorr_func_1d(yy)
        f += out

    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)

    return taus[window]


def calc_autocorr(y, num_pts=20, nstart=100, autocorr_fn=autocorr_new):
    """Compute the integrated autocorrelation time vs. num_samples."""
    lower = np.log(nstart)
    upper = np.log(y.shape[1])
    lspace = np.linspace(lower, upper, num_pts).astype(int)
    N = np.exp(lspace)

    chains, draws = y.shape
    acfs = np.zeros((len(N), chains))
    for i, n in enumerate(N):
        autocorrs = [autocorr_fn(y[i][None, :n]) for i in range(chains)]
        acfs[i] = np.array(autocorrs)

    return N, acfs


def calc_autocorr_alt(y, num_pts=20, nstart=100):
    """Alternative to the above method, `calc_autocorr`."""
    N = np.exp(
        np.linspace(np.log(nstart), np.log(y.shape[0]), num_pts).astype(int)
    )
    acfs = np.zeros((len(N), y.shape[1]))
    for i, n in enumerate(N):
        acfs[i] = integrated_autocorr(y[:n, :])

    return N, acfs


def calc_tau_int(
        qdata: dict,
        therm_frac: float = 0.2,
        min_draws: int = 1,
        min_chains: int = 1,
        num_pts: int = 50,
        autocorr_fn: callable = autocorr_new
):
    tau_int = {key: {} for key, _ in qdata.items()}
    for key, val in qdata.items():
        io.rule(f'beta: {key}')
        for k, v in val.items():
            qarr, _ = therm_arr(v, therm_frac=therm_frac)
            draws, chains = qarr.shape
            if chains < min_chains:
                io.log('\n'.join([
                    'WARNING: Skipping entry!',
                    f'\tchains: {chains} < min_chains: {min_chains}'
                ]), style='blue')

                continue

            if draws < min_draws:
                io.log('\n'.join([
                    'WARNING: Skipping entry!',
                    f'\tdraws: {draws} < min_draws: {min_draws}'
                ]), style='blue')

                continue

            t0 = time.time()
            try:
                if autocorr_fn == autocorr_new:
                    # NOTE: `calc_autocorr` expects
                    # `input.shape = (chains, draws),` so we pass `qarr.T`
                    n, tint = calc_autocorr(qarr.T, num_pts=num_pts,
                                            nstart=min_draws)
                elif autocorr_fn == integrated_autocorr:
                    n, tint = calc_autocorr_alt(qarr, num_pts=num_pts,
                                                nstart=min_draws)
            except ZeroDivisionError:
                io.log(f'ZeroDivisionError, skipping! ({key}, {k})')
                continue

            io.log(', '.join([
                f'traj_len: {k[0] * k[1]:.3g}',
                f'chains: {chains}',
                f'draws: {draws}',
                f'tau_int: {tint[-1].mean():.3g}',
                f'+/- {tint[-1].std():.3g}',
                f'took: {time.time() - t0:.3g}s',
            ]))

            # -- Only keep finite estimates
            if np.isfinite(tint[-1].mean()):
                tau_int[key][k] = {
                    'draws': n,
                    'lf': k[0],
                    'eps': k[1],
                    'tau_int': tint,
                    'chains': chains,
                    'traj_len': k[0] * k[1],
                }

    # -- Sort data by trajectory length, k[0] = (lf, eps)
    data = {}
    for key, val in tau_int.items():
        vsort = dict(sorted(val.items(), key=lambda k: k[0][0] * k[0][1]))
        for k, v in vsort.items():
            data[key] = {
                (k[0],  k[1]): {
                    'lf': k[0],
                    'eps': k[1],
                    'traj_len': k[0] * k[1],
                    'draws': v['N'],
                    'chains': v['chains'],
                    'tau_int': v['tau_int'],
                }
            }

    return data


def make_hcolors(data, cmap='viridis', skip=0):
    keys = {
        k: list(data[k].keys()) for k in data.keys()
    }
    colors = {}
    for key, val in keys.items():
        if skip > 0:
            num_items = len(val) + skip
        c = plt.get_cmap(cmap, num_items)
        colors[key] = {
            t: c(idx + skip) for idx, t in enumerate(val)
        }

    return colors
