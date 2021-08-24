import os
import sys
import platform
import time
import datetime

from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy

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
from utils.logger import Logger

logger = Logger()

sns.set_palette('bright')

THETA_LOCAL = os.path.abspath('/Users/saforem2/thetaGPU')
THETA_REMOTE = os.path.abspath('/lus/theta-fs0/projects/DLHMC/thetaGPU')
WSTR = f'[yellow]WARNING[/yellow]'

sns.set_style('whitegrid')
sns.set_palette('bright')

# -- Check if we're running on OSX (w/ appropriate LaTeX env)
if 'Darwin' in platform.system():
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=(
        r"""
        \usepackage{amsmath}
        \usepackage[sups]{XCharter}
        \usepackage[scaled=1.04,varqu,varl]{inconsolata}
        \usepackage[type1]{cabin}
        \usepackage[charter,vvarbb,scaled=1.07]{newtxmath}
        \usepackage[cal=boondoxo]{mathalfa}
        """
    ))


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
    #  logger.info(f'Looking in {p}...')
    logger.info(f'Looking in: {p}')
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
        logger.info(f'len(hdirs): {len(hdirs)}')
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
        logger.info(f'len(ldirs): {len(ldirs)}')
        if os.path.isdir(d):
            ldirs += [x for x in Path(d).rglob(f'*{glob}*') if x.is_dir()]

    return ldirs


def load_from_dir(d, fnames=None):
    if fnames is None:
        fnames = ['dq', 'charges']

    darr = [x for x in Path(d).iterdir() if x.is_dir()]

    data = {}
    for p in darr:
        for fname in fnames:
            data[fname] = {}
            files = p.rglob(f'*{fname}*')
            if len(files) > 0:
                for f in files:
                    x = io.loadz(f)
                    data[fname] = x

    return data


# pylint:disable=invalid-name,too-many-locals
def load_charges_from_dir(
        d: str,
        hmc: bool = False,
        px_cutoff: float = None
):

    """Load charge data from `d`."""
    logger.info(f'Looking in {d}...')

    if not os.path.isdir(os.path.abspath(d)):
        logger.info(', '.join([
            'WARNING: Skipping entry!',
            f'{d} is not a directory.',
        ]))

        return None

    if 'old' in str(d):
        return None

    if 'inference_hmc' in str(d) and not hmc:
        return None

    qfs = [x for x in Path(d).rglob('charges.z') if x.is_file()]
    pxfs = [x for x in Path(d).rglob('accept_prob.z') if x.is_file()]
    rpfs = [x for x in Path(d).rglob('run_params.z') if x.is_file()]
    num_runs = len(qfs)

    if num_runs == 0:
        return None

    output_arr = []
    for idx, (qf, pxf, rpf) in enumerate(zip(qfs, pxfs, rpfs)):
        params = io.loadz(rpf)
        beta = params['beta']
        lf = params['num_steps']
        run_dir, _ = os.path.split(rpf)

        if px_cutoff is not None:
            px = io.loadz(pxf)
            midpt = px.shape[0] // 2
            px_avg = np.mean(px[midpt:])
            if px_avg < px_cutoff:
                logger.info(', '.join([
                    f'{WSTR}: Bad acceptance prob.',
                    f'px_avg: {px_avg:.3g} < 0.1',
                    f'dir: {d}',
                ]))

                return None

        if 'xeps' and 'veps' in params.keys():
            xeps = np.mean(params['xeps'])
            veps = np.mean(params['veps'])
            eps = np.mean([xeps, veps])

        else:
            eps = params.get('eps', None)
            if eps is None:
                raise ValueError('Unable to determine eps.')

        #  eps = tf.reduce_mean(eps).numpy()
        logger.info('Loading data for: ' + ', '.join([
            f'beta: {str(beta)}', f'lf: {str(lf)}',
            f'eps: {str(eps)}', f'run_dir: {run_dir}',
        ]))

        charges = io.loadz(qf)
        charges = np.array(charges, dtype=int)
        output = {
            'beta': beta,
            'lf': lf,
            'eps': eps,
            'traj_len': lf * eps,
            'qarr': charges,
            'run_params': params,
            'run_dir': run_dir,
        }
        output_arr.append(output)

    return output_arr


# pylint:disable=invalid-name
def load_charge_data(dirs, hmc=False, px_cutoff=None):
    """Load in charge data from `dirs`."""
    data = {}
    #  dirmap = {}
    for d in dirs:
        output_arr = load_charges_from_dir(d, hmc, px_cutoff)
        if output_arr is None:
            continue

        if len(output_arr) > 1:
            for out in output_arr:
                beta = out['beta']
                lf = out['lf']
                eps = out['eps']
                if out[beta] not in data.keys():
                    data[beta] = {}
                elif (lf, eps) not in data[beta].keys():
                    data[beta][(lf, eps)] = out

                if out['beta'] not in data.keys():
                    data[beta] = {}

                if (lf, eps) in data[beta].keys():
                    small_delta = 1e-6 * np.random.randn()
                    data[beta][(lf, eps + small_delta)] = out

                else:
                    data[beta][(lf, eps)] = out
    return data


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
    lspace = np.linspace(lower, upper, num_pts)
    N = np.exp(lspace).astype(int)

    chains, draws = y.shape
    acfs = np.zeros((len(N), chains))
    for i, n in enumerate(N):
        autocorrs = [autocorr_fn(y[i][None, :n]) for i in range(chains)]
        acfs[i] = np.array(autocorrs)

    return N, acfs


def calc_autocorr_alt(y: np.ndarray, num_pts: int = 20, nstart: int = 100):
    """Alternative to the above method, `calc_autocorr`.

    Returns:
        N (np.ndarray): Array containing the number of draws for which the
            integrated autocorrelation was calculated.
        acfs (np.ndarray): Array of estimates of integrated autocorrelation
            times.
    """
    N = np.exp(
        np.linspace(np.log(nstart), np.log(y.shape[0]), num_pts).astype(int)
    ).astype(int)
    acfs = np.zeros((len(N), y.shape[1]))
    for i, n in enumerate(N):
        acfs[i] = integrated_autocorr(y[:n, :])

    return N, acfs


def calc_tau_int(
        qdata: dict,
        therm_frac: float = 0.2,
        min_draws: int = 1000,
        min_chains: int = 1,
        num_pts: int = 50,
        nstart: int = 100,
        use_alt_autocorr: bool = False,
):
    tau_int = {key: {} for key, _ in qdata.items()}
    for key, val in qdata.items():
        io.rule(f'beta: {key}')
        for k, v in val.items():
            qarr, _ = therm_arr(v, therm_frac=therm_frac)
            lf = k[0]
            eps = k[1]
            traj_len = lf * eps
            draws, chains = qarr.shape
            if chains < min_chains:
                logger.info(', '.join([
                    'WARNING: Skipping entry!',
                    f'\tchains: {chains} < min_chains: {min_chains}'
                ]))

                continue

            if draws < min_draws:
                logger.info(', '.join([
                    'WARNING: Skipping entry!',
                    f'\tdraws: {draws} < min_draws: {min_draws}'
                ]))

                continue

            t0 = time.time()
            try:
                if use_alt_autocorr:
                    n, tint = calc_autocorr_alt(qarr, num_pts=num_pts,
                                                nstart=nstart)
                else:
                    # NOTE: `calc_autocorr` expects
                    # `input.shape = (chains, draws),` so we pass `qarr.T`
                    n, tint = calc_autocorr(qarr.T, num_pts=num_pts,
                                            nstart=nstart)
            except ZeroDivisionError:
                logger.info(f'ZeroDivisionError, skipping! ({key}, {k})')
                continue

            # -- Only keep finite estimates
            if np.isfinite(tint[-1].mean()):
                logger.info(', '.join([
                    f'lf: {lf}',
                    f'eps: {eps:.3g}',
                    f'traj_len: {traj_len:.3g}',
                    f'chains: {chains}',
                    f'draws: {draws}',
                    f'tau_int: {tint[-1].mean():.3g}',
                    f'+/- {tint[-1].std():.3g}',
                    f'took: {time.time() - t0:.3g}s',
                ]))
                tau_int[key][k] = {
                    'lf': lf,
                    'eps': eps,
                    'traj_len': k[0] * k[1],
                    'chains': chains,
                    'draws': n,
                    'tau_int': tint,
                }

            else:
                logger.info(', '.join([
                    'WARNING: Skipping entry!',
                    f'\tint[-1] is np.nan'
                ]))

    # -- Sort data by trajectory length, k[0] = (lf, eps)
    data = {}
    for key, val in tau_int.items():
        vsort = dict(sorted(val.items(), key=lambda k: k[0][0] * k[0][1]))
        for k, v in vsort.items():
            data[key] = {
                (k[0],  k[1]): {
                    'lf': v['lf'],
                    'eps': v['eps'],
                    'traj_len': v['traj_len'],
                    'draws': v['draws'],
                    'chains': v['chains'],
                    'tau_int': v['tau_int'],
                }
            }

    return data


def get_plot_data(
        tint_data: dict,
        ndata: dict,
        cmap: str = 'Greys',
        skip: int = 10,
        hcolors: dict = None
):
    """Get data for plotting."""
    betas = list(tint_data.keys())

    template = {
        beta: {
            'hmc': {},
            'l2hmc': {},
        } for beta in betas
    }
    tint_nsamples = deepcopy(template)
    tint_traj_len = deepcopy(template)
    tint_beta = deepcopy(template)

    if hcolors is None:
        hdata = {
            key: val['hmc'] for key, val in tint_data.items()
        }
        hcolors = make_hcolors(hdata, cmap, skip=skip)

    for idx, (beta, d) in enumerate(tint_data.items()):
        hmc_avgs = []
        hmc_errs = []
        for (lf, eps), tint in d['hmc'].items():
            # -- Scale tau_int by number of leapfrog steps:
            tint_scaled = tint * lf
            #  draws = d['hmc'][(lf, eps)]['draws']

            # Compute statistics across chains (axis=1)
            chain_avg = np.mean(tint_scaled, axis=1)
            # error scaling reference:
            # https://faculty.washington.edu/stuve/log_error.pdf
            chain_err = np.std(tint_scaled, axis=1) * (0.434 / chain_avg)
            chain_len = ndata[beta]['hmc'][(lf, eps)]
            tint_nsamples[beta]['hmc'][(lf, eps)] = {
                'x': chain_len,
                'y': chain_avg,
                'yerr': chain_err,
                'color': hcolors[beta][(lf, eps)],
            }

            # Take best (last) value of the estimate for tau_int
            tint_traj_len[beta]['hmc'][(lf, eps)] = {
                'x': lf * eps,
                'y': np.mean(tint_scaled[-1]),
                'yerr': np.std(tint_scaled[-1]),
                'color': hcolors[beta][(lf, eps)],
            }

            avg = np.mean(tint_scaled)
            err = np.std(tint_scaled) / (0.434 / chain_avg)
            hmc_avgs.append(avg)
            hmc_errs.append(err)

        tint_beta[beta]['hmc'] = {
            'x': beta,
            'y': np.mean(hmc_avgs),
            'yerr': np.mean(hmc_errs),
        }

        l2hmc_avgs = []
        l2hmc_errs = []
        for (lf, eps), tint in d['l2hmc'].items():
            # -- Repeat calculation as above
            tint_scaled = tint * lf

            chain_len = ndata[beta]['l2hmc'][(lf, eps)]
            chain_avg = np.mean(tint_scaled, axis=1)
            chain_err = np.std(tint_scaled, axis=1) * (0.434 / chain_avg)
            tint_nsamples[beta]['l2hmc'][(lf, eps)] = {
                'x': chain_len,
                'y': chain_avg,
                'yerr': chain_err,
            }

            best_avg = np.mean(tint_scaled[-1])
            best_err = np.std(tint_scaled[-1]) * (0.434 / best_avg)
            tint_traj_len[beta]['l2hmc'][(lf, eps)] = {
                'x': lf * eps,
                'y': best_avg,
                'yerr': best_err,
            }

            l2hmc_avgs.append(np.mean(tint))
            l2hmc_errs.append(np.std(tint))

        tint_beta[beta]['l2hmc'] = {
            'x': beta,
            'y': np.mean(l2hmc_avgs),
            'yerr': np.mean(l2hmc_errs),
        }

    return tint_nsamples, tint_traj_len, tint_beta


def calc_tau_int_vs_draws(
        qarr: np.ndarray,
        num_pts:int = 20,
        nstart=1000,
        therm_frac=0.2
):
    qarr, _ = therm_arr(qarr, therm_frac=therm_frac)
    n, tint = calc_autocorr(qarr.T, num_pts=num_pts, nstart=nstart)
    return {'q': qarr, 'narr': n, 'tint': tint}


def _get_important_params(path):
    """Get important parameters by finding and loading from `run_params.z`."""
    params_file = os.path.join(path, 'run_params.z')
    try:
        params = io.loadz(params_file)
    except FileNotFoundError as err:
        logger.info(f'Unable to locate {params_file}.')
        raise err

    lf = params.get('num_steps', None)
    beta = params.get('beta', params.get('beta_final', None))
    xeps = params.get('xeps', None)
    traj_len = params.get('traj_len', None)
    if xeps is None:
        eps = params.get('eps', params.get('eps_avg', None))
        if eps is None:
            raise ValueError(f'Unable to determine `eps`.')
        eps = np.mean(eps)
        traj_len = lf * eps
    else:
        eps = np.mean(xeps)
        traj_len = np.sum(xeps)

    important_params = {
        'lf': lf,
        'eps': eps,
        'traj_len': traj_len,
        'beta': beta,
        'run_params': params,
    }

    return important_params


def _deal_with_new_data(tint_file: str, save: bool = False):
    loaded = io.loadz(tint_file)
    if 'tau_int_data' in tint_file:
        head = os.path.dirname(os.path.split(tint_file)[0])
    else:
        head, _ = os.path.split(str(tint_file))
    tint_arr = loaded.get('tint', loaded.get('tau_int', None))
    narr = loaded.get('narr', loaded.get('draws', None))
    if tint_arr is None or narr is None:
        raise ValueError(f'Unable to load from {tint_file}.')

    try:
        params = _get_important_params(head)
    except (ValueError, FileNotFoundError) as err:
        logger.info(f'Error loading params from {head}, skipping!')
        raise err

    lf = params.get('lf', None)
    eps = params.get('eps', None)
    traj_len = params.get('traj_len', None)
    beta = params.get('beta', None)
    data = {
        'tint': tint_arr,
        'narr': narr,
        'run_dir': head,
        'lf': lf,
        'eps': eps,
        'traj_len': traj_len,
        'beta': beta,
        'run_params': params.get('run_params', None),
    }
    if save:
        outfile = os.path.join(head, 'tint_data.z')
        logger.info(f'Saving tint data to: {outfile}.')
        io.savez(data, outfile, 'tint_data')

    return data


def deal_with_new_data(path: str, save: bool = False):
    tint_data = {}
    tint_files = [
        x for x in Path(path).rglob('*tint_data.z*') if x.is_file()
    ]
    tau_int_files = [
        x for x in Path(path).rglob(f'*tau_int_data.z*') if x.is_file()
    ]
    bad_dirs = {}
    for tint_file in tint_files:
        try:
            tint = _deal_with_new_data(tint_file, save=save)
        except (ValueError, FileNotFoundError) as err:
            logger.info(f'Unable to get tint data from: {tint_file}, skipping!')
            continue

        beta = tint['beta']
        traj_len = tint['traj_len']
        run_dir = tint['run_dir']
        if beta not in tint_data:
            tint_data[beta] = {
                tint['traj_len']: tint
            }
        else:
            if traj_len not in tint_data[beta]:
                tint_data[beta][traj_len] = tint
            else:
                # If traj_len is already in tint_data[beta], jitter it by some
                # small amount to create new entry in `tint_data[beta]`
                #  traj_len_modified = traj_len + 1e-6 * np.random.randn()
                traj_len += 1e-6 * np.random.randn()
                tint_data[beta][traj_len] = tint

        if save:
            outfile = os.path.join(run_dir, 'tint_data.z')
            logger.info(f'Saving tint_data to: {outfile}.')
            io.savez(tint, outfile, 'tint_data')

    for tau_int_file in tau_int_files:
        try:
            tint = _deal_with_new_data(tau_int_file, save=save)
        except (ValueError, FileNotFoundError) as err:
            logger.info(f'Unable to get tint data from: {tau_int_file}, skipping!')
            continue
        beta = tint['beta']
        traj_len = tint['traj_len']
        run_dir = tint['run_dir']
        if beta not in tint_data:
            tint_data[beta] = {
                tint['traj_len']: tint
            }
        else:
            if traj_len not in tint_data[beta]:
                tint_data[beta][traj_len] = tint
            else:
                traj_len += 1e-6 * np.random.randn()
                tint_data[beta][traj_len] = tint
        if save:
            outfile = os.path.join(run_dir, 'tint_data.z')
            logger.info(f'Saving tint data to: {outfile}.')
            io.savez(tint, outfile, 'tint_data')

    return tint_data


def deal_with_old_data(path: str, save: bool = False):
    tint_data = {}
    tau_int_files = [
        x for x in Path(path).rglob('*tau_int_data.z*') if x.is_file()
    ]
    for tau_int_file in tau_int_files:
        pass


def look_for_tint_data(path: str, save: bool = False):
    tint_data = {}
    new_data = deal_with_new_data(path, save=save)
    pass




def calc_tau_int_from_dir(
        input_path: str,
        hmc: bool = False,
        px_cutoff: float = None,
        therm_frac: float = 0.2,
        num_pts: int = 50,
        nstart: int = 100,
        make_plot: bool = True,
        save_data: bool = True,
        keep_charges: bool = False,
):
    """
    NOTE: `load_charges_from_dir` returns `output_arr`:
      `output_arr` is a list of dicts, each of which look like:
            output = {
                'beta': beta,
                'lf': lf,
                'eps': eps,
                'traj_len': lf * eps,
                'qarr': charges,
                'run_params': params,
                'run_dir': run_dir,
            }
    """
    output_arr = load_charges_from_dir(input_path, hmc=hmc)
    if output_arr is None:
        logger.info(', '.join([
            'WARNING: Skipping entry!',
            f'\t unable to load charge data from {input_path}',
        ]))

        return None

    tint_data = {}
    for output in output_arr:
        run_dir = output['run_dir']
        beta = output['beta']
        lf = output['lf']
        #  eps = output['eps']
        #  traj_len = output['traj_len']
        run_params = output['run_params']

        if hmc:
            data_dir = os.path.join(input_path, 'run_data')
            plot_dir = os.path.join(input_path, 'plots', 'tau_int_plots')
        else:
            run_dir = output['run_params']['run_dir']
            data_dir = os.path.join(run_dir, 'run_data')
            plot_dir = os.path.join(run_dir, 'plots')

        outfile = os.path.join(data_dir, 'tau_int_data.z')
        outfile1 = os.path.join(data_dir, 'tint_data.z')
        fdraws = os.path.join(plot_dir, 'tau_int_vs_draws.pdf')
        ftlen = os.path.join(plot_dir, 'tau_int_vs_traj_len.pdf')
        c1 = os.path.isfile(outfile)
        c11 = os.path.isfile(outfile1)
        c2 = os.path.isfile(fdraws)
        c3 = os.path.isfile(ftlen)
        if c1 or c11 or c2 or c3:
            loaded = io.loadz(outfile)
            output.update(loaded)
            logger.info(', '.join([
                'WARNING: Loading existing data'
                f'\t Found existing data at: {outfile}.',
            ]))
            loaded = io.loadz(outfile)
            n = loaded.get('draws', loaded.get('narr', None))
            tint = loaded.get('tau_int', loaded.get('tint', None))
            output.update(loaded)

        xeps_check = 'xeps' in output['run_params'].keys()
        veps_check = 'veps' in output['run_params'].keys()
        if xeps_check and veps_check:
            xeps = tf.reduce_mean(output['run_params']['xeps'])
            veps = tf.reduce_mean(output['run_params']['veps'])
            eps = tf.reduce_mean([xeps, veps]).numpy()
        else:
            eps = output['eps']
            if isinstance(eps, list):
                eps = tf.reduce_mean(eps)
            elif tf.is_tensor(eps):
                try:
                    eps = eps.numpy()
                except AttributeError:
                    eps = tf.reduce_mean(eps)

        traj_len = lf * eps

        qarr, _ = therm_arr(output['qarr'], therm_frac=therm_frac)

        n, tint = calc_autocorr(qarr.T, num_pts=num_pts, nstart=nstart)
        #  output.update({
        #      'draws': n,
        #      'tau_int': tint,
        #      'qarr.shape': qarr.shape,
        #  })
        tint_data[run_dir] = {
            'run_dir': run_dir,
            'run_params': run_params,
            'lf': lf,
            'eps': eps,
            'traj_len': traj_len,
            'narr': n,
            'tint': tint,
            'qarr.shape': qarr.shape,
        }
        if save_data:
            io.savez(tint_data, outfile, 'tint_data')
            #  io.savez(output, outfile, name='tint_data')

        if make_plot:
            #  fbeta = os.path.join(plot_dir, 'tau_int_vs_beta.pdf')
            io.check_else_make_dir(plot_dir)
            prefix = 'HMC' if hmc else 'L2HMC'
            xlabel = 'draws'
            ylabel = r'$\tau_{\mathrm{int}}$ (estimate)'
            title = (f'{prefix}, '
                     + r'$\beta=$' + f'{beta}, '
                     + r'$N_{\mathrm{lf}}=$' + f'{lf}, '
                     + r'$\varepsilon=$' + f'{eps:.2g}, '
                     + r'$\lambda=$' + f'{traj_len:.2g}')

            _, ax = plt.subplots(constrained_layout=True)
            best = []
            for t in tint.T:
                _ = ax.plot(n, t, marker='.', color='k')
                best.append(t[-1])

            _ = ax.set_ylabel(ylabel)
            _ = ax.set_xlabel(xlabel)
            _ = ax.set_title(title)

            _ = ax.set_xscale('log')
            _ = ax.set_yscale('log')
            _ = ax.grid(alpha=0.4)
            logger.info(f'Saving figure to: {fdraws}')
            _ = plt.savefig(fdraws, dpi=400, bbox_inches='tight')
            plt.close('all')

            _, ax = plt.subplots()
            for b in best:
                _ = ax.plot(traj_len, b, marker='.', color='k')
            _ = ax.set_ylabel(ylabel)
            _ = ax.set_xlabel(r'trajectory length, $\lambda$')
            _ = ax.set_title(title)
            _ = ax.set_yscale('log')
            _ = ax.grid(True, alpha=0.4)
            logger.info(f'Saving figure to: {ftlen}')
            _ = plt.savefig(ftlen, dpi=400, bbox_inches='tight')
            plt.close('all')

    return tint_data
