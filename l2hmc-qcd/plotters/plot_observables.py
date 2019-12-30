"""
plot_observables.py

Collection of helper functions for plotting lattice observables for gauge
model.
"""
import os
import pickle

from config import HAS_MATPLOTLIB, COLORS

import numpy as np
import pandas as pd

import utils.file_io as io
from lattice.lattice import u1_plaq_exact
from .plot_utils import get_run_dirs, load_pkl


if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
    import matplotlib.style as mplstyle
    mplstyle.use('fast')
try:
    import seaborn as sns
    sns.set_palette('bright', 100)
    colors = sns.color_palette()
    sns.set_style('ticks', {'xtick.major.size': 8,
                            'ytick.major.size': 8})
    HAS_SEABORN = True

except ImportError:
    HAS_SEABORN = False
    colors = COLORS


def plot_charges(charges, out_file=None, title=None, nrows=2, **kwargs):
    ls = kwargs.get('ls', '-')
    color = kwargs.get('color', 'k')
    lw = kwargs.get('lw', 0.6)

    if not isinstance(charges, np.ndarray):
        charges = np.array(charges)[0]

    if charges.shape[0] > charges.shape[1]:
        charges = charges.T

    batch_size, steps = charges.shape
    N = int(nrows)

    figsize = (6.4 * N, 4.8 * N)
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=figsize)
    for idx, ax in enumerate(axes.flatten()):
        arr_int = np.around(charges[idx])
        _ = ax.plot(charges[idx], ls=ls, color=color, lw=lw)
        _ = ax.plot(arr_int, marker='.', ls='', color='r', zorder=10)

    if title is not None:
        _ = plt.suptitle(title, fontsize=20, y=1.04)

    if out_file is not None:
        fig.tight_layout()
        io.log(f'Saving figure to: {out_file}.')
        _ = fig.savefig(out_file, dpi=200, bbox_inches='tight')

    return fig, axes


def plot_autocorrs(charges, out_file=None, title=None, nrows=4, **kwargs):
    if not isinstance(charges, np.ndarray):
        charges = np.array(charges)[0]

    if charges.shape[0] > charges.shape[1]:
        charges = charges.T

    batch_size, steps = charges.shape
    N = int(nrows)

    figsize = (6.4 * N, 4.8 * N)
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=figsize)
    for idx, ax in enumerate(axes.flatten()):
        qproj = np.around(charges[idx])
        #  try:
        pd.plotting.autocorrelation_plot(qproj, ax=ax, **kwargs)
        #      from statsmodels.graphics.tsaplots import plot_acf
        #      lags = len(qproj) - 2
        #      plot_acf(qproj, ax=ax, use_vlines=True, lags=lags)
        #  except ImportError:

    if title is not None:
        _ = plt.suptitle(title, fontsize=20, y=1.02)

    if out_file is not None:
        fig.tight_layout()
        io.log(f'Saving figure to: {out_file}.')
        _ = fig.savefig(out_file, dpi=200, bbox_inches='tight')

    return fig, axes


def weights_hist(log_dir, weights=None):
    if HAS_SEABORN:
        sns.set_palette('bright', 100)

    if weights is None:
        weights = load_pkl(os.path.join(log_dir, 'weights.pkl'))

    figs_dir = os.path.join(log_dir, 'figures', 'weights')
    io.check_else_make_dir(figs_dir)

    idx = 0
    for key, val in weights.items():
        for k1, v1 in val.items():
            for k2, v2 in v1.items():
                fig, ax = plt.subplots()
                hist_kws = {
                    'density': True,
                    #  'label': f'{key}/{k1}/{k2}',
                    'histtype': 'step',
                    'color': 'C6',
                }
                try:
                    w = v2.w.flatten()
                    b = v2.b.flatten()
                    inc = 2
                    bavg = np.mean(b)
                    berr = np.std(b)
                    blabel = (r"""$\langle b\rangle = $"""
                              + f' {bavg:.3g} +/- {berr:.3g}')
                    #  b = v2.b.flatten()
                except AttributeError:
                    w = v2.flatten()
                    b = None
                    inc = 1
                avg = np.mean(w)
                err = np.std(w)
                label = (r"""$\langle W\rangle = $"""
                         + f' {avg:.3g} +/- {err:.3g}')
                if HAS_SEABORN:
                    try:
                        sns.kdeplot(w, ax=ax, shade=True,
                                    color='C6', label=label)
                    except np.linalg.LinAlgError:
                        io.log(f'LinAlgError raised. Returning.')
                        continue

                _ = ax.hist(w, **hist_kws)
                if b is not None:
                    label = (r"""$\langle b\rangle$"""
                             + f' {avg:.3g} +/- {err:.3g}')
                    if HAS_SEABORN:
                        try:
                            sns.kdeplot(b, ax=ax, shade=True,
                                        color='C7', label=blabel)
                        except np.linalg.LinAlgError:
                            continue
                    _ = ax.hist(b, density=True, color='C7', histtype='step')
                _ = ax.set_title(f'{key}/{k1}/{k2}', fontsize=20)
                _ = ax.legend(loc='best')
                fname = f'{key}_{k1}_{k2}_weights_hist.png'
                out_file = os.path.join(figs_dir, fname)
                fig.tight_layout()
                io.log(f'Saving figure to: {out_file}.')
                fig.savefig(out_file, dpi=200, bbox_inches='tight')
                idx += inc
            plt.close('all')
            plt.clf()


def reset_plots():
    plt.close('all')
    plt.clf()


def load_plaqs(run_dir):
    plaqs_file = os.path.join(run_dir, 'observables', 'plaqs.pkl')
    return load_pkl(plaqs_file, arr=True)


def get_title_str(params, beta=None, eps=None, nw_legend=True):
    ss = params['space_size']
    ts = params['time_size']
    lf_steps = params['num_steps']
    batch_size = params['batch_size']
    clip_value = params.get('clip_value', 0)
    nw_desc = (r'($\alpha_{\mathrm{S_{x}}}$, '
               r'$\alpha_{\mathrm{T_{x}}}$, '
               r'$\alpha_{\mathrm{Q_{x}}}$, '
               r'$\alpha_{\mathrm{S_{v}}}$, '
               r'$\alpha_{\mathrm{T_{v}}}$, '
               r'$\alpha_{\mathrm{Q_{v}}}$)')

    title_str = (f"{ss} x {ts}, "
                 r"$N_{\mathrm{LF}} = $" + f"{lf_steps}, "
                 r"$N_{\mathrm{B}} = $" + f"{batch_size}, ")
    if clip_value > 0:
        title_str += f'clip: {clip_value}, '

    if beta is not None:
        title_str += r"$\beta = $" + f'{beta}, '

    if eps is not None:
        title_str += r"$\varepsilon = $" + f'{eps:.3g}, '

    if nw_legend:
        title_str += f"nw: {nw_desc}"

    return title_str


def get_run_title_str(run_params):
    """Parses `run_params` and returns string detailing parameters."""
    beta = run_params['beta']
    eps = run_params['eps']
    nw = tuple(run_params['net_weights'])
    #  run_str = run_params['run_str']
    title_str = (r"""$\beta = $""" + f'{beta}'
                 + r"""$\varepsilon = $""" + f'{eps:.3g}'
                 + f'nw: {nw}')

    return title_str


def get_obs_dict(log_dir, obs_name, run_dirs=None):
    """Load observable with `obs_name` from each `run_dir` in `run_dirs`.

    Args:
        log_dir (str): Path specifying `log_dir` to look in.
        obs_name (str): Name of observable to load (should match file name e.g.
            in `observables/obs_name.pkl`)
        run_dirs (list): List of run_dirs from which observable should be
            loaded. If not included, will look in all `run_dirs` in
                `log_dir/runs/`.

    Returns:
    """
    if run_dirs is None:
        run_dirs = get_run_dirs(log_dir)

    obs_dict = {}

    for rd in sorted(run_dirs):
        run_params = load_pkl(os.path.join(rd, 'run_params.pkl'))
        nw = tuple(run_params['net_weights'])
        if obs_name == 'plaqs':
            exact = u1_plaq_exact(run_params['beta'])
            try:
                plaqs = load_plaqs(rd)
                obs = exact - plaqs
            except FileNotFoundError:
                continue
        else:
            pkl_file = os.path.join(rd, 'observables', f'{obs_name}.pkl')
            try:
                obs = load_pkl(pkl_file)
            except FileNotFoundError:
                continue

        key = tuple(list(np.array(nw, dtype=int)))
        try:
            obs_dict[key].append(obs)
        except KeyError:
            obs_dict[key] = [obs]

    return obs_dict


def calc_stats(data, therm_frac=0.2, axis=None):
    """Calculate statistics for data.
    
    Args:
        data (np.array): If `len(data.shape) == 2`:, calculate 
    """

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    step_axis = np.argmax(data.shape)
    num_steps = data.shape[step_axis]
    therm_steps = int(therm_frac * num_steps)
    data = np.delete(data, np.s_[:therm_steps], axis=step_axis)

    if axis is None:
        avgs = np.mean(data, axis=-1)
        errs = np.std(data, axis=-1)
    else:
        avgs = np.mean(data, axis=axis)
        errs = np.std(data, axis=axis)

    x = np.arange(therm_steps, num_steps)
    avg = np.mean(avgs)
    err = np.std(errs)

    output = {
        'x': x,
        'avg': avg,
        'err': err,
        'avgs': avgs,
        'errs': errs,
        'data': data,
    }

    return output


def trace_plot(data, ax, color=None, stats=True, **kwargs):
    """Create trace plot of data."""
    if not isinstance(data, dict) and stats:
        data = calc_stats(data, **kwargs)

    if isinstance(data, dict):
        x = data.get('x', None)
        y = data.get('data', None)
        avg = data.get('avg', np.mean(y))
        err = data.get('err', np.std(y))
        avgs = data.get('avgs', np.mean(y, axis=-1))
        errs = data.get('errs', np.std(y, axis=-1))
    else:
        x = np.arange(data.shape[0])
        y = np.squeeze(data.reshape((data.shape[0], -1)))
        avgs = np.squeeze(data.mean(axis=-1))
        errs = np.squeeze(data.std(axis=-1))
        avg = data.mean()
        err = data.std()

    yp_ = avg + err
    ym_ = avg - err
    yps = avgs + errs
    yms = avgs - errs

    if color is None:
        randint = np.random.randint(0, 10)
        color = f'C{randint}'

    label = kwargs.get('label', '')
    avg_label = kwargs.get('avg_label', False)

    avg_label_ = f'{avg:.3g} +/- {err:.3g}' if avg_label else ''
    _ = ax.axhline(y=avg, color=color, label=avg_label_, rasterized=True)

    # represent errors using semi-transparent `fill_between`
    _ = ax.plot(x, np.squeeze(avgs), color=color, label=label, rasterized=True)
    _ = ax.fill_between(x, y1=np.squeeze(yps),
                        y2=np.squeeze(yms), color=color, alpha=0.3,
                        rasterized=True)

    # plot horizontal lines to show avg errors
    _ = ax.axhline(y=yp_, color=color, ls=':', alpha=0.5, rasterized=True)
    _ = ax.axhline(y=ym_, color=color, ls=':', alpha=0.5, rasterized=True)

    if kwargs.get('zeroline', False):  # draw horizontal line at y = 0
        _ = ax.axhline(y=0, color='gray', ls='--', zorder=-1, rasterized=True)

    if label != '' and avg_label:
        _ = ax.legend(loc='best')  #, fontsize='small')

    strip_yticks = kwargs.get('strip_yticks', False)
    if strip_yticks:
        #  yticks = list(ax.get_yticks())
        #  yticks.append(avg)
        #  _ = ax.set_yticks(sorted(yticks))

        _ = ax.label_outer()
        _ = ax.set_yticks([avg])
        _ = ax.set_yticklabels([f'{avg:3.2g}'])

    return ax


def kde_hist(data, stats=True, **kwargs):
    histtype = kwargs.get('histtype', 'stepfilled')
    key = kwargs.get('key', '')
    color = kwargs.get('color', 'k')
    ax = kwargs.get('ax', None)
    #  kdehist = kwargs.get('kdehist', True)
    use_avg = kwargs.get('use_avg', False)

    if ax is None:
        ax = plt.gca()

    if not isinstance(data, dict) and stats:
        data = calc_stats(data, **kwargs)

    if isinstance(data, dict):
        if use_avg:
            y = data.get('avgs', None)
        else:
            y = data.get('data', None)
        avg = data.get('avg', np.mean(y))
        err = data.get('err', np.std(y))
    else:
        y = data
        avg = data.mean()
        err = data.std()

    try:
        _ = sns.kdeplot(y.flatten(), ax=ax, color=color,
                        label=f'{avg:.3g} +/- {err:.3g}')
    except np.linalg.LinAlgError:
        pass

    hist_kws = dict(color=color,
                    alpha=0.3,
                    #  bins=50,
                    density=True,
                    label=str(key),
                    histtype=histtype)
    _ = ax.hist(y.flatten(), **hist_kws)
    if kwargs.get('zeroline', False):
        _ = ax.axvline(x=0, color='gray', ls='--', zorder=-1)

    _ = ax.axvline(x=avg, color=color, ls='--')
    _ = ax.legend(loc='best')  # , fontsize='small')
    _ = ax.set_ylabel('')
    _ = ax.set_yticklabels([])
    _ = ax.set_yticks([])
    _ = ax.label_outer()

    return ax


def grid_plot(log_dir, obs_dict=None, **kwargs):
    """Plot grid of histograms."""
    run_dirs = kwargs.get('run_dirs', None)
    filter_str = kwargs.get('filter_str', None)
    therm_frac = kwargs.get('therm_frac', 0.1)
    obs_name = kwargs.get('obs_name', None)
    strip_yticks = kwargs.get('strip_yticks', False)
    zeroline = True if obs_name == 'plaqs' else False
    kdehist = kwargs.get('kdehist', True)
    axis = kwargs.get('axis', None)
    stats = kwargs.get('stats', True)
    use_avg = kwargs.get('use_avg', False)
    plot_type = kwargs.get('plot_type', 'hist')
    out_dir = kwargs.get('out_dir', None)

    if run_dirs is None:
        run_dirs = get_run_dirs(log_dir, filter_str)

    if obs_dict is None:
        obs_dict = get_obs_dict(log_dir, obs_name, run_dirs=run_dirs)

    #  files = os.listdir(out_dir)
    #  existing = [f'{obs_name}' in f for f in files]
    #  if any(existing):
    #      io.log(f'Plot already exists. Skipping.')
    #if os.path.isfile(os.path.join(out_dir, ''))
    #if os.path.isdir(os.path.join(out_dir, ))

    #out_file = os.path.join(out_dir, f'{fname}_{plot_type}.png')
    #if os.path.isfile(out_file):
    #    io.log(f'Plot already exists. Skipping.')

    num_plots = int(len(obs_dict.keys()))
    nrows = int(np.sqrt(num_plots))
    ncols = nrows

    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols,
                             sharex='col',
                             figsize=(1.95 * 12.8, 1.85 * 12.8))
                             #  gridspec_kw={'wspace': 0., 'hspace': 0.})
    axes = axes.flatten()

    beta_arr = []
    eps_arr = []

    for idx, (k, v) in enumerate(obs_dict.items()):
        if stats:
            data = calc_stats(v, therm_frac=therm_frac, axis=axis)
        else:
            data = v

        run_params = load_pkl(os.path.join(run_dirs[idx],
                                           'run_params.pkl'))
        beta_arr.append(run_params['beta'])
        eps_arr.append(run_params['eps'])

        if plot_type == 'hist':
            axes[idx] = kde_hist(data,
                                 key=k,
                                 stats=stats,
                                 ax=axes[idx],
                                 color=colors[idx],
                                 zeroline=zeroline,
                                 kdehist=kdehist,
                                 use_avg=use_avg)

        elif plot_type in ['trace', 'trace_plot']:
            axes[idx] = trace_plot(data,
                                   axes[idx],
                                   colors[idx],
                                   stats=stats,
                                   label=k,
                                   zeroline=zeroline,
                                   avg_label=True,
                                   strip_yticks=strip_yticks)

        if idx == int(num_plots // 2):
            if obs_name == 'plaqs':
                ylabel = (r"$\langle\phi_{\mathrm{P}}\rangle "
                          r"- \phi_{\mathrm{P}}^{*}$")
            else:
                ylabel = obs_name

            _ = axes[idx].set_ylabel(ylabel, fontsize=20)

    title_kwargs = {}
    if np.allclose(eps_arr, eps_arr[0]):
        title_kwargs['eps'] = eps_arr[0]

    if np.allclose(beta_arr, beta_arr[0]):
        title_kwargs['beta'] = beta_arr[0]

    fig.subplots_adjust(hspace=0.)
    if plot_type in ['trace', 'trace_plot']:
        fig.subplots_adjust(wspace=0.2)

    params_file = os.path.join(log_dir, 'parameters.pkl')
    params = load_pkl(params_file)
    clip_value = params.get('clip_value', 0)

    title_str = get_title_str(params, **title_kwargs)
    _ = plt.suptitle(title_str, fontsize=22, y=1.02)

    if out_dir is None:
        out_dir = os.path.join(log_dir, 'figures')
    else:
        log_dir_str = log_dir.split('/')[-1]
        ld_arr = log_dir_str.split('_')
        head = ld_arr[:-1]
        tail = ld_arr[-1]
        if clip_value > 0:
            head.append(f'_clip{int(clip_value)}')

        ld_str = '_'.join((i for i in head + [tail]))
        out_dir = os.path.join(out_dir, ld_str)
        io.save_dict(params, out_dir, name='parameters')

    io.check_else_make_dir(out_dir)

    if obs_name is not None:
        if obs_name == 'plaqs':
            fname = f'plaqs_diffs'
        else:
            fname = f'{obs_name}'
    else:
        fname = plot_type

    #  out_dir = os.path.join(out_dir, fname)

    if filter_str is not None:
        fname += f'_{filter_str}'

    if clip_value > 0:
        fname += f'_clip{int(clip_value)}'
        #  out_dir = os.path.join(out_dir, filter_str)

    out_file = os.path.join(out_dir,
                            f'{fname}_{plot_type}.png')
    plt.tight_layout()
    io.log(f'saving figure to: {out_file}...')
    plt.savefig(out_file, dpi=200, bbox_inches='tight')

    return fig, axes


def plot_obs(log_dir, obs_dict=None, **kwargs):
    """Plot `plaqs_diffs` w/ hists for all run dirs in `log_dir`."""
    run_dirs = kwargs.get('run_dirs', None)
    filter_str = kwargs.get('filter_str', '')
    therm_frac = kwargs.get('therm_frac', 0.1)
    obs_name = kwargs.get('obs_name', 'plaqs')
    strip_yticks = kwargs.get('strip_yticks', False)
    zeroline = True if obs_name == 'plaqs' else False
    kdehist = kwargs.get('kdehist', True)
    axis = kwargs.get('axis', None)
    stats = kwargs.get('stats', True)
    use_avg = kwargs.get('use_avg', False)
    ncols = kwargs.get('ncols', 1)

    if run_dirs is None:
        run_dirs = get_run_dirs(log_dir, filter_str)

    if obs_dict is None:
        obs_dict = get_obs_dict(log_dir, obs_name, run_dirs=run_dirs)

    nrows = int(len(obs_dict.keys()))
    ncols = 2
    #  nrows = int(len(obs_dict.keys()) / ncols)
    #  ncols = int(2 * ncols)

    if HAS_SEABORN:
        sns.set_style('ticks', {'xtick.major.size': 8, 'ytick.major.size': 8})
        sns.set_palette('bright', len(obs_dict.keys()))
        colors = sns.color_palette()
    else:
        colors = COLORS

    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols,
                             sharex='col',
                             figsize=(12.8, 9.6),
                             gridspec_kw={'wspace': 0.,
                                          'hspace': 0.})
    #  axes = axes.reshape(axes.shape[0], -1, 2)
    #  axes = axes.reshape(-1, 2)
    #  keys = np.array(list(obs_dict.keys()))
    #  vals = np.array(list(obs_dict.values()))
    #  axes = axes.flatten()

    beta_arr = []
    eps_arr = []

    #  for idx, ax in enumerate(axes)
    #  for idx, (k, v) in enumerate(zip(keys, vals)):
    for idx, (k, v) in enumerate(obs_dict.items()):
        if stats:
            data = calc_stats(v, therm_frac=therm_frac, axis=axis)
        else:
            data = v

        run_params = load_pkl(os.path.join(run_dirs[idx],
                                           'run_params.pkl'))
        assert tuple(run_params['net_weight']) == k
        beta_arr.append(run_params['beta'])
        eps_arr.append(run_params['eps'])

        axes[idx, 0] = trace_plot(data,
                                  axes[idx, 0],
                                  colors[idx],
                                  stats=stats,
                                  zeroline=zeroline,
                                  strip_yticks=strip_yticks)

        axes[idx, 1] = kde_hist(data,
                                key=k,
                                stats=stats,
                                ax=axes[idx, 1],
                                color=colors[idx],
                                zeroline=zeroline,
                                kdehist=kdehist,
                                use_avg=use_avg)

        if idx == int(nrows // 2):
            if obs_name == 'plaqs':
                ylabel = (r"$\langle\phi_{\mathrm{P}}\rangle "
                          r"- \phi_{\mathrm{P}}^{*}$")
            else:
                ylabel = obs_name

            _ = axes[idx, 0].set_ylabel(ylabel, fontsize=18)

    title_kwargs = {}
    if np.allclose(eps_arr, eps_arr[0]):
        title_kwargs['eps'] = eps_arr[0]

    if np.allclose(beta_arr, beta_arr[0]):
        title_kwargs['beta'] = beta_arr[0]

    params_file = os.path.join(log_dir, 'parameters.pkl')
    params = load_pkl(params_file)
    title_str = get_title_str(params, **title_kwargs)
    _ = plt.suptitle(title_str, fontsize=20, y=1.04)

    plt.tight_layout()
    #  fig.subplots_adjust(hspace=0.)

    out_dir = os.path.join(log_dir, 'figures')
    if obs_name == 'plaqs':
        fname = f'plaqs_diffs'
    else:
        fname = obs_name

    out_dir = os.path.join(out_dir, fname)

    if filter_str != '':
        fname += f'_{filter_str}'
        out_dir = os.path.join(out_dir, filter_str)

    io.check_else_make_dir(out_dir)
    out_file = os.path.join(out_dir, f'{fname}_{therm_frac}.png')
    print(f'saving figure to: {out_file}...')
    plt.savefig(out_file, dpi=200, bbox_inches='tight')

    return fig, axes
