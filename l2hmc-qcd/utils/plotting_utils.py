"""
plotting.py

Methods for plotting data.
"""
import os

import arviz as az
import numpy as np
import xarray as xr
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as it

import utils.file_io as io

from dynamics.config import NetWeights
from config import NP_FLOATS, PI, PROJECT_DIR, TF_FLOATS

TF_FLOAT = TF_FLOATS[tf.keras.backend.floatx()]
NP_FLOAT = NP_FLOATS[tf.keras.backend.floatx()]

COLORS = 100 * ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

plt.style.use('default')
sns.set_context('paper')
sns.set_style('whitegrid')
sns.set_palette('bright')
plt.ticklabel_format(scilimits=None)
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


def drop_sequential_duplicates(chain):
    if tf.is_tensor(chain):
        return tf.convert_to_tensor([i[0] for i in it.groupby(chain)])
    return np.array([i[0] for i in it.groupby(chain)])


def savefig(fig, fpath):
    io.check_else_make_dir(os.path.dirname(fpath))
    io.log(f'Saving figure to: {fpath}.')
    fig.savefig(fpath, dpi=400, bbox_inches='tight')


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


def plot_energy_distributions(data, out_dir=None, title=None):
    energies = {
        'forward': {
            'start': data['Hf_start'],
            'mid': data['Hf_mid'],
            'end': data['Hf_end'],
        },
        'backward': {
            'start': data['Hb_start'],
            'mid': data['Hb_mid'],
            'end': data['Hb_end'],
        }
    }
    energies_combined = {
        'forward': {
            'start': data['Hwf_start'],
            'mid': data['Hwf_mid'],
            'end': data['Hwf_end'],
        },
        'backward': {
            'start': data['Hwb_start'],
            'mid': data['Hwb_mid'],
            'end': data['Hwb_end'],
        }
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='col',
                             constrained_layout=True)
    #  plt.tight_layout()
    axes = axes.flatten()
    for idx, (key, val) in enumerate(energies.items()):
        for k, v, in val.items():
            x, y = v
            _ = sns.distplot(y.flatten(), label=f'{key}/{k}',
                             hist=False, ax=axes[idx],
                             kde_kws={'shade': True})

    for idx, (key, val) in enumerate(energies_combined.items()):
        for k, v in val.items():
            x, y = v
            _ = sns.distplot(y.flatten(), label=f'{key}/{k}',
                             hist=False, ax=axes[idx+2],
                             kde_kws={'shade': True})

    _ = axes[0].legend(loc='best')
    _ = axes[1].legend(loc='best')
    _ = axes[2].legend(loc='best')
    _ = axes[3].legend(loc='best')
    _ = axes[1].set_xlabel(r"$\mathcal{H}$")  # , fontsize='large')
    _ = axes[3].set_xlabel(r"$\mathcal{H} - \sum\log\|\mathcal{J}\|$")
    if title is not None:
        _ = fig.suptitle(title)  # , fontsize='x-large')
    if out_dir is not None:
        out_file = os.path.join(out_dir, 'energy_dists_traj.png')
        savefig(fig, out_file)

    return fig, axes


def energy_traceplot(key, arr, out_dir=None, title=None):
    if out_dir is not None:
        out_dir = os.path.join(out_dir, 'energy_traceplots')
        io.check_else_make_dir(out_dir)

    for idx in range(arr.shape[1]):
        arr_ = arr[:, idx, :]
        steps = np.arange(arr_.shape[0])
        chains = np.arange(arr_.shape[1])
        data_arr = xr.DataArray(arr_.T,
                                dims=['chain', 'draw'],
                                coords=[chains, steps])
        new_key = f'{key}_lf{idx}'
        if out_dir is not None:
            tplot_fname = os.path.join(out_dir,
                                       f'{new_key}_traceplot.png')

        _ = mcmc_traceplot(new_key, data_arr, title, tplot_fname)


def plot_charges(steps, charges, title=None, out_dir=None):
    charges = charges.T
    if charges.shape[0] > 4:
        charges = charges[:4, :]
    fig, ax = plt.subplots(constrained_layout=True)
    for idx, q in enumerate(charges):
        ax.plot(steps, np.around(q) + 5 * idx, marker='', ls='-')
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.xmargin: 0
    ax.yaxis.set_label_coords(-0.03, 0.5)
    ax.set_ylabel(r"$\mathcal{Q}$",  # , fontsize='x-large',
                  rotation='horizontal')
    ax.set_xlabel('MC Step')  # , fontsize='x-large')
    if title is not None:
        ax.set_title(title)  # , fontsize='x-large')
    #  plt.tight_layout()

    if out_dir is not None:
        fpath = os.path.join(out_dir, 'charge_chains.png')
        savefig(fig, fpath)

    return fig, ax


def get_title_str_from_params(params):
    """Create a formatted string with relevant params from `params`."""
    eps = params.get('eps', None)
    net_weights = params.get('net_weights', None)
    num_steps = params.get('num_steps', None)
    lattice_shape = params.get('lattice_shape', None)

    title_str = (r"$N_{\mathrm{LF}} = $" + f'{num_steps}, '
                 r"$\varepsilon = $" + f'{tf.reduce_mean(eps):.4g}, ')

    if 'beta_init' in params and 'beta_final' in params:
        beta_init = params.get('beta_init', None)
        beta_final = params.get('beta_final', None)
        #  title_str = r"$\beta_{mathrm{init}} = $" + f'{beta_init}'
        title_str += (r"$\beta: $" + f'{beta_init:.3g}'
                      + r"$\rightarrow$" f'{beta_final:.3g}, ')
    elif 'beta' in params:
        beta = params.get('beta', None)
        title_str += r"$\beta = $" + f'{beta:.3g}, '

    title_str += f'shape: {tuple(lattice_shape)}'

    if net_weights == NetWeights(0., 0., 0., 0., 0., 0.):
        title_str += ', (HMC)'

    return title_str


def mcmc_avg_lineplots(data, title=None, out_dir=None):
    """Plot trace of avg."""
    for idx, (key, val) in enumerate(data.items()):
        fig, axes = plt.subplots(ncols=2, figsize=(8, 4),
                                 constrained_layout=True)
        axes = axes.flatten()
        if len(val) == 2:
            if len(val[0].shape) > len(val[1].shape):
                arr, steps = val
            else:
                steps, arr = val
        else:
            arr = val
            steps = np.arange(arr.shape[0])

        avg = np.mean(arr, axis=1)

        xlabel = 'MC Step'
        ylabel = ' '.join(key.split('_')) + r" $\text{avg}$"

        _ = axes[0].plot(steps, avg, color=COLORS[idx])
        _ = axes[0].set_xlabel(xlabel)
        _ = axes[0].set_ylabel(ylabel)
        _ = sns.distplot(arr.flatten(), hist=False,
                         color=COLORS[idx], ax=axes[1],
                         kde_kws={'shade': True})
        _ = axes[1].set_xlabel(ylabel)
        _ = axes[1].set_ylabel('')
        if title is not None:
            _ = fig.suptitle(title)

        if out_dir is not None:
            dir_ = out_dir
            if 'Hf' in key or 'Hb' in key:
                dir_ = os.path.join(out_dir, 'energies')
            if 'Hwf' in key or 'Hwb' in key:
                dir_ = os.path.join(out_dir,
                                    'energies_combined')
            if 'sld' in key or 'ldf' in key or 'ldb' in key:
                dir_ = os.path.join(out_dir, 'logdets')

            dir_ = os.path.join(dir_, 'avg_lineplots')
            io.check_else_make_dir(dir_)
            fpath = os.path.join(dir_, f'{key}_avg.png')
            savefig(fig, fpath)

    return fig, axes


def mcmc_lineplot(data, labels, title=None,
                  fpath=None, show_avg=False, **kwargs):
    """Make a simple lineplot."""
    fig, ax = plt.subplots(constrained_layout=True)

    if show_avg:
        avg = np.mean(data[1])
        ax.axhline(y=avg, color='gray',
                   label=f'avg: {avg:.3g}',
                   ls='-', marker='')
        ax.legend(loc='best')

    ax.plot(*data, **kwargs)
    ax.set_xlabel(labels[0])  # , fontsize='large')
    ax.set_ylabel(labels[1])  # , fontsize='large')
    if title is not None:
        ax.set_title(title)  # , fontsize='x-large')

    if fpath is not None:
        savefig(fig, fpath)

    return fig, ax


def mcmc_traceplot(key, val, title=None, fpath=None):
    if '_' in key:
        key = ' '.join(key.split('_'))

    az.plot_trace({key: val})
    fig = plt.gcf()
    if title is not None:
        fig.suptitle(title)  # , fontsize='x-large', y=1.06)

    #  plt.tight_layout()
    if fpath is not None:
        savefig(fig, fpath)

    return fig


# pylint:disable=unsubscriptable-object
def plot_data(train_data, out_dir, flags, thermalize=False, params=None):
    out_dir = os.path.join(out_dir, 'plots')
    io.check_else_make_dir(out_dir)

    title = None if params is None else get_title_str_from_params(params)

    logging_steps = flags.get('logging_steps', 1)
    flags_file = os.path.join(out_dir, 'FLAGS.z')
    if os.path.isfile(flags_file):
        train_flags = io.loadz(flags_file)
        logging_steps = train_flags.get('logging_steps', 1)

    data_dict = {}
    for key, val in train_data.data.items():
        if key == 'x':
            continue

        out_dir_ = out_dir
        if 'ld' in key:
            out_dir_ = os.path.join(out_dir_, 'logdets')

        if 'H' in key:
            if 'Hw' in key:
                out_dir_ = os.path.join(out_dir_, 'energies_combined')
            else:
                out_dir_ = os.path.join(out_dir_, 'energies')

        io.check_else_make_dir(out_dir_)

        arr = np.array(val)
        steps = logging_steps * np.arange(len(arr))

        if thermalize or key == 'dt':
            arr, steps = therm_arr(arr, therm_frac=0.33)
            #  steps = steps[::logging_setps]
            #  steps *= logging_steps

        labels = ('MC Step', key)
        data = (steps, arr)

        if len(arr.shape) == 1:
            lplot_fname = os.path.join(out_dir_, f'{key}.png')
            _, _ = mcmc_lineplot(data, labels, title,
                                 lplot_fname, show_avg=True)

        elif len(arr.shape) > 1:
            data_dict[key] = data
            cond1 = (key in ['Hf', 'Hb', 'Hwf', 'Hwb', 'sldf', 'sldb'])
            cond2 = (arr.shape[1] == flags.dynamics_config.get('num_steps'))
            if cond1 and cond2:
                _ = energy_traceplot(key, arr, out_dir=out_dir_, title=title)
            else:
                out_dir_ = os.path.join(out_dir_, 'traceplots')
                chains = np.arange(arr.shape[1])
                data_arr = xr.DataArray(arr.T,
                                        dims=['chain', 'draw'],
                                        coords=[chains, steps])

                tplot_fname = os.path.join(out_dir_, f'{key}_traceplot.png')
                _ = mcmc_traceplot(key, data_arr, title, tplot_fname)

        plt.close('all')

    _ = mcmc_avg_lineplots(data_dict, title, out_dir)
    _ = plot_charges(*data_dict['charges'], out_dir=out_dir, title=title)
    _ = plot_energy_distributions(data_dict, out_dir=out_dir, title=title)

    plt.close('all')
