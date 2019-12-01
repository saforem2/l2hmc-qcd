"""
plotters.py

Implements GaugeModelPlotter class, responsible for loading and plotting
gauge model observables.

Author: Sam Foreman (github: @saforem2)
Date: 04/10/2019
"""
import os

#  from config import BootstrapData, HAS_MATPLOTLIB, MARKERS
import config as cfg
from collections import Counter, namedtuple, OrderedDict

import numpy as np

from scipy.stats import sem

import utils.file_io as io
from seed_dict import seeds

from .plot_utils import (MPL_PARAMS, plot_multiple_lines,
                         _get_title, reset_plots)
from lattice.lattice import u1_plaq_exact

if cfg.HAS_MATPLOTLIB:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    MARKERS = cfg.MARKERS

    mpl.rcParams.update(MPL_PARAMS)

try:
    import seaborn as sns
    sns.set_palette('bright')
    sns.set_style('ticks', {'xtick.major.size': 8,
                            'ytick.major.size': 8})
    COLORS = sns.color_palette()
    HAS_SEABORN = True
except ImportError:
    COLORS = cfg.COLORS
    HAS_SEABORN = False


__all__ = ['GaugeModelPlotter', 'arr_from_dict', 'get_out_file']

FigAx = namedtuple('FigAx', ['fig', 'ax'])
DataErr = namedtuple('DataErr', ['data', 'err'])
BootstrapData = cfg.BootstrapData

np.random.seed(seeds['global_np'])



def arr_from_dict(d, key):
    if isinstance(d[key], dict):
        return np.array(list(d[key].values()))
    return np.array(d[key])


def get_out_file(out_dir, out_str):
    return os.path.join(out_dir, out_str + '.png')


def calc_plaq_sums(x):
    plaq_sums = (x[:, :, :, 0]
                 - x[:, :, :, 1]
                 - np.roll(x[:, :, :, 0], shift=-1, axis=2)
                 + np.roll(x[:, :, :, 1], shift=-1, axis=1))

    return plaq_sums


def calc_actions(x):
    plaq_sums = calc_plaq_sums(x)
    actions = np.sum(1. - np.cos(plaq_sums), axis=(1, 2))

    return actions


def calc_potential_energy(x, beta):
    return beta * calc_actions(x)


def calc_kinetic_energy(v):
    ke = 0.5 * sum(v ** 2, axis=1)

    return ke


def calc_hamiltonian(x, v, beta):
    return calc_potential_energy(x, beta) + calc_kinetic_energy(v)


class GaugeModelPlotter:
    def __init__(self, params, figs_dir=None, experiment=None):
        self.figs_dir = figs_dir
        self.params = params

    def plot_observables(self, data, **kwargs):
        """Plot observables."""
        xy_data, kwargs = self._plot_setup(data, **kwargs)

        self._plot_plaqs(xy_data['plaqs'], **kwargs)
        self._plot_actions(xy_data['actions'], **kwargs)
        self._plot_accept_probs(xy_data['accept_prob'], **kwargs)
        self._plot_charges(xy_data['charges'], **kwargs)
        self._plot_autocorrs(xy_data['autocorrs'], **kwargs)
        # xy_data['charges'][1] since we're only concerned with 'y' data
        self._plot_charge_probs(xy_data['charges'][1], **kwargs)
        self._plot_charges_hist(xy_data['charges'][1], **kwargs)
        #  self._plot_charge_diffs(xy_data['charge_diffs'], **kwargs)
        mean_diff = self._plot_plaqs_diffs(xy_data['plaqs_diffs'], **kwargs)

        return mean_diff

    def calc_stats(self, data, therm_frac=10):
        """Calculate observables statistics.

        Args:
            data (dict): Run data.
            therm_frac (int): Percent of total steps to ignore to account for
            thermalization.

        Returns:
            stats: Dictionary containing statistics for actions, plaquettes,
            top. charges, and charge probabilities. For each of the
            observables (actions, plaquettes, charges), the dictionary values
            consist of a tuple of the form: (data, error), and
            charge_probabilities is a dictionary of the form:
                {charge_val: charge_val_probability}
        """
        actions = np.array(data['actions'])
        plaqs = np.array(data['plaqs'])
        charges = np.array(data['charges'])

        charge_probs = {}
        counts = Counter(list(charges.flatten()))
        total_counts = np.sum(list(counts.values()))
        for key, val in counts.items():
            charge_probs[key] = val / total_counts

        charge_probs = OrderedDict(sorted(charge_probs.items(),
                                          key=lambda k: k[0]))

        def get_mean_err(x):
            num_steps = x.shape[0]
            therm_steps = num_steps // therm_frac
            x = x[therm_steps:, :]
            avg = np.mean(x, axis=0)
            err = sem(x)
            return avg, err

        stats = {
            'actions': get_mean_err(actions),
            'plaqs': get_mean_err(plaqs),
            'charges': get_mean_err(charges),
            'suscept': get_mean_err(charges ** 2),
            'charge_probs': charge_probs
        }

        return stats

    def _parse_data(self, data, beta):
        """Helper method for extracting relevant data from `data`.'"""
        accept_prob = np.array(data['px'])
        actions = np.array(data['actions'])
        plaqs = np.array(data['plaqs'])
        charges = np.array(data['charges'], dtype=int)
        charge_autocorrs = np.array(data['charges_autocorrs'])
        plaqs_diffs = plaqs - u1_plaq_exact(beta)

        def _stats(data, axis=1):
            return np.mean(data, axis=axis), sem(data, axis=axis)

        num_steps = actions.shape[0]
        warmup_steps = int(0.1 * num_steps)
        steps_arr = np.arange(warmup_steps, num_steps)

        actions_stats = _stats(actions[warmup_steps:, :])
        plaqs_stats = _stats(plaqs[warmup_steps:, :])
        accept_prob_stats = _stats(accept_prob[warmup_steps:, :])
        full_steps_arr = np.arange(num_steps)
        autocorrs_stats = _stats(charge_autocorrs.T)

        _plaq_diffs = plaqs_diffs[warmup_steps:]  # [::skip_steps]
        _plaq_diffs_stats = _stats(_plaq_diffs)

        xy_data = {
            'actions': (steps_arr, *actions_stats),
            'plaqs': (steps_arr, *plaqs_stats),
            'accept_prob': (steps_arr, *accept_prob_stats),
            'charges': (full_steps_arr, charges.T),
            'autocorrs': (full_steps_arr, *autocorrs_stats),
            'plaqs_diffs': (steps_arr, *_plaq_diffs_stats)
        }

        return xy_data

    def _plot_setup(self, data, **kwargs):
        """Prepare for plotting observables."""
        beta = kwargs.get('beta', 5.)
        run_str = kwargs.get('run_str', None)
        net_weights = kwargs.get('net_weights', [1., 1., 1.])
        dir_append = kwargs.get('dir_append', None)
        eps = kwargs.get('eps', None)

        if dir_append:
            run_str += dir_append

        self.out_dir = os.path.join(self.figs_dir, run_str)
        io.check_else_make_dir(self.out_dir)

        lf_steps = self.params['num_steps']
        bs = self.params['batch_size']
        nw = net_weights
        title_str = _get_title(lf_steps, eps, bs, beta, nw)

        kwargs.update({
            'markers': False,
            'lines': True,
            'alpha': 0.6,
            'title': title_str,
            'legend': False,
            'ret': False,
            'out_file': [],
        })

        xy_data = self._parse_data(data, beta)

        return xy_data, kwargs

    def _plot(self, xy_data, **kwargs):
        """Basic plotting wrapper."""
        x, y, yerr = xy_data

        labels = kwargs.get('labels', None)
        if labels is not None:
            xlabel = labels.get('x_label', '')
            ylabel = labels.get('y_label', '')
        else:
            xlabel = ''
            ylabel = ''

        _leg = kwargs.get('legend', False)

        if kwargs.get('two_rows', False):
            fig, (ax0, ax1) = plt.subplots(
                nrows=2, ncols=1, gridspec_kw={'height_ratios': [2.5, 1],
                                               'hspace': 0.175}
            )
            n = len(x)
            mid = n // 2
            x0 = int(mid - 0.025 * n)
            x1 = int(mid + 0.025 * n)
        else:
            fig, ax0 = plt.subplots()
            ax1 = None

        plt_kwargs = {
            'color': 'k',
            #  'lw': 1.,
            #  'ls': '-',
            'alpha': 0.8,
            'marker': ',',
        }

        ax0.plot(x, y, **plt_kwargs)
        ax0.errorbar(x, y, yerr=yerr, alpha=0.7, color='k', ecolor='gray')

        if ax1 is not None:
            ax1.plot(x[x0:x1], y[x0:x1], **plt_kwargs)
            ax1.errorbar(x[x0:x1], y[x0:x1], yerr=yerr[x0:x1],
                         alpha=0.7, color='k', ecolor='gray')

        ax1.set_xlabel(xlabel, fontsize='large')
        ax0.set_ylabel(ylabel, fontsize='large')
        ax1.set_ylabel('', fontsize='large')
        if _leg:
            ax0.legend(loc='best')

        title = kwargs.get('title', None)
        if title is not None:
            _ = ax0.set_title(title, fontsize='x-large')

        plt.tight_layout()
        if kwargs.get('save', True):
            fname = kwargs.get('fname', f'plot_{np.random.randint(10)}')
            out_file = get_out_file(self.out_dir, fname)
            io.log(f'Saving figure to: {out_file}.')
            plt.savefig(out_file, bbox_inches='tight')

        return fig, (ax0, ax1)

    def _hist(self, x, ax=None, labels=None, leg=True, **kwargs):
        if labels is not None:
            ylabel = labels.get('y_label', '')
        else:
            ylabel = ''

        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if ax is None:
            fig, ax = plt.subplots()

        x = x.flatten()
        mean = x.mean()
        err = x.std()
        label = f'{mean:<6.3g} +/- {err:^6.2g}'

        if HAS_SEABORN:
            _ = sns.kdeplot(x, ax=ax, color='k', label=label)
        _ = ax.hist(x, density=True, bins=50, alpha=0.3, histtype='stepfilled')
        _ = ax.axvline(x=mean, label=f'avg: {mean:.3g} +/- {err:.3g}')
        #  _ = ax.plot(mean, 0., marker='|', linewidth=1,
        #              markersize=10, markeredgewidth=1, label=label)

        _ = ax.set_ylabel(ylabel)
        if leg:
            ax.legend(loc='best')

        title = kwargs.get('title', None)
        if title is not None:
            _ = ax.set_title(title, fontsize='x-large')

        plt.tight_layout()
        if kwargs.get('save', True):
            fname = kwargs.get('fname', f'plot_{np.random.randint(10)}')
            fname += '_hist'
            out_file = get_out_file(self.out_dir, fname)
            io.log(f'Saving figure to: {out_file}.')
            plt.savefig(out_file, bbox_inches='tight')

        return ax, mean, err

    def _plot_actions(self, xy_data, **kwargs):
        """Plot actions."""
        labels = {
            'x_label': 'Step',
            'y_label': 'Action',
            'plt_label': 'Action'
        }

        fig_kwargs = {
            'labels': labels,
            'two_rows': True,
            'fname': 'actions',
            'save': True,
        }
        _ = self._hist(xy_data[1], leg=True, **fig_kwargs, **kwargs)
        self._plot(xy_data, **fig_kwargs, **kwargs)

    def _plot_plaqs(self, xy_data, beta, **kwargs):
        """PLot average plaquette."""
        labels = {
            'x_label': 'Step',
            'y_label': r"""$\langle \phi_{P} \rangle$""",
            'plt_label': r"""$\langle \phi_{P} \rangle$"""
        }
        fig_kwargs = {
            'labels': labels,
            'fname': 'plaqs',
            'two_rows': True,
            'save': False,
        }
        fig, (ax0, ax1) = self._plot(xy_data, **fig_kwargs, **kwargs)

        ax0.axhline(y=u1_plaq_exact(beta),
                    color='#CC0033', ls='-', lw=1.5, label='exact')
        ax1.axhline(y=u1_plaq_exact(beta),
                    color='#CC0033', ls='-', lw=1.5, label='exact')

        plt.tight_layout()

        out_file = get_out_file(self.out_dir, 'plaqs_vs_step')
        io.log(f'Saving figure to: {out_file}')
        plt.savefig(out_file, bbox_inches='tight')

        fig_kwargs['save'] = True
        fig_kwargs['leg'] = True
        _ = self._hist(xy_data[1], **fig_kwargs, **kwargs)

    def _plot_accept_probs(self, xy_data, **kwargs):
        """Plot actions."""
        labels = {
            'x_label': 'Step',
            'y_label': r"""$A(\xi^{\prime}|\xi)$""",
            'plt_label': 'accept_prob'
        }

        fig_kwargs = {
            'fname': 'accept_probs_vs_step',
            'labels': labels,
            'two_rows': True,
        }
        self._plot(xy_data, **fig_kwargs, **kwargs)

        fig_kwargs['save'] = True
        fig_kwargs['leg'] = True
        _ = self._hist(xy_data[1], **fig_kwargs, **kwargs)

    def _plot_plaqs_diffs(self, xy_data, **kwargs):
        kwargs['out_file'] = None
        kwargs['ret'] = True
        labels = {
            'x_label': 'Step',
            'y_label': r"$\delta_{\phi_{P}}$",
            'plt_label': r"$\delta_{\phi_{P}}$"
        }
        x, y, yerr = xy_data
        y_mean = np.mean(y)
        fig, ax = plt.subplots()
        _ = ax.plot(x, y, label='', marker=',', color='k', alpha=0.8)
        _ = ax.errorbar(x, y, yerr=yerr, label='', marker=None, ls='',
                        alpha=0.7, color='gray', ecolor='gray')
        _ = ax.axhline(y=0, color='#CC0033', ls='-', lw=2.)
        _ = ax.axhline(y=y_mean, label=f'avg {y_mean:.5f}',
                       color='C2', ls='-', lw=2.)

        _ = ax.set_xlabel(labels['x_label'], fontsize='large')
        _ = ax.set_ylabel(labels['y_label'], fontsize='large')
        title = kwargs.get('title', None)
        if title is not None:
            _ = ax.set_title(title, fontsize='x-large')

        ax.legend(loc='best')

        _ = plt.tight_layout()
        out_file = get_out_file(self.out_dir, 'plaqs_diffs_vs_step')
        io.log(f'Saving figure to: {out_file}.')
        plt.savefig(out_file, bbox_inches='tight')

        fig_kwargs = {
            'fname': 'plaqs',
            'labels': labels,
            'leg': True,
            'save': True,
        }
        self._hist(xy_data[1], **fig_kwargs, **kwargs)

        return y_mean

    def _plot_charges(self, xy_data, **kwargs):
        """Plot topological charges."""
        kwargs['out_file'] = get_out_file(self.out_dir, 'charges_vs_step')
        kwargs['markers'] = True
        kwargs['lines'] = False
        kwargs['alpha'] = 1.
        kwargs['ret'] = False
        xy_labels = ('Step', r'$Q$')
        plot_multiple_lines(xy_data, xy_labels, **kwargs)

        charges = np.array(xy_data[1], dtype=int)
        batch_size = charges.shape[0]
        num_steps = charges.shape[1]

        out_dir = os.path.join(self.out_dir, 'top_charge_plots')
        io.check_else_make_dir(out_dir)
        # if we have more than 10 chains in charges, only plot first 10
        for idx in range(min(batch_size, 8)):
            _, ax = plt.subplots()
            _ = ax.plot(charges[idx, :],
                        marker=MARKERS[idx],
                        color=COLORS[idx],
                        ls='',
                        alpha=0.5,
                        label=f'sample {idx}')
            _ = ax.legend(loc='best')
            _ = ax.set_xlabel(xy_labels[0], fontsize='large')
            _ = ax.set_ylabel(xy_labels[1], fontsize='large')
            _ = ax.set_title(kwargs['title'], fontsize='x-large')
            _ = plt.tight_layout()
            out_file = get_out_file(out_dir, f'top_charge_vs_step_{idx}')
            io.check_else_make_dir(os.path.dirname(out_file))
            io.log(f'Saving figure to: {out_file}')
            plt.savefig(out_file, bbox_inches='tight')
        plt.close('all')

        fig_kwargs = {
            'fname': 'charges',
            'labels': {'y_label': r"""$Q$"""},
            'leg': True,
            'save': True,
        }
        self._hist(xy_data[1], **fig_kwargs, **kwargs)

    def _plot_charge_diffs(self, xy_data, **kwargs):
        """Plot tunneling events (change in top. charge)."""
        out_file = get_out_file(self.out_dir, 'top_charge_diffs')
        steps_arr, charge_diffs = xy_data

        # ignore first two data points when plotting since the top. charge
        # should change dramatically for the very first vew steps when starting
        # from a random configuration
        _, ax = plt.subplots()
        _ = ax.plot(xy_data[0][2:], xy_data[1][2:],
                    marker=',', ls='', fillstyle='none', color='C0')
        _ = ax.set_xlabel('Steps', fontsize='large')
        _ = ax.set_ylabel(r'$\delta_{Q}$', fontsize='large')
        _ = ax.set_title(kwargs['title'], fontsize='x-large')
        _ = plt.tight_layout()
        io.log(f"Saving figure to: {out_file}")
        plt.savefig(out_file, bbox_inches='tight')

    def _plot_charge_probs(self, charges, **kwargs):
        """PLot top. charge probabilities."""
        num_steps, batch_size = charges.shape
        charges = np.array(charges, dtype=int)
        out_dir = os.path.join(self.out_dir, 'top_charge_probs')
        io.check_else_make_dir(out_dir)
        if 'title' in list(kwargs.keys()):
            title = kwargs.pop('title')
        # if we have more than 10 chains in charges, only plot first 10
        for idx in range(min(batch_size, 5)):
            counts = Counter(charges[:, idx])
            total_counts = np.sum(list(counts.values()))
            _, ax = plt.subplots()
            ax.plot(list(counts.keys()),
                    np.array(list(counts.values()) / total_counts),
                    marker=MARKERS[idx],
                    color=COLORS[idx],
                    ls='',
                    label=f'sample {idx}')
            _ = ax.legend(loc='best')
            _ = ax.set_xlabel(r"$Q$", fontsize='large')
            _ = ax.set_ylabel('Probability', fontsize='large')
            _ = ax.set_title(title, fontsize='x-large')
            _ = plt.tight_layout()
            out_file = get_out_file(out_dir, f'top_charge_vs_step_{idx}')
            io.check_else_make_dir(os.path.dirname(out_file))
            io.log(f"Saving plot to: {out_file}.")
            plt.savefig(out_file, bbox_inches='tight')
        plt.close('all')

        all_counts = Counter(list(charges.flatten()))
        total_counts = np.sum(list(counts.values()))
        _, ax = plt.subplots()
        ax.plot(list(all_counts.keys()),
                np.array(list(all_counts.values()) / (total_counts *
                                                      batch_size)),
                marker='o',
                color='C0',
                ls='',
                alpha=0.6,
                label=f'total across {batch_size} samples')
        _ = ax.legend(loc='best')
        _ = ax.set_xlabel(r"$Q$", fontsize='large')
        _ = ax.set_ylabel('Probability', fontsize='large')
        _ = ax.set_title(title, fontsize='x-large')
        _ = plt.tight_layout()
        out_file = get_out_file(self.out_dir, f'TOP_CHARGE_PROBS_ALL')
        io.check_else_make_dir(os.path.dirname(out_file))
        io.log(f"Saving plot to: {out_file}.")
        plt.savefig(out_file, bbox_inches='tight')
        plt.close('all')

    def _plot_charges_hist(self, charges, **kwargs):
        charges = np.array(charges, dtype=int)
        charges_flat = charges.flatten()
        bins = np.unique(charges_flat)

        _, ax = plt.subplots()
        _ = ax.hist(charges_flat, bins=bins)
        _ = ax.set_ylabel(r"""Topological charge, $Q$""")
        out_file = get_out_file(self.out_dir, 'top_charge_histogram')
        io.check_else_make_dir(os.path.dirname(out_file))
        io.log(f'Saving plot to: {out_file}.')
        plt.savefig(out_file, bbox_inches='tight')

    def _plot_autocorrs(self, xy_data, **kwargs):
        """Plot topological charge autocorrelations."""
        labels = {
            'x_label': 'Step',
            'y_label': 'Autocorrelation of ' + r'$Q$',
            'plt_label': 'Autocorrelation of ' + r'$Q$',
        }
        kwargs.update({
            'labels': labels,
            'fname': 'charge_autocorrs',
            'two_rows': True,
        })
        self._plot(xy_data, **kwargs)
