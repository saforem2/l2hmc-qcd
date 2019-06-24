"""
plotters.py

Implements GaugeModelPlotter class, responsible for loading and plotting
gauge model observables.

Author: Sam Foreman (github: @saforem2)
Date: 04/10/2019
"""
import os
#  import matplotlib as mpl
import numpy as np
import utils.file_io as io

from collections import Counter, OrderedDict
from scipy.stats import sem

from lattice.lattice import u1_plaq_exact
#  from utils.data_loader import DataLoader
from globals import MARKERS, COLORS
from .plot_utils import plot_multiple_lines

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True

except ImportError:
    HAS_MATPLOTLIB = False


def arr_from_dict(d, key):
    return np.array(list(d[key].values()))


def get_out_file(out_dir, out_str):
    out_file = os.path.join(out_dir, out_str + '.pdf')
    return out_file


class GaugeModelPlotter:

    def __init__(self, model, figs_dir=None):
        self.model = model
        self.figs_dir = figs_dir

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
        actions = arr_from_dict(data, 'actions')
        plaqs = arr_from_dict(data, 'plaqs')
        charges = arr_from_dict(data, 'charges')

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

    def plot_observables(self, data, beta, run_str, **weights):
        """Plot observables."""
        actions = arr_from_dict(data, 'actions')
        plaqs = arr_from_dict(data, 'plaqs')
        charges = np.array(arr_from_dict(data, 'charges'), dtype=int)
        charge_diffs = arr_from_dict(data, 'charge_diffs')
        charge_autocorrs = np.array(data['charges_autocorrs'])
        plaqs_diffs = plaqs - u1_plaq_exact(beta)

        num_steps, num_samples = actions.shape
        steps_arr = np.arange(num_steps)
        # skip 5% of total number of steps 
        # between successive points when 
        # plotting to help smooth out graph
        skip_steps = max((1, int(0.005 * num_steps)))
        # ignore first 10% of pts (warmup)
        warmup_steps = max((1, int(0.01 * num_steps)))

        #  _actions = actions[::skip_steps]
        #  _plaqs = plaqs[::skip_steps]

        #  _steps_arr = skip_steps * np.arange(_actions.shape[0])
        _charge_diffs = charge_diffs[warmup_steps:][::skip_steps]
        _plaqs_diffs = plaqs_diffs[warmup_steps:][::skip_steps]
        _steps_diffs = (
            skip_steps * np.arange(_plaqs_diffs.shape[0]) + skip_steps
        )
        #  steps_diffs = np.arange(plaqs_diffs.shape[0])

        #  out_dir = (f'{int(num_steps)}_steps_' f"beta_{beta}")
        #  self.out_dir = os.path.join(self.figs_dir, out_dir)
        #  self.out_dir = os.path.join
        self.out_dir = os.path.join(self.figs_dir, run_str)
        io.check_else_make_dir(self.out_dir)

        L = self.model.space_size
        lf_steps = self.model.num_steps
        bs = self.model.num_samples
        qw = weights['charge_weight']
        #  nw = weights['net_weight']

        title_str = (r"$L = $" + f"{L}, "
                     r"$N_{\mathrm{LF}} = $" + f"{lf_steps}, "
                     r"$\alpha_{Q} = $" + f"{qw}, "
                     r"$\beta = $" + f"{beta}, "
                     r"$N_{\mathrm{samples}} = $" + f"{bs}")

        kwargs = {
            'markers': False,
            'lines': True,
            'alpha': 0.6,
            'title': title_str,
            'legend': True,
            'ret': False,
            'out_file': [],
        }

        #  self._plot_actions((_steps_arr, _actions.T), **kwargs)
        #  self._plot_plaqs((_steps_arr, _plaqs.T), beta, **kwargs)
        self._plot_actions((steps_arr, actions.T), **kwargs)
        self._plot_plaqs((steps_arr, plaqs.T), beta, **kwargs)
        self._plot_charges((steps_arr, charges.T), **kwargs)
        self._plot_autocorrs((steps_arr, charge_autocorrs), **kwargs)
        self._plot_charge_probs(charges, **kwargs)
        self._plot_charge_diffs((_steps_diffs, _charge_diffs.T), **kwargs)
        self._plot_plaqs_diffs((_steps_diffs, _plaqs_diffs.T), **kwargs)
        #  self._plot_charge_diffs((steps_diffs, charge_diffs.T), **kwargs)
        #  self._plot_plaqs_diffs((steps_diffs, plaqs_diffs.T), **kwargs)

    def _plot_actions(self, xy_data, **kwargs):
        """Plot actions."""
        kwargs['out_file'] = get_out_file(self.out_dir, 'actions_vs_step')
        xy_labels = ('Step', 'Action')
        plot_multiple_lines(xy_data, xy_labels, **kwargs)

    def _plot_plaqs(self, xy_data, beta, **kwargs):
        """PLot average plaquette."""
        kwargs['out_file'] = None
        kwargs['ret'] = True
        xy_labels = ('Step', r"""$\langle \phi_{P} \rangle$""")

        _, ax = plot_multiple_lines(xy_data, xy_labels, **kwargs)
        _ = ax.axhline(y=u1_plaq_exact(beta),
                       color='#CC0033', ls='-', lw=1.5, label='exact')
        #  _ = ax.plot(xy_data[0], xy_data[1].mean(axis=0), lw=1.25,
        #              color='k', label='average', alpha=0.75)
        _ = plt.tight_layout()

        out_file = get_out_file(self.out_dir, 'plaqs_vs_step')
        io.log(f'Saving figure to: {out_file}')
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

    def _plot_plaqs_diffs(self, xy_data, **kwargs):
        kwargs['out_file'] = None
        kwargs['ret'] = True
        xy_labels = ('Step', r"$\delta_{\phi_{P}}$")
        _, ax = plot_multiple_lines(xy_data, xy_labels, **kwargs)
        _ = ax.axhline(y=0, color='#CC0033', ls='-', lw=1.5)
        #  _ = ax.plot(xy_data[0], xy_data[1].mean(axis=0), lw=1.25,
        #              color='k', label='average', alpha=0.75)
        _ = ax.legend(loc='best')
        _ = plt.tight_layout()

        out_file = get_out_file(self.out_dir, 'plaqs_diffs_vs_step')
        io.log(f'Saving figure to: {out_file}')
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

    def _plot_charges(self, xy_data, **kwargs):
        """Plot topological charges."""
        kwargs['out_file'] = get_out_file(self.out_dir, 'charges_vs_step')
        kwargs['markers'] = True
        kwargs['lines'] = False
        kwargs['alpha'] = 1.
        kwargs['ret'] = False
        xy_labels = ('Step', r'$Q$')
        plot_multiple_lines(xy_data, xy_labels, **kwargs)

        charges = np.array(xy_data[1].T, dtype=int)
        num_steps, num_samples = charges.shape

        out_dir = os.path.join(self.out_dir, 'top_charge_plots')
        io.check_else_make_dir(out_dir)
        # if we have more than 10 chains in charges, only plot first 10
        for idx in range(min(num_samples, 5)):
            _, ax = plt.subplots()
            _ = ax.plot(charges[:, idx],
                        marker=MARKERS[idx],
                        color=COLORS[idx],
                        ls='',
                        alpha=0.5,
                        label=f'sample {idx}')
            _ = ax.legend(loc='best')
            _ = ax.set_xlabel(xy_labels[0], fontsize=14)
            _ = ax.set_ylabel(xy_labels[1], fontsize=14)
            _ = ax.set_title(kwargs['title'], fontsize=16)
            _ = plt.tight_layout()
            out_file = get_out_file(out_dir, f'top_charge_vs_step_{idx}')
            io.check_else_make_dir(os.path.dirname(out_file))
            io.log(f'Saving figure to: {out_file}')
            plt.savefig(out_file, dpi=400, bbox_inches='tight')

            xi = 10
            xf = charges.shape[0] // 10

            _ = ax.set_xlim((xi, xf))
            out_file = out_file.rstrip('.pdf') + '_zoom.pdf'
            io.log(f'Saving figure to: {out_file}')
            plt.savefig(out_file, dpi=400, bbox_inches='tight')

        plt.close('all')

    def _plot_charge_diffs(self, xy_data, **kwargs):
        """Plot tunneling events (change in top. charge)."""
        out_file = get_out_file(self.out_dir, 'top_charge_diffs')
        steps_arr, charge_diffs = xy_data

        # ignore first two data points when plotting since the top. charge
        # should change dramatically for the very first vew steps when starting
        # from a random configuration
        _, ax = plt.subplots()
        _ = ax.plot(xy_data[0][2:], xy_data[1][2:],
                    marker='.', ls='', fillstyle='none', color='C0')
        _ = ax.set_xlabel('Steps', fontsize=14)
        _ = ax.set_ylabel(r'$\delta_{Q}$', fontsize=14)
        _ = ax.set_title(kwargs['title'], fontsize=16)
        _ = plt.tight_layout()
        io.log(f"Saving figure to: {out_file}")
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

    def _plot_charge_probs(self, charges, **kwargs):
        """PLot top. charge probabilities."""
        num_steps, num_samples = charges.shape
        charges = np.array(charges, dtype=int)
        out_dir = os.path.join(self.out_dir, 'top_charge_probs')
        io.check_else_make_dir(out_dir)
        if 'title' in list(kwargs.keys()):
            title = kwargs.pop('title')
        # if we have more than 10 chains in charges, only plot first 10
        for idx in range(min(num_samples, 5)):
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
            _ = ax.set_xlabel(r"$Q$")  # , fontsize=14)
            _ = ax.set_ylabel('Probability')  # , fontsize=14)
            _ = ax.set_title(title)  # , fontsize=16)
            _ = plt.tight_layout()
            out_file = get_out_file(out_dir, f'top_charge_vs_step_{idx}')
            io.check_else_make_dir(os.path.dirname(out_file))
            io.log(f"Saving plot to: {out_file}.")
            plt.savefig(out_file, dpi=400, bbox_inches='tight')
            #  for f in out_file:
            #      io.check_else_make_dir(os.path.dirname(f))
            #      io.log(f"Saving plot to: {f}.")
        plt.close('all')

        all_counts = Counter(list(charges.flatten()))
        total_counts = np.sum(list(counts.values()))
        _, ax = plt.subplots()
        ax.plot(list(all_counts.keys()),
                np.array(list(all_counts.values()) / (total_counts *
                                                      num_samples)),
                marker='o',
                color='C0',
                ls='',
                alpha=0.6,
                label=f'total across {num_samples} samples')
        _ = ax.legend(loc='best')
        _ = ax.set_xlabel(r"$Q$")  # , fontsize=14)
        _ = ax.set_ylabel('Probability')  # , fontsize=14)
        _ = ax.set_title(title)  # , fontsize=16)
        _ = plt.tight_layout()
        out_file = get_out_file(self.out_dir, f'TOP_CHARGE_PROBS_ALL')
        io.check_else_make_dir(os.path.dirname(out_file))
        io.log(f"Saving plot to: {out_file}.")
        plt.savefig(out_file, dpi=400, bbox_inches='tight')
        #  for f in out_file:
        plt.close('all')

    def _plot_autocorrs(self, xy_data, **kwargs):
        """Plot topological charge autocorrelations."""
        try:
            kwargs['out_file'] = get_out_file(
                self.out_dir, 'charge_autocorrs_vs_step'
            )
        except AttributeError:
            kwargs['out_file'] = None
        xy_labels = ('Step', 'Autocorrelation of ' + r'$Q$')
        return plot_multiple_lines(xy_data, xy_labels, **kwargs)
