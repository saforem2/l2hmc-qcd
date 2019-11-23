"""
plotters.py

Implements GaugeModelPlotter class, responsible for loading and plotting
gauge model observables.

Author: Sam Foreman (github: @saforem2)
Date: 04/10/2019
"""
import os

from config import BootstrapData, COLORS, HAS_MATPLOTLIB, MARKERS
from collections import Counter, namedtuple, OrderedDict

import numpy as np

from scipy.stats import sem

import utils.file_io as io
from seed_dict import seeds

from .plot_utils import MPL_PARAMS, plot_multiple_lines
from lattice.lattice import u1_plaq_exact

if HAS_MATPLOTLIB:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(MPL_PARAMS)


__all__ = ['EnergyPlotter', 'GaugeModelPlotter',
           'arr_from_dict', 'get_out_file']

FigAx = namedtuple('FigAx', ['fig', 'ax'])
DataErr = namedtuple('DataErr', ['data', 'err'])

np.random.seed(seeds['global_np'])


def reset_plots():
    plt.close('all')
    plt.clf()


def arr_from_dict(d, key):
    if isinstance(d[key], dict):
        return np.array(list(d[key].values()))
    return np.array(d[key])


def get_out_file(out_dir, out_str):
    return os.path.join(out_dir, out_str + '.png')


def _get_title(lf_steps, eps, batch_size, beta, nw):
    """Parse vaious parameters to make figure title when creating plots."""
    try:
        title_str = (r"$N_{\mathrm{LF}} = $" + f"{lf_steps}, "
                     r"$\varepsilon = $" + f"{eps:.3g}, "
                     r"$N_{\mathrm{B}} = $" + f"{batch_size}, "
                     r"$\beta =$" + f"{beta:.2g}, "
                     r"$\mathrm{nw} = $" + (f"{nw[0]:.3g}, "
                                            f"{nw[1]:.3g}, "
                                            f"{nw[2]:.3g}"))
    except ValueError:
        title_str = ''
    return title_str


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


class EnergyPlotter:
    def __init__(self, params, figs_dir=None):
        self.params = params
        self.figs_dir = figs_dir

    def bootstrap(self, data, n_boot=5000):
        boot_dist = []
        for i in range(int(n_boot)):
            resampler = np.random.randint(0, data.shape[0],
                                          data.shape[0], )
            sample = data.take(resampler, axis=0)
            boot_dist.append(np.mean(sample, axis=0))

        means_bs = np.array(boot_dist)
        err = np.sqrt(float(n_boot) / float(n_boot - 1)) * np.std(means_bs)
        mean = np.mean(means_bs)

        # NOTE BootstrapData is namedtuple of form: ('mean', 'err', 'means_bs')
        bs_data = BootstrapData(mean=mean, err=err, means_bs=means_bs)

        return bs_data

    def _plot_setup(self, **kwargs):
        """Prepare for making plots."""
        beta = kwargs.get('beta', 5.)
        run_str = kwargs.get('run_str', '')
        net_weights = kwargs.get('net_weights', [1., 1., 1.])
        eps = kwargs.get('eps', None)
        out_dir = kwargs.get('out_dir', None)
        base_dir = os.path.join(self.figs_dir, run_str, 'energy_plots')
        if out_dir is not None:
            out_dir = os.path.join(base_dir, out_dir)
        else:
            out_dir = base_dir

        io.check_else_make_dir(out_dir)

        lf_steps = self.params['num_steps']
        bs = self.params['batch_size']
        nw = net_weights
        try:
            title_str = _get_title(lf_steps, eps, bs, beta, nw)
        except ValueError:
            title_str = ''

        return title_str, out_dir

    def _plot(self, labels, data_arr, title=None, out_file=None):
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        num_plots = len(labels)
        fig, axes = plt.subplots(nrows=num_plots, ncols=1)

        for idx, (label, data) in enumerate(zip(labels, data_arr)):
            num_steps = data.shape[0]
            therm_steps = int(0.1 * num_steps)
            x = np.arange(therm_steps, num_steps)
            y = data[therm_steps:].mean(axis=1)
            yerr = data[therm_steps:].std(axis=1)
            hline = y.mean()
            label += f'  avg: {hline:.4g}'
            axes[idx].plot(x, y, color=colors[idx])  # , alpha=alphas[idx])
            axes[idx].errorbar(x, y, yerr=yerr, color=colors[idx])
            dx = int(0.05 * len(x))
            axes[idx].axhline(xmin=x[0] - dx, xmax=x[-1] + dx,
                              y=hline, label=label, color=colors[idx])

            axes[idx].legend(loc='best')
            if title is not None:
                axes[idx].set_title(title)

        fig.tight_layout()
        if out_file is not None:
            io.log(f'Saving figure to: {out_file}.')
            fig.savefig(out_file, bbox_inches='tight')

        fig_ax = FigAx(fig=fig, ax=axes)
        data_err = DataErr(data=y, err=yerr)

        return fig_ax, data_err

    def _hist(self, labels, data_arr, title=None, out_file=None, **kwargs):
        n_bins = kwargs.get('n_bins', 50)
        #  n_boot = kwargs.get('n_boot', 1000)
        is_mixed = kwargs.get('is_mixed', False)
        single_chain = kwargs.get('single_chain', False)
        #  alphas = [1. - 0.25 * i for i in range(len(labels))]
        if not is_mixed:
            num_steps = data_arr[0].shape[0]
            therm_steps = int(0.1 * num_steps)
        else:
            therm_steps = 0

        fig, ax = plt.subplots()
        for idx, (label, data) in enumerate(zip(labels, data_arr)):
            if single_chain:
                data = data[therm_steps:, -1]
                mean = data.mean()
                err = data.std()
                mean_arr = data.flatten()
                #  mean, err, mean_arr = self.bootsrap(data, n_boot=n_boot)
                mean_arr = mean_arr.flatten()
            else:
                therm_steps = int(therm_steps)
                data = data[therm_steps:, :]
                mean = data.mean()
                err = data.std()
                mean_arr = data.flatten()
                #  mean, err, mean_arr = self.bootstrap(data, n_boot=n_boot)
                #  mean_arr = mean_arr.flatten()
            label = labels[idx] + f'  avg: {mean:.4g} +/- {err:.4g}'
            kwargs = dict(ec='k',
                          alpha=0.4,
                          label=label,
                          bins=n_bins,
                          density=True,
                          histtype='stepfilled')
            ax.hist(mean_arr, **kwargs)
            #  if idx > 1:
            #      ax.hist(mean_arr, bins=n_bins, density=True,
            #              alpha=alphas[idx], label=label, histtype='step')
            #  else:
            #      ax.hist(mean_arr, bins=n_bins, density=True,
            #              alpha=alphas[idx], label=label)

        ax.legend(loc='best')
        if title is not None:
            ax.set_title(title)

        fig.tight_layout()
        if out_file is not None:
            print(f'Saving figure to: {out_file}.')
            fig.savefig(out_file, bbox_inches='tight')

        fig_ax = FigAx(fig=fig, ax=ax)
        bs_data = BootstrapData(mean=mean, err=err, means_bs=mean_arr)

        return fig_ax, bs_data

    def _potential_plots(self, energy_data, title, out_dir):
        labels = [r"""$\delta U_{\mathrm{out}}$,""",
                  r"""$\delta U_{\mathrm{proposed}}$,"""]

        #  try:
        pe_init = np.array(energy_data['potential_init'])
        pe_prop = np.array(energy_data['potential_proposed'])
        pe_out = np.array(energy_data['potential_out'])

        pe_out_diff = pe_out - pe_init
        pe_proposed_diff = pe_prop - pe_init

        if not isinstance(pe_out_diff, np.ndarray):
            pe_out_diff = np.array(pe_out_diff)
        if not isinstance(pe_proposed_diff, np.ndarray):
            pe_proposed_diff = np.array(pe_proposed_diff)

        data = [pe_out_diff, pe_proposed_diff]

        plt_file = os.path.join(out_dir, 'potential_diffs.png')
        hist_file = os.path.join(out_dir, 'potential_diffs_hist.png')
        hist_file1 = os.path.join(out_dir,
                                  'potential_diffs_hist_single_chain.png')

        _, pe_diff_plot_data = self._plot(labels, data,
                                          title=title,
                                          out_file=plt_file)
        _, pe_diff_hist_data = self._hist(labels, data,
                                          title=title,
                                          out_file=hist_file)
        _, pe_diff_hist_data_sc = self._hist(labels, data,
                                             title=title,
                                             out_file=hist_file1,
                                             single_chain=True)
        reset_plots()

        labels = [r"""$U_{\mathrm{init}}$, """,
                  r"""$U_{\mathrm{proposed}}$, """,
                  r"""$U_{\mathrm{out}}$, """]

        data = [np.array(energy_data['potential_init']),
                np.array(energy_data['potential_proposed']),
                np.array(energy_data['potential_out'])]

        plt_file = os.path.join(out_dir, 'potentials.png')
        hist_file = os.path.join(out_dir, 'potential_hist.png')
        hist_file1 = os.path.join(out_dir, 'potential_hist_single_chain.png')

        _, pe_plot_data = self._plot(labels, data,
                                     title=title,
                                     out_file=plt_file)
        _, pe_hist_data = self._hist(labels, data,
                                     title=title,
                                     out_file=hist_file)
        _, pe_hist_data_sc = self._hist(labels, data,
                                        title=title,
                                        out_file=hist_file1,
                                        single_chain=True)
        reset_plots()
        outputs = {
            'diff_plot_data': pe_diff_plot_data,
            'diff_hist_data': pe_diff_hist_data,
            'diff_hist_data_sc': pe_diff_hist_data_sc,
            'plot_data': pe_plot_data,
            'hist_data': pe_hist_data,
            'hist_data_sc': pe_hist_data_sc
        }

        return outputs

    def _kinetic_plots(self, energy_data, title, out_dir):
        ke_labels = [r"""$\delta KE_{\mathrm{out}}$,""",
                     r"""$\delta KE_{\mathrm{proposed}}$,"""]

        ke_init = np.array(energy_data['kinetic_init'])
        ke_prop = np.array(energy_data['kinetic_proposed'])
        ke_out = np.array(energy_data['kinetic_out'])

        ke_out_diff = ke_out - ke_init
        ke_proposed_diff = ke_prop - ke_init

        ke_data = [ke_out_diff, ke_proposed_diff]

        ke_f = os.path.join(out_dir, 'kinetic_diffs.png')
        keh_f = os.path.join(out_dir, 'kinetic_diffs_hist.png')
        keh_f1 = os.path.join(out_dir, 'kinetic_diffs_hist_single_chain.png')

        _, diff_plot_data = self._plot(ke_labels, ke_data,
                                       title=title,
                                       out_file=ke_f)
        _, diff_hist_data = self._hist(ke_labels, ke_data,
                                       title=title,
                                       out_file=keh_f)
        _, diff_hist_data_sc = self._hist(ke_labels, ke_data, title=title,
                                          out_file=keh_f1, single_chain=True)
        reset_plots()

        labels = [r"""$KE_{\mathrm{init}}$, """,
                  r"""$KE_{\mathrm{proposed}}$, """,
                  r"""$KE_{\mathrm{out}}$, """]

        data = [np.array(energy_data['kinetic_init']),
                np.array(energy_data['kinetic_proposed']),
                np.array(energy_data['kinetic_out'])]

        plt_file = os.path.join(out_dir, 'kinetics.png')
        hist_file = os.path.join(out_dir, 'kinetic_hist.png')
        hist_file1 = os.path.join(out_dir, 'kinetic_hist_single_chain.png')

        _, plot_data = self._plot(labels, data,
                                  title=title, out_file=plt_file)
        _, hist_data = self._hist(labels, data,
                                  title=title, out_file=hist_file)
        _, hist_data_sc = self._hist(labels, data, title=title,
                                     out_file=hist_file1, single_chain=True)
        reset_plots()

        outputs = {
            'diff_plot_data': diff_plot_data,
            'diff_hist_data': diff_hist_data,
            'diff_hist_data_sc': diff_hist_data_sc,
            'plot_data': plot_data,
            'hist_data': hist_data,
            'hist_data_sc': hist_data_sc
        }

        return outputs

    def _hamiltonian_plots(self, energy_data, title, out_dir, sld=None):
        h_labels = [r"""$\delta H_{\mathrm{out}}$,""",
                    r"""$\delta H_{\mathrm{proposed}}$,"""]

        h_init = np.array(energy_data['hamiltonian_init'])
        h_prop = np.array(energy_data['hamiltonian_proposed'])
        h_out = np.array(energy_data['hamiltonian_out'])
        if sld is not None:  # sumlogdets; expects dict. 
            sld_out = sld['out']
            sld_prop = sld['proposed']
        else:
            sld_out = sld_prop = 0
        h_data = [h_out - h_init + sld_out, h_prop - h_init + sld_prop]

        h_f = os.path.join(out_dir, 'hamiltonian_diffs.png')
        hh_f = os.path.join(out_dir, 'hamiltonian_diffs_hist.png')
        hh_f1 = os.path.join(out_dir,
                             'hamiltonian_diffs_hist_single_chain.png')

        _, diff_plot_data = self._plot(h_labels, h_data,
                                       title=title, out_file=h_f)
        _, diff_hist_data = self._hist(h_labels, h_data,
                                       title=title, out_file=hh_f)
        _, diff_hist_data_sc = self._hist(h_labels, h_data, title=title,
                                          out_file=hh_f1, single_chain=True)
        reset_plots()

        labels = [r"""$H_{\mathrm{init}}$, """,
                  r"""$H_{\mathrm{proposed}}$, """,
                  r"""$H_{\mathrm{out}}$, """]

        data = [h_init, h_prop, h_out]

        plt_file = os.path.join(out_dir, 'hamiltonians.png')
        hist_file = os.path.join(out_dir, 'hamiltonian_hist.png')
        hist_file1 = os.path.join(out_dir, 'hamiltonian_hist_single_chain.png')

        _, plot_data = self._plot(labels, data,
                                  title=title, out_file=plt_file)
        _, hist_data = self._hist(labels, data,
                                  title=title, out_file=hist_file)
        _, hist_data_sc = self._hist(labels, data, title=title,
                                     out_file=hist_file1, single_chain=True)
        reset_plots()
        outputs = {
            'diff_plot_data': diff_plot_data,
            'diff_hist_data': diff_hist_data,
            'diff_hist_data_sc': diff_hist_data_sc,
            'plot_data': plot_data,
            'hist_data': hist_data,
            'hist_data_sc': hist_data_sc
        }

        return outputs

    def plot_energies(self, energy_data, sumlogdets=None, **kwargs):
        title, out_dir = self._plot_setup(**kwargs)
        args = (energy_data, title, out_dir)
        ke_data = self._kinetic_plots(*args)
        pe_data = self._potential_plots(*args)
        h_data = self._hamiltonian_plots(*args, sld=sumlogdets)

        outputs = {
            'pe_data': pe_data,
            'ke_data': ke_data,
            'h_data': h_data
        }

        return outputs


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

        # skip 5% of total number of steps between successive points when
        # plotting to help smooth out graph
        #  skip_steps = max((1, int(0.005 * num_steps)))
        # ignore first 10% of pts (warmup)

        #  _charge_diffs = charge_diffs[warmup_steps:]  # [::skip_steps]
        _plaq_diffs = plaqs_diffs[warmup_steps:]  # [::skip_steps]
        #  _steps_diffs = (
        #      skip_steps * np.arange(_plaq_diffs.shape[0])  # + skip_steps
        #  )
        #  _plaq_diffs_avg = np.mean(_plaq_diffs, axis=1)
        #  _plaq_diffs_err = sem(_plaq_diffs, axis=1)
        _plaq_diffs_stats = _stats(_plaq_diffs)

        xy_data = {
            'actions': (steps_arr, *actions_stats),
            'plaqs': (steps_arr, *plaqs_stats),
            'accept_prob': (steps_arr, *accept_prob_stats),
            'charges': (full_steps_arr, charges.T),
            #  'charges_stats': (steps_arr, charges_stats),
            #  'charge_diffs': (x_therm, _charge_diffs.T),
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

        ax1.set_xlabel(xlabel, fontsize=14)
        ax0.set_ylabel(ylabel, fontsize=14)
        ax1.set_ylabel('', fontsize=14)
        if _leg:
            ax0.legend(loc='best')

        title = kwargs.get('title', None)
        if title is not None:
            _ = ax0.set_title(title)

        plt.tight_layout()
        if kwargs.get('save', True):
            fname = kwargs.get('fname', f'plot_{np.random.randint(10)}')
            out_file = get_out_file(self.out_dir, fname)
            io.log(f'Saving figure to: {out_file}.')
            plt.savefig(out_file, bbox_inches='tight')

        return fig, (ax0, ax1)

    def _plot_actions(self, xy_data, **kwargs):
        """Plot actions."""
        labels = {
            'x_label': 'Step',
            'y_label': 'Action',
            'plt_label': 'Action'
        }

        kwargs.update({
            'fname': 'actions_vs_step',
            'labels': labels,
            'two_rows': True,
        })
        self._plot(xy_data, **kwargs)

    def _plot_plaqs(self, xy_data, beta, **kwargs):
        """PLot average plaquette."""
        labels = {
            'x_label': 'Step',
            'y_label': r"""$\langle \phi_{P} \rangle$""",
            'plt_label': r"""$\langle \phi_{P} \rangle$"""
        }
        kwargs.update({
            'labels': labels,
            'fname': 'plaqs_vs_step',
            'two_rows': True,
            'save': False,
        })
        fig, (ax0, ax1) = self._plot(xy_data, **kwargs)

        ax0.axhline(y=u1_plaq_exact(beta),
                    color='#CC0033', ls='-', lw=1.5, label='exact')
        ax1.axhline(y=u1_plaq_exact(beta),
                    color='#CC0033', ls='-', lw=1.5, label='exact')

        plt.tight_layout()

        out_file = get_out_file(self.out_dir, 'plaqs_vs_step')
        io.log(f'Saving figure to: {out_file}')
        plt.savefig(out_file, bbox_inches='tight')

    def _plot_accept_probs(self, xy_data, **kwargs):
        """Plot actions."""
        labels = {
            'x_label': 'Step',
            'y_label': r"""$A(\xi^{\prime}|\xi)$""",
            'plt_label': 'accept_prob'
        }

        kwargs.update({
            'fname': 'accept_probs_vs_step',
            'labels': labels,
            'two_rows': True,
        })
        self._plot(xy_data, **kwargs)

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

        _ = ax.set_xlabel(labels['x_label'], fontsize=14)
        _ = ax.set_ylabel(labels['y_label'], fontsize=14)
        title = kwargs.get('title', None)
        if title is not None:
            _ = ax.set_title(title)

        ax.legend(loc='best')

        _ = plt.tight_layout()
        out_file = get_out_file(self.out_dir, 'plaqs_diffs_vs_step')
        io.log(f'Saving figure to: {out_file}.')
        plt.savefig(out_file, bbox_inches='tight')

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
        num_steps, batch_size = charges.shape

        out_dir = os.path.join(self.out_dir, 'top_charge_plots')
        io.check_else_make_dir(out_dir)
        # if we have more than 10 chains in charges, only plot first 10
        for idx in range(min(batch_size, 5)):
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
            plt.savefig(out_file, bbox_inches='tight')

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
                    marker=',', ls='', fillstyle='none', color='C0')
        _ = ax.set_xlabel('Steps', fontsize=14)
        _ = ax.set_ylabel(r'$\delta_{Q}$', fontsize=14)
        _ = ax.set_title(kwargs['title'], fontsize=16)
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
            _ = ax.set_xlabel(r"$Q$")  # , fontsize=14)
            _ = ax.set_ylabel('Probability')  # , fontsize=14)
            _ = ax.set_title(title)  # , fontsize=16)
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
        _ = ax.set_xlabel(r"$Q$")  # , fontsize=14)
        _ = ax.set_ylabel('Probability')  # , fontsize=14)
        _ = ax.set_title(title)  # , fontsize=16)
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
