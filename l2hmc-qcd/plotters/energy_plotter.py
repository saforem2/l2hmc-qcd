"""
energy_plotter.py

Implements `EnergyPlotter` class for plotting trace plots and histograms of the
energy during an inferenfce run.
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


__all__ = ['EnergyPlotter']

FigAx = namedtuple('FigAx', ['fig', 'ax'])
DataErr = namedtuple('DataErr', ['data', 'err'])
BootstrapData = cfg.BootstrapData

np.random.seed(seeds['global_np'])


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
                axes[idx].set_title(title, fontsize='x-large')

        fig.tight_layout()
        if out_file is not None:
            io.log(f'Saving figure to: {out_file}.')
            fig.savefig(out_file, bbox_inches='tight')

        fig_ax = FigAx(fig=fig, ax=axes)
        data_err = DataErr(data=y, err=yerr)

        return fig_ax, data_err

    def _hist(self, labels, data_arr, title=None, out_file=None, **hist_kws):
        n_bins = hist_kws.get('n_bins', 50)
        #  n_boot = kwargs.get('n_boot', 1000)
        is_mixed = hist_kws.get('is_mixed', False)
        single_chain = hist_kws.get('single_chain', False)
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
            hist_kws = dict(label=label,
                            alpha=0.3,
                            bins=n_bins,
                            color=COLORS[idx],
                            density=True,
                            histtype='stepfilled')
            try:
                ax = sns.kdeplot(mean_arr, ax=ax, color=COLORS[idx])
                #  ax = sns.distplot(mean_arr, ax=ax, **hist_kws)
            except ValueError:
                continue
            ax.hist(mean_arr, **hist_kws)

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
        _, hist_data = self._hist(labels, data, title=title,
                                  out_file=hist_file)
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


