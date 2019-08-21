"""
plotters.py

Implements GaugeModelPlotter class, responsible for loading and plotting
gauge model observables.

Author: Sam Foreman (github: @saforem2)
Date: 04/10/2019
"""
import os
import numpy as np
import utils.file_io as io

from collections import Counter, OrderedDict
from scipy.stats import sem

from lattice.lattice import u1_plaq_exact
from config import COLORS, MARKERS, HAS_MATPLOTLIB

if HAS_MATPLOTLIB:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    params = {
        #  'backend': 'ps',
        #  'text.latex.preamble': [r'\usepackage{gensymb}'],
        'axes.labelsize': 14,   # fontsize for x and y labels (was 10)
        'axes.titlesize': 10,
        'legend.fontsize': 10,  # was 10
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        #  'text.usetex': True,
        #  'figure.figsize': [fig_width, fig_height],
        #  'font.family': 'serif',
    }

    mpl.rcParams.update(params)


def arr_from_dict(d, key):
    return np.array(list(d[key].values()))


def get_out_file(out_dir, out_str):
    return os.path.join(out_dir, out_str + '.pdf')


def get_colors(num_samples=10, cmaps=None):
    if cmaps is None:
        cmap0 = mpl.cm.get_cmap('Greys', num_samples + 1)
        cmap1 = mpl.cm.get_cmap('Reds', num_samples + 1)
        cmap2 = mpl.cm.get_cmap('Blues', num_samples + 1)
        cmaps = (cmap0, cmap1, cmap2)

    idxs = np.linspace(0.1, 0.75, num_samples + 1)
    colors_arr = []
    for cmap in cmaps:
        colors_arr.append([cmap(i) for i in idxs])

    return colors_arr


def plot_multiple_lines(data, xy_labels, **kwargs):
    """Plot multiple lines along with their average."""
    out_file = kwargs.get('out_file', None)
    markers = kwargs.get('markers', False)
    lines = kwargs.get('lines', True)
    alpha = kwargs.get('alpha', 1.)
    legend = kwargs.get('legend', False)
    title = kwargs.get('title', None)
    ret = kwargs.get('ret', False)
    num_samples = kwargs.get('num_samples', 10)
    greys, reds, blues = get_colors(num_samples)
    if isinstance(data, list):
        data = np.array(data)

    try:
        x_data, y_data = data
    except (IndexError, ValueError):
        x_data = np.arange(data.shape[0])
        y_data = data

    x_label, y_label = xy_labels

    if y_data.shape[0] > num_samples:
        y_sample = y_data[:num_samples, :]
    else:
        y_sample = y_data

    fig, ax = plt.subplots()

    marker = None
    ls = '-'
    fillstyle = 'full'
    for idx, row in enumerate(y_sample):
        if markers:
            marker = MARKERS[idx]
            fillstyle = 'none'
            ls = '-'
        if not lines:
            ls = ''
        _ = ax.plot(x_data, row, label=f'sample {idx}', fillstyle=fillstyle,
                    marker=marker, ls=ls, alpha=alpha, lw=0.5,
                    color=greys[idx])

    _ = ax.plot(
        x_data, y_data.mean(axis=0), label='average',
        alpha=1., lw=1.0, color='k'
    )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    plt.tight_layout()
    if legend:
        ax.legend(loc='best')
    if title is not None:
        ax.set_title(title)
    if out_file is not None:
        out_dir = os.path.dirname(out_file)
        io.check_else_make_dir(out_dir)
        io.log(f'Saving figure to {out_file}.')
        fig.savefig(out_file, dpi=400, bbox_inches='tight')

    if ret:
        return fig, ax

    return 1


def plot_with_inset(data, labels=None, **kwargs):
    """Make plot with zoomed inset."""
    out_file = kwargs.get('out_file', None)
    markers = kwargs.get('markers', False)
    lines = kwargs.get('lines', True)
    alpha = kwargs.get('alpha', 7.)
    legend = kwargs.get('legend', False)
    title = kwargs.get('title', None)
    lw = kwargs.get('lw', 1.)
    ret = kwargs.get('ret', False)
    #  data_lims = kwargs.get('data_lims', None)

    color = kwargs.get('color', 'C0')
    bounds = kwargs.get('bounds', [0.2, 0.2, 0.7, 0.3])
    dx = kwargs.get('dx', 100)

    plt_label = labels.get('plt_label', None)
    x_label = labels.get('x_label', None)
    y_label = labels.get('y_label', None)
    #  num_samples = kwargs.get('num_samples', 10)
    #  greys, reds, blues = get_colors(num_samples)
    if isinstance(data, list):
        data = np.array(data)

    if isinstance(data, (tuple, np.ndarray)):
        if len(data) == 1:
            x = np.arange(data.shape[0])
            y = data
        if len(data) == 2:
            x, y = data
        elif len(data) == 3:
            x, y, yerr = data

    marker = None
    ls = '-'
    fillstyle = 'full'
    if markers:
        marker = 'o'
        fillstyle = 'none'

    if not lines:
        ls = ''

    fig, ax = plt.subplots()
    if yerr is None:
        _ = ax.plot(x, y, label=plt_label,
                    marker=marker, fillstyle=fillstyle,
                    ls=ls, alpha=alpha, lw=lw, color=color)
    else:
        _ = ax.errorbar(x, y, yerr=yerr, label=plt_label,
                        marker=marker, fillstyle=fillstyle,
                        ls=ls, alpha=alpha, lw=lw, color=color)

    axins = ax.inset_axes(bounds)

    mid_idx = len(x) // 2
    idx0 = mid_idx - dx
    idx1 = mid_idx + dx
    skip = 10

    _x = x[idx0:idx1:skip]
    _y = y[idx0:idx1:skip]
    if yerr is not None:
        _yerr = yerr[idx0:idx1:skip]
        _ymax = max(_y + abs(_yerr))
        _ymax += 0.1 * _ymax
        _ymin = min(_y - abs(_yerr))
        _ymin -= 0.1 * _y
        axins.errorbar(_x, _y, yerr=_yerr, label='',
                       marker=marker, fillstyle=fillstyle,
                       ls=ls, alpha=alpha, lw=lw, color=color)
    else:
        _ymax = max(_y)
        _ymax += 0.1 * _ymax
        _ymin = min(_y)
        _ymin -= 0.1 * _ymin
        axins.plot(_x, _y, label='',
                   marker=marker, fillstyle=fillstyle,
                   ls=ls, alpha=alpha, lw=lw, color=color)

    axins.indicate_inset_zoom(axins, label='')
    axins.xaxis.get_major_locator().set_params(nbins=3)

    _ = ax.set_xlabel(x_label, fontsize=14)
    _ = ax.set_ylabel(y_label, fontsize=14)
    _ = plt.tight_layout()
    if legend:
        _ = ax.legend(loc='best')
    if title is not None:
        _ = ax.set_title(title)
    if out_file is not None:
        out_dir = os.path.dirname(out_file)
        io.check_else_make_dir(out_dir)
        io.log(f'Saving figure to {out_file}.')
        fig.savefig(out_file, dpi=400, bbox_inches='tight')

    if ret:
        return fig, ax, axins

    return None


def plot_plaq_diffs_vs_transl_weight(xy_data, lf_steps, figs_dir):
    """Plot the average plaquette difference versus translation weight."""
    if not HAS_MATPLOTLIB:
        return
    else:
        import matplotlib.pyplot as plt

    transl_weights = [i[0][1] for i in xy_data]
    diffs = [i[1] for i in xy_data]

    fig, ax = plt.subplots()
    ax.plot(transl_weights, diffs,
            label=r'$N_{\mathrm{LF}} = $' + f'{lf_steps}')

    xlabel = 'Translation weight'
    ylabel = r"$\langle\delta_{\phi_{P}}^{(\mathrm{obs})}\rangle$"

    ax.grid(True)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    ax.legend(loc='best')
    out_file = os.path.join(figs_dir, 'avg_plaq_diff_vs_transl_weight.pdf')
    plt.savefig(out_file, dpi=400, bbox_inches='tight')


class GaugeModelPlotter:
    def __init__(self, params, figs_dir=None, experiment=None):
        self.figs_dir = figs_dir
        self.params = params
        #  self.model = model
        if experiment is not None:
            self.experiment = experiment

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

    def log_figure(self):
        try:
            self.experiment.log_figure()
        except AttributeError:
            pass

    def _parse_data(self, data, beta):
        """Helper method for extracting relevant data from `data`.'"""
        actions = arr_from_dict(data, 'actions')
        plaqs = arr_from_dict(data, 'plaqs')
        charges = np.array(arr_from_dict(data, 'charges'), dtype=int)
        charge_diffs = arr_from_dict(data, 'charge_diffs')
        charge_autocorrs = np.array(data['charges_autocorrs'])
        plaqs_diffs = plaqs - u1_plaq_exact(beta)

        actions_avg = np.mean(actions, axis=1)
        actions_err = sem(actions, axis=1)

        plaqs_avg = np.mean(plaqs, axis=1)
        plaqs_err = sem(plaqs, axis=1)

        autocorrs_avg = np.mean(charge_autocorrs.T, axis=1)
        autocorrs_err = sem(charge_autocorrs.T, axis=1)

        num_steps, num_samples = actions.shape
        steps_arr = np.arange(num_steps)

        # skip 5% of total number of steps between successive points when
        # plotting to help smooth out graph
        skip_steps = max((1, int(0.005 * num_steps)))
        # ignore first 10% of pts (warmup)
        warmup_steps = max((1, int(0.01 * num_steps)))

        _charge_diffs = charge_diffs[warmup_steps:][::skip_steps]
        _plaq_diffs = plaqs_diffs[warmup_steps:][::skip_steps]
        _steps_diffs = (
            skip_steps * np.arange(_plaq_diffs.shape[0]) + skip_steps
        )
        _plaq_diffs_avg = np.mean(_plaq_diffs, axis=1)
        _plaq_diffs_err = sem(_plaq_diffs, axis=1)

        xy_data = {
            'actions': (steps_arr, actions_avg, actions_err),
            'plaqs': (steps_arr, plaqs_avg, plaqs_err),
            'charges': (steps_arr, charges.T),
            'charge_diffs': (_steps_diffs, _charge_diffs.T),
            'autocorrs': (steps_arr, autocorrs_avg, autocorrs_err),
            'plaqs_diffs': (_steps_diffs, _plaq_diffs_avg, _plaq_diffs_err)
        }

        return xy_data

    def _plot_setup(self, data, beta, run_str, weights, dir_append=None):
        """Prepare for plotting observables."""
        if dir_append:
            run_str += dir_append

        self.out_dir = os.path.join(self.figs_dir, run_str)
        io.check_else_make_dir(self.out_dir)

        #  L = self.params['space_size']
        lf_steps = self.params['num_steps']
        bs = self.params['num_samples']  # batch size
        #  qw = weights['charge_weight']
        nw = weights['net_weights']
        sw, translw, transfw = nw
        title_str = (r"$N_{\mathrm{LF}} = $" + f"{lf_steps}, "
                     r"$N_{\mathrm{B}} = $" + f"{bs}, "
                     r"$\mathrm{nw} = $" + f"{nw[0], nw[1], nw[2]}")

        #  r"$L = $" + f"{L}, "
        #  r"$\beta = $ " + f"{beta}, "
        #  r"$\alpha_{Q} = $" + f"{qw}, "
        kwargs = {
            'markers': False,
            'lines': True,
            'alpha': 0.6,
            'title': title_str,
            'legend': False,
            'ret': False,
            'out_file': [],
        }

        xy_data = self._parse_data(data, beta)

        return xy_data, kwargs

    def plot_observables(self, data, beta, run_str, weights, dir_append=None):
        """Plot observables."""
        xy_data, kwargs = self._plot_setup(data, beta, run_str,
                                           weights, dir_append)

        self._plot_actions(xy_data['actions'], **kwargs)
        self.log_figure()
        self._plot_plaqs(xy_data['plaqs'], beta, **kwargs)
        self.log_figure()
        self._plot_charges(xy_data['charges'], **kwargs)
        self.log_figure()
        self._plot_charge_probs(xy_data['charges'][1], **kwargs)
        self.log_figure()
        self._plot_charge_diffs(xy_data['charge_diffs'], **kwargs)
        self.log_figure()
        self._plot_autocorrs(xy_data['autocorrs'], **kwargs)
        self.log_figure()
        mean_diff = self._plot_plaqs_diffs(xy_data['plaqs_diffs'], **kwargs)
        self.log_figure()

        return mean_diff

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
        #  err_kwargs = plt_kwargs.update({'lw': 1.5, 'alpha': 0.7})

        ax0.plot(x, y, **plt_kwargs)
        ax0.errorbar(x, y, yerr=yerr,
                     #  ls='-', lw=1.,
                     alpha=0.7,
                     color='k',
                     ecolor='gray')

        if ax1 is not None:
            ax1.plot(x[x0:x1:10], y[x0:x1:10], **plt_kwargs)
            ax1.errorbar(x[x0:x1:10], y[x0:x1:10], yerr=yerr[x0:x1:10],
                         #  ls='-', lw=1.,
                         alpha=0.7,
                         color='k',
                         ecolor='gray')

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
            plt.savefig(out_file, dpi=400, bbox_inches='tight')

        return fig, (ax0, ax1)

    def _plot_actions(self, xy_data, **kwargs):
        """Plot actions."""
        #  kwargs['out_file'] = get_out_file(self.out_dir, 'actions_vs_step')
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
        #  kwargs['bounds'] = [0.2, 0.6, 0.7, 0.3]

        #  xy_labels = ('Step', 'Action')
        #  plot_multiple_lines(xy_data, xy_labels, **kwargs)
        #  plot_with_inset(xy_data, labels, **kwargs)

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
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

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
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

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
        #  xy_labels = ('Step', 'Autocorrelation of ' + r'$Q$')
        #  return plot_multiple_lines(xy_data, xy_labels, **kwargs)
        #  kwargs['out_file'] = get_out_file(self.out_dir,
        #                                    'charge_autocorrs_vs_step')
        #  kwargs['ret'] = True
        #  kwargs['bounds'] = [0.2, 0.6, 0.7, 0.3]
        #  try:
        #      kwargs['out_file'] = get_out_file(
        #          self.out_dir, 'charge_autocorrs_vs_step'
        #      )
        #  except AttributeError:
        #      kwargs['out_file'] = None
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
        #  _, ax, axins = plot_with_inset(xy_data, labels, **kwargs)
        #  xy_labels = ('Step', 'Autocorrelation of ' + r'$Q$')
        #  return plot_multiple_lines(xy_data, xy_labels, **kwargs)
