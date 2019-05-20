import os
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import pickle
import numpy as np
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

#  from .formatters import latexify
import utils.file_io as io
from utils.data_loader import DataLoader


def load_and_sep(out_file, keys=('forward', 'backward')):
    with open(out_file, 'rb') as f:
        data = pickle.load(f)

    return (np.array(data[k]) for k in keys)


def smooth_data(y_data, therm_steps=10, skip_steps=100):
    skip_steps = max(1, skip_steps)
    if therm_steps > 0:
        _y_data = y_data[therm_steps:][::skip_steps]
    else:
        _y_data = y_data[::skip_steps]
    x_data = skip_steps * np.arange(_y_data.shape[0])

    return x_data, _y_data


params = {
    'backend': 'ps',
    #  'text.latex.preamble': [r'\usepackage{gensymb}'],
    'axes.labelsize': 14,   # fontsize for x and y labels (was 10)
    'axes.titlesize': 16,
    'legend.fontsize': 10,  # was 10
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    #  'text.usetex': True,
    #  'figure.figsize': [fig_width, fig_height],
    'font.family': 'serif'
}

try:
    mpl.rcParams.update(params)
except FileNotFoundError:
    params['text.usetex'] = False
    params['text.latex.preamble'] = None
    try:
        mpl.rcParams.update(params)
    except FileNotFoundError:
        pass


class LeapfrogPlotter:
    def __init__(self, figs_dir, run_logger=None,
                 run_dir=None, therm_perc=0.005, skip_perc=0.01):
        self.figs_dir = figs_dir
        self.eps_dir = os.path.join(self.figs_dir, 'eps_plots')
        io.check_else_make_dir(self.eps_dir)

        if run_logger is None:
            if run_dir is None:
                raise AttributeError(
                    """Either a `run_logger` object containing data or a
                    `run_dir` from which to load data must be specified.
                    Exiting.
                    """
                )
            else:
                self.load_data(run_dir)

        else:
            self.samples = np.array(run_logger.samples_arr)
            self.lf_f = np.array(run_logger.lf_out['forward'])
            self.lf_b = np.array(run_logger.lf_out['backward'])
            self.logdets_f = np.array(run_logger.logdets['forward'])
            self.logdets_b = np.array(run_logger.logdets['backward'])
            self.sumlogdet_f = np.array(run_logger.sumlogdet['forward'])
            self.sumlogdet_b = np.array(run_logger.sumlogdet['backward'])

        self.lf_f_diffs = self.lf_f[1:] - self.lf_f[:-1]
        self.lf_b_diffs = self.lf_b[1:] - self.lf_b[:-1]
        self.samples_diffs = self.samples[1:] - self.samples[:-1]
        self.tot_lf_steps = self.lf_f_diffs.shape[0]
        self.tot_md_steps = self.samples_diffs.shape[0]
        self.num_lf_steps = self.tot_lf_steps // self.tot_md_steps
        self.therm_steps = int(therm_perc * self.tot_lf_steps)
        self.skip_steps = int(skip_perc * self.tot_lf_steps)
        self.step_multiplier = (
            self.lf_f_diffs.shape[0] // self.samples_diffs.shape[0]
        )

    def load_data(self, run_dir):
        loader = DataLoader(run_dir)
        io.log("Loading leapfrogs...")
        self.lf_f, self.lf_b = loader.load_leapfrogs(run_dir)
        io.log("Loading logdets...")
        self.logdets_f, self.logdets_b = loader.load_logdets(run_dir)
        io.log("Loading sumlogdets...")
        self.sumlogdet_f, self.sumlogdet_b = loader.load_sumlogdets(run_dir)


    def make_plots(self, run_dir, num_samples=20):
        """Make plots of the leapfrog differences and logdets.

        Immediately after creating and saving the plots, delete these
        (no-longer needed) attributes to free up memory.

        Args:
            run_dir (str): Path to directory in which to save all of the
                relevant instance attributes.
            num_samples (int): Number of samples to include when creating
                plots.
        """
        self.plot_lf_diffs(num_samples)

        self.print_memory()

        self.save_attr('lf_forward', self.lf_f, out_dir=run_dir)
        del self.lf_f
        del self.lf_f_diffs
        self.print_memory()

        self.save_attr('lf_backward', self.lf_b, out_dir=run_dir)
        del self.lf_b
        del self.lf_b_diffs
        self.print_memory()

        self.save_attr('samples_out', self.samples, out_dir=run_dir)
        del self.samples
        del self.samples_diffs
        self.print_memory()

        self.plot_logdets(num_samples)

        self.save_attr('logdets_forward', self.logdets_f, out_dir=run_dir)
        del self.logdets_f
        self.print_memory()

        self.save_attr('logdets_backward', self.logdets_b, out_dir=run_dir)
        del self.logdets_b
        self.print_memory()

        self.save_attr('sumlogdet_forward', self.sumlogdet_f, out_dir=run_dir)
        del self.sumlogdet_f
        self.print_memory()

        self.save_attr('sumlogdet_backward', self.sumlogdet_b, out_dir=run_dir)
        del self.sumlogdet_b
        self.print_memory()

    def print_memory(self):
        if HAS_PSUTIL:
            pid = os.getpid()
            py = psutil.Process(pid)
            memory_use = py.memory_info()[0] / 2. ** 30
            io.log(80 * '-')
            io.log(f'memory use: {memory_use}')
            io.log(80 * '-')

    def get_colors(self, num_samples=20):
        reds_cmap = mpl.cm.get_cmap('Reds', num_samples + 1)
        blues_cmap = mpl.cm.get_cmap('Blues', num_samples + 1)
        idxs = np.linspace(0.1, 0.75, num_samples + 1)
        reds = [reds_cmap(i) for i in idxs]
        blues = [blues_cmap(i) for i in idxs]

        return reds, blues

    def save_attr(self, name, attr, out_dir):
        assert os.path.isdir(out_dir)
        out_file = os.path.join(out_dir, name + '.npz')
        io.log(f'Saving {name} to: {out_file}')
        np.savez_compressed(out_file, attr)

    #  def save_data(self, out_dir):
    #      assert os.path.isdir(out_dir)
    #      for key, val in self.__dict__.items():
    #          if isinstance(val, np.ndarray):
    #              self.save_attr(key, val, out_dir)

    def plot_lf_diffs(self, num_samples=10):
        reds, blues = self.get_colors(num_samples)
        samples_x_avg, samples_y_avg = smooth_data(
            np.mean(self.samples_diffs, axis=(1, 2)),
            self.therm_steps // self.step_multiplier,
            self.skip_steps // self.step_multiplier
        )

        indiv_kwargs = {
            'ls': '-',
            'alpha': 0.75,
            'lw': 0.5
        }

        fig, (ax1, ax2) = plt.subplots(2, 1)
        for idx in range(num_samples):
            xf, yf = smooth_data(np.mean(self.lf_f_diffs, axis=-1),
                                 self.therm_steps, self.skip_steps)
            xb, yb = smooth_data(np.mean(self.lf_b_diffs, axis=-1),
                                 self.therm_steps, self.skip_steps)
            _ = ax1.plot(xf, yf[:, idx],
                         color=reds[idx], **indiv_kwargs)
            _ = ax1.plot(xb, yb[:, idx],
                         color=blues[idx], **indiv_kwargs)

        xf_avg, yf_avg = smooth_data(np.mean(self.lf_f_diffs, axis=(1, 2)),
                                     self.therm_steps, self.skip_steps)
        xb_avg, yb_avg = smooth_data(np.mean(self.lf_b_diffs, axis=(1, 2)),
                                     self.therm_steps, self.skip_steps)
        _ = ax1.plot(xf_avg, yf_avg, label='avg. diff (forward)',
                     color=reds[-1], lw=1.)
        _ = ax1.plot(xb_avg, yb_avg, label='avg. diff (backward)',
                     color=blues[-1], lw=1.)

        _ = ax2.plot(samples_x_avg, samples_y_avg, color='k', lw=1.,
                     label='avg. output diff')

        _ = ax1.set_xlabel('Leapfrog step')  # , fontsize=14)
        _ = ax2.set_xlabel('MD step')

        ylabel = r'$\langle \delta\phi_{\mu}(i)\rangle$'
        _ = ax1.set_ylabel(ylabel)
        _ = ax2.set_ylabel(ylabel)

        _ = ax1.legend(loc='best')
        _ = ax2.legend(loc='best')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.5)

        out_file = os.path.join(self.figs_dir, 'leapfrog_diffs.png')
        out_file_eps = os.path.join(self.eps_dir, 'leapfrog_diffs.eps')
        io.log(f'Saving figure to: {out_file}')
        _ = plt.savefig(out_file, dpi=400, bbox_inches='tight')
        _ = plt.savefig(out_file_eps, dpi=400, bbox_inches='tight')

    def plot_logdets(self, num_samples=20):
        reds, blues = self.get_colors(num_samples)
        sumlogdet_xf_avg, sumlogdet_yf_avg = smooth_data(
            np.mean(self.sumlogdet_f, axis=-1),
            self.therm_steps // self.step_multiplier,
            self.skip_steps // self.step_multiplier
        )

        sumlogdet_xb_avg, sumlogdet_yb_avg = smooth_data(
            np.mean(self.sumlogdet_b, axis=-1),
            self.therm_steps // self.step_multiplier,
            self.skip_steps // self.step_multiplier
        )

        fig, (ax1, ax2) = plt.subplots(2, 1)
        for idx in range(num_samples):
            xf, yf = smooth_data(self.logdets_f[:, idx],
                                 self.therm_steps, self.skip_steps)
            xb, yb = smooth_data(self.logdets_b[:, idx],
                                 self.therm_steps, self.skip_steps)
            _ = ax1.plot(xf, yf, color=reds[idx], alpha=0.75, lw=0.5)
            _ = ax1.plot(xb, yb, color=blues[idx], alpha=0.75, lw=0.5)

        xf_avg, yf_avg = smooth_data(np.mean(self.logdets_f, axis=-1),
                                     self.therm_steps, self.skip_steps)
        xb_avg, yb_avg = smooth_data(np.mean(self.logdets_b, axis=-1),
                                     self.therm_steps, self.skip_steps)
        _ = ax1.plot(xf_avg, yf_avg,
                     label=r"$|\mathrm{avg. logdet (f)}|$",
                     ls='-', color=reds[-1], lw=1.)
        _ = ax1.plot(xb_avg, yb_avg,
                     label=r"$|\mathrm{avg. logdet (b)}|$",
                     ls='-', color=blues[-1], lw=1.)

        _ = ax2.plot(sumlogdet_xf_avg, sumlogdet_yf_avg,
                     label=r"$|\mathrm{sumlogdet (f)}|$",
                     color=reds[-1], lw=1., ls='-')
        _ = ax2.plot(sumlogdet_xb_avg, sumlogdet_yb_avg,
                     label=r"$|\mathrm{sumlogdet (b)}|$",
                     color=blues[-1], lw=1., ls='-')

        _ = ax1.set_xlabel('Leapfrog step')
        _ = ax2.set_xlabel('MD step')
        _ = ax1.legend(loc='best')
        _ = ax2.legend(loc='best')
        _ = fig.tight_layout()
        _ = fig.subplots_adjust(hspace=0.5)

        out_file = os.path.join(self.figs_dir, 'avg_logdets.png')
        out_file_eps = os.path.join(self.eps_dir, 'avg_logdets.eps')
        io.log(f'Saving figure to: {out_file}')
        _ = plt.savefig(out_file, dpi=400, bbox_inches='tight')
        _ = plt.savefig(out_file_eps, dpi=400, bbox_inches='tight')
