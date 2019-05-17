import os
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
    'text.latex.preamble': [r'\usepackage{gensymb}'],
    'axes.labelsize': 14,   # fontsize for x and y labels (was 10)
    'axes.titlesize': 16,
    'legend.fontsize': 10,  # was 10
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': True,
    #  'figure.figsize': [fig_width, fig_height],
    'font.family': 'serif'
}

mpl.rcParams.update(params)


class LeapfrogPlotter:
    def __init__(self, run_logger, figs_dir, therm_perc=0.005, skip_perc=0.01):
        self.figs_dir = figs_dir
        self.eps_dir = os.path.join(self.figs_dir, 'eps_plots')
        io.check_else_make_dir(self.eps_dir)
        self.run_logger = run_logger
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

    def make_plots(self, num_samples=15):
        self.plot_lf_diffs(num_samples)
        self.plot_logdets(num_samples)

    def get_colors(self, num_samples=10):
        reds_cmap = mpl.cm.get_cmap('Reds', num_samples + 1)
        blues_cmap = mpl.cm.get_cmap('Blues', num_samples + 1)
        idxs = np.linspace(0., 0.75, num_samples + 1)
        reds = [reds_cmap(i) for i in idxs]
        blues = [blues_cmap(i) for i in idxs]

        return reds, blues

    def plot_lf_diffs(self, num_samples=10):
        reds, blues = self.get_colors(num_samples)
        #  therm_steps = int(0.005 * num_leapfrog_steps)
        #  skip_steps = int(0.01 * num_leapfrog_steps)

        step_multiplier = (self.lf_f_diffs.shape[0]
                           // self.samples_diffs.shape[0])
        samples_x_avg, samples_y_avg = smooth_data(
            np.mean(self.samples_diffs, axis=(1, 2)),
            self.therm_steps // step_multiplier,
            self.skip_steps // step_multiplier
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
            _ = ax1.plot(xf, yf[:, idx], color=reds[idx], **indiv_kwargs)
            _ = ax1.plot(xb, yb[:, idx], color=blues[idx], **indiv_kwargs)

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

    def plot_logdets(self, num_samples=10):
        reds, blues = self.get_colors(num_samples)
        #  therm_steps = 10
        #  skip_steps = 100
        step_multiplier = (self.lf_f_diffs.shape[0]
                           // self.samples_diffs.shape[0])
        sumlogdet_xf_avg, sumlogdet_yf_avg = smooth_data(
            np.mean(self.sumlogdet_f, axis=-1),
            self.therm_steps // step_multiplier,
            self.skip_steps // step_multiplier
        )

        sumlogdet_xb_avg, sumlogdet_yb_avg = smooth_data(
            np.mean(self.sumlogdet_b, axis=-1),
            self.therm_steps // step_multiplier,
            self.skip_steps // step_multiplier
        )

        fig, (ax1, ax2) = plt.subplots(2, 1)
        for idx in range(num_samples):
            xf, yf = smooth_data(self.logdets_f[:, idx],
                                 self.therm_steps, self.skip_steps)
            xb, yb = smooth_data(self.logdets_b[:, idx],
                                 self.therm_steps, self.skip_steps)
            _ = ax1.plot(xf, yf, ls='-', color=reds[idx], alpha=0.75, lw=0.5)
            _ = ax1.plot(xb, yb, ls='-', color=blues[idx], alpha=0.75, lw=0.5)

        xf_avg, yf_avg = smooth_data(np.mean(self.logdets_f, axis=-1),
                                     self.therm_steps, self.skip_steps)
        xb_avg, yb_avg = smooth_data(np.mean(self.logdets_b, axis=-1),
                                     self.therm_steps, self.skip_steps)
        _ = ax1.plot(xf_avg, yf_avg, label=f'avg. logdet (forward)',
                     ls='-', color=reds[-1], lw=1.)
        _ = ax1.plot(xb_avg, yb_avg, label=f'avg. logdet (backward)',
                     ls='-', color=blues[-1], lw=1.)

        _ = ax2.plot(sumlogdet_xf_avg, sumlogdet_yf_avg,
                     label=f'sumlogdet (f)',
                     color=reds[-1], lw=1., ls='-')
        _ = ax2.plot(sumlogdet_xb_avg, sumlogdet_yb_avg,
                     label=f'sumlogdet (b)',
                     color=blues[-1], lw=1., ls='-')

        _ = ax1.set_xlabel('Leapfrog step')
        _ = ax2.set_xlabel('MD step')
        _ = ax1.legend(loc='best')
        _ = ax2.legend(loc='best')

        out_file = os.path.join(self.figs_dir, 'avg_logdets.png')
        out_file_eps = os.path.join(self.eps_dir, 'avg_logdets.eps')
        io.log(f'Saving figure to: {out_file}')
        _ = plt.savefig(out_file, dpi=400, bbox_inches='tight')
        _ = plt.savefig(out_file_eps, dpi=400, bbox_inches='tight')
