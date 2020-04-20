"""
train_plots.py

Contains helper method for plotting training data.

Author: Sam Foreman
Date: 04/20/2020
"""
import os
import utils.file_io as io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

MARKERS = 10 * ['.', 'x', '+', '^', 'v', '<', '>', 'D']
COLORS = 10 * ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
sns.set_palette('bright')


def get_title_str(params):
    """Parse `params` to create title string for plots."""
    num_steps = params.get('num_steps', None)
    batch_size = params.get('batch_size', None)
    train_steps = params.get('train_steps', None)
    time_size = params.get('time_size', None)
    space_size = params.get('space_size', None)
    plaq_weight = params.get('plaq_weight', 0.)
    charge_weight = params.get('charge_weight', 0.)
    std_weight = params.get('std_weight', 0.)

    title_str = (f'{time_size}' + r"$\times$" + f'{space_size}, '
                 + r"$N_{\mathrm{B}} = $" + f'{batch_size}, '
                 + r"$N_{\mathrm{LF}} = $" + f'{num_steps}, '
                 + r"$N_{\mathrm{train}} = $" + f'{train_steps}, ')

    if std_weight > 0:
        title_str += r"$\alpha_{\mathrm{STD}} = $" + f'{std_weight:.2g}, '
    if plaq_weight > 0:
        title_str += r"$\alpha_{\mathrm{plaq}} = $" + f'{plaq_weight:.2g}, '
    if charge_weight > 0:
        title_str += r"$\alpha_{\mathrm{Q}} = $" + f'{charge_weight:.2g}, '

    return title_str


def plot_train_data(train_data, params):
    """Plot all training data and save figures to `log_dir/train_plots`.

    Args:
        train_data (dict): Dictionary of training data.
        params (dict): Dictionary of parameters used for training.
    """
    title_str = get_title_str(params)
    log_dir = params.get('log_dir', None)
    out_dir = os.path.join(log_dir, 'train_plots')
    io.check_else_make_dir(out_dir)

    for idx, (key, val) in enumerate(train_data.items()):
        if key == 'train_op':
            continue

        val = np.array(val)
        beta = np.array(train_data['beta'])

        fig, ax = plt.subplots()
        if len(val.shape) == 1:
            ax.plot(beta, val, label=key, ls='',
                    marker=MARKERS[idx], color=COLORS[idx])
        if len(val.shape) == 2:
            ax.plot(beta, val.mean(axis=1), label=key, ls='',
                    marker=MARKERS[idx], color=COLORS[idx])
        if len(val.shape) == 3:
            ax.plot(beta, val.mean(axis=(1, 2)), label=key, ls='',
                    marker=MARKERS[idx], color=COLORS[idx])

        ax.legend(loc='best')
        ax.set_xlabel(r"$\beta$", fontsize='large')
        ax.set_title(title_str, fontsize='x-large')
        plt.tight_layout()
        out_file = os.path.join(out_dir, f'{key}.png')
        io.log(f'Saving figure to: {out_file}.')
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close('all')

