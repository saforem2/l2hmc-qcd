"""
np_inference_script.py

Runs tensorflow-independent inference on trained model.

Specifically, we want to get inference data for all of the previously trained
models that have saved weights to compare against the results obtained from
running inference with tensorflow.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

import seaborn as sns
import utils.file_io as io
import numpy as np

from config import NetWeights, Weights
from plot_script import get_matching_log_dirs
from gauge_inference_np import inference_plots
from utils.file_io import timeit
from lattice.lattice import u1_plaq_exact
from runners.runner_np import (_get_eps, create_dynamics, load_pkl,
                               run_inference_np)
from plotters.seaborn_plots import get_lf

sns.set_palette('bright')


def therm_arr(arr, therm_frac=0.25):
    step_axis = np.argmax(arr.shape)
    num_steps = arr.shape[step_axis]
    therm_steps = int(therm_frac * num_steps)
    arr = np.delete(arr, np.s_[:therm_steps], axis=step_axis)

    return arr


def build_log_dirs():
    """Build array of log_dirs."""
    root_dir = os.path.abspath('/home/foremans/DLHMC/l2hmc-qcd/gauge_logs')
    dates = [
        '2019_12_15',
        '2019_12_16',
        '2019_12_17',
        '2019_12_18',
        '2019_12_19',
        '2019_12_20',
        '2019_12_21',
        '2019_12_22',
        '2019_12_23',
        '2019_12_24',
        '2019_12_25',
        '2019_12_26',
        '2019_12_27',
        '2019_12_28',
        '2019_12_29',
        '2019_12_30',
        '2019_12_31',
        '2020_01_02',
        '2020_01_03',
        '2020_01_04',
        '2020_01_05',
        '2020_01_06',
        '2020_01_07',
        '2020_01_08',
        '2020_01_14',
        '2020_01_15',
    ]

    log_dirs = []
    for date in dates:
        ld = get_matching_log_dirs(date, root_dir=root_dir)
        for log_dir in ld:
            log_dirs += [log_dir]

    return log_dirs


def plot_plaqs_diffs(run_data_dict, log_dir, params, run_params, **kwargs):
    """Plot plaqs_diffs for all key, val pairs in `run_data_dict`."""
    fig_dir = os.path.join(log_dir, 'figures_np', 'plaqs_diffs')
    io.check_else_make_dir(fig_dir)
    out_file = os.path.join(fig_dir, 'plaqs_diffs_plots.pdf')
    title_str = (r"$N_{\mathrm{LF}} = $" + f"{params['num_steps']}, "
                 + r"$\varepsilon = $" + f"{run_params['eps']:.3g}")

    out_dir = kwargs.get('out_dir', None)
    if out_dir is not None:
        io.check_else_make_dir(out_dir)
        out_file_ = os.path.join(out_dir, 'plaqs_diffs_plots.pdf')

    if params['eps_fixed']:
        title_str += ' (fixed), '
    else:
        title_str += ', '

    title_str += (r"$N_{\mathrm{B}} = $" + f"{params['batch_size']}, "
                  + r"$\beta = $" + f"{run_params['beta']}")

    plaq_exact = u1_plaq_exact(run_params['beta'])

    fig, ax = plt.subplots()
    for net_weights, run_data in run_data_dict.items():
        plaqs = np.array(run_data['plaqs'])
        y = plaq_exact - therm_arr(plaqs).flatten()
        px_label = (r"""$\langle A(\xi^{\prime}|\xi) = $"""
                    + f"{np.mean(run_data['accept_prob']):.3g}")
        label = f'nw: {tuple(net_weights)}, ' + px_label
        sns.kdeplot(y, shade=True, label=label, ax=ax)

    ax.legend(loc='best')
    ax.set_xlabel(r"""$\delta \phi_{P}$""", fontsize='large')
    ax.set_title(title_str, fontsize='x-large')

    io.log(80 * '-')
    io.log(f'Saving plaqs_diffs figure to: {out_file}.')
    io.log(80 * '-')
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    if out_dir is not None:
        fig.savefig(out_file_, dpi=200, bbox_inches='tight')

    return fig, ax


@timeit
def run_and_plot(log_dir, net_weights, run_steps=10000,
                 beta=5., init='rand', skip=True):
    """Run `np_inference` on the saved model in `log_dir` w/ `net_weights`."""
    dataset = None
    energy_dataset = None
    try:
        params = load_pkl(os.path.join(log_dir, 'parameters.pkl'))
    except FileNotFoundError:
        io.log(f'Unable to load params from {log_dir}. Skipping!')
        return dataset, energy_dataset

    run_params = {
        'eps': _get_eps(log_dir),
        'num_steps': params['num_steps'],
        'batch_size': params['batch_size'],
        'hmc': bool(net_weights == NetWeights(0, 0, 0, 0, 0, 0)),
    }
    dynamics, lattice = create_dynamics(log_dir, **run_params)

    run_params.update({
        'beta': beta,
        'log_dir': log_dir,
        'run_steps': run_steps,
        'net_weights': net_weights,
    })
    outputs = run_inference_np(log_dir, dynamics, lattice,
                               run_params, init=init, skip=skip)
    run_data = outputs['run_data']
    energy_data = outputs['energy_data']
    run_params = outputs['run_params']
    existing_flag = outputs['existing_flag']
    try:
        if not skip and not existing_flag:
            dataset, energy_dataset = inference_plots(run_data, energy_data,
                                                      params, run_params)
        else:
            dataset = None
            energy_dataset = None
    except:
        dataset = None
        energy_dataset = None

    outputs = {
        'params': params,
        'run_params': run_params,
        'run_data': run_data,
        'dataset': dataset,
        'energy_data': energy_data,
        'energy_dataset': energy_dataset,
    }

    return outputs


@timeit
def outer_loop(log_dirs, run_steps=10000, beta=5., skip=True):
    """Loop over `log_dirs`, running inference w/ numpy on each saved model."""
    nws_arr = [
        NetWeights(1, 1, 1, 1, 1, 1),
        NetWeights(1, 0, 1, 1, 1, 1),
        NetWeights(0, 0, 0, 0, 0, 0),
    ]

    for log_dir in log_dirs:
        run_data_dict = {}
        for net_weights in nws_arr:
            try:
                outputs = run_and_plot(log_dir, net_weights,
                                       run_steps=run_steps,
                                       beta=beta, init='rand', skip=skip)
                run_data_dict[tuple(net_weights)] = outputs['run_data']
            except:
                continue
        plt.close('all')

        try:
            params = outputs['params']
            run_params = outputs['run_params']
            _, _ = plot_plaqs_diffs(run_data_dict, log_dir, params, run_params)
        except:
            continue


def main():
    """Main function."""
    log_dirs = build_log_dirs()
    outer_loop(log_dirs, run_steps=10000, skip=True)


if __name__ == '__main__':
    main()
