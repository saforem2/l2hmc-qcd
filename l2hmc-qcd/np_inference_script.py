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

from config import NetWeights
from plot_script import get_matching_log_dirs
from gauge_inference_np import inference_plots
from utils.file_io import timeit
from runners.runner_np import create_dynamics, load_pkl, run_inference_np
from lattice.lattice import u1_plaq_exact
from plotters.seaborn_plots import get_lf

import autograd.numpy as np

sns.set_palette('bright')


def main():
    root_dir = os.path.abspath('/home/foremans/DLHMC/l2hmc-qcd/gauge_logs')
    dates = ['2019_12_15',
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
             '2020_01_08']
    log_dirs = []
    for date in dates:
        ld = get_matching_log_dirs(date, root_dir=root_dir)
        for log_dir in ld:
            if 'x011' in log_dir or 'v011' in log_dir:
                continue
            log_dirs += [log_dir]

    log_dirs = sorted(log_dirs, key=get_lf)#, reverse=True)
    for log_dir in log_dirs:
        rd_l2hmc = None
        ed_l2hmc = None
        rp_l2hmc = None

        rd_hmc = None
        ed_hmc = None
        rp_hmc = None

        dynamics = None

        try:
            params = load_pkl(os.path.join(log_dir, 'parameters.pkl'))
        except FileNotFoundError:
            continue

        try:
            dynamics, lattice = create_dynamics(log_dir)
        except:
            io.log(f'Unable to create dynamics for {log_dir}. Skipping!')
            continue

        nw_l2hmc = NetWeights(x_scale=1,
                              x_translation=1,
                              x_transformation=1,
                              v_scale=1,
                              v_translation=1,
                              v_transformation=1)
        rp_l2hmc = {
            'beta': 5.,
            'eps': dynamics.eps,
            'net_weights': nw_l2hmc,
            'run_steps': 5000,
        }

        rp_l2hmc, rd_l2hmc, ed_l2hmc = run_inference_np(log_dir,
                                                        dynamics,
                                                        lattice,
                                                        rp_l2hmc,
                                                        init='rand')
        io.log(80 * '-')
        dataset_l2hmc = inference_plots(rd_l2hmc, params, rp_l2hmc)
        io.log(80 * '-')
        #plt.close('all')
        #plt.clf()
        #plt.cla()

        nw_hmc = NetWeights(x_scale=0,
                            x_translation=0,
                            x_transformation=0,
                            v_scale=0,
                            v_translation=0,
                            v_transformation=0)
        rp_hmc = {
            'beta': 5.,
            'eps': dynamics.eps,
            'net_weights': nw_hmc,
            'run_steps': 5000,
        }

        rp_hmc, rd_hmc, ed_hmc = run_inference_np(log_dir,
                                                  dynamics,
                                                  lattice,
                                                  rp_hmc,
                                                  init='rand')
        io.log(80 * '-')
        datset_hmc = inference_plots(rd_hmc, params, rp_hmc)
        io.log(80 * '-')
        plt.close('all')
        plt.clf()
        plt.cla()

        def therm_arr(arr):
            arr = np.array(arr)
            num_steps = arr.shape[0]
            therm_steps = int(0.25 * num_steps)
            arr = arr[therm_steps:, :]
            return arr

        base_dir = os.path.abspath('/home/foremans/inference_numpy_figs')
        log_str = log_dir.split('/')[-1]
        out_dir = os.path.join(base_dir, log_str)
        io.check_else_make_dir(out_dir)

        out_file = os.path.join(out_dir, 'plaqs_diffs_plots.pdf')
        title_str = (r"$N_{\mathrm{LF}} = $" + f"{params['num_steps']}, "
                     + r"$\varepsilon = $" + f"{rp_l2hmc['eps']}")

        if params['eps_fixed']:
            title_str += '(fixed), '
        else:
            title_str += ', '

        title_str += (r"$N_{\mathrm{B}}$" + f"{params['batch_size']}, "
                      + r"$\beta = $" + f"{rp_l2hmc['beta']}")

        plaq_exact = u1_plaq_exact(rp_l2hmc['beta'])
        y1 = plaq_exact - therm_arr(rd_l2hmc['plaqs']).flatten()
        y2 = plaq_exact - therm_arr(rd_hmc['plaqs']).flatten()

        fig, ax = plt.subplots()
        sns.kdeplot(y1, shade=True, label='nw: (1, 1, 1, 1, 1, 1)', ax=ax)
        sns.kdeplot(y2, shade=True, label='nw: (0, 0, 0, 0, 0, 0)', ax=ax)
        ax.set_xlabel(r"""$\delta \phi_{P}$""", fontsize='large')

        io.log(80 * '-')
        io.log(f'Saving plaqs_diffs figure to: {out_file}.')
        io.log(80 * '-')
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
        #  io.log(f'Unable to create dynamics for {log_dir}. Skipping!')
        plt.close('all')

        #  except:
        #      import pudb; pudb.set_trace()
        #      io.log(f'Unable to create dynamics for {log_dir}. Skipping!')
        #      continue


if __name__ == '__main__':
    main()
