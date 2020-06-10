"""
hmc_runner.py

Implements methods for running generic HMC using numpy.

Author: Sam Foreman (github: @saforem2)
Date: 05/29/2020
"""
import os
import shutil

import numpy as np
import seaborn as sns
import matplotlib.style as mplstyle

import utils.file_io as io

from config import NET_WEIGHTS_HMC, PI
from utils.attr_dict import AttrDict
from plotters.inference_plots import inference_plots
from .runner_np import RunnerNP, RunParams

sns.set_palette('bright')
mplstyle.use('fast')


def calc_plaqs_np(x, n=1):
    plaqs = (x[..., 0]
             - x[..., 1]
             - np.roll(x[..., 0], shift=-n, axis=2)
             + np.roll(x[..., 1], shift=-n, axis=1))
    return plaqs


def get_lattice_action_np(lattice_shape, n_arr=None):
    def lattice_action(x):
        x = np.reshape(x, lattice_shape)
        plaqs = calc_plaqs_np(x)
        action = np.sum(1. - np.cos(plaqs), axis=(1, 2))
        return action
    return lattice_action


def run_hmc_np(log_dir, train_params, run_params):
    """Run HMC using numpy."""
    runner = RunnerNP(run_params,
                      log_dir=log_dir,
                      train_params=train_params,
                      model_type='GaugeModel',
                      from_trained_model=False)
    x = np.random.uniform(-PI, PI, size=runner.config.input_shape)
    run_data = runner.inference(x=x, run_steps=run_params.run_steps)
    _, _, fig_dir = inference_plots(run_data, runner.config.run_params,
                                    runner.config)
    out_file = os.path.join(fig_dir, 'run_summary.txt')
    _, _ = run_data.log_summary(out_file, n_boot=10)
    run_data.save(run_dir=runner.config.run_dir)
    io.save_dict(run_params._asdict(), runner.config.run_dir, 'run_params')
    _ = shutil.copy2(out_file, runner.config.run_dir)

    return run_data


# pylint: disable=invalid-name, too-many-locals
def run_hmc_loop(kwargs):
    """Run HMC Loop."""
    print_steps = kwargs.get('print_steps', 100)
    batch_size = kwargs.get('batch_size', 32)
    run_steps = kwargs.get('run_steps', 10000)
    lf_arr = kwargs.get('lf_arr', [2, 3, 4])
    eps_arr = kwargs.get('eps_arr', [0.05, 0.075, 0.1, 0.125, 0.15])
    beta_arr = kwargs.get('beta_arr', [4., 4.25, 4.5, 4.75, 5.])
    time_size = kwargs.get('time_size', 16)
    space_size = kwargs.get('space_size', 16)
    bdir = os.path.abspath('~/l2hmc-qcd/gauge_logs/HMC_LOGS_NP')
    base_dir = kwargs.get('base_dir', bdir)
    dim = 2

    for lf in lf_arr:
        lf_dir = os.path.join(base_dir, f'lf{lf}_bs{batch_size}')
        print(80 * '*')
        print(f'LF: {lf}')
        for beta in beta_arr:
            print(80 * '=')
            print(f'BETA: {beta}')
            beta_dir = os.path.join(lf_dir, f'beta{beta}')
            for eps in eps_arr:
                print(80 * '-')
                print(f'eps: {eps}')
                eps_dir = os.path.join(beta_dir, f'eps{eps:.3g}')
                cond1 = os.path.isdir(eps_dir)
                rd = os.path.join(eps_dir, 'runs_np')
                cond2 = os.path.isdir(rd)
                if cond1 and cond2:
                    continue
                else:
                    io.check_else_make_dir(eps_dir)
                train_params = AttrDict({
                    'dim': dim,
                    'time_size': time_size,
                    'space_size': space_size,
                    'batch_size': batch_size,
                    'num_steps': lf,
                    'beta_final': beta,
                    'plaq_weight': 0.1,
                    'charge_weight': 0.1,
                    'log_dir': eps_dir,
                })
                io.save_dict(dict(train_params), eps_dir, 'params')

                run_params = RunParams(
                    beta=beta,
                    eps=eps,
                    init='rand',
                    run_steps=run_steps,
                    num_steps=lf,
                    batch_size=batch_size,
                    print_steps=print_steps,
                    mix_samplers=False,
                    num_singular_values=-1,
                    net_weights=NET_WEIGHTS_HMC,
                    network_type='GaugeNetwork',
                )
                _ = run_hmc_np(eps_dir, train_params, run_params)
                print(80 * '-')
            print(80 * '=')
        print(80 * '=')
