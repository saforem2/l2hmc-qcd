"""
gauge_inference_np.py

Runs tensorflow independent inference on a trained model.

Author: Sam Foreman (github: @saforem2)
Date: 01/09/2020
"""
import os

import arviz as az
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import utils.file_io as io

from config import NetWeights
from lattice.lattice import u1_plaq_exact
from runners.runner_np import (_update_params, _get_eps, create_dynamics,
                               create_lattice, load_pkl, run_inference_np)
from plotters.seaborn_plots import plot_setup
from plotters.inference_plots import inference_plots, build_dataset
from utils.file_io import timeit
from utils.parse_inference_args_np import parse_args as parse_inference_args
from loggers.inference_summarizer import InferenceSummarizer

SEPERATOR = 80 * '-'

mpl.rcParams['axes.formatter.limits'] = -4, 4


def _get_title(params, run_params):
    lf = params['num_steps']
    beta = run_params['beta']
    eps = run_params['eps']
    title_str = (r"$N_{\mathrm{LF}} = $" + f'{lf}, '
                 r"$\beta = $" + f'{beta:.1g}, '
                 r"$\varepsilon = $" + f'{eps:.3g}')

    if params['eps_fixed']:
        title_str += ' (fixed)'

    if params['clip_value'] > 0:
        clip_value = params['clip_value']
        title_str += f', clip: {clip_value}'

    return title_str


def _check_existing(out_dir, fname):
    if os.path.isfile(os.path.join(out_dir, f'{fname}.pdf')):
        timestr = io.get_timestr()
        hour_str = timestr['hour_str']
        fname += f'_{hour_str}'

    return fname


def make_csv(run_data, energy_data, run_params):
    """Make .csv file containing relevant inference data."""
    plaq_exact = u1_plaq_exact(run_params['beta'])
    csv_dict = {}
    for e_key, e_val in energy_data.items():
        arr = np.squeeze(np.array(e_val)).flatten()
        csv_dict[e_key] = arr
        _shape = arr.shape

    for r_key, r_val in run_data.items():
        arr = np.squeeze(np.array(r_val)).flatten()
        if arr.shape[0] != _shape[0]:
            factor = int(_shape[0] / arr.shape[0])
            arr = np.array(factor * [arr]).flatten()
            #  arr = np.squeeze(np.array(
            #      int((_shape[0]/arr.shape[0])) *[arr]))
        #  if arr.shape[0] == _shape[0] / 2:
        #      arr = np.squeeze(np.array([arr for _ in _shape]))
        if r_key == 'plaqs':
            csv_dict['plaqs_diffs'] = plaq_exact - np.squeeze(np.array(r_val))
        else:
            csv_dict[r_key] = arr
            #  csv_dict[r_key] = np.squeeze(np.array(r_val.flatten()))
    try:
        csv_df = pd.DataFrame(csv_dict)
    except:
        for key, val in csv_dict.items():
            n1 = val.flatten().shape[0]
            n2 = _shape[0]
            if n1 == n2:
                csv_dict[key] = val.flatten()
            else:
                csv_dict[key] = np.array(int(n2/n1) * [val]).flatten()
        csv_df = pd.DataFrame(csv_dict)

    csv_file = os.path.join(run_params['run_dir'], 'inference_data.csv')
    io.log(f'Saving inference data to {csv_file}.')
    csv_df.to_csv(csv_file, mode='a')
    return csv_dict


@timeit
def main(args):
    """Perform tensorflow-independent inference on a trained model."""
    log_dir = getattr(args, 'log_dir', None)
    if log_dir is None:
        params_file = os.path.join(os.getcwd(), 'params.pkl')
    else:
        log_dir = os.path.abspath(log_dir)
        params_file = os.path.join(log_dir, 'parameters.pkl')

    params = load_pkl(params_file)
    log_dir = params['log_dir']

    eps = _get_eps(log_dir) if args.eps is None else args.eps

    params = _update_params(params,
                            eps=eps,
                            num_steps=args.num_steps,
                            batch_size=args.batch_size,
                            num_singular_values=args.num_singular_values)

    lattice = create_lattice(params)
    _fn = lattice.calc_actions_np

    log_dir = params['log_dir']
    dynamics = create_dynamics(log_dir,
                               potential_fn=_fn,
                               x_dim=lattice.x_dim,
                               hmc=args.hmc,
                               eps=eps,
                               num_steps=args.num_steps,
                               batch_size=args.batch_size,
                               model_type='GaugeModel',
                               direction=args.direction,
                               zero_masks=args.zero_masks,
                               num_singular_values=args.num_singular_values)
    if args.hmc:
        net_weights = NetWeights(0, 0, 0, 0, 0, 0)
    else:
        net_weights = NetWeights(x_scale=args.x_scale_weight,
                                 x_translation=args.x_translation_weight,
                                 x_transformation=args.x_transformation_weight,
                                 v_scale=args.v_scale_weight,
                                 v_translation=args.v_translation_weight,
                                 v_transformation=args.v_transformation_weight)
    run_params = {
        'beta': args.beta,
        'trained_eps': dynamics.eps,
        'args_eps': args.eps,
        'eps': eps,
        'net_weights': net_weights,
        'run_steps': args.run_steps,
        'num_steps': dynamics.num_steps,
        'batch_size': lattice.batch_size,
        'direction': args.direction,
        'mix_samplers': args.mix_samplers,
        'zero_masks': args.zero_masks,
        'num_singular_values': args.num_singular_values,
    }

    for key, val in args.__dict__.items():
        if key not in run_params:
            run_params[key] = val
    outputs = run_inference_np(log_dir, dynamics, lattice,
                               run_params, init=args.init, skip=False,
                               print_steps=args.print_steps,
                               mix_samplers=args.mix_samplers)
    run_data = outputs['data']['run_data']
    energy_data = outputs['data']['energy_data']
    run_params = outputs['run_params']

    make_csv(run_data, energy_data, run_params)

    run_params = outputs['run_params']
    dataset, energy_dataset = inference_plots(outputs['data'], params,
                                              outputs['run_params'],
                                              runs_np=True)

    summarizer = InferenceSummarizer(run_params['run_dir'])
    therm_data, tunn_stats = summarizer.log_summary(n_boot=10000)

    return run_params, outputs['data'], dataset


if __name__ == '__main__':
    FLAGS = parse_inference_args()
    io.log(SEPERATOR)
    io.log('FLAGS: ')
    for key, val in FLAGS.__dict__.items():
        io.log(f'  - {key}: {val}\n')

    io.log(SEPERATOR)

    _ = main(FLAGS)
