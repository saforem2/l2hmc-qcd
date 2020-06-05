"""
gauge_inference_np.py

Runs inference on a trained L2HMC model using numpy.

Author: Sam Foreman
Date: 01/09/2020

UPDATES:
--------
    - Version 0.1 (05/07/2020): Uses new `RunnerNP` object with a new
      `RunParams` (`namedtuple`) container which simplifies the interface
      between command line arguments and the inference process.
"""
from __future__ import absolute_import, division, print_function

import os
import shutil

import numpy as np
import matplotlib as mpl

import utils.file_io as io

from config import NET_WEIGHTS_HMC, NetWeights, PI
from runners.runner_np import RunnerNP, RunParams
from plotters.inference_plots import inference_plots
from utils.file_io import timeit
from utils.parse_inference_args_np import parse_args as parse_inference_args

mpl.rcParams['axes.formatter.limits'] = -4, 4

SEPERATOR = 80 * '-'

# pylint:disable=invalid-name


def _load_configs(src_dir):
    names = ['dynamics_config.z', 'network_config.z', 'master_config.z']
    cfg_files = [os.path.join(src_dir, name) for name in names]
    cfgs = {
        n.rstrip('.z'): io.loadz(f) for n, f in zip(names, cfg_files)
    }
    return cfgs


def load_configs(log_dir=None):
    """Load configs from `log_dir`."""
    try:
        cfgs = _load_configs(log_dir)
    except FileNotFoundError:
        cfgs = _load_configs(os.getcwd())
    finally:
        raise FileNotFoundError('Unable to locate config files.')

    return cfgs


def run_hmc(args, train_params=None):
    """Run generic HMC."""
    run_params = RunParams(
        beta=args.beta,
        eps=args.hmc_eps,
        init='rand',
        run_steps=args.hmc_steps,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        print_steps=args.print_steps,
        mix_samplers=args.mix_samplers,
        num_singular_values=args.num_singular_values,
        net_weights=NET_WEIGHTS_HMC,
        network_type='GaugeNetwork',
    )

    runner_hmc = RunnerNP(run_params, args.log_dir,
                          model_type='GaugeModel',
                          train_params=train_params)
    x = np.random.uniform(-np.pi, np.pi, size=runner_hmc.config.input_shape)
    rd_hmc = runner_hmc.inference(x=x, run_steps=args.hmc_steps)
    x_out = rd_hmc.samples_arr[-1]

    return x_out, rd_hmc


@timeit
def main(FLAGS):
    """Create `RunnerNP` object and run inference on trained model."""
    net_weights = NetWeights(
        x_scale=FLAGS.x_scale_weight,
        x_translation=FLAGS.x_translation_weight,
        x_transformation=FLAGS.x_transformation_weight,
        v_scale=FLAGS.v_scale_weight,
        v_translation=FLAGS.v_translation_weight,
        v_transformation=FLAGS.v_transformation_weight
    )

    if FLAGS.log_dir is not None:
        train_params = io.loadz(os.path.join(FLAGS.log_dir, 'params.z'))
    else:
        pfile = os.path.join(os.getcwd(), 'params.z')
        if os.path.isfile(pfile):
            train_params = io.loadz(pfile)
            FLAGS.log_dir = train_params['log_dir']
        else:
            raise FileNotFoundError('Unable to locate `params.z file.')

    if FLAGS.num_steps is None:
        FLAGS.num_steps = train_params['num_steps']

    run_params = RunParams(
        beta=FLAGS.beta,
        eps=FLAGS.eps,
        init='rand',
        run_steps=FLAGS.run_steps,
        num_steps=FLAGS.num_steps,
        batch_size=FLAGS.batch_size,
        print_steps=FLAGS.print_steps,
        mix_samplers=FLAGS.mix_samplers,
        num_singular_values=FLAGS.num_singular_values,
        net_weights=net_weights,
        network_type='GaugeNetwork',
    )

    runner = RunnerNP(run_params,
                      FLAGS.log_dir,
                      model_type='GaugeModel',
                      train_params=train_params,
                      from_trained_model=True)

    if FLAGS.hmc_start:
        x, _ = run_hmc(FLAGS, train_params)
    else:
        try:
            final_state = io.loadz(os.path.join(runner.config.log_dir,
                                                'training', 'current_state.z'))
            # Only use first `FLAGS.batch_size` chains
            x = final_state['x_out'][:FLAGS.batch_size, :]
        except FileNotFoundError:
            shape = (FLAGS.batch_size, runner.config.run_params.xdim)
            x = np.random.uniform(-PI, PI, size=shape)

    run_data = runner.inference(x=x)
    runner.save_params()

    if not FLAGS.dont_save:  # dont save is False by default (i.e. save)
        run_data.save(run_dir=runner.config.run_dir)

    _, _, fig_dir = inference_plots(run_data,
                                    train_params,
                                    runner.config,
                                    runs_np=True,
                                    num_chains=10)

    out_file = os.path.join(fig_dir, 'run_summary.txt')
    _, _ = run_data.log_summary(out_file=out_file, n_boot=10)

    #  Copy summary file to `run_dir`
    _ = shutil.copy2(out_file, runner.config.run_dir)

    return run_data


if __name__ == '__main__':
    CL_FLAGS = parse_inference_args()

    io.log(SEPERATOR)
    io.log('FLAGS: ')
    for key, val in CL_FLAGS.__dict__.items():
        io.log(f' - {key}: {val}\n')

    io.log(SEPERATOR)

    _ = main(CL_FLAGS)
