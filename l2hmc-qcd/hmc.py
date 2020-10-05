"""
hmc.py

Runs generic HMC by loading in from `config.BIN_DIR/hmc_configs.json`
"""
from __future__ import absolute_import, division, print_function
import argparse
import os
import json
from utils.inference_utils import run_hmc
from utils.attr_dict import AttrDict

import utils.file_io as io


def parse_args():
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(
        description='Run generic HMC.'
    )

    parser.add_argument('--run_steps', dest='run_steps',
                        type=int, default=None, required=False,
                        help='Number of sampling steps.')

    parser.add_argument('--beta', dest='beta', type=float, default=None,
                        required=False, help='Inverse coupling constant.')

    parser.add_argument('--eps', dest='eps', type=float, default=None,
                        required=False, help='Step size')

    parser.add_argument('--num_steps', dest='num_steps',
                        type=int, default=None, required=False,
                        help='Number of LeapFrog steps per trajectory.')

    parser.add_argument('--run_loop', dest='run_loop',
                        action='store_true', default=False, required=False,
                        help='Run HMC over loop of parameters.')

    return parser.parse_args()


def multiple_runs():
    num_steps = 10
    run_steps = 5000
    betas = [2., 3., 4., 5., 6.]
    eps = [0.1, 0.125, 0.15, 0.175, 0.2]
    for b in betas:
        for e in eps:
            args = AttrDict({
                'eps': e,
                'beta': b,
                'num_steps': num_steps,
                'run_steps': run_steps
            })
            _ = main(args)


def main(args):
    """Main method for running HMC."""
    cfg_file = os.path.relpath(os.path.join('..', 'bin', 'hmc_configs.json'))
    with open(cfg_file, 'rt') as f:
        configs = json.load(f)

    configs = AttrDict(configs)

    if args.beta is not None:
        configs.beta_final = configs.beta_init = args.beta

    if args.run_steps is not None:
        configs.run_steps = args.run_steps

    if args.eps is not None:
        configs.dynamics_config['eps'] = args.eps

    if args.num_steps is not None:
        configs.dynamics_config['num_steps'] = args.num_steps

    log_dir = io.make_log_dir(configs)
    configs.log_dir = log_dir

    return run_hmc(configs, hmc_dir=log_dir)


if __name__ == '__main__':
    FLAGS = parse_args()
    FLAGS = AttrDict(FLAGS.__dict__)  # pylint:disable=protected-access
    if FLAGS.run_loop:
        multiple_runs()
    else:
        _ = main(FLAGS)
