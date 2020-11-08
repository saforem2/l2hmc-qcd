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
from config import GAUGE_LOGS_DIR, BIN_DIR

import utils.file_io as io


def parse_args():
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(
        description='Run generic HMC.'
    )

    parser.add_argument('--json_file', dest='json_file',
                        type=str, default=None, required=False,
                        help='json file containing HMC config')

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


def multiple_runs(json_file=None):
    """Perform multiple runs across a range of parameters."""
    num_steps = 10
    run_steps = 10000
    betas = [2., 3., 4., 5., 6.]
    #  eps = [0.1, 0.125, 0.15, 0.175, 0.2]
    eps = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275]
    for b in betas:
        for e in eps:
            args = AttrDict({
                'eps': e,
                'beta': b,
                'num_steps': num_steps,
                'run_steps': run_steps,
                'json_file': json_file,
            })
            _ = main(args)


def load_hmc_flags(json_file=None):
    """Load HMC flags from `BIN_DIR/hmc_configs.json`."""
    if json_file is None:
        json_file = os.path.join(BIN_DIR, 'hmc_configs.json')

    with open(json_file, 'rt') as f:
        flags = json.load(f)

    return AttrDict(flags)


# pylint:disable=no-member
def main(args):
    """Main method for running HMC."""
    #  flags = load_hmc_flags()
    flags = load_hmc_flags(args.json_file)

    if args.beta is not None:
        flags.beta = args.beta
        flags.beta_init = args.beta
        flags.beta_final = args.beta

    if args.run_steps is not None:
        flags.run_steps = args.run_steps

    if args.eps is not None:
        flags.dynamics_config['eps'] = args.eps

    if args.num_steps is not None:
        flags.dynamics_config['num_steps'] = args.num_steps

    return run_hmc(flags)


if __name__ == '__main__':
    FLAGS = parse_args()
    FLAGS = AttrDict(FLAGS.__dict__)  # pylint:disable=protected-access
    if FLAGS.run_loop:
        multiple_runs(FLAGS.json_file)
    else:
        _ = main(FLAGS)
