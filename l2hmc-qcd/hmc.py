"""
hmc.py

Runs generic HMC by loading in from `config.BIN_DIR/hmc_configs.json`
"""
from __future__ import absolute_import, division, print_function

import os
import json
import logging
import argparse

import utils
import random
import tensorflow as tf

try:
    import horovod
    import horovod.tensorflow as hvd

    try:
        RANK = hvd.rank()
    except ValueError:
        hvd.init()

    RANK = hvd.rank()
    HAS_HOROVOD = True
    logging.info(f'using horovod version: {horovod.__version__}')
    logging.info(f'using horovod from: {horovod.__file__}')
    GPUS = tf.config.experimental.list_physical_devices('GPU')
    for gpu in GPUS:
        tf.config.experimental.set_memory_growth(gpu, True)
    if GPUS:
        gpu = GPUS[hvd.local_rank()]
        tf.config.experimental.set_visible_devices(gpu, 'GPU')

except (ImportError, ModuleNotFoundError):
    HAS_HOROVOD = False

import utils.file_io as io

from config import BIN_DIR, GAUGE_LOGS_DIR
from utils.attr_dict import AttrDict
from utils.inference_utils import run_hmc


def parse_args():
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(
        description='Run generic HMC.'
    )

    parser.add_argument('--json_file', dest='json_file',
                        type=str, default=None, required=False,
                        help='json file containing HMC config')

    parser.add_argument('--x_shape', dest='x_shape',
                        type=lambda s: [int(i) for i in s.split(',')],
                        default=None, required=False,
                        help='Specify shape of lattice (batch, Lt, Lx, 2)')

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

    parser.add_argument('--overwrite', dest='overwrite',
                        action='overwrite', default=False, required=False,
                        help=('If an existing run with identical config is '
                              'found, should it be overwritten?'))

    return parser.parse_args()


def check_existing(beta, num_steps, eps):
    from config import HMC_LOGS_DIR

    root_dir = os.path.abspath(os.path.join(HMC_LOGS_DIR, '2021_01'))
    runs = os.listdir(root_dir)
    dirname = f'HMC_L16_b512_beta{beta}_lf{num_steps}_eps{eps}'
    match = False
    for run in runs:
        if run.startswith(dirname):
            match = True

    return match


def multiple_runs(flags, json_file=None):
    default = (512, 16, 16, 2)
    #  run_steps = flags.run_steps if flags.run_steps is not None else 125000
    shape = flags.x_shape if flags.x_shape is not None else default

    num_steps = [5, 10]
    eps = [0.05, 0.1, 0.2]
    betas = [2., 3., 4., 5., 6., 7.]
    #  run_steps = [50000, 50000, 50000, 100000, 100000, 100000]
    #  betas = [5.0, 6.0, 7.0]
    #num_steps = [10, 15, 20, 25]
    #  eps = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    #  eps = [0.1, 0.125, 0.15, 0.175, 0.2]

    for b in random.sample(betas, len(betas)):
        for ns in random.sample(num_steps, len(num_steps)):
            for e in random.sample(eps, len(eps)):
                run_steps = 50000 if b < 5. else 100000
                args = AttrDict({
                    'eps': e,
                    'beta': b,
                    'num_steps': ns,
                    'run_steps': run_steps,
                    'x_shape': shape,
                })
                if not flags.overwrite:
                    exists = check_existing(b, ns, e)
                    if exists:
                        io.rule('Skipping existing run!')
                        continue

                _ = main(args, json_file=json_file)


def load_hmc_flags(json_file=None):
    """Load HMC flags from `BIN_DIR/hmc_configs.json`."""
    if json_file is None:
        json_file = os.path.join(BIN_DIR, 'hmc_configs.json')

    with open(json_file, 'rt') as f:
        flags = json.load(f)

    return AttrDict(flags)


# pylint:disable=no-member
def main(args, json_file=None):
    """Main method for running HMC."""
    flags = load_hmc_flags(json_file)

    if args.get('x_shape', None) is not None:
        flags['dynamics_config']['x_shape'] = args.x_shape

    if args.get('beta', None) is not None:
        flags.beta = args.beta
        flags.beta_init = args.beta
        flags.beta_final = args.beta

    if args.get('run_steps', None) is not None:
        flags.run_steps = args.run_steps

    if args.get('eps', None) is not None:
        flags.dynamics_config['eps'] = args.eps

    if args.get('num_steps', None) is not None:
        flags.dynamics_config['num_steps'] = args.num_steps

    #  return run_hmc(flags, skip_existing=True, num_chains=4, make_plots=True)
    #  skip_existing = os.environ.get('SKIP_EXISTING', True)
    return run_hmc(flags, skip_existing=False, num_chains=4, make_plots=True)


if __name__ == '__main__':
    FLAGS = parse_args()
    FLAGS = AttrDict(FLAGS.__dict__)  # pylint:disable=protected-access
    if FLAGS.run_loop:
        multiple_runs(FLAGS, FLAGS.json_file)
    else:
        _ = main(FLAGS, FLAGS.json_file)
