"""
hmc.py

Runs generic HMC by loading in from `config.BIN_DIR/hmc_configs.json`
"""
from __future__ import absolute_import, division, print_function
import argparse
import os
import json
import logging
from tqdm.auto import trange, tqdm
import tensorflow as tf
import utils
from config import CBARS

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

from utils.inference_utils import run_hmc
from utils.attr_dict import AttrDict
from config import GAUGE_LOGS_DIR, BIN_DIR


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
    lattice_shapes = [
        (128, 4, 4, 2),
        #  (128, 16, 16, 2),
    ]
    num_steps = [10, 15, 20]
    run_steps = 5000
    betas = [2., 3., 4., 5., 6.]
    eps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    #  eps = [0.1, 0.125, 0.15, 0.175, 0.2]

    # =====
    # NOTE: Color tuples for tqdm formatting follow the pattern:
    # (left_text, bar, right_text, reset)
    lstup = (CBARS['reset'], CBARS['red'], CBARS['reset'], CBARS['reset'])
    nstup = (CBARS['reset'], CBARS['blue'], CBARS['reset'], CBARS['reset'])
    btup = (CBARS['reset'], CBARS['magenta'], CBARS['reset'], CBARS['reset'])
    etup = (CBARS['reset'], CBARS['cyan'], CBARS['reset'], CBARS['reset'])
    eps = tqdm(eps, desc='eps', unit='step',
               bar_format=("%s{l_bar}%s{bar}%s{r_bar}%s" % etup))
    betas = tqdm(betas, desc='betas', unit='step',
                 bar_format=("%s{l_bar}%s{bar}%s{r_bar}%s" % btup))
    num_steps = tqdm(num_steps, desc='num_steps', unit='step',
                     bar_format=("%s{l_bar}%s{bar}%s{r_bar}%s" % nstup))
    lattice_shapes = tqdm(lattice_shapes, desc='lattice_shapes', unit='step',
                          bar_format=("%s{l_bar}%s{bar}%s{r_bar}%s" % lstup))
    # pylint:disable=invalid-name
    for ls in lattice_shapes:
        for ns in num_steps:
            for b in betas:
                for e in eps:
                    args = AttrDict({
                        'eps': e,
                        'beta': b,
                        'num_steps': ns,
                        'run_steps': run_steps,
                        'lattice_shape': ls,
                    })
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

    if args.get('lattice_shape', None) is not None:
        flags['dynamics_config']['lattice_shape'] = args.lattice_shape
        #  flags.dynamics_config['lattice_shape'] = args.lattice_shape

    if args.get('beta', None) is not None:
        flags.beta = args.beta
        flags.beta_init = args.beta
        flags.beta_final = args.beta

    if args.get('run_steps', None) is not None:
        flags.run_steps = args.run_steps

    if args.get('eps', None) is not None:
        flags.dynamics_config['eps'] = args.eps

    if args.get('num-steps', None) is not None:
        flags.dynamics_config['num_steps'] = args.num_steps

    return run_hmc(flags)


if __name__ == '__main__':
    FLAGS = parse_args()
    FLAGS = AttrDict(FLAGS.__dict__)  # pylint:disable=protected-access
    if FLAGS.run_loop:
        multiple_runs(FLAGS.json_file)
    else:
        _ = main(FLAGS, FLAGS.json_file)
