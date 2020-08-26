"""
run.py

Run inference on trained model.
"""
# pylint:disable=wrong-import-position
# noqa:disable-all
from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
import horovod.tensorflow as hvd

hvd.init()
RANK = hvd.rank()
IS_CHIEF = (RANK == 0)

# noqa: E402
import utils.file_io as io

from config import TF_FLOAT
from utils.attr_dict import AttrDict
from utils.inference_utils import load_and_run, run_hmc
from utils.parse_inference_args import parse_args


def main(args, random_start=True):
    """Run inference on trained model from `log_dir/checkpoints/`."""
    if not IS_CHIEF:
        return

    io.print_flags(args)
    skip = not args.get('overwrite', False)

    # If no `log_dir` specified, run generic HMC
    log_dir = args.get('log_dir', None)
    if log_dir is None:
        io.log('`log_dir` not specified, running generic HMC...')
        _ = run_hmc(args=args, hmc_dir=None, skip_existing=skip)
        return

    # Otherwise, load training flags
    train_flags_file = os.path.join(log_dir, 'training', 'FLAGS.z')
    train_flags = io.loadz(train_flags_file)

    beta = args.get('beta', None)
    eps = args.get('eps', None)

    if beta is None:
        io.log('Using `beta_final` from training flags')
        beta = train_flags['beta_final']
    if eps is None:
        eps_file = os.path.join(log_dir, 'training', 'train_data', 'eps.z')
        io.log(f'Loading `eps` from {eps_file}')
        eps_arr = io.loadz(eps_file)
        eps = tf.cast(eps_arr[-1], TF_FLOAT)

    # Update `args` with values from training flags
    args.update({
        'eps': eps,
        'beta': beta,
        'num_steps': int(train_flags['num_steps']),
        'lattice_shape': train_flags['lattice_shape'],
    })

    # Run generic HMC using trained step-size (by loading it from
    _ = run_hmc(args=args, hmc_dir=None, skip_existing=skip)

    # `x` will be randomly initialized if passed as `None`
    x = None
    if not random_start:
        # Load the last configuration from the end of training run
        x_file = os.path.join(args.log_dir, 'training',
                              'train_data', 'x_rank0.z')
        x = io.loadz(x_file) if os.path.isfile(x_file) else None

    # Run inference on trained model from `args.log_dir`
    args['hmc'] = False  # Ensure we're running L2HMC
    _ = load_and_run(args, x=x)

    return


if __name__ == '__main__':
    ARGS = parse_args()
    ARGS = AttrDict(ARGS.__dict__)
    if IS_CHIEF:
        main(ARGS, random_start=True)
