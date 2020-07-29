"""
run.py

Run inference on trained model.
"""
from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import horovod.tensorflow as hvd
hvd.init()
RANK = hvd.rank()
IS_CHIEF = (RANK == 0)

import utils.file_io as io
from config import TF_FLOAT
from utils.attr_dict import AttrDict
from utils.parse_inference_args import parse_args
from utils.inference_utils import load_and_run, run_hmc


def main(args):
    """Main method for running inference."""
    io.log(80 * '=')
    io.log('Running inference with:')
    io.log('\n'.join([f'  {key}: {val}' for key, val in args.items()]))
    log_dir = args.get('log_dir', None)
    skip = not args.get('overwrite', False)

    if log_dir is None:
        _, _, _ = run_hmc(args=args, hmc_dir=None, skip_existing=skip)
        return

    train_flags_file = os.path.join(args.log_dir, 'training', 'FLAGS.z')
    train_flags = io.loadz(train_flags_file)
    if args.eps is None:
        eps_file = os.path.join(args.log_dir, 'training',
                                'train_data', 'eps.z')
        eps_arr = io.loadz(eps_file)
        args.eps = tf.cast(eps_arr[-1], TF_FLOAT)
    if args.beta is None:
        args.beta = train_flags['beta_final']
    args.update({
        'num_steps': int(train_flags['num_steps']),
        'lattice_shape': train_flags['lattice_shape'],
    })

    #  hmc_dir = os.path.join(args.log_dir, 'inference_hmc')
    #  _, _, x = run_hmc(args=args, hmc_dir=hmc_dir, skip_existing=skip)

    args.hmc = False
    #  _, _, _ = load_and_run(args, x=x)

    x_train = io.loadz(os.path.join(args.log_dir, 'training',
                                    'train_data', 'x_rank0.z'))
    _, _, _ = load_and_run(args, x=x_train)

    io.log(80 * '=')
    return


if __name__ == '__main__':
    ARGS = parse_args()
    ARGS = AttrDict(ARGS.__dict__)
    if IS_CHIEF:
        main(ARGS)
