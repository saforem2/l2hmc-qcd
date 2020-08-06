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


def run(args, log_dir=None, random_start=True):
    """Run inference on trained model from `log_dir/checkpoints/`."""
    io.log('\n'.join([80 * '=', 'Running inference with:']))
    io.print_flags(args)
    skip = not args.get('overwrite', False)
    if log_dir is None:
        log_dir = args.get('log_dir', None)
        if log_dir is None:
            io.log('`log_dir` not specified, running generic HMC...')
            _ = run_hmc(args=args, hmc_dir=None, skip_existing=skip)
            return

    _ = run_hmc(args=args, hmc_dir=None, skip_existing=skip)

    train_flags_file = os.path.join(log_dir, 'training', 'FLAGS.z')
    train_flags = io.loadz(train_flags_file)

    if args.beta is None:
        io.log('Using `beta_final` from training flags')
        args.beta = train_flags['beta_final']
    if args.eps is None:
        eps_file = os.path.join(log_dir, 'training', 'train_data', 'eps.z')
        io.log(fLoading `eps` from {eps_file}')
        eps_arr = io.loadz(eps_file)
        args.eps = tf.cast(eps_arr[-1], TF_FLOAT)

    args.update({
        'hmc': False,
        'num_steps': int(train_flags['num_steps']),
        'lattice_shape': train_flags['lattice_shape'],
    })

    if random_start:
        io.log('Running inference with random start...')
        _ = load_and_run(args)
    else:
        x_file = os.path.join(args.log_dir, 'training',
                              'train_data', 'x_rank0.z')
        if os.path.isfile(x_file):
            x = io.loadz(x_file)
        else:
            x = None
            io.log('\n'.join([f'Unable to load from: {x_file}',
                              'Running with random start...']))
        _ = load_and_run(args, x=x)

    #  _, _, _ = load_and_run(args, x=x)

    x_train = io.loadz(os.path.join(args.log_dir, 'training',
                                    'train_data', 'x_rank0.z'))
    _, _, _ = load_and_run(args, x=x_train)


def main(args):
    """Main method for running inference."""
    run(args, args.log_dir, random_start=True)

    '''
    io.log(80 * '=')
    io.log('Running inference with:')
    io.print_flags(args)
    #  io.log('\n'.join([f'  {key}: {val}' for key, val in args.items()]))
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
    '''
    #  hmc_dir = os.path.join(args.log_dir, 'inference_hmc')
    #  _, _, x = run_hmc(args=args, hmc_dir=hmc_dir, skip_existing=skip)

    #  args.hmc = False
    #  #  _, _, _ = load_and_run(args, x=x)
    #
    #  x_train = io.loadz(os.path.join(args.log_dir, 'training',
    #                                  'train_data', 'x_rank0.z'))
    #  _, _, _ = load_and_run(args, x=x_train)

    io.log(80 * '=')
    return


if __name__ == '__main__':
    ARGS = parse_args()
    ARGS = AttrDict(ARGS.__dict__)
    if IS_CHIEF:
        main(ARGS)
