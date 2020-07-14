"""
run.py

Run inference on trained model.
"""
from __future__ import absolute_import, division, print_function

import os

import utils.file_io as io
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
        _, _ = run_hmc(args=args, hmc_dir=None, skip_existing=skip)
        return

    hmc_dir = os.path.join(args.log_dir, 'inference_hmc')

    _, _, x_hmc = run_hmc(args=args, hmc_dir=hmc_dir, skip_existing=skip)
    _, _, _ = load_and_run(args, x=x_hmc)

    io.log(80 * '=')
    return



if __name__ == '__main__':
    ARGS = parse_args()
    ARGS = AttrDict(ARGS.__dict__)
    main(ARGS)
