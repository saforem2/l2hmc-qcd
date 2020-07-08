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
    io.log(80 * '-')
    io.log('Running inference with:')
    io.log('\n'.join([f'  {key}: {val}' for key, val in args.items()]))
    skip_existing = args.get('overwrite', False)
    if args.get('log_dir', None) is not None:
        _, _ = load_and_run(args)
        hmc_dir = os.path.join(args.log_dir, 'inference_hmc')
        _, _ = run_hmc(args=args, hmc_dir=hmc_dir, skip_existing=skip_existing)
    else:
        #  log_file = os.path.join(os.getcwd(), 'hmc_dirs.txt')
        _, _ = run_hmc(args=args, hmc_dir=None, skip_existing=skip_existing)
    io.log(80 * '-')



if __name__ == '__main__':
    ARGS = parse_args()
    ARGS = AttrDict(ARGS.__dict__)
    main(ARGS)
