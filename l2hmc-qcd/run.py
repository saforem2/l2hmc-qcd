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


if __name__ == '__main__':
    ARGS = parse_args()
    ARGS = AttrDict(ARGS.__dict__)
    LOG_DIR = ARGS.get('log_dir', None)
    io.log(80 * '-')
    io.log('Running inference with:')
    io.log('\n'.join([f'  {key}: {val}' for key, val in ARGS.items()]))
    if LOG_DIR:
        _, _ = load_and_run(ARGS)
    else:
        LOG_FILE = os.path.join(os.getcwd(), 'hmc_dirs.txt')
        _, _ = run_hmc(args=ARGS, hmc_dir=None, skip_existing=True)
    io.log(80 * '-')
