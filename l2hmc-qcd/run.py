"""
run.py

Run inference on trained model.
"""
from __future__ import absolute_import, division, print_function

import os

from utils.attr_dict import AttrDict
from utils.parse_inference_args import parse_args
from utils.inference_utils import load_and_run, run_hmc


if __name__ == '__main__':
    ARGS = parse_args()
    ARGS = AttrDict(ARGS.__dict__)
    LOG_DIR = ARGS.get('log_dir', None)
    if LOG_DIR:
        _, _ = load_and_run(ARGS)
    else:
        LOG_FILE = os.path.join(os.getcwd(), 'hmc_dirs.txt')
        _, _ = run_hmc(args=ARGS, hmc_dir=None, log_file=LOG_FILE)

