"""
parse_configs.py

Implements a method for parsing configuration objects from JSON file.
"""
from __future__ import absolute_import, division, print_function
import argparse
import json

from tensorflow.python.ops.gen_math_ops import log1p_eager_fallback
from utils.attr_dict import AttrDict


def parse_configs():
    """Parse configs from JSON file."""
    parser = argparse.ArgumentParser(
        'L2HMC algorithm applied to a 2D U(1) lattice gauge model.'
    )
    parser.add_argument("--log_dir",
                        dest="log_dir",
                        type=str,
                        default=None,
                        required=False,
                        help=("""Log directory to use from previous run.  If
                        this argument is not passed, a new directory will be
                        created."""))
    parser.add_argument("--json_file",
                        dest="json_file",
                        type=str,
                        default=None,
                        required=True,
                        help=("""Path to JSON file containing configs."""))
    args = parser.parse_args()
    log_dir = args.log_dir
    with open(args.json_file, 'rt') as f:
        targs = argparse.Namespace()
        targs.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=targs)

    if log_dir is not None:
        args.__dict__['log_dir'] = log_dir

    flags = AttrDict(args.__dict__)
    for key, val in flags.items():
        if isinstance(val, dict):
            flags[key] = AttrDict(val)

    return args
