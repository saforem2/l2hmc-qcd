"""
run.py

Run inference on trained model.
"""
from __future__ import absolute_import, division, print_function

from utils.attr_dict import AttrDict
from utils.parse_inference_args import parse_args
from utils.inference_utils import load_and_run


if __name__ == '__main__':
    ARGS = parse_args()
    ARGS = AttrDict(ARGS.__dict__)
    _, _ = load_and_run(ARGS)
