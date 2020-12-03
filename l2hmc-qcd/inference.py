"""
inference.py

Run inference by loading network/trained model from `log_dir`.
"""
import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import utils.file_io as io
from utils.attr_dict import AttrDict
from dataclasses import dataclass
from collections import namedtuple


from utils.inference_utils import run_inference_from_log_dir, InferenceResults
sns.set_palette('bright')


def parse_args():
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(
        description='Run inference from `log_dir`.'
    )
    parser.add_argument('--log_dir', dest='log_dir',
                        type=str, default=None, required=True,
                        help=('Path to `log_dir` containing either '
                              'network weights or checkpoint object.'))

    parser.add_argument('--run_steps', dest='run_steps',
                        type=int, default=5000, required=False,
                        help='Number of inference steps.')

    parser.add_argument('--beta', dest='beta',
                        type=float, default=None, required=False,
                        help='Value of beta at which to run inference.')

    parser.add_argument('--therm_frac', dest='therm_frac',
                        type=float, default=0.33, required=False,
                        help=('Percentage of total chain length to drop '
                              '(from beginning, to account for burn-in).'))

    parser.add_argument('--num_chains', dest='num_chains',
                        type=int, default=None, required=False,
                        help='Number of chains to include when plotting.')

    parser.add_argument('--no_plots', action='store_true', default=False,
                        required=False, help='If passed, do not make plots.')

    parser.add_argument('--train_steps', dest='train_steps',
                        type=int, default=10, required=False,
                        help=('Number of training steps to perform '
                              'prior to running inference. (Useful '
                              'for getting final `eps` if unknown.)'))

    return parser.parse_args()



def inference(
        log_dir,
        run_steps: int = None,
        beta: float = None,
        make_plots: bool = True,
        train_steps: int = 100,
        therm_frac: float = 0.33,
        num_chains: int = None,
):
    """Run inference on model stored in `log_dir`."""
    inference_results = run_inference_from_log_dir(log_dir=log_dir,
                                                   run_steps=run_steps,
                                                   beta=beta,
                                                   therm_frac=therm_frac,
                                                   num_chains=num_chains,
                                                   make_plots=make_plots,
                                                   train_steps=train_steps)
    return inference_results


if __name__ == '__main__':
    args = parse_args()
    args = AttrDict(args.__dict__)
    inference(log_dir=args.log_dir,
              run_steps=args.run_steps,
              beta=args.beta,
              num_chains=args.num_chains,
              therm_frac=args.therm_frac,
              make_plots=(not args.no_plots),
              train_steps=args.train_steps)
