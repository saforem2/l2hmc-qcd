"""
inference_run.py

Run inference on log_dirs.
"""
from __future__ import absolute_import, division, print_function
import os
import argparse
import numpy as np
import tensorflow as tf
try:
    import horovod
    import horovod.tensorflow as hvd
    try:
        RANK = hvd.rank()
    except ValueError:
        hvd.init()

    RANK = hvd.rank()
    HAS_HOROVOD = True
    GPUS = tf.config.experimental.list_physical_devices('GPU')
    for gpu in GPUS:
        tf.config.experimental.set_memory_growth(gpu, True)
    if GPUS:
        gpu = GPUS[hvd.local_rank()]
        tf.config.experimental.set_visible_devices(gpu, 'GPU')
except ImportError:
    HAS_HOROVOD = False

from pathlib import Path
import utils.file_io as io
from utils.attr_dict import AttrDict
from utils.inference_utils import run_inference_from_log_dir, InferenceResults
import seaborn as sns
sns.set_palette('bright')


def parse_args():
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(
        description='Run inference from `log_dir`.'
    )
    parser.add_argument('--log_dir', dest='log_dir',
                        type=str, default=None, required=True,
                        help='`log_dir` containing model to run.')

    return parser.parse_args()



def run_from_dir(d):
    log_dir_file = os.path.join(d, 'log_dirs.txt')
    if os.path.isfile(log_dir_file):
        print(120 * '#' + '\n'
              + f'Running inference from: {d}'
              + 120 * '#' + '\n')
        with open(log_dir_file, 'r') as f:
            lines = f.readlines()

        log_dir = os.path.abspath(lines[-1])
        if os.path.isdir(log_dir):
            run_steps = int(1.25e5)
            train_steps = 1000
            therm_frac = 0.2
            num_chains = 8
            x_shape = (8, 16, 16, 2)
            return run_inference_from_log_dir(log_dir=log_dir,
                                              run_steps=run_steps,
                                              therm_frac=therm_frac,
                                              num_chains=num_chains,
                                              make_plots=True,
                                              train_steps=train_steps)
    return None


if __name__ == '__main__':
    FLAGS = parse_args()
    FLAGS = AttrDict(FLAGS.__dict__)
    _ = run_from_dir(FLAGS.log_dir)
