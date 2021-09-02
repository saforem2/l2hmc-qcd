"""
test.py

Test training on 2D U(1) model.
"""
from __future__ import absolute_import, annotations, division, print_function

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Union, Any
from pathlib import Path


warnings.filterwarnings(action='once', category=UserWarning)
warnings.filterwarnings('once', 'keras')

MODULEPATH = os.path.join(os.path.dirname(__file__), '..')
if MODULEPATH not in sys.path:
    sys.path.append(MODULEPATH)


from config import BIN_DIR
from utils.hvd_init import RANK
from utils.inference_utils import InferenceResults, run
from utils.logger import Logger
from utils.training_utils import TrainOutputs, train

logger = Logger()


# pylint:disable=import-outside-toplevel, invalid-name, broad-except
TIMING_FILE = os.path.join(BIN_DIR, 'test_benchmarks.log')
LOG_FILE = os.path.join(BIN_DIR, 'log_dirs.txt')


@dataclass
class TestOutputs:
    train: TrainOutputs
    run: Union[InferenceResults, None]


def parse_args():
    """Method for parsing CLI flags."""
    description = (
        "Various test functions to make sure everything runs as expected."
    )

    parser = argparse.ArgumentParser(
        description=description,
    )
    parser.add_argument('--configs_file', default=None, type=str,
                        help=("""Path to configs file to use for testing."""))

    args = parser.parse_args()

    return args



def get_configs(fpath: os.PathLike = None) -> dict[str, Any]:
    """Get fresh copy of `bin/test_configs.json` for running tests."""
    if fpath is None:
        fpath = Path(BIN_DIR).joinpath('test_configs.json')
        logger.warning(f'Loading default configs from: {str(fpath)}')

    with open(fpath, 'r') as f:
        configs = json.load(f)

    return configs


def main(configs: dict[str, Any], **kwargs):
    t0 = time.time()
    train_out = train(configs, **kwargs)
    run_out = None
    #  if RANK == 0:
    #      run_out = run(train_out.dynamics, configs, make_plots=True,
    #                    runs_dir=os.path.join(train_out.logdir, 'inference'))

    logger.info(f'Passed! Took: {time.time() - t0:.4f} seconds')
    return TestOutputs(train_out, run_out)


if __name__ == '__main__':
    ARGS = parse_args()
    configs = get_configs(ARGS.configs_file)
    _ = main(configs)
