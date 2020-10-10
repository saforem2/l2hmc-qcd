"""
test_dynamics.py
"""
import os
import json

from typing import Union
from pathlib import Path

import numpy as np
import tensorflow as tf

import utils.file_io as io

from config import BIN_DIR
from utils.attr_dict import AttrDict

TEST_CONFIGS_FILE = os.path.join(BIN_DIR, 'test_configs.json')


def load_test_configs(json_file: Union[str, Path] = None):
    """Load test configs, if specified.

    If not specified, load from `BIN_DIR/test_configs.json`.

    Returns:
        configs (AttrDict): Configs.
    """
    if json_file is None:
        json_file = os.path.join(BIN_DIR, 'test_configs.json')

    try:
        with open(json_file, 'rt') as f:
            configs = json.load(f)
    except FileNotFoundError:
        io.log(f'Unable to load configs from: {json_file}. Exiting.')
        raise

    return configs


class TestDynamics:
    def __init__(self, configs):
        pass
