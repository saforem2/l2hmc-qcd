"""
test_dynamics.py
"""
import os
import json

from typing import Callable, Union
from pathlib import Path

import numpy as np
import tensorflow as tf

import utils.file_io as io

from config import BIN_DIR
from network import LearningRateConfig, NetworkConfig
from dynamics import DynamicsConfig
from dynamics.generic_dynamics import GenericDynamics
from utils.attr_dict import AttrDict

TEST_CONFIGS_FILE = os.path.join(BIN_DIR, 'test_configs.json')


def identity(x):
    return x


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


class TestGenericDynamics(GenericDynamics):

    # pylint:disable=too-many-arguments
    def __init__(
            self,
            params: AttrDict,
            config: DynamicsConfig,
            network_config: NetworkConfig,
            lr_config: LearningRateConfig,
            potential_fn: Callable,
            normalizer: Callable = identity,
            name: str = 'GenericDynamics'
    ):
        super(TestGenericDynamics, self).__init__(
            name=name,
            params=params,
            config=config,
            lr_config=lr_config,
            normalizer=normalizer,
            potential_fn=potential_fn,
            network_config=network_config,
        )
