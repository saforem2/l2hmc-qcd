"""
config.py

Implements various configuration objects
"""
from __future__ import absolute_import, annotations, division, print_function
import abc
from collections import namedtuple
from dataclasses import dataclass, asdict, field
import json
import os
from pathlib import Path
from typing import Callable

# from pydantic import BaseModel
import tensorflow as tf


SRC = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = SRC.parent
LOGS_DIR = PROJECT_DIR.joinpath('logs')


ACTIVATIONS = {
    'relu': tf.nn.relu,
    'tanh': tf.nn.tanh,
    'leaky_relu': tf.nn.leaky_relu,
}

State = namedtuple('State', ['x', 'v', 'beta'])

MonteCarloStates = namedtuple('MonteCarloStates', ['init', 'proposed', 'out'])


@dataclass
class BaseConfig:
    def to_json(self):
        return json.dumps(self.__dict__)

    def to_file(self, fpath: os.PathLike):
        with open(fpath, 'w') as f:
            json.dump(self.to_json(), f, indent=4)


class NetWeight(BaseConfig):
    """Object for selectively scaling different components of learned fns.

    Explicitly,
     - s: scales the v (x) scaling function in the v (x) updates
     - t: scales the translation function in the update
     - q: scales the force (v) transformation function in the v (x) updates
    """
    s: float = 1.
    t: float = 1.
    q: float = 1.


class NetWeights(BaseConfig):
    """Separate NetWeight objects for scaling the `x` and `v` networks."""
    x: NetWeight = field(default_factory=NetWeight)
    v: NetWeight = field(default_factory=NetWeight)


@dataclass
class LearningRateConfig(BaseConfig):
    """Learning rate configuration object."""
    lr_init: float
    warmup_steps: int = 0
    decay_steps: int = -1
    decay_rate: float = 1.0


@dataclass
class NetworkConfig(BaseConfig):
    units: list[int]
    activation_fn: Callable
    dropout_prob: float
    use_batch_norm: bool = True


class ConvolutionConfig(BaseConfig):
    input_shape: list[int]
    filters: list[int]
    sizes: list[int]
    pool: list[int]
    activation: Callable
    paddings: list[int]
    use_batch_norm: bool = False


class DynamicsConfig(BaseConfig):
    xdim: int
    num_steps: int
    hmc: bool = False
    eps: float = 0.01
    eps_fixed: bool = False
    separate_networks: bool = False
