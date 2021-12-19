"""
config.py

Contains configuration objects for networks.
"""
from __future__ import absolute_import, division, print_function, annotations

import json
from os import PathLike
from typing import Callable, Union

import tensorflow as tf

from dataclasses import asdict, dataclass, field


@dataclass
class BaseConfig:
    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_file(self, fpath: PathLike):
        with open(fpath, 'w') as f:
            json.dump(self.to_json(), f, indent=4)


@dataclass
class LearningRateConfig(BaseConfig):
    """Learning rate configuration object."""
    lr_init: float
    decay_steps: int = field(default_factory=int)
    decay_rate: float = field(default_factory=float)
    warmup_steps: int = field(default_factory=int)


@dataclass
class NetworkConfig(BaseConfig):
    """Network configuration object."""
    units: list
    activation_fn: Callable
    dropout_prob: float = field(default_factory=float)
    use_batch_norm: bool = field(default_factory=bool)


@dataclass
class ConvolutionConfig(BaseConfig):
    """Convolutional network configuration."""
    input_shape: tuple
    filters: tuple
    sizes: tuple
    pool: tuple
    activation: Callable = field(default_factory=Callable)
    paddings: tuple = field(default_factory=tuple)
    use_batch_norm: bool = field(default_factory=bool)
