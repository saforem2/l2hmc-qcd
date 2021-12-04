"""
dynamics.py

Tensorflow implementation of the dynamics engine for the L2HMC sampler
"""
from __future__ import absolute_import, print_function, division, annotations
from pathlib import Path
from typing import Union
from math import pi
import json
from dataclasses import dataclass, asdict
import numpy as np
import tensorflow as tf

ListLike = Union[tuple, list]
PathLike = Union[str, Path]

PI = tf.constant(pi)
TWO_PI = 2. * PI


def rand_unif(shape: ListLike, a: float, b: float) -> tf.Tensor:
    return tf.random.uniform(shape, *(a, b))


def random_angle_unif(shape: ListLike):
    return TWO_PI * tf.random.uniform(shape) - pi


@dataclass
class Config:
    def to_json(self):
        return json.dumps(asdict(self))

    def to_file(self, fpath: PathLike):
        with open(str(fpath), 'w') as f:
            json.dump(asdict(self), f)


@dataclass
class fnWeights(Config):
    s: float = 1.
    t: float = 1.
    q: float = 1.


@dataclass
class NetWeights(Config):
    x: fnWeights = fnWeights()
    v: fnWeights = fnWeights()


@dataclass
class LossWeights(Config):
    aux: float = 0.
    plaq: float = 0.
    charge: float = 0.01
    use_mixed: bool = False


@dataclass
class DynamicsConfig(Config):
    lf: int
    xdim: int
    eps: float = 0.1
    eps_fixed: bool = False
    clip_val: float = 0


def to_u1(x: tf.Tensor) -> tf.Tensor:
    return ((x + PI) % TWO_PI) - PI

def project_angle(x: tf.Tensor) -> tf.Tensor:
    return x - TWO_PI * tf.floor((x + PI) / TWO_PI)


class GaugeDynamics(tf.keras.Model):
    def __init__(
        self,
        dynamics_config: DynamicsConfig,
        network_config: NetworkConfig,
    ):
        super().__init__()
        self.cfg = dynamics_config
        self.net_cfg = network_config
