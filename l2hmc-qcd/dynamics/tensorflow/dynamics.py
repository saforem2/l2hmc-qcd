"""
dynamics.py

Tensorflow implementation of the dynamics engine for the L2HMC sampler
"""
from __future__ import absolute_import, print_function, division, annotations
from collections import namedtuple
from pathlib import Path
from typing import Union
from math import pi
import json
from dataclasses import dataclass, asdict, field
import numpy as np
import tensorflow as tf
from ...lattice.tensorflow.lattice import Lattice
from ...network.config import (
    NetworkConfig, ConvolutionConfig, LearningRateConfig
)

ListLike = Union[tuple, list]
PathLike = Union[str, Path]

PI = tf.constant(pi)
TWO_PI = 2. * PI


def rand_unif(shape: ListLike, a: float, b: float) -> tf.Tensor:
    return tf.random.uniform(shape, *(a, b))


def random_angle_unif(shape: ListLike):
    return TWO_PI * tf.random.uniform(shape) - pi


@dataclass
class BaseConfig:
    def to_json(self):
        return json.dumps(asdict(self))

    def to_file(self, fpath: PathLike):
        with open(str(fpath), 'w') as f:
            json.dump(asdict(self), f)



fnWeights = namedtuple('fnWeights', ['s', 't', 'q'])


class NetWeights:
    def __init__(self, x: fnWeights = None, v: fnWeights = None):
        self.x = x if x is not None else fnWeights(1., 1., 1.)
        self.v  = v if v is not None else fnWeights(1., 1., 1.)


@dataclass
class LossWeights(BaseConfig):
    aux: float = 0.
    plaq: float = 0.
    charge: float = 0.01
    use_mixed: bool = False


@dataclass
class DynamicsConfig(BaseConfig):
    lf: int
    L: int
    eps: float = 0.1
    hmc: bool = False
    eps_fixed: bool = False
    clip_val: float = 0
    net_weights: NetWeights = field(default_factory=NetWeights)


def to_u1(x: tf.Tensor) -> tf.Tensor:
    return ((x + PI) % TWO_PI) - PI

def project_angle(x: tf.Tensor) -> tf.Tensor:
    return x - TWO_PI * tf.floor((x + PI) / TWO_PI)


class GaugeDynamics(tf.keras.models.Model):
    def __init__(
        self,
        dynamics_config: DynamicsConfig,
        network_config: NetworkConfig,
        conv_config: ConvolutionConfig = None,
    ):
        super().__init__()
        self.cfg = dynamics_config
        self.net_cfg = network_config
        self.conv_cfg = conv_config

        self.nw = self.cfg.net_weights
        self.x_shape = (-1, self.cfg.L, self.cfg.L, 2)
        self.lattice = Lattice(self.x_shape)
        self.potential_fn = self.lattice.action
        self.xdim = self.cfg.L * self.cfg.L * 2
        self.masks = self._build_masks()

        if self.config.hmc:
            self.nw = NetWeights((0., 0., 0.), (0., 0., 0.))
            self._use_ncp = False
            self.xnet, self.vnet = self._build_hmc_networks()
        else:
            self._use_ncp = (self.nw.x.s != 0)
            self.xnet, self.vnet = self._build_networks(
                net_config=self.net_cfg,
                conv_config=self.conv_cfg,
            )




