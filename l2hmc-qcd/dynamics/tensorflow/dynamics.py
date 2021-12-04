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


def rand_unif(shape: ListLike, a: float, b: float) -> tf.Tensor:
    return tf.random.uniform(shape, *(a, b))


def random_angle_unif(shape: ListLike):
    return 2 * pi * tf.random.uniform(shape) - pi


@dataclass
class fnWeights:
    s: float = 1.
    t: float = 1.
    q: float = 1.


@dataclass
class NetWeights:
    x: fnWeights = fnWeights()
    v: fnWeights = fnWeights()


@dataclass
class LossWeights:
    aux: float = 0.
    plaq: float = 0.
    charge: float = 0.01
    use_mixed: bool = False


@dataclass
class DynamicsConfig:
    lf: int
    xdim: int
    eps: float = 0.1
    eps_fixed: bool = False


