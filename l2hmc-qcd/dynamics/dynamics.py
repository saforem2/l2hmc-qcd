"""
dynamics.py

Implements Dynamics class.
"""
from __future__ import absolute_import, division, print_function, annotations
from dataclasses import dataclass, field
from typing import Callable, Dict


import tensorflow as tf


Networks = Dict[str, tf.keras.Model]


@dataclass
class DynamicsConfig:
    eps: float
    lf: int
    xshape: tuple[int]
    eps_fixed: bool = False
    aux_weight: float = 0.
    loss_scale: float = 1.
    use_mixed_loss: bool = False
    verbose: bool = False



@dataclass
class lfNetworks:
    xnet: tuple[list[tf.keras.Model]]
    vnet: list[tf.keras.Model]



class Dynamics(tf.keras.Model):
    def __init__(
            self,
            config: DynamicsConfig,
            networks: lfNetworks,
            potential_fn: Callable[[tf.Tensor], tf.Tensor],
            **kwargs
    ):
        super(Dynamics, self).__init__()
        self.xnet = networks.xnet
        self.vnet = networks.vnet
        self.config = config
        pass
