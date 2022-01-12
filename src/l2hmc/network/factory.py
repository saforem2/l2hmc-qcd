"""
network/factory.py

Contains implementation of NetworkFactory, an Abstract
Base Class for building networks.
"""
from __future__ import absolute_import, division, print_function, annotations
from abc import ABC, abstractmethod
from dataclasses import asdict

from typing import Optional

from src.l2hmc.configs import (InputSpec, NetworkConfig,
                               NetWeights, NetWeight,)


class BaseNetworkFactory(ABC):
    def __init__(
            self,
            input_spec: InputSpec,
            network_config: NetworkConfig,
            net_weights: Optional[NetWeights] = None,
    ):
        if net_weights is None:
            net_weights = NetWeights(x=NetWeight(1., 1., 1.),  # (s, t, q)
                                     v=NetWeight(1., 1., 1.))

        self.nw = net_weights
        self.input_spec = input_spec
        self.network_config = network_config
        self.config = {
            'net_weights': asdict(self.nw),
            'input_spec': asdict(self.input_spec),
            'network_config': asdict(self.network_config),
        }

    def get_build_configs(self):
        return {
            'xnet': {
                'net_weight': self.nw.x,
                'xshape': self.input_spec.xshape,
                'input_shapes': self.input_spec.xnet,
                'network_config': self.network_config,
            },
            'vnet': {
                'net_weight': self.nw.v,
                'xshape': self.input_spec.xshape,
                'input_shapes': self.input_spec.vnet,
                'network_config': self.network_config,
            }
        }

    @abstractmethod
    def build_networks(self, nleapfrog: int = 0):
        """Build Networks."""
        pass
