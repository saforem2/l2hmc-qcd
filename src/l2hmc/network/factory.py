"""
network/factory.py

Contains implementation of NetworkFactory, an Abstract
Base Class for building networks.
"""
from __future__ import absolute_import, annotations, division, print_function
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Optional

from l2hmc.configs import (
    ConvolutionConfig,
    InputSpec,
    NetWeight,
    NetWeights,
    NetworkConfig,
)


class BaseNetworkFactory(ABC):
    def __init__(
            self,
            input_spec: InputSpec,
            network_config: NetworkConfig,
            conv_config: Optional[ConvolutionConfig] = None,
            net_weights: Optional[NetWeights] = None,
    ):
        if net_weights is None:
            net_weights = NetWeights(x=NetWeight(1., 1., 1.),  # (s, t, q)
                                     v=NetWeight(1., 1., 1.))

        self.nw = net_weights
        self.input_spec = input_spec
        self.network_config = network_config
        self.conv_config = conv_config
        self.config = {
            'net_weights': self.nw,
            'input_spec': self.input_spec,
            'network_config': self.network_config,
            # 'network_config': asdict(self.network_config),
        }
        if conv_config is not None:
            self.config.update({'conv_config': asdict(self.conv_config)})

    def get_build_configs(self):
        return {
            'xnet': {
                'net_weight': self.nw.x,
                'xshape': self.input_spec.xshape,
                'input_shapes': self.input_spec.xnet,
                'network_config': self.network_config,
                'conv_config': self.conv_config,
            },
            'vnet': {
                'net_weight': self.nw.v,
                'xshape': self.input_spec.xshape,
                'input_shapes': self.input_spec.vnet,
                'network_config': self.network_config,
            }
        }

    @abstractmethod
    def build_networks(
            self,
            n: int = 0,
            split_xnets: bool = True,
            group: str = 'U1',
    ):
        """Build Networks."""
        pass
