"""
network.py

Contains a pytorch implementation of `network/functional_net.py`.

Author: Sam Foreman (github: @saforem2)
Date: 11/21/2020
"""
from __future__ import absolute_import, division, print_function, annotations
from collections import namedtuple
import os
import sys
import torch

from typing import Callable, Union
from dataclasses import dataclass, field
from torch import nn
import torch.nn.functional as F

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


MODULEPATH = os.path.join(os.path.dirname(__file__), '..', '..')
if MODULEPATH not in sys.path:
    sys.path.append(MODULEPATH)


ACTIVATION_FNS = {'relu': F.relu, 'swish': nn.SiLU()}


@dataclass
class LearningRateConfig:
    lr_init: float = 0.001
    decay_steps: int = field(default_factory=int)
    decay_rate: float = field(default_factory=float)
    warmup_steps: int = 0


@dataclass
class ConvolutionConfig:
    """Configuration object for convolutional block of network."""
    input_shape: tuple[int]
    filters: list
    sizes: list
    pool_sizes: list = field(default_factory=list)
    conv_activations: list = field(default_factory=list)
    conv_paddings: list = field(default_factory=list)
    use_batch_norm: bool = True
    name: str = field(default_factory=str)


@dataclass
class NetworkConfig:
    """Configuration object for model network."""
    units: Union[tuple[int], list[int]]
    # lattice_size: int
    dropout_prob: float = 0.
    activation_fn: Callable = field(default_factory=torch.nn.ReLU())
    use_batch_norm: bool = True


# @dataclass
# class NetworkOutputs:
#     s: torch.Tensor
#     t: torch.Tensor
#     q: torch.Tensor

NetworkOutputs = "tuple[torch.Tensor, torch.Tensor, torch.Tensor]"

State = namedtuple('State', ['x', 'v', 'beta'])

@dataclass
class NetworkInputs:
    x: torch.Tensor
    v: torch.Tensor


def xy_repr(x: torch.Tensor):
    return torch.stack((torch.cos(x), torch.sin(x)), dim=1)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


def init_zero_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)


from collections import namedtuple
NetworkOutputs = namedtuple('NetworkOutputs', ['s', 't', 'q'])

# pylint:disable=invalid-name
class GenericNetwork(nn.Module):
    def __init__(self, xdim: int, net_config: NetworkConfig):
        super().__init__()
        # self.xdim = net_config.lattice_size
        self.xdim = xdim

        self.net_config = net_config
        self.units = net_config.units
        self.input_shapes = {'x': self.xdim, 'v': self.xdim}
        #  self.batch_size, self.xdim = input_shape
        self.scale_coeff = torch.zeros(1, self.xdim, device=DEVICE)
        self.transf_coeff = torch.zeros(1, self.xdim, device=DEVICE)
        self.h1 = net_config.units[0]
        self.h2 = net_config.units[1]

        if net_config.dropout_prob > 0:
            self.dropout = nn.Dropout(net_config.dropout_prob)

        self.x_layer = nn.Linear(self.input_shapes['x'], self.units[0])
        self.v_layer = nn.Linear(self.input_shapes['v'], self.units[0])

        self.h_layer1 = nn.Linear(self.units[0], self.units[1])
        self.h_layer2 = nn.Linear(self.units[1], self.units[1])

        if net_config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.units[1])

        self.scale_layer = nn.Linear(self.units[1], self.xdim)
        self.translation_layer = nn.Linear(self.units[1], self.xdim)
        self.transformation_layer = nn.Linear(self.units[1], self.xdim)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except batch dim
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

    def forward(
            self,
            inputs: NetworkInputs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.x_layer(inputs.x)
        v = self.v_layer(inputs.v)
        z = (x + v).clamp(min=0)
        #  z = (x + v + t).clamp(min=0)
        z = self.h_layer1(z)
        z = self.h_layer2(z)
        if self.net_config.dropout_prob > 0:
            z = self.dropout(z)

        if self.net_config.use_batch_norm:
            z = self.batch_norm(z)

        scale = self.scale_layer(z)
        transl = self.translation_layer(z)
        transf = self.transformation_layer(z)

        scale = torch.exp(self.scale_coeff) * torch.tanh(scale)
        transf = torch.exp(self.transf_coeff) * torch.tanh(transf)

        return scale, transl, transf


class GaugeNetwork(nn.Module):
    def __init__(
            self,
            xshape: tuple,
            net_config: NetworkConfig,
            conv_config: ConvolutionConfig = None,
            input_shapes: dict[str, int] = None,
            factor: float = 1.,
            batch_size: tuple = None,
    ):
        super().__init__()
        if len(xshape) == 4:
            batch_size, T, X, d = xshape
        elif len(xshape) == 3:
            T, X, d = xshape
        else: 
            raise ValueError(f'Incorrect shape passed for xshape')

        self.xdim = T * X * d
        if input_shapes is None:
            input_shapes = {
                'x': 2 * self.xdim, 'v': self.xdim,
            }

        self.net_config = net_config
        self.units = net_config.units
        self.input_shapes = input_shapes

        self.s_coeff = nn.Parameter(torch.zeros(1, self.xdim))
        self.q_coeff = nn.Parameter(torch.zeros(1, self.xdim))

        self.x_layer = nn.Linear(self.input_shapes['x'], self.units[0])
        self.v_layer = nn.Linear(self.input_shapes['v'], self.units[0])

        self.activation_fn = net_config.activation_fn

        self.hidden_layers = nn.ModuleList()
        for idx, units in enumerate(self.units[1:]):
            h = nn.Linear(self.units[idx], units)
            #nn.init.kaiming_normal_(h.weight, mode='fan_in')
            self.hidden_layers.append(h)

        self.scale = nn.Linear(self.units[-1], self.xdim)
        self.transl = nn.Linear(self.units[-1], self.xdim)
        self.transf = nn.Linear(self.units[-1], self.xdim)

        if net_config.dropout_prob > 0:
            self.dropout = nn.Dropout(net_config.dropout_prob)

        if net_config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.units[-1])

    def forward(
            self,
            inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # stack `x` as (cos(x), sin(x))
        x, v = inputs

        # v = v.reshape(v.shape[0], -1)
        x = x.reshape(x.shape[0], -1)
        #xy = self.flatten(torch.stack([torch.cos(x), torch.sin(x)], dim=-1))
        z = self.activation_fn(self.x_layer(x) + self.v_layer(v))
        #  z = (x + v).clamp(min=0)
        for layer in self.hidden_layers:
            z = self.activation_fn(layer(z))

        # z = self.activation_fn(z)

        if self.net_config.dropout_prob > 0:
            z = self.dropout(z)

        if self.net_config.use_batch_norm:
            z = self.batch_norm(z)

        scale = torch.exp(self.s_coeff) * torch.tanh(self.scale(z))
        transl = self.transl(z)
        transf = torch.exp(self.q_coeff) * torch.tanh(self.transf(z))

        return NetworkOutputs(s=scale, t=transl, q=transf)
