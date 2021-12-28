"""
network.py

Contains the pytorch implementation of the Normalizing Flow network

used to train the L2HMC model.
"""
from __future__ import absolute_import, division, print_function, annotations
from collections import namedtuple
import os
import numpy as np
import sys

from src.l2hmc.configs import NetworkConfig

import torch

from typing import Callable, Union, NamedTuple
from dataclasses import dataclass, field
from collections import namedtuple



from torch import nn
import torch.nn.functional as F

DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu')
)

ACTIVATION_FNS = {
    'elu': F.elu,
    'tanh': F.tanh,
    'relu': F.relu,
    'swish': F.silu,
    'leaky_relu': F.leaky_relu,
}

class NetworkInputs(NamedTuple):
    x: torch.Tensor
    v: torch.Tensor

class NetworkOutputs(NamedTuple):
    s: torch.Tensor
    t: torch.Tensor
    q: torch.Tensor


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def xy_repr(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.cos(x), torch.sin(x)), dim=1)


def init_weights(m: nn.Module, use_zeros: bool = False):
    if isinstance(m, nn.Linear):
        if use_zeros:
            torch.nn.init.zeros_(m.weight)
        else:
            torch.nn.init.kaiming_normal_(m.weight)


class Network(nn.Module):
    def __init__(
            self,
            xshape: tuple[int],
            network_config: NetworkConfig,
            input_shapes: dict[str, int] = None,
    ):
        super().__init__()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.xshape = xshape
        self.net_config = network_config
        self.xdim = np.cumprod(xshape[1:])[-1]

        if input_shapes is None:
            input_shapes = {'x': self.xdim, 'v': self.xdim}

        self.input_shapes = input_shapes
        self.units = self.net_config.units

        self.s_coeff = nn.Parameter(torch.zeros(1, self.xdim))
        self.q_coeff = nn.Parameter(torch.zeros(1, self.xdim))

        self.x_layer = nn.Linear(self.input_shapes['x'], self.units[0])
        self.v_layer = nn.Linear(self.input_shapes['v'], self.units[0])

        act_fn = self.net_config.activation_fn
        if isinstance(act_fn, str):
            act_fn = ACTIVATION_FNS.get(act_fn, None)

        assert isinstance(act_fn, Callable)
        self.activation_fn = act_fn

        self.hidden_layers = nn.ModuleList()
        for idx, units in enumerate(self.units[1:]):
            h = nn.Linear(self.units[idx], units)
            self.hidden_layers.append(h)

        self.scale = nn.Linear(self.units[-1], self.xdim)
        self.transl = nn.Linear(self.units[-1], self.xdim)
        self.transf = nn.Linear(self.units[-1], self.xdim)

        if self.net_config.dropout_prob > 0:
            self.dropout = nn.Dropout(self.net_config.dropout_prob)

        if self.net_config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.units[-1])


    def forward(self, inputs: NetworkInputs) -> NetworkOutputs:
        x, v = inputs
        v = self.v_layer(v)
        x = self.x_layer(flatten(x))

        z = self.activation_fn(x + v)
        for layer in self.hidden_layers:
            z = self.activation_fn(layer(z))

        if self.net_config.dropout_prob > 0:
            z = self.dropout(z)

        if self.net_config.use_batch_norm:
            z = self.batch_norm(z)

        scale = torch.exp(self.s_coeff) * torch.tanh(self.scale(z))
        transl = self.transl(z)
        transf = torch.exp(self.q_coeff) * torch.tanh(self.transf(z))

        return NetworkOutputs(scale, transl, transf)
