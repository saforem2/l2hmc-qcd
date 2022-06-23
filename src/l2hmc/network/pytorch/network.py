"""
network.py

Contains the pytorch implementation of the Normalizing Flow network

used to train the L2HMC model.
"""
from __future__ import absolute_import, annotations, division, print_function
from typing import Callable, Optional

import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d

from l2hmc.configs import (
    NetWeight,
    NetworkConfig,
    ConvolutionConfig
)

from l2hmc.network.factory import BaseNetworkFactory

log = logging.getLogger(__name__)

Tensor = torch.Tensor

ACTIVATION_FNS = {
    'elu': F.elu,
    'tanh': F.tanh,
    'relu': F.relu,
    'swish': F.silu,
    'leaky_relu': F.leaky_relu,
}


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def xy_repr(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.cos(x), torch.sin(x)), dim=1)


def init_weights_old(m: nn.Module, use_zeros: bool = False):
    if isinstance(m, nn.Linear):
        if use_zeros:
            torch.nn.init.zeros_(m.weight)
        else:
            torch.nn.init.kaiming_normal_(m.weight)


@torch.no_grad()
def init_weights(m, method='xavier_uniform'):
    if isinstance(m, nn.Linear):
        # m.bias.fill_()
        if method == 'zeros':
            nn.init.zeros_(m.weight)
        elif method == 'xavier_normal':
            nn.init.xavier_normal_(m.weight)
        elif method == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight)
        else:
            try:
                method = getattr(nn.init, method)
                if method is not None and callable(method):
                    method(m.weight)
            except NameError:
                log.warning('. '.join([
                    f'Unable to initialize weights with {method}',
                    'Falling back to default: xavier_uniform_'

                ]))


def zero_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class PeriodicPadding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) >= 3, 'Expected len(x.shape) >= 3'
        assert isinstance(x, Tensor)
        x0 = x[:, :, -self.size:, :]
        x1 = x[:, :, 0:self.size, :]
        x = torch.concat([x0, x, x1], 2)

        y0 = x[:, :, :, -self.size:]
        y1 = x[:, :, :, 0:self.size]
        x = torch.concat([y0, x, y1], 3)

        return x


class ScaledTanh(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
    ) -> None:
        super().__init__()
        self.coeff = torch.zeros(1, out_features, requires_grad=True)
        self.layer = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )
        self.tanh = nn.Tanh()
        self._with_cuda = False
        if torch.cuda.is_available():
            self._with_cuda = True
            self.coeff = self.coeff.cuda()
            self.layer = self.layer.cuda()
            self.cuda()

    def forward(self, x):
        if self._with_cuda:
            x = x.cuda()
        return torch.exp(self.coeff) * self.tanh(self.layer(x))


class Network(nn.Module):
    def __init__(
            self,
            xshape: tuple[int],
            network_config: NetworkConfig,
            input_shapes: Optional[dict[str, tuple[int, int]]] = None,
            net_weight: Optional[NetWeight] = None,
            conv_config: Optional[ConvolutionConfig] = None,
            name: Optional[str] = None,
    ):
        super().__init__()
        if net_weight is None:
            net_weight = NetWeight(1., 1., 1.)

        self.name = name if name is not None else 'network'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.xshape = xshape
        self.net_config = network_config
        self.nw = net_weight

        self.xdim = int(np.cumprod(xshape[1:])[-1])

        if input_shapes is None:
            input_shapes = {
                'x': (int(self.xdim), int(2)),
                'v': (int(self.xdim), int(2)),
            }

        self.input_shapes = {}
        for key, val in input_shapes.items():
            if isinstance(val, (list, tuple)):
                self.input_shapes[key] = np.cumprod(val)[-1]
            elif isinstance(val, int):
                self.input_shapes[key] = val
            else:
                raise ValueError(
                    'Unexpected value in input_shapes!\n'
                    f'\tinput_shapes: {input_shapes}\n'
                    f'\t  val: {val}'
                )

        act_fn = self.net_config.activation_fn
        if isinstance(act_fn, str):
            act_fn = ACTIVATION_FNS.get(act_fn, None)

        assert isinstance(act_fn, Callable)
        self.activation_fn = act_fn

        self.units = self.net_config.units

        self.s_coeff = nn.parameter.Parameter(
            torch.zeros(1, self.xdim, device=self.device)
        )
        self.q_coeff = nn.parameter.Parameter(
            torch.zeros(1, self.xdim, device=self.device)
        )

        if conv_config is not None and len(conv_config.filters) > 0:
            self.conv_config = conv_config
            if len(xshape) == 3:
                d, nt, nx = xshape[0], xshape[1], xshape[2]
            elif len(xshape) == 4:
                _, d, nt, nx = xshape[0], xshape[1], xshape[2], xshape[3]
            else:
                raise ValueError(f'Invalid value for `xshape`: {xshape}')

            self.nt = nt
            self.nx = nx
            self.d = d
            # p0 = PeriodicPadding(conv_config.sizes[0] - 1)
            conv_stack = [
                PeriodicPadding(conv_config.sizes[0] - 1),
                Conv2d(d, conv_config.filters[0], conv_config.sizes[0])
            ]
            iterable = zip(conv_config.filters[1:], conv_config.sizes[1:])
            for idx, (f, n) in enumerate(iterable):
                conv_stack.append(PeriodicPadding(n - 1))
                conv_stack.append(nn.LazyConv2d(n, f))
                # , padding=(n-1), padding_mode='circular'))
                # conv_stack.append(self.activation_fn)
                if (idx + 1) % 2 == 0:
                    conv_stack.append(nn.MaxPool2d(conv_config.pool[idx]))

            conv_stack.append(nn.Flatten(1))
            if network_config.use_batch_norm:
                conv_stack.append(nn.BatchNorm1d(-1))

            self.conv_stack = nn.ModuleList(conv_stack)

        else:
            self.conv_stack = []

        self.flatten = nn.Flatten(1)
        self.x_layer = nn.Linear(
            self.input_shapes['x'],  # input
            self.units[0],           # output
            device=self.device
        )
        self.v_layer = nn.Linear(
            self.input_shapes['v'],
            self.units[0],
            device=self.device
        )

        self.hidden_layers = nn.Sequential()
        for idx, units in enumerate(self.units[1:]):
            h = nn.Linear(self.units[idx], units)
            self.hidden_layers.append(h)

        self.scale = ScaledTanh(self.units[-1], self.xdim)
        self.transf = ScaledTanh(self.units[-1], self.xdim)
        # self.scale = nn.Linear(self.units[-1], self.xdim, device=self.device)
        self.transl = nn.Linear(self.units[-1], self.xdim)

        if self.net_config.dropout_prob > 0:
            self.dropout = nn.Dropout(self.net_config.dropout_prob)

        if self.net_config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.units[-1],
                                             device=self.device)

    def forward(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        x = inputs[0].to(self.device)
        v = inputs[1].to(self.device)

        if len(self.conv_stack) > 0:
            try:
                x = x.reshape(-1, self.d + 2, self.nt, self.nx)
            except ValueError:
                x = x.reshape(-1, self.d, self.nt, self.nx)

            for layer in self.conv_stack:
                x = self.activation_fn(layer(x))
            if self.net_config.use_batch_norm:
                x = self.batch_norm(x)

        v = self.v_layer(v.flatten(1))
        x = self.x_layer(x.flatten(1))
        # v = self.v_layer(self.flatten(v))
        z = self.activation_fn(x + v)
        for layer in self.hidden_layers:
            z = self.activation_fn(layer(z))

        if self.net_config.dropout_prob > 0:
            z = self.dropout(z)

        if self.net_config.use_batch_norm:
            z = self.batch_norm(z)

        # scale = torch.tanh(self.scale(z))
        # transl = self.transl(z)
        # transf = torch.tanh(self.transf(z))
        # s = self.nw.s * torch.exp(self.s_coeff) * scale
        # t = self.nw.t * transl
        # q = self.nw.q * torch.exp(self.q_coeff) * transf
        s = (self.nw.s * torch.exp(self.s_coeff)) * F.tanh(self.scale(z))
        t = (self.nw.t * self.transl(z))
        q = (self.nw.q * torch.exp(self.q_coeff)) * F.tanh(self.transf(z))

        return s, t, q


class NetworkFactory(BaseNetworkFactory):
    def build_networks(self, n: int, split_xnets: bool) -> nn.ModuleDict:
        """Build LeapfrogNetwork."""
        # TODO: if n == 0: build hmcNetwork (return zeros)
        assert n >= 1, 'Must build at least one network'

        cfg = self.get_build_configs()
        if n == 1:
            return nn.ModuleDict({
                'xnet': Network(**cfg['xnet']),
                'vnet': Network(**cfg['vnet']),
            })

        vnet = nn.ModuleDict({
            str(i): Network(**cfg['vnet'], name=f'vnet/lf{i}')
            for i in range(n)
        })

        if split_xnets:
            xnet = {}
            for i in range(n):
                n1 = f'xnet/lf{i}/first'
                n2 = f'xnet/lf{i}/second'
                xnet[str(i)] = nn.ModuleDict({
                    'first': Network(**cfg['xnet'], name=n1),
                    'second': Network(**cfg['xnet'], name=n2),
                })
            xnet = nn.ModuleDict(xnet)
        else:
            xnet = nn.ModuleDict({
                str(i): Network(**cfg['xnet']) for i in range(n)
            })

        return nn.ModuleDict({'xnet': xnet, 'vnet': vnet})
