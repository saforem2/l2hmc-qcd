"""
pytorch/network.py

Contains the pytorch implementation of the Normalizing Flow network

used to train the L2HMC model.
"""
from __future__ import absolute_import, annotations, division, print_function
from typing import Any, Callable, Optional
import logging

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from l2hmc.configs import (
    NetWeight,
    NetworkConfig,
    ConvolutionConfig
)

from l2hmc.network.factory import BaseNetworkFactory


log = logging.getLogger(__name__)

Tensor = torch.Tensor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ACTIVATION_FNS = {
    'elu': nn.ELU(),
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'swish': nn.SiLU(),
    'leaky_relu': nn.LeakyReLU(),
}


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def xy_repr(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.cos(x), torch.sin(x)), dim=1)


def init_all(model, init_func, *params, **kwargs):
    """Recursively initialize all parameters in model using init_func.

    Example:
        >>> model = Network()
        >>> init_all(model, torch.nn.init.normal_, mean=0., std=1.)
        >>> # or
        >>> init_all(model, torch.nn.init.constant_, 1.)
    """
    for p in model.parameters():
        init_func(p, *params, **kwargs)


def init_all_by_shape(model: nn.Module, init_funcs: dict[str | int, Callable]):
    """Recursively initialize parameters in model using init_funcs.

    Example:

    ```python
    >>> model = nn.Module()
    >>> init_funcs = {
        'default': lambda x: torch.nn.init.constant(x, 1.),
        1: lambda x: torch.nn.init.normal(x, mean=0., std=1.)  # e.g. bias
        2: lambda x: torch.nn.init.xavier_normal_(x, gain=1.)   # weight
        3: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.)  # conv1D
        4: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.)  # conv2D
    }
    >>> init_all(model, init_funcs)
    ```
    """
    assert 'default' in init_funcs, 'init_funcs must have `default` entry'
    for p in model.parameters():
        if hasattr(p, 'shape'):
            init_func = init_funcs.get(
                str(len(p.shape)),
                init_funcs['default']
            )
            init_func(p)


@torch.no_grad()
def init_weights(m, method='xavier_uniform'):
    if isinstance(m, nn.Linear):
        # m.bias.fill_()
        if method == 'zeros':
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
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
        super(ScaledTanh, self).__init__()
        self.coeff = nn.parameter.Parameter(
            torch.zeros(
                1,
                out_features,
                requires_grad=True,
                device=DEVICE
            ).exp()
        )
        # self.coeff = torch.zeros(1, out_features, requires_grad=True)
        self.layer = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            device=DEVICE,
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
        return self.coeff * F.tanh(self.layer(x))


def calc_output_size(
        # in_channels: int,
        # out_channels: int,
        hw: tuple[int, int],
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
):
    from math import floor
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    h = floor(1 + (
        (
            hw[0]
            + (2 * pad)
            - (dilation * (kernel_size[0] - 1))
            - 1
        ) / stride
    ))
    w = floor(1 + (
        (
            hw[1]
            + (2 * pad)
            - (dilation * (kernel_size[1] - 1))
            - 1
        ) / stride
    ))
    return h, w


class ConvStack(nn.Module):
    def __init__(
            self,
            xshape: list[int],
            conv_config: ConvolutionConfig,
            activation_fn: Any,
            use_batch_norm: bool = False,
    ) -> None:
        super(ConvStack, self).__init__()
        if len(xshape) == 3:
            d, nt, nx = xshape[0], xshape[1], xshape[2]
        elif len(xshape) == 4:
            _, d, nt, nx = xshape[0], xshape[1], xshape[2], xshape[3]
        else:
            raise ValueError(f'Invalid value for `xshape`: {xshape}')

        self.d = d
        self.nt = nt
        self.nx = nx
        self._with_cuda = torch.cuda.is_available()
        self.xshape = xshape
        self.xdim = np.cumprod(xshape[1:])[-1]
        self.activation_fn = activation_fn
        self.layers = nn.ModuleList([
            PeriodicPadding(conv_config.sizes[0] - 1),
            # in_channels, out_channels, kernel_size
            nn.LazyConv2d(
                conv_config.filters[0],
                conv_config.sizes[0],
                # padding='same',
                # padding_mode='circular',
            )
        ])
        for idx, (f, n) in enumerate(zip(
                conv_config.filters[1:],
                conv_config.sizes[1:],
        )):
            self.layers.append(PeriodicPadding(n - 1))
            self.layers.append(nn.LazyConv2d(f, n))
            if (idx + 1) % 2 == 0:
                p = 2 if conv_config.pool is None else conv_config.pool[idx]
                self.layers.append(nn.MaxPool2d(p))

            self.layers.append(self.activation_fn)

        self.layers.append(nn.Flatten())
        if use_batch_norm:
            self.layers.append(nn.BatchNorm1d(-1))
        self.layers.append(nn.LazyLinear(self.xdim))
        self.layers.append(self.activation_fn)

    def forward(self, x: Tensor) -> Tensor:
        x.requires_grad_(True)
        x = x.to(DEVICE)
        if x.shape != self.xshape:
            try:
                x = x.reshape(x.shape[0], self.d + 2, self.nt, self.nx)
            except (ValueError, RuntimeError):
                x = x.reshape(*self.xshape)

        for layer in self.layers:
            x = layer(x)

        return x


class InputLayer(nn.Module):
    def __init__(
            self,
            xshape: list[int],
            network_config: NetworkConfig,
            activation_fn: Callable[[Tensor], Tensor],
            conv_config: Optional[ConvolutionConfig] = None,
            input_shapes: Optional[dict[str, int]] = None,
    ) -> None:
        super(InputLayer, self).__init__()
        self.xshape = xshape
        self.net_config = network_config
        self.units = self.net_config.units
        self.xdim = np.cumprod(xshape[1:])[-1]
        self._with_cuda = torch.cuda.is_available()

        if input_shapes is None:
            input_shapes = {'x': self.xdim, 'v': self.xdim}

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

        self.conv_config = conv_config
        # if conv_config is not None:
        self.activation_fn = activation_fn
        if conv_config is not None and len(conv_config.filters) > 0:
            conv_stack = ConvStack(
                xshape=xshape,
                conv_config=conv_config,
                activation_fn=self.activation_fn,
            )
        else:
            conv_stack = None

        self.conv_stack = conv_stack
        # self.register_module('conv_stack', self.conv_stack)
        self.xlayer = nn.LazyLinear(
            self.net_config.units[0],
            device=DEVICE
        )
        self.vlayer = nn.LazyLinear(
            self.net_config.units[0],
            device=DEVICE
        )

    def forward(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> Tensor:
        x, v = inputs
        x.requires_grad_(True)
        v.requires_grad_(True)
        if self._with_cuda:
            x = x.cuda()
            v = v.cuda()
        self.vlayer.to(v.dtype)
        self.xlayer.to(x.dtype)
        if self.conv_stack is not None:
            x = self.conv_stack(x)

        v = self.vlayer(flatten(v))
        x = self.xlayer(flatten(x))
        return self.activation_fn(x + v)


class LeapfrogLayer(nn.Module):
    def __init__(
            self,
            xshape: list[int],
            network_config: NetworkConfig,
            input_shapes: Optional[dict[str, int]] = None,
            net_weight: Optional[NetWeight] = None,
            conv_config: Optional[ConvolutionConfig] = None,
            name: Optional[str] = None,
    ):
        super(LeapfrogLayer, self).__init__()
        if net_weight is None:
            net_weight = NetWeight(1., 1., 1.)

        self.xshape = xshape
        self.nw = net_weight
        self.net_config = network_config
        self.name = name if name is not None else 'network'
        self.xdim = np.cumprod(xshape[1:])[-1]
        self._with_cuda = torch.cuda.is_available()
        act_fn = self.net_config.activation_fn
        if isinstance(act_fn, str):
            act_fn = ACTIVATION_FNS.get(act_fn, None)

        assert isinstance(act_fn, Callable)
        self.activation_fn = act_fn

        self.input_layer = InputLayer(
            xshape=xshape,
            network_config=network_config,
            activation_fn=self.activation_fn,
            conv_config=conv_config,
            input_shapes=input_shapes,
        )

        self.units = self.net_config.units
        self.hidden_layers = nn.ModuleList()
        for idx, units in enumerate(self.units[1:]):
            h = nn.Linear(self.units[idx], units)
            self.hidden_layers.append(h)

        self.scale = ScaledTanh(self.units[-1], self.xdim)
        self.transf = ScaledTanh(self.units[-1], self.xdim)
        self.transl = nn.Linear(self.units[-1], self.xdim)

        if self.net_config.dropout_prob > 0:
            self.dropout = nn.Dropout(self.net_config.dropout_prob)

        if self.net_config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.units[-1], device=DEVICE)

        if torch.cuda.is_available():
            self.cuda()
            self.input_layer.cuda()
            self.hidden_layers.cuda()
            self.scale.cuda()
            self.transf.cuda()
            self.transl.cuda()

    def set_net_weight(self, net_weight: NetWeight):
        self.nw = net_weight

    def forward(
            self,
            inputs: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        x, v = inputs
        x.requires_grad_(True)
        v.requires_grad_(True)
        if self._with_cuda:
            x = x.cuda()
            v = v.cuda()
        self.input_layer.to(x.dtype)
        z = self.input_layer((x, v))
        for layer in self.hidden_layers:
            layer.to(z.dtype)
            z = self.activation_fn(layer(z))

        if self.net_config.dropout_prob > 0:
            z = self.dropout(z)

        if self.net_config.use_batch_norm:
            z = self.batch_norm(z)

        self.scale.to(z.dtype)
        self.transf.to(z.dtype)
        self.transl.to(z.dtype)
        scale = self.scale(z)
        transf = self.transf(z)
        transl = self.transl(z)

        return (
            self.nw.s * scale,
            self.nw.t * transl,
            self.nw.q * transf,
        )


def get_network(
        xshape: list[int],
        network_config: NetworkConfig,
        input_shapes: Optional[dict[str, int]] = None,
        net_weight: Optional[NetWeight] = None,
        conv_config: Optional[ConvolutionConfig] = None,
        name: Optional[str] = None,
) -> LeapfrogLayer:
    return LeapfrogLayer(
        xshape=xshape,
        network_config=network_config,
        input_shapes=input_shapes,
        net_weight=net_weight,
        conv_config=conv_config,
        name=name
    )


def get_and_call_network(
        xshape: list[int],
        *,
        network_config: NetworkConfig,
        is_xnet: bool,
        input_shapes: Optional[dict[str, int]] = None,
        net_weight: Optional[NetWeight] = None,
        conv_config: Optional[ConvolutionConfig] = None,
        name: Optional[str] = None,
) -> LeapfrogLayer:
    """Wrapper function for instantiating created LeapfrogLayers."""
    net = get_network(
        xshape=xshape,
        network_config=network_config,
        input_shapes=input_shapes,
        net_weight=net_weight,
        conv_config=conv_config,
        name=name,
    )
    x = torch.rand(xshape)
    v = torch.rand_like(x)
    if is_xnet:
        # TODO: Generalize for 4D SU(3)
        x = torch.cat([x.cos(), x.sin()], dim=1)

    _ = net((x, v))
    return net


class NetworkFactory(BaseNetworkFactory):
    def build_networks(self, n: int, split_xnets: bool) -> nn.ModuleDict:
        """Build LeapfrogNetwork."""
        # TODO: if n == 0: build hmcNetwork (return zeros)
        assert n >= 1, 'Must build at least one network'

        cfg = self.get_build_configs()
        if n == 1:
            return nn.ModuleDict({
                'xnet': get_and_call_network(**cfg['xnet'], is_xnet=True),
                'vnet': get_and_call_network(**cfg['vnet'], is_xnet=False),
                # 'xnet': Network(**cfg['xnet']),
                # 'vnet': Network(**cfg['vnet']),
            })

        vnet = nn.ModuleDict()
        xnet = nn.ModuleDict()
        for lf in range(n):
            vnet[f'{lf}'] = get_and_call_network(
                **cfg['vnet'],
                is_xnet=False,
                name=f'vnet/{lf}',
            )
            if split_xnets:
                xnet[f'{lf}'] = nn.ModuleDict({
                    'first': get_and_call_network(
                        **cfg['xnet'],
                        is_xnet=True,
                        name=f'xnet/{lf}/first'
                    ),
                    'second': get_and_call_network(
                        **cfg['xnet'],
                        is_xnet=True,
                        name=f'xnet/{lf}/second'
                    ),
                })
            else:
                xnet[f'{lf}'] = get_and_call_network(
                    **cfg['xnet'],
                    is_xnet=True,
                    name=f'xnet/{lf}'
                )

        return nn.ModuleDict({'xnet': xnet, 'vnet': vnet})
