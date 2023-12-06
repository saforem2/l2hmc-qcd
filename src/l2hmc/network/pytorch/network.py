"""
pytorch/network.py

Contains the pytorch implementation of the Normalizing Flow network

used to train the L2HMC model.
"""
from __future__ import absolute_import, annotations, division, print_function
from typing import Any, Callable, Optional, Sequence
import logging

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from l2hmc.group.su3.pytorch.group import SU3
from l2hmc.group.u1.pytorch.group import U1Phase


from l2hmc import DEVICE
from l2hmc.configs import (
    NetWeight,
    NetworkConfig,
    ConvolutionConfig
)

from l2hmc.network.factory import BaseNetworkFactory
# from l2hmc.utils.logger import get_pylogger


# log = get_pylogger(__name__)
# from l2hmc import get_logger
# log = get_logger(__name__)
log = logging.getLogger(__name__)

Tensor = torch.Tensor

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ACTIVATION_FNS = {
    'elu': nn.ELU(inplace=True),
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(inplace=True),
    'swish': nn.SiLU(),
    'leaky_relu': nn.LeakyReLU(inplace=True),
}


def nested_children(m: nn.Module) -> dict[str, nn.Module]:
    children = dict(m.named_children())
    if not list(children.keys()):
        return {m._get_name(): m}

    return {
        name: nested_children(child) if isinstance(child, nn.Module) else child
        for name, child in children.items()
    }


def flatten(x: torch.Tensor) -> torch.Tensor:
    # return x.reshape(x.shape[0], -1)
    return x.view(x.shape[0], -1)


def xy_repr(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.cos(x), torch.sin(x)), dim=1)


def dummy_network(
        inputs: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor, Tensor]:
    x, _ = inputs
    return (
        torch.zeros_like(x),
        torch.zeros_like(x),
        torch.zeros_like(x)
    )


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
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x: Tensor) -> Tensor:
        if torch.cuda.is_available():
            x = x.cuda()

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
            )
        )
        self.layer = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            # bias=False,
            device=DEVICE,
        )
        # self.tanh = nn.Tanh()
        self._with_cuda = torch.cuda.is_available()
        if self._with_cuda:
            self.coeff = self.coeff.cuda()
            self.layer = self.layer.cuda()
            self.cuda()

    def forward(self, x):
        if self._with_cuda:
            x = x.cuda()
        return self.coeff.exp() * F.tanh(self.layer(x))


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
            xshape: Sequence[int],
            conv_config: ConvolutionConfig,
            activation_fn: Any,
            use_batch_norm: bool = False,
    ) -> None:
        super(ConvStack, self).__init__()
        # ------------------------------------------------
        # NOTE:
        # - xshape will be of the form
        #   [batch, dim, *lattice_shape, *link_shape]
        # - Cases:
        #   1. xshape: [dim, nt, nx]
        #   2. xshape: [batch, dim, nt, nx]
        #   3. xshape: [batch, dim, nt, nx, ny, nz, 3, 3]
        # ------------------------------------------------
        # -- CASE 1: [dim, nt, nx] -----------------------
        if len(xshape) == 3:
            # lattice_shape: [nt, nx]
            ny, nz = 0, 0
            d, nt, nx = xshape[0], xshape[1], xshape[2]

        # -- CASE 2: [batch, dim, nt, nx] ----------------
        elif len(xshape) == 4:
            # lattice_shape: [nt, nx]
            ny, nz = 0, 0
            _, d, nt, nx = xshape[0], xshape[1], xshape[2], xshape[3]

        # -- CASE 2: [batch, dim, nt, nx, ny, nz, 3, 3] --
        elif len(xshape) == 8:
            # link_shape: [3, 3]
            # lattice_shape = [nt, nx, ny, nz]
            d = xshape[1]
            nt, nx, ny, nz = xshape[2], xshape[3], xshape[4], xshape[5]
        else:
            raise ValueError(f'Invalid value for xshape: {xshape}')
        # ---------------------------------------------------------------

        self.d = d
        self.nt = nt
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self._with_cuda = torch.cuda.is_available()
        self.xshape = xshape
        self.xdim = np.cumprod(xshape[1:])[-1]
        self.activation_fn = activation_fn
        self.layers = nn.ModuleList()
        if conv_config.filters is not None:
            if (nfilters := len(list(conv_config.filters))) > 0:
                if (
                        conv_config.sizes is not None
                        and nfilters == len(conv_config.sizes)
                ):
                    self.layers.append(
                        PeriodicPadding(
                            conv_config.sizes[0] - 1
                        )
                    )
                    # in_channels, out_channels, kernel_size
                    self.layers.append(
                        nn.LazyConv2d(
                            conv_config.filters[0],
                            conv_config.sizes[0],
                        )
                    )
                    for idx, (f, n) in enumerate(zip(
                            conv_config.filters[1:],
                            conv_config.sizes[1:],
                    )):
                        self.layers.append(PeriodicPadding(n - 1))
                        self.layers.append(nn.LazyConv2d(f, n))
                        if (idx + 1) % 2 == 0:
                            p = (
                                2 if conv_config.pool is None
                                else conv_config.pool[idx]
                            )
                            self.layers.append(nn.MaxPool2d(p))

                        self.layers.append(self.activation_fn)

        self.layers.append(nn.Flatten())
        if use_batch_norm:
            self.layers.append(nn.BatchNorm1d(-1))
        self.layers.append(nn.LazyLinear(self.xdim))
        self.layers.append(self.activation_fn)
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x: Tensor) -> Tensor:
        x.requires_grad_(True)
        x = x.to(DEVICE)
        # self.to(x.dtype)
        # self.to(x.dtype)
        if x.shape != self.xshape:
            try:
                x = x.reshape(x.shape[0], self.d + 2, self.nt, self.nx)
            except (ValueError, RuntimeError):
                x = x.reshape((x.shape[0], *self.xshape[1:]))
                # x = x.reshape(*self.xshape)

        for layer in self.layers:
            x = layer(x)

        return x


class InputLayer(nn.Module):
    def __init__(
            self,
            xshape: Sequence[int],
            network_config: NetworkConfig,
            activation_fn: Callable[[Tensor], Tensor],
            vshape: Optional[Sequence[int]] = None,
            conv_config: Optional[ConvolutionConfig] = None,
            input_shapes: Optional[dict[str, Sequence[int] | int]] = None,
    ) -> None:
        super(InputLayer, self).__init__()
        self.xshape = xshape
        self.vshape = xshape if vshape is None else vshape
        self.net_config = network_config
        self.units = self.net_config.units
        # self._dtype = torch.get_default_dtype()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.xdim = np.cumprod(self.xshape[1:])[-1]
        self.vdim = np.cumprod(self.vshape[1:])[-1]
        self._with_cuda = torch.cuda.is_available()

        if input_shapes is None:
            input_shapes = {'x': self.xdim, 'v': self.xdim}

        self.input_shapes = {}
        for key, val in input_shapes.items():
            # if isinstance(val, Sequence[int]):
            if isinstance(val, int):
                self.input_shapes[key] = val
            else:
                try:
                    self.input_shapes[key] = np.cumprod(val)[-1]
                except Exception as e:
                    log.error(
                        'Unexpected value in input_shapes!\n'
                        f'\tinput_shapes: {input_shapes}\n'
                        f'\t  val: {val}'
                    )
                    raise e
                # raise ValueError(
                #     'Unexpected value in input_shapes!\n'
                #     f'\tinput_shapes: {input_shapes}\n'
                #     f'\t  val: {val}'
                # )

        self.conv_config = conv_config
        self.activation_fn = activation_fn
        conv_stack = nn.Identity()
        # vconv_stack = nn.Identity()
        if conv_config is not None:
            if (
                    conv_config.filters is not None
                    and len(conv_config.filters) > 0
            ):
                conv_stack = ConvStack(
                    xshape=xshape,
                    conv_config=conv_config,
                    activation_fn=self.activation_fn,
                )
                # vconv_stack = ConvStackr
                #     xshape=xshape,
                #     conv_config=conv_config,
                #     activation_fn=self.activation_fn
                # )
        # self.register_module('conv_stack', conv_stack)
        self.conv_stack = conv_stack
        self.xlayer = nn.LazyLinear(
            self.net_config.units[0],
            device=DEVICE
        )
        self.vlayer = nn.LazyLinear(
            self.net_config.units[0],
            device=DEVICE
        )
        if torch.cuda.is_available():
            self.conv_stack.cuda()
            self.xlayer.cuda()
            self.vlayer.cuda()
            self.cuda()
            # self.to(self._dtype)

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
            # x = x.to(self._device).to(self._dtype)
            # v = v.to(self._device).to(self._dtype)
            # x = x.cuda()
            # v = v.cuda()
        # self.vlayer.to(v.dtype)
        # self.xlayer.to(x.dtype)
        if self.conv_stack is not None:
            x = self.conv_stack(x)

        v = self.vlayer(flatten(v))
        x = self.xlayer(flatten(x))
        return self.activation_fn(x + v)


class LeapfrogLayer(nn.Module):
    def __init__(
            self,
            xshape: Sequence[int],
            network_config: NetworkConfig,
            input_shapes: Optional[dict[str, int | Sequence[int]]] = None,
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
        # self._dtype = torch.get_default_dtype()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

        # xdim = self.input_layer.xlayer.weight.shape[-1]
        xdim = np.cumprod(xshape[1:])[-1]
        self.scale = ScaledTanh(self.units[-1], xdim)
        self.transf = ScaledTanh(self.units[-1], xdim)
        self.transl = nn.Linear(self.units[-1], xdim)

        self.dropout = nn.Dropout(self.net_config.dropout_prob)
        # if self.net_config.dropout_prob > 0:
        #     self.dropout = nn.Dropout(self.net_config.dropout_prob)

        if self.net_config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.units[-1], device=DEVICE)

        if torch.cuda.is_available():
            self.cuda()
            # self.to(self._dtype)
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
        # x = x.to(self._device).to(self._dtype)
        # v = v.to(self._device).to(self._dtype)
        if self._with_cuda:
            x = x.cuda()
            v = v.cuda()

            # self.to(x.dtype)
        # self.input_layer.to(x.dtype)
        z = self.input_layer((x, v))
        for layer in self.hidden_layers:
            # layer.to(z.dtype)
            z = self.activation_fn(layer(z))

        if self.net_config.dropout_prob > 0 and self.dropout is not None:
            z = self.dropout(z)

        if self.net_config.use_batch_norm:
            z = self.batch_norm(z)

        s = self.nw.s * self.scale(z)
        t = self.nw.t * self.transl(z)
        q = self.nw.q * self.transf(z)
        return (s, t, q)


def get_network(
        xshape: Sequence[int],
        network_config: NetworkConfig,
        input_shapes: Optional[dict[str, int | Sequence[int]]] = None,
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
        xshape: Sequence[int],
        *,
        network_config: NetworkConfig,
        is_xnet: bool,
        group: U1Phase | SU3,
        input_shapes: Optional[dict[str, int | Sequence[int]]] = None,
        net_weight: Optional[NetWeight] = None,
        conv_config: Optional[ConvolutionConfig] = None,
        name: Optional[str] = None,
) -> LeapfrogLayer:
    """Wrapper function for instantiating created LeapfrogLayers."""
    # if isinstance(group, U1Phase):
    #     x = torch.rand(xshape)
    # elif isinstance(group, SU3):
    x = group.random(xshape)
    v = group.random_momentum(xshape)

    net = get_network(
        xshape=xshape,
        network_config=network_config,
        input_shapes=input_shapes,
        net_weight=net_weight,
        conv_config=conv_config,
        name=name,
    )

    if torch.cuda.is_available():
        net.cuda()
        x = x.cuda()
        v = v.cuda()
        # net = net.to(x.dtype)

    if isinstance(group, U1Phase):
        # TODO: Generalize for 4D SU(3)
        # x = torch.cat([x.cos(), x.sin()], dim=1)
        if is_xnet:
            x = group.group_to_vec(x)
    if (
            isinstance(group, SU3)
            or getattr(group, '_name', None) == 'SU3'
    ):
        if is_xnet:
            x = torch.cat([x.real, x.imag], dim=1)
            v = torch.cat([v.real, v.imag], dim=1)
        else:
            x = group.group_to_vec(x)
            v = group.group_to_vec(v)
        # if is_xnet:
        #     x = torch.stack([x.real, x.imag], 1)
        #     pass
        # x = group.group_to_vec(x)
        # x = group.group_to_vec(x)
        # v = group.group_to_vec(v)

    # except RuntimeError as e:
    #     # net = net.to(x.dtype)
    #     _ = net((x, v))
    _ = net((x, v))
    return net


class NetworkFactory(BaseNetworkFactory):
    def build_xnet(
            self,
            group: SU3 | U1Phase,
            name: Optional[str] = None,
    ) -> LeapfrogLayer:
        xname = 'xnet' if name is None else f'xnet/{name}'
        return get_and_call_network(
            xshape=self.input_spec.xshape,
            network_config=self.network_config,
            is_xnet=True,
            group=group,
            input_shapes=self.input_spec.xnet,
            net_weight=self.nw.x,
            conv_config=self.conv_config,
            name=xname,
        )

    def build_vnet(
            self,
            group: SU3 | U1Phase,
            name: Optional[str] = None,
    ) -> LeapfrogLayer:
        vname = 'vnet' if name is None else f'vnet/{name}'
        return get_and_call_network(
            xshape=self.input_spec.xshape,
            network_config=self.network_config,
            is_xnet=False,
            group=group,
            input_shapes=self.input_spec.vnet,
            net_weight=self.nw.v,
            conv_config=self.conv_config,
            name=vname,
        )

    def build_networks(
            self,
            n: int,
            split_xnets: bool,
            group: SU3 | U1Phase,
    ) -> nn.ModuleDict:
        """Build LeapfrogNetwork."""
        # TODO: if n == 0: build hmcNetwork (return zeros)
        assert n >= 1, 'Must build at least one network'
        if n == 1:
            # xnet = get_and_call_network(
            #     xshape=self.input_spec.xshape,
            #     network_config=self.network_config,
            #     is_xnet=True,
            #     group=group,
            #     input_shapes=self.input_spec.xnet,
            #     net_weight=self.nw.x,
            #     conv_config=self.conv_config,
            #     name='xnet'
            # )
            # vnet = get_and_call_network(
            #     xshape=self.input_spec.xshape,
            #     network_config=self.network_config,
            #     is_xnet=False,
            #     group=group,
            #     input_shapes=self.input_spec.vnet,
            #     net_weight=self.nw.v,
            #     conv_config=self.conv_config,
            #     name='vnet'
            # )
            return nn.ModuleDict({
                'xnet': self.build_xnet(group=group),
                'vnet': self.build_vnet(group=group),
            })

        vnet = nn.ModuleDict()
        xnet = nn.ModuleDict()
        for lf in range(n):
            vnet[f'{lf}'] = self.build_vnet(
                group=group,
                name=f'{lf}',
            )
            # vnet[f'{lf}'] = get_and_call_network(
            #     **cfg['vnet'],
            #     is_xnet=False,
            #     name=f'vnet/{lf}',
            #     group=group,
            # )
            # vnet[f'{lf}'] = get_and_call_network(
            #     xshape=self.input_spec.xshape,
            #     network_config=self.network_config,
            #     is_xnet=False,
            #     group=group,
            #     input_shapes=self.input_spec.vnet,
            #     net_weight=self.nw.v,
            #     conv_config=self.conv_config,
            #     name=f'vnet/{lf}'
            # )
            # vnet[f'{lf}'] = LeapfrogLayer(
            #     **cfg['vnet'],
            #     name='vnet',
            # )
            if split_xnets:
                # xnet[f'{lf}'] = nn.ModuleDict({
                #     'first': LeapfrogLayer(
                #         **cfg['xnet'],
                #         name='xnet/{lf}/first',
                #     ),
                #     'second': LeapfrogLayer(
                #         **cfg['xnet'],
                #         name='xnet/{lf}/first',
                #     )
                # })
                # xnet[f'{lf}'] = get_and_call_network(
                #     xshape=self.input_spec.xshape,
                #     network_config=self.network_config,
                #     is_xnet=True,
                #     group=group,
                #     input_shapes=self.input_spec.xnet,
                #     net_weight=self.nw.x,
                #     conv_config=self.conv_config,
                #     name=f'xnet/{lf}'
                # )
                # xnet[f'{lf}'] = nn.ModuleDict({
                #     'first': get_and_call_network(
                #         xshape=self.input_spec.xshape,
                #         network_config=self.network_config,
                #         is_xnet=True,
                #         group=group,
                #         input_shapes=self.input_spec.xnet,
                #         net_weight=self.nw.x,
                #         conv_config=self.conv_config,
                #         name=f'xnet/{lf}/first',
                #     ),
                #     'second': get_and_call_network(
                #         xshape=self.input_spec.xshape,
                #         network_config=self.network_config,
                #         is_xnet=True,
                #         group=group,
                #         input_shapes=self.input_spec.xnet,
                #         net_weight=self.nw.x,
                #         conv_config=self.conv_config,
                #         name=f'xnet/{lf}/second',
                #     ),
                # })
                xnet[f'{lf}'] = nn.ModuleDict({
                    'first': self.build_xnet(group=group, name=f'{lf}/first'),
                    'second': self.build_xnet(group=group, name=f'{lf}/second')
                })
            else:
                xnet[f'{lf}'] = self.build_xnet(group=group, name=f'{lf}')
                # xnet[f'{lf}'] = get_and_call_network(
                #     xshape=self.input_spec.xshape,
                #     network_config=self.network_config,
                #     is_xnet=True,
                #     group=group,
                #     input_shapes=self.input_spec.xnet,
                #     net_weight=self.nw.x,
                #     conv_config=self.conv_config,
                #     name=f'xnet/{lf}'
                # )
                # xnet[f'{lf}'] = get_and_call_network(
                #     **cfg['xnet'],
                #     is_xnet=True,
                #     name=f'xnet/{lf}',
                #     group=group,
                # )
                # xnet[f'{lf}'] = LeapfrogLayer(
                #     **cfg['xnet'],
                #     name=f'xnet/{lf}',
                # )

        return nn.ModuleDict({'xnet': xnet, 'vnet': vnet})
