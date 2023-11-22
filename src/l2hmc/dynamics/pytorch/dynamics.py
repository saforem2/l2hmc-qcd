"""
pytorch/dynamics.py

Pytorch implementation of Dynamics object for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass
import logging
from math import pi as PI
import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Union
from typing import Tuple

import numpy as np
import torch
from torch import nn

# from l2hmc import get_logger
import l2hmc.configs as cfgs
from l2hmc.group.u1.pytorch.group import U1Phase
from l2hmc.lattice.u1.pytorch.lattice import LatticeU1
from l2hmc.group.su3.pytorch.group import SU3
from l2hmc.lattice.su3.pytorch.lattice import LatticeSU3
from l2hmc.network.pytorch.network import NetworkFactory, dummy_network


log = logging.getLogger(__name__)
# log = get_logger(__name__)

TWO_PI = 2. * PI

Shape = Union[tuple, list]
Tensor = torch.Tensor
Array = np.ndarray
FLOAT = torch.get_default_dtype()


# (xinit, beta)
DynamicsInput = Tuple[Tensor, Tensor]
# (xout, metrics)
DynamicsOutput = Tuple[Tensor, dict]


@dataclass
class State:
    x: Tensor               # gauge links
    v: Tensor               # conj. momenta
    beta: Tensor            # inv. coupling const.

    def __post_init__(self):
        self.nb = self.x.shape[0]
        self.xshape = self.x.shape

    def flatten(self) -> State:
        return State(
            x=self.x.flatten(1),
            v=self.v.flatten(1),
            beta=self.beta,
        )

    def to_numpy(self):
        return {
            'x': self.x.detach().numpy(),  # type:ignore
            'v': self.v.detach().numpy(),  # type:ignore
            'beta': self.beta.detach().numpy(),  # type:ignore
        }


@dataclass
class MonteCarloStates:
    init: State             # Input state
    proposed: State         # Proposal state
    out: State              # Output state (after acc/rej)


def to_u1(x: Tensor) -> Tensor:
    """Returns x as U(1) link variable in [-pi, pi]."""
    return ((x + PI) % TWO_PI) - PI


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1. / (1. + torch.exp(-x))


def rand_unif(
        shape: Shape,
        a: float,
        b: float,
        requires_grad: bool
) -> Tensor:
    """Returns tensor from random uniform distribution ~U[a, b]."""
    rand = (a - b) * torch.rand(tuple(shape)) + b
    return rand.clone().detach().requires_grad_(requires_grad)


def random_angle(shape: Shape, requires_grad: bool = True) -> Tensor:
    """Returns random angle with `shape` and values in (-pi, pi)."""
    return rand_unif(shape, -PI, PI, requires_grad=requires_grad)


# TODO: Remove or finish implementation ?
class Mask:
    def __init__(self, m: Tensor):
        self.m = m
        # complement: 1. - m
        self.mb = torch.ones_like(self.m) - self.m

    def combine(self, x: Tensor, y: Tensor):
        return self.m * x + self.mb * y


class Dynamics(nn.Module):
    def __init__(
            self,
            potential_fn: Callable,
            config: cfgs.DynamicsConfig,
            network_factory: Optional[NetworkFactory] = None,
    ):
        """Initialization method."""
        super().__init__()
        self.config = config
        self.xdim = self.config.xdim
        self.xshape = self.config.xshape
        self.potential_fn = potential_fn
        self.nlf = self.config.nleapfrog
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.config.group == 'U1':
            self.g = U1Phase()
            self.lattice = LatticeU1(self.config.nchains,
                                     self.config.latvolume)
        elif self.config.group == 'SU3':
            self.g = SU3()
            self.lattice = LatticeSU3(self.config.nchains,
                                      self.config.latvolume)
        self.network_factory = network_factory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if network_factory is not None:
            self._networks_built = True
            self.networks = self._build_networks(network_factory)
            self.xnet = self.networks['xnet']
            self.vnet = self.networks['vnet']
            self.register_module('xnet', self.networks['xnet'])
            self.register_module('vnet', self.networks['vnet'])
        else:
            self._networks_built = False
            self.xnet = dummy_network
            self.vnet = dummy_network
            self.networks = {
                'xnet': self.xnet,
                'vnet': self.vnet,
            }
        self.masks = self._build_masks()
        self._dtype: torch.dtype = torch.get_default_dtype()
        if self._networks_built:
            vnet = self._get_vnet(0)
            self._dtype: torch.dtype = (
                vnet.input_layer.xlayer.weight.dtype  # type:ignore
            )
        rg = (not self.config.eps_fixed)
        # logxeps = nn.ParameterList(
        self.xeps = nn.ParameterList(
            [
                nn.parameter.Parameter(
                    # torch.log(
                    torch.tensor(self.config.eps, requires_grad=rg),
                    # ),
                    requires_grad=rg,
                )
                for _ in range(self.config.nleapfrog)
            ]
        )
        self.veps = nn.ParameterList(
            [
                nn.parameter.Parameter(
                    # torch.log(
                    torch.tensor(self.config.eps, requires_grad=rg),
                    # ),
                    requires_grad=rg,
                )
                for _ in range(self.config.nleapfrog)
            ]
        )
        # self.xeps = logxeps.exp_()
        # self.veps = logveps.exp_()
        # self.xeps = torch.stack([
        #     eps for _ in range(self.config.nleapfrog)
        # ])
        # self.veps = torch.stack([
        #     eps for _ in range(self.config.nleapfrog)
        # ])
        # assert isinstance(self.xeps, nn.ParameterList)
        # assert isinstance(self.veps, nn.ParameterList)
        if torch.cuda.is_available():
            self.xeps = self.xeps.cuda()
            self.veps = self.veps.cuda()
            self.cuda()
            if self._networks_built:
                self.xnet.cuda()
                self.vnet.cuda()

    def get_models(self) -> dict:
        if self.config.use_separate_networks:
            xnet = {}
            vnet = {}
            for lf in range(self.config.nleapfrog):
                vnet[str(lf)] = self._get_vnet(lf)
                if self.config.use_split_xnets:
                    xnet[str(lf)] = {
                        '0': self._get_xnet(lf, first=True),
                        '1': self._get_xnet(lf, first=False),
                    }
                else:
                    xnet[str(lf)] = self._get_xnet(lf, first=True)
        else:
            vnet = self._get_vnet(0)
            if self.config.use_split_xnets:
                xnet = {
                    '0': self._get_xnet(0, first=True),
                    '1': self._get_xnet(0, first=False),
                }
            else:
                xnet = self._get_xnet(0, first=True)
        return {'xnet': xnet, 'vnet': vnet}

    def _build_networks(
            self,
            network_factory: NetworkFactory
    ) -> nn.ModuleDict:
        """Build networks."""
        split = self.config.use_split_xnets
        n = self.nlf if self.config.use_separate_networks else 1
        return network_factory.build_networks(
            n,
            split,
            group=self.g,
        )

    def _build_xnet(
            self,
            xshape: Sequence[int],
            network_config: cfgs.NetworkConfig,
            input_shapes: Optional[dict[str, int | Sequence[int]]] = None,
            net_weight: Optional[cfgs.NetWeight] = None,
            conv_config: Optional[cfgs.ConvolutionConfig] = None,
            name: Optional[str] = None,
    ):
        # return LeapfrogLayer
        pass

    def _build_networks1(
            self,
            network_factory: NetworkFactory
    ) -> nn.ModuleDict:
        """Build networks."""
        split = self.config.use_split_xnets
        n = self.nlf if self.config.use_separate_networks else 1

    @torch.no_grad()
    def _init_weight(
            self,
            m: nn.Module,
            idx: int,
            method='xavier_uniform',
            rval: Optional[float] = None,
            constant: Optional[float] = None,
            bias: Optional[bool] = None,
            min: Optional[float] = None,
            max: Optional[float] = None,
            mean: Optional[float] = None,
            std: Optional[float] = None,
    ):
        def log_init(idx: int, s: str) -> None:
            if idx == 0:
                log.info(s)
        b = m.bias
        w = m.weight
        assert isinstance(b, torch.Tensor)
        assert isinstance(w, torch.Tensor)
        if method is None:
            if rval is not None:
                log_init(
                    idx=idx,
                    s=('Initializing weights from '
                       f'~ U[-{rval:.2f}, {rval:.2f}]')
                )
                nn.init.uniform(m.weight, a=-rval, b=rval)
                if bias and m.bias is not None:
                    nn.init.uniform(m.bias, a=-rval, b=rval)
            if constant is not None:
                log_init(
                    idx=idx,
                    s=f'Initializing weights with const = {constant}'
                )
                nn.init.constant_(w, constant)
                # if bias is not None:
                if bias and m.bias is not None:
                    log_init(
                        idx=idx,
                        s=f'Initializing weights with bias: {bias}'
                    )
                    if m.bias is not None:
                        nn.init.constant_(b, bias)
        elif method in ['zero', 'zeros']:
            log_init(
                idx=idx,
                s='Iniitializing weights with 0 s'
            )
            nn.init.zeros_(w)
            # if bias:
            if bias and m.bias is not None:
                nn.init.zeros_(b)
        elif method == 'uniform':
            a = 0.0 if min is None else min
            b = 1.0 if max is None else max
            a = torch.Parameter.parameter(a)
            b = torch.Parameter.parameter(b)
            log_init(idx=idx, s=f'Initializing weights ~ U[{a}, {b}]')
            nn.init.uniform_(w, a=a, b=b)
            if bias and b is not None:
                nn.init.uniform_(b, a=a, b=b)
        elif method == 'normal':
            mean = 0.0 if mean is None else mean
            std = 1.0 if std is None else std
            log_init(
                idx=idx,
                s=(f'Initializing weights '
                    f'~ N[{mean:.2f}; {std:.2f}]')
            )
            nn.init.normal_(w, mean=mean, std=std)
            if bias and b is not None:
                nn.init.normal_(b, mean=mean, std=std)
        elif method == 'xavier_uniform':
            log_init(idx=idx, s=f'Initializing weights with: {method}')
            nn.init.xavier_uniform_(w)
            if bias and b is not None:
                nn.init.xavier_uniform_(b)
        elif method == 'kaiming_uniform':
            log_init(idx=idx, s=f'Initializing weights with: {method}')
            nn.init.kaiming_uniform_(w)
            if bias and b is not None:
                nn.init.kaiming_uniform_(b)
        elif method == 'xavier_normal':
            log_init(idx=idx, s=f'Initializing weights with: {method}')
            nn.init.xavier_normal_(w)
            if bias and b is not None:
                nn.init.xavier_normal_(b)
        elif method == 'kaiming_normal':
            log_init(idx=idx, s=f'Initializing weights with: {method}')
            nn.init.kaiming_normal_(w)
            if bias and b is not None:
                nn.init.kaiming_normal_(b)
        else:
            try:
                method = getattr(nn.init, method)
                if method is not None and callable(method):
                    method(m.weight)
                    # if bias:
                    if bias and m.bias is not None:
                        method(m.bias)
            except NameError:
                log.warning('. '.join([
                    'Unable to initialize weights',
                    f' with {method}',
                    'Falling back to default: xavier_uniform_'
                ]))
                nn.init.xavier_uniform_(w)
                # if bias:
                if bias and m.bias is not None:
                    nn.init.xavier_uniform_(b)

    @torch.no_grad()
    def init_weights(
            self,
            *,
            method='xavier_uniform',
            rval: Optional[float] = None,
            constant: Optional[float] = None,
            bias: Optional[bool] = None,
            min: Optional[float] = None,
            max: Optional[float] = None,
            mean: Optional[float] = None,
            std: Optional[float] = None,
            xeps: Optional[float] = None,
            veps: Optional[float] = None,
            # fn: Optional[Callable] = None,
    ):
        def log_init(idx: int, s: str) -> None:
            if idx == 0:
                log.info(s)
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        # for idx, m in enumerate(self.modules()):
        for idx, m in enumerate(self.trainable_parameters()):
            if not isinstance(getattr(m, 'weight', None), torch.Tensor):
                continue
            # if hasattr(m, 'weight'):
            # if isinstance(m, nn.Module):
            # if isinstance(m, nn.Linear):
            # if bias is not None:
            #     if isinstance(bias, float):
            #         log_init(
            #             idx=idx,
            #             s=f'initializing bias with: {bias:.4f}'
            #         )
            #         nn.init.constant_(m.bias, bias)
            #     else:
            #         log_init(
            #             idx=idx,
            #             s='Initializing bias-es with weights!'
            #         )
            #         SET_BIAS = True
            # else:
            #     SET_BIAS = False
            if method is None:
                if rval is not None:
                    log_init(
                        idx=idx,
                        s=('Initializing weights from '
                           f'~ U[-{rval:.2f}, {rval:.2f}]')
                    )
                    nn.init.uniform(m.weight, a=-rval, b=rval)
                    if bias and m.bias is not None:
                        nn.init.uniform(m.bias, a=-rval, b=rval)
                if constant is not None:
                    log_init(
                        idx=idx,
                        s=f'Initializing weights with const = {constant}'
                    )
                    nn.init.constant_(m.weight, constant)
                    # if bias is not None:
                    if bias and m.bias is not None:
                        log_init(
                            idx=idx,
                            s=f'Initializing weights with bias: {bias}'
                        )
                        if m.bias is not None:
                            nn.init.constant_(m.bias, bias)
            elif method in ['zero', 'zeros']:
                log_init(
                    idx=idx,
                    s='Iniitializing weights with 0 s'
                )
                nn.init.zeros_(m.weight)
                # if bias:
                if bias and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif method == 'uniform':
                a = 0.0 if min is None else min
                b = 1.0 if max is None else max
                log_init(idx=idx, s=f'Initializing weights ~ U[{a}, {b}]')
                nn.init.uniform_(m.weight, a=a, b=b)
                # if bias:
                if bias and m.bias is not None:
                    nn.init.uniform_(m.bias, a=a, b=b)
            elif method == 'normal':
                mean = 0.0 if mean is None else mean
                std = 1.0 if std is None else std
                log_init(
                    idx=idx,
                    s=(f'Initializing weights '
                        f'~ N[{mean:.2f}; {std:.2f}]')
                )
                nn.init.normal_(m.weight, mean=mean, std=std)
                # if bias:
                if bias and m.bias is not None:
                    nn.init.normal_(m.bias, mean=mean, std=std)
            elif method == 'xavier_uniform':
                log_init(idx=idx, s=f'Initializing weights with: {method}')
                nn.init.xavier_uniform_(m.weight)
                # if bias:
                if bias and m.bias is not None:
                    nn.init.xavier_uniform_(m.bias)
            elif method == 'kaiming_uniform':
                log_init(idx=idx, s=f'Initializing weights with: {method}')
                nn.init.kaiming_uniform_(m.weight)
                # if bias:
                if bias and m.bias is not None:
                    nn.init.kaiming_uniform_(m.bias)
            elif method == 'xavier_normal':
                log_init(idx=idx, s=f'Initializing weights with: {method}')
                nn.init.xavier_normal_(m.weight)
                # if bias:
                if bias and m.bias is not None:
                    nn.init.xavier_normal_(m.bias)
            elif method == 'kaiming_normal':
                log_init(idx=idx, s=f'Initializing weights with: {method}')
                nn.init.kaiming_normal_(m.weight)
                # if bias:
                if bias and m.bias is not None:
                    nn.init.kaiming_normal_(m.bias)
            else:
                try:
                    method = getattr(nn.init, method)
                    if method is not None and callable(method):
                        method(m.weight)
                        # if bias:
                        if bias and m.bias is not None:
                            method(m.bias)
                except NameError:
                    log.warning('. '.join([
                        'Unable to initialize weights',
                        f' with {method}',
                        'Falling back to default: xavier_uniform_'
                    ]))
                    nn.init.xavier_uniform_(m.weight)
                    # if bias:
                    if bias and m.bias is not None:
                        nn.init.xavier_uniform_(m.bias)

        if xeps is not None or veps is not None:
            rg = not self.config.eps_fixed
            if xeps is not None:
                xeps = self.config.eps if xeps is None else xeps
                self.xeps = nn.ParameterList(
                    [
                        nn.parameter.Parameter(
                            torch.tensor(xeps, requires_grad=rg),
                            requires_grad=rg,
                        )
                        for _ in range(self.config.nleapfrog)
                    ]
                )
            if veps is not None:
                veps = self.config.eps if veps is None else veps
                self.veps = nn.ParameterList(
                    [
                        nn.parameter.Parameter(
                            torch.tensor(veps, requires_grad=rg),
                            requires_grad=rg,
                        )
                        for _ in range(self.config.nleapfrog)
                    ]
                )

    def save(self, outdir: os.PathLike) -> None:
        netdir = Path(outdir).joinpath('networks')
        outfile = netdir.joinpath('dynamics.pt')
        netdir.mkdir(exist_ok=True, parents=True)
        self.save_eps(outdir=netdir)
        torch.save(self.state_dict(), outfile.as_posix())

    def save_eps(self, outdir: os.PathLike) -> None:
        netdir = Path(outdir).joinpath('networks')
        fxeps = netdir.joinpath('xeps.npy')
        fveps = netdir.joinpath('veps.npy')
        xeps = np.array([
            i.detach().cpu().numpy() for i in self.xeps
        ])
        veps = np.array([
            i.detach().cpu().numpy() for i in self.veps
        ])
        np.save(fxeps, xeps)
        np.save(fveps, veps)
        np.savetxt(netdir.joinpath('xeps.txt').as_posix(), xeps)
        np.savetxt(netdir.joinpath('veps.txt').as_posix(), veps)

    def load(self, outdir: os.PathLike) -> None:
        netdir = Path(outdir).joinpath('networks')
        netfile = netdir.joinpath('dynamics.pt')
        self.load_state_dict(torch.load(netfile))
        eps = self.load_eps(outdir)
        self.assign_eps(eps)

    def load_eps(
            self,
            outdir: os.PathLike
    ) -> dict[str, dict[str, Tensor]]:
        netdir = Path(outdir).joinpath('networks')
        fxeps = netdir.joinpath('xeps.npy')
        fveps = netdir.joinpath('veps.npy')
        xe = torch.from_numpy(np.load(fxeps))
        ve = torch.from_numpy(np.load(fveps))
        xeps = {}
        veps = {}
        rg = (not self.config.eps_fixed)
        for lf in range(self.config.nleapfrog):
            xeps[str(lf)] = torch.tensor(xe[lf], requires_grad=rg)
            veps[str(lf)] = torch.tensor(ve[lf], requires_grad=rg)

        return {'xeps': xeps, 'veps': veps}

    def restore_eps(self, outdir: os.PathLike) -> None:
        eps = self.load_eps(Path(outdir).joinpath('networks'))
        self.assign_eps(eps)

    def assign_eps(
            self,
            eps: float | tuple[float, float] | dict[str, dict[str, Tensor]]
    ) -> None:
        if isinstance(eps, dict):
            xe = eps['xeps']
            ve = eps['veps']
        elif isinstance(eps, tuple):
            xe_, ve_ = eps
            xe = {str(n): xe_ for n in range(self.config.nleapfrog)}
            ve = {str(n): ve_ for n in range(self.config.nleapfrog)}
        elif isinstance(eps, float):
            xe = {str(n): eps for n in range(self.config.nleapfrog)}
            ve = {str(n): eps for n in range(self.config.nleapfrog)}
        else:
            raise TypeError

        rg = (not self.config.eps_fixed)
        self.xeps = nn.ParameterList()
        self.veps = nn.ParameterList()
        for lf in range(self.config.nleapfrog):
            self.xeps.append(nn.parameter.Parameter(
                data=torch.tensor(xe[str(lf)]), requires_grad=rg
            ))
            self.veps.append(nn.parameter.Parameter(
                data=torch.tensor(ve[str(lf)]), requires_grad=rg
            ))

    def forward(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, dict]:
        x, beta = inputs
        x = x.to(self._device)  # .to(self._dtype)
        beta = beta.to(self._device)  # .to(self._dtype)
        return (
            self.apply_transition_fb(inputs)
            if self.config.merge_directions
            else self.apply_transition(inputs)
        )

    def flatten(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], -1)

    def apply_transition_hmc(
            self,
            inputs: tuple[Tensor, Tensor],
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> tuple[Tensor, dict]:
        data = self.generate_proposal_hmc(inputs, eps=eps, nleapfrog=nleapfrog)
        ma_, mr_ = self._get_accept_masks(data['metrics']['acc'])
        ma_ = ma_.to(inputs[0].device)
        mr_ = mr_.to(inputs[0].device)
        ma = ma_[:, None]
        mr = mr_[:, None]
        xinit = data['init'].x.flatten(1)
        vinit = data['init'].v.flatten(1)
        xprop = data['proposed'].x.flatten(1)
        vprop = data['proposed'].v.flatten(1)
        vout = ma * vprop + mr * vinit
        xout = ma * xprop + mr * xinit
        state_out = State(x=xout, v=vout, beta=data['init'].beta)
        mc_states = MonteCarloStates(init=data['init'],
                                     proposed=data['proposed'],
                                     out=state_out)
        data['metrics'].update({
            'acc_mask': ma_,
            'mc_states': mc_states,
        })
        return xout, data['metrics']

    def apply_transition_fb(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, dict]:
        data = self.generate_proposal_fb(inputs)
        ma_, mr_ = self._get_accept_masks(data['metrics']['acc'])
        ma_ = ma_.to(inputs[0].device)
        mr_ = mr_.to(inputs[0].device)
        ma = ma_[:, None]
        mr = mr_[:, None]
        # NOTE: We construct output states by combining
        #   output = (acc_mask * proposed) + (reject_mask * init)
        # v_out = torch.where(
        #     ma.to(torch.bool),
        #     self.flatten(data['proposed'].v),
        #     self.flatten(data['init'].v)
        # )
        xinit = data['init'].x.flatten(1)
        vinit = data['init'].v.flatten(1)
        xprop = data['proposed'].x.flatten(1)
        vprop = data['proposed'].v.flatten(1)
        vout = ma * vprop + mr * vinit
        xout = ma * xprop + mr * xinit
        # x_out = torch.where(
        #     ma.to(torch.bool),
        #     self.flatten(data['proposed'].x),
        #     self.flatten(data['init'].x)
        # )
        # NOTE: sumlogdet = (accept * logdet) + (reject * 0)
        sumlogdet = ma_ * data['metrics']['sumlogdet']
        state_out = State(x=xout, v=vout, beta=data['init'].beta)
        mc_states = MonteCarloStates(
            init=data['init'],
            proposed=data['proposed'],
            out=state_out
        )
        data['metrics'].update({
            'beta': data['init'].beta,
            'acc_mask': ma_,
            'sumlogdet': sumlogdet,
            'mc_states': mc_states,
        })
        return xout, data['metrics']

    def apply_transition(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        if torch.rand(1) > 0.5:
            data = self.generate_proposal(inputs, forward=True)
        else:
            data = self.generate_proposal(inputs, forward=False)
        # metrics = data['metrics']
        ma_, mr_ = self._get_accept_masks(data['metrics']['acc'])
        if torch.cuda.is_available():
            ma_ = ma_.cuda()
            mr_ = mr_.cuda()
        ma = ma_[:, None]
        mr = mr_[:, None]
        xinit = data['init'].x.flatten(1)
        vinit = data['init'].v.flatten(1)
        xprop = data['proposed'].x.flatten(1)
        vprop = data['proposed'].v.flatten(1)
        v_out = ma * vprop + mr * vinit
        x_out = ma * xprop + mr * xinit
        # v_out = ma * data['proposed'].v + mr * data['init'].v
        # x_out = ma * data['proposed'].x + mr * xinit
        sumlogdet = ma_ * data['metrics']['sumlogdet']
        state_out = State(x=x_out, v=v_out, beta=beta)
        mc_states = MonteCarloStates(
            init=data['init'],
            proposed=data['proposed'],
            out=state_out
        )
        data['metrics'].update({
            'beta': beta,
            'acc': data['metrics']['acc'],
            'acc_mask': ma_,
            'sumlogdet': sumlogdet,
            'mc_states': mc_states,
        })
        return x_out, data['metrics']

    def apply_transition_both(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, dict]:
        x, beta = inputs
        fwd = self.generate_proposal(inputs, forward=True)
        bwd = self.generate_proposal(inputs, forward=False)
        mf_, mb_ = self._get_direction_masks(batch_size=x.shape[0])
        if torch.cuda.is_available():
            mf_ = mf_.cuda()
            mb_ = mb_.cuda()
        mf = mf_[:, None]
        mb = mb_[:, None]
        v_init = mf * fwd['init'].v + mb * bwd['init'].v
        # -------------------------------------------------------------------
        # NOTE:
        #   To get the output, we combine forward and backward proposals
        #   where mf, mb are (forward/backward) masks and primed vars are
        #   proposals, e.g. xf' is the proposal from obtained from xf
        #   by running the dynamics in the forward direction. Explicitly,
        #                xp = mf * xf' + mb * xb',
        #                vp = mf * vf' + mb * vb'
        # -------------------------------------------------------------------
        xp = mf * fwd['proposed'].x + mb * bwd['proposed'].x
        vp = mf * fwd['proposed'].v + mb * bwd['proposed'].v
        mfwd = fwd['metrics']
        mbwd = bwd['metrics']
        logdetp = mf_ * mfwd['sumlogdet'] + mb_ * mbwd['sumlogdet']
        acc = mf_ * mfwd['acc'] + mb_ * mbwd['acc']
        ma_, mr_ = self._get_accept_masks(acc)
        if torch.cuda.is_available():
            ma_ = ma_.cuda()
            mr_ = mr_.cuda()
        ma = ma_[:, None]
        mr = mr_[:, None]
        v_out = ma * vp + mr * v_init
        x_out = ma * xp + mr * x
        logdet = ma_ * logdetp  # NOTE: + mr_ * logdet_init = 0

        state_init = State(x=x, v=v_init, beta=beta)
        state_prop = State(x=xp, v=vp, beta=beta)
        state_out = State(x=x_out, v=v_out, beta=beta)
        mc_states = MonteCarloStates(init=state_init,
                                     proposed=state_prop,
                                     out=state_out)
        metrics = {}
        for (key, vf), (_, vb) in zip(mfwd.items(), mbwd.items()):
            try:
                vprop = ma_ * (mf_ * vf + mb_ * vb)
            except RuntimeError:
                vprop = ma * (mf * vf + mb * vb)
            metrics[key] = vprop

        metrics.update({
            'acc': acc,
            'acc_mask': ma_,
            'sumlogdet': logdet,
            'mc_states': mc_states,
        })
        return x_out, metrics

    def random_state(self, beta: float) -> State:
        x = self.g.random(list(self.xshape)).to(self.device)
        v = self.g.random_momentum(list(self.xshape)).to(x.device)
        # with torch.no_grad():
        #     x = self.g.projectSU(x)
        #     v = self.g.projectTAH(v)
        return State(x=x, v=v, beta=torch.tensor(beta).to(self.device))

    def test_reversibility(self) -> dict[str, Tensor]:
        state = self.random_state(beta=1.)
        state_fwd, _ = self.transition_kernel(state, forward=True)
        state_, _ = self.transition_kernel(state_fwd, forward=False)
        dx = torch.abs(state.x - state_.x)
        dv = torch.abs(state.v - state_.v)
        return {'dx': dx.detach().numpy(), 'dv': dv.detach().numpy()}

    def generate_proposal_hmc(
            self,
            inputs: tuple[Tensor, Tensor],
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> dict:
        x, beta = inputs
        xshape = [x.shape[0], *self.xshape[1:]]
        v = self.g.random_momentum(xshape).to(x.device)

        init = State(x=x, v=v, beta=beta)
        proposed, metrics = self.transition_kernel_hmc(
            init,
            eps=eps,
            nleapfrog=nleapfrog
        )
        return {'init': init, 'proposed': proposed, 'metrics': metrics}

    def generate_proposal_fb(
            self,
            inputs: tuple[Tensor, Tensor],
    ) -> dict:
        x, beta = inputs
        v = self.g.random_momentum(
            (x.shape[0], *self.xshape[1:])
        ).to(x.device)
        init = State(x=x, v=v, beta=beta)
        proposed, metrics = self.transition_kernel_fb(
            State(x, v, beta)
        )
        return {'init': init, 'proposed': proposed, 'metrics': metrics}

    def generate_proposal(
            self,
            inputs: tuple[Tensor, Tensor],
            forward: bool,
    ) -> dict:
        x, beta = inputs
        xshape = [x.shape[0], *self.xshape[1:]]
        v = self.g.random_momentum(xshape).to(x.device)
        state_init = State(x=x, v=v, beta=beta)
        state_prop, metrics = self.transition_kernel(state_init, forward)
        return {'init': state_init, 'proposed': state_prop, 'metrics': metrics}

    def get_metrics(
            self,
            state: State,
            logdet: Tensor,
            step: Optional[int] = None,
            extras: Optional[dict[str, Tensor]] = None,
    ) -> dict:
        energy = self.hamiltonian(state)
        logprob = energy - logdet
        metrics = {
            'energy': energy,
            'logprob': logprob,
            'logdet': logdet,
        }
        if extras is not None:
            metrics.update(extras)
        if step is not None:
            metrics.update({
                'xeps': self.xeps[step],
                'veps': self.veps[step]
            })
        return metrics

    def update_history(
            self,
            metrics: dict,
            history: dict,
    ):
        for key, val in metrics.items():
            try:
                history[key].append(val)
            except KeyError:
                history[key] = [val]
        return history

    def leapfrog_hmc(
            self,
            state: State,
            eps: Optional[float] = None,
    ) -> State:
        eps = self.config.eps if eps is None else eps
        x_ = state.x.reshape_as(state.v)
        force1 = self.grad_potential(x_, state.beta)       # f = dU / dx
        v1 = state.v - 0.5 * eps * force1                  # v -= ½ veps * f
        xp = self.g.update_gauge(x_, eps * v1)             # x += eps * V
        # xp = x_ + eps * v1                               # x += xeps * v
        force2 = self.grad_potential(xp, state.beta)       # calc force, again
        v2 = v1 - 0.5 * eps * force2                       # v -= ½ veps * f
        return State(x=xp, v=v2, beta=state.beta)          # output: (x', v')

    def transition_kernel_hmc(
            self,
            state: State,
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> tuple[State, dict]:
        state_ = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = torch.zeros(state.x.shape[0],
                                # dtype=state.x.real.dtype,
                                device=state.x.device)
        history = {}
        if self.config.verbose:
            history = self.update_history(
                self.get_metrics(state_, sumlogdet),
                history=history,
            )
        eps = self.config.eps_hmc if eps is None else eps
        nlf = (
            self.config.nleapfrog if not self.config.merge_directions
            else 2 * self.config.nleapfrog
        )
        if eps is None:
            eps = 1. / nlf
            log.warning(f'Using eps = 1 / {nlf} = {eps:.3f}')
        assert nlf <= 2 * self.config.nleapfrog
        nleapfrog = nlf if nleapfrog is None else nleapfrog
        for _ in range(nleapfrog):
            state_ = self.leapfrog_hmc(state_, eps=eps)
            if self.config.verbose:
                history = self.update_history(
                    self.get_metrics(state_, sumlogdet),
                    history=history,
                )
        acc = self.compute_accept_prob(state, state_, sumlogdet)
        history.update({'acc': acc, 'sumlogdet': sumlogdet})
        if self.config.verbose:
            for key, val in history.items():
                if isinstance(val, list) and isinstance(val[0], Tensor):
                    history[key] = torch.stack(val)
        return state_, history

    def transition_kernel_fb(
            self,
            state: State,
    ) -> tuple[State, dict]:
        sumlogdet = torch.zeros(
            (state.x.shape[0],),
            device=state.x.device,
        )
        sldf = torch.zeros_like(sumlogdet)
        sldb = torch.zeros_like(sumlogdet)
        state_ = State(x=state.x, v=state.v, beta=state.beta)
        history = {}
        if self.config.verbose:
            extras = {
                'sldf': sldf,
                'sldb': sldb,
                # 'sldfb': sldf + sldb,
                'sld': sumlogdet,
            }
            history = self.update_history(
                self.get_metrics(state_, sumlogdet, step=0, extras=extras),
                history=history,
            )
        # Forward
        for step in range(self.config.nleapfrog):
            state_, logdet = self._forward_lf(step, state_)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                # sldf += logdet
                sldf = sldf + logdet
                extras = {
                    'sldf': sldf,
                    'sldb': sldb,
                    # 'sldfb': sldf + sldb,
                    'sld': sumlogdet,
                }
                metrics = self.get_metrics(
                    state_,
                    sumlogdet,
                    step=step,
                    extras=extras
                )
                history = self.update_history(metrics=metrics, history=history)
        # Flip momentum
        # m1 = -1.0 * torch.ones_like(state_.v)
        state_ = State(state_.x, (-state_.v), state_.beta)
        # Backward
        for step in range(self.config.nleapfrog):
            state_, logdet = self._backward_lf(step, state_)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                # sldb += loget
                sldb = sldb + logdet
                extras = {
                    'sldf': torch.zeros_like(sldb),
                    'sldb': sldb,
                    # 'sldfb': sldf + sldb,
                    'sld': sumlogdet,
                }
                # Reverse step count to correctly order metrics at each step
                metrics = self.get_metrics(
                    state_,
                    sumlogdet,
                    step=(self.config.nleapfrog - step - 1),
                    extras=extras,
                )
                history = self.update_history(metrics=metrics, history=history)
        acc = self.compute_accept_prob(state, state_, sumlogdet)
        history.update({'acc': acc, 'sumlogdet': sumlogdet})
        if self.config.verbose:
            for key, val in history.items():
                if isinstance(val, list) and isinstance(val[0], Tensor):
                    history[key] = torch.stack(val)
        return state_, history

    def transition_kernel(
            self,
            state: State,
            forward: bool
    ) -> tuple[State, dict]:
        """Implements the transition kernel."""
        lf_fn = self._forward_lf if forward else self._backward_lf
        # Copy initial state into proposed state
        sinit = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = torch.zeros(state.x.shape[0],
                                # dtype=state.x.dtype,
                                device=state.x.device)
        history = {}
        if self.config.verbose:
            metrics = self.get_metrics(state, sumlogdet)
            history = self.update_history(metrics, history=history)
        for step in range(self.config.nleapfrog):
            state, logdet = lf_fn(step, state)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                metrics = self.get_metrics(state, sumlogdet, step=step)
                history = self.update_history(metrics, history=history)
        acc = self.compute_accept_prob(
            state_init=state,
            state_prop=sinit,
            sumlogdet=sumlogdet,
        )
        history.update({'acc': acc, 'sumlogdet': sumlogdet})
        if self.config.verbose:
            for key, val in history.items():
                if isinstance(val, list) and isinstance(val[0], Tensor):
                    history[key] = torch.stack(val)
        return state, history

    def compute_accept_prob(
            self,
            state_init: State,
            state_prop: State,
            sumlogdet: Tensor,
    ) -> Tensor:
        h_init = self.hamiltonian(state_init)
        h_prop = self.hamiltonian(state_prop)
        if sumlogdet.is_complex():
            log.warning('Complex sumlogdet! Taking norm...?')
            sumlogdet = sumlogdet.norm()
        dh = h_init - h_prop + sumlogdet
        return torch.exp(
            torch.minimum(dh, torch.zeros_like(dh, device=dh.device))
        ).to(state_init.x.device)

    @staticmethod
    def _get_accept_masks(px: Tensor) -> tuple[Tensor, Tensor]:
        acc = (
            px > torch.rand_like(px).to(px.device)
        ).to(torch.float)
        rej = torch.ones_like(acc) - acc
        return acc, rej

    @staticmethod
    def _get_direction_masks(batch_size: int) -> tuple[Tensor, Tensor]:
        """Returns (forward_mask, backward_mask)."""
        fwd = (torch.rand(batch_size) > 0.5)  # .to(torch.float)
        bwd = torch.ones_like(fwd) - fwd
        return fwd, bwd

    def _get_mask(self, step: int) -> tuple[Tensor, Tensor]:
        m = self.masks[step]
        mb = torch.ones_like(m) - m
        return m, mb

    def _build_masks(self):
        """Construct different binary masks for different time steps."""
        masks = []
        for _ in range(self.config.nleapfrog):
            _idx = np.arange(self.xdim)
            idx = np.random.permutation(_idx)[:self.xdim // 2]
            mask = np.zeros((self.xdim,), dtype=np.float32)
            mask[idx] = 1.
            masks.append(torch.from_numpy(mask[None, :]))
        return masks

    def _get_vnet(self, step: int) -> nn.Module | Callable:
        """Returns momentum network to be used for updating v."""
        if not self._networks_built:
            return self.vnet
        # vnet = self.networks.get_submodule('vnet')
        if self.config.use_separate_networks:
            return self.vnet.get_submodule(str(step))
        return self.vnet

    def _get_xnet(
            self,
            step: int,
            first: bool = False
    ) -> nn.Module | Callable:
        """Returns position network to be used for updating x."""
        # xnet = self.networks.get_submodule('xnet')
        if not self._networks_built:
            return self.xnet
        if self.config.use_separate_networks:
            xnet = self.xnet.get_submodule(str(step))
            if self.config.use_split_xnets:
                return xnet.get_submodule('first') if first else xnet.get_submodule('second')
            return xnet
        return self.xnet

    def _stack_as_xy(self, x: Tensor) -> Tensor:
        """Returns -pi < x <= pi stacked as [cos(x), sin(x)]"""
        # TODO: Deal with ConvNet here
        return torch.stack([x.cos(), x.sin()], dim=-1).to(self.device)

    def _call_vnet(
            self,
            step: int,
            inputs: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Call the momentum update network used to update v.
        Args:
            inputs: (x, force) tuple
        Returns:
            s, t, q: Scaling, Translation, and Transformation functions
        """
        x, force = inputs
        if self.config.group == 'SU3':
            x = self.group_to_vec(x)
            force = self.group_to_vec(force)
        vnet = self._get_vnet(step)
        assert callable(vnet)
        return vnet((x, force))

    def _call_xnet(
            self,
            step: int,
            inputs: tuple[Tensor, Tensor],
            first: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Call the position network used to update x.
        Args:
            inputs: (m * x, v) tuple, where (m * x) is a masking operation.
        Returns:
            s, t, q: Scaling, Translation, and Transformation functions
        """
        x, v = inputs
        xnet = self._get_xnet(step, first)
        assert callable(xnet)
        if self.config.group == 'U1':
            x = self.g.group_to_vec(x)
        if self.config.group == 'SU3':
            # x = self.unflatten(x)
            x = torch.stack([x.real, x.imag], 1)
            v = torch.stack([v.real, v.imag], 1)
            # v = self.group_to_vec(v)
            # x = self.group_to_vec(x)
        return xnet((x, v))

    def _forward_lf(self, step: int, state: State) -> tuple[State, Tensor]:
        """Complete update (leapfrog step) in the forward direction.
        Explicitly:
            1. Update momentum (v' <-- v)
            2. Update half of position (x' <-- m * x)
            3. Update other half of position (x'' <-- (1 - m) * x)
            4. Update momentum (v'' <-- v')
        """
        m, mb = self._get_mask(step)
        m, mb = m.to(self.device), mb.to(self.device)
        # sumlogdet = torch.zeros(state.x.shape[0], device=self.device)
        state, logdet = self._update_v_fwd(step, state)
        sumlogdet = logdet
        state, logdet = self._update_x_fwd(step, state, m, first=True)
        sumlogdet = sumlogdet + logdet
        state, logdet = self._update_x_fwd(step, state, mb, first=False)
        sumlogdet = sumlogdet + logdet
        state, logdet = self._update_v_fwd(step, state)
        sumlogdet = sumlogdet + logdet
        return state, sumlogdet

    def _backward_lf(self, step: int, state: State) -> tuple[State, Tensor]:
        """Complete update (leapfrog step) in the backward direction
        Explicitly:
            1. Update momentum (v' <-- v)
            2. Update half of position       (x'  <-- (1 - m) * x)
            3. Update other half of position (x'' <-- m * x)
            4. Update momentum (v'' <-- v)
        """
        # NOTE: Reverse the step count, i.e. count from end of trajectory
        step_r = self.config.nleapfrog - step - 1
        m, mb = self._get_mask(step_r)
        m, mb = m.to(self.device), mb.to(self.device)
        state, logdet = self._update_v_bwd(step_r, state)
        sumlogdet = logdet
        state, logdet = self._update_x_bwd(step_r, state, mb, first=False)
        sumlogdet = sumlogdet + logdet
        state, logdet = self._update_x_bwd(step_r, state, m, first=True)
        sumlogdet = sumlogdet + logdet
        state, logdet = self._update_v_bwd(step_r, state)
        sumlogdet = sumlogdet + logdet
        return state, sumlogdet

    def unflatten(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], *self.xshape[1:])

    def group_to_vec(self, x: Tensor) -> Tensor:
        """For SU(3), returns 8-component real-valued vector from SU(3) mtx"""
        return self.g.group_to_vec(self.unflatten(x))

    def vec_to_group(self, x: Tensor) -> Tensor:
        """Inverts `group_to_vec` operation."""
        x = self.unflatten(x)
        if self.config.group == 'SU3':
            return self.g.vec_to_group(x)
        return torch.complex(x[..., 0], x[..., 1])

    def _update_v_fwd_hmc(self, step: int, state: State) -> Tensor:
        eps = sigmoid(self.veps[step].log())
        x_ = state.x.reshape_as(state.v)
        force1 = self.grad_potential(x_, state.beta)       # f = dU / dx
        return state.v - 0.5 * eps * force1                # v -= ½ veps * f

    def _update_v_bwd_hmc(self, step: int, state: State) -> Tensor:
        eps = sigmoid(self.veps[step].log())
        x_ = state.x.reshape_as(state.v)
        force1 = self.grad_potential(x_, state.beta)       # f = dU / dx
        return state.v + 0.5 * eps * force1                # v -= ½ veps * f

    def _update_x_fwd_hmc(self, step: int, state: State) -> Tensor:
        eps = sigmoid(self.xeps[step].log())
        x = self.g.update_gauge(state.x, eps * state.v)   # x += eps * V
        return x

    def _update_x_bwd_hmc(self, step: int, state: State) -> Tensor:
        eps = sigmoid(self.xeps[step].log())
        x = self.g.update_gauge(state.x, -eps * state.v)
        return x

    def _update_v_fwd(self, step: int, state: State) -> tuple[State, Tensor]:
        """Single v update in the forward direction"""
        assert self.config.group in ['U1', 'SU3']
        force = self.grad_potential(state.x, state.beta)
        eps = sigmoid(self.veps[step].log())
        s, t, q = self._call_vnet(step, (state.x, force))
        logjac = eps * s / 2.
        logdet = self.flatten(logjac).sum(1)
        exp_s = logjac.exp().reshape_as(state.v)
        exp_q = (eps * q).exp().reshape_as(state.v)
        t = t.reshape_as(state.v)
        force = force.reshape_as(state.v)
        force_new = (force * exp_q + t)
        vf = (exp_s * state.v) - 0.5 * eps * force_new
        return State(state.x, vf, state.beta), logdet.real

    def _update_v_bwd(self, step: int, state: State) -> tuple[State, Tensor]:
        """Single v update in the backward direction"""
        assert self.config.group in ['U1', 'SU3']
        # vNet: (x, force) --> (s, t, q)
        force = self.grad_potential(state.x, state.beta)
        eps = sigmoid(self.veps[step].log())  # .clamp_min_(0.)
        s, t, q = self._call_vnet(step, (state.x, force))
        logjac = (-eps * s / 2.)
        logdet = self.flatten(logjac).sum(1)
        exp_s = logjac.exp().reshape_as(state.v)
        exp_q = (eps * q).exp().reshape_as(state.v)
        t = t.reshape_as(state.v)
        force = force.reshape_as(state.v)
        force_new = (force * exp_q + t)
        vb = exp_s * (state.v + 0.5 * eps * force_new)
        return State(state.x, vb, state.beta), logdet.real

    def _update_v_fwd_deprecated(self, step: int, state: State) -> tuple[State, Tensor]:
        """Single v update in the forward direction"""
        # assert self.config.group in ['U1', 'SU3']
        # force = self.grad_potential(state.x, state.beta)
        # eps = sigmoid(self.veps[step].log())
        # s, t, q = self._call_vnet(step, (state.x, force))
        # logjac = eps * s / 2.
        # logdet = self.flatten(logjac).sum(1)
        # exp_s = logjac.exp().reshape_as(state.v)
        # exp_q = (eps * q).exp().reshape_as(state.v)
        # t = t.reshape_as(state.v)
        # force = force.reshape_as(state.v)
        # force_new = (force * exp_q + t)
        # vf = (exp_s * state.v) - 0.5 * eps * force_new
        # if self.config.group == 'U1':
        #     # s = self.g.projectTAH(self.unflatten(s))
        #     # t = self.g.projectTAH(self.unflatten(t))
        #     # q = self.g.projectTAH(self.unflatten(q))
        #     # NOTE: For x ∈ U(1), we define ProjectTAH(x) := x
        #
        #     # logjac = eps * s / 2.  # jacobian factor, also used in exp_s below
        #     # logdet = self.flatten(logjac).sum(1)
        #     # force = force.reshape_as(state.v)
        #     # exp_s = logjac.exp().reshape_as(state.v)
        #     # exp_q = (eps * q).exp().reshape_as(state.v)
        #     # t = t.reshape_as(state.v)
        #     # exp_s = self.g.projectTAH(logjac.exp().reshape_as(state.v))
        #     # exp_q = self.g.projectTAH((eps * q).exp().reshape_as(state.v))
        #     # t = self.g.projectTAH(t.reshape_as(state.v))
        #     # force_new = self.g.projectTAH(force * exp_q + t)
        #     # vf = (exp_s * state.v) - 0.5 * eps * force_new
        #     # vf = (exp_s * state.v) - (0.5 * eps * (force * exp_q + t))
        # elif self.config.group == 'SU3':
        #     # eps = self.veps[step]
        #     # vf = self._update_v_fwd_hmc(step, state)
        #     # logdet = torch.zeros(vf.shape[0]).to(vf.device)
        #     # logjac = eps * s / 2.
        #     # logdet = self.flatten(logjac).sum(1)
        #     # exp_s = logjac.exp().reshape_as(state.v)
        #     # t = t.reshape_as(state.v)
        #     # force = force.reshape_as(state.v)
        #     # exp_q = (eps * q).exp().reshape_as(state.v)
        #     # force_ = exp_q * force + t
        #     # v_ = exp_s * state.v
        #     # vf = v_ - 0.5 * eps * force_
        #     # vf = (exp_s * state.v) - 0.5 * eps * force
        #     # vf = self.g.projectTAH(vf)
        # else:
        #     raise ValueError(f'Unexpected group: {self.config.group}')
        return State(state.x, vf, state.beta), logdet.real

    def _update_v_bwd_deprecated(self, step: int, state: State) -> tuple[State, Tensor]:
        """Single v update in the backward direction"""
        # vNet: (x, force) --> (s, t, q)
        force = self.grad_potential(state.x, state.beta)
        eps = sigmoid(self.veps[step].log())  # .clamp_min_(0.)
        if self.config.group == 'U1':
            s, t, q = self._call_vnet(step, (state.x, force))
            # s = self.g.projectTAH(self.unflatten(s))
            # t = self.g.projectTAH(self.unflatten(t))
            # q = self.g.projectTAH(self.unflatten(q))
            logjac = (-eps * s / 2.)  # jacobian factor, also used in exp_s below
            logdet = self.flatten(logjac).sum(1)
            exp_s = self.g.projectTAH(logjac.exp().reshape_as(state.v))
            exp_q = self.g.projectTAH((eps * q).exp().reshape_as(state.v))
            t = self.g.projectTAH(t.reshape_as(state.v))
            force = force.reshape_as(state.v)
            force_new = self.g.projectTAH(force * exp_q + t)
            vb = exp_s * (state.v + 0.5 * eps * force_new)
            # vb = exp_s * (state.v + 0.5 * eps * (force * exp_q + t))
        elif self.config.group == 'SU3':
            # eps = self.veps[step]
            # vb = self.g.projectTAH(vb)
            # vb = self._update_v_bwd_hmc(step, state)
            # logdet = torch.zeros(vb.shape[0]).to(vb.device)
            s, t, q = self._call_vnet(step, (state.x, force))
            logjac = (-eps * s / 2.)
            logdet = self.flatten(logjac).sum(1)
            exp_s = logjac.exp().reshape_as(state.v)
            exp_q = (eps * q).exp().reshape_as(state.v)
            force = force.reshape_as(state.v)
            t = t.reshape_as(state.v)
            vb = exp_s * (state.v + 0.5 * eps * (force * exp_q + t))
        else:
            raise ValueError(f'Unexpected group: {self.config.group}')
        return State(state.x, vb, state.beta), logdet.real

    def _update_x_fwd(
            self,
            step: int,
            state: State,
            m: Tensor,
            first: bool,
    ) -> tuple[State, Tensor]:
        """Single x update in the forward direction."""
        eps = sigmoid(self.xeps[step].log())
        m = self.unflatten(m)
        mb = (torch.ones_like(m) - m).to(self.device)
        xm_init = m * state.x
        if self.config.group == 'U1':
            x = self.flatten(state.x)
            v = self.flatten(state.v)
            s, t, q = self._call_xnet(step, (xm_init, state.v), first=first)
            s = eps * s
            q = eps * q
            exp_s = s.exp()
            exp_q = q.exp()
            if self.config.use_ncp:
                halfx = (x / 2.).flatten(1)
                _x = 2. * (halfx.tan() * exp_s).atan()
                xp = self.unflatten(_x + eps * (v * exp_q + t))
                xf = xm_init + (mb * xp)
                cterm = (halfx.cos()) ** 2
                sterm = (exp_s * halfx.sin()) ** 2
                logdet_ = (exp_s / (cterm + sterm)).log()
                logdet = (mb.flatten(1) * logdet_).sum(dim=1)
            else:
                xp = x * exp_s + eps * (v * exp_q + t)
                xf = xm_init + (mb * xp)
                logdet = (mb * s).sum(dim=1)
            xf = self.g.compat_proj(xf)
        elif self.config.group == 'SU3':
            # xf = xm_init + mb * self.g.update_gauge(state.x, eps * state.v)
            # eps = self.xeps[step]
            # xf = xm_init + mb * self.g.update_gauge(state.x, eps * state.v)
            xf = xm_init + self.g.update_gauge(mb * state.x, eps * state.v)
            logdet = torch.zeros(state.x.shape[0]).to(state.x.device)
        else:
            raise ValueError('Unexpected value for self.config.group')
        return State(x=xf, v=state.v, beta=state.beta), logdet.real

    def _update_x_bwd(
            self,
            step: int,
            state: State,
            m: Tensor,
            first: bool,
    ) -> tuple[State, Tensor]:
        """Update the position in the backward direction."""
        # assert isinstance(self.xeps, nn.ParameterList)
        eps = sigmoid(self.xeps[step].log())
        m = self.unflatten(m)
        mb = (torch.ones_like(m) - m).to(self.device)
        xm_init = m * state.x
        if self.config.group == 'U1':
            x = state.x.reshape((state.x.shape[0], -1))
            v = state.v.reshape_as(x)
            inputs = (xm_init, state.v)
            s, t, q = self._call_xnet(step, inputs, first=first)
            s = (-eps) * s
            q = eps * q
            exp_s = s.exp()
            exp_q = q.exp()
            if self.config.use_ncp:
                halfx = x / 2.
                halfx_scale = exp_s * halfx.tan()
                x1 = 2. * halfx_scale.atan()
                x2 = exp_s * eps * (v * exp_q + t)
                xnew = self.unflatten(x1 - x2)
                xb = xm_init + (mb * xnew)
                cterm = halfx.cos() ** 2
                sterm = (exp_s * halfx.sin()) ** 2
                logdet_ = (exp_s / (cterm + sterm)).log()
                logdet = (mb.flatten(1) * logdet_).sum(dim=1)
            else:
                xnew = exp_s * (state.x - eps * (state.v * exp_q + t))
                xb = xm_init + (mb * xnew)
                logdet = (mb * s).sum(dim=1)
            xb = self.g.compat_proj(xb)
        elif self.config.group == 'SU3':
            # eps = self.xeps[step]
            # xb = xm_init + self.g.update_gauge(mb * state.x, - eps * state.v)
            xb = xm_init + self.g.update_gauge(mb * state.x, - eps * state.v)
            # x_ = state.x.reshape_as(state.v)
            # xb = xm_init + mb * self.g.update_gauge(state.x, - eps * state.v)
            logdet = torch.zeros(state.x.shape[0]).to(state.x.device)
        else:
            raise ValueError('Unexpected value for `self.g`')
        return State(x=xb, v=state.v, beta=state.beta), logdet.real

    def hamiltonian(self, state: State) -> Tensor:
        """Returns the total energy H = KE + PE."""
        kinetic = self.kinetic_energy(state.v)
        potential = self.potential_energy(state.x, state.beta)
        return kinetic + potential

    def kinetic_energy(self, v: Tensor) -> Tensor:
        """Returns the kinetic energy, KE = 0.5 * v ** 2."""
        return self.g.kinetic_energy(v)

    def potential_energy(self, x: Tensor, beta: Tensor):
        """Returns the potential energy, PE = beta * action(x)."""
        return self.potential_fn(x, beta)

    def grad_potential(
            self,
            x: Tensor,
            beta: Tensor,
    ) -> Tensor:
        """Compute the gradient of the potential function."""
        return self.lattice.grad_action(x, beta)

    @staticmethod
    def complexify(
            x: Tensor,
            dim: int = 1,
    ) -> Tensor:
        """Complexify real-valued tensor.
        Example:
            >>> x = torch.rand((N, 2, C, H, W))
            >>> xc = complexify(x, dim=1)
            >>> assert xc == torch.complex(x[:, 0, ...], x[:, 1, ...])
            >>> y = torch.rand((N, C, H, W, 2))
            >>> yT = y.transpose(0, 4)
            >>> yT.shape
            (2, C, H, W, N)
            >>> yr, yi = yT
            >>> yr.shape
            (C, H, W, N)
            >>> yrT = yr.transpose(0, dim-1)
            >>> yrT.shape
            (N, C, H, W)
            >>> yc = complexify(y, dim=4)
            >>> assert yc == torch.complex(y[])
        """
        assert len(x.shape) >= 2
        assert x.shape[dim] == 2
        if dim != 1:
            assert x.shape[dim] == 2
            xT = x.transpose(0, dim)
            xr, xi = xT
            xr = xr.transpose(0, dim-1)
            xi = xi.transpose(0, dim-1)
            return torch.complex(xr, xi)
        if len(x.shape) == 2:
            return torch.complex(x[:, 0], x[:, 1])
        return torch.complex(x[:, 0, ...], x[:, 1, ...])
