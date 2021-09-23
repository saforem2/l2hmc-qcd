"""
dynamics.py

Implements dynamics engine for augmented L2HMC leapfrog integrator.
"""
from __future__ import absolute_import, print_function, division, annotations
from pathlib import Path
from typing import Union
import torch
import json
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from dataclasses import dataclass, asdict, field
from math import pi as PI

from utils.logger import Logger, in_notebook
from network.pytorch.network import (NetworkConfig, LearningRateConfig,
                                     ConvolutionConfig, GaugeNetwork, init_weights)
from lattice.pytorch.lattice import Lattice
# from utils.pytorch.metrics import History, Metrics

from utils.data_containers import History, StepTimer, Metrics

# from utils.data_containers import History, StepTimer, Metrics, innerHistory

# logger = Logger()
if in_notebook:
    import rich.console as console
    logger = console.Console(width=135,
                             log_path=False,
                             log_time_format='[%x %X]',
                             color_system='truecolor')
else:
    logger = Logger()
# if in_notebook:
#     logger = logger.console

TWO_PI = 2. * PI

NetworkOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def rand_unif(shape: Union[tuple, list], a: float, b: float, requires_grad: bool):
    return (a - b) * torch.rand(shape, requires_grad=requires_grad) + b

def random_angle(shape: tuple, requires_grad: bool = True):
    return TWO_PI * torch.rand(shape, requires_grad=requires_grad) - PI


def grab(x: torch.Tensor):
    return x.detach().cpu().numpy()


@dataclass
class NetWeights:
    sx: float = 1.
    tx: float = 1.
    qx: float = 1.
    sv: float = 1.
    tv: float = 1.
    qv: float = 1.

    def asdict(self):
        return asdict(self)


@dataclass
class DynamicsConfig:
    eps: float
    num_steps: int
    x_shape: tuple
    hmc: bool = False
    eps_fixed: bool = False
    aux_weight: float = 0.
    use_mixed_loss: bool = False
    plaq_weight: float = 0.
    charge_weight: float = 0.
    optimizer: str = 'adam'
    clip_val: float = 0.
    separate_networks: bool = True
    net_weights: NetWeights = field(default_factory=NetWeights)

    def __post_init__(self):
        if self.net_weights is None:
            self.net_weights = NetWeights()

        if isinstance(self.net_weights, (tuple, list)):
            self.net_weights = NetWeights(*self.net_weights)

    def to_file(self, f: Union[str, Path]):
        json.dump(asdict(self), Path(f).resolve().open('w'))

    def asdict(self):
        return asdict(self)


@dataclass
class EnergyMetrics:
    H: torch.Tensor
    Hw: torch.Tensor
    logdets: torch.Tensor


@dataclass
class State:
    x: torch.Tensor
    v: torch.Tensor
    beta: float

@dataclass
class MonteCarloStates:
    init: State
    prop: State
    out: State


def to_u1(x: torch.Tensor) -> torch.Tensor:
    return ((x + PI) % TWO_PI) - PI


def project_angle(x):
    """For x in [-4pi, 4pi], returns x in [-pi, pi]."""
    return x - TWO_PI * torch.floor((x + PI) / TWO_PI)



def save_dynamics(dynamics: GaugeDynamics, outdir: Union[str, Path]):
    outfile = Path(outdir).joinpath('gauge_dynamics.pt')
    torch.save(dynamics.state_dict(), outfile)


# Metrics = dict[str, Union[torch.Tensor, np.ndarray]]

class Clamp(torch.autograd.Function):
    def __init__(self, a: float = 0., b: float = 1., **kwargs):
        super().__init__(**kwargs)
        self._a = a
        self._b = b

    def forward(self, ctx, input):
        return input.clamp(min=self._a, max=self._b)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class GaugeDynamics(nn.Module):
    def __init__(
            self,
            dynamics_config: DynamicsConfig,
            network_config: NetworkConfig,
            lr_config: LearningRateConfig,
            conv_config: ConvolutionConfig = None,
    ):
        super().__init__()

        self.config = dynamics_config
        self.net_config = network_config
        self.lr_config = lr_config
        self.conv_config = conv_config
        self.lf = self.config.num_steps
        self.nw = self.config.net_weights

        self._use_ncp = (self.nw.sx != 0)
        self._verbose = True

        self.aux_weight = self.config.aux_weight
        self.plaq_weight = self.config.plaq_weight
        self.charge_weight = self.config.charge_weight
        self.clip_val = self.config.clip_val

        self.lattice_shape = self.config.x_shape
        self.lattice = Lattice(self.lattice_shape)
        self.potential_fn = self.lattice.calc_actions

        self.batch_size = self.lattice_shape[0]
        self.xdim = np.cumprod(self.lattice_shape[1:])[-1]

        self.masks = self._build_masks()

        xeps = []
        veps = []
        # clamp = Clamp(0., 1.)
        rg = (not self.config.eps_fixed)
        for _ in range(self.lf):
            alpha = torch.log(torch.tensor(self.config.eps))
            xe = nn.parameter.Parameter(torch.exp(alpha), requires_grad=rg)
            ve = nn.parameter.Parameter(torch.exp(alpha), requires_grad=rg)
            xeps.append(xe)
            veps.append(ve)
            # xeps.append(clamp.apply(xe))
            # veps.append(clamp.apply(ve))
            # xeps.append(nn.Parameter(torch.tensor(self.config.eps), requires_grad=rg))
            # veps.append(nn.Parameter(torch.tensor(self.config.eps), requires_grad=rg))

        self.xeps = nn.ParameterList(xeps)
        self.veps = nn.ParameterList(veps)

        # -- Build networks ------------------------------------------
        if self.config.hmc:
            self.vnet = [
                lambda x: self._hmc_net(x) for _ in range(self.lf)
            ]
            self.xnet0 = [
                lambda x: self._hmc_net(x) for _ in range(self.lf)
            ]
            self.xnet1 = [
                lambda x: self._hmc_net(x) for _ in range(self.lf)
            ]
        else:
            networks = self.build_networks(self.net_config)
            if self.config.separate_networks:
                self.vnet = networks['v']
                self.xnet0 = networks['x0']
                self.xnet1 = networks['x1']
            else:
                self.vnet = networks['v']
                self.xnet = networks['x']

    def forward(
            self,
            inputs: tuple[torch.Tensor, float]
    ) -> tuple[MonteCarloStates, Metrics]:

        return self.apply_transition(inputs)

    def apply_transition(
            self,
            inputs: tuple[torch.Tensor, float]
    ) -> tuple[MonteCarloStates, Metrics]:
        if self.config.hmc:
            return self._hmc_transition(inputs)

        x, beta = inputs
        sf_init, sf_prop, metricsf = self._transition(inputs, forward=True)
        sb_init, sb_prop, metricsb = self._transition(inputs, forward=False)

        sldf = metricsf.sumlogdet
        pxf = metricsf.accept_prob

        sldb = metricsb.sumlogdet
        pxb = metricsb.accept_prob

        mf_, mb_ = self._get_direction_masks(batch_size=x.shape[0])
        mf = mf_[:, None]
        mb = mb_[:, None]

        v_init = mf * sf_init.v + mb * sb_init.v

        x_prop = mf * sf_prop.x + mb * sb_prop.x
        v_prop = mf * sf_prop.v + mb * sb_prop.v
        sld_prop = mf_ * sldf + mb_ * sldb

        accept_prob = mf_ * pxf + mb_ * pxb

        ma_, mr_ = self._get_accept_masks(accept_prob)
        ma = ma_[:, None]
        mr = mr_[:, None]

        v_out = ma * v_prop + mr * v_init
        x_out = to_u1(ma * x_prop + mr * x)
        sumlogdet = ma_ * sld_prop

        state_init = State(x=x, v=v_init, beta=beta)
        state_prop = State(x=x_prop, v=v_prop, beta=beta)
        state_out = State(x=x_out, v=v_out, beta=beta)
        mc_states = MonteCarloStates(state_init, state_prop, state_out)
        outputs = {
            'accept_prob': accept_prob,
            'accept_mask': ma_,
            'sumlogdet': sumlogdet,
            'sumlogdet_prop': sld_prop,
        }

        return mc_states, outputs

    def _transition(
            self,
            inputs: tuple[torch.Tensor, float],
            forward: bool
    ):
        x, beta = inputs
        v = torch.randn_like(x)
        state = State(x=to_u1(x), v=v, beta=beta)
        state_, metrics = self.transition_kernel(state, forward)

        return state, state_, metrics

    def md_update(
            self,
            inputs: tuple[torch.Tensor, float],
            forward: bool = True,
    ):
        """Perform the molecular dynamics (MD) update w/o accept/reject.

        NOTE: We simulate the dynamics both forward and backward, and use
        sampled Bernoulli masks to compute the actual solutions
        """
        x, beta = inputs
        v = torch.randn_like(x)
        state = State(x=to_u1(x), v=v, beta=beta)
        # sumlogdet = torch.zeros(state.x.shape[0])
        lf_fn = self._forward_lf if forward else self._backward_lf
        for step in range(self.lf):
            state, _ = lf_fn(step, state)
            # sumlogdet = sumlogdet + logdet

        return to_u1(state.x)

    def _transition_kernel(self, state: State, forward: bool):
        """Implements the transition kernel."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        # copy initial state into proposed state `state_`
        state_ = State(x=state.x, v=state.v, beta=state.beta)
        # history = History(['sinQ', 'intQ',
        #                    'plaqs', 'p4x4',
        #                    'logdets', 'sumlogdet',
        #                    'H', 'Hw', 'accept_prob'])
        history = History()
        metrics = Metrics()

        sumlogdet = torch.zeros(state.x.shape[0])

        for step in range(self.lf):
            state_, logdet = lf_fn(step, state_)
            sumlogdet = sumlogdet + logdet
            if self._verbose:
                metrics = self.get_metrics(state_, sumlogdet)
                history.update(metrics.to_dict())

        accept_prob = self.compute_accept_prob(state, state_, sumlogdet)
        history.sumlogdet = sumlogdet
        history.accept_prob = accept_prob

        if self._verbose:
            metrics = self.get_metrics(state_, sumlogdet)
            history.update(metrics.to_dict())

        return State(state_.x, state_.v, state_.beta), history

    def transition_kernel(self, state: State, forward: bool):
        """Transition kernel of the augmented leapfrog integrator."""
        return self._transition_kernel(state, forward)

    def get_network_configs(
            self,
            net_config: NetworkConfig = None,
            # conv_config: ConvolutionConfig = None,
    ) -> dict[str, dict]:
        if net_config is None:
            net_config = self.net_config

        # if conv_config is None:
        #     conv_config =
        # if self.config.use_conv_net and conv_config is not None:
        #     xshape = (*self.lattice_shape[1:], 2)
        xnet_cfg = {
            # 'factor': 2.0,
            'net_config': net_config,
            # 'conv_config': conv_config,
            'xshape': self.lattice_shape,
            'input_shapes': {'x': 2 * self.xdim, 'v': self.xdim}
        }

        vnet_cfg = {
            # 'factor': 1.0,
            'net_config': net_config,
            # 'conv_config': conv_config,
            'xshape': self.lattice_shape,
            'input_shapes': {'x': self.xdim, 'v': self.xdim}
        }

        return {'xnet': xnet_cfg, 'vnet': vnet_cfg}

    def build_networks(
            self,
            net_config: NetworkConfig = None,
            # conv_config: ConvolutionConfig = None,
    ) -> dict[str, torch.nn.ModuleList]:
        cfgs = self.get_network_configs(net_config)
        if self.config.separate_networks:
            vnet = nn.ModuleList([
                GaugeNetwork(**cfgs['vnet']) for _ in range(self.lf)
            ])
            xnet0 = nn.ModuleList([
                GaugeNetwork(**cfgs['xnet']) for _ in range(self.lf)
            ])
            xnet1 = nn.ModuleList([
                GaugeNetwork(**cfgs['xnet']) for _ in range(self.lf)
            ])
            networks = {'v': vnet, 'x0': xnet0, 'x1': xnet1}
        else:
            vnet = GaugeNetwork(**cfgs['vnet'])
            xnet = GaugeNetwork(**cfgs['xnet'])
            networks = {'v': vnet, 'x': xnet}

        return networks

    def compute_accept_prob(
            self,
            state_init: State,
            state_prop: State,
            sumlogdet: torch.Tensor
    ):
        h_init = self.hamiltonian(state_init)
        h_prop = self.hamiltonian(state_prop)
        dh = h_init - h_prop + sumlogdet
        prob = torch.exp(torch.minimum(dh, torch.zeros_like(dh)))
        return prob

    @staticmethod
    def _get_accept_masks(accept_prob: torch.Tensor):
        ma = (accept_prob > torch.rand_like(accept_prob)).to(torch.float)
        mr = 1. - ma

        return ma, mr

    def _get_direction_masks(self, batch_size: int):
        mf = (torch.rand(batch_size) > 0.5).to(torch.float)
        mb = 1. - mf

        return mf, mb

    def calc_losses(
            self,
            states: MonteCarloStates,
            accept_prob: torch.Tensor,
    ):
        q_init = self.lattice.calc_charges(x=states.init.x, use_sin=True)
        q_prop = self.lattice.calc_charges(x=states.prop.x, use_sin=True)
        qloss = (accept_prob * (q_prop - q_init) ** 2) + 1e-4
        qloss = qloss / self.charge_weight

        return qloss.mean()

    def _hmc_transition(self, inputs: tuple[torch.Tensor, float]):
        x, beta = inputs
        sf_init, sf_prop, metricsf = self._transition(inputs, forward=True)
        sb_init, sb_prop, metricsb = self._transition(inputs, forward=False)
        mf_, mb_ = self._get_direction_masks(batch_size=x.shape[0])
        mf = mf_[:, None]
        mb = mb_[:, None]

        v_init = mf * sf_init.v + mb * sb_init.v
        x_prop = mf * sf_prop.x + mb * sb_prop.x
        v_prop = mf * sf_prop.v + mb * sb_prop.v
        sld_prop = mf_ * metricsf.sumlogdet + mb_ * metricsb.sumlogdet
        accept_prob = mf_ * metricsf.accept_prob + mb_ * metricsb.accept_prob
        ma_, mr_ = self._get_accept_masks(accept_prob)
        ma = ma_[:, None]
        mr = mr_[:, None]
        v_out = ma * v_prop + mr * v_init
        x_out = to_u1(ma * x_prop + mr * x)

        sumlogdet = ma_ * sld_prop

        state_init = State(x=x, v=v_init, beta=beta)
        state_prop = State(x=x_prop, v=v_prop, beta=beta)
        state_out = State(x=x_out, v=v_out, beta=beta)
        mc_states = MonteCarloStates(init=state_init,
                                     prop=state_prop,
                                     out=state_out)
        outputs = {
            'accept_prob': accept_prob,
            'accept_mask': ma_,
            'sumlogdet': sumlogdet,
            'sumlogdet_prop': sld_prop,
        }

        return mc_states, outputs

    def _get_mask(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.masks[step]
        return m, 1. - m

    def _build_masks(self):
        """Construct different binary masks for different time steps."""
        masks = []
        for _ in range(self.config.num_steps):
            _idx = np.arange(self.xdim)
            idx = np.random.permutation(_idx)[:self.xdim // 2]
            mask = np.zeros((self.xdim,), dtype=np.float32)
            mask[idx] = 1.
            masks.append(torch.from_numpy(mask[None, :]))
        # masks = []
        # zeros = np.zeros(self.lattice_shape, dtype=np.float32)
        # p = zeros.copy()
        # for idx, _ in np.ndenumerate(zeros):
        #     p[idx] = (sum(idx) % 2 == 0)

        # for i in range(self.lf):
        #     m = p if i % 2 == 0 else (1. - p)
        #     mask = m.reshape((state.x.shape[0], -1))
        #     mask = torch.from_numpy(mask)
        #     masks.append(mask)

        return masks

    def _init_metrics(self, state: State) -> History:
        """Initialize metrics using info from `state`."""
        logdets = torch.zeros(state.x.shape[0])
        accept_prob = torch.zeros(state.x.shape[0])
        sumlogdet = torch.zeros(state.x.shape[0])
        energy = self.hamiltonian(state)
        energy_scaled = energy - logdets
        metrics = self.lattice.calc_observables(x=state.x)
        return History(dict(H=[energy],
                            Hw=[energy_scaled],
                            logdets=[logdets],
                            Qs=[metrics.Qs],
                            Qi=[metrics.Qi],
                            plaqs=[metrics.plaqs],
                            p4x4=[metrics.p4x4],
                            accept_prob=accept_prob,
                            sumlogdet=sumlogdet))

    def get_metrics(self, state: State, logdet: torch.Tensor) -> Metrics:
        energy = self.hamiltonian(state)
        energy_scaled = energy - logdet
        lmetrics = self.lattice.calc_observables(x=state.x)
        emetrics = {'H': energy, 'Hw': energy_scaled, 'logdets': logdet}
        metrics = {**lmetrics, **emetrics}

        return Metrics(**metrics)


    @staticmethod
    def _hmc_net(x: torch.Tensor) -> NetworkOutputs:
        return (torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x))

    def _call_vnet(
        self,
        step: int,
        inputs: tuple[torch.Tensor, torch.Tensor],
        hmc: bool = False,
    ) -> NetworkOutputs:
        """Call the nth momentum network."""
        if hmc or self.config.hmc:
            return self._hmc_net(inputs[0])

        if self.config.separate_networks:
            vnet = self.vnet[step]
        else:
            vnet = self.vnet

        return vnet(inputs)

    def _convert_to_cartesian(
            self,
            x: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        if mask.shape[0] == 2:
            mask, _ = mask

        xcos = mask * torch.cos(x)
        xsin = mask * torch.sin(x)
        # TODO: Deal with ConvNet here
        # if self.config.use_conv_net:
        #     xcos = xcos.reshape(self.lattice_shape)
        #     xsin = xsin.reshape(self.lattice_shape)
        return torch.stack([xcos, xsin], dim=-1)

    def _call_xnet(
            self,
            step: int,
            inputs: tuple[torch.Tensor, torch.Tensor],
            mask: torch.Tensor,
            first: bool = False,
            hmc: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Call `self.xnet` to get sx, tx, qx for updating x."""
        if hmc or self.config.hmc:
            return self._hmc_net(inputs[0])

        x, v = inputs
        x = self._convert_to_cartesian(x, mask)
        if self.config.separate_networks:
            xnet = self.xnet0[step] if first else self.xnet1[step]
        else:
            xnet = self.xnet

        return xnet((x, v))

    def _hmc_forward_lf(self, step: int, state: State):
        state_, _ = self._hmc_update_v_forward(step, state)
        state_, _ = self._hmc_update_x_forward(step, state_)
        state_, _ = self._hmc_update_v_forward(step, state_)

        return state_, torch.zeros(state.x.shape[0])

    def _hmc_backward_lf(self, step: int, state: State):
        step_r = self.lf - step - 1
        state_, _ = self._hmc_update_v_backward(step_r, state)
        state_, _ = self._hmc_update_x_backward(step_r, state_)
        state_, _ = self._hmc_update_v_backward(step_r, state_)

        return state_, torch.zeros(state.x.shape[0])

    def _forward_lf(self, step: int, state: State):
        if self.config.hmc:
            return self._hmc_forward_lf(step, state)

        m, mc = self._get_mask(step)
        sumlogdet = torch.zeros(state.x.shape[0])

        state_, logdet = self._update_v_forward(step, state)
        sumlogdet = sumlogdet + logdet

        state_, logdet = self._update_x_forward(step, state_,
                                                (m, mc), first=True)
        sumlogdet = sumlogdet + logdet

        state_, logdet = self._update_x_forward(step, state_,
                                                (mc, m), first=False)
        sumlogdet = sumlogdet + logdet

        state_, logdet = self._update_v_forward(step, state_)
        sumlogdet = sumlogdet + logdet

        return state_, sumlogdet

    def _backward_lf(self, step: int, state: State):
        if self.config.hmc:
            return self._hmc_backward_lf(step, state)

        step_r = self.lf - step - 1
        m, mc = self._get_mask(step_r)
        sumlogdet = torch.zeros(state.x.shape[0])

        state_, logdet = self._update_v_backward(step_r, state)
        sumlogdet = sumlogdet + logdet

        state, logdet = self._update_x_backward(step_r, state_,
                                                (mc, m), first=False)
        sumlogdet = sumlogdet + logdet

        state_, logdet = self._update_x_backward(step_r, state_,
                                                (m, mc), first=True)
        sumlogdet = sumlogdet + logdet

        state_, logdet = self._update_v_backward(step_r, state_)
        sumlogdet = sumlogdet + logdet

        return state_, sumlogdet

    def _hmc_update_v_forward(self, step: int, state: State):
        x = to_u1(state.x)
        grad = self.grad_potential(x, state.beta)
        eps = self.veps[step]
        vf = state.v - 0.5 * eps * grad
        logdet = torch.zeros(state.x.shape[0])
        state_ = State(x=x, v=vf, beta=state.beta)
        return state_, logdet

    def _update_v_forward(self, step: int, state: State):
        if self.config.hmc:
            return self._hmc_update_v_forward(step, state)

        x = to_u1(state.x)
        # v = state.v
        eps = self.veps[step]
        grad = self.grad_potential(x, state.beta)
        s_, t_, q_ = self._call_vnet(step, (x, grad))

        s = 0.5 * eps * self.nw.sv * s_
        t = self.nw.tv * t_
        q = eps * self.nw.qv * q_

        vf = state.v * torch.exp(s) - 0.5 * eps * (grad * torch.exp(q) - t)
        state_ = State(x=x, v=vf, beta=state.beta)
        logdet = torch.sum(s, dim=1)

        return state_, logdet

    def _hmc_update_v_backward(self, step: int, state: State):
        x = to_u1(state.x)
        eps = self.veps[step]
        grad = self.grad_potential(x, state.beta)
        vb = state.v - 0.5 * eps * grad
        logdet = torch.zeros(state.x.shape[0])
        state_ = State(x=x, v=vb, beta=state.beta)

        return state_, logdet

    def _update_v_backward(self, step: int, state: State):
        if self.config.hmc:
            return self._hmc_update_v_backward(step, state)

        x = to_u1(state.x)
        v = state.v
        eps = self.veps[step]
        grad = self.grad_potential(x, state.beta)
        s_, t_, q_ = self._call_vnet(step, (x, grad))

        s = -0.5 * eps * self.nw.sv * s_
        t = self.nw.tv * t_
        q = eps * self.nw.qv * q_

        vb = torch.exp(s) * (v + 0.5 * eps * (grad * torch.exp(q) - t))
        state_ = State(x=x, v=vb, beta=state.beta)
        logdet = torch.sum(s, dim=1)

        return state_, logdet

    def _hmc_update_x_forward(self, step: int, state: State):
        eps = self.xeps[step]
        v = state.v
        x = to_u1(state.x)
        xf = to_u1(x + eps * v)
        state_ = State(x=xf, v=v, beta=state.beta)

        return state_, torch.zeros(state.x.shape[0])

    def _update_x_forward(
            self,
            step: int,
            state: State,
            masks: tuple[torch.Tensor, torch.Tensor],
            first: bool
    ):
        if self.config.hmc:
            return self._hmc_update_x_forward(step, state)

        m, mc = masks
        eps = self.xeps[step]
        x = state.x
        v = state.v
        s_, t_, q_ = self._call_xnet(step, (x, v), m, first=first)

        s = eps * self.nw.sx * s_
        t = self.nw.tx * t_
        q = eps * self.nw.qx * q_

        exp_s = torch.exp(s)
        exp_q = torch.exp(q)

        if self._use_ncp:
            _x = 2 * torch.atan(torch.tan(x / 2.) * exp_s)
            xnew = _x + eps * (v * exp_q + t)
            xf = (m * x) + (mc * xnew)

            cterm = torch.cos(x / 2.) ** 2
            sterm = (exp_s * torch.sin(x / 2.)) ** 2
            logdet_ = torch.log(exp_s / (cterm + sterm))
            logdet = torch.sum(mc * logdet_, dim=1)
        else:
            xnew = x * exp_s + eps * (v * exp_q + t)
            xf = (m * x) + (mc * xnew)
            logdet = torch.sum(mc * s, dim=1)

        state_ = State(x=to_u1(xf), v=v, beta=state.beta)

        return state_, logdet

    def _hmc_update_x_backward(self, step: int, state: State):
        """Update `x` in the backward direction for HMC."""
        xf = to_u1(to_u1(state.x) + self.xeps[step] * state.v)
        state_ = State(x=xf, v=state.v, beta=state.beta)

        return state_, torch.zeros(state.x.shape[0])

    def _update_x_backward(
            self,
            step: int,
            state: State,
            masks: tuple[torch.Tensor, torch.Tensor],
            first: bool,
    ):
        if self.config.hmc:
            return self._hmc_update_x_backward(step, state)

        m, mc = masks
        eps = self.xeps[step]
        x = to_u1(state.x)
        v = state.v
        s_, t_, q_ = self._call_xnet(step, (x, v), m, first)

        s = self.nw.sx * (-eps * s_)
        t = self.nw.tx * t_
        q = self.nw.qx * (eps * q_)

        exp_s = torch.exp(s)
        exp_q = torch.exp(q)

        if self._use_ncp:
            x1 = 2. * torch.atan(exp_s * torch.tan(x / 2.))
            x2 = exp_s * eps * (v * exp_q + t)
            xnew = x1 - x2
            xb = (m * x) + (mc * xnew)

            cterm = torch.cos(x / 2.) ** 2
            sterm = (exp_s * torch.sin(x / 2.)) ** 2
            logdet_ = torch.log(exp_s / (cterm + sterm))
            logdet = (mc * logdet_).sum(dim=1)
        else:
            xnew = exp_s * (x - eps * (v * exp_q +  t))
            xb = m * x + mc * xnew 
            logdet = (mc * s).sum(dim=1)

        state_ = State(x=to_u1(xb), v=v, beta=state.beta)

        return state_, logdet

    def hamiltonian(self, state: State):
        kinetic = self.kinetic_energy(state.v)
        potential = self.potential_energy(state.x, state.beta)

        return kinetic + potential

    @staticmethod
    def kinetic_energy(v: torch.Tensor):
        return 0.5 * (v ** 2).sum(dim=-1)

    def potential_energy(self, x: torch.Tensor, beta: float):
        return beta * self.potential_fn(to_u1(x))

    def grad_potential(
            self,
            x: torch.Tensor,
            beta: float,
            create_graph: bool = True
    ):
        x = to_u1(x)
        x.requires_grad_(True)
        s = self.potential_energy(x, beta)
        dsdx, = torch.autograd.grad(s, x,
                                    # retain_graph=True,
                                    create_graph=create_graph,
                                    grad_outputs=torch.ones(x.shape[0]))
        # return dsdx.detach()
        return dsdx

    def calc_metrics(self, mc_states: MonteCarloStates) -> Metrics:
        x0 = to_u1(mc_states.init.x)
        x1 = to_u1(mc_states.out.x)
        metrics = self.lattice.calc_observables(x1)

        q0 = self.lattice.calc_both_charges(x0)
        q1 = self.lattice.calc_both_charges(x1)
        metrics['dQi'] = torch.abs(q1.Qi - q0.Qi)
        metrics['dQs'] = torch.abs(q1.Qs - q0.Qs)

        return metrics

    def train_step(
            self,
            inputs: tuple[torch.Tensor, torch.Tensor],
            optimizer: optim.Optimizer
    ) -> tuple[torch.Tensor, Metrics]:
        """Perform a single training step."""
        x, beta = inputs
        loss = torch.tensor(0.0)
        if torch.cuda.is_available():
            x, loss = x.cuda(), loss.cuda()

        mc_states, outputs = self((to_u1(x), beta))
        loss = -self.calc_losses(mc_states, outputs['accept_prob'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss.detach()

        metrics = self.calc_metrics(mc_states)
        metrics.update({
            'loss': loss.detach(),
            'acc': outputs['accept_mask'].detach(),
            'px': outputs['accept_prob'].detach()
        })

        return mc_states.out.x.detach(), metrics



def train_step(
        inputs: tuple[torch.Tensor, float],
        dynamics: GaugeDynamics,
        optimizer: optim.Optimizer,
        timer: StepTimer = None,
):
    """Perform a single training step"""
    timer = StepTimer() if timer is None else timer
    dynamics.train()
    x, beta = inputs

    # loss = torch.tensor(0.0)

    if torch.cuda.is_available():
        x = x.cuda()

    # optimizer.zero_grad()

    timer.start()
    mc_states, outputs = dynamics((to_u1(x), beta))
    loss = - dynamics.calc_losses(mc_states, outputs['accept_prob'])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    dt = timer.stop()
    # loss.detach()


    # x.detach()
    # loss.detach()

    metrics = {
        'dt': dt,
        'loss': loss.detach(),
        'beta': mc_states.out.beta,
        'acc': outputs['accept_mask'].detach(),
        'px': outputs['accept_prob'].detach()
    }
    dmetrics = dynamics.calc_metrics(mc_states)
    metrics.update(**dmetrics)

    return to_u1(mc_states.out.x).detach(), metrics


def test_step(
    inputs: tuple[torch.Tensor, float],
    dynamics: GaugeDynamics,
    timer: StepTimer = None,
):
    """Perform a single training step"""
    dynamics.eval()

    x, beta = inputs
    timer = StepTimer() if timer is None else timer

    loss = torch.tensor(0.0)

    if torch.cuda.is_available():
        x = x.cuda()
        loss = loss.cuda()

    timer.start()
    mc_states, outputs = dynamics((to_u1(x), beta))
    # TODO: Turn off `x.requires_grad_ = False` here???
    loss = - dynamics.calc_losses(mc_states, outputs['accept_prob'])
    dt = timer.stop()

    metrics = {
        'dt': dt,
        'loss': loss.detach(),
        'beta': mc_states.out.beta,
        'acc': outputs['accept_mask'].detach(),
        'px': outputs['accept_prob'].detach()
    }
    dmetrics = dynamics.calc_metrics(mc_states)
    metrics.update(**dmetrics)

    return mc_states.out.x, metrics


@dataclass
class Steps:
    train: int
    test: int
    log: int = 1
    save: int = 0

    def __post_init__(self):
        if self.save == 0:
            self.save == int(self.train // 4)


def train(
        dynamics: GaugeDynamics,
        optimizer: optim.Optimizer,
        steps: Steps,
        beta: Union[list[float], float],
        window: int = 10,
        x: torch.Tensor = None,
        skip: Union[str, list[str]] = None,
        keep: Union[str, list[str]] = None,
        history: History = None,
) -> History:
    """Train dynamics."""
    dynamics.train()

    if x is None:
        x = random_angle(dynamics.config.x_shape, requires_grad=True)
        x = x.reshape(x.shape[0], -1)

    train_logs = []
    if history is None:
        history = History()

    should_print = (not in_notebook())

    # logger.log(f'Training for {steps.train} steps...')

    if isinstance(beta, list):
        assert len(beta) == steps.train
    elif isinstance(beta, float):
        beta = np.array(steps.train * [beta], dtype=np.float32).tolist()
    else:
        raise ValueError(f'Unexpected value for beta: {beta}')

    assert isinstance(beta, list)
    assert len(beta) == steps.train
    assert isinstance(beta[0], float)

    for step, b in zip(range(steps.train), beta):
        x, metrics = train_step((to_u1(x), b), dynamics=dynamics,
                                optimizer=optimizer, timer=history.timer)
        history.update(metrics, step)
        pre = [f'{step}/{steps.train}']
        mstr = history.metrics_summary(window=window, pre=pre,
                                       keep=keep, skip=skip,
                                       should_print=should_print)
        if not should_print:
            logger.log(mstr)

        train_logs.append(mstr)

    logger.log(80 * '-')
    rate = history.timer.get_eval_rate(evals_per_step=dynamics.config.num_steps)
    logger.log(f'Done training! took: {rate["total_time"]}')
    logger.log(f'Timing info:')
    for key, val in rate.items():
        logger.log(f' - {key}={val}')

    return history


def test(
        dynamics: GaugeDynamics,
        steps: Steps,
        beta: Union[list[float], float],
        x: torch.Tensor = None,
        skip: Union[str, list[str]] = None,
        keep: Union[str, list[str]] = None,
        history: History = None,
        nchains_test: int = None,
) -> History:
    """Run training and evaluate the trained model."""
    logger.log(80 * '-')
    logger.log(f'Running inference...')
    should_print = not in_notebook()

    dynamics.eval()

    if history is None:
        history = History()

    if x is None:
        x = random_angle(dynamics.config.x_shape, requires_grad=True)

    if isinstance(beta, float):
        beta = np.array(steps.train * [beta], dtype=np.float32).tolist()

    assert isinstance(beta, list)
    assert len(beta) == steps.train
    assert isinstance(beta[0], float)

    test_logs = []
    test_beta = beta[-1]
    x = x.reshape(x.shape[0], -1)

    for step in range(steps.test):
        x, metrics = test_step((x, test_beta), dynamics, timer=history.timer)
        history.update(metrics, step)
        pre = [f'{step}/{steps.test}']
        mstr = history.metrics_summary(window=0, pre=pre,
                                       keep=keep, skip=skip,
                                       should_print=should_print)
        if not should_print:
            logger.log(mstr)

        test_logs.append(mstr)

    rate = history.timer.get_eval_rate(evals_per_step=dynamics.config.num_steps)
    logger.log(f'Done training! took: {rate["total_time"]}')
    logger.log(f'Timing info:')
    for key, val in rate.items():
        logger.log(f' - {key}={val}')

    return history


def train_and_test(
        dynamics: GaugeDynamics,
        optimizer: optim.Optimizer,
        steps: Steps,
        beta: Union[list[float], float],
        window: int = 10,
        x: torch.Tensor = None,
        skip: Union[str, list[str]] = None,
        keep: Union[str, list[str]] = None,
        train_history: History = None,
        test_history: History = None,
        nchains_test: int = None,
) -> dict[str, History]:
    """Train and test"""
    train_out = train(dynamics=dynamics, optimizer=optimizer,
                      steps=steps, beta=beta, keep=keep,
                      skip=skip, window=window, history=train_history)
    test_out = test(dynamics=dynamics, steps=steps,
                    beta=beta, x=x, skip=skip,
                    keep=keep, history=test_history, nchains_test=nchains_test)

    return {'train': train_out, 'test': test_out}
