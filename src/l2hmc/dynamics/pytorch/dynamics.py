"""
dynamics.py

Contains pytorch implementation of Dynamics object for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass
from math import pi
from typing import Callable, Union

import time
from src.l2hmc.configs import DynamicsConfig
from src.l2hmc.loss.pytorch.loss import LatticeLoss
from src.l2hmc.network.pytorch.network import (
    NetworkFactory,
    NetworkInputs,
    NetworkOutputs,
)
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PI = np.pi
TWO_PI = 2. * PI

Shape = Union[tuple, list]
Tensor = torch.Tensor


def to_u1(x: Tensor) -> Tensor:
    return ((x + PI) % TWO_PI) - PI


def rand_unif(shape: Shape,
              a: float,
              b: float,
              requires_grad: bool) -> Tensor:
    """Draw tensor from random uniform distribution U[a, b]"""
    return torch.tensor((a - b) * torch.rand(shape) + b).requires_grad_(requires_grad)


def random_angle(shape: Shape, requires_grad: bool = True) -> Tensor:
    return rand_unif(shape, -pi, pi, requires_grad=requires_grad)


# class DynamicsInput(NamedTuple):
#     x: Tensor
#     beta: float

DynamicsInput = tuple[Tensor, float]


@dataclass
class State:
    x: Tensor
    v: Tensor
    beta: float


@dataclass
class MonteCarloStates:
    init: State
    proposed: State
    out: State


class Mask:
    def __init__(self, m: Tensor):
        self.m = m
        self.mb = torch.ones_like(self.m) - self.m  # complement: 1. - m

    def combine(self, x: Tensor, y: Tensor):
        return self.m * x + self.mb * y


class Dynamics(nn.Module):
    def __init__(
            self,
            potential_fn: Callable,
            config: DynamicsConfig,
            network_factory: NetworkFactory,
            # loss_fn: Callable,
            verbose: bool = True,
    ):
        """Initialization method."""
        super().__init__()
        self._verbose = verbose
        self.config = config
        self.xdim = self.config.xdim
        # self.loss_fn = loss_fn
        # self.optimizer = self.configure_optimizers()
        # self.lattice = lattice
        # self.potential_fn = self.lattice.action
        self.xshape = network_factory.input_spec.xshape
        self.potential_fn = potential_fn
        self.network_factory = network_factory
        self.nlf = self.config.nleapfrog
        self.num_networks = (
            self.nlf if self.config.use_separate_networks else 1
        )
        self.networks = network_factory.build_networks(
            n=self.num_networks,
            split_xnets=self.config.use_split_xnets
        )
        self.masks = self._build_masks()

        xeps = []
        veps = []
        rg = (not self.config.eps_fixed)
        for _ in range(self.config.nleapfrog):
            alpha = torch.log(torch.tensor(self.config.eps))
            xe = torch.clamp(torch.exp(alpha), min=0., max=1.)
            ve = torch.clamp(torch.exp(alpha), min=0., max=1.)
            xeps.append(nn.parameter.Parameter(xe, requires_grad=rg))
            veps.append(nn.parameter.Parameter(ve, requires_grad=rg))
            # xeps.append(torch.clamp(xe, min=0., max=1.))
            # veps.append(torch.clamp(ve, min=0., max=1.))

        self.xeps = nn.ParameterList(xeps)
        self.veps = nn.ParameterList(veps)

    def forward(self, inputs: DynamicsInput) -> tuple[Tensor, dict]:
        return self.apply_transition(inputs)

    def apply_transition(self, inputs: DynamicsInput) -> tuple[Tensor, dict]:
        x, beta = inputs
        fwd = self._transition(inputs, forward=True)
        bwd = self._transition(inputs, forward=False)

        # sf_init = fwd['init']
        # sf_prop = fwd['proposed']
        # metricsf = fwd['metrics']

        mf_, mb_ = self._get_direction_masks(batch_size=x.shape[0])
        mf = mf_.unsqueeze(-1)  # mf = mf_[:, None]
        mb = mb_.unsqueeze(-1)  # mb = mb_[:, None]

        v_init = mf * fwd['init'].v + mb * bwd['init'].v

        xp = mf * fwd['proposed'].x + mb * bwd['proposed'].x
        vp = mf * fwd['proposed'].v + mb * bwd['proposed'].v

        mfwd = fwd['metrics']
        mbwd = bwd['metrics']

        logdetp = mf_ * mfwd['sumlogdet'] + mb_ * mbwd['sumlogdet']

        acc = mf_ * mfwd['acc'] + mb_ * mbwd['acc']
        ma_, mr_ = self._get_accept_masks(acc)
        ma = ma_.unsqueeze(-1)
        mr = mr_.unsqueeze(-1)

        v_out = ma * vp + mr * v_init
        x_out = ma * xp + mr * x
        logdet = ma_ * logdetp  # + mr_ * logdet_init = 0

        state_init = State(x=x, v=v_init, beta=beta)
        state_prop = State(x=xp, v=vp, beta=beta)
        state_out = State(x=x_out, v=v_out, beta=beta)
        mc_states = MonteCarloStates(init=state_init,
                                     proposed=state_prop,
                                     out=state_out)
        metrics = {
            'acc': acc,
            'acc_mask': ma_,
            'logdet': logdet,
            'mc_states': mc_states,
            'forward': mfwd,
            'backward': mbwd,
        }

        return mc_states.out.x, metrics

    def _transition(
            self,
            inputs: DynamicsInput,
            forward: bool
    ) -> dict:
        """Run the transition kernel"""
        return self.generate_proposal(inputs, forward)

    def generate_proposal(
            self,
            inputs: DynamicsInput,
            forward: bool,
    ) -> dict:
        x, beta = inputs
        v = torch.randn_like(x)
        state_init = State(x=x, v=v, beta=beta)
        state_prop, metrics = self.transition_kernel(state_init, forward)

        return {'init': state_init, 'proposed': state_prop, 'metrics': metrics}

    def get_metrics(self, state: State, logdet: Tensor) -> dict[str, Tensor]:
        return {
            'energy': (h := self.hamiltonian(state)),
            'logprob': h - logdet,
            'logdet': logdet,
        }

    def _transition_kernel(
            self,
            state: State,
            forward: bool
    ) -> tuple[State, dict]:
        """Implements the transition kernel."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        # Copy initial state into proposed state
        state_ = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = torch.zeros(state.x.shape[0], device=state.x.device)
        # lf_metrics = {n: {} for n in range(self.config.nleapfrog)}
        history = {}
        if self._verbose:
            history = {
                'xeps': [self.xeps[0], *self.xeps],
                'veps': [self.veps[0], *self.veps],
                'energy': [(h := self.hamiltonian(state))],
                'logdet': [sumlogdet],
                'logprob': [h - sumlogdet],
            }

        for step in range(self.config.nleapfrog):
            state_, logdet = lf_fn(step, state_)
            sumlogdet = sumlogdet + logdet
            if self._verbose:
                metrics = self.get_metrics(state_, sumlogdet)
                for key, val in metrics.items():
                    history[key].append(val)

        acc = self.compute_accept_prob(state, state_, sumlogdet)
        history.update({'acc': acc, 'sumlogdet': sumlogdet})
        new_state = State(x=state_.x, v=state_.v, beta=state_.beta)

        if self._verbose:
            for key, val in history.items():
                if isinstance(val, list):
                    history[key] = torch.stack(val).detach().numpy()

        return new_state, history

    def transition_kernel(
            self,
            state: State,
            forward: bool
    ) -> tuple[State, dict]:
        return self._transition_kernel(state, forward)

    def compute_accept_prob(
        self,
        state_init: State,
        state_prop: State,
        sumlogdet: Tensor,
    ) -> Tensor:
        h_init = self.hamiltonian(state_init)
        h_prop = self.hamiltonian(state_prop)
        dh = h_init - h_prop + sumlogdet
        prob = torch.exp(
            torch.minimum(dh, torch.zeros_like(dh, device=dh.device))
        )

        return prob

    @staticmethod
    def _get_accept_masks(px: Tensor) -> tuple[Tensor, Tensor]:
        acc = (px > torch.rand_like(px).to(DEVICE)).to(torch.float)
        rej = torch.ones_like(acc) - acc
        return acc, rej

    @staticmethod
    def _get_direction_masks(batch_size: int) -> tuple[Tensor, Tensor]:
        """Returns (forward_mask, backward_mask)."""
        fwd = (torch.rand(batch_size).to(DEVICE) > 0.5).to(torch.float)
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
            masks.append(torch.from_numpy(mask[None, :]).to(DEVICE))

        return masks

    def _get_vnet(self, step: int) -> nn.Module:
        vnet = self.networks.get_submodule('vnet')
        if self.config.use_separate_networks:
            return vnet.get_submodule(str(step))
        return vnet

    def _call_vnet(
            self,
            step: int,
            inputs: NetworkInputs,
    ) -> NetworkOutputs:
        """Call the momentum update network for a step along the trajectory"""
        vnet = self._get_vnet(step)
        assert callable(vnet)
        return vnet(inputs)

    def _get_xnet(self, step: int, first: bool = False) -> nn.Module:
        xnet = self.networks.get_submodule('xnet')
        if self.config.use_separate_networks:
            xnet = xnet.get_submodule(str(step))
            if self.config.use_split_xnets:
                if first:
                    return xnet.get_submodule('first')
                return xnet.get_submodule('second')
            return xnet
        return xnet

    def _stack_as_xy(self, x: Tensor) -> Tensor:
        # TODO: Deal with ConvNet here
        # if self.config.use_conv_net:
        #     xcos = xcos.reshape(self.lattice_shape)
        #     xsin = xsin.reshape(self.lattice_shape)
        return torch.stack([torch.cos(x), torch.sin(x)], dim=-1)

    def _call_xnet(
            self,
            step: int,
            inputs: NetworkInputs,
            first: bool = False,
    ) -> NetworkOutputs:
        """Call the position update network for a step along the trajectory."""
        x, v = inputs
        x = self._stack_as_xy(x)
        xnet = self._get_xnet(step, first)
        assert callable(xnet)
        return xnet((x, v))

    def _forward_lf(self, step: int, state: State) -> tuple[State, Tensor]:
        """Complete update (leapfrog step) in the forward direction. """
        m, mb = self._get_mask(step)
        sumlogdet = torch.zeros(state.x.shape[0], device=state.x.device)

        state_, logdet = self._update_v_fwd(step, state)
        sumlogdet = sumlogdet + logdet

        state_, logdet = self._update_x_fwd(step, state_, m, first=True)
        sumlogdet = sumlogdet + logdet

        state_, logdet = self._update_x_fwd(step, state_, mb, first=False)
        sumlogdet = sumlogdet + logdet

        state_, logdet = self._update_v_fwd(step, state_)

        return state_, sumlogdet

    def _backward_lf(self, step: int, state: State) -> tuple[State, Tensor]:
        """Complete update (leapfrog step) in the backward direction"""
        # NOTE: Reverse the step count, i.e. count from end of trajectory
        step = self.config.nleapfrog - step - 1
        m, mb = self._get_mask(step)
        sumlogdet = torch.zeros(state.x.shape[0], device=state.x.device)

        state_, logdet = self._update_v_bwd(step, state)
        sumlogdet = sumlogdet + logdet

        state_, logdet = self._update_x_bwd(step, state_, mb, first=False)
        sumlogdet = sumlogdet + logdet

        state_, logdet = self._update_x_bwd(step, state_, m, first=True)
        sumlogdet = sumlogdet + logdet

        state_, logdet = self._update_v_bwd(step, state_)
        sumlogdet = sumlogdet + logdet

        return state_, sumlogdet

    def _update_v_fwd(self, step: int, state: State) -> tuple[State, Tensor]:
        """Single v update in the forward direction"""
        eps = self.veps[step]
        grad = self.grad_potential(state.x, state.beta)
        s, t, q = self._call_vnet(step, (state.x, grad))

        s = torch.mul(0.5, eps * s)
        q = eps * q

        vf = state.v * torch.exp(s) - 0.5 * eps * (grad * torch.exp(q) - t)
        logdet = torch.sum(s, dim=1)

        return State(state.x, vf, state.beta), logdet

    def _update_v_bwd(self, step: int, state: State) -> tuple[State, Tensor]:
        """Single v update in the backward direction"""
        eps = self.veps[step]
        grad = self.grad_potential(state.x, state.beta)
        s, t, q = self._call_vnet(step, (state.x, grad))

        s = torch.mul(-0.5, eps * s)
        q = eps * q

        vb = torch.exp(s) * (state.v + 0.5 * eps * (grad * torch.exp(q) - t))
        logdet = torch.sum(s, dim=1)

        return State(state.x, vb, state.beta), logdet

    def _update_x_fwd(
            self,
            step: int,
            state: State,
            m: Tensor,
            first: bool,
    ) -> tuple[State, Tensor]:
        """Single x update in the forward direction"""
        eps = self.xeps[step]
        mb = torch.ones_like(m) - m
        s, t, q = self._call_xnet(step, (m * state.x, state.v), first=first)
        s = eps * s
        q = eps * q
        exp_s = torch.exp(s)
        exp_q = torch.exp(q)
        if self.config.use_ncp:
            _x = 2 * torch.atan(torch.tan(state.x / 2.) * exp_s)
            xnew = _x + eps * (state.v * exp_q + t)
            xf = (m * state.x) + (mb * xnew)
            cterm = torch.cos(state.x / 2.) ** 2
            sterm = (exp_s * torch.sin(state.x / 2.)) ** 2
            logdet_ = torch.log(exp_s / (cterm + sterm))
            logdet = (mb * logdet_).sum(dim=1)
        else:
            xnew = state.x * exp_s + eps * (state.v * exp_q + t)
            xf = (m * state.x) + (mb * xnew)
            logdet = (mb * s).sum(dim=1)

        return State(x=xf, v=state.v, beta=state.beta), logdet

    def _update_x_bwd(
            self,
            step: int,
            state: State,
            m: Tensor,
            first: bool,
    ) -> tuple[State, Tensor]:
        eps = self.xeps[step]
        mb = torch.ones_like(m) - m
        s, t, q = self._call_xnet(step, (m * state.x, state.v), first=first)
        s = (-eps) * s
        q = eps * q

        exp_s = torch.exp(s)
        exp_q = torch.exp(q)
        if self.config.use_ncp:
            x1 = 2. * torch.atan(exp_s * torch.tan(state.x / 2.))
            x2 = exp_s * eps * (state.v * exp_q + t)
            xnew = x1 - x2
            xb = (m * state.x) + (mb * xnew)

            cterm = torch.cos(state.x / 2.) ** 2
            sterm = (exp_s * torch.sin(state.x / 2.)) ** 2
            logdet_ = torch.log(exp_s / (cterm + sterm))
            logdet = (mb * logdet_).sum(dim=1)
        else:
            xnew = exp_s * (state.x - eps * (state.v * exp_q + t))
            xb = (m * state.x) + (mb * xnew)
            logdet = (mb * s).sum(dim=1)

        return State(x=xb, v=state.v, beta=state.beta), logdet

    def hamiltonian(self, state: State) -> Tensor:
        kinetic = self.kinetic_energy(state.v)
        potential = self.potential_energy(state.x, state.beta)
        return kinetic + potential

    def kinetic_energy(self, v: Tensor) -> Tensor:
        return (0.5 * (v ** 2)).sum(dim=-1)

    def potential_energy(self, x: Tensor, beta: float):
        return torch.mul(beta, self.potential_fn(x))

    def grad_potential(
        self,
        x: Tensor,
        beta: float,
        create_graph: bool = True,
    ) -> Tensor:
        x.requires_grad_(True)
        s = self.potential_energy(x, beta)
        id = torch.ones(x.shape[0], device=x.device)
        dsdx, = torch.autograd.grad(s, x,
                                    create_graph=create_graph,
                                    grad_outputs=id)
        return dsdx
