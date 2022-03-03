"""
pytorch/dynamics.py

Pytorch implementation of Dynamics object for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass
from math import pi as PI
import os
from pathlib import Path
import logging
from typing import Callable, Union
from typing import Tuple

from l2hmc.configs import DynamicsConfig
from l2hmc.network.pytorch.network import (
    NetworkFactory,
)
import numpy as np
import torch
from torch import nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

log = logging.getLogger(__name__)

TWO_PI = 2. * PI

Shape = Union[tuple, list]
Tensor = torch.Tensor
Array = np.ndarray


DynamicsInput = Tuple[Tensor, Tensor]  # (xinit, beta)
DynamicsOutput = Tuple[Tensor, dict]  # (xout, metrics)


@dataclass
class State:
    x: Tensor               # gauge links
    v: Tensor               # conj. momenta
    beta: Tensor            # inv. coupling const.


@dataclass
class MonteCarloStates:
    init: State             # Input state
    proposed: State         # Proposal state
    out: State              # Output state (after acc/rej)


def to_u1(x: Tensor) -> Tensor:
    """Returns x as U(1) link variable in [-pi, pi)."""
    return ((x + PI) % TWO_PI) - PI


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
    """Returns random angle with `shape` and values in [-pi, pi)."""
    return rand_unif(shape, -PI, PI, requires_grad=requires_grad)


# TODO: Remove or finish implementation ?
class Mask:
    def __init__(self, m: Tensor):
        self.m = m
        self.mb = torch.ones_like(self.m) - self.m  # complement: 1. - m

    def combine(self, x: Tensor, y: Tensor):
        return self.m * x + self.mb * y


def grab(x: Tensor) -> Array:
    """Detach tensor and return as numpy array on CPU."""
    return x.detach().cpu().numpy()


class Dynamics(nn.Module):
    def __init__(
            self,
            potential_fn: Callable,
            config: DynamicsConfig,
            network_factory: NetworkFactory,
    ):
        """Initialization method."""
        super(Dynamics, self).__init__()
        self.config = config
        self.xdim = self.config.xdim
        self.xshape = tuple(network_factory.input_spec.xshape)
        self.potential_fn = potential_fn
        self.network_factory = network_factory
        self.nlf = self.config.nleapfrog
        self.networks = network_factory.build_networks(
            n=(self.nlf if self.config.use_separate_networks else 1),
            split_xnets=self.config.use_split_xnets
        )
        self.masks = self._build_masks()
        xeps = {}
        veps = {}
        rg = (not self.config.eps_fixed)
        for lf in range(self.config.nleapfrog):
            xeps[str(lf)] = nn.parameter.Parameter(
                data=torch.tensor(self.config.eps), requires_grad=rg
            )
            veps[str(lf)] = nn.parameter.Parameter(
                data=torch.tensor(self.config.eps), requires_grad=rg
            )

        self.xeps = nn.ParameterDict(xeps)
        self.veps = nn.ParameterDict(veps)

    def save(self, outdir: os.PathLike) -> None:
        netdir = Path(outdir).joinpath('networks')
        fxeps = netdir.joinpath('xeps.npy')
        fveps = netdir.joinpath('veps.npy')
        outfile = netdir.joinpath('dynamics.pt')
        netdir.mkdir(exist_ok=True, parents=True)
        # log.info(
        #     f'Saving `xeps`, `veps`, `self.state_dict()` to: {netdir}'
        # )
        xeps = np.array([
            i.detach().cpu().numpy() for i in list(self.xeps.values())
        ])
        veps = np.array([
            i.detach().cpu().numpy() for i in list(self.veps.values())
        ])
        np.save(fxeps, xeps)
        np.save(fveps, veps)
        np.savetxt(netdir.joinpath('xeps.txt').as_posix(), xeps)
        np.savetxt(netdir.joinpath('veps.txt').as_posix(), veps)
        # log.info(f'Saving dynamics to: {outfile}')
        torch.save(self.state_dict(), outfile.as_posix())

    def save_eps(self, outdir: os.PathLike) -> None:
        netdir = Path(outdir).joinpath('networks')
        fxeps = netdir.joinpath('xeps.npy')
        fveps = netdir.joinpath('veps.npy')
        xeps = np.array([
            i.detach().cpu().numpy() for i in list(self.xeps.values())
        ])
        veps = np.array([
            i.detach().cpu().numpy() for i in list(self.veps.values())
        ])
        # log.info(f'Saving `xeps`, `veps` to: {netdir}')
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
            eps: dict[str, dict[str, Tensor]]
    ) -> None:
        xe = eps['xeps']
        ve = eps['veps']
        rg = (not self.config.eps_fixed)
        xeps = {}
        veps = {}
        for lf in range(self.config.nleapfrog):
            xe[str(lf)] = nn.parameter.Parameter(
                data=torch.tensor(xe[str(lf)]), requires_grad=rg
            )
            ve[str(lf)] = nn.parameter.Parameter(
                data=torch.tensor(ve[str(lf)]), requires_grad=rg
            )

        xeps = nn.ParameterDict(xeps)
        veps = nn.ParameterDict(veps)

        self.xeps = xeps
        self.veps = veps

    def forward(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, dict]:
        if self.config.merge_directions:
            return self.apply_transition_fb(inputs)

        return self.apply_transition(inputs)

    def apply_transition_hmc(
            self,
            inputs: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, dict]:
        data = self.generate_proposal_hmc(inputs)
        ma_, mr_ = self._get_accept_masks(data['metrics']['acc'])
        ma = ma_.unsqueeze(-1)
        mr = mr_.unsqueeze(-1)

        vout = ma * data['proposed'].v + mr * data['init'].v
        xout = ma * data['proposed'].x + mr * data['init'].x
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
        ma = ma_.unsqueeze(-1)
        mr = mr_.unsqueeze(-1)

        v_out = ma * data['proposed'].v + mr * data['init'].v
        x_out = ma * data['proposed'].x + mr * data['init'].x

        # NOTE: sumlogdet = (accept * logdet) + (reject * 0)
        sumlogdet = ma_ * data['metrics']['sumlogdet']

        state_out = State(x=x_out, v=v_out, beta=data['init'].beta)
        mc_states = MonteCarloStates(init=data['init'],
                                     proposed=data['proposed'],
                                     out=state_out)
        data['metrics'].update({
            'acc_mask': ma_,
            'sumlogdet': sumlogdet,
            'mc_states': mc_states,
        })

        return x_out, data['metrics']

    def apply_transition(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, dict]:
        x, beta = inputs
        fwd = self.generate_proposal(inputs, forward=True)
        bwd = self.generate_proposal(inputs, forward=False)

        mf_, mb_ = self._get_direction_masks(batch_size=x.shape[0])
        mf = mf_.unsqueeze(-1)  # mf = mf_[:, None]
        mb = mb_.unsqueeze(-1)  # mb = mb_[:, None]

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
        ma = ma_.unsqueeze(-1)
        mr = mr_.unsqueeze(-1)

        v_out = ma * vp + mr * v_init
        x_out = ma * xp + mr * x
        logdet = ma_ * logdetp  # NOTE: + mr_ * logdet_init = 0

        state_init = State(x=x, v=v_init, beta=beta)
        state_prop = State(x=xp, v=vp, beta=beta)
        state_out = State(x=x_out, v=v_out, beta=beta)
        mc_states = MonteCarloStates(init=state_init,
                                     proposed=state_prop,
                                     out=state_out)
        metrics = {
            'acc': acc,
            'acc_mask': ma_,
            'sumlogdet': logdet,
            'mc_states': mc_states,
        }
        metrics.update(**{f'fwd/{key}': val for key, val in mfwd.items()})
        metrics.update(**{f'bwd/{key}': val for key, val in mbwd.items()})

        return x_out, metrics

    def random_state(self, beta: float) -> State:
        x = torch.rand(tuple(self.xshape)).reshape(self.xshape[0], -1)
        v = torch.randn_like(x)
        return State(x=x, v=v, beta=torch.tensor(beta))

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
    ) -> dict:
        x, beta = inputs
        v = torch.randn_like(x)
        init = State(x=x, v=v, beta=beta)
        proposed, metrics = self.transition_kernel_hmc(init)

        return {'init': init, 'proposed': proposed, 'metrics': metrics}

    def generate_proposal_fb(
            self,
            inputs: tuple[Tensor, Tensor],
    ) -> dict:
        x, beta = inputs
        v = torch.randn_like(x)
        init = State(x=x, v=v, beta=beta)
        proposed, metrics = self.transition_kernel_fb(init)

        return {'init': init, 'proposed': proposed, 'metrics': metrics}

    def generate_proposal(
            self,
            inputs: tuple[Tensor, Tensor],
            forward: bool,
    ) -> dict:
        x, beta = inputs
        v = torch.randn_like(x)
        state_init = State(x=x, v=v, beta=beta)
        state_prop, metrics = self.transition_kernel(state_init, forward)

        return {'init': state_init, 'proposed': state_prop, 'metrics': metrics}

    def get_metrics(
            self,
            state: State,
            logdet: Tensor,
            step: int = None
    ) -> dict:
        energy = self.hamiltonian(state)
        logprob = energy - logdet
        metrics = {
            'energy': energy,
            'logprob': logprob,
            'logdet': logdet,
        }
        if step is not None:
            metrics.update({
                'xeps': self.xeps[str(step)].clone().detach(),
                'veps': self.veps[str(step)].clone().detach(),
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
            step: int,
    ) -> State:
        force1 = self.grad_potential(state.x, state.beta)
        v1 = state.v - 0.5 * self.veps[str(step)] * force1

        xp = state.x + self.xeps[str(step)] * v1

        force2 = self.grad_potential(xp, state.beta)
        v2 = v1 - 0.5 * self.veps[str(step)] * force2

        return State(x=xp, v=v2, beta=state.beta)

    def transition_kernel_hmc(
            self,
            state: State,
    ) -> tuple[State, dict]:
        state_ = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = torch.zeros(state.x.shape[0], device=state.x.device)
        metrics = self.get_metrics(state_, sumlogdet)
        history = self.update_history(metrics, history={})
        for step in range(self.config.nleapfrog):
            state_ = self.leapfrog_hmc(state_, step)
            if self.config.verbose:
                metrics = self.get_metrics(state_, sumlogdet, step=step)
                history = self.update_history(metrics, history=history)

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
        state_ = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = torch.zeros(state.x.shape[0], device=state.x.device)
        metrics = self.get_metrics(state_, sumlogdet)
        history = self.update_history(metrics, history={})

        # Forward
        for step in range(self.config.nleapfrog):
            state_, logdet = self._forward_lf(step, state_)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                metrics = self.get_metrics(state_, sumlogdet, step=step)
                history = self.update_history(metrics, history=history)

        # Flip momentum
        state_ = State(state_.x, -1. * state_.v, state_.beta)

        # Backward
        for step in range(self.config.nleapfrog):
            state_, logdet = self._backward_lf(step, state_)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                metrics = self.get_metrics(state_, sumlogdet, step=step)
                history = self.update_history(metrics, history=history)

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
        state_ = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = torch.zeros(state.x.shape[0], device=state.x.device)
        metrics = self.get_metrics(state_, sumlogdet, step=0)
        history = self.update_history(metrics, history={})

        for step in range(self.config.nleapfrog):
            state_, logdet = lf_fn(step, state_)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                metrics = self.get_metrics(state_, sumlogdet, step=step)
                history = self.update_history(metrics, history=history)

        acc = self.compute_accept_prob(state, state_, sumlogdet)
        history.update({'acc': acc, 'sumlogdet': sumlogdet})
        if self.config.verbose:
            for key, val in history.items():
                if isinstance(val, list) and isinstance(val[0], Tensor):
                    history[key] = torch.stack(val).detach()

        return state_, history

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
        """Returns momentum network to be used for updating v."""
        vnet = self.networks.get_submodule('vnet')
        if self.config.use_separate_networks:
            return vnet.get_submodule(str(step))
        return vnet

    def _get_xnet(self, step: int, first: bool = False) -> nn.Module:
        """Returns position network to be used for updating x."""
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
        """Returns -pi < x <= pi stacked as [cos(x), sin(x)]"""
        # TODO: Deal with ConvNet here
        # if self.config.use_conv_net:
        #     xcos = xcos.reshape(self.lattice_shape)
        #     xsin = xsin.reshape(self.lattice_shape)
        return torch.stack([torch.cos(x), torch.sin(x)], dim=-1)

    def _call_vnet(
            self,
            step: int,
            inputs: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Call the momentum update network for a step along the trajectory"""
        vnet = self._get_vnet(step)
        assert callable(vnet)
        return vnet(inputs)

    def _call_xnet(
            self,
            step: int,
            inputs: tuple[Tensor, Tensor],
            first: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
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

        state, logdet = self._update_v_fwd(step, state)
        sumlogdet = sumlogdet + logdet

        state, logdet = self._update_x_fwd(step, state, m, first=True)
        sumlogdet = sumlogdet + logdet

        state, logdet = self._update_x_fwd(step, state, mb, first=False)
        sumlogdet = sumlogdet + logdet

        state, logdet = self._update_v_fwd(step, state)
        sumlogdet = sumlogdet + logdet

        return state, sumlogdet

    def _backward_lf(self, step: int, state: State) -> tuple[State, Tensor]:
        """Complete update (leapfrog step) in the backward direction"""
        # NOTE: Reverse the step count, i.e. count from end of trajectory
        step_r = self.config.nleapfrog - step - 1

        m, mb = self._get_mask(step_r)
        sumlogdet = torch.zeros(state.x.shape[0], device=state.x.device)

        state, logdet = self._update_v_bwd(step_r, state)
        sumlogdet = sumlogdet + logdet

        state, logdet = self._update_x_bwd(step_r, state, mb, first=False)
        sumlogdet = sumlogdet + logdet

        state, logdet = self._update_x_bwd(step_r, state, m, first=True)
        sumlogdet = sumlogdet + logdet

        state, logdet = self._update_v_bwd(step_r, state)
        sumlogdet = sumlogdet + logdet

        return state, sumlogdet

    def _update_v_fwd(self, step: int, state: State) -> tuple[State, Tensor]:
        """Single v update in the forward direction"""
        eps = self.veps[str(step)]
        force = self.grad_potential(state.x, state.beta)
        s, t, q = self._call_vnet(step, (state.x, force))

        jac = eps * s / 2.  # jacobian factor, also used in exp_s below
        logdet = jac.sum(dim=1)
        exp_s = torch.exp(jac)
        exp_q = torch.exp(eps * q)
        vf = exp_s * state.v - 0.5 * eps * (force * exp_q + t)

        return State(state.x, vf, state.beta), logdet

    def _update_v_bwd(self, step: int, state: State) -> tuple[State, Tensor]:
        """Single v update in the backward direction"""
        eps = self.veps[str(step)]
        force = self.grad_potential(state.x, state.beta)
        s, t, q = self._call_vnet(step, (state.x, force))

        jac = (-eps * s / 2.)  # jacobian factor, also used in exp_s below
        logdet = jac.sum(dim=1)
        exp_s = torch.exp(jac)
        exp_q = torch.exp(eps * q)
        vb = exp_s * (state.v + 0.5 * eps * (force * exp_q + t))

        return State(state.x, vb, state.beta), logdet

    def _update_x_fwd(
            self,
            step: int,
            state: State,
            m: Tensor,
            first: bool,
    ) -> tuple[State, Tensor]:
        """Single x update in the forward direction"""
        eps = self.xeps[str(step)]
        mb = torch.ones_like(m) - m
        xm_init = m * state.x
        inputs = (xm_init, state.v)
        s, t, q = self._call_xnet(step, inputs, first=first)
        s = eps * s
        q = eps * q
        exp_s = torch.exp(s)
        exp_q = torch.exp(q)
        if self.config.use_ncp:
            halfx = state.x / 2.
            _x = 2. * torch.atan(torch.tan(halfx) * exp_s)
            xp = _x + eps * (state.v * exp_q + t)
            xf = xm_init + (mb * xp)
            cterm = torch.cos(halfx) ** 2
            sterm = (exp_s * torch.sin(state.x / 2.)) ** 2
            logdet_ = torch.log(exp_s / (cterm + sterm))
            logdet = (mb * logdet_).sum(dim=1)
        else:
            xp = state.x * exp_s + eps * (state.v * exp_q + t)
            xf = xm_init + (mb * xp)
            logdet = (mb * s).sum(dim=1)

        return State(x=xf, v=state.v, beta=state.beta), logdet

    def _update_x_bwd(
            self,
            step: int,
            state: State,
            m: Tensor,
            first: bool,
    ) -> tuple[State, Tensor]:
        """Update the position in the backward direction."""
        eps = self.xeps[str(step)]
        mb = torch.ones_like(m) - m
        xm_init = m * state.x
        inputs = (xm_init, state.v)
        s, t, q = self._call_xnet(step, inputs, first=first)
        s = (-eps) * s
        q = eps * q
        exp_s = torch.exp(s)
        exp_q = torch.exp(q)
        if self.config.use_ncp:
            halfx = state.x / 2.
            halfx_scale = exp_s * torch.tan(halfx)
            x1 = 2. * torch.atan(halfx_scale)
            x2 = exp_s * eps * (state.v * exp_q + t)
            xnew = x1 - x2
            xb = xm_init + (mb * xnew)

            cterm = torch.cos(halfx) ** 2
            sterm = (exp_s * torch.sin(halfx)) ** 2
            logdet_ = torch.log(exp_s / (cterm + sterm))
            logdet = (mb * logdet_).sum(dim=1)
        else:
            xnew = exp_s * (state.x - eps * (state.v * exp_q + t))
            xb = xm_init + (mb * xnew)
            logdet = (mb * s).sum(dim=1)

        return State(x=xb, v=state.v, beta=state.beta), logdet

    def hamiltonian(self, state: State) -> Tensor:
        """Returns the total energy H = KE + PE."""
        kinetic = self.kinetic_energy(state.v)
        potential = self.potential_energy(state.x, state.beta)
        return kinetic + potential

    def kinetic_energy(self, v: Tensor) -> Tensor:
        """Returns the kinetic energy, KE = 0.5 * v ** 2."""
        return 0.5 * (v ** 2).sum(dim=-1)

    def potential_energy(self, x: Tensor, beta: Tensor):
        """Returns the potential energy, PE = beta * action(x)."""
        return beta * self.potential_fn(x)

    def grad_potential(
            self,
            x: Tensor,
            beta: Tensor,
            create_graph: bool = True,
    ) -> Tensor:
        """Compute the gradient of the potential function."""
        x.requires_grad_(True)
        s = self.potential_energy(x, beta)
        id = torch.ones(x.shape[0], device=x.device)
        dsdx, = torch.autograd.grad(s, x,
                                    create_graph=create_graph,
                                    grad_outputs=id)
        return dsdx
