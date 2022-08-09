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
from typing import Callable, Optional, Union
from typing import Tuple

import numpy as np
import torch
from torch import nn

import l2hmc.configs as cfgs
from l2hmc.group.u1.pytorch.group import U1Phase
from l2hmc.group.su3.pytorch.group import SU3
from l2hmc.network.pytorch.network import NetworkFactory

log = logging.getLogger(__name__)

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


def grab(x: Tensor) -> Array:
    """Detach tensor and return as numpy array on CPU."""
    return x.detach().cpu().numpy()


class Dynamics(nn.Module):
    def __init__(
            self,
            potential_fn: Callable,
            config: cfgs.DynamicsConfig,
            network_factory: NetworkFactory
    ):
        """Initialization method."""
        super().__init__()
        self.config = config
        self.xdim = self.config.xdim
        # self.xshape = tuple(network_factory.input_spec.xshape)
        self.xshape = self.config.xshape
        self.potential_fn = potential_fn
        self.nlf = self.config.nleapfrog

        self.network_factory = network_factory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.networks = self._build_networks(network_factory)
        self.xnet = self.networks['xnet']
        self.vnet = self.networks['vnet']
        self.register_module('xnet', self.networks['xnet'])
        self.register_module('vnet', self.networks['vnet'])
        # assert isinstance(self.networks, nn.ModuleDict)
        # self.xnet = self.networks.get_submodule['xnet']
        # self.vnet = self.networks.get_submodule['vnet']
        # num_nets = self.nlf if self.config.use_separate_networks else 1
        # self.networks = network_factory.build_networks(
        #     n=num_nets, split_xnets=self.config.use_split_xnets
        # )
        self.masks = self._build_masks()

        if self.config.group == 'U1':
            self.g = U1Phase()
        elif self.config.group == 'SU3':
            self.g = SU3()

        # self.xeps = nn.ParameterDict()
        # self.veps = nn.ParameterDict()
        # rg = (not self.config.eps_fixed)
        # xeps = {}
        # veps = {}
        self.xeps = nn.ParameterList([
            nn.parameter.Parameter(
                torch.tensor(
                    self.config.eps,
                ).clamp(min=0.0),
                requires_grad=True
            )
            for _ in range(self.config.nleapfrog)
        ])
        self.veps = nn.ParameterList([
            nn.parameter.Parameter(
                torch.tensor(
                    self.config.eps,
                ).clamp(min=0.0),
                requires_grad=True,
            )
            for _ in range(self.config.nleapfrog)
        ])
        assert isinstance(self.xeps, nn.ParameterList)
        assert isinstance(self.veps, nn.ParameterList)
        if torch.cuda.is_available():
            self.xeps = self.xeps.cuda()
            self.veps = self.veps.cuda()
            self.xnet.cuda()
            self.vnet.cuda()
            # self.networks.cuda()
            self.cuda()

    def _build_networks(self, network_factory: NetworkFactory):
        """Build networks."""
        split = self.config.use_split_xnets
        n = self.nlf if self.config.use_separate_networks else 1
        networks = network_factory.build_networks(n, split)
        # return networks['xnet'], networks['vnet']
        return networks

    @torch.no_grad()
    def init_weights(
            self,
            method='xavier_uniform',
            rval: Optional[float] = None,
            constant: Optional[float] = None,
            bias: Optional[float] = None,
            min: Optional[float] = None,
            max: Optional[float] = None,
            mean: Optional[float] = None,
            std: Optional[float] = None,
    ):
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method in ['zero', 'zeros']:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    if rval is not None:
                        nn.init.uniform_(m.weight, a=-rval, b=rval)
                    if bias is not None:
                        if m.bias is not None:
                            nn.init.constant_(m.bias, bias)
                    if constant is not None:
                        nn.init.constant_(m.weight, constant)
                    else:
                        if method == 'uniform':
                            a = 0.0 if min is None else min
                            b = 1.0 if max is None else max
                            nn.init.uniform_(m.weight, a=a, b=b)
                        elif method == 'normal':
                            mean = 0.0 if mean is None else mean
                            std = 1.0 if std is None else std
                            nn.init.normal_(m.weight, mean=mean, std=std)
                        elif method == 'xavier_uniform':
                            nn.init.xavier_uniform_(m.weight)
                        elif method == 'kaiming_uniform':
                            nn.init.kaiming_uniform_(m.weight)
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
                                    'Unable to initialize weights',
                                    f' with {method}',
                                    'Falling back to default: xavier_uniform_'
                                ]))
                                nn.init.xavier_uniform_(m.weight)

    def save(self, outdir: os.PathLike) -> None:
        netdir = Path(outdir).joinpath('networks')
        outfile = netdir.joinpath('dynamics.pt')
        netdir.mkdir(exist_ok=True, parents=True)
        assert isinstance(self.xeps, nn.ParameterList)
        assert isinstance(self.veps, nn.ParameterList)
        self.save_eps(outdir=netdir)
        torch.save(self.state_dict(), outfile.as_posix())

    def save_eps(self, outdir: os.PathLike) -> None:
        netdir = Path(outdir).joinpath('networks')
        fxeps = netdir.joinpath('xeps.npy')
        fveps = netdir.joinpath('veps.npy')
        assert isinstance(self.xeps, nn.ParameterList)
        assert isinstance(self.veps, nn.ParameterList)
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
            inputs: tuple[Tensor, Tensor]  # x, beta
    ) -> tuple[Tensor, dict]:
        if self.config.merge_directions:
            outputs = self.apply_transition_fb(inputs)
        else:
            outputs = self.apply_transition(inputs)

        return outputs

    def flatten(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], -1)

    def apply_transition_hmc(
            self,
            inputs: tuple[Tensor, Tensor],  # x, beta
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> tuple[Tensor, dict]:
        data = self.generate_proposal_hmc(inputs, eps=eps, nleapfrog=nleapfrog)
        ma_, mr_ = self._get_accept_masks(data['metrics']['acc'])
        ma_ = ma_.to(inputs[0].device)
        mr_ = mr_.to(inputs[0].device)
        # ma = ma_.unsqueeze(-1)
        # mr = mr_.unsqueeze(-1)
        ma = ma_[:, None]
        mr = mr_[:, None]

        xinit = data['init'].x.flatten(1)
        vinit = data['init'].v.flatten(1)
        xprop = data['proposed'].x.flatten(1)
        vprop = data['proposed'].v.flatten(1)

        vout = ma * vprop + mr * vinit
        xout = ma * xprop + mr * xinit
        # if isinstance(self.g, g.SU3):
        #     xout = self.g.vec_to_group(xout)

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
            inputs: tuple[Tensor, Tensor]  # x, beta
    ) -> tuple[Tensor, dict]:
        data = self.generate_proposal_fb(inputs)
        ma_, mr_ = self._get_accept_masks(data['metrics']['acc'])
        ma_ = ma_.to(inputs[0].device)
        mr_ = mr_.to(inputs[0].device)
        ma = ma_[:, None]
        mr = mr_[:, None]

        # NOTE: We construct output states by combining
        #   output = (acc_mask * proposed) + (reject_mask * init)
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
        xinit, beta = inputs
        if torch.rand(1) > 0.5:
            data = self.generate_proposal(inputs, forward=True)
        else:
            data = self.generate_proposal(inputs, forward=False)

        metrics = data['metrics']
        ma_, mr_ = self._get_accept_masks(metrics['acc'])
        if torch.cuda.is_available():
            ma_ = ma_.cuda()
            mr_ = mr_.cuda()
        # ma = ma_.unsqueeze(-1)
        # mr = mr_.unsqueeze(-1)
        ma = ma_[:, None]
        mr = mr_[:, None]

        v_out = ma * data['proposed'].v + mr * data['init'].v
        x_out = ma * data['proposed'].x + mr * xinit
        logdet = ma_ * metrics['sumlogdet']  # NOTE: + mr_ * logdet_init = 0

        state_out = State(x=x_out, v=v_out, beta=beta)
        mc_states = MonteCarloStates(init=data['init'],
                                     proposed=data['proposed'],
                                     out=state_out)

        metrics.update({
            'acc': metrics['acc'],
            'acc_mask': ma_,
            'sumlogdet': logdet,
            'mc_states': mc_states,
        })

        # metrics.update(**{f'fwd/{key}': val for key, val in mfwd.items()})
        # metrics.update(**{f'bwd/{key}': val for key, val in mbwd.items()})

        return x_out, metrics

    def apply_transition_both(
            self,
            inputs: tuple[Tensor, Tensor]  # x, beta
    ) -> tuple[Tensor, dict]:
        x, beta = inputs
        fwd = self.generate_proposal(inputs, forward=True)
        bwd = self.generate_proposal(inputs, forward=False)

        mf_, mb_ = self._get_direction_masks(batch_size=x.shape[0])
        if torch.cuda.is_available():
            mf_ = mf_.cuda()
            mb_ = mb_.cuda()
        # mf_ = mf_.to(x.device)
        # mb_ = mb_.to(x.device)
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
        # ma = ma_.unsqueeze(-1)
        # mr = mr_.unsqueeze(-1)
        ma = ma_[:, None]
        mr = mr_[:, None]
        # mr = mr_.unsqueeze(-1)

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

        # metrics.update(**{f'fwd/{key}': val for key, val in mfwd.items()})
        # metrics.update(**{f'bwd/{key}': val for key, val in mbwd.items()})

        return x_out, metrics

    def random_state(self, beta: float) -> State:
        x = self.g.random(list(self.xshape)).to(self.device)
        v = self.g.random_momentum(list(self.xshape)).to(x.device)
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
            inputs: tuple[Tensor, Tensor],  # x, beta
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> dict:
        x, beta = inputs
        # v = torch.randn_like(x).to(x.device)
        xshape = [x.shape[0], *self.xshape[1:]]
        v = self.g.random_momentum(xshape).to(x.device)
        # v = v.reshape_as(x).to(x.device)

        init = State(x=x, v=v, beta=beta)
        proposed, metrics = self.transition_kernel_hmc(
            init,
            eps=eps,
            nleapfrog=nleapfrog
        )

        return {'init': init, 'proposed': proposed, 'metrics': metrics}

    def generate_proposal_fb(
            self,
            inputs: tuple[Tensor, Tensor],  # x, beta
    ) -> dict:
        x, beta = inputs
        xshape = [x.shape[0], *self.xshape[1:]]
        v = self.g.random_momentum(xshape).to(x.device)
        # v = self.g.random_momentum(list(x.shape)).to(x.device)
        # v = torch.randn_like(x).to(x.device)
        init = State(x=x, v=v, beta=beta)
        proposed, metrics = self.transition_kernel_fb(init)

        return {'init': init, 'proposed': proposed, 'metrics': metrics}

    def generate_proposal(
            self,
            inputs: tuple[Tensor, Tensor],  # x, beta
            forward: bool,
    ) -> dict:
        x, beta = inputs
        xshape = [x.shape[0], *self.xshape[1:]]
        v = self.g.random_momentum(xshape).to(x.device)
        # v = self.g.random_momentum(list(self.xshape)).to(x.device)
        # v = torch.randn_like(x).to(x.device)
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
            assert isinstance(self.xeps, nn.ParameterList)
            assert isinstance(self.veps, nn.ParameterList)
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
                                dtype=state.x.real.dtype,
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
        sumlogdet = torch.zeros((state.x.shape[0],),
                                # dtype=state.x.real.dtype,
                                device=state.x.device)
        sldf = torch.zeros_like(sumlogdet)
        sldb = torch.zeros_like(sumlogdet)

        state_ = State(x=state.x, v=state.v, beta=state.beta)

        history = {}
        if self.config.verbose:
            extras = {'sldf': sldf, 'sldb': sldb, 'sld': sumlogdet}
            history = self.update_history(
                self.get_metrics(state_, sumlogdet, step=0, extras=extras),
                history=history,
            )

        # Forward
        for step in range(self.config.nleapfrog):
            state_, logdet = self._forward_lf(step, state_)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                sldf += logdet
                extras = {'sldf': sldf, 'sldb': sldb, 'sld': sumlogdet}
                metrics = self.get_metrics(
                    state_,
                    sumlogdet,
                    step=step,
                    extras=extras
                )
                history = self.update_history(metrics=metrics, history=history)

        # Flip momentum
        m1 = -1.0 * torch.ones_like(state_.v)
        state_ = State(state_.x, (m1 * state_.v), state_.beta)
        # sumlogdet *= -1

        # Backward
        for step in range(self.config.nleapfrog):
            state_, logdet = self._backward_lf(step, state_)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                sldb += logdet
                extras = {'sldf': sldf, 'sldb': sldb, 'sld': sumlogdet}
                # Reverse step count to correctly order metrics at each step
                metrics = self.get_metrics(
                    state_,
                    sumlogdet,
                    step=(self.config.nleapfrog-step-1),
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
                                dtype=state.x.dtype,
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
        dh = h_init - h_prop + sumlogdet
        prob = torch.exp(
            torch.minimum(dh, torch.zeros_like(dh, device=dh.device))
        ).to(state_init.x.device)

        # return prob
        # return torch.where(
        #     torch.isfinite(prob),
        #     prob,
        #     torch.zeros_like(prob)
        # )
        # return torch.where(prob.nan_to_num)
        # return prob.nan_to_num(0.)
        return prob

    @staticmethod
    def _get_accept_masks(px: Tensor) -> tuple[Tensor, Tensor]:
        acc = (
            px > torch.rand_like(px).to(px.device)
        ).to(torch.float)
        rej = torch.ones_like(acc) - acc

        # runif = torch.rand_like(px)  # .to(torch.float)
        # acc = (px > runif).to(torch.float)
        # rej = torch.ones_like(acc) - acc
        # acc = (px > torch.rand_like(px).to(px.device)).to(torch.float)
        # rej = torch.ones_like(acc) - acc
        # return acc.to(px.device), rej.to(px.device)
        return acc, rej

    @staticmethod
    def _get_direction_masks(batch_size: int) -> tuple[Tensor, Tensor]:
        """Returns (forward_mask, backward_mask)."""
        fwd = (torch.rand(batch_size) > 0.5).to(torch.float)
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
            # masks.append(torch.from_numpy(mask[None, :]).to(self.device))

        # return torch.stack(list(masks)).to(self.device)
        # return torch.from_numpy(np.array(masks)).float().to(self.device)
        return masks

    def _get_vnet(self, step: int) -> nn.Module:
        """Returns momentum network to be used for updating v."""
        # vnet = self.networks.get_submodule('vnet')
        if self.config.use_separate_networks:
            return self.vnet.get_submodule(str(step))
        return self.vnet

    def _get_xnet(self, step: int, first: bool = False) -> nn.Module:
        """Returns position network to be used for updating x."""
        # xnet = self.networks.get_submodule('xnet')
        if self.config.use_separate_networks:
            xnet = self.xnet.get_submodule(str(step))
            if self.config.use_split_xnets:
                if first:
                    return xnet.get_submodule('first')
                return xnet.get_submodule('second')
            return xnet
        return self.xnet

    def _stack_as_xy(self, x: Tensor) -> Tensor:
        """Returns -pi < x <= pi stacked as [cos(x), sin(x)]"""
        # TODO: Deal with ConvNet here
        # if self.config.use_conv_net:
        #     xcos = xcos.reshape(self.lattice_shape)
        #     xsin = xsin.reshape(self.lattice_shape)
        return torch.stack([x.cos(), x.sin()], dim=-1).to(self.device)

    def _call_vnet(
            self,
            step: int,
            inputs: tuple[Tensor, Tensor],  # (x, ∂S/∂x)
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Call the momentum update network used to update v.

        Args:
            inputs: (x, force) tuple
        Returns:
            s, t, q: Scaling, Translation, and Transformation functions
        """
        vnet = self._get_vnet(step)
        assert callable(vnet)
        x, force = inputs
        # if isinstance(self.g, U1Phase):
        # x = self._stack_as_xy(x)
        # elif isinstance(self.g, SU3):
        # x = self.g.group_to_vec(x)

        # x, force = x.to(self.device), force.to(self.device)
        if torch.cuda.is_available():
            x, force = x.cuda(), force.cuda()

        return vnet((x, force))

    def _call_vnet_dummy(
            self,
            step: int,
            inputs: tuple[Tensor, Tensor],  # (x, ∂S/∂x)
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Call the momentum update network used to update v.

        Args:
            inputs: (x, force) tuple
        Returns:
            s, t, q: Scaling, Translation, and Transformation functions
        """
        vnet = self._get_vnet(step)
        assert callable(vnet)
        x, force = inputs
        if torch.cuda.is_available():
            x, force = x.cuda(), force.cuda()

        return (
            torch.zeros_like(x),
            torch.zeros_like(x),
            torch.zeros_like(x),
        )

    def _call_xnet_dummy(
            self,
            step: int,
            inputs: tuple[Tensor, Tensor],
            first: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Call the position network used to update x.

        Args:
            inputs: (m * x, v) tuple, where (m * x) is a masking operation.
        Returns:
            s, t, q: Scaling, Translation, and Transformation functions
        """
        xnet = self._get_xnet(step, first)
        assert callable(xnet)
        x, v = inputs
        x = self.g.group_to_vec(x)

        # x, v = x.to(self.device), v.to(self.device)
        if torch.cuda.is_available():
            x, v = x.cuda(), v.cuda()

        return (
            torch.zeros_like(v),
            torch.zeros_like(v),
            torch.zeros_like(v),
        )

    def _call_xnet(
            self,
            step: int,
            inputs: tuple[Tensor, Tensor],
            first: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Call the position network used to update x.

        Args:
            inputs: (m * x, v) tuple, where (m * x) is a masking operation.
        Returns:
            s, t, q: Scaling, Translation, and Transformation functions
        """
        xnet = self._get_xnet(step, first)
        assert callable(xnet)
        x, v = inputs
        x = self.g.group_to_vec(x)

        # x, v = x.to(self.device), v.to(self.device)
        if torch.cuda.is_available():
            x, v = x.cuda(), v.cuda()

        return xnet((x, v))

    def _forward_lf(self, step: int, state: State) -> tuple[State, Tensor]:
        """Complete update (leapfrog step) in the forward direction. """
        m, mb = self._get_mask(step)
        m, mb = m.to(self.device), mb.to(self.device)
        sumlogdet = torch.zeros(state.x.shape[0],
                                # dtype=state.x.real.dtype,
                                device=self.device)

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
        m, mb = m.to(self.device), mb.to(self.device)
        sumlogdet = torch.zeros((state.x.shape[0],),
                                # dtype=state.x.real.dtype,
                                device=self.device)

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
        assert isinstance(self.veps, nn.ParameterList)
        eps = self.veps[step]
        force = self.grad_potential(state.x, state.beta)
        s, t, q = self._call_vnet(step, (state.x, force))

        jac = eps * s / 2.  # jacobian factor, also used in exp_s below
        logdet = jac.sum(dim=1)
        exp_s = jac.exp()
        exp_q = (eps * q).exp()
        # exp_s = torch.exp(jac)
        # exp_q = torch.exp(eps * q)
        vf = exp_s * state.v - 0.5 * eps * (force * exp_q + t)

        return State(state.x, vf, state.beta), logdet

    def _update_v_bwd(self, step: int, state: State) -> tuple[State, Tensor]:
        """Single v update in the backward direction"""
        assert isinstance(self.veps, nn.ParameterList)
        eps = self.veps[step]
        force = self.grad_potential(state.x, state.beta)
        s, t, q = self._call_vnet(step, (state.x, force))

        logjac = (-eps * s / 2.)  # jacobian factor, also used in exp_s below
        logdet = logjac.sum(dim=1)
        exp_s = torch.exp(logjac)
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
        assert isinstance(self.xeps, nn.ParameterList)
        eps = self.xeps[step]
        mb = (torch.ones_like(m) - m).to(self.device)
        xm_init = m * state.x
        inputs = (xm_init, state.v)
        s, t, q = self._call_xnet(step, inputs, first=first)
        # s, t, q = self._call_xnet_dummy(step, inputs, first=first)
        s = eps * s
        q = eps * q
        exp_s = s.exp()
        exp_q = q.exp()
        if isinstance(self.g, U1Phase):
            if self.config.use_ncp:
                halfx = state.x / 2.
                _x = 2. * (halfx.tan() * exp_s).atan()
                xp = _x + eps * (state.v * exp_q + t)
                xf = xm_init + (mb * xp)
                cterm = (halfx.cos()) ** 2
                sterm = (exp_s * halfx.sin()) ** 2
                logdet_ = (exp_s / (cterm + sterm)).log()
                logdet = (mb * logdet_).sum(dim=1)
            else:
                xp = state.x * exp_s + eps * (state.v * exp_q + t)
                xf = xm_init + (mb * xp)
                logdet = (mb * s).sum(dim=1)

        elif isinstance(self.g, SU3):
            x = self.g.group_to_vec(state.x)
            # v = self.g.group_to_vec(state.v)
            xm_init = self.g.group_to_vec(xm_init)
            xp = x * exp_s + eps * (state.v * exp_q + t)
            xf = xm_init + (mb * xp)
            # xp = state.x * exp_s + eps * (state.v * exp_q + t)
            # xf = xm_init + (mb * xp)
            logdet = (mb * s).sum(dim=1)
        else:
            raise ValueError('Unexpected value for `self.g`')

        # xf = self.g.compat_proj(xf)
        return State(x=xf, v=state.v, beta=state.beta), logdet

    def _update_x_bwd(
            self,
            step: int,
            state: State,
            m: Tensor,
            first: bool,
    ) -> tuple[State, Tensor]:
        """Update the position in the backward direction."""
        assert isinstance(self.xeps, nn.ParameterList)
        eps = self.xeps[step]
        mb = (torch.ones_like(m) - m).to(self.device)
        xm_init = m * state.x
        inputs = (xm_init, state.v)
        s, t, q = self._call_xnet(step, inputs, first=first)
        # s, t, q = self._call_xnet_dummy(step, inputs, first=first)
        s = (-eps) * s
        q = eps * q
        exp_s = s.exp()
        exp_q = q.exp()
        # exp_s = torch.exp(s)
        # exp_q = torch.exp(q)
        if isinstance(self.g, U1Phase):
            if self.config.use_ncp:
                halfx = state.x / 2.
                halfx_scale = exp_s * halfx.tan()
                x1 = 2. * halfx_scale.atan()
                x2 = exp_s * eps * (state.v * exp_q + t)
                xnew = x1 - x2
                xb = xm_init + (mb * xnew)

                cterm = halfx.cos() ** 2
                sterm = (exp_s * halfx.sin()) ** 2
                logdet_ = (exp_s / (cterm + sterm)).log()
                logdet = (mb * logdet_).sum(dim=1)
            else:
                xnew = exp_s * (state.x - eps * (state.v * exp_q + t))
                xb = xm_init + (mb * xnew)
                logdet = (mb * s).sum(dim=1)
        elif isinstance(self.g, SU3):
            xnew = exp_s * (state.x - eps * (state.v * exp_q + t))
            xb = xm_init + (mb * xnew)
            logdet = (mb * s).sum(dim=1)
        else:
            raise ValueError('Unexpected value for `self.g`')

        # xb = self.g.compat_proj(xb)
        return State(x=xb, v=state.v, beta=state.beta), logdet

    def hamiltonian(self, state: State) -> Tensor:
        """Returns the total energy H = KE + PE."""
        kinetic = self.kinetic_energy(state.v)
        potential = self.potential_energy(state.x, state.beta)
        return kinetic + potential

    def kinetic_energy(self, v: Tensor) -> Tensor:
        """Returns the kinetic energy, KE = 0.5 * v ** 2."""
        # return 0.5 * v.reshape(v.shape[0], -1).square()
        return self.g.kinetic_energy(v)

    def potential_energy(self, x: Tensor, beta: Tensor):
        """Returns the potential energy, PE = beta * action(x)."""
        # return beta * self.potential_fn(x)
        return self.potential_fn(x, beta)

    def grad_potential(
            self,
            x: Tensor,
            beta: Tensor,
            # create_graph: bool = True,
    ) -> Tensor:
        """Compute the gradient of the potential function."""
        x = x.to(self.device)
        x.requires_grad_(True)
        s = self.potential_energy(x, beta)
        id = torch.ones(x.shape[0], device=self.device)
        dsdx, = torch.autograd.grad(s, x,
                                    create_graph=True,
                                    retain_graph=True,
                                    # create_graph=create_graph,
                                    # retain_graph=True,
                                    grad_outputs=id)
        return dsdx
