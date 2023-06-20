"""
tensorflow/dynamics.py

Tensorflow implementation of Dynamics object for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass
from math import pi
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional
from typing import Tuple
import logging
# from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import IndexedSlices

from l2hmc import configs as cfgs
from l2hmc.network.tensorflow.network import dummy_network, NetworkFactory
from l2hmc.group.u1.tensorflow.group import U1Phase
from l2hmc.group.su3.tensorflow.group import SU3
from l2hmc.lattice.u1.tensorflow.lattice import LatticeU1
from l2hmc.lattice.su3.tensorflow.lattice import LatticeSU3


Tensor = tf.Tensor
Model = tf.keras.Model
TensorLike = tf.types.experimental.TensorLike
TF_FLOAT = tf.dtypes.as_dtype(tf.keras.backend.floatx())

PI = tf.constant(pi, dtype=TF_FLOAT)
TWO = tf.constant(2., dtype=TF_FLOAT)
TWO_PI = TWO * PI

log = logging.getLogger(__name__)


DynamicsOutput = Tuple[TensorLike, dict]


def to_u1(x: Tensor) -> Tensor:
    return (tf.add(x, PI) % TWO_PI) - PI


@dataclass
class State:
    x: Tensor
    v: Tensor
    beta: Tensor

    def flatten(self):
        x = tf.reshape(self.x, (self.x.shape[0], -1))
        v = tf.reshape(self.v, (self.v.shape[0], -1))
        beta = tf.constant(tf.cast(self.beta, x.dtype))
        return State(x, v, beta)

    def __post_init__(self):
        assert isinstance(self.x, Tensor)
        assert isinstance(self.v, Tensor)
        assert isinstance(self.beta, Tensor)

    def to_numpy(self):
        if tf.executing_eagerly():
            return {
                'x': self.x.numpy(),  # type:ignore
                'v': self.v.numpy(),  # type:ignore
                'beta': self.beta.numpy(),  # type:ignore
            }
        return {
            'x': self.x,
            'v': self.v,
            'beta': self.beta,
        }


@dataclass
class MonteCarloStates:
    init: State
    proposed: State
    out: State


@dataclass
class MonteCarloProposal:
    init: State
    proposed: State


def xy_repr(x: Tensor) -> Tensor:
    return tf.stack([tf.math.cos(x), tf.math.sin(x)], axis=-1)


def sigmoid(x: Tensor | Any) -> Tensor:
    return 1. / (1. + tf.exp(tf.negative(x)))


class Dynamics(Model):
    def __init__(
            self,
            potential_fn: Callable,
            config: cfgs.DynamicsConfig,
            network_factory: Optional[NetworkFactory] = None,
    ):
        """Initialization."""
        super(Dynamics, self).__init__()
        self.config = config
        self.group = config.group
        self.xdim = self.config.xdim
        self.xshape = self.config.xshape
        self.potential_fn = potential_fn
        self.nlf = self.config.nleapfrog

        if self.config.group == 'U1':
            self.g = U1Phase()
            self.lattice = LatticeU1(self.config.nchains,
                                     self.config.latvolume)
        elif self.config.group == 'SU3':
            self.g = SU3()
            self.lattice = LatticeSU3(self.config.nchains,
                                      self.config.latvolume)
        else:
            raise ValueError('Unexpected value for `self.config.group`')

        assert isinstance(self.g, (U1Phase, SU3))
        self.network_factory = network_factory
        if network_factory is not None:
            self._networks_built = True
            self.networks = self._build_networks(network_factory)
            self.xnet = self.networks['xnet']
            self.vnet = self.networks['vnet']
            # self.networks = {'xnet': self.xnet, 'vnet': self.vnet}
        else:
            self._networks_built = False
            self.xnet = dummy_network
            self.vnet = dummy_network
            self.networks = {
                'xnet': self.xnet,
                'vnet': self.vnet
            }
        self.masks = self._build_masks()
        self.xeps = []
        self.veps = []
        ekwargs = {
            # 'dtype': TF_FLOAT,
            'initial_value': self.config.eps,
            'trainable': (not self.config.eps_fixed),
            'constraint': tf.keras.constraints.non_neg(),
        }
        for lf in range(self.config.nleapfrog):
            xalpha = tf.Variable(name=f'xeps_lf{lf}', **ekwargs)
            valpha = tf.Variable(name=f'veps_lf{lf}', **ekwargs)
            self.xeps.append(xalpha)
            self.veps.append(valpha)

    def get_models(self) -> dict:
        if self.config.use_separate_networks:
            xnet = {}
            vnet = {}
            for lf in range(self.config.nleapfrog):
                vnet[str(lf)] = self._get_vnet(lf)
                xnet[f'{lf}/first'] = self._get_xnet(lf, first=True)
                if self.config.use_split_xnets:
                    xnet[f'{lf}/second'] = self._get_xnet(lf, first=False)
                    # xnet[str(lf)] = {
                    #     '0': self._get_xnet(lf, first=True),
                    #     '1': self._get_xnet(lf, first=False),
                    # }
                else:
                    xnet[str(lf)] = self._get_xnet(lf, first=True)
        else:
            vnet = self._get_vnet(0)
            xnet = self._get_xnet(0, first=True)
            if self.config.use_split_xnets:
                xnet1 = self._get_xnet(0, first=False)
                xnet = {
                    'first': xnet,
                    'second': xnet1,
                }
                # xnet = {
                #     '0': self._get_xnet(0, first=True),
                #     '1': self._get_xnet(0, first=False),
                # }
            else:
                xnet = self._get_xnet(0, first=True)

        return {'xnet': xnet, 'vnet': vnet}

    def get_weights_dict(self) -> dict:
        weights = {}
        if self.config.use_separate_networks:
            for lf in range(self.config.nleapfrog):
                vnet = self._get_vnet(lf)
                weights |= vnet.get_weights_dict()
                # weights.update(vnet.get_weights_dict())

                xnet0 = self._get_xnet(lf, first=True)
                # weights.update(xnet0.get_weights_dict())
                weights |= xnet0.get_weights_dict()
                if self.config.use_split_xnets:
                    xnet1 = self._get_xnet(lf, first=False)
                    # weights.update(xnet1.get_weights_dict())
                    weights |= xnet1.get_weights_dict()
        else:
            vnet = self._get_vnet(0)
            weights = vnet.get_weights_dict()
            xnet0 = self._get_xnet(0, first=True)
            weights.update(xnet0.get_weights_dict())
            if self.config.use_split_xnets:
                xnet1 = self._get_xnet(0, first=False)
                weights.update(xnet1.get_weights_dict())

        weights = {
            f'model/{k}': v for k, v in weights.items()
        }
        return weights

    def _build_networks(
            self,
            network_factory: NetworkFactory
    ) -> dict:
        """Build networks."""
        split = self.config.use_split_xnets
        n = self.nlf if self.config.use_separate_networks else 1
        # return networks['xnet'], networks['vnet']
        # return networks
        return network_factory.build_networks(
            n,
            split,
            group=self.g,
        )

    def call(
            self,
            inputs: tuple[Tensor, Tensor],
            training: bool = True
    ) -> tuple[Tensor, dict]:
        """Call Dynamics object.

        Args:
            inputs: Pair of inputs: (x, β) to use for generating new state x'.
            training (bool): Indicates training or evaluation of model
        """
        if self.config.merge_directions:
            return self.apply_transition_fb(inputs, training=training)
        return self.apply_transition(inputs, training=training)

    @staticmethod
    def flatten(x: Tensor | IndexedSlices | Any) -> Tensor:
        return tf.reshape(x, (x.shape[0], -1))

    def random_state(self, beta: float = 1.) -> State:
        """Returns a random State."""
        x = self.flatten(self.g.random(list(self.xshape)))
        v = self.flatten(self.g.random_momentum(list(self.xshape)))
        return State(x=x, v=v, beta=tf.constant(beta, dtype=TF_FLOAT))

    def test_reversibility(self) -> dict[str, Tensor]:
        """Test reversibility i.e. backward(forward(state)) = state"""
        state = self.random_state(beta=1.)
        state_fwd, _ = self.transition_kernel(state, forward=True)
        state_, _ = self.transition_kernel(state_fwd, forward=False)
        dx = tf.abs(tf.subtract(state.x, state_.x))
        dv = tf.abs(tf.subtract(state.v, state_.v))

        return {'dx': dx, 'dv': dv}

    def apply_transition_hmc(
            self,
            inputs: tuple[Tensor, Tensor],
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> tuple[Tensor, dict]:
        data = self.generate_proposal_hmc(inputs, eps, nleapfrog=nleapfrog)
        ma_, mr_ = self._get_accept_masks(data['metrics']['acc'])
        ma_ = tf.constant(ma_, dtype=TF_FLOAT)
        mr_ = tf.constant(mr_, dtype=TF_FLOAT)
        ma = ma_[:, None]

        xinit = self.flatten(data['init'].x)
        vinit = self.flatten(data['init'].v)
        xprop = self.flatten(data['proposed'].x)
        vprop = self.flatten(data['proposed'].v)

        vout = tf.where(tf.cast(ma, bool), vprop, vinit)
        xout = tf.where(tf.cast(ma, bool), xprop, xinit)

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
            inputs: tuple[Tensor, Tensor],
            training: bool = True,
    ) -> tuple[Tensor, dict]:
        """Apply transition using single forward/backward update."""
        data = self.generate_proposal_fb(inputs, training=training)
        ma_, _ = self._get_accept_masks(data['metrics']['acc'])
        ma = ma_[:, None]
        vprop = self.flatten(data['proposed'].v)
        xprop = self.flatten(data['proposed'].x)
        v_out = tf.where(
            tf.cast(ma, bool),
            vprop,
            tf.cast(self.flatten(data['init'].v), vprop.dtype)
        )
        x_out = tf.where(
            tf.cast(ma, bool),
            xprop,
            tf.cast(self.flatten(data['init'].x), xprop.dtype)
        )
        sld = data['metrics']['sumlogdet']
        sumlogdet = tf.cast(ma_, sld.dtype) * sld

        state_out = State(x=x_out, v=v_out, beta=data['init'].beta)
        mc_states = MonteCarloStates(
            init=data['init'],
            proposed=data['proposed'],
            out=state_out
        )
        data['metrics'].update({
            'acc_mask': ma_,
            'sumlogdet': sumlogdet,
            'mc_states': mc_states,
        })

        return x_out, data['metrics']

    def apply_transition(
            self,
            inputs: tuple[Tensor, Tensor],
            training: bool = True
    ) -> tuple[Tensor,  dict]:
        """Apply transition using masks to combine forward/backward updates."""
        x, beta = inputs
        fwd = self.generate_proposal(inputs, forward=True, training=training)
        bwd = self.generate_proposal(inputs, forward=False, training=training)

        # assert isinstance(x, Tensor)
        mf_, mb_ = self._get_direction_masks(batch_size=x.shape[0])
        mf = mf_[:, None]
        mb = mb_[:, None]

        x_init = tf.where(tf.cast(mf, bool), fwd['init'].x, bwd['init'].x)
        v_init = tf.where(tf.cast(mf, bool), fwd['init'].v, bwd['init'].v)

        x_prop = tf.where(
            tf.cast(mf, bool),
            fwd['proposed'].x,
            bwd['proposed'].x
        )
        v_prop = tf.where(
            tf.cast(mf, bool),
            fwd['proposed'].v,
            bwd['proposed'].v
        )

        mfwd = fwd['metrics']
        mbwd = bwd['metrics']

        logdet_prop = tf.where(
            tf.cast(mf_, bool),
            mfwd['sumlogdet'],
            mbwd['sumlogdet']
        )

        acc = mf_ * mfwd['acc'] + mb_ * mbwd['acc']
        ma_, _ = self._get_accept_masks(acc)
        ma = ma_[:, None]

        v_out = tf.where(
            tf.cast(ma, bool),
            v_prop,
            v_init
        )
        x_out = tf.where(
            tf.cast(ma, bool),
            x_prop,
            x_init,
        )
        sumlogdet = tf.where(
            tf.cast(ma_, bool),
            logdet_prop,
            tf.zeros_like(logdet_prop)
        )

        init = State(x=x, v=v_init, beta=beta)
        prop = State(x=x_prop, v=v_prop, beta=beta)
        out = State(x=x_out, v=v_out, beta=beta)

        mc_states = MonteCarloStates(init=init, proposed=prop, out=out)

        metrics = {}
        for (key, vf), (_, vb) in zip(mfwd.items(), mbwd.items()):
            try:
                vfb = ma_ * (mf_ * vf + mb_ * vb)  # + mr_ * v0
            except ValueError:
                vfb = ma * (mf * vf + mb * vb)  # + mr * v0

            metrics[key] = vfb

        metrics.update({
            'acc': acc,
            'acc_mask': ma_,
            'sumlogdet': sumlogdet,
            'mc_states': mc_states,
        })

        return x_out, metrics

    def generate_proposal_hmc(
            self,
            inputs: tuple[Tensor, Tensor],
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> dict:
        x, beta = inputs
        assert isinstance(x, Tensor)
        xshape = [x.shape[0], *self.xshape[1:]]
        v = self.g.random_momentum(xshape)
        init = State(x, v, beta)
        proposed, metrics = self.transition_kernel_hmc(init,
                                                       eps=eps,
                                                       nleapfrog=nleapfrog)

        return {'init': init, 'proposed': proposed, 'metrics': metrics}

    def generate_proposal_fb(
            self,
            inputs: tuple[Tensor, Tensor],
            training: bool = True,
    ) -> dict:
        """Generate proposal using single forward/backward update.

        Inputs:
          inputs: Tuple of (x, beta)
          training: Currently training model?

        Returns dict of 'init', and 'proposed' states, along with 'metrics'.
        """
        x, beta = inputs
        assert isinstance(x, Tensor)
        xshape = [x.shape[0], *self.xshape[1:]]
        v = self.flatten(self.g.random_momentum(xshape))
        init = State(x, v, beta)
        proposed, metrics = self.transition_kernel_fb(init, training=training)

        return {'init': init, 'proposed': proposed, 'metrics': metrics}

    def generate_proposal(
            self,
            inputs: tuple[Tensor, Tensor],
            forward: bool,
            training: bool = True,
    ) -> dict:
        """Generate proposal using direction specified by 'forward'.

        Returns dict of 'init', and 'proposed' states, along with 'metrics'.
        """
        x, beta = inputs
        assert isinstance(x, Tensor)
        xshape = [x.shape[0], *self.xshape[1:]]
        v = self.flatten(self.g.random_momentum(xshape))
        state_init = State(x=x, v=v, beta=beta)
        state_prop, metrics = self.transition_kernel(state_init,
                                                     forward=forward,
                                                     training=training)

        return {'init': state_init, 'proposed': state_prop, 'metrics': metrics}

    def get_metrics(
            self,
            state: State,
            logdet: Tensor,
            step: Optional[int] = None,
            extras: Optional[dict[str, Tensor]] = None,
    ) -> dict:
        """Returns dict of various metrics about input State."""
        energy = self.hamiltonian(state)
        logprob = tf.subtract(energy, tf.cast(logdet, energy.dtype))
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
                'veps': self.veps[step],
            })

        return metrics

    def update_history(
            self,
            metrics: dict,
            history: dict,
    ) -> dict:
        """Update history with items from metrics."""
        for key, val in metrics.items():
            try:
                history[key].append(val)
            except KeyError:
                history[key] = [val]

        return history

    def leapfrog_hmc(
            self,
            state: State,
            eps: float | tf.Tensor,
    ) -> State:
        """Perform standard HMC leapfrog update."""
        x = tf.reshape(state.x, state.v.shape)
        force = self.grad_potential(x, state.beta)        # f = dU / dx
        eps = tf.constant(eps, dtype=force.dtype)
        # halfeps = tf.cast(tf.scalar_mul(0.5, eps), dtype=force.dtype)
        # halfeps = tf.scalar_mul(0.5, eps)
        # halfeps = tf.constant(eps / 2.0, dtype=force.dtype)
        halfeps = 0.5 * eps  # type:ignore
        v = state.v - halfeps * force
        x = self.g.update_gauge(x, eps * v)
        force = self.grad_potential(x, state.beta)       # calc force, again
        v -= halfeps * force
        return State(x=x, v=v, beta=state.beta)          # output: (x', v')

    def transition_kernel_hmc(
            self,
            state: State,
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> tuple[State, dict]:
        """Run the generic HMC transition kernel."""
        state_ = State(x=state.x, v=state.v, beta=state.beta)
        assert isinstance(state.x, Tensor)
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=state.x.dtype)
        history = {}
        if self.config.verbose:
            history = self.update_history(
                self.get_metrics(state_, sumlogdet),
                history={}
            )
        eps = self.config.eps_hmc if eps is None else eps
        nlf = (
            2 * self.config.nleapfrog if self.config.merge_directions
            else self.config.nleapfrog
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
                    history[key] = tf.stack(val)

        return state_, history

    def transition_kernel_fb(
            self,
            state: State,
            training: bool = True,
    ) -> tuple[State, dict]:
        """Run the transition kernel using single forward/backward update.

        Returns:
         tuple of output state, and history of metrics tracked during traj.
        """
        state_ = State(state.x, state.v, state.beta)
        assert isinstance(state.x, Tensor)
        sumlogdet = tf.zeros(
            (state.x.shape[0],),
            # dtype=tf.math.real(state.x).dtype
        )
        sldf = tf.zeros_like(sumlogdet)
        sldb = tf.zeros_like(sumlogdet)

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
            state_, logdet = self._forward_lf(step, state_, training)
            logdet = tf.cast(logdet, sumlogdet.dtype)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
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
        state_ = State(state_.x, tf.negative(state_.v), state_.beta)  # noqa
        # Backward
        for step in range(self.config.nleapfrog):
            state_, logdet = self._backward_lf(step, state_, training)
            logdet = tf.cast(logdet, sumlogdet.dtype)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                # sldb += logdet
                sldb = sldb + logdet
                extras = {
                    'sldf': tf.zeros_like(sldb),
                    'sldb': sldb,
                    # 'sldfb': sldf + sldb,
                    'sld': sumlogdet,
                }
                # Reverse step count to correctly order metrics
                metrics = self.get_metrics(
                    state_,
                    sumlogdet,
                    step=(self.config.nleapfrog - step - 1),
                    extras=extras
                )
                history = self.update_history(metrics=metrics, history=history)

        acc = self.compute_accept_prob(state, state_, sumlogdet)
        history.update({'acc': acc, 'sumlogdet': sumlogdet})
        if self.config.verbose:
            for key, val in history.items():
                if isinstance(val, list) and isinstance(val[0], Tensor):
                    history[key] = tf.stack(val)

        return state_, history

    def transition_kernel(
            self,
            state: State,
            forward: bool,
            training: bool = True,
    ) -> tuple[State, dict]:
        """Implements the directional transition kernel.

        Returns:
         tuple of output state, and history of metrics tracked during traj.
        """
        lf_fn = self._forward_lf if forward else self._backward_lf

        # Copy initial state into proposed state
        state_ = State(x=state.x, v=state.v, beta=state.beta)
        assert isinstance(state.x, Tensor)
        sumlogdet = tf.zeros((state.x.shape[0],))

        history = {}
        if self.config.verbose:
            metrics = self.get_metrics(state_, sumlogdet)
            history = self.update_history(metrics, history=history)

        for step in range(self.config.nleapfrog):
            state_, logdet = lf_fn(step, state_, training)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                metrics = self.get_metrics(state_, sumlogdet, step=step)
                history = self.update_history(metrics, history=history)

        acc = self.compute_accept_prob(state, state_, sumlogdet)
        history.update({'acc': acc, 'sumlogdet': sumlogdet})
        if self.config.verbose:
            for key, val in history.items():
                if isinstance(val, list) and isinstance(val[0], Tensor):
                    history[key] = tf.stack(val)  # type: ignore

        return state_, history

    def compute_accept_prob(
            self,
            state_init: State,
            state_prop: State,
            sumlogdet: Tensor,
    ) -> Tensor:
        """Compute the acceptance probability."""
        h_init = self.hamiltonian(state_init)
        h_prop = self.hamiltonian(state_prop)
        dh = tf.add(
            tf.subtract(h_init, h_prop),
            tf.cast(sumlogdet, h_init.dtype)
        )
        # dh = h_init - h_prop + sumlogdet
        prob = tf.exp(tf.minimum(dh, tf.zeros_like(dh)))

        return tf.where(tf.math.is_finite(prob), prob, tf.zeros_like(prob))

    @staticmethod
    def _get_accept_masks(px: Tensor) -> tuple:
        """Convert acceptance probability to binary mask of accept/rejects."""
        acc = tf.cast(
            px > tf.random.uniform(tf.shape(px), dtype=TF_FLOAT),
            dtype=TF_FLOAT,
        )
        rej = tf.ones_like(acc) - acc

        return (acc, rej)

    def _get_direction_masks(self, batch_size) -> tuple:
        """Get masks for combining forward/backward updates."""
        fwd = tf.cast(
            tf.random.uniform((batch_size,), dtype=TF_FLOAT) > 0.5,
            dtype=TF_FLOAT,
        )
        bwd = tf.ones_like(fwd) - fwd

        return fwd, bwd

    def _get_mask(self, i: int) -> tuple[Tensor, Tensor]:
        """Returns mask used for sequentially updating x."""
        m = self.masks[i]
        mb = tf.ones_like(m) - m
        return (m, mb)

    def _build_masks(self):
        """Construct different binary masks for different lf steps."""
        masks = []
        for _ in range(self.config.nleapfrog):
            # Need to use numpy.random here bc tf would generate different
            # random values across different calls
            _idx = np.arange(self.xdim)
            idx = np.random.permutation(_idx)[:self.xdim // 2]
            mask = np.zeros((self.xdim,))
            mask[idx] = 1.
            mask = tf.constant(mask, dtype=TF_FLOAT)
            masks.append(mask[None, :])

        return masks

    def _get_vnet(self, step: int) -> Callable:
        """Returns momentum network to be used for updating v."""
        if not self._networks_built:
            return self.vnet
        vnet = self.vnet
        # assert isinstance(vnet, (dict, tf.keras.Model))
        if self.config.use_separate_networks and isinstance(vnet, dict):
            return vnet[str(step)]
        # assert isinstance(vnet, (CallableNetwork))
        return self.vnet

    def _get_xnets(
            self,
            step: int,
    ) -> list:
        xnets = [
            self._get_xnet(step, first=True)
        ]
        if self.config.use_separate_networks:
            xnets.append(
                self._get_xnet(step, first=False)
            )
        return xnets

    def _get_all_xnets(self) -> list[Model]:
        xnets = []
        for step in range(self.config.nleapfrog):
            nets = self._get_xnets(step)
            for net in nets:
                xnets.append(net)
        return xnets

    def _get_all_vnets(self) -> list:
        return [
            self._get_vnet(step)
            for step in range(self.config.nleapfrog)
        ]
        # for step in range(self.config.nleapfrog):
        #     nets = self._get_vnet(step)

    @staticmethod
    def rename_weight(
            name: str,
            sep: Optional[str] = None,
    ) -> str:
        new_name = (
            name.rstrip(':0').replace('kernel', 'weight')
        )
        new_name = re.sub(r'\_\d', '', new_name)
        if sep is not None:
            new_name.replace('.', '/')
            new_name.replace('/', sep)

        return new_name

    def get_all_weights(self) -> dict:
        xnets = self._get_all_xnets()
        vnets = self._get_all_vnets()
        weights = {}
        for xnet in xnets:
            weights.update({
                f'{self.rename_weight(w.name)}': w
                for w in xnet.weights
            })
        for vnet in vnets:
            weights.update({
                # self.format_weight_name(w.name): w
                f'{self.rename_weight(w.name)}': w
                for w in vnet.weights
            })

        return cfgs.flatten_dict(weights)

    def _get_xnet(
            self,
            step: int,
            first: bool
    ) -> Callable:
        """Returns position network to be used for updating x."""
        if not self._networks_built:
            return self.xnet
        xnet = self.xnet
        # assert isinstance(xnet, (tf.keras.Model, dict))
        if self.config.use_separate_networks and isinstance(xnet, dict):
            xnet = xnet[str(step)]
            if self.config.use_split_xnets:
                if first:
                    return xnet['first']
                return xnet['second']
            return xnet
        return xnet

    def _stack_as_xy(self, x: Tensor) -> Tensor:
        """Returns -pi < x <= pi stacked as [cos(x), sin(x)]"""
        return tf.stack([tf.math.cos(x), tf.math.sin(x)], axis=-1)

    def _call_vnet(
            self,
            step: int,
            inputs: tuple[Tensor, Tensor],  # (x, ∂S/∂x)
            training: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Calls the momentum network used to update v.

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
        s, t, q = vnet((x, force), training)
        # return (
        #     tf.cast(s, TF_FLOAT),
        #     tf.cast(t, TF_FLOAT),
        #     tf.cast(q, TF_FLOAT)
        # )
        return (s, t, q)

    def _call_xnet(
            self,
            step: int,
            inputs: tuple[Tensor, Tensor],  # (m * x, v)
            first: bool,
            training: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Call the position network used to update x.

        Args:
            inputs: (m * x, v) tuple, where (m * x) is a masking operation.
        Returns:
            s, t, q: Scaling, Translation, and Transformation functions
        """
        x, v = inputs
        assert isinstance(x, Tensor) and isinstance(v, Tensor)
        xnet = self._get_xnet(step, first)
        if self.config.group == 'U1':
            x = self.g.group_to_vec(x)

        elif self.config.group == 'SU3':
            x = self.unflatten(x)
            x = tf.stack([tf.math.real(x), tf.math.imag(x)], 1)
            v = tf.stack([tf.math.real(v), tf.math.imag(v)], 1)
        # s, t, q = xnet((x, v), training)
        # return (s, t, q)
        # return xnet((x, v), training=training)
        s, t, q = xnet((x, v), training=training)
        # return (
        #     tf.cast(s, TF_FLOAT),
        #     tf.cast(t, TF_FLOAT),
        #     tf.cast(q, TF_FLOAT)
        # )
        return (s, t, q)

    def _forward_lf(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Complete update (leapfrog step) in the forward direction."""
        m, mb = self._get_mask(step)
        # m = tf.cast(m, state.x.dtype)
        # mb = tf.cast(mb, state.x.dtype)
        # sumlogdet = tf.zeros((state.x.shape[0],), dtype=state.x.dtype)
        # assert isinstance(state.x, Tensor)
        # assert isinstance(m, Tensor) and isinstance(mb, Tensor)
        # sumlogdet = sumlogdet + tf.cast(logdet, sumlogdet.dtype)
        state, logdet = self._update_v_fwd(step, state, training=training)
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=logdet.dtype)
        sumlogdet = sumlogdet + logdet
        state, logdet = self._update_x_fwd(step, state, m,
                                           first=True, training=training)
        sumlogdet = sumlogdet + logdet
        state, logdet = self._update_x_fwd(step, state, mb,
                                           first=False, training=training)
        sumlogdet = sumlogdet + logdet
        state, logdet = self._update_v_fwd(step, state, training=training)
        sumlogdet = sumlogdet + logdet

        return state, sumlogdet

    def _backward_lf(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Complete update (leapfrog step) in the backward direction."""
        # Note: Reverse the step count, i.e. count from end of trajectory.
        step_r = self.config.nleapfrog - step - 1
        m, mb = self._get_mask(step_r)
        # sumlogdet = tf.zeros((state.x.shape[0],), dtype=state.x.dtype)
        # m = tf.cast(m, state.x.dtype)
        # mb = tf.cast(mb, state.x.dtype)
        # assert isinstance(m, Tensor) and isinstance(mb, Tensor)
        # sumlogdet = sumlogdet + tf.cast(logdet, sumlogdet.dtype)

        state, logdet = self._update_v_bwd(step_r, state, training=training)
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=logdet.dtype)
        sumlogdet = sumlogdet + logdet
        state, logdet = self._update_x_bwd(step_r, state, mb,
                                           first=False, training=training)
        sumlogdet = sumlogdet + logdet
        state, logdet = self._update_x_bwd(step_r, state, m,
                                           first=True, training=training)
        sumlogdet = sumlogdet + logdet
        state, logdet = self._update_v_bwd(step_r, state, training=training)
        sumlogdet = sumlogdet + logdet

        return state, sumlogdet

    def unflatten(self, x: Tensor) -> Tensor:
        return tf.reshape(x, (x.shape[0], *self.xshape[1:]))

    def group_to_vec(self, x: Tensor) -> Tensor:
        """For x in SU(3), returns an 8-component real-valued vector"""
        return self.g.group_to_vec(self.unflatten(x))

    def vec_to_group(self, x: Tensor) -> Tensor:
        if x.shape[1:] != self.xshape[1:]:
            x = self.unflatten(x)

        if self.config.group == 'SU3':
            return self.g.vec_to_group(x)

        xrT, xiT = tf.transpose(x)
        return tf.complex(tf.transpose(xrT), tf.transpose(xiT))

    def _update_v_bwd(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Update the momentum in the backward direction."""
        eps = self.veps[step]
        force = self.grad_potential(state.x, state.beta)
        x = state.x
        v = state.v
        # vNet: (x, force) --> (s, t, q)
        s, t, q = self._call_vnet(step, (x, force), training=training)
        # eps = tf.cast(self.veps[step], s.dtype)
        logjac = (-eps * s / 2.)
        logdet = tf.reduce_sum(logjac, axis=1)
        v = tf.cast(tf.reshape(v, (-1, *self.xshape[1:])), s.dtype)
        exp_s = tf.reshape(tf.exp(logjac), v.shape)
        exp_q = tf.reshape(tf.exp(eps * q), v.shape)
        t = tf.reshape(t, v.shape)
        force = tf.cast(tf.reshape(force, v.shape), s.dtype)
        vb = exp_s * (v + 0.5 * eps * (force * exp_q + t))
        return State(state.x, vb, state.beta), logdet

    def _update_v_fwd(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Update the momentum in the forward direction."""
        eps = self.veps[step]
        force = self.grad_potential(state.x, state.beta)
        # vNet: (x, F) --> (s, t, q)
        s, t, q = self._call_vnet(step, (state.x, force), training=training)
        # eps = tf.constant(self.veps[step], dtype=s.dtype)
        logjac = eps * s / 2.  # jacobian factor, also used in exp_s below
        logdet = tf.reduce_sum(self.flatten(logjac), axis=1)
        force = tf.cast(tf.reshape(force, state.v.shape), s.dtype)
        exp_s = tf.reshape(tf.exp(logjac), state.v.shape)
        exp_q = tf.reshape(tf.exp(eps * q), state.v.shape)
        t = tf.reshape(t, state.v.shape)
        v = tf.cast(state.v, exp_s.dtype)
        vf = exp_s * v - 0.5 * eps * (force * exp_q + t)
        return State(state.x, vf, state.beta), logdet

    def _update_v_fwd1(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Update the momentum in the forward direction."""
        eps = sigmoid(tf.math.log(self.veps[step]))
        # force = tf.cast(tf.reshape(force, state.v.shape), state.v.dtype)
        # exp_s = tf.cast(
        #     tf.reshape(tf.exp(logjac), state.v.shape),
        #     state.v.dtype
        # )
        # exp_q = tf.cast(
        #     tf.reshape(tf.exp(tf.multiply(eps, q)), state.v.shape),
        #     state.v.dtype
        # )
        # t = tf.cast(tf.reshape(t, state.v.shape), state.v.dtype)
        # vf = exp_s * state.v - 0.5 * eps * (force * exp_q + t)
        # vf = tf.multiply(exp_s, state.v) - 0.5 * eps * (force * exp_q + t)
        # halfeps = tf.cast(0.5 * eps, state.x.dtype)
        # vf = (
        #     tf.multiply(exp_s, state.v)
        #     - halfeps * (tf.multiply(force, exp_q) + t)
        # )
        force = self.grad_potential(state.x, state.beta)
        # vNet: (x, F) --> (s, t, q)
        s, t, q = self._call_vnet(step, (state.x, force), training=training)
        eps = tf.cast(eps, s.dtype)
        halfeps = tf.scalar_mul(0.5, eps)
        assert eps is not None and isinstance(eps, tf.Tensor)
        logjac = tf.scalar_mul(halfeps, s)
        logdet = tf.reduce_sum(self.flatten(logjac), axis=1)
        force = tf.cast(tf.reshape(force, state.v.shape), s.dtype)
        exp_s = tf.reshape(tf.exp(logjac), state.v.shape)
        exp_q = tf.reshape(tf.exp(tf.scalar_mul(eps, q)), state.v.shape)
        t = tf.reshape(t, state.v.shape)
        v = tf.cast(state.v, exp_s.dtype)
        vf = exp_s * v - tf.scalar_mul(halfeps, (force * exp_q + t))

        return State(state.x, vf, state.beta), logdet

    def _update_v_bwd1(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Update the momentum in the backward direction."""
        eps = sigmoid(tf.math.log(self.veps[step]))
        force = self.grad_potential(state.x, state.beta)
        x = state.x
        v = state.v
        # vNet: (x, force) --> (s, t, q)
        s, t, q = self._call_vnet(step, (x, force), training=training)
        eps = tf.cast(eps, s.dtype)
        assert eps is not None and isinstance(eps, tf.Tensor)
        halfeps = tf.scalar_mul(0.5, eps)
        logjac = tf.scalar_mul(tf.negative(halfeps), s)
        # logjac = (-eps * s / 2.)
        logdet = tf.reduce_sum(logjac, axis=1)
        v = tf.cast(tf.reshape(v, (-1, *self.xshape[1:])), s.dtype)
        exp_s = tf.reshape(tf.exp(logjac), v.shape)
        exp_q = tf.reshape(tf.scalar_mul(eps, q), v.shape)
        t = tf.reshape(t, v.shape)
        force = tf.cast(tf.reshape(force, v.shape), s.dtype)
        vb = (
            exp_s * tf.math.add(v, tf.scalar_mul(halfeps, (force * exp_q + t)))
        )
        # exp_s = tf.cast(tf.reshape(tf.exp(logjac), v.shape), v.dtype)
        # exp_q = tf.cast(tf.reshape(tf.exp(eps * q), v.shape), v.dtype)
        # t = tf.cast(tf.reshape(t, v.shape), v.dtype)
        # halfeps = tf.cast(0.5 * eps, state.x.dtype)
        # vb = exp_s * (v + halfeps * (force * exp_q + t))
        # s, t, q = self._call_vnet(step, (state.x, force), training=training)
        # halfeps = tf.scalar_mul(0.5, eps)
        # logjac = tf.scalar_mul(tf.scalar_mul(-1., halfeps), s)
        # v = tf.reshape(state.v, (-1, *self.xshape[1:]))
        # logdet = tf.reduce_sum(logjac, axis=1)
        # exp_s = tf.reshape(tf.exp(logjac), v.shape)
        # exp_q = tf.reshape(tf.exp(tf.scalar_mul(eps, q)), v.shape)
        # t = tf.reshape(t, v.shape)
        # force = tf.reshape(force, v.shape)
        # vb = exp_s * (v + tf.scalar_mul(halfeps, (force * exp_q + t)))

        return State(state.x, vb, state.beta), tf.math.real(logdet)

    def _update_x_fwd(
            self,
            step: int,
            state: State,
            m: Tensor,
            first: bool,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        eps = sigmoid(tf.math.log(self.xeps[step]))
        mb = (tf.ones_like(m) - m)
        x = self.flatten(state.x)
        v = self.flatten(state.v)
        m = tf.cast(m, x.dtype)
        xm_init = m * x
        # xNet: (m * x, v) --> (s, t, q)
        s, t, q = self._call_xnet(
            step,
            (xm_init, state.v),
            first=first,
            training=training
        )
        eps = tf.cast(eps, s.dtype)
        x = tf.cast(x, s.dtype)
        v = tf.cast(v, s.dtype)
        mb = tf.cast(mb, s.dtype)
        xm_init = tf.cast(xm_init, s.dtype)
        s = eps * s
        q = eps * q
        exp_s = tf.exp(s)
        exp_q = tf.exp(q)
        if self.config.group == 'U1':
            if self.config.use_ncp:
                halfx = self.flatten(x / 2.)
                _x = 2. * tf.math.atan(tf.math.tan(halfx) * exp_s)
                xp = _x + eps * (v * exp_q + t)
                xf = xm_init + (mb * xp)
                cterm = (tf.math.cos(halfx)) ** 2
                sterm = (exp_s * tf.math.sin(halfx)) ** 2
                logdet_ = tf.math.log(exp_s / (cterm + sterm))
                logdet = tf.reduce_sum(mb * logdet_, axis=1)
                # halfx = self.flatten(tf.math.multiply(0.5, x))
                # _x = tf.math.multiply(
                #     2.,
                #     tf.math.atan(tf.math.tan(halfx) * exp_s)
                # )
                # xp = _x + eps * (v * exp_q + t)
                # xf = xm_init + (mb * xp)
                # cterm = tf.square(tf.math.cos(halfx))
                # sterm = (exp_s * tf.math.sin(halfx)) ** 2
                # logdet_ = tf.math.log(exp_s / (cterm + sterm))
                # mbld = tf.multiply(mb, logdet_)
                # logdet = tf.reduce_sum(mbld, axis=1)
            else:
                xp = x * exp_s + eps * (v * exp_q + t)
                xf = xm_init + (mb * xp)
                logdet = tf.reduce_sum(mb * s, axis=1)
        elif self.config.group == 'SU3':
            x = self.unflatten(x)
            xm_init = self.unflatten(xm_init)
            exp_s = tf.cast(tf.reshape(exp_s, x.shape), x.dtype)
            exp_q = tf.cast(tf.reshape(exp_q, x.shape), x.dtype)
            t = tf.cast(tf.reshape(t, x.shape), x.dtype)
            eps = tf.cast(eps, x.dtype)
            v = tf.reshape(v, x.shape)
            xp = self.g.update_gauge(x, eps * (v * exp_q + t))
            xf = xm_init + tf.reshape(mb * self.flatten(xp), xm_init.shape)
            logdet = tf.reduce_sum(mb * tf.cast(s, x.dtype), axis=1)
            # x = self.unflatten(x)
            # xm_init = self.unflatten(x)
            # exp_s = tf.cast(tf.reshape(exp_s, x.shape), x.dtype)
            # exp_q = tf.cast(tf.reshape(exp_q, x.shape), x.dtype)
            # t = tf.cast(tf.reshape(t, x.shape), x.dtype)
            # eps = tf.cast(eps, x.dtype)
            # v = tf.reshape(v, x.shape)
            # xp = self.g.update_gauge(x, eps * (v * exp_q + t))
            # mbxp = tf.multiply(mb, self.flatten(xp))
            # mbs = tf.multiply(mb, tf.cast(s, x.dtype))
            # xf = xm_init + tf.reshape(mbxp, xm_init.shape)
            # logdet = tf.reduce_sum(mbs, axis=1)
            # xf = (
            #     xm_init
            #     + tf.reshape((mb * self.flatten(xp)), xm_init.shape)
            # )
            # logdet = tf.reduce_sum(mb * tf.cast(s, x.dtype), axis=1)
        else:
            raise ValueError(
                'Unexpected value for self.config.group: '
                f'{self.config.group}'
            )

        xf = self.g.compat_proj(xf)
        return State(x=xf, v=state.v, beta=state.beta), tf.math.real(logdet)

    def _update_x_bwd(
            self,
            step: int,
            state: State,
            m: Tensor,
            first: bool,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Update the position in the backward direction."""
        eps = sigmoid(tf.math.log(self.xeps[step]))  # .clamp_min_(0.)
        m = tf.cast(m, state.x.dtype)
        x = tf.reshape(state.x, (state.x.shape[0], -1))
        v = tf.reshape(state.v, x.shape)
        # xm_init = tf.math.multiply(tf.cast(m, x.dtype), x)
        xm_init = m * x
        # xNet: (m * x, v) --> (s, t, q)
        s, t, q = self._call_xnet(
            step,
            (xm_init, state.v),
            first=first,
            training=training
        )
        eps = tf.cast(eps, s.dtype)
        x = tf.cast(x, s.dtype)
        v = tf.cast(v, s.dtype)
        mb = tf.cast(tf.ones_like(m) - m, s.dtype)
        xm_init = tf.cast(xm_init, s.dtype)
        s = (-eps) * s
        q = eps * q
        exp_s = tf.math.exp(s)
        exp_q = tf.math.exp(q)
        if self.config.group == 'U1':
            if self.config.use_ncp:
                halfx = x / 2.
                halfx_scale = exp_s * tf.math.tan(halfx)
                x1 = 2. * tf.math.atan(halfx_scale)
                x2 = exp_s * eps * (v * exp_q + t)
                xnew = x1 - x2
                xb = xm_init + (mb * xnew)
                cterm = tf.math.cos(halfx) ** 2
                sterm = (exp_s * tf.math.sin(halfx)) ** 2
                logdet_ = tf.math.log(exp_s / (cterm + sterm))
                logdet = tf.reduce_sum(mb * logdet_, axis=1)
            else:
                xnew = exp_s * (state.x - eps * (state.v * exp_q + t))
                xb = xm_init + (mb * xnew)
                logdet = tf.reduce_sum(mb * s, axis=1)

        elif self.config.group == 'SU3':
            exp_s = tf.cast(tf.reshape(exp_s, state.x.shape), state.x.dtype)
            exp_q = tf.cast(tf.reshape(exp_q, state.x.shape), state.x.dtype)
            t = tf.cast(tf.reshape(t, state.x.shape), state.x.dtype)
            eps = tf.cast(eps, state.x.dtype)
            xnew = exp_s * self.g.update_gauge(
                state.x,
                -(eps * (state.v * exp_q + t))
            )
            xb = tf.reshape(
                xm_init + (mb * self.flatten(xnew)),
                (-1, *self.xshape[1:])
            )
            logdet = tf.reduce_sum(mb * tf.cast(s, mb.dtype), axis=1)
        else:
            raise ValueError('Unexpected value for `self.g`')

        xb = self.g.compat_proj(xb)
        return State(x=xb, v=state.v, beta=state.beta), tf.math.real(logdet)

    def hamiltonian(self, state: State) -> Tensor:
        """Returns the total energy H = KE + PE."""
        kinetic = self.kinetic_energy(state.v)
        potential = tf.cast(
            self.potential_energy(state.x, state.beta),
            kinetic.dtype
        )
        # return tf.add(kinetic, potential)
        return kinetic + potential

    def kinetic_energy(self, v: Tensor) -> Tensor:
        """Returns the kinetic energy, KE = 0.5 * v ** 2."""
        return self.g.kinetic_energy(
            tf.reshape(v, (-1, *self.xshape[1:]))
        )

    def potential_energy(self, x: Tensor, beta: Tensor) -> Tensor:
        """Returns the potential energy, PE = beta * action(x)."""
        return self.potential_fn(x=x, beta=beta)

    def grad_potential(self, x: Tensor, beta: Tensor) -> Tensor:
        """Compute the gradient of the potential function."""
        return self.lattice.grad_action(self.unflatten(x), beta)

    def load_networks(self, d: os.PathLike):
        d = Path(d)
        assert d.is_dir(), f'Directory {d} does not exist'
        fveps = d.joinpath('veps.npy')
        fxeps = d.joinpath('xeps.npy')
        veps = tf.Variable(np.load(fveps.as_posix()), dtype=TF_FLOAT)
        xeps = tf.Variable(np.load(fxeps.as_posix()), dtype=TF_FLOAT)
        if self.config.use_separate_networks:
            xnet = {}
            vnet = {}
            for lf in range(self.config.nleapfrog):
                fvnet = d.joinpath(f'vnet-{lf}')
                fxnet1 = d.joinpath(f'xnet-{lf}_first')
                vnet[str(lf)] = tf.keras.models.load_model(fvnet.as_posix())
                xnet[str(lf)] = {
                    'first': tf.keras.models.load_model(fxnet1.as_posix())
                }
                if self.config.use_split_xnets:
                    fxnet2 = d.joinpath(f'xnet-{lf}_second')
                    xnet[str(lf)].update({
                        'second': tf.keras.models.load_model(fxnet2.as_posix())
                    })
        else:
            vnet = tf.keras.models.load_model(d.joinpath('vnet').as_posix())
            xnet1 = tf.keras.models.load_model(
                d.joinpath('xnet_first').as_posix()
            )
            vnet = {'0': vnet}
            xnet = {'0': {'first': xnet1}}
            if self.config.use_split_xnets:
                xnet2 = tf.keras.models.load_model(
                    d.joinpath('xnet_second').as_posix()
                )
                xnet['0'].update({'second': xnet2})

        return {'vnet': vnet, 'xnet': xnet, 'veps': veps, 'xeps': xeps}

    def save_networks(self, outdir: os.PathLike) -> None:
        """Save networks to `outdir`."""
        outdir = Path(outdir).joinpath('networks')
        outdir.mkdir(exist_ok=True, parents=True)

        try:
            self.save(
                outdir.joinpath('dynamics').as_posix(),
                save_format='tf',
            )
        except Exception as e:
            log.exception(e)

        if self.config.use_separate_networks:
            for lf in range(self.config.nleapfrog):
                fvnet = outdir.joinpath(f'vnet-{lf}').as_posix()
                fxnet1 = outdir.joinpath(f'xnet-{lf}_first').as_posix()

                vnet = self._get_vnet(lf)
                xnet1 = self._get_xnet(lf, first=True)

                vnet.save(fvnet, save_format='tf')
                xnet1.save(fxnet1, save_format='tf')

                if self.config.use_split_xnets:
                    xnet2 = self._get_xnet(lf, first=False)
                    fxnet2 = outdir.joinpath(f'xnet-{lf}_second').as_posix()
                    xnet2.save(fxnet2)
        else:
            vnet = self._get_vnet(0)
            xnet1 = self._get_xnet(0, first=True)

            fvnet = outdir.joinpath('vnet').as_posix()
            fxnet1 = outdir.joinpath('xnet_first').as_posix()

            vnet.save(fvnet, save_format='tf')
            xnet1.save(fxnet1, save_format='tf')

            if self.config.use_split_xnets:
                xnet2 = self._get_xnet(0, first=False)
                fxnet2 = outdir.joinpath('xnet_second').as_posix()
                xnet2.save(fxnet2, save_format='tf')
