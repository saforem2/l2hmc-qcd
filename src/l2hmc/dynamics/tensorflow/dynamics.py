"""
tensorflow/dynamics.py

Tensorflow implementation of Dynamics object for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass
from math import pi
import os
from pathlib import Path
from typing import Callable, Optional
from typing import Tuple
import logging

import numpy as np
import tensorflow as tf

from l2hmc import configs as cfgs
from l2hmc.network.tensorflow.network import NetworkFactory
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
        # type: ignore
        return {
            'x': self.x.numpy(),
            'v': self.v.numpy(),
            'beta': self.beta.numpy(),
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


def dummy_network(
        x: Tensor,
        *_,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        tf.zeros_like(x),
        tf.zeros_like(x),
        tf.zeros_like(x)
    )


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
        # self.midpt = self.config.nleapfrog // 2
        self.network_factory = network_factory
        if network_factory is not None:
            self._networks_built = True
            self.xnet, self.vnet = self._build_networks(network_factory)
            self.networks = {'xnet': self.xnet, 'vnet': self.vnet}
        else:
            self._networks_built = False
            self.xnet = dummy_network
            self.vnet = dummy_network
            self.networks = {
                'xnet': self.xnet,
                'vnet': self.vnet
            }
        self.masks = self._build_masks()
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

        self.xeps = []
        self.veps = []
        ekwargs = {
            'dtype': TF_FLOAT,
            'initial_value': self.config.eps,
            'trainable': (not self.config.eps_fixed),
            'constraint': tf.keras.constraints.non_neg(),
        }
        for lf in range(self.config.nleapfrog):
            xalpha = tf.Variable(name=f'xeps_lf{lf}', **ekwargs)
            valpha = tf.Variable(name=f'veps_lf{lf}', **ekwargs)
            self.xeps.append(xalpha)
            self.veps.append(valpha)

    def _build_networks(self, network_factory):
        """Build networks."""
        split = self.config.use_split_xnets
        n = self.nlf if self.config.use_separate_networks else 1
        networks = network_factory.build_networks(n, split)
        return networks['xnet'], networks['vnet']

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
    def flatten(x: Tensor) -> Tensor:
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

        return {'dx': dx.numpy(), 'dv': dv.numpy()}

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
        ma_, mr_ = self._get_accept_masks(data['metrics']['acc'])
        ma = ma_[:, None]
        ma_ = tf.cast(ma_, dtype=TF_FLOAT)  # data['proposed'].x.dtype)
        mr_ = tf.cast(mr_, dtype=TF_FLOAT)  # data['proposed'].x.dtype)
        v_out = tf.where(
            tf.cast(ma, bool),
            self.flatten(data['proposed'].v),
            self.flatten(data['init'].v)
        )
        x_out = tf.where(
            tf.cast(ma, bool),
            self.flatten(data['proposed'].x),
            self.flatten(data['init'].x),
        )
        sumlogdet = tf.cast(ma_, x_out.dtype) * data['metrics']['sumlogdet']

        state_out = State(x=x_out, v=v_out, beta=data['init'].beta)
        mc_states = MonteCarloStates(init=data['init'],
                                     proposed=data['proposed'],
                                     out=state_out)
        data['metrics'].update({
            'acc_mask': ma_,
            'sumlogdet': sumlogdet,
            'mc_states': mc_states,
        })

        return mc_states.out.x, data['metrics']

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
            eps: float,
    ) -> State:
        """Perform standard HMC leapfrog update."""
        x = tf.reshape(state.x, state.v.shape)
        force = self.grad_potential(x, state.beta)        # f = dU / dx
        eps = tf.constant(eps, dtype=force.dtype)
        halfeps = tf.constant(eps / 2.0, dtype=force.dtype)
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
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=state.x.dtype)
        sldf = tf.zeros_like(sumlogdet)
        sldb = tf.zeros_like(sumlogdet)

        history = {}
        if self.config.verbose:
            extras = {'sldf': sldf, 'sldb': sldb, 'sld': sumlogdet}
            history = self.update_history(
                self.get_metrics(state_, sumlogdet, step=0, extras=extras),
                history=history,
            )

        # Forward
        for step in range(self.config.nleapfrog):
            state_, logdet = self._forward_lf(step, state_, training)
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
        # m1 = -1.0 * tf.ones_like(state_.v)
        vneg = tf.negative(state_.v)
        assert isinstance(vneg, tf.Tensor)
        state_ = State(state_.x, vneg, state_.beta)

        # Backward
        for step in range(self.config.nleapfrog):
            state_, logdet = self._backward_lf(step, state_, training)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                sldb += logdet
                extras = {'sldf': sldf, 'sldb': sldb, 'sld': sumlogdet}
                # Reverse step count to correctly order metrics at each step
                step = self.config.nleapfrog - step - 1
                metrics = self.get_metrics(
                    state_,
                    sumlogdet,
                    step=step,
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
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=state.x.dtype)

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
        assert isinstance(vnet, (dict, tf.keras.Model))
        if self.config.use_separate_networks and isinstance(vnet, dict):
            return vnet[str(step)]
        # assert isinstance(vnet, (CallableNetwork))
        return self.vnet

    def _get_xnet(
            self,
            step: int,
            first: bool
    ) -> Callable:
        """Returns position network to be used for updating x."""
        if not self._networks_built:
            return self.xnet
        xnet = self.xnet
        assert isinstance(xnet, (tf.keras.Model, dict))
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
        vnet = self._get_vnet(step)
        assert callable(vnet)
        # x = self.g.group_to_vec(x)
        return vnet((x, force), training)

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
        if isinstance(self.g, U1Phase):
            x = self.g.group_to_vec(x)

        elif isinstance(self.g, SU3):
            x = tf.reshape(x, self.xshape)
            x = tf.stack([tf.math.real(x), tf.math.imag(x)], 1)
        # x = self.g.group_to_vec(x)
        # assert xnet is not None and isinstance(xnet, LeapfrogLayer)
        assert callable(xnet)
        s, t, q = xnet((x, v), training)
        # return xnet((x, v), training)
        return (s, t, q)

    def _forward_lf(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Complete update (leapfrog step) in the forward direction."""
        m, mb = self._get_mask(step)
        m = tf.cast(m, state.x.dtype)
        mb = tf.cast(mb, state.x.dtype)
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=state.x.dtype)
        assert isinstance(state.x, Tensor)
        assert isinstance(m, Tensor) and isinstance(mb, Tensor)

        state, logdet = self._update_v_fwd(step, state, training=training)
        sumlogdet = sumlogdet + tf.cast(logdet, sumlogdet.dtype)

        state, logdet = self._update_x_fwd(step, state, m,
                                           first=True, training=training)
        sumlogdet = sumlogdet + tf.cast(logdet, sumlogdet.dtype)

        state, logdet = self._update_x_fwd(step, state, mb,
                                           first=False, training=training)
        sumlogdet = sumlogdet + tf.cast(logdet, sumlogdet.dtype)

        state, logdet = self._update_v_fwd(step, state, training=training)
        sumlogdet = sumlogdet + tf.cast(logdet, sumlogdet.dtype)

        return state, sumlogdet

    def _backward_lf(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Complete update (leapfrog step) in the backward direction."""
        # Note: Reverse the step count, i.e. count from end of trajectory.
        assert isinstance(state.x, Tensor)
        step_r = self.config.nleapfrog - step - 1

        m, mb = self._get_mask(step_r)
        m = tf.cast(m, state.x.dtype)
        mb = tf.cast(mb, state.x.dtype)
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=state.x.dtype)
        assert isinstance(m, Tensor) and isinstance(mb, Tensor)

        state, logdet = self._update_v_bwd(step_r, state, training=training)
        sumlogdet = sumlogdet + tf.cast(logdet, sumlogdet.dtype)

        state, logdet = self._update_x_bwd(step_r, state, mb,
                                           first=False, training=training)
        sumlogdet = sumlogdet + tf.cast(logdet, sumlogdet.dtype)

        state, logdet = self._update_x_bwd(step_r, state, m,
                                           first=True, training=training)
        sumlogdet = sumlogdet + tf.cast(logdet, sumlogdet.dtype)

        state, logdet = self._update_v_bwd(step_r, state, training=training)
        sumlogdet = sumlogdet + tf.cast(logdet, sumlogdet.dtype)

        return state, sumlogdet

    def _update_v_fwd(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Update the momentum in the forward direction."""
        eps = tf.cast(self.veps[step], state.x.dtype)
        x = tf.reshape(state.x, (-1, *self.xshape[1:]))
        force = tf.reshape(
            self.grad_potential(x, state.beta),
            state.v.shape
        )
        s, t, q = self._call_vnet(step, (state.x, force), training=training)

        # jacobian factor, used in exp_s below
        halfeps = tf.scalar_mul(0.5, eps)
        jac = tf.scalar_mul(halfeps, s)
        # jac = eps * s / 2.  # jacobian factor, also used in exp_s below
        logdet = tf.reduce_sum(jac, axis=1)

        exp_s = tf.reshape(tf.exp(jac), state.v.shape)
        exp_q = tf.reshape(tf.exp(tf.scalar_mul(eps, q)), force.shape)
        t = tf.reshape(t, force.shape)
        # exp_q = tf.exp(eps * q)
        vf = exp_s * state.v - tf.scalar_mul(halfeps, (force * exp_q + t))

        return State(state.x, vf, state.beta), logdet

    def _update_v_bwd(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Update the momentum in the backward direction."""
        eps = tf.cast(self.veps[step], state.x.dtype)
        force = self.grad_potential(state.x, state.beta)
        s, t, q = self._call_vnet(step, (state.x, force), training=training)
        halfeps = tf.scalar_mul(0.5, eps)
        logjac = tf.scalar_mul(tf.scalar_mul(-1., halfeps), s)
        # logjac = (-eps * s / 2.)
        v = tf.reshape(state.v, (-1, *self.xshape[1:]))
        logdet = tf.reduce_sum(logjac, axis=1)
        exp_s = tf.reshape(tf.exp(logjac), v.shape)
        exp_q = tf.reshape(tf.exp(tf.scalar_mul(eps, q)), v.shape)
        t = tf.reshape(t, v.shape)
        force = tf.reshape(force, v.shape)
        vb = exp_s * (v + tf.scalar_mul(halfeps, (force * exp_q + t)))

        return State(state.x, vb, state.beta), logdet

    def _update_x_fwd(
            self,
            step: int,
            state: State,
            m: Tensor,
            first: bool,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Single x update in the forward direction"""
        eps = self.xeps[step]
        mb = tf.ones_like(m) - m
        x = tf.reshape(state.x, (state.x.shape[0], -1))
        v = tf.reshape(state.v, x.shape)
        xm_init = tf.multiply(m, x)
        inputs = (xm_init, state.v)
        s, t, q = self._call_xnet(step, inputs, first=first, training=training)
        s = tf.scalar_mul(eps, s)
        q = tf.scalar_mul(eps, q)
        exp_s = tf.exp(s)
        exp_q = tf.exp(q)
        if isinstance(self.g, U1Phase):
            if self.config.use_ncp:
                halfx = self.flatten(x / TWO)
                _x = TWO * tf.math.atan(tf.math.tan(halfx) * exp_s)
                xp = _x + eps * (v * exp_q + t)
                xf = xm_init + (mb * xp)
                cterm = tf.math.square(tf.math.cos(halfx))
                sterm = (exp_s * tf.math.sin(halfx)) ** 2
                logdet_ = tf.math.log(exp_s / (cterm + sterm))
                logdet = tf.reduce_sum(mb * logdet_, axis=1)
            else:
                xp = x * exp_s + eps * (v * exp_q + t)
                xf = xm_init + (mb * xp)
                logdet = tf.reduce_sum((mb * s), axis=1)

        elif isinstance(self.g, SU3):
            x = tf.reshape(state.x, self.xshape)
            xm_init = tf.reshape(xm_init, self.xshape)
            exp_s = tf.cast(tf.reshape(exp_s, self.xshape), x.dtype)
            exp_q = tf.cast(tf.reshape(exp_q, self.xshape), x.dtype)
            t = tf.cast(tf.reshape(t, self.xshape), x.dtype)
            eps = tf.cast(eps, x.dtype)
            v = tf.reshape(state.v, self.xshape)
            xp = x * exp_s + eps * (v * exp_q + t)
            xf = xm_init + tf.reshape((mb * self.flatten(xp)), xm_init.shape)
            logdet = tf.reduce_sum(mb * tf.cast(s, x.dtype), axis=1)
        else:
            raise ValueError('Unexpected value for `self.g`')

        xf = self.g.compat_proj(xf)
        logdet = tf.cast(logdet, TF_FLOAT)
        assert isinstance(logdet, Tensor)
        return State(x=xf, v=state.v, beta=state.beta), logdet

    def _update_x_bwd(
            self,
            step: int,
            state: State,
            m: Tensor,
            first: bool,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Update the position in the backward direction."""
        eps = self.xeps[step]

        mb = tf.ones_like(m) - m
        x = tf.reshape(state.x, (state.x.shape[0], -1))
        v = tf.reshape(state.v, x.shape)
        xm_init = tf.multiply(m, x)
        inputs = (xm_init, state.v)
        s, t, q = self._call_xnet(step, inputs, first=first, training=training)
        s = tf.scalar_mul(tf.scalar_mul(-1., eps), s)
        q = tf.scalar_mul(eps, q)
        exp_q = tf.exp(q)
        exp_s = tf.exp(s)

        if isinstance(self.g, U1Phase):
            if self.config.use_ncp:
                halfx = x / TWO
                halfx_scale = exp_s * tf.tan(halfx)
                x1 = TWO * tf.atan(halfx_scale)
                x2 = exp_s * eps * (v * exp_q + t)
                xnew = x1 - x2
                xb = (
                    xm_init
                    + tf.reshape(mb * self.flatten(xnew), xm_init.shape)
                )

                cterm = tf.math.square(tf.cos(halfx))
                sterm = (exp_s * tf.sin(halfx)) ** 2
                logdet_ = tf.reshape(
                    tf.math.log(exp_s / (cterm + sterm)),
                    (xm_init.shape[0], -1)
                )
                logdet = tf.reduce_sum(mb * logdet_, axis=1)
            else:
                xnew = exp_s * (state.x - eps * (state.v * exp_q + t))
                xb = xm_init + mb * xnew
                logdet = tf.reduce_sum(mb * s, axis=1)
        elif isinstance(self.g, SU3):
            exp_s = tf.cast(tf.reshape(exp_s, state.x.shape), state.x.dtype)
            exp_q = tf.cast(tf.reshape(exp_q, state.x.shape), state.x.dtype)
            t = tf.cast(tf.reshape(t, state.x.shape), state.x.dtype)
            eps = tf.cast(eps, state.x.dtype)
            xnew = exp_s * self.g.update_gauge(
                state.x,
                -(eps * (state.v * exp_q + t))  # type:ignore
            )
            xb = tf.reshape(
                xm_init + (mb * self.flatten(xnew)),
                self.xshape
            )
            logdet = tf.math.real(
                tf.reduce_sum(mb * tf.cast(s, mb.dtype), axis=1)
            )
        else:
            raise ValueError('Unexpected value for `self.g`')

        xb = self.g.compat_proj(xb)
        return State(x=xb, v=state.v, beta=state.beta), logdet

    def hamiltonian(self, state: State) -> Tensor:
        """Returns the total energy H = KE + PE."""
        kinetic = self.kinetic_energy(state.v)
        potential = self.potential_energy(state.x, state.beta)
        return tf.add(kinetic, potential)

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
        return self.lattice.grad_action(x, beta)

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
            pass

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
