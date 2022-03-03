"""
tensorflow/dynamics.py

Tensorflow implementation of Dynamics object for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass
from math import pi as PI
import os
from pathlib import Path
from typing import Callable
from typing import Tuple
import logging

import numpy as np
import tensorflow as tf

from l2hmc.configs import DynamicsConfig
from l2hmc.network.tensorflow.network import NetworkFactory

TWO_PI = 2. * PI
TWO = tf.constant(2.)

Tensor = tf.Tensor
Model = tf.keras.Model
TF_FLOAT = tf.keras.backend.floatx()


log = logging.getLogger(__name__)


DynamicsOutput = Tuple[Tensor, dict]


def to_u1(x: Tensor) -> Tensor:
    return (tf.add(x, PI) % TWO_PI) - PI


@dataclass
class State:
    x: Tensor
    v: Tensor
    beta: Tensor


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


CallableNetwork = Callable[[Tuple[Tensor, Tensor], bool],
                           Tuple[Tensor, Tensor, Tensor]]


class Dynamics(Model):
    def __init__(
            self,
            potential_fn: Callable,
            config: DynamicsConfig,
            network_factory: NetworkFactory,
    ):
        """Initialization."""
        super(Dynamics, self).__init__()
        # TODO: Implement reversibility check
        self.config = config
        self.xdim = self.config.xdim
        self.xshape = tuple(network_factory.input_spec.xshape)
        self.potential_fn = potential_fn
        self.nlf = self.config.nleapfrog
        self.midpt = self.config.nleapfrog // 2
        self.xnet, self.vnet = self._build_networks(network_factory)
        self.masks = self._build_masks()

        self.xeps = []
        self.veps = []
        for lf in range(self.config.nleapfrog):
            self.xeps.append(
                tf.Variable(initial_value=self.config.eps,
                            name=f'xeps_lf{lf}', dtype=TF_FLOAT,
                            trainable=(not self.config.eps_fixed),
                            constraint=tf.keras.constraints.non_neg())
            )
            self.veps.append(
                tf.Variable(initial_value=self.config.eps,
                            name=f'veps_lf{lf}', dtype=TF_FLOAT,
                            trainable=(not self.config.eps_fixed),
                            constraint=tf.keras.constraints.non_neg())
            )

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
        """Call Dynamics object."""
        if self.config.merge_directions:
            return self.apply_transition_fb(inputs, training=training)
        return self.apply_transition(inputs, training=training)

    def random_state(self, beta: float = 1.) -> State:
        """Returns a random State."""
        x = tf.random.uniform(self.config.xshape, dtype=TF_FLOAT)
        x = tf.reshape(x, (self.config.xshape[0], -1))
        v = tf.random.normal(x.shape, dtype=TF_FLOAT)
        return State(x=x, v=v, beta=tf.constant(beta))

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
    ) -> tuple[Tensor, dict]:
        data = self.generate_proposal_hmc(inputs)
        ma_, mr_ = self._get_accept_masks(data['metrics']['acc'])
        ma = ma_[:, None]
        mr = mr_[:, None]
        vout = ma * data['proposed'].v + mr * data['init'].v
        xout = ma * data['proposed'].x + mr * data['init'].x
        sumlogdet = ma_ * data['metrics']['sumlogdet']
        state_out = State(x=xout, v=vout, beta=data['init'].beta)
        mc_states = MonteCarloStates(init=data['init'],
                                     proposed=data['proposed'],
                                     out=state_out)
        data['metrics'].update({
            'acc_mask': ma_,
            'sumlogdet': sumlogdet,
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
        mr = mr_[:, None]

        v_out = ma * data['proposed'].v + mr * data['init'].v
        x_out = ma * data['proposed'].x + mr * data['init'].x
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

        mf_, mb_ = self._get_direction_masks(batch_size=x.shape[0])
        mf = mf_[:, None]
        mb = mb_[:, None]

        v_init = mf * fwd['init'].v + mb * bwd['init'].v

        x_prop = mf * fwd['proposed'].x + mb * bwd['proposed'].x
        v_prop = mf * fwd['proposed'].v + mb * bwd['proposed'].v

        mfwd = fwd['metrics']
        mbwd = bwd['metrics']

        logdet_prop = mf_ * mfwd['sumlogdet'] + mb_ * mbwd['sumlogdet']

        acc = mf_ * mfwd['acc'] + mb_ * mbwd['acc']
        ma_, mr_ = self._get_accept_masks(acc)
        ma = ma_[:, None]
        mr = mr_[:, None]

        v_out = ma * v_prop + mr * v_init
        x_out = ma * x_prop + mr * x
        sumlogdet = ma_ * logdet_prop  # + mr_ * logdet_init (= 0.)

        init = State(x=x, v=v_init, beta=beta)
        prop = State(x=x_prop, v=v_prop, beta=beta)
        out = State(x=x_out, v=v_out, beta=beta)

        mc_states = MonteCarloStates(init=init, proposed=prop, out=out)

        metrics = {
            'acc': acc,
            'acc_mask': ma_,
            'sumlogdet': sumlogdet,
            'mc_states': mc_states,
        }
        metrics.update({f'fwd/{k}': v for k, v in mfwd.items()})
        metrics.update({f'bwd/{k}': v for k, v in mbwd.items()})

        return x_out, metrics

    def generate_proposal_hmc(
            self,
            inputs: tuple[Tensor, Tensor],
    ) -> dict:
        x, beta = inputs
        v = tf.random.normal(x.shape, dtype=x.dtype)
        init = State(x, v, beta)
        proposed, metrics = self.transition_kernel_hmc(init)

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
        v = tf.random.normal(x.shape, dtype=x.dtype)
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
        v = tf.random.normal(x.shape, dtype=TF_FLOAT)
        state_init = State(x=x, v=v, beta=beta)
        state_prop, metrics = self.transition_kernel(state_init,
                                                     forward=forward,
                                                     training=training)

        return {'init': state_init, 'proposed': state_prop, 'metrics': metrics}

    def get_metrics(
            self,
            state: State,
            logdet: Tensor,
            step: int = None,
    ) -> dict:
        """Returns dict of various metrics about input State."""
        energy = self.hamiltonian(state)
        logprob = tf.subtract(energy, logdet)
        metrics = {
            'energy': energy,
            'logprob': logprob,
            'logdet': logdet,
        }
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
            step: int,
    ) -> State:
        """Perform standard HMC leapfrog update."""
        force1 = self.grad_potential(state.x, state.beta)
        v1 = state.v - 0.5 * self.veps[step] * force1

        xp = state.x + self.xeps[step] * v1

        force2 = self.grad_potential(xp, state.beta)
        v2 = v1 - 0.5 * self.veps[step] * force2

        return State(x=xp, v=v2, beta=state.beta)

    def transition_kernel_hmc(
            self,
            state: State,
    ) -> tuple[State, dict]:
        """Run the generic HMC transition kernel."""
        state_ = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=TF_FLOAT)
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
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=TF_FLOAT)

        metrics = self.get_metrics(state_, sumlogdet)
        history = self.update_history(metrics, history={})

        # Forward
        for step in range(self.config.nleapfrog):
            state_, logdet = self._forward_lf(step, state_, training)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                metrics = self.get_metrics(state_, sumlogdet, step=step)
                history = self.update_history(metrics, history=history)

        # Flip momentum
        m1 = -1.0 * tf.ones_like(state_.v)
        state_ = State(state_.x, m1 * state_.v, state_.beta)

        # Backward
        for step in range(self.config.nleapfrog):
            state_, logdet = self._backward_lf(step, state_, training)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                step = self.config.nleapfrog - step - 1
                metrics = self.get_metrics(state_, sumlogdet, step=step)
                history = self.update_history(metrics, history=history)

        acc = self.compute_accept_prob(state, state_, sumlogdet)
        history['acc'] = acc
        history['sumlogdet'] = sumlogdet
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
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=TF_FLOAT)
        metrics = self.get_metrics(state_, sumlogdet)
        history = self.update_history(metrics, history={})

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
        dh = tf.add(tf.subtract(h_init, h_prop), sumlogdet)
        # dh = h_init - h_prop + sumlogdet
        prob = tf.exp(tf.minimum(dh, 0.))

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

    def _get_mask(self, i: int) -> tuple[Tensor, Tensor]:
        """Returns mask used for sequentially updating x."""
        m = self.masks[i]
        mb = tf.ones_like(m) - m
        return (m, mb)

    def _get_vnet(self, step: int) -> CallableNetwork:
        """Returns momentum network to be used for updating v."""
        vnet = self.vnet
        if self.config.use_separate_networks:
            return vnet[str(step)]
        return vnet

    def _call_vnet(
            self,
            step: int,
            inputs: tuple[Tensor, Tensor],
            training: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Calls the momentum network used to update v."""
        vnet = self._get_vnet(step)
        assert callable(vnet)
        return vnet(inputs, training)

    def _get_xnet(self, step: int, first: bool) -> CallableNetwork:
        """Returns position network to be used for updating x."""
        xnet = self.xnet
        if self.config.use_separate_networks:
            xnet = xnet[str(step)]
            if self.config.use_split_xnets:
                if first:
                    return xnet['first']
                return xnet['second']
            return xnet
        return xnet

    def _stack_as_xy(self, x: Tensor):
        """Returns -pi < x <= pi stacked as [cos(x), sin(x)]"""
        return tf.stack([tf.math.cos(x), tf.math.sin(x)], axis=-1)

    def _call_xnet(
            self,
            step: int,
            inputs: tuple[Tensor, Tensor],
            first: bool,
            training: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Calls the position network used to update x."""
        x, v = inputs
        x = self._stack_as_xy(x)
        xnet = self._get_xnet(step, first)
        assert callable(xnet)
        return xnet((x, v), training)

    def _forward_lf(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Complete update (leapfrog step) in the forward direction."""
        m, mb = self._get_mask(step)
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=TF_FLOAT)

        state, logdet = self._update_v_fwd(step, state, training=training)
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
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=TF_FLOAT)

        state, logdet = self._update_v_bwd(step_r, state, training=training)
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

    def _update_v_fwd(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Update the momentum in the forward direction."""
        eps = self.veps[step]
        force = self.grad_potential(state.x, state.beta)
        s, t, q = self._call_vnet(step, (state.x, force), training=training)

        jac = eps * s / 2.  # jacobian factor, also used in exp_s below
        logdet = tf.reduce_sum(jac, axis=1)

        exp_s = tf.exp(jac)
        exp_q = tf.exp(eps * q)
        vf = exp_s * state.v - 0.5 * eps * (force * exp_q + t)

        return State(state.x, vf, state.beta), logdet

    def _update_v_bwd(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Update the momentum in the backward direction."""
        eps = self.veps[step]
        force = self.grad_potential(state.x, state.beta)
        s, t, q = self._call_vnet(step, (state.x, force), training=training)
        jac = (-eps * s / 2.)
        logdet = tf.reduce_sum(jac, axis=1)
        exp_s = tf.exp(jac)
        exp_q = tf.exp(eps * q)
        vb = exp_s * (state.v + 0.5 * eps * (force * exp_q + t))

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
        xm_init = tf.multiply(m, state.x)
        inputs = (xm_init, state.v)
        s, t, q = self._call_xnet(step, inputs, first=first, training=training)
        s = eps * s
        q = eps * q
        exp_s = tf.exp(s)
        exp_q = tf.exp(q)
        if self.config.use_ncp:
            halfx = state.x / TWO
            _x = TWO * tf.math.atan(tf.math.tan(halfx) * exp_s)
            xp = _x + eps * (state.v * exp_q + t)
            xf = xm_init + (mb * xp)
            cterm = tf.math.square(tf.math.cos(halfx))
            sterm = (exp_s * tf.math.sin(halfx)) ** 2
            logdet_ = tf.math.log(exp_s / (cterm + sterm))
            logdet = tf.reduce_sum(mb * logdet_, axis=1)
        else:
            xp = state.x * exp_s + eps * (state.v * exp_q + t)
            xf = xm_init + (mb * xp)
            logdet = tf.reduce_sum(mb * s, axis=1)

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
        xm_init = tf.multiply(m, state.x)
        inputs = (xm_init, state.v)
        s, t, q = self._call_xnet(step, inputs, first=first, training=training)
        s = tf.scalar_mul(-eps, s)
        q = tf.scalar_mul(eps, q)
        exp_q = tf.exp(q)
        exp_s = tf.exp(s)

        if self.config.use_ncp:
            halfx = state.x / TWO
            halfx_scale = exp_s * tf.tan(halfx)
            x1 = TWO * tf.atan(halfx_scale)
            x2 = exp_s * eps * (state.v * exp_q + t)
            xnew = x1 - x2
            xb = xm_init + (mb * xnew)

            cterm = tf.math.square(tf.cos(halfx))
            sterm = (exp_s * tf.sin(halfx)) ** 2
            logdet_ = tf.math.log(exp_s / (cterm + sterm))
            logdet = tf.reduce_sum(mb * logdet_, axis=1)
        else:
            xnew = exp_s * (state.x - eps * (state.v * exp_q + t))
            xb = xm_init + mb * xnew
            logdet = tf.reduce_sum(mb * s, axis=1)

        return State(x=xb, v=state.v, beta=state.beta), logdet

    def hamiltonian(self, state: State) -> Tensor:
        """Returns the total energy H = KE + PE."""
        kinetic = self.kinetic_energy(state.v)
        potential = self.potential_energy(state.x, state.beta)
        return tf.add(kinetic, potential)

    @staticmethod
    def kinetic_energy(v: Tensor) -> Tensor:
        """Returns the kinetic energy, KE = 0.5 * v ** 2."""
        return tf.reduce_sum(tf.math.square(v), axis=-1) / 2.

    def potential_energy(self, x: Tensor, beta: Tensor) -> Tensor:
        """Returns the potential energy, PE = beta * action(x)."""
        return tf.multiply(beta, self.potential_fn(x))

    def grad_potential(self, x: Tensor, beta: Tensor) -> Tensor:
        """Compute the gradient of the potential function."""
        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                tape.watch(x)
                pe = self.potential_energy(x, beta)
            grad = tape.gradient(pe, x)
        else:
            grad = tf.gradients(self.potential_energy(x, beta), [x])[0]

        return grad

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
        # log.info(f'Saving `xeps`, `veps`, `vnet`, `xnet` to: {outdir}')

        veps = np.array([e.numpy() for e in self.veps])
        xeps = np.array([e.numpy() for e in self.xeps])

        np.savetxt(outdir.joinpath('veps.txt').as_posix(), xeps)
        np.savetxt(outdir.joinpath('xeps.txt').as_posix(), veps)
        np.save(outdir.joinpath('veps.npy').as_posix(), veps)
        np.save(outdir.joinpath('xeps.npy').as_posix(), xeps)

        if self.config.use_separate_networks:
            for lf in range(self.config.nleapfrog):
                fvnet = outdir.joinpath(f'vnet-{lf}').as_posix()
                fxnet1 = outdir.joinpath(f'xnet-{lf}_first').as_posix()

                vnet = self._get_vnet(lf)
                xnet1 = self._get_xnet(lf, first=True)

                # log.info(f'Saving `vnet-{lf} to {fvnet}')
                # log.info(f'Saving `xnet-{lf}_first to {fxnet1}')

                vnet.save(fvnet)
                xnet1.save(fxnet1)

                if self.config.use_split_xnets:
                    xnet2 = self._get_xnet(lf, first=False)
                    fxnet2 = outdir.joinpath(f'xnet-{lf}_second').as_posix()
                    # log.info(f'Saving `xnet-{lf}_second to {fxnet2}')
                    xnet2.save(fxnet2)
        else:
            vnet = self._get_vnet(0)
            xnet1 = self._get_xnet(0, first=True)

            fvnet = outdir.joinpath('vnet').as_posix()
            fxnet1 = outdir.joinpath('xnet_first').as_posix()

            # log.info(f'Saving vnet to: {fvnet}')
            vnet.save(fvnet)
            # log.info(f'Saving xnet_first to: {fxnet1}')
            xnet1.save(fxnet1)

            if self.config.use_split_xnets:
                xnet2 = self._get_xnet(0, first=False)
                fxnet2 = outdir.joinpath('xnet_second').as_posix()
                # log.info(f'Saving xnet_second to: {fxnet2}')
                xnet2.save(fxnet2)
