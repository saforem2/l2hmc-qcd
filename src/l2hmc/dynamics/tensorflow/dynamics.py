"""
tensorflow/dynamics.py

Tensorflow implementation of Dynamics object for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass
from typing import Callable
from math import pi as PI
from typing import Tuple


import numpy as np
import tensorflow as tf

from l2hmc.configs import DynamicsConfig
from l2hmc.network.tensorflow.network import NetworkFactory

TWO_PI = 2. * PI
TWO = tf.constant(2.)

Tensor = tf.Tensor
Model = tf.keras.Model
TF_FLOAT = tf.keras.backend.floatx()


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


CallableNetwork = Callable[[tuple[Tensor, Tensor], bool],
                           tuple[Tensor, Tensor, Tensor]]


class Dynamics(Model):
    def __init__(
            self,
            potential_fn: Callable,
            config: DynamicsConfig,
            network_factory: NetworkFactory,
    ):
        if config.merge_directions:
            assert config.nleapfrog % 2 == 1, (' '.join([
                'If `config.merge_directions`, ',
                'we restrict `config.nleapfrog % 2 == 0` ',
                'to preserve reversibility.'
            ]))

        super(Dynamics, self).__init__()
        self.config = config
        self.xdim = self.config.xdim
        self.xshape = network_factory.input_spec.xshape
        self.potential_fn = potential_fn
        self.nlf = self.config.nleapfrog
        self.midpt = self.config.nleapfrog // 2
        self.xnet, self.vnet = self._build_networks(network_factory)
        self.masks = self._build_masks()

        self.xeps = [
            self._get_eps(name=f'xeps/lf{lf}')
            for lf in range(self.config.nleapfrog)
        ]
        self.veps = [
            self._get_eps(name=f'veps/lf{lf}')
            for lf in range(self.config.nleapfrog)
        ]

    def _build_networks(self, network_factory):
        split = self.config.use_split_xnets
        n = self.nlf if self.config.use_separate_networks else 1
        networks = network_factory.build_networks(n, split)
        return networks['xnet'], networks['vnet']

    def _get_eps(self, name: str) -> tf.Variable:
        # if not self.config.merge_directions:
        constraint = tf.keras.constraints.non_neg()
        einit = tf.cast(self.config.eps, dtype=TF_FLOAT)
        alpha = tf.constant(tf.math.exp(tf.math.log(einit)))
        # else:
        #     constraint = None
        #     init = tf.constant(tf.cast(self.config.eps, dtype=TF_FLOAT))

        return tf.Variable(initial_value=alpha,
                           name=name,
                           dtype=TF_FLOAT,
                           constraint=constraint,
                           trainable=(not self.config.eps_fixed))

    def call(
            self,
            inputs: tuple[Tensor, Tensor],
            training: bool = True
    ) -> tuple[Tensor, dict]:
        """Call Dynamics object."""
        if self.config.merge_directions:
            return self.apply_transition_fb(inputs, training=training)
        return self.apply_transition(inputs, training=training)

    def apply_transition_fb(
            self,
            inputs: tuple[Tensor, Tensor],
            training: bool = True,
    ) -> tuple[Tensor, dict]:
        data = self._transition_fb(inputs, training)
        ma_, mr_ = self._get_accept_masks(data['metrics']['acc'])
        ma = ma_[:, None]
        mr = mr_[:, None]

        v_out = ma * data['proposed'].v + mr * data['init'].v
        x_out = ma * data['proposed'].x + mr * data['init'].x
        logdet = ma_ * data['metrics']['sumlogdet']

        state_out = State(x=x_out, v=v_out, beta=data['init'].beta)
        mc_states = MonteCarloStates(init=data['init'],
                                     proposed=data['proposed'],
                                     out=state_out)
        metrics = {
            'acc': data['metrics']['acc'],
            'acc_mask': ma_,
            'sumlogdet': logdet,
            'mc_states': mc_states,
        }

        return (mc_states.out.x, metrics)

    def apply_transition(
            self,
            inputs: tuple[Tensor, Tensor],
            training: bool = True
    ) -> tuple[Tensor,  dict]:
        x, beta = inputs
        fwd = self._transition(inputs, forward=True, training=training)
        bwd = self._transition(inputs, forward=False, training=training)

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
        logdet = ma_ * logdet_prop  # + mr_ * logdet_init (= 0.)

        init = State(x=x, v=v_init, beta=beta)
        prop = State(x=x_prop, v=v_prop, beta=beta)
        out = State(x=x_out, v=v_out, beta=beta)

        mc_states = MonteCarloStates(init=init, proposed=prop, out=out)

        metrics = {
            'acc': acc,
            'acc_mask': ma_,
            'sumlogdet': logdet,
            'mc_states': mc_states,
            # 'logdet': logdet,
            # 'forward': mfwd,
            # 'backward': mbwd,
        }
        metrics.update({f'forward/{k}': v for k, v in mfwd.items()})
        metrics.update({f'backward/{k}': v for k, v in mbwd.items()})

        return (mc_states.out.x, metrics)

    def _transition_fb(
            self, inputs: tuple[Tensor, Tensor], training: bool = True,
    ) -> dict:
        """Run the transition kernel to generate a proposal configuration."""
        return self.generate_proposal_fb(inputs, training=training)

    def _transition(
            self,
            inputs: tuple[Tensor, Tensor],
            forward: bool,
            training: bool = True,
    ) -> dict:
        """Run the transition kernel to generate a proposal configuration."""
        return self.generate_proposal(inputs, forward, training=training)

    def generate_proposal_fb(
            self,
            inputs: tuple[Tensor, Tensor],
            training: bool = True,
    ) -> dict:
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
        x, beta = inputs
        v = tf.random.normal(x.shape, dtype=TF_FLOAT)
        state_init = State(x=x, v=v, beta=beta)
        state_prop, metrics = self.transition_kernel(state_init,
                                                     forward=forward,
                                                     training=training)

        return {'init': state_init, 'proposed': state_prop, 'metrics': metrics}

    def _new_history(self, state: State) -> dict:
        """Create new history object to track metrics over trajectory.."""
        if not self.config.verbose:
            return {}

        kwargs = {
            'dynamic_size': False,
            'clear_after_read': False,
            'size': self.config.nleapfrog+1,
            'element_shape': (state.x.shape[0],),
        }
        # sumlogdet = tf.TensorArray(TF_FLOAT, **kwargs)
        # energies = tf.TensorArray(TF_FLOAT, **kwargs)
        # logprobs = tf.TensorArray(TF_FLOAT, **kwargs)

        energy = self.hamiltonian(state)
        logdet = tf.zeros(state.x.shape[0], dtype=TF_FLOAT)

        return {
            # by duplicating the first elements in xeps and veps,
            # all of the entries in this dict have size nleapfrog + 1
            # 'xeps': [self.xeps[0], *self.xeps],
            # 'veps': [self.veps[0], *self.veps],
            # 'logdet': logdets.write(0, logdet),
            'sumlogdet': logdet,  # .write(0, logdet),
            'energy': energy,  # .write(0, energy),
            'logprob': energy - logdet,  # .write(0, energy - logdet),
        }

    def _get_metrics(self, s: State, sumlogdet: Tensor) -> dict[str, Tensor]:
        h = self.hamiltonian(s)
        logprob = tf.subtract(h, sumlogdet)
        return {
            'energy': h,
            'sumlogdet': sumlogdet,
            'logprob': logprob,
        }

    @staticmethod
    def _update_history(history: dict, data: dict, step: int) -> dict:
        for key, val in data.items():
            try:
                history[key] = history[key].write(step, val)
            except AttributeError:
                continue

        return history

    def _transition_kernel_fb(
            self,
            state: State,
            training: bool = True,
    ) -> tuple[State, dict]:
        state_ = State(state.x, state.v, state.beta)
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=TF_FLOAT)
        # history = self._new_history(state_)
        history = {}

        def _update_history(hist: dict, data: dict, step: int) -> dict:
            for key, val in data.items():
                # try:
                #     hist[key] = hist[key].write(step, val)
                # except AttributeError:
                #     continue
                try:
                    hist[key].append(val)
                except KeyError:
                    hist[key] = [val]

            return hist

        def _stack_history(hist: dict) -> dict:
            for key, val in hist.items():
                if isinstance(val, tf.TensorArray):
                    val = val.stack()
                if tf.is_tensor(val):
                    try:
                        val = val.numpy()  # type: ignore
                    except Exception:
                        continue
                hist[key] = val
            return hist

        # Loop over leapfrog steps
        for step in range(self.config.nleapfrog):
            if step <= self.config.nleapfrog // 2:
                state_, logdet = self._forward_lf(step, state_, training)
            else:
                step_r = self.config.nleapfrog - step - 1
                state_, logdet = self._backward_lf(step_r, state_, training)

            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                metrics = self._get_metrics(state_, sumlogdet)
                history = _update_history(history, metrics, step+1)

        acc = self.compute_accept_prob(state, state_, sumlogdet)
        history.update({
            'acc': acc,
            'sumlogdet': sumlogdet,
            'xeps': tf.convert_to_tensor([self.xeps[0], *self.xeps]),
            'veps': tf.convert_to_tensor([self.veps[0], *self.veps]),
            # 'xeps': [self.xeps[0], *self.xeps],
            # 'veps': [self.veps[0], *self.veps],
        })
        if self.config.verbose:
            history = _stack_history(history)

        return state_, history

    def _transition_kernel(
            self,
            state: State,
            forward: bool,
            training: bool = True,
    ) -> tuple[State, dict]:
        """Implements the transition kernel."""
        lf_fn = self._forward_lf if forward else self._backward_lf
        # Copy initial state into proposed state
        state_ = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=TF_FLOAT)
        history = self._new_history(state_)

        def _update_history(hist: dict, data: dict, step: int) -> dict:
            # we implicitly assume that all of data's entries are TensorArray's
            for key, val in data.items():
                try:
                    hist[key] = hist[key].write(step, val)
                except AttributeError:  # can't call write method on entry
                    continue

            return hist

        def _stack_history(hist: dict) -> dict:
            for key, val in hist.items():
                if isinstance(val, tf.TensorArray):
                    val = val.stack()
                if tf.is_tensor(val):
                    try:
                        val = val.numpy()  # type: ignore
                    except AttributeError:
                        continue
                hist[key] = val
            return hist

        # Loop over leapfrog steps
        for step in range(self.config.nleapfrog):
            state_, logdet = lf_fn(step, state_, training)
            sumlogdet = sumlogdet + logdet
            if self.config.verbose:
                metrics = self._get_metrics(state_, sumlogdet)
                history = _update_history(history, metrics, step+1)

        acc = self.compute_accept_prob(state, state_, sumlogdet)
        history.update({
            'acc': acc,
            'sumlogdet': sumlogdet,
            'xeps': tf.convert_to_tensor([self.xeps[0], *self.xeps]),
            'veps': tf.convert_to_tensor([self.veps[0], *self.veps]),
        })
        if self.config.verbose:
            history = _stack_history(history)

        return state_, history

    def transition_kernel(
            self,
            state: State,
            forward: bool,
            training: bool = True,
    ) -> tuple[State, dict]:
        """Transition kernel of the augmented leapfrog integrator."""
        return self._transition_kernel(state, forward, training)

    def transition_kernel_fb(
            self,
            state: State,
            training: bool = True,
    ) -> tuple[State, dict]:
        return self._transition_kernel_fb(state, training=training)

    def compute_accept_prob(
            self,
            state_init: State,
            state_prop: State,
            sumlogdet: Tensor,
    ) -> Tensor:
        h_init = self.hamiltonian(state_init)
        h_prop = self.hamiltonian(state_prop)
        dh = tf.add(tf.subtract(h_init, h_prop), sumlogdet)
        # dh = h_init - h_prop + sumlogdet
        prob = tf.exp(tf.minimum(dh, 0.))

        return tf.where(tf.math.is_finite(prob), prob, tf.zeros_like(prob))

    @staticmethod
    def _get_accept_masks(px: Tensor) -> tuple:
        acc = tf.cast(
            px > tf.random.uniform(tf.shape(px), dtype=TF_FLOAT),
            dtype=TF_FLOAT,
        )
        rej = tf.ones_like(acc) - acc

        return acc, rej

    def _get_direction_masks(self, batch_size) -> tuple:
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
        m = self.masks[i]
        mb = tf.ones_like(m) - m
        return (m, mb)

    def _get_vnet(self, step: int) -> CallableNetwork:
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
        vnet = self._get_vnet(step)
        assert callable(vnet)
        return vnet(inputs, training)

    def _get_xnet(self, step: int, first: bool) -> CallableNetwork:
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
        sumlogdet += logdet

        state, logdet = self._update_x_fwd(step, state, m,
                                           first=True, training=training)
        sumlogdet += logdet

        state, logdet = self._update_x_fwd(step, state, mb,
                                           first=False, training=training)
        sumlogdet += logdet

        state, logdet = self._update_v_fwd(step, state, training=training)
        sumlogdet += logdet

        return state, sumlogdet

    def _backward_lf(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        """Complete update (leapfrog step) in the backward direction."""
        # if self.config.merge_directions:
        #     step_r = step
        # else:
        # Note: Reverse the step count, i.e. count from end of trajectory.
        step_r = self.config.nleapfrog - step - 1

        m, mb = self._get_mask(step_r)
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=TF_FLOAT)

        state, logdet = self._update_v_bwd(step_r, state, training=training)
        sumlogdet += logdet

        state, logdet = self._update_x_bwd(step_r, state, mb,
                                           first=False, training=training)
        sumlogdet += logdet

        state, logdet = self._update_x_bwd(step_r, state, m,
                                           first=True, training=training)
        sumlogdet += logdet

        state, logdet = self._update_v_bwd(step_r, state, training=training)
        sumlogdet += logdet

        return state, sumlogdet

    def _update_v_fwd(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        eps = self.veps[step]
        # half_eps = tf.cast(eps / 2, dtype=TF_FLOAT)
        force = self.grad_potential(state.x, state.beta)
        s, t, q = self._call_vnet(step, (state.x, force), training=training)

        jac = eps * s / 2.  # jacobian factor, also used in exp_s below
        logdet = tf.reduce_sum(jac, axis=1)

        exp_s = tf.exp(jac)
        exp_q = tf.exp(eps * q)
        vf = exp_s * state.v - (eps * (force * exp_q + t) / 2.)

        # s = tf.scalar_mul(half_eps, s)
        # q = tf.scalar_mul(eps, q)

        # vf = state.v * tf.exp(s) - half_eps * (force * tf.exp(q) - t)
        # logdet = tf.reduce_sum(s, axis=1)

        return State(state.x, vf, state.beta), logdet

    def _update_v_bwd(
            self,
            step: int,
            state: State,
            training: bool = True,
    ) -> tuple[State, Tensor]:
        eps = self.veps[step]
        # half_eps = tf.cast(eps / 2, dtype=TF_FLOAT)
        force = self.grad_potential(state.x, state.beta)
        s, t, q = self._call_vnet(step, (state.x, force), training=training)
        jac = (-eps * s / 2.)
        logdet = tf.reduce_sum(jac, axis=1)
        exp_s = tf.exp(jac)
        exp_q = tf.exp(eps * q)
        vb = exp_s * (state.v + 0.5 * eps * (force * exp_q + t))
        # s = tf.scalar_mul(half_eps, s)
        # q = tf.scalar_mul(eps, q)
        # vb = tf.exp(s) * (state.v + half_eps * (force * tf.exp(q) - t))
        # logdet = tf.reduce_sum(s, axis=1)

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
            # halfx = tf.divide(state.x, 2.)
            halfx = state.x / tf.constant(2.)
            _x = tf.constant(2.) * tf.math.atan(tf.math.tan(halfx) * exp_s)
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
        kinetic = self.kinetic_energy(state.v)
        potential = self.potential_energy(state.x, state.beta)
        return tf.add(kinetic, potential)

    @staticmethod
    def kinetic_energy(v: Tensor) -> Tensor:
        return tf.reduce_sum(tf.math.square(v), axis=1) / 2.

    def potential_energy(self, x: Tensor, beta: Tensor) -> Tensor:
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
