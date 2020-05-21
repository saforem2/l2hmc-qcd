"""
dynamics.py

Dynamics engine for L2HMC sampler on Lattice Gauge Models.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Reference [Robust Parameter Estimation with a Neural Network Enhanced
Hamiltonian Markov Chain Monte Carlo Sampler]
https://infoscience.epfl.ch/record/264887/files/robust_parameter_estimation.pdf

Author: Sam Foreman (github: @saforem2)
Date: 1/14/2019
"""
from __future__ import absolute_import, division, print_function

from typing import Callable, List, NoReturn, Tuple
from collections import namedtuple

import numpy as np
import tensorflow as tf

from config import NP_FLOAT, TF_FLOAT, TF_INT
from seed_dict import seeds, vnet_seeds, xnet_seeds
from network import NetworkConfig
#  from network.network import FullNet
from network.gauge_network import GaugeNetwork

# pylint:disable=invalid-name,too-many-locals,too-many-arguments

PI = np.pi
TWO_PI = 2 * PI

DynamicsConfig = namedtuple('DynamicsConfig', [
    'num_steps', 'eps', 'input_shape',
    'hmc', 'eps_trainable', 'net_weights',
    'model_type',
])

State = namedtuple('State', ['x', 'v', 'beta'])
MonteCarloStates = namedtuple('MonteCarloStates', ['init', 'proposed', 'out'])


def cast_float(x, dtype=NP_FLOAT):
    """Cast `x` to `dtype`."""
    if dtype == np.float64:
        return np.float64(x)
    if dtype == np.float32:
        return np.float32(x)
    return np.float(x)


def convert_to_angle(x):
    """Returns x in -pi <= x < pi."""
    x = tf.math.floormod(x + PI, TWO_PI) - PI
    return x


# pylint:disable=too-many-instance-attributes,unused-argument
class Dynamics(tf.keras.Model):
    """DynamicsObject for training the L2HMC sampler."""
    def __init__(self,
                 potential_fn: Callable[[tf.Tensor], tf.Tensor],
                 dynamics_config: DynamicsConfig,
                 network_config: NetworkConfig) -> NoReturn:
        """Initialization method."""
        super(Dynamics, self).__init__(name='Dynamics')
        np.random.seed(seeds['global_np'])

        self.config = dynamics_config
        self.net_config = network_config
        self._potential_fn = potential_fn
        self._model_type = self.config.model_type

        self.net_weights = self.config.net_weights
        self._xsw = self.net_weights.x_scale
        self._xtw = self.net_weights.x_translation
        self._xqw = self.net_weights.x_transformation
        self._vsw = self.net_weights.v_scale
        self._vtw = self.net_weights.v_translation
        self._vqw = self.net_weights.v_transformation

        self.x_shape = self.config.input_shape
        self.batch_size = self.x_shape[0]
        self.xdim = np.cumprod(self.x_shape[1:])[-1]

        self.eps = self._build_eps(use_log=False)
        self.masks = self._build_masks()

        self.xnet, self.vnet = self.build_network(network_config)

    def build_network(self, network_config):
        """Build the networks to be used during training."""
        if self.config.hmc:
            xnet = lambda inputs, is_training: [  # noqa: E731
                tf.zeros_like(inputs[0]) for _ in range(3)
            ]
            vnet = lambda inputs, is_training: [  # noqa: E731
                tf.zeros_like(inputs[0]) for _ in range(3)
            ]

        else:
            if network_config.type == 'GaugeNetwork':
                xnet = GaugeNetwork(network_config, self.xdim, factor=2.,
                                    net_seeds=xnet_seeds, name='XNet')
                vnet = GaugeNetwork(network_config, self.xdim, factor=1.,
                                    net_seeds=vnet_seeds, name='VNet')

            # TODO: Update `CartesianNet` and remainder of network objects to
            # use generic `NetworkConfig` instead of explicitly passing
            # parameters.
            else:
                #  net_params['factor'] = 2.
                #  net_params['net_name'] = 'x'
                #  net_params['net_seeds'] = xnet_seeds
                #  xnet = FullNet(model_name='XNet', **net_params)
                #
                #  net_params['factor'] = 1.
                #  net_params['net_name'] = 'v'
                #  net_params['net_seeds'] = vnet_seeds
                #  vnet = FullNet(model_name='VNet', **net_params)
                pass

        return xnet, vnet

    def _build_eps(self, use_log=False):
        """Create `self.eps` (i.e. the step size) as a `tf.Variable`.

        Args:
            use_log (bool): If True, initialize `log_eps` as the actual
            `tf.Variable` and set `self.eps = tf.exp(log_eps)`; otherwise, set
            `self.eps` as a `tf.Variable` directly.

        Returns:
            eps: The (trainable) step size to be used in the L2HMC algorithm.
        """
        if use_log:
            init = tf.math.log(tf.constant(self.config.eps))
        else:
            init = tf.constant(self.config.eps)

        kwargs = {
            'name': 'eps',
            'dtype': TF_FLOAT,
            'trainable': self.config.eps_trainable,
        }

        return tf.Variable(initial_value=init, **kwargs)

    def call(self, inputs, training=None):
        """Obtain a new state from `inputs`."""
        return self.apply_transition(inputs, training=None)

    def apply_transition(self, inputs, training=None):
        """Propose a new state and perform the accept reject step.

        We simulate the molecular dynamics update both forward and backward,
        and use sampled masks to compute the actual solutions.
        """
        x_init, beta = inputs
        sf_init, sf_prop, pxf, sldf = self._transition(inputs,
                                                       forward=True,
                                                       training=training)
        sb_init, sb_prop, pxb, sldb = self._transition(inputs,
                                                       forward=False,
                                                       training=training)
        mask_f, mask_b = self._get_direction_masks()
        v_init = sf_init.v * mask_f[:, None] + sb_init.v * mask_b[:, None]
        x_prop = sf_prop.x * mask_f[:, None] + sb_prop.x * mask_b[:, None]
        v_prop = sf_prop.v * mask_f[:, None] + sb_prop.v * mask_b[:, None]
        sld_prop = sldf * mask_f + sldb * mask_b  # sumlogdet proposed

        state_init = State(x_init, v_init, beta)
        state_prop = State(x_prop, v_prop, beta)

        accept_prob = pxf * mask_f + pxb * mask_b
        mask_a, mask_r = self._get_accept_masks(accept_prob)

        x_out = x_prop * mask_a[:, None] + x_init * mask_r[:, None]
        v_out = v_prop * mask_a[:, None] + v_init * mask_r[:, None]
        sumlogdet = sld_prop * mask_a

        state_out = State(x_out, v_out, beta)
        mc_states = MonteCarloStates(state_init, state_prop, state_out)
        sld_states = MonteCarloStates(0., sld_prop, sumlogdet)

        return mc_states, accept_prob, sld_states

    def _transition(self, inputs, forward=True, training=None):
        x_init, beta = inputs
        v_init = tf.random.normal(tf.shape(x_init))
        state = State(x_init, v_init, beta)
        state_prop, px, sld = self.transition_kernel(state, forward, training)

        return state, state_prop, px, sld

    def transition_kernel(self, state, forward, training=None):
        """Transition kernel of the augmented leapfrog integrator."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        step = tf.constant(0., name='md_step', dtype=TF_FLOAT)
        logdet = tf.zeros((self.batch_size,), dtype=TF_FLOAT)

        def body(step, state, logdet):
            new_state, j = lf_fn(step, state, training)

            return step+1, new_state, logdet+j

        def cond(step, *args):
            return tf.less(step, self.config.num_steps)

        _, state_prop, sumlogdet = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[step, state, logdet]
        )

        accept_prob = self._compute_accept_prob(state, state_prop, sumlogdet)

        return state_prop, accept_prob, sumlogdet

    def _forward_lf(self, step, state, training=None):
        t = self._get_time(step, tile=tf.shape(state.x)[0])
        m, mc = self._get_mask(step)

        sumlogdet = 0.
        state, logdet = self._update_v_forward(state, t, training)
        sumlogdet += logdet
        state, logdet = self._update_x_forward(state, t, (m, mc), training)
        sumlogdet += logdet
        state, logdet = self._update_x_forward(state, t, (mc, m), training)
        sumlogdet += logdet
        state, logdet = self._update_v_forward(state, t, training)
        sumlogdet += logdet

        return state, sumlogdet

    def _backward_lf(self, step, state, training=None):
        step_r = self.config.num_steps - step - 1
        t = self._get_time(step_r)
        m, mc = self._get_mask(step_r)

        sumlogdet = 0.
        state, logdet = self._update_v_backward(state, t, training)
        sumlogdet += logdet
        state, logdet = self._update_x_backward(state, t, (mc, m), training)
        sumlogdet += logdet
        state, logdet = self._update_x_backward(state, t, (m, mc), training)
        sumlogdet += logdet
        state, logdet = self._update_v_backward(state, t, training)
        sumlogdet += logdet

        return state, sumlogdet

    def _update_v_forward(self, state, t, training):
        """Update the momentum `v` in the forward leapfrog step.

        Args:
            state (State): Input state
            t (float): Current leapfrog step, represented as periodic time.
            training (bool): Currently training?

        Returns:
            new_state (State): New state, with updated momentum.
            logdet (float): Jacobian factor.
        """
        if self._model_type == 'GaugeModel':
            x = convert_to_angle(state.x)

        grad = self.grad_potential(x, state.beta)
        #  net_outputs = self.vnet((x, grad, t), training)
        #  Sv, Tv, Qv = self._scale_momentum_outputs(net_outputs)
        Sv, Tv, Qv = self.vnet((x, grad, t), training)

        scale = self._vsw * (0.5 * self.eps * Sv)
        transl = self._vtw * Tv
        transf = self._vqw * (self.eps * Qv)

        exp_s = tf.exp(scale)
        exp_q = tf.exp(transf)

        vf = state.v * exp_s - 0.5 * self.eps * (grad * exp_q + transl)
        logdet = tf.reduce_sum(scale, axis=-1)

        state_out = State(x, vf, state.beta)

        return state_out, logdet

    def _update_x_forward(self, state, t, masks, training):
        """Update the position `x` in the forward leapfrog step.

        Args:
            state (State): Input state
            t (float): Current leapfrog step, represented as periodic time.
            training (bool): Currently training?

        Returns:
            new_state (State): New state, with updated momentum.
            logdet (float): Jacobian factor.
        """
        if self._model_type == 'GaugeModel':
            x = convert_to_angle(state.x)

        m, mc = masks
        #  net_outputs = self.xnet((state.v, m * x, t), training)
        #  Sx, Tx, Qx = self._scale_position_outputs(net_outputs)
        Sx, Tx, Qx = self.xnet((state.v, m * x, t), training)

        scale = self._xsw * (self.eps * Sx)
        transl = self._xtw * Tx
        transf = self._xqw * (self.eps * Qx)

        exp_s = tf.exp(scale)
        exp_q = tf.exp(transf)

        y = x * exp_s + self.eps * (state.v * exp_q + transl)
        xf = m * x + mc * y

        if self._model_type == 'GaugeModel':
            xf = convert_to_angle(xf)

        state_out = State(xf, state.v, state.beta)
        logdet = tf.reduce_sum(mc * scale, axis=-1)

        return state_out, logdet

    def _update_v_backward(self, state, t, training):
        """Update the momentum `v` in the backward leapfrog step.

        Args:
            state (State): Input state.
            t (float): Current leapfrog step, represented as periodic time.
            training (bool): Currently training?

        Returns:
            new_state (State): New state, with updated momentum.
            logdet (float): Jacobian factor.
        """
        if self._model_type == 'GaugeModel':
            x = convert_to_angle(state.x)

        grad = self.grad_potential(x, state.beta)
        #  net_outputs = self.vnet((x, grad, t), training)
        #  Sv, Tv, Qv = self._scale_momentum_outputs(net_outputs)
        Sv, Tv, Qv = self.vnet((x, grad, t), training)

        scale = self._vsw * (-0.5 * self.eps * Sv)
        transl = self._vtw * Tv
        transf = self._vqw * (self.eps * Qv)

        exp_s = tf.exp(scale)
        exp_q = tf.exp(transf)

        #  exp_s = tf.exp(-0.5 * self.eps * Sv)
        #  exp_q = tf.exp(0.5 * self.eps * Qv)
        vb = exp_s * (state.v + 0.5 * self.eps * (grad * exp_q + transl))

        state_out = State(x, vb, state.beta)
        logdet = tf.reduce_sum(scale, axis=-1)

        return state_out, logdet

    def _update_x_backward(self, state, t, masks, training):
        """Update the position `x` in the forward leapfrog step.

        Args:
            state (State): Input state
            t (float): Current leapfrog step, represented as periodic time.
            training (bool): Currently training?

        Returns:
            new_state (State): New state, with updated momentum.
            logdet (float): Jacobian factor.
        """
        x = state.x
        if self._model_type == 'GaugeModel':
            x = convert_to_angle(state.x)

        m, mc = masks
        Sx, Tx, Qx = self.xnet((state.v, m * x, t), training)
        #  Sx, Tx, Qx = self._scale_position_outputs(net_outputs)

        scale = self._xsw * (-self.eps * Sx)
        transl = self._xtw * Tx
        transf = self._xqw * (self.eps * Qx)

        exp_s = tf.exp(scale)
        exp_q = tf.exp(transf)

        y = exp_s * (x - self.eps * (state.v * exp_q + transl))
        xb = m * x + mc * y
        if self._model_type == 'GaugeModel':
            xb = convert_to_angle(xb)

        state_out = State(xb, state.v, state.beta)
        logdet = tf.reduce_sum(mc * scale, axis=1)

        return state_out, logdet

    def _compute_accept_prob(self, state_init, state_prop, sumlogdet):
        """Compute the probability of accepting the proposed state.

        Args:
            state_init (State): Initial state.
            state_prop (State): Proposed state.
            sumlogdet (float): Sum of the log of the determinant of the
                transformation.

        Returns:
            accept_prob (tf.Tensor): Acceptance probabilities.
        """
        h_init = self.hamiltonian(state_init)
        h_prop = self.hamiltonian(state_prop)
        dh = h_init - h_prop + sumlogdet
        prob = tf.exp(tf.minimum(dh, 0.))

        return tf.where(tf.math.is_finite(prob), prob, tf.zeros_like(prob))

    def _get_time(self, i, tile=1):
        """Format the MCMC step as [cos(...), sin(...)]."""
        trig_t = tf.squeeze([
            tf.cos(TWO_PI * i / self.config.num_steps),
            tf.sin(TWO_PI * i / self.config.num_steps),
        ])

        t = tf.tile(tf.expand_dims(trig_t, 0), (tile, 1))

        return t

    def _get_direction_masks(self):
        forward_mask = tf.cast(
            tf.random.uniform((self.batch_size,)) > 0.5,
            dtype=TF_FLOAT,
        )
        backward_mask = 1. - forward_mask

        return forward_mask, backward_mask

    @staticmethod
    def _get_accept_masks(accept_prob):
        accept_mask = tf.cast(
            accept_prob > tf.random.uniform(tf.shape(accept_prob)),
            dtype=TF_FLOAT,
        )
        reject_mask = 1. - accept_mask

        return accept_mask, reject_mask

    def _build_masks(self):
        """Construct different binary masks for different time steps."""
        masks = []
        for _ in range(self.config.num_steps):
            # Need to use numpy.random here because tensorflow would generate
            # different random values across different calls.
            #  _idx = np.arange(self.xdim)
            #  idx = np.random.permutation(_idx)[:self.xdim // 2]
            #  mask = np.zeros((self.xdim,))
            #  mask[idx] = 1.
            mask = np.arange(self.xdim) % 2
            mask = tf.constant(mask, dtype=TF_FLOAT)
            masks.append(mask[None, :])

        return masks

    def _get_mask(self, step):
        """Retrieve the binary mask associated with the time step `step`."""
        if tf.executing_eagerly():
            m = self.masks[int(step)]
        else:
            m = tf.gather(self.masks, tf.cast(step, dtype=TF_INT))

        return m, 1. - m

    def grad_potential(self, x, beta):
        """Get the gradient of the potential function at current `x`."""
        if self._model_type == 'GaugeModel':
            x = convert_to_angle(x)
        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                tape.watch(x)
                action = self.potential_energy(x, beta)
            grad = tape.gradient(action, x)
            grad = tf.reshape(grad, (self.batch_size, -1))

        else:
            grad = tf.gradients(self.potential_energy(x, beta), [x])[0]

        return grad

    def potential_energy(self, x, beta):
        """Compute the potential energy as beta times the potential fn."""
        if self._model_type == 'GaugeModel':
            x = convert_to_angle(x)

        return beta * self._potential_fn(x)

    @staticmethod
    def kinetic_energy(v):
        """Compute the kinetic energy of the momentum as 0.5 * (v ** 2)."""
        return 0.5 * tf.reduce_sum(v**2, axis=-1)

    def hamiltonian(self, state):
        """Compute the overall hamiltonian."""
        kinetic = self.kinetic_energy(state.v)
        potential = self.potential_energy(state.x, state.beta)

        return potential + kinetic
