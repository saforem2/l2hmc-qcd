"""
dynamics_base.py

Implementes `BaseDynamics`, an abstract base class for the dynamics engine of
the L2HMC sampler.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Reference [Robust Parameter Estimation with a Neural Network Enhanced
Hamiltonian Markov Chain Monte Carlo Sampler]
https://infoscience.epfl.ch/record/264887/files/robust_parameter_estimation.pdf

Author: Sam Foreman (github: @saforem2)
Date: 6/30/2020
"""
from __future__ import absolute_import, division, print_function

import os


from typing import Callable, NoReturn

import numpy as np
import tensorflow as tf

from config import (DynamicsConfig, MonteCarloStates, NET_WEIGHTS_HMC,
                    NET_WEIGHTS_L2HMC, NetworkConfig, State,
                    TF_FLOAT, TF_INT, lrConfig, BIN_DIR)
from network.gauge_network import GaugeNetwork
from utils.attr_dict import AttrDict
from utils.seed_dict import vnet_seeds, xnet_seeds
from utils.learning_rate import WarmupExponentialDecay
from utils.file_io import timeit

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


TIMING_FILE = os.path.join(BIN_DIR, 'timing_file.log')


def identity(x):
    """Returns x"""
    return x


# pylint:disable=invalid-name, too-many-arguments
# pylint:disable=attribute-defined-outside-init, too-many-locals
# pylint:disable=too-many-instance-attributes
class BaseDynamics(tf.keras.Model):
    """Dynamics object for training the L2HMC sampler."""

    def __init__(
            self,
            params: AttrDict,
            config: DynamicsConfig,
            network_config: NetworkConfig,
            potential_fn: Callable[[tf.Tensor], tf.Tensor],
            lr_config: lrConfig = None,
            normalizer: Callable[[tf.Tensor], tf.Tensor] = None,
            name: str = 'Dynamics',
    ):
        """Initialization method.

        * NOTE: `normalizer` is meant to be a function that forces the
            configurations to exist in some restricted space. For example,
            in the case of a 2D U(1) lattice gauge model, we restrict the
            configurations `x` to reside in [-pi, pi] (the U(1) gauge group).
            If not specified, `normalizer` will default to `identity` defined
            above, which just returns the input.
        """
        super(BaseDynamics, self).__init__(name=name)
        self._model_type = config.model_type

        self.params = params
        self.mask_type = params.get('mask_type', 'rand')
        self.config = config
        self.net_config = network_config
        self.potential_fn = potential_fn
        self.normalizer = normalizer if normalizer is not None else identity

        self._parse_params(params)
        self.eps = self._build_eps(use_log=False)
        self.masks = self._build_masks(self.mask_type)
        #  self._construct_time()
        if self.config.hmc:
            self.xnets, self.vnets = self._build_hmc_networks()
        else:
            self._build_networks()

        if self._has_trainable_params:
            self.lr_config = lr_config
            self.lr = self._create_lr(lr_config)
            self.optimizer = self._create_optimizer()

    def _parse_params(self, params):
        """Set instance attributes from `params`."""
        self.xdim = params.get('xdim', None)
        self.batch_size = params.get('batch_size', None)
        self.using_hvd = params.get('horovod', False)
        self.x_shape = (self.batch_size, self.xdim)
        self.clip_val = params.get('clip_val', 0.)
        #  self.loss_scale = params.get('loss_scale', 0.1)

        # Determine if there are any parameters to be trained
        self._has_trainable_params = True
        if self.config.hmc and not self.config.eps_trainable:
            self._has_trainable_params = False

        self.net_weights = (
            NET_WEIGHTS_HMC if self.config.hmc
            else NET_WEIGHTS_L2HMC
        )

        self._xsw = self.net_weights.x_scale
        self._xtw = self.net_weights.x_translation
        self._xqw = self.net_weights.x_transformation
        self._vsw = self.net_weights.v_scale
        self._vtw = self.net_weights.v_translation
        self._vqw = self.net_weights.v_transformation

    def call(self, inputs, training=None):
        """Obtain a new state from `inputs`."""
        return self.apply_transition(inputs, training=training)

    def calc_losses(self, inputs):
        """Calculate the total loss."""
        raise NotImplementedError

    def train_step(
            self,
            inputs: tuple,
            first_step: bool,
    ):
        """Perform a single training step."""
        raise NotImplementedError

    def test_step(self, inputs):
        """Perform a single inference step."""
        raise NotImplementedError

    def _apply_transition(self, inputs, training=None):
        """Propose a new state and perform the accept/reject step."""
        forward = tf.cast((tf.random.uniform(shape=[]) < 0.5), dtype=tf.bool)
        state_init, state_prop, px, sld = self._transition(inputs,
                                                           forward=forward,
                                                           training=training)
        ma, mr = self._get_accept_masks(px)
        ma_tensor = ma[:, None]
        mr_tensor = mr[:, None]
        x_out = state_prop.x * ma_tensor + state_init.x * mr_tensor
        v_out = state_prop.v * ma_tensor + state_init.v * mr_tensor
        sumlogdet = sld * ma

        state_out = State(x_out, v_out, state_init.beta)
        sld_states = MonteCarloStates(0., sld, sumlogdet)
        mc_states = MonteCarloStates(state_init, state_prop, state_out)

        return mc_states, px, sld_states

    def apply_transition(self, inputs, training=None):
        """Propose a new state and perform the accept/reject step.

        NOTE: We simulate the molecular dynamics update in both the forward and
        backward directions, and use sampled masks to compute the actual
        solutions.
        """
        x_init, beta = inputs
        # Simulate the dynamics both forward and backward;
        # Use sampled Bernoulli masks to compute the actual solutions
        sf_init, sf_prop, pxf, sldf = self._transition(inputs,
                                                       forward=True,
                                                       training=training)
        sb_init, sb_prop, pxb, sldb = self._transition(inputs,
                                                       forward=False,
                                                       training=training)

        # Combine the forward / backward outputs;
        # these are the proposed configuration
        mf_, mb_ = self._get_direction_masks()
        mf = mf_[:, None]
        mb = mb_[:, None]
        v_init = mf * sf_init.v + mb * sb_init.v
        x_prop = mf * sf_prop.x + mb * sb_prop.x
        v_prop = mf * sf_prop.v + mb * sb_prop.v
        sld_prop = mf_ * sldf + mb_ * sldb
        # Compute the acceptance probability
        accept_prob = mf_ * pxf + mb_ * pxb

        ma_, mr_ = self._get_accept_masks(accept_prob)
        ma = ma_[:, None]
        mr = mr_[:, None]

        # Construct the output configuration
        x_out = ma * x_prop + mr * x_init
        v_out = ma * v_prop + mr * v_init
        sumlogdet = ma_ * sld_prop  # NOTE: initial sumlogdet = 0

        state_init = State(x=x_init, v=v_init, beta=beta)
        state_prop = State(x=x_prop, v=v_prop, beta=beta)
        state_out = State(x=x_out, v=v_out, beta=beta)
        mc_states = MonteCarloStates(init=state_init,
                                     proposed=state_prop,
                                     out=state_out)
        sld_states = MonteCarloStates(init=0.,
                                      proposed=sld_prop,
                                      out=sumlogdet)

        return mc_states, accept_prob, sld_states

    def _transition(self, inputs, forward, training=None):
        """Run the augmented leapfrog integrator."""
        x_init, beta = inputs
        v_init = tf.random.normal(tf.shape(x_init))
        state_init = State(x=x_init, v=v_init, beta=beta)
        #  if self.config.separate_networks:
        #      tk_fn = self.transition_kernel_for
        #  else:
        #      tk_fn = self.transition_kernel_while

        state_prop, px, sld = self.transition_kernel_while(
            state_init, forward, training
        )

        return state_init, state_prop, px, sld

    def transition_kernel_while(self, state, forward, training=None):
        """Transition kernel using a `tf.while_loop` implementation."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        #  state_prop = State(*state)
        step = tf.constant(0, dtype=tf.int64)
        sld = tf.zeros((self.batch_size,), dtype=state.x.dtype)
        #  cond = tf.less(step, self.config.num_steps)
        #  while tf.less(step, self.config.num_steps):
        #      state_prop, logdet = lf_fn(step, state_prop, training=training)
        #      step += 1
        #      sumlogdet += logdet

        def body(step, state, logdet):
            new_state, logdet = lf_fn(step, state, training=training)
            return step+1, new_state, sld+logdet

        # pylint:disable=unused-argument
        def cond(step, *args):
            return tf.less(step, self.config.num_steps)

        _, state_prop, sumlogdet = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[step, state, sld]
        )

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)

        return state_prop, accept_prob, sumlogdet

    def transition_kernel_for(self, state, forward, training=None):
        """Transition kernel of the augmented leapfrog integrator."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        state_prop = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = tf.zeros((self.batch_size,), dtype=TF_FLOAT)
        for step in tf.range(self.config.num_steps):
            state_prop, logdet = lf_fn(step, state_prop, training)
            sumlogdet += logdet

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)

        return state_prop, accept_prob, sumlogdet

    def compute_accept_prob(self,
                            state_init: State,
                            state_prop: State,
                            sumlogdet: tf.Tensor):
        """Compute the acceptance prob. of `state_prop` given `state_init`.

        Returns: tf.Tensor
        """
        h_init = self.hamiltonian(state_init)
        h_prop = self.hamiltonian(state_prop)
        dh = h_init - h_prop + sumlogdet
        prob = tf.exp(tf.minimum(dh, 0.))

        return tf.where(tf.math.is_finite(prob), prob, tf.zeros_like(prob))

    def _forward_lf(self, step, state, training=None):
        """Run the augmented leapfrog integrator in the forward direction."""
        m, mc = self._get_mask(step)  # pylint: disable=invalid-name
        xnet, vnet = self._get_network(step)
        t = self._get_time_old(step, tile=tf.shape(state.x)[0])

        sumlogdet = tf.constant(0., dtype=state.x.dtype)

        state, logdet = self._update_v_forward(vnet, state, t, training)
        sumlogdet += logdet
        state, logdet = self._update_x_forward(xnet, state, t,
                                               (m, mc), training)
        sumlogdet += logdet
        state, logdet = self._update_x_forward(xnet, state, t,
                                               (mc, m), training)
        sumlogdet += logdet
        state, logdet = self._update_v_forward(vnet, state, t, training)
        sumlogdet += logdet

        return state, sumlogdet

    def _backward_lf(self, step, state, training=None):
        """Run the augmented leapfrog integrator in the backward direction."""
        step_r = self.config.num_steps - step - 1
        t = self._get_time_old(step_r, tile=tf.shape(state.x)[0])
        m, mc = self._get_mask(step_r)
        xnet, vnet = self._get_network(step_r)

        sumlogdet = 0.
        state, logdet = self._update_v_backward(vnet, state, t, training)
        sumlogdet += logdet
        state, logdet = self._update_x_backward(xnet, state, t,
                                                (mc, m), training)
        sumlogdet += logdet
        state, logdet = self._update_x_backward(xnet, state, t,
                                                (m, mc), training)
        sumlogdet += logdet
        state, logdet = self._update_v_backward(vnet, state, t, training)
        sumlogdet += logdet

        return state, sumlogdet

    def _update_v_forward(self, network, state, t, training):
        """Update the momentum `v` in the forward leapfrog step.

        Args:
            network (tf.keras.Layers): Network to use
            state (State): Input state
            t (float): Current leapfrog step, represented as periodic time.
            training (bool): Currently training?

        Returns:
            new_state (State): New state, with updated momentum.
            logdet (float): Jacobian factor
        """
        x = self.normalizer(state.x)

        grad = self.grad_potential(x, state.beta)
        S, T, Q = network((x, grad, t), training)

        transl = self._vtw * T
        scale = self._vsw * (0.5 * self.eps * S)
        transf = self._vqw * (self.eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        vf = state.v * expS - 0.5 * self.eps * (grad * expQ + transl)

        state_out = State(x=x, v=vf, beta=state.beta)
        logdet = tf.reduce_sum(scale, axis=1)

        return state_out, logdet

    def _update_x_forward(self, network, state, t, masks, training):
        """Update the position `x` in the forward leapfrog step.

        Args:
            state (State): Input state
            t (float): Current leapfrog step, represented as periodic time.
            training (bool): Currently training?

        Returns:
            new_state (State): New state, with updated position.
            logdet (float): Jacobian factor.
        """
        x = self.normalizer(state.x)

        m, mc = masks
        S, T, Q = network((state.v, m * x, t), training)

        transl = self._xtw * T
        scale = self._xsw * (self.eps * S)
        transf = self._xqw * (self.eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        y = x * expS + self.eps * (state.v * expQ + transl)
        xf = m * x + mc * y

        xf = self.normalizer(xf)

        state_out = State(x=xf, v=state.v, beta=state.beta)
        logdet = tf.reduce_sum(mc * scale, axis=1)

        return state_out, logdet

    def _update_v_backward(self, network, state, t, training):
        """Update the momentum `v` in the backward leapfrog step.

        Args:
            state (State): Input state.
            t (float): Current leapfrog step, represented as periodic time.
            training (bool): Currently training?

        Returns:
            new_state (State): New state, with updated momentum.
            logdet (float): Jacobian factor.
        """
        x = self.normalizer(state.x)

        grad = self.grad_potential(x, state.beta)
        S, T, Q = network((x, grad, t), training)

        scale = self._vsw * (-0.5 * self.eps * S)
        transf = self._vqw * (self.eps * Q)
        transl = self._vtw * T

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        vb = expS * (state.v + 0.5 * self.eps * (grad * expQ + transl))

        state_out = State(x=x, v=vb, beta=state.beta)
        logdet = tf.reduce_sum(scale, axis=1)

        return state_out, logdet

    def _update_x_backward(self, network, state, t, masks, training):
        """Update the position `x` in the forward leapfrog step.

        Args:
            state (State): Input state
            t (float): Current leapfrog step, represented as periodic time.
            training (bool): Currently training?

        Returns:
            new_state (State): New state, with updated momentum.
            logdet (float): Jacobian factor.
        """
        x = self.normalizer(state.x)

        m, mc = masks
        S, T, Q = network((state.v, m * x, t), training)

        scale = self._xsw * (-self.eps * S)
        transl = self._xtw * T
        transf = self._xqw * (self.eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        y = expS * (x - self.eps * (state.v * expQ + transl))
        xb = m * x + mc * y

        xb = self.normalizer(xb)

        state_out = State(x=xb, v=state.v, beta=state.beta)
        logdet = tf.reduce_sum(mc * scale, axis=1)

        return state_out, logdet

    def grad_potential(self, x, beta):
        """Compute the gradient of the potential function."""
        with tf.name_scope('grad_potential'):
            x = self.normalizer(x)
            if tf.executing_eagerly():
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    pe = self.potential_energy(x, beta)
                grad = tf.reshape(tape.gradient(pe, x), (self.batch_size, -1))
            else:
                grad = tf.gradients(self.potential_energy(x, beta), [x])[0]

        return grad

    def potential_energy(self, x, beta):
        """Compute the potential energy as beta times the potential fn."""
        with tf.name_scope('potential_energy'):
            x = self.normalizer(x)
            pe = beta * self.potential_fn(x)

        return pe

    @staticmethod
    def kinetic_energy(v):
        """Compute the kinetic energy of the momentum as 0.5 * (v ** 2)."""
        with tf.name_scope('kinetic_energy'):
            return 0.5 * tf.reduce_sum(v ** 2, axis=1)

    def hamiltonian(self, state):
        """Compute the overall Hamiltonian."""
        with tf.name_scope('hamiltonian'):
            kinetic = self.kinetic_energy(state.v)
            potential = self.potential_energy(state.x, state.beta)

        return potential + kinetic

    def _construct_time(self, tile):
        """Convert leapfrog step index into sinusoidal time."""
        #  t = self._get_time(step_r, tile=tf.shape(state.x)[0])
        self.ts = []
        for i in range(self.config.num_steps):
            t = tf.constant([
                np.cos(2 * np.pi * i / self.config.num_steps),
                np.sin(2 * np.pi * i / self.config.num_steps)
            ], dtype=TF_FLOAT)

            self.ts.append(tf.tile(tf.expand_dims(t, 0), (tile, 1)))

    def _get_time(self, i):
        """Format the MCMC step as [cos(...), sin(...)]."""
        return tf.gather(self.ts, i)

    def _get_time_old(self, i, tile=1):
        """Format the MCMC step as [cos(...), sin(...)]."""
        i = tf.cast(i, dtype=TF_FLOAT)
        trig_t = tf.squeeze([
            tf.cos(2 * np.pi * i / self.config.num_steps),
            tf.sin(2 * np.pi * i / self.config.num_steps),
        ])
        t = tf.tile(tf.expand_dims(trig_t, 0), (tile, 1))

        return t

    def _get_direction_masks(self):
        """Decide direction uniformly."""
        forward_mask = tf.cast(
            tf.random.uniform((self.batch_size,)) > 0.5,
            dtype=TF_FLOAT,
        )
        backward_mask = 1. - forward_mask

        return forward_mask, backward_mask

    @staticmethod
    def _get_accept_masks(accept_prob):
        """Create binary array to pick out which idxs are accepted."""
        accept_mask = tf.cast(
            accept_prob > tf.random.uniform(tf.shape(accept_prob)),
            dtype=TF_FLOAT,
        )
        reject_mask = 1. - accept_mask

        return accept_mask, reject_mask

    def _build_masks(self, mask_type=None):
        """Construct different binary masks for different time steps."""
        masks = []
        if mask_type == 'checkerboard':
            odds = np.arange(self.xdim) % 2.
            evens = 1. - odds

        for i in range(self.config.num_steps):
            # Need to use numpy.random here because tensorflow would
            # generate different random values across different calls.
            if mask_type == 'checkerboard':
                mask = evens if (i % 2 == 0) else odds
            else:  # use randomly generated masks
                _idx = np.arange(self.xdim)
                idx = np.random.permutation(_idx)[:self.xdim // 2]
                mask = np.zeros((self.xdim,))
                mask[idx] = 1.

            mask = tf.constant(mask, dtype=TF_FLOAT)
            masks.append(mask[None, :])

        return masks

    def _get_mask(self, i):
        """Retrieve the binary mask for the i-th leapfrog step."""
        if tf.executing_eagerly():
            m = self.masks[int(i)]
        else:
            m = tf.gather(self.masks, tf.cast(i, dtype=TF_INT))

        return m, 1. - m

    def _build_networks(self):
        """Must implement method for building the network."""
        raise NotImplementedError

    @staticmethod
    def _build_hmc_networks():
        # pylint:disable=unused-argument
        xnets = lambda inputs, is_training: [  # noqa: E731
            tf.zeros_like(inputs[0]) for _ in range(3)
        ]
        vnets = lambda inputs, is_training: [  # noqa: E731
            tf.zeros_like(inputs[0]) for _ in range(3)
        ]

        return xnets, vnets

    def _get_network(self, step):
        return self.xnets, self.vnets

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

        return tf.Variable(initial_value=init,
                           name='eps', dtype=TF_FLOAT,
                           trainable=self.config.eps_trainable)

    def _create_lr(self, lr_config=None):
        """Create the learning rate schedule to be used during training."""
        if lr_config is None:
            lr_config = self.lr_config

        if lr_config.warmup_steps > 0:
            return WarmupExponentialDecay(lr_config, staircase=True,
                                          name='WarmupExponentialDecay')

        return tf.keras.optimizers.schedules.ExponentialDecay(
            lr_config.init,
            decay_steps=lr_config.decay_steps,
            decay_rate=lr_config.decay_rate,
            staircase=True,
        )

    def _create_optimizer(self):
        """Create the optimizer to be used for backpropagating gradients."""
        if tf.executing_eagerly():
            return tf.keras.optimizers.Adam(self.lr)

        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        if self.using_hvd:
            optimizer = hvd.DistributedOptimizer(optimizer)

        return optimizer
