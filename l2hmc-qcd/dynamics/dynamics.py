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


from typing import Callable, NoReturn

import numpy as np
import tensorflow as tf

from config import (DynamicsConfig, MonteCarloStates, NET_WEIGHTS_HMC,
                    NET_WEIGHTS_L2HMC, NetworkConfig, State, TWO_PI,
                    TF_FLOAT, TF_INT, lrConfig)
from network.gauge_network import GaugeNetwork
from utils.attr_dict import AttrDict
from utils.seed_dict import vnet_seeds, xnet_seeds
from utils.learning_rate import WarmupExponentialDecay

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


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
    ) -> NoReturn:
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
        self.config = config
        self.net_config = network_config
        self.potential_fn = potential_fn
        self.normalizer = normalizer if normalizer is not None else identity

        self._parse_params(params)
        self.eps = self._build_eps(use_log=False)
        self.masks = self._build_masks(random=False)
        self._build_networks()

        if self._has_trainable_params:
            self.lr_config = lr_config
            self.lr = self._create_lr(lr_config)
            self.optimizer = self._create_optimizer()

    def compile(self, optimizer, loss=None):
        """Compile the `tf.keras.models.Model` object."""
        super(BaseDynamics, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def _parse_params(self, params):
        """Set instance attributes from `params`."""
        self.xdim = params.get('xdim', None)
        self.batch_size = params.get('batch_size', None)
        self.using_hvd = params.get('horovod', False)
        self.x_shape = (self.batch_size, self.xdim)

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

    def train_step(self, inputs, first_step):
        """Perform a single training step."""
        raise NotImplementedError

    def test_step(self, inputs):
        """Perform a single inference step."""
        raise NotImplementedError

    def apply_transition(self, inputs, training=None):
        """Propose a new state and perform the accept/reject step.

        NOTE: We simulate the molecular dynamics update in both the forward and
        backward directions, and use sampled masks to compute the actual
        solutions.
        """
        x_init, beta = inputs
        with tf.name_scope('forward'):
            sf_init, sf_prop, pxf, sldf = self._transition(inputs,
                                                           forward=True,
                                                           training=training)
        with tf.name_scope('backward'):
            sb_init, sb_prop, pxb, sldb = self._transition(inputs,
                                                           forward=False,
                                                           training=training)

        with tf.name_scope('proposed'):
            mf, mb = self._get_direction_masks()
            mf_tensor = mf[:, None]
            mb_tensor = mb[:, None]
            v_init = sf_init.v * mf_tensor + sb_init.v * mb_tensor
            x_prop = sf_prop.x * mf_tensor + sb_prop.x * mb_tensor
            v_prop = sf_prop.v * mf_tensor + sb_prop.v * mb_tensor
            sld_prop = sldf * mf + sldb * mb

        with tf.name_scope('outputs'):
            with tf.name_scope('accept_prob'):
                accept_prob = pxf * mf + pxb * mb

            ma, mr = self._get_accept_masks(accept_prob)
            ma_tensor = ma[:, None]
            mr_tensor = mr[:, None]

            x_out = x_prop * ma_tensor + x_init * mr_tensor
            v_out = v_prop * ma_tensor + v_init * mr_tensor
            sumlogdet = sld_prop * ma

            state_init = State(x_init, v_init, beta)
            state_prop = State(x_prop, v_prop, beta)
            state_out = State(x_out, v_out, beta)
            mc_states = MonteCarloStates(state_init, state_prop, state_out)
            sld_states = MonteCarloStates(0., sld_prop, sumlogdet)

        return mc_states, accept_prob, sld_states

    def _transition(self, inputs, forward, training=None):
        """Run the augmented leapfrog integrator."""
        x_init, beta = inputs
        v_init = tf.random.normal(tf.shape(x_init))
        state = State(x_init, v_init, beta)
        #  state_prop, px, sld = self.transition_kernel_while(state,
        #                                                     forward,
        #                                                     training)
        state_prop, px, sld = self.transition_kernel_for(state,
                                                         forward,
                                                         training)

        return state, state_prop, px, sld

    def transition_kernel_while(self, state, forward, training=None):
        """Transition kernel using a `tf.while_loop` implementation."""
        lf_fn = self._forward_lf if forward else self._backward_lf
        step = tf.constant(0., dtype=TF_FLOAT)
        sld = tf.zeros((self.batch_size,), dtype=state.x.dtype)

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

        state_prop = State(*state)
        sumlogdet = tf.zeros((self.batch_size,), dtype=TF_FLOAT)
        for step in range(self.config.num_steps):
            state_prop, logdet = lf_fn(step, state_prop, training)
            sumlogdet += logdet

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)

        return state_prop, accept_prob, sumlogdet

    def compute_accept_prob(self,
                            state_init: State,
                            state_prop: State,
                            sumlogdet: tf.Tensor) -> tf.Tensor:
        """Compute the acceptance prob. of `state_prop` given `state_init`."""
        h_init = self.hamiltonian(state_init)
        h_prop = self.hamiltonian(state_prop)
        dh = h_init - h_prop + sumlogdet
        prob = tf.exp(tf.minimum(dh, 0.))

        return tf.where(tf.math.is_finite(prob), prob, tf.zeros_like(prob))

    def _forward_lf(self, step, state, training=None):
        """Run the augmented leapfrog integrator in the forward direction."""
        t = self._get_time(step, tile=tf.shape(state.x)[0])
        m, mc = self._get_mask(step)  # pylint: disable=invalid-name
        xnet, vnet = self._get_network(step)
        sumlogdet = 0.

        state, logdet = self._update_v_forward(vnet, state, t, training)
        sumlogdet += logdet

        state, logdet = self._update_x_forward(xnet, state, t,
                                               (m, mc), training)
        sumlogdet += logdet

        state, logdet = self._update_x_forward(xnet, state, t,
                                               (mc, m), training)
        sumlogdet += logdet

        state, logdet = self._update_v_forward(vnet, state, t, training)

        return state, sumlogdet

    def _backward_lf(self, step, state, training=None):
        """Run the augmented leapfrog integrator in the backward direction."""
        step_r = self.config.num_steps - step - 1
        t = self._get_time(step_r)
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

        state_out = State(x, vf, state.beta)
        logdet = tf.reduce_sum(scale, axis=-1)

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

        state_out = State(xf, state.v, state.beta)
        logdet = tf.reduce_sum(mc * scale, axis=-1)

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

        state_out = State(x, vb, state.beta)
        logdet = tf.reduce_sum(scale, axis=-1)

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

        state_out = State(xb, state.v, state.beta)
        logdet = tf.reduce_sum(mc * scale, axis=-1)

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
            return 0.5 * tf.reduce_sum(v ** 2, axis=-1)

    def hamiltonian(self, state):
        """Compute the overall Hamiltonian."""
        with tf.name_scope('hamiltonian'):
            kinetic = self.kinetic_energy(state.v)
            potential = self.potential_energy(state.x, state.beta)

        return potential + kinetic

    def _get_time(self, i, tile=1):
        """Format the MCMC step as [cos(...), sin(...)]."""
        i = tf.cast(i, dtype=TF_FLOAT)
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
        """Create binary array to pick out which idxs are accepted."""
        accept_mask = tf.cast(
            accept_prob > tf.random.uniform(tf.shape(accept_prob)),
            dtype=TF_FLOAT,
        )
        reject_mask = 1. - accept_mask

        return accept_mask, reject_mask

    def _build_masks(self, random=False):
        """Construct different binary masks for different time steps."""
        with tf.name_scope('build_masks'):
            masks = []
            if not random:
                odds = np.arange(self.xdim) % 2.
                evens = 1. - odds

            for i in range(self.config.num_steps):
                # Need to use numpy.random here because tensorflow would
                # generate different random values across different calls.
                if random:
                    _idx = np.arange(self.xdim)
                    idx = np.random.permutation(_idx)[:self.xdim // 2]
                    mask = np.zeros((self.xdim,))
                    mask[idx] = 1.
                else:
                    mask = evens if (i % 2 == 0) else odds

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

    def _build_networks(self):
        if self.config.hmc:
            self.xnets = lambda inputs, is_training: [
                tf.zeros_like(inputs[0]) for _ in range(3)
            ]
            self.vnets = lambda inputs, is_training: [
                tf.zeros_like(inputs[0]) for _ in range(3)
            ]
        else:
            if self.config.separate_networks:
                def _new_net(name, seeds_):
                    self.xnets = []
                    self.vnets = []
                    factor = 2. if name == 'XNet' else 1.
                    for idx in range(self.config.num_steps):
                        new_seeds = {
                            key: int(idx * val) for key, val in seeds_.items()
                        }
                        net = GaugeNetwork(self.net_config,
                                           self.xdim, factor=factor,
                                           net_seeds=new_seeds,
                                           name=f'{name}_step{idx}')
                        if name == 'XNet':
                            setattr(self, f'xnets{idx}', net)
                            self.xnets.append(net)
                        elif name == 'VNet':
                            setattr(self, f'vnets{idx}', net)
                            self.vnets.append(net)

                _new_net('XNet', xnet_seeds)
                _new_net('VNet', vnet_seeds)

            else:
                self.xnets = GaugeNetwork(self.net_config,
                                          xdim=self.xdim, factor=2.,
                                          net_seeds=xnet_seeds, name='XNet')
                self.vnets = GaugeNetwork(self.net_config,
                                          xdim=self.xdim, factor=1.,
                                          net_seeds=vnet_seeds, name='VNet')

    def _get_network(self, step):
        if self.config.hmc or not self.config.separate_networks:
            return self.xnets, self.vnets

        step_int = int(step)
        xnet = getattr(self, f'xnets{step_int}', None)
        vnet = getattr(self, f'vnets{step_int}', None)

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

        return tf.Variable(initial_value=init, dtype=TF_FLOAT,
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
