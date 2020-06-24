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
import time

from config import NP_FLOAT, TF_FLOAT, TF_INT
from utils.seed_dict import seeds, vnet_seeds, xnet_seeds
from utils.attr_dict import AttrDict
from utils.learning_rate import WarmupExponentialDecay
from network import NetworkConfig
from lattice.lattice import GaugeLattice
from lattice.utils import u1_plaq_exact_tf
from network.gauge_network import GaugeNetwork

try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


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


def convert_to_angle(x):
    """Returns x in -pi <= x < pi."""
    x = tf.math.floormod(x + PI, TWO_PI) - PI
    return x


def exp_mult_cooling(step, temp_init, temp_final, num_steps, alpha=None):
    """Annealing function."""
    if alpha is None:
        alpha = tf.exp(
            (tf.math.log(temp_final) - tf.math.log(temp_init)) / num_steps
        )

    temp = temp_init * (alpha ** step)

    return tf.cast(temp, TF_FLOAT)


def get_betas(steps, beta_init, beta_final):
    """Get array of betas to use in annealing schedule."""
    t_init = 1. / beta_init
    t_final = 1. / beta_final
    t_arr = [
        exp_mult_cooling(i, t_init, t_final, steps) for i in range(steps)
    ]

    return 1. / tf.convert_to_tensor(np.array(t_arr))


# pylint:disable=too-many-instance-attributes,unused-argument
class Dynamics(tf.keras.Model):
    """DynamicsObject for training the L2HMC sampler."""

    def __init__(
            self,
            params: AttrDict,
            dynamics_config: DynamicsConfig,
            network_config: NetworkConfig,
            potential_fn: Callable[[tf.Tensor], tf.Tensor] = None,
    ) -> NoReturn:
        """Initialization method."""
        super(Dynamics, self).__init__(name='Dynamics')
        np.random.seed(seeds['global_np'])

        self.params = params
        self.config = dynamics_config
        self.net_config = network_config
        #  self._potential_fn = potential_fn
        self._model_type = self.config.model_type

        self.parse_params(params)
        self.lattice = GaugeLattice(self.lattice_shape)
        if potential_fn is None:
            self.potential_fn = self.lattice.calc_actions
        else:
            self.potential_fn = potential_fn

        self.eps = self._build_eps(use_log=False)
        self.masks = self._build_masks(random=False)

        self._build_networks()

        if self._has_trainable_params:
            self.lr = self._create_lr(warmup=self.warmup_lr)
            self.optimizer = self._create_optimizer()

    # pylint:disable=attribute-defined-outside-init
    def compile(self, optimizer):
        """Compile the Model."""
        super(Dynamics, self).compile()
        self.optimizer = optimizer

    # pylint:disable=attribute-defined-outside-init
    def parse_params(self, params):
        """Set instance attributes from `params`."""
        self.params = AttrDict(params)

        self.net_weights = self.config.net_weights
        self._xsw = self.net_weights.x_scale
        self._xtw = self.net_weights.x_translation
        self._xqw = self.net_weights.x_transformation
        self._vsw = self.net_weights.v_scale
        self._vtw = self.net_weights.v_translation
        self._vqw = self.net_weights.v_transformation

        self.separate_networks = params.get('separate_networks', False)

        lattice_shape = params.get('lattice_shape', None)
        if lattice_shape is not None:
            batch_size, time_size, space_size, dim = lattice_shape
        else:
            batch_size = params.get('batch_size', None)
            time_size = params.get('time_size', None)
            space_size = params.get('space_size', None)
            dim = params.get('dim', 2)
            lattice_shape = (batch_size, time_size, space_size, dim)

        self.batch_size = batch_size
        self.lattice_shape = lattice_shape
        self.xdim = time_size * space_size * dim
        self.x_shape = (batch_size, self.xdim)

        self.plaq_weight = params.get('plaq_weight', 0.)
        self.charge_weight = params.get('charge_weight', 0.)

        self.print_steps = params.get('print_steps', 10)
        self.run_steps = params.get('run_steps', int(1e3))
        self.logging_steps = params.get('logging_steps', 50)
        self.save_run_data = params.get('save_run_data', True)
        self.save_train_data = True
        self.save_steps = params.get('save_steps', None)

        self._should_compile = True
        eager_execution = params.get('eager_execution', False)
        if tf.executing_eagerly() or eager_execution:
            self._should_compile = False

            # Determine if there are any parameters to be trained
        self._has_trainable_params = True
        if self.config.hmc and not self.config.eps_trainable:
            self._has_trainable_params = False

        # If there exist parameters to be optimized, setup optimizer
        if self._has_trainable_params:
            self.lr_init = params.get('lr_init', None)
            self.warmup_lr = params.get('warmup_lr', False)
            self.warmup_steps = params.get('warmup_steps', None)
            self.using_hvd = params.get('horovod', False)
            self.lr_decay_steps = params.get('lr_decay_steps', None)
            self.lr_decay_rate = params.get('lr_decay_rate', None)

            self.train_steps = params.get('train_steps', None)
            self.beta_init = params.get('beta_init', None)
            self.beta_final = params.get('beta_final', None)
            beta = params.get('beta', None)
            if self.beta_init == self.beta_final or beta is None:
                self.beta = self.beta_init
                self.betas = tf.convert_to_tensor(
                    tf.cast(self.beta * np.ones(self.train_steps),
                            dtype=TF_FLOAT)
                )
            else:
                if self.train_steps is not None:
                    self.betas = get_betas(self.train_steps,
                                           self.beta_init,
                                           self.beta_final)

    def call(self, inputs, training=None):
        """Obtain a new state from `inputs`."""
        return self.apply_transition(inputs, training=None)

    def calc_loss(self, inputs):
        """Calculate the total loss."""
        # Unpack the inputs
        x_init, x_prop, accept_prob = inputs
        ps_init = self.lattice.calc_plaq_sums(samples=x_init)
        ps_prop = self.lattice.calc_plaq_sums(samples=x_prop)
        q_init = self.lattice.calc_top_charges(plaq_sums=ps_init)

        plaq_loss = tf.constant(0., dtype=TF_FLOAT, name='plaq_loss')
        if self.plaq_weight > 0:
            dplaq = 2 * (1. - tf.math.cos(ps_prop - ps_init))
            ploss = accept_prob * tf.reduce_sum(dplaq, axis=(1, 2))
            plaq_loss = tf.reduce_mean(-ploss / self.plaq_weight, axis=0)

        charge_loss = tf.constant(0., dtype=TF_FLOAT, name='charge_loss')
        if self.charge_weight > 0:
            q_prop = self.lattice.calc_top_charges(plaq_sums=ps_prop)
            qloss = accept_prob * (q_prop - q_init) ** 2
            charge_loss = tf.reduce_mean(-qloss / self.charge_weight, axis=0)

        total_loss = plaq_loss + charge_loss

        return total_loss, q_init

    def train_step(self, x, beta):
        """Perform a single training step."""
        global_step = self.optimizer.iterations
        if tf.executing_eagerly():
            first_step = (global_step.numpy() == 0)
        else:
            first_step = (global_step == 0)

        t0 = time.time()
        with tf.GradientTape() as tape:
            states, px, sld_states = self((x, beta), training=True)
            inputs = (states.init.x, states.proposed.x, px)
            loss, q_old = self.calc_loss(inputs)

        dt = time.time() - t0
        if self.using_hvd:
            # Horovod: add Horovod Distributed GradientTape
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables)
        )
        # Horovod:
        #   Broadcast initial variable states from rank 0 to all other
        #   processes. This is necessary to ensure consistent initialization of
        #   all workers when training is started with random weights or
        #   restored from a checkpoint.
        # NOTE:
        #   Broadcast should be done after the first gradient step to ensure
        #   optimizer initialization.
        if first_step and self.using_hvd:
            hvd.broadcast_variables(self.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        plaqs, q_new = self.calc_observables(
            tf.reshape(states.out.x, self.lattice_shape),
            beta
        )

        metrics = AttrDict({
            'dt': dt,
            'beta': beta,
            'loss': loss,
            'plaqs': plaqs,
            'eps': self.eps,
            'charges': q_new,
            'accept_prob': px,
            'sumlogdet': sld_states.out,
            'dq': tf.math.abs(q_new - q_old),
        })

        return states.out.x, metrics

    def test_step(self, x, beta):
        """Perform a single inference step."""
        t0 = time.time()
        states, px, sld_states = self((x, beta), training=False)
        inputs = (states.init.x, states.proposed.x, px)
        loss, q_old = self.calc_loss(inputs)
        dt = time.time() - t0
        plaqs, q_new = self.calc_observables(
            tf.reshape(states.out.x, self.lattice_shape),
            beta
        )

        metrics = AttrDict({
            'dt': dt,
            'beta': beta,
            'loss': loss,
            'plaqs': plaqs,
            'eps': self.eps,
            'charges': q_new,
            'accept_prob': px,
            'sumlogdet': sld_states.out,
            'dq': tf.math.abs(q_new - q_old),
        })

        return states.out.x, metrics

    def calc_observables(self, x, beta, use_sin=True):
        """Calculate observabless."""
        ps = self.lattice.calc_plaq_sums(x)
        plaqs = self.lattice.calc_plaqs(plaq_sums=ps)
        charges = self.lattice.calc_top_charges(plaq_sums=ps, use_sin=use_sin)
        plaqs_err = u1_plaq_exact_tf(beta) - plaqs

        return plaqs_err, charges

    def apply_transition(self, inputs, training=None):
        """Propose a new state and perform the accept reject step.

        We simulate the molecular dynamics update both forward and backward,
        and use sampled masks to compute the actual solutions.
        """
        x_init, beta = inputs
        with tf.name_scope('transition_forward'):
            sf_init, sf_prop, pxf, sldf = self._transition(inputs,
                                                           forward=True,
                                                           training=training)
        with tf.name_scope('transition_backward'):
            sb_init, sb_prop, pxb, sldb = self._transition(inputs,
                                                           forward=False,
                                                           training=training)
        with tf.name_scope('combined_fb'):
            mask_f, mask_b = self._get_direction_masks()
            v_init = sf_init.v * mask_f[:, None] + sb_init.v * mask_b[:, None]
            x_prop = sf_prop.x * mask_f[:, None] + sb_prop.x * mask_b[:, None]
            v_prop = sf_prop.v * mask_f[:, None] + sb_prop.v * mask_b[:, None]
            sld_prop = sldf * mask_f + sldb * mask_b  # sumlogdet proposed

            state_init = State(x_init, v_init, beta)
            state_prop = State(x_prop, v_prop, beta)

        with tf.name_scope('outputs'):
            with tf.name_scope('accept_prob'):
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
        #  with tf.name_scope('transition_kernel'):
        state_prop, px, sld = self.transition_kernel(state,
                                                     forward,
                                                     training)

        return state, state_prop, px, sld

    def transition_kernel_while_loop(self, state, forward, training=None):
        """Transition kernel using a `tf.while_loop` implementation."""
        lf_fn = self._forward_lf if forward else self._backward_lf
        step = 0.
        step = tf.constant(0., dtype=TF_FLOAT)
        sld = tf.zeros((self.batch_size,), dtype=TF_FLOAT)

        def body(step, state, logdet):
            new_state, logdet = lf_fn(step, state, training=training)
            return step+1, new_state, sld+logdet

        def cond(step, *args):
            return tf.less(step, self.config.num_steps)

        _, state_prop, sumlogdet = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[step, state, sld]
        )

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)

        return state_prop, accept_prob, sumlogdet

    def transition_kernel(self, state, forward, training=None):
        """Transition kernel of the augmented leapfrog integrator."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        state_prop = State(*state)
        sumlogdet = tf.zeros((self.batch_size,), dtype=TF_FLOAT)
        for step in range(self.config.num_steps):
            state_prop, logdet = lf_fn(step, state_prop, training)
            sumlogdet += logdet

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)

        return state_prop, accept_prob, sumlogdet

    def compute_accept_prob(self, state_init, state_prop, sumlogdet):
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

    def _forward_lf(self, step, state, training=None):
        #  with tf.name_scope('forward_lf'):
        t = self._get_time(step, tile=tf.shape(state.x)[0])
        m, mc = self._get_mask(step)
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
        sumlogdet += logdet

        return state, sumlogdet

    def _backward_lf(self, step, state, training=None):
        #  with tf.name_scope('backward_lf'):
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
        Sv, Tv, Qv = network((x, grad, t), training)

        scale = self._vsw * (0.5 * self.eps * Sv)
        transl = self._vtw * Tv
        transf = self._vqw * (self.eps * Qv)

        exp_s = tf.exp(scale)
        exp_q = tf.exp(transf)

        vf = state.v * exp_s - 0.5 * self.eps * (grad * exp_q + transl)
        logdet = tf.reduce_sum(scale, axis=-1)

        state_out = State(x, vf, state.beta)

        return state_out, logdet

    def _update_x_forward(self, network, state, t, masks, training):
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
        Sx, Tx, Qx = network((state.v, m * x, t), training)

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
        if self._model_type == 'GaugeModel':
            x = convert_to_angle(state.x)

        grad = self.grad_potential(x, state.beta)
        #  Sv, Tv, Qv = self.vnet((x, grad, t), training)
        Sv, Tv, Qv = network((x, grad, t), training)

        scale = self._vsw * (-0.5 * self.eps * Sv)
        transl = self._vtw * Tv
        transf = self._vqw * (self.eps * Qv)

        exp_s = tf.exp(scale)
        exp_q = tf.exp(transf)

        vb = exp_s * (state.v + 0.5 * self.eps * (grad * exp_q + transl))

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
        x = state.x
        if self._model_type == 'GaugeModel':
            x = convert_to_angle(state.x)

        m, mc = masks
        #  Sx, Tx, Qx = self.xnet((state.v, m * x, t), training)
        Sx, Tx, Qx = network((state.v, m * x, t), training)
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

    def grad_potential(self, x, beta):
        """Get the gradient of the potential function at current `x`."""
        with tf.name_scope('grad_potential'):
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
        with tf.name_scope('potential_energy'):
            if self._model_type == 'GaugeModel':
                x = convert_to_angle(x)

            potential_energy = beta * self.potential_fn(x)

        return potential_energy

    @staticmethod
    def kinetic_energy(v):
        """Compute the kinetic energy of the momentum as 0.5 * (v ** 2)."""
        with tf.name_scope('kinetic_energy'):
            return 0.5 * tf.reduce_sum(v**2, axis=-1)

    def hamiltonian(self, state):
        """Compute the overall hamiltonian."""
        with tf.name_scope('hamiltonian'):
            kinetic = self.kinetic_energy(state.v)
            potential = self.potential_energy(state.x, state.beta)

        return potential + kinetic

    def _create_lr(self, warmup=False):
        """Create the learning rate schedule to be used during training."""
        if warmup:
            name = 'WarmupExponentialDecay'
            warmup_steps = getattr(self, 'warmup_steps', None)
            if warmup_steps is None:
                warmup_steps = self.train_steps // 10

            return WarmupExponentialDecay(self.lr_init, self.lr_decay_steps,
                                          self.lr_decay_rate, warmup_steps,
                                          staircase=True, name=name)

        return tf.keras.optimizers.schedules.ExponentialDecay(
            self.lr_init,
            decay_steps=self.lr_decay_steps,
            decay_rate=self.lr_decay_rate,
            staircase=True,
        )

    def _create_optimizer(self):
        """Create the optimizer to use for backpropagation."""
        if tf.executing_eagerly():
            return tf.keras.optimizers.Adam(self.lr)

        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        if self.using_hvd:
            optimizer = hvd.DistributedOptimizer(optimizer)

        return optimizer

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

    # pylint:disable=attribute-defined-outside-init
    def _build_networks(self):
        if self.config.hmc:
            self.xnets = lambda inputs, is_training: [
                tf.zeros_like(inputs[0]) for _ in range(3)
            ]
            self.vnets = lambda inputs, is_training: [
                tf.zeros_like(inputs[0]) for _ in range(3)
            ]

        else:
            if self.separate_networks:
                def _separate_nets(name, seeds_):
                    nets = []
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

                    return nets

                _separate_nets('XNet', xnet_seeds)
                _separate_nets('VNet', vnet_seeds)

            else:
                self.xnets = GaugeNetwork(self.net_config,
                                          xdim=self.xdim, factor=2.,
                                          net_seeds=xnet_seeds, name='XNet')
                self.vnets = GaugeNetwork(self.net_config,
                                          xdim=self.xdim, factor=1.,
                                          net_seeds=vnet_seeds, name='VNet')

    def _get_network(self, step):
        if self.config.hmc or not self.separate_networks:
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
