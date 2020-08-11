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
import time

from typing import Callable, Union, List

import numpy as np
import tensorflow as tf

import utils.file_io as io

from config import (BIN_DIR, DynamicsConfig, lrConfig, MonteCarloStates,
                    NetWeights, NetworkConfig, State, TF_FLOAT, TF_INT)
from utils.attr_dict import AttrDict
from utils.learning_rate import WarmupExponentialDecay

# pylint:disable=unused-import
from utils.file_io import timeit  # noqa:E401
from utils.seed_dict import vnet_seeds  # noqa:E401
from utils.seed_dict import xnet_seeds  # noqa:E401

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


TIMING_FILE = os.path.join(BIN_DIR, 'timing_file.log')


def identity(x):
    """Returns x"""
    return x


def timed(fn):
    def wrap(*args, **kwargs):
        """Function to be timed."""
        start = time.time()
        result = fn(*args, **kwargs)
        stop = time.time()

        return result, stop - start
    return wrap


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
        self._model_type = config.get('model_type', 'BaseDynamics')

        #  self.params = params
        self.mask_type = params.get('mask_type', 'rand')
        self.config = config
        self.net_config = network_config
        self.potential_fn = potential_fn
        self.normalizer = normalizer if normalizer is not None else identity

        self.params = self._parse_params(params)
        self.eps = self._build_eps(use_log=False)
        self.masks = self._build_masks(self.mask_type)
        #  self._construct_time()
        if self.config.hmc:
            self.xnets, self.vnets = self._build_hmc_networks()
            self.net_weights = NetWeights(*(6 * [0.]))
        else:
            self._build_networks()

        if self._has_trainable_params:
            self.lr_config = lr_config
            self.lr = self._create_lr(lr_config)
            self.optimizer = self._create_optimizer()

    def save_config(self, config_dir):
        """Helper method for saving configuration objects."""
        io.save_dict(self.config, config_dir, name='dynamics_config')
        io.save_dict(self.net_config, config_dir, name='network_config')
        io.save_dict(self.lr_config, config_dir, name='lr_config')
        io.save_dict(self.params, config_dir, name='dynamics_params')

    def _parse_params(self, params, net_weights=None):
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

        if net_weights is None:
            if self.config.hmc:
                net_weights = NetWeights(*(6 * [0.]))
            else:
                net_weights = NetWeights(*(6 * [1.]))

        self.net_weights = net_weights
        self._xsw = self.net_weights.x_scale
        self._xtw = self.net_weights.x_translation
        self._xqw = self.net_weights.x_transformation
        self._vsw = self.net_weights.v_scale
        self._vtw = self.net_weights.v_translation
        self._vqw = self.net_weights.v_transformation

        params = AttrDict({
            'xdim': self.xdim,
            'batch_size': self.batch_size,
            'using_hvd': self.using_hvd,
            'x_shape': self.x_shape,
            'clip_val': self.clip_val,
            'net_weights': self.net_weights,
        })

        return params

    def call(self, inputs, training=None, mask=None):
        """Calls the model on new inputs.

        In this case `call` just reapplies all ops in the graph to the new
        inputs (e.g. build a new computational graph from the provided inputs).

        NOTE: Custom implementation calls `self.apply_transition` and returns a
        list of the states (init, proposed, out), accept probability, and the
        sumlogdet states (init, proposed, out):
            [mc_states: MonteCarloStates,
             accept_prob: tf.Tensor,
             sld_states: MonteCarloStates]

        Arguments:
            inputs: A tensor or list of tensors.
            training: Boolean or boolean scalar tensor, indicating whether to
                run the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be either a tensor or
                None (no mask).

        Returns:
            A tensor if there is a single output, or a list of tensors if there
            are more than one outputs.
        """
        return self.apply_transition(inputs, training=training)

    def calc_losses(self, states: MonteCarloStates, accept_prob: tf.Tensor):
        """Calculate the total loss."""
        raise NotImplementedError

    @staticmethod
    def calc_esjd(x: tf.Tensor,
                  y: tf.Tensor,
                  accept_prob: tf.Tensor):
        """Calculate the expected squared jump distance, ESJD."""
        return accept_prob * tf.reduce_sum((x - y) ** 2, axis=1) + 1e-4

    def _mixed_loss(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            accept_prob: tf.Tensor,
            scale: tf.Tensor = tf.constant(1., dtype=TF_FLOAT)
    ):
        """Compute the mixed loss as: scale / esjd - esjd / scale."""
        esjd = self.calc_esjd(x, y, accept_prob)
        loss = tf.reduce_mean(scale / esjd) - tf.reduce_mean(esjd / scale)

        return loss

    def train_step(self, data):
        """Perform a single training step."""
        raise NotImplementedError

    def test_step(self, data):
        """Perform a single inference step."""
        raise NotImplementedError

    def _random_direction(
            self,
            inputs: Union[tf.Tensor, List[tf.Tensor]],
            training: bool = None
    ) -> (MonteCarloStates, tf.Tensor, MonteCarloStates):
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

    def apply_transition(
            self,
            inputs,
            training: bool = None,
    ) -> (MonteCarloStates, tf.Tensor, MonteCarloStates):
        """Propose a new state and perform the accept/reject step.

        NOTE: We simulate the dynamics both forward and backward, and use
        sampled Bernoulli masks to compute the actual solutions
        """
        x, beta = inputs
        sf_init, sf_prop, pxf, sldf = self._transition(inputs, forward=True,
                                                       training=training)
        sb_init, sb_prop, pxb, sldb = self._transition(inputs, forward=False,
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

        # ma_: accept_mask, mr_: reject mask
        ma_, mr_ = self._get_accept_masks(accept_prob)
        ma = ma_[:, None]
        mr = mr_[:, None]

        # Construct the output configuration
        v_out = ma * v_prop + mr * v_init
        x_out = self.normalizer(ma * x_prop + mr * x)
        sumlogdet = ma_ * sld_prop  # NOTE: initial sumlogdet = 0

        state_init = State(x=x, v=v_init, beta=beta)
        state_prop = State(x=x_prop, v=v_prop, beta=beta)
        state_out = State(x=x_out, v=v_out, beta=beta)

        mc_states = MonteCarloStates(state_init, state_prop, state_out)
        sld_states = MonteCarloStates(0., sld_prop, sumlogdet)

        return mc_states, accept_prob, sld_states

    def md_update(
            self,
            inputs: Union[tf.Tensor, List[tf.Tensor]],
            training: bool = None
    ) -> (MonteCarloStates, MonteCarloStates):
        """Perform the molecular dynamics (MD) update w/o accept/reject.

        NOTE: We simulate the dynamics both forward and backward, and use
        sampled Bernoulli masks to compute the actual solutions
        """
        x, beta = inputs
        sf_init, sf_prop, _, sldf = self._transition(inputs, forward=True,
                                                     training=training)
        sb_init, sb_prop, _, sldb = self._transition(inputs, forward=False,
                                                     training=training)
        # Decide direction uniformly
        mf_, mb_ = self._get_direction_masks()
        mf = mf_[:, None]
        mb = mb_[:, None]

        v_init = mf * sf_init.v + mb * sb_init.v
        x_prop = mf * sf_prop.x + mb * sb_prop.x
        v_prop = mf * sf_prop.v + mb * sb_prop.v
        sld_prop = mf_ * sldf + mb_ * sldb
        x_prop = self.normalizer(x_prop)

        state_init = State(x=x, v=v_init, beta=beta)
        state_prop = State(x=x_prop, v=v_prop, beta=beta)
        mc_states = MonteCarloStates(init=state_init,
                                     proposed=state_prop,
                                     out=state_prop)
        sld_states = MonteCarloStates(init=0.,
                                      proposed=sld_prop,
                                      out=sld_prop)

        return mc_states, sld_states

    def _transition(
            self,
            inputs: Union[tf.Tensor, List[tf.Tensor]],
            forward: bool,
            training: bool = None
    ) -> (State, State, tf.Tensor, State):
        """Run the augmented leapfrog integrator."""
        x, beta = inputs
        v = tf.random.normal(tf.shape(x))
        state = State(x=x, v=v, beta=beta)
        state_, px, sld = self.transition_kernel(state, forward, training)

        return state, state_, px, sld

    def test_reversibility(
            self,
            data: Union[tf.Tensor, List[tf.Tensor]],
            training: bool = None
    ):
        """Test reversibility.

        NOTE:
         1. Run forward then backward
                 (x, v) -> (xf, vf)
                 (xf, vf) -> (xb, vb)
            check that x == xb, v == vb

         2. Run backward then forward
                 (x, v) -> (xb, vb)
                 (xb, vb) -> (xf, vf)
            check that x == xf, v == vf
        """
        dxf, dvf = self._test_reversibility(data, forward_first=True,
                                            training=training)
        dxb, dvb = self._test_reversibility(data, forward_first=False,
                                            training=training)
        output = AttrDict({
            'dxf': dxf,
            'dvf': dvf,
            'dxb': dxb,
            'dvb': dvb,
        })

        return output

    def _test_reversibility(
            self,
            data: Union[tf.Tensor, List[tf.Tensor]],
            forward_first: bool,
            training: bool = None
    ):
        """Helper method for `self.test_reversibility`.

        NOTE:
         - If `forward_first`, test (1.) (forward then backward).
         - Else, test (2.) (backward then forward).
        """
        x, beta = data
        v = tf.random.normal(tf.shape(x))

        #  Run from (x, v) -> (x1, v1)
        state1, _, _ = self.transition_kernel(State(x, v, beta),
                                              forward_first, training)
        # Run from (x1, v1) --> (x2, v2)
        state2, _, _ = self.transition_kernel(state1,
                                              not forward_first, training)
        # If reversible, then x2 == x, v2 == v
        dx = x - state2.x
        dv = v - state2.v

        return dx, dv

    def test_transition_kernels(self, x, beta, forward, training=None):
        """Test for difference between while and for loop."""
        v = tf.random.normal(tf.shape(x))
        state = State(x=x, v=v, beta=beta)

        def _timeit(state, forward, training, fn):
            start = time.time()
            state_, px_, sld_ = fn(state, forward, training)
            dt = time.time() - start
            return state_, px_, sld_, dt

        def _sum_mean_diff(x, y, name):
            dxy = x - y
            return {f'{name}_sum': tf.reduce_sum(dxy),
                    f'{name}_avg': tf.reduce_mean(dxy)}

        state_w, px_w, sld_w, dt_w = _timeit(state, forward, training,
                                             self.transition_kernel_while)
        state_f, px_f, sld_f, dt_f = _timeit(state, forward, training,
                                             self.transition_kernel)
        out = {
            'dt_w': dt_w,
            'dt_f': dt_f,
        }

        names = ['dpx', 'dsld', 'dx', 'dv']
        vars_w = [px_w, sld_w, state_w.x, state_w.v]
        vars_f = [px_f, sld_f, state_f.x, state_f.v]
        for idx, (vw, vf) in enumerate(zip(vars_w, vars_f)):
            d = _sum_mean_diff(vw, vf, names[idx])
            out.update(d)

        return AttrDict(out)

    def transition_kernel_while(self, state, forward, training=None):
        """Transition kernel using a `tf.while_loop` implementation."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        step = tf.constant(0, dtype=tf.int64)
        sld = tf.zeros((self.batch_size,), dtype=state.x.dtype)

        def body(step, state, sld):
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

    def transition_kernel(
            self,
            state: State,
            forward: bool,
            training: bool = None
    ):
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
        t = self._get_time(step, tile=tf.shape(state.x)[0])

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
        m, mc = self._get_mask(step_r)
        xnet, vnet = self._get_network(step_r)
        t = self._get_time(step_r, tile=tf.shape(state.x)[0])

        #  sumlogdet = 0.
        sumlogdet = tf.constant(0., dtype=state.x.dtype)

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

        vf = state.v * expS - 0.5 * self.eps * (grad * expQ - transl)

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

        vb = expS * (state.v + 0.5 * self.eps * (grad * expQ - transl))

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

    def _get_time_bad(self, i):
        """Format the MCMC step as [cos(...), sin(...)]."""
        return tf.gather(self.ts, i)

    def _get_time(self, i, tile=1):
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

    # pylint:disable=unused-argument
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
            init = tf.math.log(tf.constant(self.config.eps, dtype=TF_FLOAT))
        else:
            init = tf.constant(self.config.eps, dtype=TF_FLOAT)

        return tf.Variable(initial_value=init,
                           name='eps', dtype=TF_FLOAT,
                           trainable=self.config.eps_trainable)

    def _create_lr(self, lr_config=None):
        """Create the learning rate schedule to be used during training."""
        if lr_config is None:
            lr_config = self.lr_config

        warmup_steps = lr_config.get('warmup_steps', None)
        if warmup_steps is not None and warmup_steps > 0:
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
