"""
gauge_dynamics.py

Implements the GaugeDynamics class by subclassing the `BaseDynamics` class.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Reference [Robust Parameter Estimation with a Neural Network Enhanced
Hamiltonian Markov Chain Monte Carlo Sampler]
https://infoscience.epfl.ch/record/264887/files/robust_parameter_estimation.pdf

Author: Sam Foreman (github: @saforem2)
Date: 7/3/2020
"""
from __future__ import absolute_import, division, print_function

import time

from typing import NoReturn

import numpy as np
import tensorflow as tf

from config import (DynamicsConfig, lrConfig, NetWeights, NetworkConfig, PI,
                    State, TWO_PI)
from utils.attr_dict import AttrDict
from dynamics.dynamics import BaseDynamics
from lattice.utils import u1_plaq_exact_tf
from lattice.lattice import GaugeLattice

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


def convert_to_angle(x):
    """Returns x in -pi <= x < pi."""
    x = tf.math.floormod(x + PI, TWO_PI) - PI
    return x


# pylint:disable=attribute-defined-outside-init
# pylint:disable=too-many-instance-attributes,unused-argument
# pylint:disable=invalid-name,too-many-locals,too-many-arguments
class GaugeDynamics(BaseDynamics):
    """Implements the dynamics engine for the L2HMC sampler."""
    def __init__(
            self,
            params: AttrDict,
            config: DynamicsConfig,
            network_config: NetworkConfig,
            lr_config: lrConfig,
    ) -> NoReturn:

        self.plaq_weight = params.get('plaq_weight', 10.)
        self.charge_weight = params.get('charge_weight', 0.1)

        self.lattice_shape = params.get('lattice_shape', None)
        self.lattice = GaugeLattice(self.lattice_shape)

        params.update({
            'batch_size': self.lattice_shape[0],
            'xdim': np.cumprod(self.lattice_shape[1:])[-1],
        })

        super(GaugeDynamics, self).__init__(
            params=params,
            config=config,
            name='GaugeDynamics',
            normalizer=convert_to_angle,
            network_config=network_config,
            lr_config=lr_config,
            potential_fn=self.lattice.calc_actions,
        )

        if self.config.use_ncp and not self.config.hmc:
            self.net_weights = NetWeights(1., 1., 1., 1., 1., 1.)
            self._xsw = self.net_weights.x_scale
            self._xtw = self.net_weights.x_translation
            self._xqw = self.net_weights.x_transformation
            self._vsw = self.net_weights.v_scale
            self._vtw = self.net_weights.v_translation
            self._vqw = self.net_weights.v_transformation

    def calc_losses(self, inputs):
        """Calculate the total loss."""
        states, accept_prob = inputs
        dtype = states.init.x.dtype

        ps_init = self.lattice.calc_plaq_sums(samples=states.init.x)
        ps_prop = self.lattice.calc_plaq_sums(samples=states.proposed.x)

        # Calculate the plaquette loss
        ploss = tf.constant(0., dtype=dtype)
        if self.plaq_weight > 0:
            dplaq = 2 * (1. - tf.math.cos(ps_prop - ps_init))
            ploss = accept_prob * tf.reduce_sum(dplaq, axis=(1, 2))
            ploss = tf.reduce_mean(-ploss / self.plaq_weight, axis=0)

        # Calculate the charge loss
        qloss = tf.constant(0., dtype=dtype)
        if self.charge_weight > 0:
            q_init = self.lattice.calc_top_charges(plaq_sums=ps_init,
                                                   use_sin=True)
            q_prop = self.lattice.calc_top_charges(plaq_sums=ps_prop,
                                                   use_sin=True)

            qloss = accept_prob * (q_prop - q_init) ** 2
            qloss = tf.reduce_mean(-qloss / self.charge_weight)

        return ploss, qloss

    def train_step(self, x, beta, first_step):
        """Perform a single training step."""
        start = time.time()
        with tf.GradientTape() as tape:
            tape.watch(x)
            tape.watch(beta)
            states, px, sld = self((x, beta), training=True)
            ploss, qloss = self.calc_losses((states, px))
            loss = ploss + qloss

        if self.using_hvd:
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables),
        )

        metrics = AttrDict({
            'dt': time.time() - start,
            'loss': loss,
            'ploss': ploss,
            'qloss': qloss,
            'accept_prob': px,
            'eps': self.eps,
            'beta': states.init.beta,
            'sumlogdet': sld.out,
        })

        observables = self.calc_observables(states, use_sin=True)
        metrics.update(**observables)

        # Horovod:
        #    Broadcast initial variable states from rank 0 to all other
        #    processes. This is necessary to ensure consistent initialization
        #    of all workers when training is started with random weights or
        #    restored from a checkpoint.
        # NOTE:
        #    Broadcast should be done after the first gradient step to ensure
        #    optimizer intialization.
        #  if self.optimizer.iterations.numpy() == 0 and self.using_hvd:
        if first_step:
            hvd.broadcast_variables(self.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        return states.out.x, metrics

    def test_step(self, inputs):
        """Perform a single inference step."""
        start = time.time()
        states, px, sld = self(inputs, training=False)
        ploss, qloss = self.calc_losses((states, px))
        loss = ploss + qloss

        metrics = AttrDict({
            'dt': time.time() - start,
            'loss': loss,
            'ploss': ploss,
            'qloss': qloss,
            'accept_prob': px,
            'eps': self.eps,
            'beta': states.init.beta,
            'sumlogdet': sld.out,
        })

        observables = self.calc_observables(states, use_sin=False)
        metrics.update(**observables)

        return states.out.x, metrics

    def _calc_observables(self, state, use_sin=False):
        """Calculate the observables for a particular state.

        NOTE: We track the error in the plaquette instead of the actual value.
        """
        x = tf.reshape(state.x, self.lattice_shape)
        ps = self.lattice.calc_plaq_sums(samples=x)
        plaqs = self.lattice.calc_plaqs(plaq_sums=ps)
        plaqs_err = u1_plaq_exact_tf(state.beta) - plaqs
        charges = self.lattice.calc_top_charges(
            plaq_sums=ps, use_sin=use_sin
        )

        return plaqs_err, charges

    def calc_observables(self, states, use_sin=False):
        """Calculate observables."""
        _, q_init = self._calc_observables(states.init, use_sin=use_sin)
        plaqs, q_out = self._calc_observables(states.out, use_sin=use_sin)

        observables = AttrDict({
            'dq': tf.math.abs(q_out - q_init),
            'charges': q_out,
            'plaqs': plaqs,
        })

        return observables

    def _update_x_foward(self, network, state, t, masks, training):
        """Update the position `x` in the forward leapfrog step."""
        if not self.config.use_ncp:
            return super()._update_x_forward(network, state,
                                             t, masks, training)
        m, mc = masks

        # map x from [-pi, pi] to [-inf, inf]
        x = tf.math.tan(state.x / 2.)

        S, T, Q = network((state.v, m * x, t), training)

        transl = self._xtw * T
        transf = self._xqw * (self.eps * Q)
        scale = self._xsw * (self.eps * S)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        y = x * expS + self.eps * (state.v * expQ + transl)
        xf = (m * x) + (mc * y)

        # map xf from [-inf, inf] back to [-pi, pi]
        xf = 2. * tf.math.atan(xf)
        state_out = State(xf, state.v, state.beta)
        logdet = tf.reduce_sum(mc * scale, axis=-1)

        return state_out, logdet

    def _update_x_backward(self, network, state, t, masks, training):
        """Update the position `x` in the backward leapfrog step."""
        if not self.config.use_ncp:
            return super()._update_x_backward(network, state,
                                              t, masks, training)
        m, mc = masks
        # map x from [-pi, pi] to [-inf, inf]
        x = tf.math.tan(state.x / 2.)

        S, T, Q = network((state.v, m * x, t), training)

        transl = self._xtw * T
        transf = self._xqw * (self.eps * Q)
        scale = self._xsw * (-self.eps * S)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        y = expS * (x - self.eps * (state.v * expQ + transl))
        xb = (m * x) + (mc * y)

        # map xb from [-inf, inf] back to [-pi, pi]
        xb = 2. * tf.math.atan(xb)
        state_out = State(xb, state.v, state.beta)
        logdet = tf.reduce_sum(mc * scale, axis=-1)

        return state_out, logdet
