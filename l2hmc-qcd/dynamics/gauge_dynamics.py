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

from typing import Callable, List, NoReturn, Tuple
from collections import namedtuple

import numpy as np
import tensorflow as tf

from config import (DynamicsConfig, NetworkConfig, NP_FLOAT, PI, TF_FLOAT,
                    TF_INT, TWO_PI, lrConfig)
from dynamics.dynamics import BaseDynamics
from utils.attr_dict import AttrDict
from utils.seed_dict import seeds, vnet_seeds, xnet_seeds
from utils.learning_rate import WarmupExponentialDecay
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

    def train_step(self, inputs, first_step=False):
        """Perform a single training step."""
        start = time.time()
        with tf.GradientTape() as tape:
            states, px, sld = self(inputs, training=True)
            ploss, qloss = self.calc_losses((states, px))
            loss = ploss + qloss

        if self.using_hvd:
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
