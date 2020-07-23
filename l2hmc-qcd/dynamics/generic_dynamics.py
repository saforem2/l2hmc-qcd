"""
generic_dynamics.py

Implements the `GenericDynamics` object by subclassing the `BaseDynamics`
class.

Author: Sam Foreman
Date: 7/15/2020
"""
from __future__ import absolute_import, division, print_function

import time
import tensorflow as tf

from config import (BIN_DIR, DynamicsConfig, lrConfig, NetworkConfig,
                    NetWeights, PI, TF_FLOAT, MonteCarloStates)
from dynamics.base_dynamics import BaseDynamics
from network.generic_network import GenericNetwork
from utils.attr_dict import AttrDict
try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


def identity(x):
    """Returns whatever is passed."""
    return x


class GenericDynamics(BaseDynamics):
    """Implements a generic `Dynamics` object, defined by `potential_fn`."""

    def __init__(self,
                 params: AttrDict,
                 config: DynamicsConfig,
                 network_config: NetworkConfig,
                 lr_config: lrConfig,
                 potential_fn: callable,
                 normalizer: callable = identity,
                 name: str = 'GenericDynamics'):
        super(GenericDynamics, self).__init__(
            name=name,
            params=params,
            config=config,
            lr_config=lr_config,
            normalizer=normalizer,
            potential_fn=potential_fn,
            network_config=network_config,
        )

        if not self.config.hmc:
            self.net_weights = NetWeights(1., 1., 1., 1., 1., 1.)
            self._xsw = self.net_weights.x_scale
            self._xtw = self.net_weights.x_translation
            self._xqw = self.net_weights.x_transformation
            self._vsw = self.net_weights.v_scale
            self._vtw = self.net_weights.v_translation
            self._vqw = self.net_weights.v_transformation

    def _build_networks(self):
        self.xnets = GenericNetwork(self.net_config, factor=2.,
                                    xdim=self.xdim, name='XNet')
        self.vnets = GenericNetwork(self.net_config, factor=1.,
                                    xdim=self.xdim, name='VNet')

    def calc_losses(self, states: MonteCarloStates, accept_prob: tf.Tensor):
        """Calculate the total sampling loss."""
        loss = self._mixed_loss(states.init.x, states.proposed.x, accept_prob)

        return loss

    def train_step(self,
                   x: tf.Tensor,
                   beta: tf.Tensor,
                   first_step: bool = False):
        """Perform a single training step."""
        start = time.time()
        with tf.GradientTape() as tape:
            states, accept_prob, sumlogdet = self(x, beta, training=True)
            loss = self.calc_losses(states, accept_prob)
            if self.using_hvd:
                tape = hvd.DistributedGradientTape(tape)

            grads = tape.gradient(loss, self.trainable_variables)
            if self.clip_val > 0:
                grads = [tf.clip_by_norm(g, self.clip_val) for g in grads]

            self.optimizer.apply_gradients(
                zip(grads, self.trainable_variables)
            )

            metrics = AttrDict({
                'dt': time.time() - start,
                'loss': loss,
                'accept_prob': accept_prob,
                'eps': self.eps,
                'beta': states.init.beta,
                'sumlogdet': sumlogdet.out,
            })

            return states.out.x, metrics

    def test_step(self, x: tf.Tensor, beta: tf.Tensor):
        """Perform a single inference step."""
        start = time.time()
        states, accept_prob, sumlogdet = self(x, beta, training=False)
        loss = self.calc_losses(states, accept_prob)

        metrics = AttrDict({
            'dt': time.time() - start,
            'loss': loss,
            'accept_prob': accept_prob,
            'eps': self.eps,
            'beta': states.init.beta,
            'sumlogdet': sumlogdet.out,
        })

        return states.out.x, metrics
