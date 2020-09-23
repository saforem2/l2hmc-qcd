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
import horovod.tensorflow as hvd

from config import (DynamicsConfig, LearningRateConfig, NetworkConfig,
                    NetWeights, MonteCarloStates)
from dynamics.base_dynamics import BaseDynamics, DynamicsConfig
from network.generic_network import GenericNetwork
from utils.attr_dict import AttrDict

NUM_RANKS = hvd.size()


def identity(x):
    """Returns whatever is passed."""
    return x


# pylint:disable=too-many-ancestors,too-many-instance-attributes
class GenericDynamics(BaseDynamics):
    """Implements a generic `Dynamics` object, defined by `potential_fn`."""

    # pylint:disable=too-many-arguments
    def __init__(
            self,
            params: AttrDict,
            config: DynamicsConfig,
            network_config: NetworkConfig,
            lr_config: LearningRateConfig,
            potential_fn: callable,
            normalizer: callable = identity,
            name: str = 'GenericDynamics'
    ):
        """Initialization method for generic (Euclidean) Dynamics."""
        super(GenericDynamics, self).__init__(
            name=name,
            params=params,
            config=config,
            lr_config=lr_config,
            normalizer=normalizer,
            potential_fn=potential_fn,
            network_config=network_config,
            should_build=True
        )

        if not self.config.hmc:
            self.net_weights = NetWeights(1., 1., 1., 1., 1., 1.)
            self._xsw = self.net_weights.x_scale
            self._xtw = self.net_weights.x_translation
            self._xqw = self.net_weights.x_transformation
            self._vsw = self.net_weights.v_scale
            self._vtw = self.net_weights.v_translation
            self._vqw = self.net_weights.v_transformation

    def call(self, inputs, training=None):
        return self.apply_transition(inputs, training)

    def _build(self, params, config, network_config, lr_config, **kwargs):
        """Build the model."""
        self.config = config
        self.net_config = network_config
        self.eps = self._build_eps(use_log=False)
        self.masks = self._build_masks()
        if self.config.hmc:
            net_weights = NetWeights(0., 0., 0., 0., 0., 0.)
        else:
            net_weights = NetWeights(1., 1., 1., 1., 1., 1.)
        self.params = self._parse_params(params, net_weights=net_weights)
        self.xnet, self.vnet = self._build_networks()
        if self._has_trainable_params:
            self.lr_config = lr_config
            self.lr = self._create_lr(lr_config)
            self.optimizer = self._create_optimizer()

    def _build_networks(self):
        xnet = GenericNetwork(self.net_config, factor=2.,
                              xdim=self.xdim, name='XNet')
        vnet = GenericNetwork(self.net_config, factor=1.,
                              xdim=self.xdim, name='VNet')
        return xnet, vnet

    def calc_losses(self, states: MonteCarloStates, accept_prob: tf.Tensor):
        """Calculate the total sampling loss."""
        loss = self._mixed_loss(states.init.x, states.proposed.x, accept_prob)
        return loss

    def train_step(self, data):
        """Perform a single training step."""
        x, beta = data
        start = time.time()
        with tf.GradientTape() as tape:
            states, accept_prob, sumlogdet = self((x, beta), training=True)
            loss = self.calc_losses(states, accept_prob)
            #  self.add_metric(self.loss_metric(loss))

            if self.aux_weight > 0:
                z = tf.random.normal(x.shape, dtype=x.dtype)
                states_, accept_prob_, _ = self((z, beta), training=True)
                loss_ = self.calc_losses(states_, accept_prob_)
                loss += loss_

        if NUM_RANKS > 1:
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #  if self.clip_val > 0:
        #      grads = [tf.clip_by_norm(g, self.clip_val) for g in grads]

        #  self.optimizer.apply_gradients(
        #      zip(grads, self.trainable_variables)
        #  )

        metrics = AttrDict({
            'dt': time.time() - start,
            'loss': loss,
            'accept_prob': accept_prob,
            'eps': self.eps,
            'beta': states.init.beta,
            'sumlogdet': sumlogdet.out,
        })

        #  if self.optimizer.iterations == 0:
        #      for key, val in metrics.items():
        #          self.add_metric(
        #          self.metrics[key].update_state(val)
        #
        if self.optimizer.iterations == 0 and NUM_RANKS > 1:
            hvd.broadcast_variables(self.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        return states.out.x, metrics

    def test_step(self, data):
        """Perform a single inference step."""
        x, beta = data
        start = time.time()
        states, accept_prob, sumlogdet = self((x, beta), training=False)
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
