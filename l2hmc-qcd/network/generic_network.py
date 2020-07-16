"""
generic_network.py

Implements a generic feed-forward neural network for augmenting the leapfrog
integrator.
"""
from __future__ import absolute_import, division, print_function

import typing

from collections import namedtuple

import tensorflow as tf

from config import QNAME, SNAME, TNAME
from network.layers import dense_layer, ScaledTanhLayer

NetworkConfig = namedtuple('NetworkConfig', [
    'type', 'units', 'dropout_prob', 'activation_fn'
])


# pylint:disable=too-many-instance-attributes, too-few-public-methods
class GenericNetwork(tf.keras.layers.Layer):
    """Implements a generic feed forward neural network."""

    # pylint:disable=too-many-arguments
    def __init__(self,
                 config: NetworkConfig,
                 xdim: int,
                 factor: float = 1.,
                 net_seeds: dict = None,
                 zero_init: bool = False,
                 name: str = 'GaugeNetwork') -> typing.NoReturn:
        """Initialization method.

        Args:
            config (NetworkConfig): Configuration specifying various network
                properties
            xdim (int): Dimensionality of target space (features dim.).
            factor (float): Scaling factor used in weight initialization of
                `custom_dense` layers.
            net_seeds (dict): Dictionary of random(int) seeds for
                reproducibility.
            name (str): Name of network.
        """
        super(GenericNetwork, self).__init__(name=name)

        self.xdim = xdim
        self.factor = factor
        self._config = config
        self.activation = config.activation_fn

        with tf.name_scope(name):
            if config.dropout_prob > 0:
                self.dropout = tf.keras.layers.Dropout(config.dropout_prob)

            self.x_layer = dense_layer(name='x_layer', factor=factor/3.,
                                       units=config.units[0],
                                       input_shape=(2 * xdim,),
                                       zero_init=zero_init,
                                       seed=net_seeds['x_layer'])
            self.v_layer = dense_layer(name='v_layer',
                                       factor=1./3,
                                       units=config.units[0],
                                       zero_init=zero_init,
                                       input_shape=(1 * xdim,),
                                       seed=net_seeds['v_layer'])
            self.t_layer = dense_layer(name='t_layer',
                                       factor=1/3.,
                                       units=config.units[0],
                                       zero_init=zero_init,
                                       input_shape=(2 * xdim,),
                                       seed=net_seeds['t_layer'])

            def make_hlayer(i, units):
                return dense_layer(factor=1.,
                                   units=units,
                                   zero_init=zero_init,
                                   name=f'h_layer{i}',
                                   seed=int(i * net_seeds['h_layer']))

            self.hidden_layers = [
                make_hlayer(j, k) for j, k in enumerate(config.units[1:])
            ]

            self.scale_layer = ScaledTanhLayer(units=xdim,
                                               factor=0.001,
                                               name='scale',
                                               zero_init=zero_init,
                                               seed=net_seeds[SNAME])
            self.translation_layer = dense_layer(units=xdim,
                                                 factor=0.001,
                                                 name='translation',
                                                 zero_init=zero_init,
                                                 seed=net_seeds[TNAME])
            self.transformation_layer = ScaledTanhLayer(units=xdim,
                                                        factor=0.001,
                                                        name='transformation',
                                                        zero_init=zero_init,
                                                        seed=net_seeds[QNAME])

    # pylint:disable=invalid-name
    def call(self, inputs, training=None):
        """Call the network (forward-pass)."""
        v, x, t = inputs
        h = self.v_layer(v) + self.x_layer(x) + self.t_layer(t)
        h = self.activation(h)
        for layer in self.hidden_layers:
            h = self.activation(layer(h))

        if self._config.dropout_prob > 0 and training:
            h = self.dropout(h, training=training)

        scale = self.scale_layer(h)
        translation = self.translation_layer(h)
        transformation = self.transformation_layer(h)

        return scale, translation, transformation
