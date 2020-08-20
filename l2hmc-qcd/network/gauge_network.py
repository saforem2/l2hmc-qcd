"""
gauge_network.py

Implements a feed-forward neural network that operates on the stacked Cartesian
representation of an angular variable phi in U(1).

Author: Sam Foreman
Date: 04/11/2020
"""
# pylint: disable=invalid-name, too-many-instance-attributes
# pylint: disable=too-many-arguments, too-few-public-methods
from __future__ import absolute_import, division, print_function

from collections import namedtuple

import numpy as np
import tensorflow as tf

import utils.file_io as io

from config import QCOEFF, QNAME, SCOEFF, SNAME, TNAME, Weights
from .layers import (dense_layer, DenseLayerNP, relu, ScaledTanhLayer,
                     ScaledTanhLayerNP, StackedLayer, StackedLayerNP)

NetworkConfig = namedtuple('NetworkConfig', [
    'type', 'units', 'dropout_prob', 'activation_fn'
])


def _get_layer_weights(layer):
    """Get an individual layers' weights."""
    w, b = layer.weights
    return Weights(w=w.numpy(), b=b.numpy())


def convert_to_image(x):
    """Create image from lattice by doubling the size."""
    y = np.zeros((2 * x.shape[0], 2 * x.shape[1]))
    y[::2, 1::2] = x[:, :, 0]
    y[1::2, ::2] = x[:, :, 1]
    return y


def get_layer_weights(net):
    """Helper method for extracting layer weights."""
    wdict = {
        'x_layer': _get_layer_weights(net.xlayer.layer),
        'v_layer': _get_layer_weights(net.vlayer.layer),
        't_layer': _get_layer_weights(net.t_layer),
        'hidden_layers': [
            _get_layer_weights(i) for i in net.hidden_layers
        ],
        'scale_layer': (
            _get_layer_weights(net.scale_layer.layer)
        ),
        'translation_layer': (
            _get_layer_weights(net.translation_layer)
        ),
        'transformation_layer': (
            _get_layer_weights(net.transformation_layer.layer)
        ),
    }
    coeffs = [
        net.scale_layer.coeff.numpy(),
        net.transformation_layer.coeff.numpy()
    ]
    wdict['coeff_scale'] = coeffs[0]
    wdict['coeff_transformation'] = coeffs[1]

    return wdict


def save_layer_weights(net, out_file):
    """Save all layer weights from `net` to `out_file`."""
    weights_dict = get_layer_weights(net)
    io.savez(weights_dict, out_file, name=net.name)


class GaugeNetwork(tf.keras.layers.Layer):
    """GaugeNetwork. Implements stacked Cartesian repr. of `GenericNet`."""

    def __init__(
            self,
            config: NetworkConfig,
            xdim: int,
            factor: float = 1.,
            net_seeds: dict = None,
            zero_init: bool = False,
            name: str = 'GaugeNetwork'
    ):
        """Initialization method.

        Args:
            config (NetworkConfig): Configuration specifying various network
                properties.
            xdim (int): Dimensionality of target space (features dim.).
            factor (float): Scaling factor used in weight initialization of
                `custom_dense` layers.
            net_seeds (dict): Dictionary of random (int) seeds for
                reproducibility.
            name (str): Name of network.
            **kwargs (keyword arguments): Passed to `tf.keras.Model.__init__`.
        """
        super(GaugeNetwork, self).__init__(name=name)

        self.xdim = xdim
        self.factor = factor
        self._config = config
        self.activation = config.activation_fn
        with tf.name_scope(name):
            if config.dropout_prob > 0:
                self.dropout = tf.keras.layers.Dropout(config.dropout_prob)

            self.scale_coeff = tf.Variable(name='scale_coeff',
                                           trainable=True,
                                           initial_value=tf.zeros((xdim,)))

            self.transf_coeff = tf.Variable(name='transf_coeff',
                                            trainable=True,
                                            initial_value=tf.zeros((xdim,)))

            #  seed=net_seeds['x_layer'])
            self.x_layer = StackedLayer(name='x_layer',
                                        factor=factor/3.,
                                        zero_init=zero_init,
                                        units=config.units[0],
                                        input_shape=(2 * xdim,))

            #  seed=net_seeds['v_layer'])
            self.v_layer = StackedLayer(name='v_layer',
                                        factor=1./3.,
                                        zero_init=zero_init,
                                        units=config.units[0],
                                        input_shape=(2 * xdim,))

            #  seed=net_seeds['t_layer'])
            self.t_layer = dense_layer(name='t_layer',
                                       factor=1./3.,
                                       zero_init=zero_init,
                                       units=config.units[0],
                                       input_shape=(2 * xdim,))

            def make_hlayer(i, units):
                #  seed=int(i * net_seeds['h_layer']))
                return dense_layer(factor=1.,
                                   units=units,
                                   zero_init=zero_init,
                                   name=f'h_layer{i}')

            self.hidden_layers = [
                make_hlayer(i, n) for i, n in enumerate(config.units[1:])
            ]

            #  seed=net_seeds[SNAME],
            self.scale_layer = dense_layer(units=xdim,
                                           factor=0.001,
                                           zero_init=zero_init,
                                           name='scale')

            #  seed=net_seeds[TNAME])
            self.translation_layer = dense_layer(units=xdim,
                                                 factor=0.001,
                                                 zero_init=zero_init,
                                                 name='translation')

            self.transformation_layer = dense_layer(units=xdim,
                                                    factor=0.001,
                                                    zero_init=zero_init,
                                                    name='transformation')
            #  self.scale_layer = ScaledTanhLayer(name='scale',
            #                                     factor=0.001,
            #                                     zero_init=zero_init,
            #                                     units=xdim)

            #  self.transformation_layer = ScaledTanhLayer(name='transformation',
            #                                              zero_init=zero_init,
            #                                              factor=0.001,
            #                                              units=xdim)

        self.layers_dict = {
            'x_layer': self.x_layer.layer,
            'v_layer': self.v_layer.layer,
            't_layer': self.t_layer,
            'hidden_layers': self.hidden_layers,
            'scale_layer': self.scale_layer,
            'translation_layer': self.translation_layer,
            'transformation_layer': self.transformation_layer,
        }

    @staticmethod
    def _get_layer_weights(layer, sess=None):
        # pylint:disable=invalid-name
        if sess is None or tf.executing_eagerly():
            w, b = layer.weights
            return Weights(w=w.numpy(), b=b.numpy())

        w, b = sess.run(layer.weights)
        return Weights(w=w, b=b)

    def get_layer_weights(self, sess=None):
        """Get dictionary of layer weights."""
        weights_dict = {
            'x_layer': self._get_layer_weights(self.x_layer.layer),
            'v_layer': self._get_layer_weights(self.v_layer.layer),
            't_layer': self._get_layer_weights(self.t_layer),
            'hidden_layers': [
                self._get_layer_weights(j) for j in self.hidden_layers
            ],
            'scale_layer': (
                self._get_layer_weights(self.scale_layer)
            ),
            'translation_layer': (
                self._get_layer_weights(self.translation_layer)
            ),
            'transformation_layer': (
                self._get_layer_weights(self.transformation_layer)
            ),
        }

        if sess is None or tf.executing_eagerly:
            coeffs = [self.scale_coeff.numpy(),
                      self.transf_coeff.numpy()]
        else:
            coeffs = sess.run([self.scale_coeff,
                               self.transf_coeff])

        weights_dict[SCOEFF] = coeffs[0]
        weights_dict[QCOEFF] = coeffs[1]

        return weights_dict

    def save_layer_weights(self, sess=None, out_file=None):
        """Save all layer weights to `out_file`."""
        weights_dict = self.get_layer_weights(sess=sess)
        io.savez(weights_dict, out_file, name=self.name)

        return weights_dict

    # pylint:disable=invalid-name
    def call(self, inputs, training=None):
        """Call the network (forward-pass)."""
        v, x, t = inputs
        h = self.activation(
            self.v_layer(v) + self.x_layer(x) + self.t_layer(t)
        )

        for layer in self.hidden_layers:
            h = self.activation(layer(h))

        if self._config.dropout_prob > 0 and training:
            h = self.dropout(h, training=training)

        scale = (tf.exp(self.scale_coeff)
                 * tf.nn.tanh(self.scale_layer(h)))

        translation = self.translation_layer(h)

        transformation = (tf.exp(self.transf_coeff)
                          * tf.nn.tanh(self.transformation_layer(h)))

        return scale, translation, transformation


class GaugeNetworkNP:
    """Implements numpy version of `GaugeNetwork`."""

    def __init__(self, weights, activation=relu):
        self.activation = activation
        self.x_layer = StackedLayerNP(weights['x_layer'])
        self.v_layer = StackedLayerNP(weights['v_layer'])
        self.t_layer = DenseLayerNP(weights['t_layer'])

        self.hidden_layers = [
            DenseLayerNP(w) for w in weights['hidden_layers']
        ]

        self.translation_layer = DenseLayerNP(weights[TNAME])

        self.scale_layer = ScaledTanhLayerNP(weights[SCOEFF],
                                             weights[SNAME])
        self.transformation_layer = ScaledTanhLayerNP(weights[QCOEFF],
                                                      weights[QNAME])

    def __call__(self, inputs):
        v, x, t = inputs
        v = self.v_layer(v)
        x = self.x_layer(x)
        t = self.t_layer(t)
        h = self.activation(v + x + t)

        for layer in self.hidden_layers:
            h = self.activation(layer(h))

        scale = self.scale_layer(h)
        translation = self.translation_layer(h)
        transformation = self.transformation_layer(h)

        return scale, translation, transformation
