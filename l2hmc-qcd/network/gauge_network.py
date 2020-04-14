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

import pickle

import tensorflow as tf

from .layers import (DenseLayerNP, relu, ScaledTanhLayer, ScaledTanhLayerNP,
                     StackedLayer, StackedLayerNP)
from .network_utils import custom_dense
from config import Weights
from network import QCOEFF, QNAME, SCOEFF, SNAME, TNAME


class GaugeNetwork(tf.keras.Model):
    """GaugeNetwork. Implements stacked Cartesian repr. of `GenericNet`."""
    def __init__(self,
                 x_dim=None,
                 factor=None,
                 net_seeds=None,
                 num_hidden1=None,
                 num_hidden2=None,
                 input_shape=None,
                 name='GaugeNetwork',
                 dropout_prob=0.,
                 activation=tf.nn.relu):
        """Initialization method.

        Args:
            x_dim (int): Dimensionality of target distribution.
            factor (float): Scaling factor introduced into weight
                initialization.
            net_seeds (dict): Dictionary of seeds defined layer-wise.
            num_hidden1 (int): Number of units in first hidden layers.
            num_hidden2 (int): Number of units in second hidden layers.
            input_shape (tuple): Expected input shape (w/o batch_dim).
            name (str): Name of network.
            dropout_prob (float): Dropout probability. If = 0, no dropout.
            activation (callable): Activation function to use.
        """
        super(GaugeNetwork, self).__init__(name=name)
        self.x_dim = x_dim
        self.factor = factor
        self.dropout_prob = dropout_prob
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.activation = activation
        self._input_shape = input_shape
        with tf.name_scope(name):
            if dropout_prob > 0.:
                self.dropout = tf.keras.layers.Dropout(
                    dropout_prob, seed=net_seeds['dropout']
                )

            self.x_layer = StackedLayer(name='x_layer',
                                        factor=factor/3.,
                                        units=num_hidden1,
                                        seed=net_seeds['x_layer'],
                                        input_shape=(2 * x_dim,))

            self.v_layer = StackedLayer(name='v_layer',
                                        factor=1./3.,
                                        units=num_hidden1,
                                        seed=net_seeds['v_layer'],
                                        input_shape=(2 * x_dim,))

            self.t_layer = custom_dense(name='t_layer',
                                        factor=1./3.,
                                        units=num_hidden1,
                                        seed=net_seeds['t_layer'],
                                        input_shape=(2 * x_dim,))

            def _dense_layer(n):
                return custom_dense(name=f'h_layer{n}',
                                    factor=1.,
                                    units=num_hidden2,
                                    seed=int(n * net_seeds['h_layer']))

            self.hidden_layers = [_dense_layer(n) for n in range(5)]

            #  self.h_layer1 = custom_dense(name='h_layer1',
            #                               factor=1.,
            #                               units=num_hidden2,
            #                               seed=net_seeds['h_layer'])
            #
            #  self.h_layer2 = custom_dense(name='h_layer2',
            #                               factor=1.,
            #                               units=num_hidden2,
            #                               seed=int(2 * net_seeds['h_layer']))
            #
            #  self.h_layer = custom_dense(name='h_layer',
            #                              factor=1.,
            #                              units=num_hidden2,
            #                              seed=int(3 * net_seeds['h_layer']))

            self.translation_layer = custom_dense(name=TNAME,
                                                  factor=0.001,
                                                  units=x_dim,
                                                  seed=net_seeds[TNAME])

            self.scale_layer = ScaledTanhLayer(name='scale',
                                               factor=0.001,
                                               units=x_dim,
                                               seed=net_seeds[SNAME])

            self.transformation_layer = ScaledTanhLayer(name='transformation',
                                                        factor=0.001,
                                                        units=x_dim,
                                                        seed=net_seeds[QNAME])

        self.layers_dict = {
            'x_layer': self.x_layer.layer,
            'v_layer': self.v_layer.layer,
            't_layer': self.t_layer,
            #  'h_layer1': self.h_layer1,
            #  'h_layer2': self.h_layer2,
            #  'h_layer': self.h_layer,
            'scale_layer': self.scale_layer.layer,
            'translation_layer': self.translation_layer,
            'transformation_layer': self.transformation_layer.layer,
        }

        for idx, hidden_layer in enumerate(self.hidden_layers):
            self.layers_dict[f'h_layer{idx}'] = hidden_layer

    def get_weights(self, sess):
        """Get dictionary of layer weights."""
        def _weights(layer):
            w, b = sess.run(layer.weights)
            return Weights(w=w, b=b)

        weights_dict = {
            'x_layer': _weights(self.x_layer.layer),
            'v_layer': _weights(self.v_layer.layer),
            't_layer': _weights(self.t_layer),
            #  'h_layer1': _weights(self.h_layer1),
            #  'h_layer2': _weights(self.h_layer2),
            #  'h_layer': _weights(self.h_layer),
            'scale_layer': _weights(self.scale_layer.layer),
            'translation_layer': _weights(self.translation_layer),
            'transformation_layer': _weights(self.transformation_layer.layer),
        }

        for idx, layer in enumerate(self.hidden_layers):
            weights_dict[f'h_layer{idx}'] = _weights(layer)

        coeffs = sess.run([self.scale_layer.coeff,
                           self.transformation_layer.coeff])

        weights_dict[SCOEFF] = coeffs[0]
        weights_dict[QCOEFF] = coeffs[1]

        return weights_dict

    def save_weights(self, sess, out_file):
        """Save all layer weights to `out_file`."""
        weights_dict = self.get_weights(sess)
        with open(out_file, 'wb') as f:
            pickle.dump(weights_dict, f)

        #  fpath, ext = out_file.split('.')
        #  types_file = f'{fpath}_types.{ext}'
        #  with open(types_file, 'wb') as f:
        #      pickle.dump(self.layers_types, f)

    def call(self, inputs, train_phase):
        """Call the network (forward-pass)."""
        v, x, t = inputs

        x = self.x_layer(x)
        v = self.v_layer(v)
        t = self.t_layer(t)

        h = self.activation(x + v + t)
        for layer in self.hidden_layers:
            h = self.activation(layer(h))

        #  h = self.activation(x + v + t)
        #  h = self.activation(self.h_layer(h))
        #  h = self.activation(self.h_layer1(h))
        #  h = self.activation(self.h_layer2(h))

        if self.dropout_prob > 0:
            h = self.dropout(h, training=train_phase)

        scale = self.scale_layer(h)
        translation = self.translation_layer(h)
        transformation = self.transformation_layer(h)

        return scale, translation, transformation


class GaugeNetworkNP:
    """Implements numpy version of `GaugeNetwork`."""
    def __init__(self, weights, activation=relu):
        self.activation = activation
        self.x_layer = StackedLayerNP(weights['x_layer'])
        self.v_layer = StackedLayerNP(weights['v_layer'])
        self.t_layer = DenseLayerNP(weights['t_layer'])

        def _dense_layer(n):
            return DenseLayerNP(weights[f'h_layer{n}'])
        #
        self.hidden_layers = [_dense_layer(i) for i in range(5)]
        #  self.h_layer = DenseLayerNP(weights['h_layer'])

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
        #  h = self.activation(self.h_layer(h))

        scale = self.scale_layer(h)
        translation = self.translation_layer(h)
        transformation = self.transformation_layer(h)

        return scale, translation, transformation
