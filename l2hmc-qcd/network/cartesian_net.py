"""
cartesian_net.py

Implements a feed-forward neural network that operates on the cartesian
represenrepresentation of an angular variable phi in U(1).

Author: Sam Foreman
Date: 04/02/2020
"""
# pylint:disable=invalid-name
from __future__ import absolute_import, division, print_function

import pickle

import numpy as np
import tensorflow as tf

from .layers import (CartesianLayer, CartesianLayerNP, DenseLayerNP, relu,
                     ScaledTanhLayer, ScaledTanhLayerNP)
from .network_utils import custom_dense
from config import NP_FLOAT, TF_FLOAT, TF_INT, Weights
from seed_dict import seeds, vnet_seeds, xnet_seeds

SNAME = 'scale_layer'
SCOEFF = 'coeff_scale'
TNAME = 'translation_layer'
QNAME = 'transformation_layer'
QCOEFF = 'coeff_transformation'


class CartesianNet(tf.keras.Model):
    """CartesianNet. Implements cartesian representation of `GenericNet`."""
    def __init__(self,
                 x_dim=None,
                 factor=None,
                 net_seeds=None,
                 num_hidden1=None,
                 num_hidden2=None,
                 input_shape=None,
                 name='CartesianNet',
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
        super(CartesianNet, self).__init__(name=name)

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

            self.x_layer = CartesianLayer(name='x_layer',
                                          factor=factor/3.,
                                          units=num_hidden1,
                                          seed=net_seeds['x_layer'],
                                          input_shape=(x_dim,))

            self.v_layer = CartesianLayer(name='v_layer',
                                          factor=1./3.,
                                          units=num_hidden1,
                                          seed=net_seeds['v_layer'],
                                          input_shape=(x_dim,))

            self.t_layer = custom_dense(name='t_layer',
                                        factor=1./3.,
                                        units=self.num_hidden1,
                                        seed=net_seeds['t_layer'])

            self.h_layer = custom_dense(name='h_layer',
                                        factor=1.,
                                        units=self.num_hidden2,
                                        seed=net_seeds['h_layer'])

            self.translation_layer = custom_dense(name=TNAME,
                                                  factor=0.001,
                                                  units=self.x_dim,
                                                  seed=net_seeds[TNAME])

            self.scale_layer = ScaledTanhLayer(name='scale',
                                               factor=0.001,
                                               units=self.x_dim,
                                               seed=net_seeds[SNAME])

            self.transformation_layer = ScaledTanhLayer(name='transformation',
                                                        factor=0.001,
                                                        units=self.x_dim,
                                                        seed=net_seeds[QNAME])
        self.layers_types = {
            'x_layer': 'CartesianLayer',
            'v_layer': 'CartesianLayer',
            't_layer': 'custom_dense',
            'h_layer': 'custom_dense',
            'translation_layer': 'custom_dense',
            'scale_layer': 'ScaledTanhLayer',
            'transformation_layer': 'ScaledTanhLayer',
        }

        self.layers_dict = {
            'x_layer': {
                'x_layer': self.x_layer.x_layer,
                'y_layer': self.x_layer.y_layer,
            },
            'v_layer': {
                'x_layer': self.v_layer.x_layer,
                'y_layer': self.v_layer.y_layer,
            },
            't_layer': self.t_layer,
            'h_layer': self.h_layer,
            'scale_layer': self.scale_layer.layer,
            'translation_layer': self.translation_layer,
            'transformation_layer': self.transformation_layer.layer,
        }

    def get_weights(self, sess):
        """Extract numerical values of all layer weights."""
        def _weights(layer):
            w, b = sess.run(layer.weights)
            return Weights(w=w, b=b)

        weights_dict = {
            'x_layer': {
                'x_layer': _weights(self.x_layer.x_layer),
                'y_layer': _weights(self.x_layer.y_layer),
            },
            'v_layer': {
                'x_layer': _weights(self.v_layer.x_layer),
                'y_layer': _weights(self.v_layer.y_layer),
            },
            't_layer': _weights(self.t_layer),
            'h_layer': _weights(self.h_layer),
            'scale_layer': _weights(self.scale_layer.layer),
            'translation_layer': _weights(self.translation_layer),
            'transformation_layer': _weights(self.transformation_layer.layer),
        }

        coeffs = sess.run([self.scale_layer.coeff,
                           self.transformation_layer.coeff])

        weights_dict[SCOEFF] = coeffs[0]
        weights_dict[QCOEFF] = coeffs[1]

        return weights_dict

    def save_weights(self, sess, out_file):
        """Save all layers weights to `out_file`."""
        weights_dict = self.get_weights(sess)
        with open(out_file, 'wb') as f:
            pickle.dump(weights_dict, f)

        fpath, ext = out_file.split('.')
        types_file = f'{fpath}_types.{ext}'
        with open(types_file, 'wb') as f:
            pickle.dump(self.layers_types, f)

    # pylint:disable=invalid-name
    def call(self, inputs, train_phase):
        v, x, t = inputs

        xx, xy = self.x_layer(tf.cos(x), tf.sin(x))
        vx, vy = self.v_layer(tf.cos(v), tf.sin(v))

        t_out = self.t_layer(t)
        #  tx = tf.cos(t_out)
        #  ty = tf.sin(t_out)

        x_sum = xx + vx + t_out
        y_sum = xy + vy + t_out
        phi = tf.mod(tf.atan2(y_sum, x_sum), 2 * np.pi)

        h = self.activation(self.h_layer(phi))
        #  h = self.activation(tf.mod(self.h_layer(phi), 2 * np.pi))
        scale = self.scale_layer(h)
        translation = self.translation_layer(h)
        transformation = self.transformation_layer(h)

        return scale, translation, transformation


class CartesianNetNP:
    def __init__(self, weights, activation=relu):
        self.activation = activation
        self.x_layer = CartesianLayerNP(weights['x_layer'])
        self.v_layer = CartesianLayerNP(weights['v_layer'])
        self.t_layer = DenseLayerNP(weights['t_layer'])
        self.h_layer = DenseLayerNP(weights['h_layer'])
        self.translation_layer = DenseLayerNP(weights[TNAME])

        self.scale_layer = ScaledTanhLayerNP(weights[SCOEFF],
                                             weights[SNAME])

        self.transformation_layer = ScaledTanhLayerNP(weights[QCOEFF],
                                                      weights[QNAME])
    def __call__(self, inputs):
        v, x, t = inputs
        xx, xy = self.x_layer(np.cos(x), np.sin(x))
        vx, vy = self.v_layer(np.cos(v), np.sin(v))

        t_out = self.t_layer(t)
        x_sum = xx + vx + t_out
        y_sum = xy + vy + t_out
        phi = np.mod(np.arctan2(y_sum, x_sum), 2 * np.pi)
        h = self.activation(self.h_layer(phi))
        scale = self.scale_layer(h)
        translation = self.translation_layer(h)
        transformation = self.transformation_layer(h)

        return scale, translation, transformation
