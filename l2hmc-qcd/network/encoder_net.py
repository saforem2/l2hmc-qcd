"""
pair_network.py

Implements new neural network architecture for dealing with (cos, sin) repr.

Author: Sam Foreman
Date: 03/23/2020
"""
from __future__ import absolute_import, division, print_function

import pickle

import tensorflow as tf

import config as cfg

from .layers import (DenseLayerNP, EncodingLayer, EncodingLayerNP, linear,
                     relu, ScaledTanhLayer, ScaledTanhLayerNP)
from .network_utils import custom_dense, tf_zeros
from seed_dict import seeds, vnet_seeds, xnet_seeds

TF_FLOAT = cfg.TF_FLOAT
TF_INT = cfg.TF_INT
Weights = cfg.Weights

SNAME = 'scale_layer'
SCOEFF = 'coeff_scale'
TNAME = 'translation_layer'
QNAME = 'transformation_layer'
QCOEFF = 'coeff_transformation'


# pylint:disable=too-many-arguments, too-many-instance-attributes
class EncoderNet(tf.keras.Model):
    """PairedGenericNet."""
    def __init__(self,
                 x_dim=None,
                 factor=None,
                 net_seeds=None,
                 num_hidden1=None,
                 num_hidden2=None,
                 input_shape=None,
                 name='EncoderNet',
                 dropout_prob=0.,
                 activation=tf.nn.relu):
        """Initialization method.

        Args:
            model_name (str): Name of the model.
            kwargs (dict): Keyword argument used to specify network properties.
        """
        super(EncoderNet, self).__init__(name=name)

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

            self.x_layer = EncodingLayer(name='x_layer',
                                         factor=self.factor/3.,
                                         units=self.num_hidden1,
                                         seed=net_seeds['x_layer'],
                                         input_shape=(x_dim,))

            self.v_layer = EncodingLayer(name='v_layer',
                                         factor=1./3.,
                                         units=self.num_hidden1,
                                         seed=net_seeds['v_layer'],
                                         input_shape=(x_dim,))

            self.t_layer = custom_dense(name='t_layer',
                                        factor=1./3.,
                                        units=self.num_hidden1,
                                        seed=net_seeds['t_layer'],
                                        input_shape=(2,))

            self.h_layer = custom_dense(name='h_layer',
                                        factor=1.,
                                        units=self.num_hidden2,
                                        seed=net_seeds['h_layer'])

            self.translation_layer = custom_dense(name=TNAME,
                                                  factor=0.001,
                                                  units=self.x_dim,
                                                  seed=net_seeds[TNAME])

            self.scale_layer = ScaledTanhLayer(name='scale', factor=0.001,
                                               units=self.x_dim,
                                               seed=net_seeds[SNAME])

            self.transformation_layer = ScaledTanhLayer(name='transformation',
                                                        factor=0.001,
                                                        units=self.x_dim,
                                                        seed=net_seeds[QNAME])

            if self.dropout_prob > 0.:
                self.dropout = tf.keras.layers.Dropout(
                    self.dropout_prob, seed=net_seeds['dropout'],
                )

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

            #  sname = 'scale_layer'
            #  self.scale_layer = custom_dense(name=sname,
            #                                  factor=0.001,
            #                                  units=self.x_dim,
            #                                  seed=net_seeds[sname])
            #
            #  qname = 'transformation_layer'
            #  self.transformation_layer = custom_dense(name=qname,
            #                                           factor=0.001,
            #                                           units=self.x_dim,
            #                                           seed=net_seeds[qname])
            #
            #  tname = 'translation_layer'
            #  self.translation_layer = custom_dense(name=tname,
            #                                        factor=0.001,
            #                                        units=self.x_dim,
            #                                        seed=net_seeds[tname])
            #
            #  self.coeff_scale = tf.Variable(
            #      name='coeff_scale',
            #      trainable=True,
            #      dtype=TF_FLOAT,
            #      initial_value=tf.zeros([1, self.x_dim], dtype=TF_FLOAT),
            #  )
            #
            #  self.coeff_transformation = tf.Variable(
            #      name='coeff_transformation',
            #      trainable=True,
            #      dtype=TF_FLOAT,
            #      initial_value=tf.zeros([1, self.x_dim], dtype=TF_FLOAT),
            #  )

    def get_weights(self, sess):
        """Extract numerical values of all layer weights."""
        w_dict = {}
        for name, layer in self.layers_dict.items():
            w_dict[name] = {}
            if isinstance(layer, dict):
                for subname, sublayer in layer.items():
                    w, b = sess.run(sublayer.weights)
                    w_dict[name][subname] = Weights(w=w, b=b)
            else:
                w, b = sess.run(layer.weights)
                w_dict[name] = Weights(w, b)

        coeffs = sess.run([self.scale_layer.coeff,
                           self.transformation_layer.coeff])

        w_dict[SCOEFF] = coeffs[0]
        w_dict[QCOEFF] = coeffs[1]

        return w_dict

    def save_weights(self, sess, out_file):
        """Save all layer weights to `out_file`."""
        weights_dict = self.get_weights(sess)
        with open(out_file, 'wb') as f:
            pickle.dump(weights_dict, f)

        return weights_dict

    def call(self, inputs, is_training):
        """Call the network (forward-pass)."""
        v, x, t = inputs

        x_out = self.x_layer(x)
        v_out = self.v_layer(v)
        t_out = self.t_layer(t)

        # pylint:disable=invalid-name
        h = self.activation(x_out + v_out + t_out)
        h = self.activation(self.h_layer(h))

        # dropout gets applied to the output of the previous layer
        if self.dropout_prob > 0.:
            h = self.dropout(h, training=is_training)

        scale = self.scale_layer(h)
        translation = self.translation_layer(h)
        transformation = self.transformation_layer(h)

        return scale, translation, transformation


class EncoderNetNP:
    """Implements numpy version of `EncoderNet` defined above."""
    def __init__(self,
                 weights,
                 name=None,
                 activation=relu):
        self.name = name
        self.activation = activation
        self.x_layer = EncodingLayerNP(weights['x_layer'])
        self.v_layer = EncodingLayerNP(weights['v_layer'])

        self.t_layer = DenseLayerNP(weights['t_layer'])
        self.h_layer = DenseLayerNP(weights['h_layer'])
        self.translation_layer = DenseLayerNP(weights[TNAME])

        self.scale_layer = ScaledTanhLayerNP(weights[SCOEFF],
                                             weights[SNAME])
        self.transformation_layer = ScaledTanhLayerNP(weights[QCOEFF],
                                                      weights[QNAME])
        #  self.scale_layer = DenseLayerNP(weights[SNAME])
        #  self.transformation_layer = DenseLayerNP(weights[QNAME])

    def __call__(self, inputs):
        v, x, t = inputs

        x = self.x_layer(x)
        v = self.v_layer(v)
        t = self.t_layer(t)
        h_out = self.activation(x + v + t)
        h_out = self.activation(self.h_layer(h_out))
        scale = self.scale_layer(h_out)
        translation = self.translation_layer(h_out)
        transformation = self.transformation_layer(h_out)

        return scale, translation, transformation
