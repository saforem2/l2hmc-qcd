"""
Generic, fully-connected neural network architecture for running L2HMC on a
gauge lattice configuration of links.

NOTE: Lattices are flattened before being passed as input to the network.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Author: Sam Foreman (github: @saforem2)
Date: 01/16/2019

"""
#  pylint: disable=invalid-name, attribute-defined-outside-init
#  pylint: disable=too-few-public-methods
#  pylint: disable=relative-beyond-top-level
#  pylint: disable=too-many-instance-attributes
from __future__ import absolute_import, division, print_function

import pickle

import tensorflow as tf

import config as cfg

from .layers import DenseLayerNP, relu, ScaledTanh, ScaledTanhNP
from .network_utils import custom_dense, tf_zeros
from seed_dict import seeds, vnet_seeds, xnet_seeds

TF_FLOAT = cfg.TF_FLOAT
TF_INT = cfg.TF_INT
Weights = cfg.Weights

SNAME = 'scale_layer'
SCOEFF = 'coeff_scale'

QNAME = 'transformation_layer'
QCOEFF = 'coeff_transformation'

TNAME = 'translation_layer'


class GenericNet(tf.keras.Model):
    """Generic (fully-connected) network used in training L2HMC."""
    def __init__(self, model_name, **kwargs):
        """
        Initialization method.

        Args:
            model_name: Name of the model.
            kwargs: Keyword arguments used to specify specifics of
                convolutional structure.
        """
        super(GenericNet, self).__init__(name=model_name)

        self.x_dim = kwargs.get('x_dim', None)
        self.net_name = kwargs.get('net_name', None)
        self.factor = kwargs.get('factor', None)
        self.dropout_prob = kwargs.get('dropout_prob', 0.)
        self.num_hidden1 = kwargs.get('num_hidden1', None)
        self.num_hidden2 = kwargs.get('num_hidden2', None)
        self.activation = kwargs.get('generic_activation', tf.nn.relu)
        self._input_shape = kwargs.get('input_shape', None)
        net_seeds = kwargs.get('net_seeds', None)

        with tf.name_scope(self.net_name):
            #  input_shape=(self.x_dim,))
            self.x_layer = custom_dense(name='x_layer',
                                        factor=self.factor/3.,
                                        units=self.num_hidden1,
                                        seed=net_seeds['x_layer'])

            self.v_layer = custom_dense(name='v_layer',
                                        factor=1./3.,
                                        units=self.num_hidden1,
                                        seed=net_seeds['v_layer'])

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

            self.scale_layer = ScaledTanhLayer(name='scale', factor=0.001,
                                               units=self.x_dim,
                                               seed=net_seeds[SNAME])

            self.transformation_layer = ScaledTanhLayer(name='transformation',
                                                        factor=0.001,
                                                        units=self.x_dim,
                                                        seed=net_seeds[QNAME])

            # Dropout layer (only defined if `dropout_prob > 0`
            if self.dropout_prob > 0:
                self.dropout = tf.keras.layers.Dropout(
                    self.dropout_prob, seed=net_seeds['dropout'],
                )

            self.layers_types = {
                'x_layer': 'custom_dense',
                'v_layer': 'custom_dense',
                't_layer': 'custom_dense',
                'h_layer': 'custom_dense',
                'scale_layer': 'ScaledTanhLayer',
                'translation_layer': 'custom_dense',
                'transformation_layer': 'ScaledTanhLayer'
            }

            self.layers_dict = {
                'x_layer': self.x_layer,
                'v_layer': self.v_layer,
                't_layer': self.t_layer,
                'h_layer': self.h_layer,
                'scale_layer': self.scale_layer.layer,
                'translation_layer': self.translation_layer,
                'transformation_layer': self.transformation_layer.layer
            }

    def get_layer_weights(self, sess):
        """Extract numerical values of all layer weights."""
        weights_dict = {}
        for name, layer in self.layers_dict.items():
            weights_dict[name] = {}
            if isinstance(layer, dict):
                for subname, sublayer in layer.items():
                    w, b = sess.run(sublayer.weights)
                    weights_dict[name][subname] = Weights(w=w, b=b)
            else:
                w, b = sess.run(layer.weights)
                weights_dict[name] = Weights(w=w, b=b)

        coeffs = sess.run([self.scale_layer.coeff,
                           self.transformation_layer.coeff])

        weights_dict[SCOEFF] = coeffs[0]
        weights_dict[QCOEFF] = coeffs[1]

        return weights_dict

    def save_weights(self, sess, out_file):
        """Save all layer weights to `out_file`."""
        weights_dict = self.get_layer_weights(sess)
        with open(out_file, 'wb') as f:
            pickle.dump(weights_dict, f)

        fpath, ext = out_file.split('.')
        types_file = f'{fpath}_types.{ext}'
        with open(types_file, 'wb') as f:
            pickle.dump(self.layers_types, f)

    def call(self, inputs, train_phase):
        """Call network."""
        v, x, t = inputs

        x = self.x_layer(x)
        v = self.v_layer(v)
        t = self.t_layer(t)
        h = self.activation(x + v + t)
        h = self.activation(self.h_layer(h))

        # dropout gets applied to the output of the previous layer
        if self.dropout_prob > 0:
            h = self.dropout(h, training=train_phase)

        scale = self.scale_layer(h)
        translation = self.translation_layer(h)
        transformation = self.transformation_layer(h)

        return scale, translation, transformation


class GenericNetNP:
    def __init__(self, weights, activation=relu):
        #  self.name = name
        self.activation = activation
        self.x_layer = DenseLayerNP(weights['x_layer'])
        self.v_layer = DenseLayerNP(weights['v_layer'])
        self.t_layer = DenseLayerNP(weights['t_layer'])
        self.h_layer = DenseLayerNP(weights['h_layer'])
        self.translation_layer = DenseLayerNP(weights[TNAME])

        self.scale_layer = ScaledTanhLayerNP(weights[SCOEFF],
                                             weights[SNAME])
        self.transformation_layer = ScaledTanhLayerNP(weights[QCOEFF],
                                                      weights[QNAME])
        #  self.scale_layer = DenseLayerNP(weights['scale_layer'])

    def __call__(self, inputs):
        v, x, t = inputs
        v = self.v_layer(v)
        x = self.x_layer(x)
        t = self.t_layer(t)
        h = self.activation(v + x + t)
        h = self.activation(self.h_layer(h))
        scale = self.scale_layer(h)
        translation = self.translation_layer(h)
        transformation = self.transformation_layer(h)

        return scale, translation, transformation
