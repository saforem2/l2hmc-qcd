"""
layers.py

Collection of network layers.

Author: Sam Foreman
Date: 04/01/2020
"""
# pylint:disable=too-few-public-methods,relative-beyond-top-level
from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from config import TF_FLOAT, NP_FLOAT, Weights

from .network_utils import tf_zeros, custom_dense


def relu(x):
    """Rectified Linear Unit Activation Function."""
    return np.where(x >= 0, x, 0)


def linear(x):
    """Linear activation function. Simply returns `x`."""
    return x


def cast_array(x, dtype=NP_FLOAT):
    """Cast the array `x` to `dtype`."""
    return np.array(x, dtype=dtype)


class DenseLayerNP:
    """Implements fully-connected Dense layer using numpy."""
    def __init__(self, weights, activation=linear):
        self.activation = activation
        self.weights = weights
        self.w = weights.w
        self.b = weights.b

    def __call__(self, x):
        return self.activation(np.dot(x, self.w) + self.b)



class ScaledTanhLayer:
    """Wrapper class for dense layer + exp scaled tanh output."""
    def __init__(self, name, factor, units, seed):
        layer_name = f'{name}_layer'
        coeff_name = f'coeff_{name}'
        with tf.name_scope(name):
            self.coeff = tf.Variable(name=coeff_name,
                                     trainable=True,
                                     dtype=TF_FLOAT,
                                     initial_value=tf_zeros([1, units]))

            self.layer = custom_dense(name=layer_name, factor=factor,
                                      units=units, seed=seed)

    def __call__(self, x):
        return tf.exp(self.coeff) * tf.nn.tanh(self.layer(x))


class ScaledTanhLayerNP:
    """Implements numpy version of `ScaledTanhLayer`."""
    def __init__(self, coeff_weight, layer_weight):
        self.coeff = coeff_weight
        self.layer = DenseLayerNP(layer_weight)

    def __call__(self, x):
        return np.exp(self.coeff) * np.tanh(self.layer(x))


class CartesianLayer:
    """Implements `CartesianLayer`."""
    def __init__(self, name, factor, units, seed, **kwargs):
        xseed = int(2 * seed)
        yseed = int(3 * seed)
        self.x_layer = custom_dense(name=f'{name}_x',
                                    factor=factor/2,
                                    units=units,
                                    seed=xseed,
                                    **kwargs)
        self.y_layer = custom_dense(name=f'{name}_y',
                                    factor=factor/2,
                                    units=units,
                                    seed=yseed,
                                    **kwargs)
    def __call__(self, x, y):
        xout = self.x_layer(x)
        yout = self.y_layer(y)

        return xout, yout


class CartesianLayerNP:
    """Implements numpy version of `CartesianLayer`."""
    def __init__(self, weights):
        self.x_layer = DenseLayerNP(weights['x_layer'])
        self.y_layer = DenseLayerNP(weights['y_layer'])

    def __call__(self, x, y):
        return self.x_layer(x), self.y_layer(y)



class EncodingLayer:
    """Implements the EncodingLayer."""
    def __init__(self, name, factor, units, seed, **kwargs):
        xseed = int(2 * seed)
        yseed = int(3 * seed)
        self.x_layer = custom_dense(name=f'{name}_x',
                                    factor=factor/2.,
                                    units=units,
                                    seed=xseed,
                                    **kwargs)
        self.y_layer = custom_dense(name=f'{name}_y',
                                    factor=factor/2.,
                                    units=units,
                                    seed=yseed,
                                    **kwargs)

    @staticmethod
    def encode(phi):
        """Encode the angle `phi` 𝜙 --> [cos(𝜙), sin(𝜙)]."""
        return tf.convert_to_tensor([tf.cos(phi), tf.sin(phi)])

    @staticmethod
    def decode(x, y):
        """Decode [cos(𝜙), sin(𝜙)] --> 𝜙."""
        return tf.atan2(y, x)

    def __call__(self, phi):
        phi_enc = self.encode(phi)
        xout = self.x_layer(phi_enc[0])  # W1 * cos(phi) + b1
        yout = self.y_layer(phi_enc[1])  # W2 * sin(phi) + b2

        return self.decode(xout, yout)


class EncodingLayerNP:
    """Implements the numpy analog of `EncodingLayer` defined above."""
    def __init__(self, weights, activation=linear):
        self.weights = weights
        self.x_layer = DenseLayerNP(weights['x_layer'])
        self.y_layer = DenseLayerNP(weights['y_layer'])

    @staticmethod
    def decode(x, y):
        """Inverse of `self.encode`."""
        return np.arctan2(y, x)

    @staticmethod
    def encode(phi):
        """Encode the angle `phi` 𝜙 --> [cos(𝜙), sin(𝜙)]."""
        return np.array([np.cos(phi), np.sin(phi)])

    def __call__(self, phi):
        phi_enc = self.encode(phi)
        xout = self.x_layer(phi_enc[0])
        yout = self.y_layer(phi_enc[1])

        return self.decode(xout, yout)


