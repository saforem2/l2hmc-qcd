"""
dynamics_np.py

Implements `GenericNetNp`, a simple, fully-connected (feed-forward) neural net
in pure numpy.

Author: Sam Foreman (github: @saforem2)
Date: 01/18/2020
"""
from __future__ import absolute_import, division, print_function
from network.activation_functions import ReLU

import numpy as np

import config
from config import Weights, NP_FLOAT


def relu(x):
    """Rectified Linear Unit Activation Function."""
    return np.where(x >= 0, x, 0)


def linear(x):
    """Linear activation function. Simply returns `x`."""
    return x


def cast_array(x, dtype=NP_FLOAT):
    return np.array(x, dtype=dtype)


class DenseLayerNP:
    """Implements fully-connected Dense layer using numpy."""
    def __init__(self, weights, activation=linear):
        self.activation = activation
        self.weights = weights
        self.w = weights.w
        self.b = weights.b

    def __call__(self, x):
        x = cast_array(x)
        return self.activation(np.dot(x, self.w) + self.b)


class GenericNetNP:
    def __init__(self, weights, name=None, activation=relu):
        self.name = name
        self.activation = activation
        self.x_layer = DenseLayerNP(weights['x_layer'])
        self.v_layer = DenseLayerNP(weights['v_layer'])
        self.t_layer = DenseLayerNP(weights['t_layer'])
        self.h_layer = DenseLayerNP(weights['h_layer'])
        self.scale_layer = DenseLayerNP(weights['scale_layer'])

        transl_weight = weights['translation_layer']
        transf_weight = weights['transformation_layer']
        self.translation_layer = DenseLayerNP(transl_weight)
        self.transformation_layer = DenseLayerNP(transf_weight)

        self.coeff_scale = weights['coeff_scale']
        self.coeff_transformation = weights['coeff_transformation']

    def __call__(self, inputs):
        v, x, t = inputs
        v = self.v_layer(v)
        x = self.x_layer(x)
        t = self.t_layer(t)
        h = self.activation(v + x + t)
        h = self.activation(self.h_layer(h))

        S = np.tanh(self.scale_layer(h))
        Q = np.tanh(self.transformation_layer(h))

        scale = S * np.exp(self.coeff_scale)
        translation = self.translation_layer(h)
        transformation = Q * np.exp(self.coeff_transformation)

        return scale, translation, transformation
