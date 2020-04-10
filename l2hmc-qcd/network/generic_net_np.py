"""
generic_net_np.py

Implements `GenericNetNp`, a simple, fully-connected (feed-forward) neural net
in pure numpy.

Author: Sam Foreman (github: @saforem2)
Date: 01/18/2020
"""
# pylint:disable=invalid-name, too-few-public-methods
from __future__ import absolute_import, division, print_function
from network.activation_functions import ReLU

import numpy as np

import config
from config import Weights, NP_FLOAT



class PairLayerNP:
    def __init__(self, weights, activation=linear):
        self.activation = activation
        self.weights = weights
        self.x_layer = DenseLayerNP(weights['x_layer'])
        self.y_layer = DenseLayerNP(weights['y_layer'])

    @staticmethod
    def encode(x):
        """Encode `x` as `(cos(x), sin(x))`."""
        return np.cos(x), np.sin(x)

    @staticmethod
    def decode(x, y):
        """Inverse of `self.encode`."""
        return np.arctan2(y, x)

    def __call__(self, x):
        _x, _y = self.encode(x)
        xout = self.x_layer(_x)
        yout = self.y_layer(_y)

        return np.array([xout, yout])


class EncodingLayerNP:
    def __init__(self, weights, activation=linear):
        self.weights = weights
        self.x_layer = DenseLayerNP(weights['x_layer'])
        self.y_layer = DenseLayerNP(weights['y_layer'])

    @staticmethod
    def encode(phi):
        """Encode the angle `phi` 𝜙 --> [cos(𝜙), sin(𝜙)]."""
        return np.array([np.cos(phi), np.sin(phi)])

    @staticmethod
    def decode(x, y):
        return np.arctan2(y, x)

    def __call__(self, phi):
        phi_enc = self.encode(phi)
        xout = self.x_layer(phi_enc[0])
        yout = self.y_layer(phi_enc[1])

        return self.decode(xout, yout)


class EncoderNetNP:
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

        sname = 'scale_layer'
        tname = 'translation_layer'
        qname = 'transformation_layer'

        self.scale_layer = DenseLayerNP(weights[sname])
        self.translation_layer = DenseLayerNP(weights[tname])
        self.transformation_layer = DenseLayerNP(weights[qname])

        self.coeff_scale = weights['coeff_scale']
        self.coeff_transformation = weights['coeff_transformation']

    def __call__(self, inputs):
        v, x, t = inputs

        xout = self.x_layer(x)
        vout = self.v_layer(v)
        tout = self.t_layer(t)

        h = self.activation(xout + vout + tout)
        h = self.activation(self.h_layer(h))

        scale = np.tanh(self.scale_layer(h))
        scale *= np.exp(self.coeff_scale)

        transformation = np.tanh(self.transformation_layer(h))
        transformation *= np.exp(self.coeff_transformation)

        translation = self.translation_layer(h)

        return scale, translation, transformation


class PairedGenericNetNP:
    def __init__(self, weights, name=None, activation=np.tanh):
        self.name = name
        self.activation = activation
        self.x_layer = PairLayerNP(weights['x_layer'])
        self.v_layer = PairLayerNP(weights['v_layer'])
        self.t_layer = PairLayerNP(weights['t_layer'])

        #  self.h_layer = DenseLayerNP(weights['h_layer'])
        #  self.scale_layer = DenseLayerNP(weights['scale_layer'])
        self.h_layer = DenseLayerNP(weights['h_layer'])
        self.scale_layer = DenseLayerNP(weights['scale_layer'])

        transl_weight = weights['translation_layer']
        self.translation_layer = DenseLayerNP(transl_weight)
        #  self.translation_layer = DenseLayerNP(transl_weight)

        transf_weight = weights['transformation_layer']
        self.transformation_layer = DenseLayerNP(transf_weight)
        #  self.transformation_layer = DenseLayerNP(transf_weight)

        self.coeff_scale = weights['coeff_scale']
        self.coeff_transformation = weights['coeff_transformation']

    def __call__(self, inputs):
        v, x, t = inputs
        v = self.v_layer(v)
        x = self.x_layer(x)
        t = self.t_layer(t)
        xvt = x + v + t

        h = self.activation(np.arctan2(xvt[1], xvt[0]))

        #  h = self.activation(v + x + t)
        h = self.activation(self.h_layer(h))

        S = np.tanh(self.scale_layer(h))
        Q = np.tanh(self.transformation_layer(h))

        scale = S * np.exp(self.coeff_scale)
        translation = self.translation_layer(h)
        transformation = Q * np.exp(self.coeff_transformation)

        return scale, translation, transformation



