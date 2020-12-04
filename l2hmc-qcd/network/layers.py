"""
layers.py

Collection of network layers.

Author: Sam Foreman
Date: 04/01/2020
"""
# pylint:disable=too-few-public-methods,relative-beyond-top-level
from __future__ import absolute_import, division, print_function

import numpy as np
from typing import Union, Callable
import tensorflow as tf

from config import NP_FLOATS, TF_FLOATS, Weights
import utils.file_io as io


TF_FLOAT = TF_FLOATS[tf.keras.backend.floatx()]
NP_FLOAT = NP_FLOATS[tf.keras.backend.floatx()]

layers = tf.keras.layers

# pylint:disable=too-many-arguments


def swish(x):
    return tf.keras.activations.swish(x)


def relu(x):
    """Rectified Linear Unit Activation Function."""
    return np.where(x >= 0, x, 0)


def linear(x):
    """Linear activation function. Simply returns `x`."""
    return x


def cast_array(x, dtype=NP_FLOAT):
    """Cast the array `x` to `dtype`."""
    return np.array(x, dtype=dtype)


def map_angle_to_reals(theta):
    """Maps theta in [-pi, pi] to the real line [-inf, inf]."""
    return tf.math.tan(theta / 2.)


def periodic_padding(image, padding=1):
    """Create a periodic padding to emulate periodic boundary conditions."""
    upper_pad = image[-padding:, :]
    lower_pad = image[:padding, :]
    partial_image = tf.concat([upper_pad, image, lower_pad], axis=0)
    left_pad = partial_image[:, -padding:]
    right_pad = partial_image[:, :padding]
    padded_image = tf.concat([left_pad, partial_image, right_pad], axis=1)

    return padded_image


def convert_to_image(x):
    """Create image from lattice by doubling the size."""
    y = np.zeros((2 * x.shape[0], 2 * x.shape[1]))
    y[::2, 1::2] = x[:, :, 0]
    y[1::2, ::2] = x[:, :, 1]
    return y


def _get_layer_weights(layer):
    """Get an individual layers' weights."""
    w, b = layer.weights
    return Weights(w=w.numpy(), b=b.numpy())


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


def dense_layer(units, seed=None, factor=1.,
                zero_init=False, name=None, **kwargs):
    """Custom dense layer with specified weight initialization."""
    if zero_init:
        kern_init = tf.zeros_initializer()

    kern_init = tf.keras.initializers.VarianceScaling(
        seed=seed,
        mode='fan_in',
        scale=2.*factor,
        distribution='truncated_normal',
    )

    return tf.keras.layers.Dense(
        units=units,
        name=name,
        use_bias=True,
        kernel_initializer=kern_init,
        bias_initializer=tf.zeros_initializer(),
        **kwargs
    )


class ScaledTanhLayer(layers.Layer):
    """Implements a custom dense layer that is scaled by a trainable var."""
    def __init__(
            self, units: int, factor: float, name='ScaledTanhLayer'
    ):
        super(ScaledTanhLayer, self).__init__(name=name)
        self.units = units
        self.factor = factor
        self.coeff = tf.Variable(
            initial_value=tf.zeros([1, units]),
            name=f'{name}/coeff', trainable=True
        )
        self.dense = layers.Dense(
            units, kernel_initializer=tf.keras.initializers.VarianceScaling(
                mode='fan_in', scale=2.*self.factor,
                distribution='truncated_normal'
            )
        )

    def get_config(self):
        config = super(ScaledTanhLayer, self).get_config()
        config.update({
            'units': self.units,
            'factor': self.factor,
            'coeff': self.coeff.numpy(),
        })

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        out = tf.keras.activations.tanh(self.dense(inputs))

        return tf.exp(self.coeff) * out


class StackedLayer:
    """Wrapper class that stacks [cos(x), sin(x)] inputs."""

    def __init__(self, units, factor=1., name='StackedLayer',
                 seed=None, zero_init=False, **kwargs):
        """Initialization method."""
        self.layer = dense_layer(name=name, seed=seed,
                                 units=units, factor=factor,
                                 zero_init=zero_init, **kwargs)

    def __call__(self, phi):
        phi = tf.concat([tf.cos(phi), tf.sin(phi)], axis=-1)
        return self.layer(phi)
