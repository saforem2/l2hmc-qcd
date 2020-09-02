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

#  from tensorflow.contrib.framework import add_arg_scope, arg_scope

from config import NP_FLOAT, TF_FLOAT, Weights
import utils.file_io as io
from .network_utils import custom_dense, tf_zeros


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




# pylint: disable=invalid-name
class DenseLayerNP:
    """Implements fully-connected Dense layer using numpy."""

    def __init__(self, weights, activation=linear):
        self.activation = activation
        self.weights = weights
        self.w = weights.w
        self.b = weights.b

    def __call__(self, x):
        return self.activation(np.dot(x, self.w) + self.b)


def dense_layer(units, seed=None, factor=1.,
                zero_init=False, name=None, **kwargs):
    """Custom dense layer with specified weight initialization."""
    if zero_init:
        kern_init = tf.zeros_initializer()

    try:
        kern_init = tf.keras.initializers.VarianceScaling(
            seed=seed,
            mode='fan_in',
            scale=2.*factor,
            distribution='truncated_normal',
        )
    except AttributeError:
        kern_init = tf.contrib.layers.variance_scaling_initializer(
            seed=seed,
            mode='FAN_IN',
            uniform=False,
            dtype=TF_FLOAT,
            factor=2.*factor,
        )

    return tf.keras.layers.Dense(
        units=units,
        name=name,
        use_bias=True,
        kernel_initializer=kern_init,
        bias_initializer=tf.zeros_initializer(),
        **kwargs
    )


class ScaledTanhLayer:
    """Wrapper class for dense layer + exp scaled tanh output."""

    def __init__(self, name, factor, units, seed=None, zero_init=False):
        self.coeff, self.layer = self._build(name, factor,
                                             units, seed,
                                             zero_init)

    @staticmethod
    def _build(name, factor, units, seed=None, zero_init=False):
        layer_name = f'{name}_layer'
        coeff_name = f'coeff_{name}'
        with tf.name_scope(name):
            coeff = tf.Variable(name=coeff_name,
                                trainable=True,
                                dtype=TF_FLOAT,
                                initial_value=tf_zeros([1, units]))

            layer = dense_layer(seed=seed,
                                units=units,
                                factor=factor,
                                zero_init=zero_init,
                                name=layer_name)

        return coeff, layer

    def __call__(self, x):
        return tf.exp(self.coeff) * tf.nn.tanh(self.layer(x))


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


class StackedLayerNP:
    """Numpy version of `StackedLayer`."""

    def __init__(self, weights):
        self.layer = DenseLayerNP(weights)

    def __call__(self, phi):
        phi = np.concatenate([np.cos(phi), np.sin(phi)], axis=-1)
        return self.layer(phi)


class ScaledTanhLayerNP:
    """Implements numpy version of `ScaledTanhLayer`."""

    def __init__(self, coeff_weight, layer_weight):
        self.coeff = coeff_weight
        self.layer = DenseLayerNP(layer_weight)

    def __call__(self, x):
        return np.exp(self.coeff) * np.tanh(self.layer(x))


class CartesianLayer:
    """Implements `CartesianLayer`."""

    def __init__(self, name, factor, units,
                 seed=None, zero_init=False, **kwargs):
        xseed = int(2 * seed) if seed is not None else seed
        yseed = int(3 * seed) if seed is not None else seed
        self.x_layer = dense_layer(name=f'{name}_x', factor=factor/2,
                                   units=units, seed=xseed,
                                   zero_init=zero_init, **kwargs)
        self.y_layer = dense_layer(name=f'{name}_y', factor=factor/2,
                                   units=units, seed=yseed,
                                   zero_init=zero_init, **kwargs)

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

    def __init__(self, name, factor, units,
                 seed=None, zero_init=False, **kwargs):
        xseed = int(2 * seed) if seed is not None else seed
        yseed = int(3 * seed) if seed is not None else seed
        self.x_layer = dense_layer(name=f'{name}_x', factor=factor/2.,
                                   units=units, seed=xseed,
                                   zero_init=zero_init, **kwargs)
        self.y_layer = dense_layer(name=f'{name}_y', factor=factor/2.,
                                   units=units, seed=yseed,
                                   zero_init=zero_init, **kwargs)

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


def _assign_moving_average(orig_val, new_val, momentum, name):
    """Assign moving average."""
    with tf.name_scope(name):
        scaled_diff = (1 - momentum) * (new_val - orig_val)
        return tf.assign_add(orig_val, scaled_diff)


#  @add_arg_scope
def batch_norm(x,
               phase,
               axis=-1,
               shift=True,
               scale=True,
               momentum=0.99,
               eps=1e-3,
               internal_update=False,
               scope=None,
               reuse=None):
    """Implements a `BatchNormalization` layer."""
    C = x._shape_as_list()[axis]
    ndim = len(x.shape)
    var_shape = [1] * (ndim - 1) + [C]

    with tf.variable_scope(scope, 'batch_norm', reuse=reuse):
        def training():
            m, v = tf.nn.moments(x, list(range(ndim - 1)), keep_dims=True)
            update_m = _assign_moving_average(moving_m,
                                              m, momentum,
                                              'update_mean')
            update_v = _assign_moving_average(moving_v,
                                              v, momentum,
                                              'update_var')
            #  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_m)
            #  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_v)
            tf.add_to_collection('update_ops', update_m)
            tf.add_to_collection('update_ops', update_v)

            if internal_update:
                with tf.control_dependencies([update_m, update_v]):
                    output = (x - m) * tf.rsqrt(v + eps)
            else:
                output = (x - m) * tf.rsqrt(v + eps)
            return output

        def testing():
            m, v = moving_m, moving_v
            output = (x - m) * tf.rsqrt(v + eps)
            return output

        # Get mean and variance, normalize input
        moving_m = tf.get_variable('mean', var_shape,
                                   initializer=tf.zeros_initializer,
                                   trainable=False)
        moving_v = tf.get_variable('var', var_shape,
                                   initializer=tf.ones_initializer,
                                   trainable=False)

        if isinstance(phase, bool):
            output = training() if phase else testing()
        else:
            output = tf.cond(phase, training, testing)

        if scale:
            output *= tf.get_variable('gamma', var_shape,
                                      initializer=tf.ones_initializer)

        if shift:
            output += tf.get_variable('beta', var_shape,
                                      initializer=tf.zeros_initializer)

    return output
