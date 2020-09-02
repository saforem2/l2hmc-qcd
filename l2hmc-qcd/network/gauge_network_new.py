"""
gauge_network.py

Implements the `GaugeNetwork` for training the L2HMC sampler on a 2D U(1)
lattice gauge theory model.

@author: Sam Foreman
@date: 09/01/2020
"""
from __future__ import absolute_import, division, print_function

import cmath

from typing import Callable, Dict, List, Optional, Union
from collections import namedtuple

import numpy as np
import tensorflow as tf

import utils.file_io as io

from config import Weights
from utils.attr_dict import AttrDict

# pylint:disable=arguments-differ
class ConcatenatedDense(tf.keras.layers.Layer):
    """Layer that converts from an angular repr to a [x, y] repr."""
    def __init__(
            self,
            units: int,
            kernel_initializer: Union[Callable, str] = 'glorot_uniform',
            **kwargs
    ):
        super(ConcatenatedDense, self).__init__(**kwargs)
        self._name = kwargs.get('name', 'ConcatenatedDense')
        self.dense_x = tf.keras.layers.Dense(
            units=units, kernel_initializer=kernel_initializer,
        )
        self.dense_y = tf.keras.layers.Dense(
            units=units, kernel_initializer=kernel_initializer,
        )

    def call(self, phi):
        """Call the layer (forward-pass)."""
        x = self.dense_x(tf.math.cos(phi))
        y = self.dense_y(tf.math.sin(phi))
        #  xy_inputs = tf.squeeze(
        #      tf.concat([tf.cos(inputs), tf.sin(inputs)], axis=-1)
        #  )
        #  return self.dense(xy_inputs)
        #  return tf.math.angle(x + 1j * y)
        return tf.math.angle(tf.complex(x, y))


class ScaledTanhLayer(tf.keras.layers.Layer):
    """Implements a custom dense layer that is scaled by a trainable var."""
    def __init__(
            self,
            units: int,
            kernel_initializer: Union[Callable, str] = 'glorot_uniform',
            **kwargs
    ):
        super(ScaledTanhLayer, self).__init__(**kwargs)
        name = kwargs.get('name', 'ScaledTanhLayer')
        self.coeff = tf.Variable(initial_value=tf.zeros([1, units]),
                                 name=f'{name}/coeff', trainable=True)
        self.dense = tf.keras.layers.Dense(
            units, kernel_initializer=kernel_initializer
        )

    def call(self, inputs):
        out = tf.keras.activations.tanh(self.dense(inputs))
        return tf.exp(self.coeff) * out


class NetworkConfig(AttrDict):
    """Configuration object for `GaugeNetwork` object."""
    def __init__(
            self,
            units: List,
            name: Optional[str] = None,
            dropout_prob: Optional[float] = 0.,
            activation_fn: Optional[Callable] = tf.nn.relu
    ):
        super(NetworkConfig, self).__init__(
            name=name,
            units=units,
            dropout_prob=dropout_prob,
            activation_fn=activation_fn,
        )


# pylint:disable=too-many-arguments, too-many-instance-attributes
class GaugeNetwork(tf.keras.models.Model):
    """Implements the Feed-Forward NN for carrying out the L2HMC algorithm."""
    def __init__(
            self,
            config: NetworkConfig,
            xdim: int,
            factor: Optional[float] = 1.,
            k_init: Optional[Union[Callable, str]] = 'glorot_uniform',
            **kwargs,
    ):
        super(GaugeNetwork, self).__init__(**kwargs)
        self.xdim = xdim
        self.factor = factor
        self._config = config
        self.activation = config.activation_fn
        name = kwargs.get('name', 'GaugeNetwork')
        with tf.name_scope(name):
            if config.dropout_prob > 0:
                self.dropout = tf.keras.layers.Dropout(config.dropout_prob)

            #  self.x_layer = ConcatenatedDense(name='x_layer',
            #                                   units=config.units[0],
            #                                   #  input_shape=(2 * xdim,),
            #                                   kernel_initializer=k_init)
            self.x_layer = tf.keras.layers.Dense(name='x_layer',
                                                 units=config.units[0],
                                                 #  input_shape=(xdim,),
                                                 kernel_initializer=k_init)
            self.v_layer = tf.keras.layers.Dense(name='v_layer',
                                                 units=config.units[0],
                                                 #  input_shape=(xdim,),
                                                 kernel_initializer=k_init)
            self.t_layer = tf.keras.layers.Dense(name='t_layer',
                                                 units=config.units[0],
                                                 #  input_shape=(2 * xdim,),
                                                 kernel_initializer=k_init)
            self.hidden_layers = [
                tf.keras.layers.Dense(name=f'h_layer{i}', units=n)
                for i, n in enumerate(config.units[1:])
            ]

            self.scale_layer = ScaledTanhLayer(
                name='scale', units=xdim, kernel_initializer=k_init,
            )
            self.translation_layer = tf.keras.layers.Dense(
                name='translation', units=xdim, kernel_initializer=k_init
            )
            self.transformation_layer = ScaledTanhLayer(
                name='transformation', units=xdim, kernel_initializer=k_init,
            )

    def call(self, inputs, training=None):
        """Call the network (forward-pass)."""
        v, x, t = inputs
        xr = tf.complex(tf.math.cos(x), tf.math.sin(x))

        v_ = self.v_layer(v)
        t_ = self.t_layer(t)
        x_ = tf.math.angle(self.x_layer(xr))
        h = self.activation(x_ + v_ + t_)
        #  h = self.activation(
        #      self.v_layer(v) + self.x_layer(x) + self.t_layer(t)
        #  )
        for layer in self.hidden_layers:
            h = self.activation(layer(h))

        if self._config.dropout_prob > 0 and training:
            h = self.dropout(h, training=training)

        scale = self.scale_layer(h)
        translation = self.translation_layer(h)
        transformation = self.transformation_layer(h)

        return scale, translation, transformation
