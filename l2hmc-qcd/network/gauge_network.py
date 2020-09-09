"""
gauge_network.py

Implements the `GaugeNetwork` for training the L2HMC sampler on a 2D U(1)
lattice gauge theory model.

@author: Sam Foreman
@date: 09/01/2020
"""
from __future__ import absolute_import, division, print_function


from typing import Callable, List, Optional, Union
from config import NetworkConfig

import tensorflow as tf

from tensorflow.keras import layers


from utils.attr_dict import AttrDict

ACTIVATIONS = {
    'tanh': tf.keras.activations.tanh,
    'relu': tf.keras.activations.relu,
    'linear': tf.keras.activations.linear,
}


def vs_init(factor):
    return tf.keras.initializers.VarianceScaling(
        mode='fan_in',
        scale=2.*factor,
        distribution='truncated_normal',
    )


# pylint:disable=arguments-differ
class ConcatenatedDense(layers.Layer):
    """Layer that converts from an angular repr to a [x, y] repr."""
    def __init__(
            self,
            units: int,
            kernel_initializer: Union[Callable, str] = vs_init(1.),
            **kwargs
    ):
        super(ConcatenatedDense, self).__init__(**kwargs)
        self._name = kwargs.get('name', 'ConcatenatedDense')
        self.layer = layers.Dense(units=2*units,
                                  kernel_initializer=kernel_initializer)
        #  self.dense_x = layers.Dense(
        #      units=units, kernel_initializer=kernel_initializer,
        #  )
        #  self.dense_y = layers.Dense(
        #      units=units, kernel_initializer=kernel_initializer,
        #  )

    def call(self, phi):
        """Call the layer (forward-pass)."""
        phi = tf.concat([tf.math.cos(phi), tf.math.sin(phi)], axis=-1)
        return self.layer(phi)
        #  x = self.dense_x(tf.math.cos(phi))
        #  y = self.dense_y(tf.math.sin(phi))
        #  xy_inputs = tf.squeeze(
        #      tf.concat([tf.cos(inputs), tf.sin(inputs)], axis=-1)
        #  )
        #  return self.dense(xy_inputs)
        #  return tf.math.angle(x + 1j * y)
        #  return tf.math.angle(tf.complex(x, y))
        #  return self.layer(phi)


class ScaledTanhLayer(layers.Layer):
    """Implements a custom dense layer that is scaled by a trainable var."""
    def __init__(
            self,
            units: int,
            kernel_initializer: Union[Callable, str] = None,
            **kwargs
    ):
        super(ScaledTanhLayer, self).__init__(**kwargs)
        name = kwargs.get('name', 'ScaledTanhLayer')
        self.coeff = tf.Variable(initial_value=tf.zeros([1, units]),
                                 name=f'{name}/coeff', trainable=True)
        self.dense = layers.Dense(
            units, kernel_initializer=kernel_initializer
        )

    def call(self, inputs):
        out = tf.keras.activations.tanh(self.dense(inputs))
        return tf.exp(self.coeff) * out


# pylint:disable=too-many-arguments, too-many-instance-attributes,
# pylint:disable=too-many-ancestors
class GaugeNetwork(layers.Layer):
    """Implements the Feed-Forward NN for carrying out the L2HMC algorithm."""
    def __init__(
            self,
            config: NetworkConfig,
            xdim: int,
            factor: Optional[float] = 1.,
            kernel_initializer: Optional[Union[str, Callable]] = None,
            **kwargs,
    ):
        super(GaugeNetwork, self).__init__(**kwargs)
        self.xdim = xdim
        self.factor = factor
        self._config = config
        self._kernel_initializer = kernel_initializer
        #  if kernel_initializer is None:
        #      def vs_init(scale):
        #          return tf.keras.initializers.VarianceScaling(scale)
        #
        #      kinits = AttrDict({
        #          'x': vs_init(factor / 3.),
        #          'v': vs_init(1 / 3.),
        #          't': vs_init(1 / 3.),
        #          'h1': vs_init(1.),
        #          'h2': vs_init(1.),
        #          'sk': vs_init(0.001),
        #          'tk': vs_init(0.001),
        #          'qk': vs_init(0.001),
        #      })
        #
        #  if kernel_initializer == 'zeros':
        #      self._kinit = 'zeros'
        #  else:
        #      self._kernel_initializer = lambda scale: (
        #          tf.keras.initializers.VarianceScaling(scale)
        #      )
        #  self._kernel_initializer = kernel_initializer

        xk_init = self._get_kern_init(factor/3.)
        vk_init = self._get_kern_init(1./3.)
        tk_init = self._get_kern_init(1./3.)
        h1_init = self._get_kern_init(1.)
        h2_init = self._get_kern_init(1.)

        sk_init = self._get_kern_init(0.001)
        tk_init = self._get_kern_init(0.001)
        qk_init = self._get_kern_init(0.001)

        name = kwargs.get('name', 'GaugeNetwork')
        with tf.name_scope(name):
            self.activation = ACTIVATIONS.get(config.activation_fn,
                                              ACTIVATIONS['relu'])
            if config.dropout_prob > 0:
                self.dropout = layers.Dropout(config.dropout_prob)

            self.v_layer = layers.Dense(name='v_layer',
                                        units=config.units[0],
                                        kernel_initializer=vk_init)

            self.x_layer = layers.Dense(name='x_layer',
                                        units=config.units[0],
                                        kernel_initializer=xk_init)
            #  self.x_layer = ConcatenatedDense(name='x_layer',
            #                                   units=config.units[0],
            #                                   kernel_initializer=xk_init)

            #  self.x_layer = ConcatenatedDense(2 * config.units[0],
            #  self.x_layer = layers.Dense(name='x_layer',
            #                              units=config.units[0],
            #                              kernel_initializer=xk_init)
            self.t_layer = layers.Dense(name='t_layer',
                                        units=config.units[0],
                                        kernel_initializer=tk_init)
            self.h_layer1 = layers.Dense(name='h_layer1',
                                         units=config.units[1],
                                         kernel_initializer=h1_init)
            self.h_layer2 = layers.Dense(name='h_layer2',
                                         units=config.units[2],
                                         kernel_initializer=h2_init)

            self.scale_layer = ScaledTanhLayer(
                name='scale', units=xdim, kernel_initializer=sk_init,
            )
            self.translation_layer = layers.Dense(
                name='translation', units=xdim, kernel_initializer=tk_init,
            )
            self.transformation_layer = ScaledTanhLayer(
                name='transformation', units=xdim, kernel_initializer=qk_init,
            )

    def _get_kern_init(self, factor=1.):
        if self._kernel_initializer == 'zeros':
            return 'zeros'
        return vs_init(factor)

    def call(self, inputs, training=None):
        """Call the network (forward-pass)."""
        v, x, t = inputs
        #  xc = tf.complex(tf.math.cos(x), tf.math.sin(x))

        t_out = self.t_layer(t)
        v_out = tf.math.angle(self.v_layer(
            tf.complex(tf.math.cos(v), tf.math.sin(v))
        ))
        x_out = tf.math.angle(self.x_layer(
            tf.complex(tf.math.cos(x), tf.math.sin(x))
        ))
        h = self.activation(x_out + v_out + t_out)
        h = self.activation(self.h_layer1(h))
        h = self.activation(self.h_layer2(h))

        #  h = self.activation(
        #      self.v_layer(v) + self.x_layer(x) + self.t_layer(t)
        #  )
        #  for layer in self.hidden_layers:
        #      h = self.activation(layer(h))

        if self._config.dropout_prob > 0 and training:
            h = self.dropout(h, training=training)

        scale = self.scale_layer(h)
        translation = self.translation_layer(h)
        transformation = self.transformation_layer(h)

        return scale, translation, transformation
