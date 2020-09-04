"""
gauge_network.py

Implements the `GaugeNetwork` for training the L2HMC sampler on a 2D U(1)
lattice gauge theory model.

@author: Sam Foreman
@date: 09/01/2020
"""
from __future__ import absolute_import, division, print_function


from typing import Callable, List, Optional, Union

import tensorflow as tf

from tensorflow.keras import layers


from utils.attr_dict import AttrDict


class ConvolutionBlock2D(layers.Layer):
    """Implements a block consisting of: 2 x [Conv2D, MaxPooling2D]."""
    def __init__(                # pylint:disable=too-many-arguments
            self,
            input_shape: tuple,  # (N, H, W, C)
            filters: tuple,      # number of filters in layers 1, 2: (n1, n2)
            sizes: tuple,        # ^ sizes: [x, y] (square) or [(x, y), (u, v)]
            pool_sizes: tuple = None,      # MaxPool sizes (same as ^)
            activations: str = None,       # activation function to use
            paddings: str = None,          # Padding to use 'valid' or 'same'
            use_batch_norm: bool = False,  # Use BatchNorm layer?
            **kwargs,
    ):
        super(ConvolutionBlock2D, self).__init__(**kwargs)
        if pool_sizes is None:
            pool_sizes = 2 * [(2, 2)]
        if activations is None:
            activations = 2 * ['relu']
        if isinstance(activations, str):
            activations = 2 * [activations]
        if isinstance(paddings, str):
            paddings = 2 * [paddings]

        self.conv1 = layers.Conv2D(
            filters=filters[0],
            kernel_size=sizes[0],
            activation=activations[0],
            input_shape=input_shape,
            padding=paddings[0],
        )
        self.pool1 = layers.MaxPooling2D(pool_sizes[0])

        self.conv2 = layers.Conv2D(
            filters=filters[1],
            kernel_size=sizes[1],
            activation=activations[1],
            padding=paddings[0],
        )
        self.pool2 = layers.MaxPooling2D(pool_sizes[1])

        self.flatten = layers.Flatten()

        self._use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = layers.BatchNormalization(axis=-1)

    def call(self, inputs, training=None, **kwargs):
        inputs = self.pool1(self.conv1(inputs))
        inputs = self.flatten(self.pool2(self.conv2(x)))
        if self._use_batch_norm:
            inputs = self.batch_norm(x, training=training)

        return inputs

# pylint:disable=arguments-differ
class ConcatenatedDense(layers.Layer):
    """Layer that converts from an angular repr to a [x, y] repr."""
    def __init__(
            self,
            units: int,
            kernel_initializer: Union[Callable, str] = 'glorot_uniform',
            **kwargs
    ):
        super(ConcatenatedDense, self).__init__(**kwargs)
        self._name = kwargs.get('name', 'ConcatenatedDense')
        self.dense_x = layers.Dense(
            units=units, kernel_initializer=kernel_initializer,
        )
        self.dense_y = layers.Dense(
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


class ScaledTanhLayer(layers.Layer):
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
        self.dense = layers.Dense(
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


def vs_init(factor):
    return tf.keras.initializers.VarianceScaling(
        mode='fan_in',
        scale=2.*factor,
        distribution='truncated_normal',
    )


# pylint:disable=too-many-arguments, too-many-instance-attributes,
# pylint:disable=too-many-ancestors
class GaugeNetwork(layers.Layer):
    """Implements the Feed-Forward NN for carrying out the L2HMC algorithm."""
    def __init__(
            self,
            config: NetworkConfig,
            xdim: int,
            factor: Optional[float] = 1.,
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
                self.dropout = layers.Dropout(config.dropout_prob)

            kwargs = {
                'units': config.units[0],
            }
            self.x_layer = layers.Dense(name='x_layer', units=config.units[0],
                                        kernel_initializer=vs_init(factor/3.))
            self.v_layer = layers.Dense(name='v_layer', units=config.units[0],
                                        kernel_initializer=vs_init(1./3.))
            self.t_layer = layers.Dense(name='t_layer', units=config.units[0],
                                        kernel_initializer=vs_init(1./3.))
            self.h_layer1 = layers.Dense(name='h_layer1',
                                         units=config.units[1],
                                         kernel_initializer=vs_init(1.))
            self.h_layer2 = layers.Dense(name='h_layer2',
                                         units=config.units[2],
                                         kernel_initializer=vs_init(1.))

            #  self.hidden_layers = [
            #      layers.Dense(name=f'h_layer{i}', units=n)
            #      for i, n in enumerate(config.units[1:])
            #  ]

            self.scale_layer = ScaledTanhLayer(
                name='scale', units=xdim,
                kernel_initializer=vs_init(0.001),
            )
            self.translation_layer = layers.Dense(
                name='translation', units=xdim,
                kernel_initializer=vs_init(0.001)
            )
            self.transformation_layer = ScaledTanhLayer(
                name='transformation', units=xdim,
                kernel_initializer=vs_init(0.001),
            )

    def call(self, inputs, training=None):
        """Call the network (forward-pass)."""
        v, x, t = inputs
        x_rect = tf.complex(tf.math.cos(x), tf.math.sin(x))

        v_out = self.v_layer(v)
        t_out = self.t_layer(t)
        x_out = tf.math.angle(self.x_layer(x_rect))
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
