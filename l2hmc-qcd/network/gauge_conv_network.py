"""
gauge_conv_network.py

Prepends a `ConvolutionBlock2D` at the beginning of the `GaugeNetwork`
used for training the L2HMC sampler on a 2D U(1) lattice gauge theory model.

@author: Sam Foreman
@date: 09/02/2020
"""
# pylint:disable=too-many-arguments,too-many-ancestors
# pylint:disable=arguments-differ,invalid-name
from __future__ import absolute_import, division, print_function

from typing import List, Optional, Tuple, Union, Callable

import tensorflow as tf

from tensorflow.keras.layers import InputSpec
from tensorflow.python.keras.utils import conv_utils
from network.gauge_network import vs_init

from utils.attr_dict import AttrDict
from network.gauge_network import GaugeNetwork, NetworkConfig

layers = tf.keras.layers


def periodic_image(x, size):
    """Apply wrapped padding (periodic boundary conditions) to `input`.

    NOTE: The input is expected to have shape (N, H, W, C), then this applies
    periodic boundary conditions along the (H, W) axes.
    """
    x = tf.concat([x[:, -size:, :, :], x, x[:, 0:size, :, :]], 1)
    x = tf.concat([x[:, :, -size:, :], x, x[:, :, 0:size, :]], 2)
    return x


class ConvolutionConfig(AttrDict):
    """Defines a configuration object for passing to `ConvolutionBlock2D`."""
    def __init__(
            self,
            input_shape: List[int],  # expected input shape
            filters: List[int],      # number of filters to use
            sizes: List[int],        # filter sizes to use
            pool_sizes: Optional[List[int]] = None,  # MaxPooling2D sizes
            conv_activations: Optional[List[str]] = None,  # Activation fns
            conv_paddings: Optional[List[str]] = None,  # Paddings to use
            use_batch_norm: Optional[bool] = False,  # Use batch normalization?
            name: Optional[str] = None,  # Name of model
    ):
        super(ConvolutionConfig, self).__init__(
            input_shape=input_shape,
            filters=filters,
            sizes=sizes,
            pool_sizes=pool_sizes,
            conv_activations=conv_activations,
            conv_paddings=conv_paddings,
            use_batch_norm=use_batch_norm,
            name=name,
        )


class ConvolutionBlock2D(layers.Layer):
    """Implements a block consisting of: 2 x [Conv2D, MaxPooling2D]."""
    def __init__(self, config: ConvolutionConfig, **kwargs):
        super(ConvolutionBlock2D, self).__init__(**kwargs)
        self._config = config
        if config.pool_sizes is None:
            config.pool_sizes = 2 * [(2, 2)]
        if config.conv_activations is None:
            config.conv_activations = 2 * ['relu']
        if isinstance(config.conv_paddings, str):
            config.conv_paddings = 2 * [config.conv_paddings]

        self.conv1 = layers.Conv2D(
            filters=config.filters[0],
            kernel_size=config.sizes[0],
            activation=config.conv_activations[0],
            input_shape=config.input_shape,
            padding=config.conv_paddings[0],
            name='conv1'
        )
        self.pool1 = layers.MaxPooling2D(config.pool_sizes[0], name='pool1')

        self.conv2 = layers.Conv2D(
            filters=config.filters[1],
            kernel_size=config.sizes[1],
            activation=config.conv_activations[1],
            padding=config.conv_paddings[0],
            name='conv2',
        )
        self.pool2 = layers.MaxPooling2D(config.pool_sizes[1], name='pool2')

        self.flatten = layers.Flatten(name='flatten')

        if config.use_batch_norm:
            self.batch_norm = layers.BatchNormalization(axis=-1,
                                                        name='batch_norm')

    def call(self, inputs, training=None):
        inputs = tf.reshape(inputs, (-1, *self._config.input_shape))
        inputs = periodic_image(inputs, self._config.sizes[0] - 1)
        y1 = self.pool1(self.conv1(inputs))
        y2 = self.flatten(self.pool2(self.conv2(y1)))
        if self._config.use_batch_norm:
            y2 = self.batch_norm(y2, training=training)

        return y2


class GaugeNetworkConv2D(tf.keras.models.Model):
    """"Prepends a conv. structure at the beginning of `GaugeNetwork`."""
    def __init__(
            self,
            conv_config: ConvolutionConfig,
            config: NetworkConfig,
            xdim: int,
            factor: Optional[float] = 1.,
            kernel_initializer: Optional[Union[str, Callable]] = None,
            **kwargs,
    ):
        super(GaugeNetworkConv2D, self).__init__(**kwargs)
        self._kernel_initializer = kernel_initializer
        with tf.name_scope('x_conv_block'):
            self.x_conv_block = ConvolutionBlock2D(conv_config, **kwargs)
        #  with tf.name_scope('v_conv_block'):
        #      self.v_conv_block = ConvolutionBlock2D(conv_config, **kwargs)
        with tf.name_scope('GaugeNetwork'):
            self.gauge_net = GaugeNetwork(
                config, xdim, factor,
                kernel_initializer=kernel_initializer)

    def _get_kern_init(self, factor=1.):
        if self._kernel_initializer == 'zeros':
            return 'zeros'
        return vs_init(factor)

    def call(self, inputs, training=None):
        """Call the network (forward-pass)."""
        #  v, x, t = inputs
        x, v, t = inputs
        # pylint:disable=protected-access
        #  v_conv = self.v_conv_block(v, training=training)
        x_conv = self.x_conv_block(x, training=training)
        return self.gauge_net((x_conv, v, t), training=training)
