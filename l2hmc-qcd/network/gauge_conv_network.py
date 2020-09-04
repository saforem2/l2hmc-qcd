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

from typing import List, Optional, Tuple, Union

import tensorflow as tf

from tensorflow.keras.layers import InputSpec
from tensorflow.python.keras.utils import conv_utils

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
            input_shape: Union[List[int], Tuple[int]],
            filters: Union[List[int], Tuple[int]],
            sizes: Union[List[int], Tuple[int]],
            pool_sizes: Optional[Union[List[int], Tuple[int]]] = None,
            conv_activations: Optional[Union[List[str], Tuple[str]]] = None,
            conv_paddings: Optional[Union[Tuple[str], str]] = None,
            use_batch_norm: Optional[bool] = False,
            name: Optional[str] = None,
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
        )
        self.pool1 = layers.MaxPooling2D(config.pool_sizes[0])

        self.conv2 = layers.Conv2D(
            filters=config.filters[1],
            kernel_size=config.sizes[1],
            activation=config.conv_activations[1],
            padding=config.conv_paddings[0],
        )
        self.pool2 = layers.MaxPooling2D(config.pool_sizes[1])

        self.flatten = layers.Flatten()

        if config.use_batch_norm:
            self.batch_norm = layers.BatchNormalization(axis=-1)

    def call(self, inputs, training=None):
        inputs = tf.reshape(inputs, (-1, *self._config.input_shape))
        inputs = periodic_image(inputs, self._config.sizes[0] - 1)
        y1 = self.pool1(self.conv1(inputs))
        y2 = self.flatten(self.pool2(self.conv2(y1)))
        if self._config.use_batch_norm:
            y2 = self.batch_norm(y2, training=training)

        return y2


class GaugeNetworkConv2D(layers.Layer):
    """"Prepends a conv. structure at the beginning of `GaugeNetwork`."""
    def __init__(
            self,
            conv_config: ConvolutionConfig,
            config: NetworkConfig,
            xdim: int,
            factor: Optional[float] = 1.,
            **kwargs,
    ):
        super(GaugeNetworkConv2D, self).__init__(**kwargs)
        self.x_conv_block = ConvolutionBlock2D(conv_config, **kwargs)
        self.v_conv_block = ConvolutionBlock2D(conv_config, **kwargs)
        self.gauge_net = GaugeNetwork(config, xdim, factor, **kwargs)

    def call(self, inputs, training=None):
        """Call the network (forward-pass)."""
        v, x, t = inputs
        # pylint:disable=protected-access
        x_conv = self.x_conv_block(x, training=training)
        v_conv = self.v_conv_block(v, training=training)
        return self.gauge_net((v_conv, x_conv, t), training=training)
