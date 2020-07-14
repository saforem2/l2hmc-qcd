"""
conv_net.py

Implements both 2D and 3D convolutional neural networks.


Author: Sam Foreman (github: @saforem2)
Date: 06/14/2019
"""
from __future__ import absolute_import, division, print_function

import pickle

from collections import namedtuple

import numpy as np
import tensorflow as tf

from config import TF_FLOAT, Weights
from network.layers import batch_norm
from utils.seed_dict import seeds

np.random.seed(seeds['global_np'])

if tf.__version__.startswith('1.'):
    tf.set_random_seed(seeds['global_tf'])


PoolConfig = namedtuple('PoolConfig', ['pool_sizes', 'strides'])

ConvConfig = namedtuple('ConvConfig', [
    'filters', 'kernel_sizes', 'strides', 'padding', 'activation'
])


class ConvNet2D(tf.keras.layers.Layer):
    """Block of 2D convolutional layers."""

    def __init__(self,
                 config: ConvConfig,
                 pool_config: PoolConfig,
                 input_shape: tuple,
                 lattice_shape: tuple,
                 data_format: str = 'channels_last',
                 use_batch_norm: bool = False) -> None:
        """Initialization method."""
        super(ConvNet2D, self).__init__()
        self.config = config
        self.pool_config = pool_config
        self.data_format = data_format
        self.input_shape = input_shape
        self.lattice_shape = lattice_shape
        self.use_batch_norm = use_batch_norm
        self.conv1 = tf.keras.layers.Conv2D(
            name='ConvLayer1',
            filters=self.config.filters[0],
            kernel_size=self.config.kernel_sizes[0],
            strides=self.config.strides[0],
            activation=self.config.activation,
            padding=self.config.padding,
            data_format=self.data_format,
            input_shape=self.lattice_shape
        )

        self.conv2 = tf.keras.layers.Conv2D(
            name='ConvLayer2',
            filters=self.config.filters[1],
            kernel_size=self.config.kernel_sizes[1],
            strides=self.config.strides[1],
            activation=self.config.activation,
            padding=self.config.padding,
            data_format=self.data_format
        )

        self.pool1 = tf.keras.layers.MaxPooling2D(
            name='MaxPool1',
            pool_size=self.pool_config.pool_sizes[0],
            strides=self.pool_config.pool_sizes[1],
        )

        self.pool1 = tf.keras.layers.MaxPooling2D(
            name='MaxPool1',
            pool_size=self.pool_config.pool_sizes[0],
            strides=self.pool_config.pool_sizes[1],
        )

    def reshape_4D(self, tensor):
        """
        Reshape tensor to be compatible with tf.keras.layers.Conv2D.

        If self.data_format is 'channels_first', and input `tensor` has shape
        (N, 2, L, L), the output tensor has shape (N, 1, 2, L, L).

        If self.data_format is 'channels_last' and input `tensor` has shape
        (N, L, L, 2), the output tensor has shape (N, 2, L, L, 1).
        """
        N, H, W, C = self._input_shape
        if self.data_format == 'channels_first':
            #  N, C, H, W = self._input_shape
            #  N, D, H, W = tensor.shape
            if isinstance(tensor, np.ndarray):
                return np.reshape(tensor, (N, C, H, W))

            return tf.reshape(tensor, (N, C, H, W))

        if self.data_format == 'channels_last':
            #  N, H, W, C = self._input_shape
            if isinstance(tensor, np.ndarray):
                return np.reshape(tensor, (N, H, W, C))

            return tf.reshape(tensor, (N, H, W, C))

        raise AttributeError("`self.data_format` should be one of "
                             "'channels_first' or 'channels_last'")

    def call(self, inputs, training=None):
        """Call the layer (forward-pass)."""


class ConvNet2D_1(tf.keras.Model):
    """Convolutional block used in ConvNet3D."""

    def __init__(self, model_name, **kwargs):
        """
        Initialization method.

        Args:
            model_name: Name of the model.
            kwargs: Keyword arguments used to specify specifics of
                convolutional structure.
        """
        super(ConvNet2D, self).__init__(name=model_name)

        self.data_format = 'channels_last'
        self.activation = kwargs.get('conv_activation', tf.nn.relu)
        self.num_filters = kwargs.get('num_filters', [32, 32])
        self.filter_sizes = kwargs.get('filter_sizes', [4, 4])

        for key, val in kwargs.items():
            setattr(self, key, val)

        self.activation = kwargs.get('conv_act', tf.nn.relu)
        activation2 = None if self.use_bn else self.activation

        if model_name == 'ConvNet2Dx':
            self.bn_name = 'batch_norm_x'
        elif model_name == 'ConvNet2Dv':
            self.bn_name = 'batch_norm_v'

        if self.name_scope == 'ConvNetX':
            conv1_name = 'conv_x1'
            pool1_name = 'pool_x1'
            conv2_name = 'conv_x2'
            pool2_name = 'pool_x2'
        elif self.name_scope == 'ConvNetV':
            conv1_name = 'conv_v1'
            pool1_name = 'pool_v1'
            conv2_name = 'conv_v2'
            pool2_name = 'pool_v2'

        with tf.name_scope(self.name_scope):

            self.conv1 = tf.keras.layers.Conv2D(
                filters=self.num_filters[0],
                kernel_size=self.filter_sizes[0],
                activation=self.activation,
                #  input_shape=self._input_shape[1:],
                name=conv1_name,
                padding='same',
                dtype=TF_FLOAT,
                data_format=self.data_format
            )

            #  with tf.name_scope('pool_x1'):
            self.max_pool1 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=2,
                #  padding='same',
                name=pool1_name,
            )

            self.conv2 = tf.keras.layers.Conv2D(
                filters=self.num_filters[1],
                kernel_size=self.filter_sizes[1],
                activation=activation2,
                padding='same',
                name=conv2_name,
                dtype=TF_FLOAT,
                data_format=self.data_format
            )

            self.max_pool2 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=2,
                #  padding='same',
                name=pool2_name,
            )

            self.flatten = tf.keras.layers.Flatten(name='flatten')

        self.layers_dict = {
            'conv1': self.conv1,
            'conv2': self.conv2,
        }

    def reshape_4D(self, tensor):
        """
        Reshape tensor to be compatible with tf.keras.layers.Conv3D.

        If self.data_format is 'channels_first', and input `tensor` has shape
        (N, 2, L, L), the output tensor has shape (N, 1, 2, L, L).

        If self.data_format is 'channels_last' and input `tensor` has shape
        (N, L, L, 2), the output tensor has shape (N, 2, L, L, 1).
        """
        N, H, W, C = self._input_shape
        if self.data_format == 'channels_first':
            #  N, C, H, W = self._input_shape
            #  N, D, H, W = tensor.shape
            if isinstance(tensor, np.ndarray):
                return np.reshape(tensor, (N, C, H, W))

            return tf.reshape(tensor, (N, C, H, W))

        if self.data_format == 'channels_last':
            #  N, H, W, C = self._input_shape
            if isinstance(tensor, np.ndarray):
                return np.reshape(tensor, (N, H, W, C))

            return tf.reshape(tensor, (N, H, W, C))

        raise AttributeError("`self.data_format` should be one of "
                             "'channels_first' or 'channels_last'")

    def get_layer_weights(self, sess):
        """Extract numerical values of all layer weights."""
        weights_dict = {}
        for name, layer in self.layers_dict.items():
            weights_dict[name] = {}
            if isinstance(layer, dict):
                for subname, sublayer in layer.items():
                    w, b = sess.run(sublayer.weights)
                    weights_dict[name][subname] = Weights(w=w, b=b)
            else:
                w, b = sess.run(layer.weights)
                weights_dict[name] = Weights(w=w, b=b)

        return weights_dict

    def save_layer_weights(self, sess, out_file):
        """Save all layer weights to `out_file`."""
        weights_dict = self.get_layer_weights(sess)
        with open(out_file, 'wb') as f:
            pickle.dump(weights_dict, f)

    def call(self, inputs, train_phase):
        """Forward pass through the network."""
        inputs = self.reshape_4D(inputs)

        inputs = self.max_pool1(self.conv1(inputs))
        inputs = self.conv2(inputs)
        if self.use_bn:
            inputs = self.activation(batch_norm(inputs, train_phase,
                                                axis=self.bn_axis,
                                                internal_update=True,
                                                scope=self.bn_name,
                                                reuse=tf.AUTO_REUSE))
        inputs = self.max_pool2(inputs)
        inputs = self.flatten(inputs)

        return inputs


class ConvNet3D(tf.keras.Model):
    """Convolutional block used in ConvNet3D."""

    def __init__(self, model_name, **kwargs):
        """
        Initialization method.

        Args:
            model_name: Name of the model.
            kwargs: Keyword arguments used to specify specifics of
                convolutional structure.
        """
        super(ConvNet3D, self).__init__(name=model_name)

        self.data_format = 'channels_last'

        for key, val in kwargs.items():
            setattr(self, key, val)

        if model_name == 'ConvNet3Dx':
            self.bn_name = 'batch_norm_x'
        elif model_name == 'ConvNet3Dv':
            self.bn_name = 'batch_norm_v'

        if self.name_scope == 'x_conv_block':
            conv1_name = 'conv_x1'
            pool1_name = 'pool_x1'
            conv2_name = 'conv_x2'
            pool2_name = 'pool_x2'
        elif self.name_scope == 'v_conv_block':
            conv1_name = 'conv_v1'
            pool1_name = 'pool_v1'
            conv2_name = 'conv_v2'
            pool2_name = 'pool_v2'

        with tf.name_scope(self.name_scope):
            if self.use_bn:
                if self.data_format == 'channels_first':
                    self.bn_axis = 1
                elif self.data_format == 'channels_last':
                    self.bn_axis = -1
                else:
                    raise AttributeError("Expected 'data_format' "
                                         "to be 'channels_first' "
                                         "or 'channels_last'.")

            self.activation = kwargs.get('conv_act', tf.nn.relu)
            activation2 = None if self.use_bn else self.activation

            self.conv1 = tf.keras.layers.Conv3D(
                filters=self.num_filters[0],
                kernel_size=self.filter_sizes[0],
                activation=self.activation,
                #  input_shape=self._input_shape,
                padding='same',
                name=conv1_name,
                dtype=TF_FLOAT,
                data_format=self.data_format
            )

            self.max_pool1 = tf.keras.layers.MaxPooling3D(
                pool_size=(2, 2, 2),
                strides=2,
                padding='same',
                name=pool1_name,
            )

            self.conv2 = tf.keras.layers.Conv3D(
                filters=self.num_filters[1],
                kernel_size=self.filter_sizes[1],
                activation=activation2,
                #  activation=None if self.use_bn else tf.nn.relu,
                #  initializer=HE_INIT,
                padding='same',
                name=conv2_name,
                dtype=TF_FLOAT,
                data_format=self.data_format
            )

            self.max_pool2 = tf.keras.layers.MaxPooling3D(
                pool_size=(2, 2, 2),
                strides=2,
                padding='same',
                name=pool2_name,
            )

            self.flatten = tf.keras.layers.Flatten(name='flatten')

        self.layers_dict = {
            'conv1': self.conv1,
            'conv2': self.conv2,
        }

    def reshape_5D(self, tensor):
        """
        Reshape tensor to be compatible with tf.keras.layers.Conv3D.

        If self.data_format is 'channels_first', and input `tensor` has shape
        (N, 2, L, L), the output tensor has shape (N, 1, 2, L, L).

        If self.data_format is 'channels_last' and input `tensor` has shape
        (N, L, L, 2), the output tensor has shape (N, 2, L, L, 1).
        """
        N, H, W, C = self._input_shape
        if self.data_format == 'channels_first':
            #  N, C, H, W = self._input_shape
            #  N, D, H, W = tensor.shape
            if isinstance(tensor, np.ndarray):
                return np.reshape(tensor, (N, 1, C, H, W))

            return tf.reshape(tensor, (N, 1, C, H, W))

        if self.data_format == 'channels_last':
            #  N, H, W, C = self._input_shape
            if isinstance(tensor, np.ndarray):
                return np.reshape(tensor, (N, H, W, C, 1))

            return tf.reshape(tensor, (N, H, W, C, 1))

        raise AttributeError("`self.data_format` should be one of "
                             "'channels_first' or 'channels_last'")

    def get_layer_weights(self, sess):
        """Extract numerical values of all layer weights."""
        weights_dict = {}
        for name, layer in self.layers_dict.items():
            weights_dict[name] = {}
            if isinstance(layer, dict):
                for subname, sublayer in layer.items():
                    w, b = sess.run(sublayer.weights)
                    weights_dict[name][subname] = Weights(w=w, b=b)
            else:
                w, b = sess.run(layer.weights)
                weights_dict[name] = Weights(w=w, b=b)

        return weights_dict

    def save_layer_weights(self, sess, out_file):
        """Save all layer weights to `out_file`."""
        weights_dict = self.get_layer_weights(sess)
        with open(out_file, 'wb') as f:
            pickle.dump(weights_dict, f)

    def call(self, inputs, train_phase):
        """Forward pass through the network."""
        inputs = self.reshape_5D(inputs)

        inputs = self.max_pool1(self.conv1(inputs))
        inputs = self.conv2(inputs)
        if self.use_bn:
            inputs = self.activation(batch_norm(inputs, train_phase,
                                                axis=self.bn_axis,
                                                internal_update=True,
                                                scope=self.bn_name,
                                                reuse=tf.AUTO_REUSE))
        inputs = self.max_pool2(inputs)
        inputs = self.flatten(inputs)

        return inputs
