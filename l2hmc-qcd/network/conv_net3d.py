"""
conv_net.py

Convolutional neural network architecture for running L2HMC on a gauge lattice
configuration of links.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Author: Sam Foreman (github: @saforem2)
Date: 01/16/2019
"""
import numpy as np
import tensorflow as tf
from .network_utils import (custom_dense, batch_norm,
                            add_elements_to_collection)
import utils.file_io as io
from tensorflow.keras import backend as K
#  import utils.file_io as io

from globals import GLOBAL_SEED, TF_FLOAT, NP_FLOAT


np.random.seed(GLOBAL_SEED)

if '2.' not in tf.__version__:
    tf.set_random_seed(GLOBAL_SEED)


#  HE_INIT = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

def flatten_list(_list):
    return [item for sublist in _list for item in sublist]


class ConvBlock(tf.keras.Model):
    """Convolutional block used in ConvNet3D."""
    def __init__(self, model_name, **kwargs):
        """
        Initialization method.

        Args:
            model_name: Name of the model.
            kwargs: Keyword arguments used to specify specifics of
                convolutional structure.
        """
        super(ConvBlock, self).__init__(name=model_name)

        for key, val in kwargs.items():
            setattr(self, key, val)

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

            #  if self.dropout_prob > 0:
            #      self.dropout = tf.keras.layers.Dropout(self.dropout_prob,
            #                                             seed=GLOBAL_SEED)

            self.conv1 = tf.keras.layers.Conv3D(
                filters=self.num_filters,
                kernel_size=self.filter_sizes[0],
                activation=kwargs.get('conv_act', tf.nn.relu),
                input_shape=self._input_shape,
                padding='same',
                name='conv1',
                dtype=TF_FLOAT,
                data_format=self.data_format
            )

            self.max_pool1 = tf.keras.layers.MaxPooling3D(
                pool_size=(2, 2, 2),
                strides=2,
                padding='same',
                name='pool1',
            )

            activation2 = (None if self.use_bn
                           else kwargs.get('conv_act', tf.nn.relu))

            self.conv2 = tf.keras.layers.Conv3D(
                filters=2*self.num_filters,
                kernel_size=self.filter_sizes[1],
                activation=activation2,
                #  activation=None if self.use_bn else tf.nn.relu,
                #  initializer=HE_INIT,
                padding='same',
                name='conv2',
                dtype=TF_FLOAT,
                data_format=self.data_format
            )

            self.max_pool2 = tf.keras.layers.MaxPooling3D(
                pool_size=(2, 2, 2),
                strides=2,
                padding='same',
                name='pool2',
            )

            self.flatten = tf.keras.layers.Flatten(name='flatten')

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

    def call(self, input, train_phase):
        """Forward pass through the network."""
        #  if input.shape[1:] != self._input_shape[1:]:
        #      input = tf.reshape(input, (-1, *self._input_shape[1:]))
        input = self.reshape_5D(input)

        input = self.max_pool1(self.conv1(input))
        input = self.conv2(input)
        if self.use_bn:
            input = batch_norm(input, train_phase,
                               axis=self.bn_axis,
                               internal_update=True)
        input = tf.nn.tanh(input)
        input = self.max_pool2(input)
        input = self.flatten(input)
        #  if self.dropout_prob > 0:
        #      input = self.dropout(input, training=train_phase)

        return input


class GenericNet(tf.keras.Model):
    """Generic (fully-connected) network used in training L2HMC."""
    def __init__(self, model_name, **kwargs):
        """
        Initialization method.

        Args:
            model_name: Name of the model.
            kwargs: Keyword arguments used to specify specifics of
                convolutional structure.
        """
        super(GenericNet, self).__init__(name=model_name)

        for key, val in kwargs.items():
            setattr(self, key, val)

        with tf.name_scope(self.name_scope):
            self.coeff_scale = tf.Variable(
                initial_value=tf.zeros([1, self.x_dim]),
                name='coeff_scale',
                trainable=True,
                dtype=TF_FLOAT
            )

            #  with tf.name_scope('coeff_transformation'):
            self.coeff_transformation = tf.Variable(
                initial_value=tf.zeros([1, self.x_dim]),
                name='coeff_transformation',
                trainable=True,
                dtype=TF_FLOAT
            )

            if self.dropout_prob > 0:
                self.dropout_x = tf.keras.layers.Dropout(self.dropout_prob,
                                                         seed=GLOBAL_SEED)
                self.dropout_v = tf.keras.layers.Dropout(self.dropout_prob,
                                                         seed=GLOBAL_SEED)

            x_factor = self.factor / 3.
            self.x_layer = custom_dense(self.num_hidden, x_factor, name='fc_x')
            self.v_layer = custom_dense(self.num_hidden, 1./3., name='fc_v')
            self.t_layer = custom_dense(self.num_hidden, 1./3., name='fc_t')

            self.h_layer = custom_dense(self.num_hidden,
                                        name='fc_h')

            self.scale_layer = custom_dense(
                self.x_dim, 0.001, name='fc_scale'
            )

            self.translation_layer = custom_dense(
                self.x_dim, 0.001, 'fc_translation'
            )

            self.transformation_layer = custom_dense(
                self.x_dim, 0.001, 'fc_transformation'
            )

    def call(self, inputs):
        v, x, t = inputs

        with tf.name_scope('fc_layers'):
            v = tf.nn.relu(self.v_layer(v))
            x = tf.nn.relu(self.x_layer(x))

            # dropout gets applied to the output of the previous layer
            if self.dropout_prob > 0:
                v = self.dropout_v(v)
                x = self.dropout_x(x)

            t = tf.nn.relu(self.t_layer(t))

            h = tf.nn.relu(v + x + t)
            h = tf.nn.relu(self.h_layer(h))
            #  h = tf.nn.relu(self.h_layer1(h))

            translation = self.translation_layer(h)

            scale = (tf.nn.tanh(self.scale_layer(h))
                     * tf.exp(self.coeff_scale))

            transformation = (tf.nn.tanh(self.transformation_layer(h))
                              * tf.exp(self.coeff_transformation))

        return scale, translation, transformation


class FullNet3D(tf.keras.Model):
    """Complete network used for training L2HMC model."""
    def __init__(self, model_name, **kwargs):
        """
        Initialization method.

        Args:
            model_name: Name of the model.
            kwargs: Keyword arguments used to specify specifics of
                convolutional structure.
        """
        super(FullNet3D, self).__init__(name=model_name)
        kwargs['name_scope'] = 'x_conv_block'
        self.x_conv_block = ConvBlock("ConvBlockX", **kwargs)

        kwargs['name_scope'] = 'v_conv_block'
        self.v_conv_block = ConvBlock("ConvBlockV", **kwargs)

        kwargs['name_scope'] = 'generic_block'
        self.generic_block = GenericNet("GenericNet", **kwargs)

    def call(self, inputs, train_phase):
        v, x, t = inputs

        v = self.v_conv_block(v, train_phase)
        x = self.x_conv_block(x, train_phase)

        scale, translation, transformation = self.generic_block([v, x, t])

        return scale, translation, transformation
