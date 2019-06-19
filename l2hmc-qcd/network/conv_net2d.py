"""
conv_net2d.py

Implements a convolutional neural network using 2D convolutions.

Author: Sam Foreman (github: @saforem2)
Date: 06/14/2019
"""
import numpy as np
import tensorflow as tf

from globals import GLOBAL_SEED, TF_FLOAT
from .network_utils import custom_dense


np.random.seed(GLOBAL_SEED)

if '2.' not in tf.__version__:
    tf.set_random_seed(GLOBAL_SEED)


class ConvNet2D(tf.keras.Model):
    """Conv. neural net with different initialization scale based on input."""

    def __init__(self, model_name, **kwargs):
        super(ConvNet2D, self).__init__(name=model_name)

        for key, val in kwargs.items():
            setattr(self, key, val)

        if self.use_bn:
            if self.data_format == 'channels_first':
                self.bn_axis = 1
            elif self.data_format == 'channels_last':
                self.bn_axis = -1
            else:
                raise AttributeError("Expected 'data_format' to be "
                                     "'channels_first'  or 'channels_last'")

        with tf.name_scope(self.name_scope):
            with tf.name_scope('coeff_scale'):
                self.coeff_scale = tf.Variable(
                    initial_value=tf.zeros([1, self.x_dim]),
                    name='coeff_scale',
                    trainable=True,
                    dtype=TF_FLOAT
                )

            with tf.name_scope('coeff_transformation'):
                self.coeff_transformation = tf.Variable(
                    initial_value=tf.zeros([1, self.x_dim]),
                    name='coeff_transformation',
                    trainable=True,
                    dtype=TF_FLOAT
                )

            with tf.name_scope('conv_layers'):
                with tf.name_scope('conv_x1'):
                    self.conv_x1 = tf.keras.layers.Conv2D(
                        filters=self.num_filters,
                        kernel_size=self.filter_sizes[0],
                        activation=tf.nn.relu,
                        input_shape=self._input_shape[1:],
                        #  padding='same',
                        name='conv_x1',
                        dtype=TF_FLOAT,
                        data_format=self.data_format

                    )

                with tf.name_scope('pool_x1'):
                    self.max_pool_x1 = tf.keras.layers.MaxPooling2D(
                        pool_size=(2, 2),
                        strides=2,
                        #  padding='same',
                        name='pool_x1',
                    )

                with tf.name_scope('conv_v1'):
                    self.conv_v1 = tf.keras.layers.Conv2D(
                        filters=self.num_filters,
                        kernel_size=self.filter_sizes[0],
                        activation=tf.nn.relu,
                        input_shape=self._input_shape[1:],
                        #  padding='same',
                        name='conv_v1',
                        dtype=TF_FLOAT,
                        data_format=self.data_format
                    )

                with tf.name_scope('pool_v1'):
                    self.max_pool_v1 = tf.keras.layers.MaxPooling2D(
                        pool_size=(2, 2),
                        strides=2,
                        #  padding='same',
                        name='pool_v1'
                    )

                with tf.name_scope('conv_x2'):
                    self.conv_x2 = tf.keras.layers.Conv2D(
                        filters=2*self.num_filters,
                        kernel_size=self.filter_sizes[1],
                        activation=tf.nn.relu,
                        #  padding='same',
                        name='conv_x2',
                        dtype=TF_FLOAT,
                        data_format=self.data_format
                    )

                with tf.name_scope('pool_x2'):
                    self.max_pool_x2 = tf.keras.layers.MaxPooling2D(
                        pool_size=(2, 2),
                        strides=2,
                        #  padding='same',
                        name='pool_x2'
                    )

                with tf.name_scope('conv_v2'):
                    self.conv_v2 = tf.keras.layers.Conv2D(
                        filters=2*self.num_filters,
                        kernel_size=self.filter_sizes[1],
                        activation=tf.nn.relu,
                        #  padding='same',
                        name='conv_v2',
                        dtype=TF_FLOAT,
                        data_format=self.data_format
                    )

                with tf.name_scope('pool_v2'):
                    self.max_pool_v2 = tf.keras.layers.MaxPooling2D(
                        pool_size=(2, 2),
                        strides=2,
                        #  padding='same',
                        name='pool_v2'
                    )

            with tf.name_scope('fc_layers'):
                with tf.name_scope('flatten'):
                    self.flatten = tf.keras.layers.Flatten(name='flatten')

                with tf.name_scope('x_layer'):
                    self.x_layer = custom_dense(self.num_hidden,
                                                self.factor/3.,
                                                name='x_layer')

                with tf.name_scope('v_layer'):
                    self.v_layer = custom_dense(self.num_hidden,
                                                1./3.,
                                                name='v_layer')

                with tf.name_scope('t_layer'):
                    self.t_layer = custom_dense(self.num_hidden,
                                                1./3.,
                                                name='t_layer')

                with tf.name_scope('h_layer'):
                    self.h_layer = custom_dense(self.num_hidden,
                                                name='h_layer')

                with tf.name_scope('scale_layer'):
                    self.scale_layer = custom_dense(
                        self.x_dim, 0.001, name='scale_layer'
                    )

                with tf.name_scope('translation_layer'):
                    self.translation_layer = custom_dense(
                        self.x_dim, 0.001, 'translation_layer'
                    )

                with tf.name_scope('transformation_layer'):
                    self.transformation_layer = custom_dense(
                        self.x_dim, 0.001, 'transformation_layer'
                    )

    def _reshape(self, tensor):
        """Reshape tensor to be compatible with tf.keras.layers.Conv2D."""
        if self.data_format == 'channels_first':
            # batch_size, num_channels, height, width
            N, D, H, W = self._input_shape
            #  N, D, H, W = tensor.shape
            if isinstance(tensor, np.ndarray):
                return np.reshape(tensor, (N, D, H, W))

            return tf.reshape(tensor, (N, D, H, W))

        if self.data_format == 'channels_last':
            N, H, W, D = self._input_shape
            if isinstance(tensor, np.ndarray):
                return np.reshape(tensor, (N, H, W, D))

            return tf.reshape(tensor, (N, H, W, D))

        raise AttributeError("`self.data_format` should be one of "
                             "'channels_first' or 'channels_last'")

    def call(self, inputs):
        """Forward pass through the network."""
        v, x, t = inputs

        with tf.name_scope('reshape'):
            v = self._reshape(v)
            x = self._reshape(x)

        with tf.name_scope('x'):
            x = self.max_pool_x1(self.conv_x1(x))
            x = self.max_pool_x2(self.conv_x2(x))
            if self.use_bn:
                x = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(x)
            x = tf.nn.relu(self.x_layer(self.flatten(x)))

        with tf.name_scope('v'):
            v = self.max_pool_v1(self.conv_v1(v))
            v = self.max_pool_v2(self.conv_v2(v))
            if self.use_bn:
                v = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(v)
            v = tf.nn.relu(self.v_layer(self.flatten(v)))
            #  v = self.flatten(v)
            #  v = tf.nn.relu(self.v_layer(v))

        with tf.name_scope('t'):
            t = tf.nn.relu(self.t_layer(t))

        with tf.name_scope('h'):
            h = tf.nn.relu(v + x + t)
            h = tf.nn.relu(self.h_layer(h))

        with tf.name_scope('scale'):
            scale = (self.coeff_scale
                     * tf.nn.tanh(self.scale_layer(h)))

        with tf.name_scope('transformation'):
            transformation = (self.coeff_transformation
                              * tf.nn.tanh(self.transformation_layer(h)))

        with tf.name_scope('translation'):
            translation = self.translation_layer(h)

        #  with tf.name_scope('scale'):
        #      scale = (tf.nn.tanh(self.scale_layer(h))
        #               * tf.exp(self.coeff_scale))
        #
        #  with tf.name_scope('transformation'):
        #      transformation = (tf.nn.tanh(self.transformation_layer(h))
        #                        * tf.exp(self.coeff_transformation))

        return scale, translation, transformation
