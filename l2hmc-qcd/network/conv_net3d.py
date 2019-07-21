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


# pylint:disable=too-many-arguments, too-many-instance-attributes
class ConvNet3D(tf.keras.Model):
    """Conv. neural net with different initialization scale based on input."""

    def __init__(self, model_name, **kwargs):
        """Initialization method.

        Attributes:
            coeff_scale: Multiplicative factor (lambda_s in original paper p.
                13) multiplying tanh(W_s h_2 + b_s).
            coeff_transformation: Multiplicative factor (lambda_q in original
                paper p. 13) multiplying tanh(W_q h_2 + b_q).
            data_format: String (either 'channels_first' or 'channels_last').
                'channels_first' ('channels_last') is default for GPU (CPU).
                This value is automatically determined and set to the
                appropriate value.
        """
        super(ConvNet3D, self).__init__(name=model_name)

        for key, val in kwargs.items():
            setattr(self, key, val)

        if self.use_bn:
            if self.data_format == 'channels_first':
                self.bn_axis = 1
                self._data_format = 'NCHW'
            elif self.data_format == 'channels_last':
                self.bn_axis = -1
                self._data_format = 'NHWC'
            else:
                raise AttributeError("Expected 'data_format' to be "
                                     "'channels_first'  or 'channels_last'")
            #  with tf.variable_scope('batch_norm_x'):
                #  self.batch_norm_x = tf.keras.layers.BatchNormalization(
                #      axis=self.bn_axis, trainable=True
                #  )
            #  with tf.variable_scope('batch_norm_v'):
                #  self.batch_norm_v = tf.keras.layers.BatchNormalization(
                #      axis=self.bn_axis, trainable=True
                #  )
            #  self.batch_norm_v = tf.keras.layers.BatchNormalization(
            #      axis=self.bn_axis
            #  )
            #  with tf.variable_scope('batch_norm_v'):
            #  self.batch_norm_v = tf.layers.BatchNormalization(
            #      axis=self.bn_axis,
            #  )
            #  with tf.variable_scope('batch_norm_x'):
            #  self.batch_norm_x = tf.layers.BatchNormalization(
            #      axis=self.bn_axis
            #  )
            #  add_elements_to_collection(self.batch_norm_v.updates,
            #                             tf.GraphKeys.UPDATE_OPS)
            #  self.batch_norm_x = tf.keras.layers.BatchNormalization(
            #      axis=self.bn_axis
            #  )
            #  add_elements_to_collection(self.batch_norm_x.updates,
            #                             tf.GraphKeys.UPDATE_OPS)

        if self.dropout_prob > 0:
            self.dropout_x = tf.keras.layers.Dropout(self.dropout_prob,
                                                     seed=GLOBAL_SEED)
            self.dropout_v = tf.keras.layers.Dropout(self.dropout_prob,
                                                     seed=GLOBAL_SEED)

        #  with tf.name_scope('coeff_scale'):
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

            #  with tf.name_scope('conv_layers'):
            #  with tf.name_scope('conv_x1'):
            self.conv_x1 = tf.keras.layers.Conv3D(
                filters=self.num_filters,
                kernel_size=self.filter_sizes[0],
                activation=tf.nn.relu,
                #  initializer=HE_INIT,
                input_shape=self._input_shape,
                padding='same',
                name='conv_x1',
                dtype=TF_FLOAT,
                data_format=self.data_format
            )

            #  with tf.name_scope('pool_x1'):
            self.max_pool_x1 = tf.keras.layers.MaxPooling3D(
                pool_size=(2, 2, 2),
                strides=2,
                padding='same',
                name='pool_x1',
            )

            #  with tf.name_scope('conv_v1'):
            self.conv_v1 = tf.keras.layers.Conv3D(
                filters=self.num_filters,
                kernel_size=self.filter_sizes[0],
                activation=tf.nn.relu,
                #  initializer=HE_INIT,
                input_shape=self._input_shape,
                padding='same',
                name='conv_v1',
                dtype=TF_FLOAT,
                data_format=self.data_format
            )

            #  with tf.name_scope('pool_v1'):
            self.max_pool_v1 = tf.keras.layers.MaxPooling3D(
                pool_size=(2, 2, 2),
                strides=2,
                padding='same',
                name='pool_v1'
            )

            #  with tf.name_scope('conv_x2'):
            self.conv_x2 = tf.keras.layers.Conv3D(
                filters=2*self.num_filters,
                kernel_size=self.filter_sizes[1],
                activation=None if self.use_bn else tf.nn.relu,
                #  initializer=HE_INIT,
                padding='same',
                name='conv_x2',
                dtype=TF_FLOAT,
                data_format=self.data_format
            )

            #  with tf.name_scope('pool_x2'):
            self.max_pool_x2 = tf.keras.layers.MaxPooling3D(
                pool_size=(2, 2, 2),
                strides=2,
                padding='same',
                name='pool_x2'
            )

            #  with tf.name_scope('conv_v2'):
            self.conv_v2 = tf.keras.layers.Conv3D(
                filters=2 * self.num_filters,
                kernel_size=self.filter_sizes[1],
                activation=None if self.use_bn else tf.nn.relu,
                #  initializer=HE_INIT,
                padding='same',
                name='conv_v2',
                dtype=TF_FLOAT,
                data_format=self.data_format
            )

            #  with tf.name_scope('pool_v2'):
            self.max_pool_v2 = tf.keras.layers.MaxPooling3D(
                pool_size=(2, 2, 2),
                strides=2,
                padding='same',
                name='pool_v2'
            )

            #  with tf.name_scope('fc_layers'):
            #  with tf.name_scope('flatten'):
            self.flatten = tf.keras.layers.Flatten(
                data_format=self.data_format,
                name='flatten'
            )

            #  with tf.name_scope('x_layer'):
            self.x_layer = custom_dense(self.num_hidden,
                                        self.factor/3.,
                                        name='x_layer')

            #  with tf.name_scope('v_layer'):
            self.v_layer = custom_dense(self.num_hidden,
                                        1./3.,
                                        name='v_layer')

            #  with tf.name_scope('t_layer'):
            self.t_layer = custom_dense(self.num_hidden,
                                        1./3.,
                                        name='t_layer')

            #  with tf.name_scope('h_layer'):
            self.h_layer = custom_dense(self.num_hidden,
                                        name='h_layer')

            #  with tf.name_scope('scale_layer'):
            self.scale_layer = custom_dense(
                self.x_dim, 0.001, name='scale_layer'
            )

            #  with tf.name_scope('translation_layer'):
            self.translation_layer = custom_dense(
                self.x_dim, 0.001, 'translation_layer'
            )

            #  with tf.name_scope('transformation_layer'):
            self.transformation_layer = custom_dense(
                self.x_dim, 0.001, 'transformation_layer'
            )

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

    # pylint: disable=invalid-name, arguments-differ
    def call(self, inputs, train_phase):
        """Forward pass through network.

        Args:
            input (list or tuple): Inputs to the network (x, v, t).
            train_phase (bool or tf.placeholder): Run the network in
            either `training` phase or `inference` phase.

       Returns:
           scale, translation, transformation (S, T, Q functions from paper)
        """
        v, x, t = inputs

        #  assert train_phase == K.learning_phase()
        #  io.log(f'K.learning_phase(): {K.learning_phase()}')
        #  io.log(f'train_phase: {train_phase}')

        #  with tf.name_scope('reshape'):
        v = self.reshape_5D(v)
        x = self.reshape_5D(x)

        with tf.name_scope('x_layers'):
            x = self.max_pool_x1(self.conv_x1(x))
            #  x = self.max_pool_x2(self.conv_x2(x))
            #  x = batch_norm(x, axis=self.bn_axis, is_training=train_phase)
            #  x = self.batch_norm_x(x, training=train_phase)
            x = self.conv_x2(x)
            if self.use_bn:
                x = batch_norm(x, train_phase,
                               axis=self.bn_axis,
                               internal_update=True)
                #  x = self.batch_norm_x(x, training=train_phase)
                #  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                #                       self.batch_norm_x.updates)
                #  x = self.batch_norm_x(x, training=train_phase)
                #  x = tf.contrib.layers.batch_norm(x, is_training=train_phase,
                #                                   data_format=self._data_format,
                #                                   updates_collections=None)
            x = tf.nn.relu(x)
            x = self.max_pool_x2(x)
            x = self.flatten(x)
            if self.dropout_prob > 0:
                x = self.dropout_x(x, training=train_phase)
            #  x = self.x_layer(x)
            x = tf.nn.relu(self.x_layer(x))

        with tf.name_scope('v_layers'):
            v = self.max_pool_v1(self.conv_v1(v))
            #  v = self.max_pool_v2(self.conv_v2(v))
            v = self.conv_v2(v)
            #  v = batch_norm(v, axis=self.bn_axis, is_training=train_phase)
            #  v = batch_norm(v, axis=self.bn_axis, is_training=train_phase)
            #  v = self.batch_norm_v(v, training=train_phase)
            #  v = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(v)
            if self.use_bn:
                v = batch_norm(v, train_phase,
                               axis=self.bn_axis,
                               internal_update=True)
                #  v = self.batch_norm_v(v, training=train_phase)
                #  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                #                       self.batch_norm_v.updates)
                #  v = self.batch_norm_v(v, training=train_phase)
                #  v = tf.contrib.layers.batch_norm(v, is_training=train_phase,
                #                                   data_format=self._data_format,
                #                                   updates_collections=None)
            v = tf.nn.relu(v)
            #  v = self.v_layer(v)
            v = self.max_pool_v2(v)
            if self.dropout_prob > 0:
                v = self.dropout_v(v, training=train_phase)
            v = self.flatten(v)
            v = tf.nn.relu(self.v_layer(v))

        with tf.name_scope('t_layer'):
            #  t = self.t_layer(t)
            t = tf.nn.relu(self.t_layer(t))

        def reshape(t, name):
            return tf.squeeze(
                tf.reshape(t, shape=self._input_shape, name=name)
            )

        with tf.name_scope('generic_layers'):
            h = tf.nn.relu(v + x + t)
            h = tf.nn.relu(self.h_layer(h))
            #  h = tf.nn.relu(self.h_layer1(h))

            with tf.name_scope('translation'):
                translation = self.translation_layer(h)

            with tf.name_scope('scale'):
                scale = (tf.nn.tanh(self.scale_layer(h))
                         * tf.exp(self.coeff_scale))

            with tf.name_scope('transformation'):
                transformation = (self.transformation_layer(h)
                                  * tf.exp(self.coeff_transformation))

        return scale, translation, transformation

