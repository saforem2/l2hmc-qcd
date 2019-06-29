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
from .network_utils import custom_dense

from globals import GLOBAL_SEED, TF_FLOAT, NP_FLOAT


np.random.seed(GLOBAL_SEED)

if '2.' not in tf.__version__:
    tf.set_random_seed(GLOBAL_SEED)


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
                    self.conv_x1 = tf.keras.layers.Conv3D(
                        filters=self.num_filters,
                        kernel_size=self.filter_sizes[0],
                        activation=tf.nn.relu,
                        input_shape=self._input_shape,
                        padding='same',
                        name='conv_x1',
                        dtype=TF_FLOAT,
                        data_format=self.data_format

                    )

                with tf.name_scope('pool_x1'):
                    self.max_pool_x1 = tf.keras.layers.MaxPooling3D(
                        pool_size=(2, 2, 2),
                        strides=2,
                        padding='same',
                        name='pool_x1',
                    )

                with tf.name_scope('conv_v1'):
                    self.conv_v1 = tf.keras.layers.Conv3D(
                        filters=self.num_filters,
                        kernel_size=self.filter_sizes[0],
                        activation=tf.nn.relu,
                        input_shape=self._input_shape,
                        padding='same',
                        name='conv_v1',
                        dtype=TF_FLOAT,
                        data_format=self.data_format
                    )

                with tf.name_scope('pool_v1'):
                    self.max_pool_v1 = tf.keras.layers.MaxPooling3D(
                        pool_size=(2, 2, 2),
                        strides=2,
                        padding='same',
                        name='pool_v1'
                    )

                with tf.name_scope('conv_x2'):
                    self.conv_x2 = tf.keras.layers.Conv3D(
                        filters=2*self.num_filters,
                        kernel_size=self.filter_sizes[1],
                        activation=tf.nn.relu,
                        padding='same',
                        name='conv_x2',
                        dtype=TF_FLOAT,
                        data_format=self.data_format
                    )

                with tf.name_scope('pool_x2'):
                    self.max_pool_x2 = tf.keras.layers.MaxPooling3D(
                        pool_size=(2, 2, 2),
                        strides=2,
                        padding='same',
                        name='pool_x2'
                    )

                with tf.name_scope('conv_v2'):
                    self.conv_v2 = tf.keras.layers.Conv3D(
                        filters=2 * self.num_filters,
                        kernel_size=self.filter_sizes[1],
                        activation=tf.nn.relu,
                        padding='same',
                        name='conv_v2',
                        dtype=TF_FLOAT,
                        data_format=self.data_format
                    )

                with tf.name_scope('pool_v2'):
                    self.max_pool_v2 = tf.keras.layers.MaxPooling3D(
                        pool_size=(2, 2, 2),
                        strides=2,
                        padding='same',
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

    def reshape_5D(self, tensor):
        """
        Reshape tensor to be compatible with tf.keras.layers.Conv3D.

        If self.data_format is 'channels_first', and input `tensor` has shape
        (N, 2, L, L), the output tensor has shape (N, 1, 2, L, L).

        If self.data_format is 'channels_last' and input `tensor` has shape
        (N, L, L, 2), the output tensor has shape (N, 2, L, L, 1).
        """
        if self.data_format == 'channels_first':
            N, D, H, W = self._input_shape
            #  N, D, H, W = tensor.shape
            if isinstance(tensor, np.ndarray):
                return np.reshape(tensor, (N, 1, H, W, D))

            return tf.reshape(tensor, (N, 1, D, H, W))

        if self.data_format == 'channels_last':
            #  N, H, W, D = tensor.shape
            N, H, W, D = self._input_shape
            if isinstance(tensor, np.ndarray):
                return np.reshape(tensor, (N, H, W, D, 1))

            return tf.reshape(tensor, (N, H, W, D, 1))

        raise AttributeError("`self.data_format` should be one of "
                             "'channels_first' or 'channels_last'")

    # pylint: disable=invalid-name, arguments-differ
    def call(self, inputs):
        """Forward pass through network.

        NOTE: Data flow of forward pass is outlined below.
        ============================================================
        * inputs: x, v, t
        ------------------------------------------------------------
            x -->
                (conv_x1, max_pool_x1) --> (conv_x1, max_pool_x2) -->
                batch_norm --> flatten_x --> x_layer --> x_out 

            v -->
                (conv_v1, max_pool_v1), --> (conv_v1, max_pool_v2) -->
                batch_norm --> flatten_v --> v_layer --> v_out

            t --> t_layer --> t_out 

            x_out + v_out + t_out --> h_layer --> h_out
        ============================================================
        * h_out is then fed to three separate layers:
        ------------------------------------------------------------
            (1.) h_out --> (scale_layer, tanh) * exp(coeff_scale)
                 output: scale (S function in orig. paper)

            (2.) h_out --> translation_layer --> translation_out
                 output: translation (T function in orig. paper)

            (3.) h_out --> 
                    (transformation_layer, tanh) * exp(coeff_transformation)
                 output: transformation (Q function in orig. paper)
          ============================================================

       Returns:
           scale, translation, transformation (S, T, Q functions from paper)
        """
        v, x, t, net_weights = inputs
        scale_weight = net_weights[0]
        transformation_weight = net_weights[1]
        translation_weight = net_weights[2]

        with tf.name_scope('reshape'):
            v = self.reshape_5D(v)
            x = self.reshape_5D(x)

        with tf.name_scope('x'):
            x = self.max_pool_x1(self.conv_x1(x))
            x = self.max_pool_x2(self.conv_x2(x))
            if self.use_bn:
                x = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(x)
            x = self.flatten(x)
            x = tf.nn.relu(self.x_layer(x))

        with tf.name_scope('v'):
            v = self.max_pool_v1(self.conv_v1(v))
            v = self.max_pool_v2(self.conv_v2(v))
            if self.use_bn:
                v = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(v)
            v = self.flatten(v)
            v = tf.nn.relu(self.v_layer(v))

        with tf.name_scope('t'):
            t = tf.nn.relu(self.t_layer(t))

        with tf.name_scope('h'):
            h = tf.nn.relu(v + x + t)
            h = tf.nn.relu(self.h_layer(h))

        def reshape(t, name):
            return tf.squeeze(
                tf.reshape(t, shape=self._input_shape, name=name)
            )

        with tf.name_scope('translation'):
            translation = translation_weight * self.translation_layer(h)

        with tf.name_scope('scale'):
            scale = (scale_weight
                     * tf.nn.tanh(self.scale_layer(h))
                     * tf.exp(self.coeff_scale))

        with tf.name_scope('transformation'):
            transformation = (transformation_weight
                              * self.transformation_layer(h)
                              * tf.exp(self.coeff_transformation))

        return scale, translation, transformation

