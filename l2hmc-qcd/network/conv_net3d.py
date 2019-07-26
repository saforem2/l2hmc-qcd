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
from .network_utils import batch_norm
#  import utils.file_io as io

from variables import GLOBAL_SEED, TF_FLOAT


np.random.seed(GLOBAL_SEED)

if '2.' not in tf.__version__:
    tf.set_random_seed(GLOBAL_SEED)


#  HE_INIT = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

def flatten_list(_list):
    return [item for sublist in _list for item in sublist]


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
        input = self.reshape_5D(input)

        input = self.max_pool1(self.conv1(input))
        input = self.conv2(input)
        if self.use_bn:
            input = self.activation(batch_norm(input, train_phase,
                                               axis=self.bn_axis,
                                               internal_update=True,
                                               scope=self.bn_name,
                                               reuse=tf.AUTO_REUSE))
        input = self.max_pool2(input)
        input = self.flatten(input)

        return input
