"""
Generic, fully-connected neural network architecture for running L2HMC on a
gauge lattice configuration of links.

NOTE: Lattices are flattened before being passed as input to the network.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Author: Sam Foreman (github: @saforem2)
Date: 01/16/2019
"""
import tensorflow as tf
import numpy as np

from globals import TF_FLOAT

from .network_utils import custom_dense


class GenericNet(tf.keras.Model):
    """Conv. neural net with different initialization scale based on input."""
    def __init__(self, model_name='GenericNet', **kwargs):
        """Initialization method."""

        super(GenericNet, self).__init__(name=model_name)

        for key, val in kwargs.items():
            setattr(self, key, val)

        if self.name_scope is None:
            self.name_scope = model_name

        if self.use_bn:
            self.bn_axis = -1

        #  with tf.variable_scope(variable_scope):
        with tf.name_scope(self.name_scope):
            #  self.flatten = tf.keras.layers.Flatten(name='flatten')

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
                                            1./3., name='t_layer')

            with tf.name_scope('h_layer'):
                self.h_layer = custom_dense(self.num_hidden,
                                            name='h_layer')

            with tf.name_scope('scale_layer'):
                self.scale_layer = custom_dense(self.x_dim, 0.001,
                                                name='scale_layer')

            with tf.name_scope('translation_layer'):
                self.translation_layer = custom_dense(
                    self.x_dim,
                    0.001,
                    name='translation_layer'
                )

            with tf.name_scope('transformation_layer'):
                self.transformation_layer = custom_dense(
                    self.x_dim,
                    0.001,
                    name='transformation_layer'
                )

            with tf.name_scope('coeff_scale'):
                self.coeff_scale = tf.Variable(
                    initial_value=tf.zeros([1, self.x_dim]),
                    name='coeff_scale',
                    trainable=True,
                    dtype=TF_FLOAT,
                )

            with tf.name_scope('coeff_transformation'):
                self.coeff_transformation = tf.Variable(
                    initial_value=tf.zeros([1, self.x_dim]),
                    name='coeff_transformation',
                    trainable=True,
                    dtype=TF_FLOAT
                )

    def _reshape(self, tensor):
        N, D, H, W = self._input_shape
        if isinstance(tensor, np.ndarray):
            return np.reshape(tensor, (N, D * H * W))

        return tf.reshape(tensor, (N, D * H * W))

    # pylint: disable=invalid-name, arguments-differ
    def call(self, inputs):
        """call method.

        NOTE Architecture looks like:

            * inputs: x, v, t
                x --> FLATTEN_X --> X_LAYER --> X_OUT
                v --> FLATTEN_V --> V_LAYER --> V_OUT
                t --> T_LAYER --> T_OUT

                X_OUT + V_OUT + T_OUT --> H_LAYER --> H_OUT

            * H_OUT is then fed to three separate layers:
                (1.) H_OUT -->
                       TANH(SCALE_LAYER) * exp(COEFF_SCALE) --> SCALE_OUT

                     input: H_OUT
                     output: scale
            
                (2.) H_OUT --> TRANSLATION_LAYER --> TRANSLATION_OUT

                     input: H_OUT
                     output: translation

                (3.) H_OUT -->
                       TANH(SCALE_LAYER)*exp(COEFF_TRANSFORMATION) -->
                         TRANFORMATION_OUT

                     input: H_OUT
                     output: transformation

       Returns:
           scale, translation, transformation
        """
        v, x, t = inputs

        x = self._reshape(x)
        v = self._reshape(v)

        h = self.v_layer(v) + self.x_layer(x) + self.t_layer(t)
        if self.use_bn:
            h = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(h)
        h = tf.nn.relu(h)
        h = self.h_layer(h)
        h = tf.nn.relu(h)

        with tf.name_scope('scale'):
            scale = (tf.exp(self.coeff_scale)
                     * tf.nn.tanh(self.scale_layer(h)))

        with tf.name_scope('transformation'):
            transformation = (tf.exp(self.coeff_transformation)
                              * tf.nn.tanh(self.transformation_layer(h)))

        with tf.name_scope('translation'):
            translation = self.translation_layer(h)

        #  scale = tf.nn.tanh(self.scale_layer(h)) * tf.exp(self.coeff_scale)

        #  transformation = (tf.nn.tanh(self.transformation_layer(h))
        #                    * tf.exp(self.coeff_transformation))

        return scale, translation, transformation
