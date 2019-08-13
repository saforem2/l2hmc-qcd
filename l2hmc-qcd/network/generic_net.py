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

from config import TF_FLOAT, GLOBAL_SEED

from .network_utils import custom_dense


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
                initial_value=tf.zeros([1, self.x_dim], dtype=TF_FLOAT),
                name='coeff_scale',
                trainable=True,
                dtype=TF_FLOAT
            )

            #  with tf.name_scope('coeff_transformation'):
            self.coeff_transformation = tf.Variable(
                initial_value=tf.zeros([1, self.x_dim], dtype=TF_FLOAT),
                name='coeff_transformation',
                trainable=True,
                dtype=TF_FLOAT
            )

            if self.dropout_prob > 0:
                self.dropout = tf.keras.layers.Dropout(self.dropout_prob,
                                                       seed=GLOBAL_SEED,)

            x_factor = self.factor / 3.
            self.x_layer = custom_dense(self.num_hidden,
                                        x_factor,
                                        name='x_layer')
            self.v_layer = custom_dense(self.num_hidden,
                                        1./3.,
                                        name='v_layer')
            self.t_layer = custom_dense(self.num_hidden,
                                        1./3.,
                                        name='t_layer')

            self.h_layer = custom_dense(self.num_hidden,
                                        name='hidden_layer')

            self.scale_layer = custom_dense(self.x_dim,
                                            0.001,
                                            name='scale_layer')

            self.translation_layer = custom_dense(self.x_dim,
                                                  0.001,
                                                  'transl_layer')

            self.transformation_layer = custom_dense(self.x_dim,
                                                     0.001,
                                                     'transf_layer')

    def call(self, inputs, train_phase):
        v, x, t = inputs

        #  v = tf.nn.relu(self.v_layer(v))
        #  x = tf.nn.relu(self.x_layer(x))
        #  t = tf.nn.relu(self.t_layer(t))

        v = self.v_layer(v)
        x = self.x_layer(x)
        t = self.t_layer(t)

        h = tf.nn.relu(v + x + t)
        h = tf.nn.relu(self.h_layer(h))

        # dropout gets applied to the output of the previous layer
        if self.dropout_prob > 0:
            h = self.dropout(h, training=train_phase)

        translation = self.translation_layer(h)

        scale = (tf.nn.tanh(self.scale_layer(h))
                 * tf.exp(self.coeff_scale, name='exp_coeff_scale'))

        #  translation = tf.zeros_like(scale, name='translation')

        transformation = (tf.nn.tanh(self.transformation_layer(h))
                          * tf.exp(self.coeff_transformation,
                                   name='exp_coeff_transformation'))

        return scale, translation, transformation
