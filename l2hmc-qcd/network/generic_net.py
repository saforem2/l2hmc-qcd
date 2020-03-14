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

from seed_dict import seeds, xnet_seeds, vnet_seeds

from .network_utils import custom_dense

import config as cfg


TF_FLOAT = cfg.TF_FLOAT

# pylint: disable=invalid-name


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

        self.activation = kwargs.get('generic_activation', tf.nn.relu)
        net_seeds = kwargs.get('net_seeds', None)

        with tf.name_scope(self.net_name):
            self.x_layer = custom_dense(name='x_layer',
                                        factor=self.factor/3.,
                                        units=self.num_hidden1,
                                        seed=net_seeds['x_layer'])
            #  input_shape=(self.x_dim,))
            self.v_layer = custom_dense(name='v_layer',
                                        factor=1./3,
                                        units=self.num_hidden1,
                                        seed=net_seeds['v_layer'])

            self.t_layer = custom_dense(name='t_layer',
                                        factor=1./3.,
                                        units=self.num_hidden1,
                                        seed=net_seeds['t_layer'])

            self.h_layer = custom_dense(name='h_layer',
                                        factor=1.,
                                        units=self.num_hidden2,
                                        seed=net_seeds['h_layer'])

            # Scale layer
            sname = 'scale_layer'
            self.scale_layer = custom_dense(name=sname,
                                            factor=0.001,
                                            units=self.x_dim,
                                            seed=net_seeds[sname])
            #  Translation layer
            qname = 'transformation_layer'
            self.transformation_layer = custom_dense(name=qname,
                                                     factor=0.001,
                                                     units=self.x_dim,
                                                     seed=net_seeds[qname])
            # Translation layer
            tname = 'translation_layer'
            self.translation_layer = custom_dense(name=tname,
                                                  factor=0.001,
                                                  units=self.x_dim,
                                                  seed=net_seeds[tname])
            # Scale layer coefficient
            self.coeff_scale = tf.Variable(
                name='coeff_scale',
                trainable=True,
                dtype=TF_FLOAT,
                initial_value=tf.zeros([1, self.x_dim], dtype=TF_FLOAT),
                #  use_resource=True,
            )

            # Transformation layer coefficient
            self.coeff_transformation = tf.Variable(
                name='coeff_transformation',
                trainable=True,
                dtype=TF_FLOAT,
                initial_value=tf.zeros([1, self.x_dim], dtype=TF_FLOAT),
                #  use_resource=True,
            )

            # Dropout layer (only defined if `dropout_prob > 0`
            if self.dropout_prob > 0:
                self.dropout = tf.keras.layers.Dropout(
                    self.dropout_prob, seed=net_seeds['dropout'],
                )

    def call(self, inputs, train_phase):
        """Call network."""
        v, x, t = inputs
        #  with tf.name_scope('v'):
        #  with tf.name_scope('x'):
        #  with tf.name_scope('t'):

        with tf.name_scope('hidden_layer'):
            h = self.activation(
                self.v_layer(v) + self.x_layer(x) + self.t_layer(t)
            )
            h = self.activation(self.h_layer(h))

            # dropout gets applied to the output of the previous layer
            if self.dropout_prob > 0:
                h = self.dropout(h, training=train_phase)

        with tf.name_scope('scale'):
            scale = tf.nn.tanh(self.scale_layer(h))
            scale *= tf.exp(self.coeff_scale)

        with tf.name_scope('transformation'):
            transformation = tf.nn.tanh(self.transformation_layer(h))
            transformation *= tf.exp(self.coeff_transformation)

        with tf.name_scope('translation'):
            translation = self.translation_layer(h)

        return scale, translation, transformation
