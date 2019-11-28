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
        #  if self.name_scope == 'x':
        #      net_seeds = xnet_seeds
        #  elif self.name_scope == 'v':
        #      net_seeds = vnet_seeds

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
                self.dropout = tf.keras.layers.Dropout(
                    self.dropout_prob, seed=net_seeds['dropout'],
                )

            self.x_layer = custom_dense(units=self.num_hidden1,
                                        seed=net_seeds['x_layer'],
                                        factor=self.factor/3.,
                                        name='x_layer',
                                        input_shape=(self.x_dim,))
            self.v_layer = custom_dense(units=self.num_hidden1,
                                        seed=net_seeds['v_layer'],
                                        factor=1./3.,
                                        name='v_layer',
                                        input_shape=(self.x_dim,))
            self.t_layer = custom_dense(units=self.num_hidden1,
                                        seed=net_seeds['t_layer'],
                                        factor=1./3.,
                                        name='t_layer',
                                        input_shape=(self.x_dim,))

            self.h_layer = custom_dense(units=self.num_hidden2,
                                        seed=net_seeds['h_layer'],
                                        factor=1.,
                                        name='hidden_layer')

            self.scale_layer = custom_dense(units=self.x_dim,
                                            seed=net_seeds['scale_layer'],
                                            factor=0.001,
                                            name='scale_layer')

            transl_seed = net_seeds['translation_layer']
            self.translation_layer = custom_dense(units=self.x_dim,
                                                  seed=transl_seed,
                                                  factor=0.001,
                                                  name='translation_layer')

            transf_name = 'transformation_layer'
            transf_seed = net_seeds['transformation_layer']
            self.transformation_layer = custom_dense(units=self.x_dim,
                                                     seed=transf_seed,
                                                     factor=0.001,
                                                     name=transf_name)

    def call(self, inputs, train_phase):
        v, x, t = inputs

        with tf.name_scope('v'):
            v = self.v_layer(v)
        with tf.name_scope('x'):
            x = self.x_layer(x)
        with tf.name_scope('t'):
            t = self.t_layer(t)

        with tf.name_scope('hidden_layer'):
            h = self.activation(v + x + t)
            h = self.activation(self.h_layer(h))

            # dropout gets applied to the output of the previous layer
            if self.dropout_prob > 0:
                h = self.dropout(h, training=train_phase)

        with tf.name_scope('scale'):
            scale = (tf.nn.tanh(self.scale_layer(h))
                     * tf.exp(self.coeff_scale, name='exp_coeff_scale'))

        with tf.name_scope('transformation'):
            transformation = (tf.nn.tanh(self.transformation_layer(h))
                              * tf.exp(self.coeff_transformation,
                                       name='exp_coeff_transformation'))

        with tf.name_scope('translation'):
            translation = self.translation_layer(h)

        return scale, translation, transformation
