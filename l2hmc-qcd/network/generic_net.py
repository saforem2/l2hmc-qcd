"""
generic_net.py

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

class GenericNet(tf.keras.Model):
    """Conv. neural net with different initialization scale based on input."""
    def __init__(self, model_name='GenericNet', **kwargs):
        """Initialization method."""

        super(GenericNet, self).__init__(name=model_name)

        for key, val in kwargs.items():
            setattr(self, key, val)

        if self.name_scope is None:
            self.name_scope = model_name



        #  with tf.variable_scope(variable_scope):
        with tf.name_scope(self.name_scope):
            self.flatten = tf.keras.layers.Flatten(name='flatten')

            with tf.name_scope('x_layer'):
                self.x_layer = _custom_dense(self.num_hidden,
                                             self.factor/3.,
                                             name='x_layer')

            with tf.name_scope('v_layer'):
                self.v_layer = _custom_dense(self.num_hidden,
                                             1./3.,
                                             name='v_layer')

            with tf.name_scope('t_layer'):
                self.t_layer = _custom_dense(self.num_hidden,
                                             1./3.,
                                             name='t_layer')

            with tf.name_scope('h_layer'):
                self.h_layer = _custom_dense(self.num_hidden,
                                             name='h_layer')

            with tf.name_scope('scale_layer'):
                self.scale_layer = _custom_dense(self.x_dim,
                                                 0.001,
                                                 name='h_layer')

            with tf.name_scope('coeff_scale'):
                self.coeff_scale = tf.Variable(
                    initial_value=tf.zeros([1, self.x_dim]),
                    name='coeff_scale',
                    trainable=True,
                    dtype=tf.float32,
                )

            with tf.name_scope('translation_layer'):
                self.translation_layer = _custom_dense(
                    self.x_dim,
                    0.001,
                    name='translation_layer'
                )

            with tf.name_scope('transformation_layer'):
                self.transformation_layer = _custom_dense(
                    self.x_dim,
                    0.001,
                    name='transformation_layer'
                )

            with tf.name_scope('coeff_transformation'):
                self.coeff_transformation = tf.Variable(
                    initial_value=tf.zeros([1, self.x_dim]),
                    name='coeff_transformation',
                    trainable=True,
                    dtype=tf.float32
                )


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

        x = self.flatten(x)
        v = self.flatten(v)

        h = self.v_layer(v) + self.x_layer(x) + self.t_layer(t)
        h = tf.nn.relu(h)
        h = self.h_layer(h)
        h = tf.nn.relu(h)

        scale = tf.nn.tanh(self.scale_layer(h)) * tf.exp(self.coeff_scale)

        translation = self.translation_layer(h)

        transformation = (self.transformation_layer(h)
                          * tf.exp(self.coeff_transformation))
        #
        return scale, translation, transformation


def _custom_dense(units, factor=1., name=None):
    """Custom dense layer with specified weight intialization."""
    return tf.keras.layers.Dense(
        units=units,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=factor * 2.,
            mode='FAN_IN',
            uniform=False
        ),
        bias_initializer=tf.constant_initializer(0., dtype=tf.float32),
        name=name
    )
