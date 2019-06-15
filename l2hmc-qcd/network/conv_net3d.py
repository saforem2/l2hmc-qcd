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

from globals import GLOBAL_SEED, TF_FLOAT, NP_FLOAT


np.random.seed(GLOBAL_SEED)

if '2.' not in tf.__version__:
    tf.set_random_seed(GLOBAL_SEED)


def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer, TF_FLOAT)
    return var


def variable_with_weight_decay(name, shape, stddev, wd, cpu=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: Name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: Add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this variable.

    Returns:
        Variable Tensor
    """
    if cpu:
        var = variable_on_cpu(
            name, shape, tf.truncated_normal_initializer(stddev=stddev,
                                                         dtype=TF_FLOAT)
        )
    else:
        var = tf.get_variable(
            name, shape, tf.truncated_normal_initializer(stddev=stddev,
                                                         dtype=TF_FLOAT)
        )
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def create_periodic_padding(samples, filter_size):
    """Create periodic padding for multiple samples, using filter_size."""
    original_size = np.shape(samples)
    N = original_size[1]  # number of links in lattice
    #  N = np.shape(samples)[1] # number of links in lattice
    padding = filter_size - 1

    samples = tf.reshape(samples, shape=(samples.shape[0], -1))

    x = []
    for sample in samples:
        padded = np.zeros((N + 2 * padding), N + 2 * padding, 2)
        # lower left corner
        padded[:padding, :padding, :] = sample[N-padding:, N-padding:, :]
        # lower middle
        padded[padding:N+padding, :padding, :] = sample[:, N-padding:, :]
        # loewr right corner
        padded[N+padding:, :padding, :] = sample[:padding, N-padding:, :]
        # left side
        padded[:padding, padding: N+padding, :] = sample[N-padding:, :, :]
        # center
        padded[:padding:N+padding, padding:N+padding, :] = sample[:, :, :]
        # right side
        padded[N+padding:, padding:N+padding:, :] = sample[:padding, :, :]
        # top middle
        padded[:padding:N+padding, N+padding:, :] = sample[:, :padding, :]
        # top right corner
        padded[N+padding:, N+padding:, :] = sample[:padding, :padding, :]

        x.append(padded)

    return np.array(x, dtype=NP_FLOAT).reshape(*original_size)


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
                    self.scale_layer = _custom_dense(
                        self.x_dim, 0.001, name='scale_layer'
                    )

                with tf.name_scope('translation_layer'):
                    self.translation_layer = _custom_dense(
                        self.x_dim, 0.001, 'translation_layer'
                    )

                with tf.name_scope('transformation_layer'):
                    self.transformation_layer = _custom_dense(
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
        v, x, t = inputs
        #  scale_weight = net_weights[0]
        #  transformation_weight = net_weights[1]
        #  translation_weight = net_weights[2]

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
            translation = self.translation_layer(h)

        with tf.name_scope('scale'):
            scale = (tf.nn.tanh(self.scale_layer(h))
                     * tf.exp(self.coeff_scale))

        with tf.name_scope('transformation'):
            transformation = (self.transformation_layer(h)
                              * tf.exp(self.coeff_transformation))

        return scale, translation, transformation


# pylint:disable=too-many-arguments, too-many-instance-attributes
class ConvNet2D(tf.keras.Model):
    """Conv. neural net with different initialization scale based on input."""

    def __init__(self, model_name, **kwargs):
        """Initialization method."""

        super(ConvNet2D, self).__init__(name=model_name)

        for key, val in kwargs.items():
            setattr(self, key, val)

        with tf.name_scope(self.name_scope):

            self.coeff_scale = tf.Variable(
                initial_value=tf.zeros([1, self.x_dim]),
                name='coeff_scale',
                trainable=True,
                dtype=TF_FLOAT
            )

            self.coeff_transformation = tf.Variable(
                initial_value=tf.zeros([1, self.x_dim]),
                name='coeff_transformation',
                trainable=True,
                dtype=TF_FLOAT
            )

            self.conv_x1 = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=self.filter_sizes[0],
                activation=tf.nn.relu,
                input_shape=self._input_shape,
                name='conv_x1',
                dtype=TF_FLOAT,
                data_format=self.data_format

            )

            self.max_pool_x1 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=2,
                name='max_pool_x1'
            )

            self.conv_v1 = tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=self.filter_sizes[0],
                activation=tf.nn.relu,
                input_shape=self._input_shape,
                name='conv_v1',
                dtype=TF_FLOAT,
                data_format=self.data_format
            )

            self.max_pool_v1 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=2,
                name='max_pool_x1'
            )

            self.conv_x2 = tf.keras.layers.Conv2D(
                filters=2*self.num_filters,
                kernel_size=self.filter_sizes[1],
                activation=tf.nn.relu,
                name='conv_x2',
                dtype=TF_FLOAT,
                data_format=self.data_format
            )

            self.max_pool_x2 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=2,
                name='max_pool_x1'
            )

            self.conv_v2 = tf.keras.layers.Conv2D(
                filters=2 * self.num_filters,
                kernel_size=self.filter_sizes[1],
                activation=tf.nn.relu,
                name='conv_v2',
                dtype=TF_FLOAT,
                data_format=self.data_format
            )

            self.max_pool_v2 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=2,
                name='max_pool_x1'
            )

            self.flatten = tf.keras.layers.Flatten(name='flatten')

            self.x_layer = _custom_dense(self.num_hidden, self.factor/3.,
                                         name='x_layer')

            self.v_layer = _custom_dense(self.num_hidden, 1./3.,
                                         name='v_layer')

            self.t_layer = _custom_dense(self.num_hidden, 1./3.,
                                         name='t_layer')

            self.h_layer = _custom_dense(self.num_hidden, name='h_layer')

            self.scale_layer = _custom_dense(self.x_dim, 0.001,
                                             name='scale_layer')

            self.translation_layer = _custom_dense(self.x_dim, 0.001,
                                                   name='translation_layer')

            self.transformation_layer = _custom_dense(
                self.x_dim,
                0.001,
                name='transformation_layer'
            )

    # pylint: disable=invalid-name, arguments-differ
    def call(self, inputs):
        """call method.

        Args: 
            inputs: Tuple consisting of (v, x, t) (momenta, x, time).

        Returns:
           scale, translation, transformation

        NOTE: Architecture looks like
            - inputs: x, v, t
                x -->
                    CONV_X1, MAX_POOL_X1, --> CONV_X1, MAX_POOL_X2 -->
                    FLATTEN_X --> X_LAYER --> X_OUT

                v -->
                    CONV_V1, MAX_POOL_V1, --> CONV_V1, MAX_POOL_V2 -->
                    FLATTEN_V --> V_LAYER --> V_OUT

                t --> T_LAYER --> T_OUT

                X_OUT + V_OUT + T_OUT --> H_LAYER --> H_OUT

            - H_OUT is then fed to three separate layers:
                (1.) H_OUT --> (SCALE_LAYER, TANH) * exp(COEFF_SCALE)
                     output: scale
                (2.) H_OUT --> TRANSLATION_LAYER --> TRANSLATION_OUT
                     output: translation
                (3.) H_OUT --> (TRANSFORMATION_LAYER, TANH)
                                * exp(COEFF_TRANSFORMATION)

                     output: transformation
        """
        v, x, t = inputs

        x = self.max_pool_x1(self.conv_x1(x))
        x = tf.nn.local_response_normalization(x)
        x = self.max_pool_x2(self.conv_x2(x))
        x = tf.nn.local_response_normalization(x)
        x = self.flatten(x)

        v = self.max_pool_v1(self.conv_v1(v))
        v = tf.nn.local_response_normalization(v)
        v = self.max_pool_v2(self.conv_v2(v))
        v = tf.nn.local_response_normalization(v)
        v = self.flatten(v)

        h = tf.nn.relu(self.v_layer(v) + self.x_layer(x) + self.t_layer(t))
        h = tf.nn.relu(self.h_layer(h))
        #  h = self.hidden_layer1(h)

        def reshape(t, name):
            return tf.reshape(t, shape=self._input_shape, name=name)

        translation = reshape(self.translation_layer(h), name='translation')

        scale = reshape(
            tf.nn.tanh(self.scale_layer(h)) * tf.exp(self.coeff_scale),
            name='scale'
        )

        transformation = reshape(
            self.transformation_layer(h) * tf.exp(self.coeff_transformation),
            name='transformation'
        )

        return scale, translation, transformation


def _custom_dense(units, factor=1., name=None):
    """Custom dense layer with specified weight intialization."""
    if '2.' not in tf.__version__:
        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=factor,
            mode='fan_in',
            distribution='uniform',
            dtype=TF_FLOAT,
            seed=GLOBAL_SEED,
        )
    else:
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=factor,
            mode='FAN_IN',
            seed=GLOBAL_SEED,
            uniform=True,
        )

    return tf.keras.layers.Dense(
        units=units,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        bias_initializer=tf.constant_initializer(0., dtype=TF_FLOAT),
        name=name
    )
