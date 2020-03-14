import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope, arg_scope

import config as cfg
from seed_dict import seeds

TF_FLOAT = cfg.TF_FLOAT
NP_FLOAT = cfg.NP_FLOAT

np.random.seed(seeds['global_np'])

if '2.' not in tf.__version__:
    tf.set_random_seed(seeds['global_tf'])


# pylint: disable=no-member


def activation_model(model):
    """Create Keras Model that outputs activations of all conv./pool layers.

    Args:
        model (tf.keraas.Model): Model for which we wish to visualize
            activations.
    Returns:
        activation_model (tf.keras.Model): Model that outputs the activations
            for each layer in `model.
    """
    layer_outputs = [layer.output for layer in model.layers]

    output_model = tf.keras.models.Model(inputs=model.input,
                                         output=layer_outputs)

    return output_model


def flatten(_list):
    """Flatten nested list."""
    return [item for sublist in _list for item in sublist]


def add_elements_to_collection(elements, collection_list):
    """Add list of `elements` to `collection_list`."""
    elements = flatten(elements)
    collection_list = flatten(collection_list)
    #  collection_list = tf.nest.flatten(collection_list)
    for name in collection_list:
        collection = tf.get_collection_ref(name)
        collection_set = set(collection)
        for element in elements:
            if element not in collection_set:
                collection.append(element)


def _assign_moving_average(orig_val, new_val, momentum, name):
    """Assign moving average."""
    with tf.name_scope(name):
        scaled_diff = (1 - momentum) * (new_val - orig_val)
        return tf.assign_add(orig_val, scaled_diff)


@add_arg_scope
def batch_norm(x,
               phase,
               axis=-1,
               shift=True,
               scale=True,
               momentum=0.99,
               eps=1e-3,
               internal_update=False,
               scope=None,
               reuse=None):
    """Implements a `BatchNormalization` layer."""
    C = x._shape_as_list()[axis]
    ndim = len(x.shape)
    var_shape = [1] * (ndim - 1) + [C]

    with tf.variable_scope(scope, 'batch_norm', reuse=reuse):
        def training():
            m, v = tf.nn.moments(x, list(range(ndim - 1)), keep_dims=True)
            update_m = _assign_moving_average(moving_m,
                                              m, momentum,
                                              'update_mean')
            update_v = _assign_moving_average(moving_v,
                                              v, momentum,
                                              'update_var')
            #  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_m)
            #  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_v)
            tf.add_to_collection('update_ops', update_m)
            tf.add_to_collection('update_ops', update_v)

            if internal_update:
                with tf.control_dependencies([update_m, update_v]):
                    output = (x - m) * tf.rsqrt(v + eps)
            else:
                output = (x - m) * tf.rsqrt(v + eps)
            return output

        def testing():
            m, v = moving_m, moving_v
            output = (x - m) * tf.rsqrt(v + eps)
            return output

        # Get mean and variance, normalize input
        moving_m = tf.get_variable('mean', var_shape,
                                   initializer=tf.zeros_initializer,
                                   trainable=False)
        moving_v = tf.get_variable('var', var_shape,
                                   initializer=tf.ones_initializer,
                                   trainable=False)

        if isinstance(phase, bool):
            output = training() if phase else testing()
        else:
            output = tf.cond(phase, training, testing)

        if scale:
            output *= tf.get_variable('gamma', var_shape,
                                      initializer=tf.ones_initializer)

        if shift:
            output += tf.get_variable('beta', var_shape,
                                      initializer=tf.zeros_initializer)

    return output


def custom_dense(units=100, seed=None, factor=1., name=None):
    """Custom dense layer with specified weight intialization."""
    try:
        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=2.*factor,
            mode='fan_in',
            distribution='truncated_normal',
            dtype=TF_FLOAT,
            seed=seed,
        )

    except AttributeError:
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
            seed=seed,
            mode='FAN_IN',
            uniform=False,
            dtype=TF_FLOAT,
            factor=2.*factor,
        )

    bias_initializer = tf.constant_initializer(0., dtype=TF_FLOAT)

    return tf.keras.layers.Dense(
        units=units,
        name=name,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        #  **kwargs
    )


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
