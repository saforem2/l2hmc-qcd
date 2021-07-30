"""
functional_net.py

Contains a functional `tf.keras.Model` that implements the network architecture
from `GaugeeNetwork` in `network/gauge_network.py`.

Author: Sam Foreman (github: @saforem2)
Date: 09/14/2020

"""
import numpy as np
# pylint:disable=invalid-name
from typing import Tuple

import tensorflow as tf

from tensorflow import keras

from network.config import ConvolutionConfig, NetworkConfig
#  from network.layers import ScaledTanhLayer

layers = tf.keras.layers

ACTIVATION_FNS = {
    'relu': tf.keras.activations.relu,
    'tanh': tf.keras.activations.tanh,
    'swish': tf.keras.activations.swish,
}


def custom_dense(units, scale=1., name=None, activation=None):
    """Implements a `layers.Dense` object with custom kernel initializer."""
    kinit = tf.keras.initializers.VarianceScaling(
        mode='fan_in', scale=2.*scale,
        distribution='truncated_normal',
    )

    return layers.Dense(units,
                        name=f'{name}_layer',
                        activation=activation,
                        kernel_initializer=kinit)


def vs_init(factor, kernel_initializer=None):
    """Create `VarianceScaling` initializer for network weights."""
    if kernel_initializer == 'zeros':
        return 'zeros'
    return tf.keras.initializers.VarianceScaling(
        mode='fan_in',
        scale=2.*factor,
        distribution='truncated_normal'
    )


# pylint:disable=unused-argument
class PeriodicPadding(layers.Layer):
    """Implements PeriodicPadding as a `tf.keras.layers.Layer`."""
    def __init__(self, size, **kwargs):
        super(PeriodicPadding, self).__init__(**kwargs)
        self._size = size

    def call(self, inputs, training=None, masks=None):
        """Call the network (forward-pass)."""
        z1 = inputs[:, -self._size:, :, ...]
        z2 = inputs[:, 0:self._size, :, ...]

        inputs = tf.concat([z1, inputs, z2], 1)

        z3 = inputs[:, :, -self._size:, ...]
        z4 = inputs[:, :, 0:self._size, ...]

        inputs = tf.concat([z3, inputs, z4], 2)

        return inputs


# pylint:disable=too-many-locals, too-many-arguments
def get_generic_network(
        input_shape: Tuple,
        net_config: NetworkConfig,
        kernel_initializer: str = None,
        input_shapes: Tuple = None,
        factor: float = 1.,
        name: str = None,
):
    """Returns a (functional) `tf.keras.Model`."""
    h1 = net_config.units[0]
    h2 = net_config.units[1]
    batch_size, xdim = input_shape
    scale_coeff = tf.Variable(initial_value=tf.zeros([1, xdim]),
                              name='scale_coeff', trainable=True)
    transf_coeff = tf.Variable(initial_value=tf.zeros([1, xdim]),
                               name='transf_coeff', trainable=True)
    if name is None:
        name = 'GenericNetwork'

    def s_(x):
        return f'{name}_{x}'

    if input_shapes is None:
        input_shapes = {
            'x': (input_shape[1],),
            'v': (input_shape[1],),
            #  't': (2,),
        }

    def get_input(s):
        return keras.Input(input_shapes[s], name=s_(s),
                           batch_size=batch_size)

    with tf.name_scope(name):
        x_input = get_input('x')
        v_input = get_input('v')
        #  t_input = get_input('t')

        x = custom_dense(h1, factor/2., f'{name}_x')(x_input)
        v = custom_dense(h1, 1./2., f'{name}_v')(v_input)
        #  t = custom_dense(h1, 1./3., f'{name}_t')(t_input)
        z = layers.Add()([x, v])
        #  z = layers.Add()([x, v, t])
        z = keras.activations.relu(z)
        z = custom_dense(h2, 1., f'{name}_h1')(z)
        z = custom_dense(h2, 1., f'{name}_h2')(z)
        if net_config.dropout_prob > 0:
            z = layers.Dropout(net_config.dropout_prob)(z)

        transl = custom_dense(xdim, 0.001, name=f'{name}_transl')(z)
        scale = tf.exp(scale_coeff) * tf.keras.activations.tanh(
            custom_dense(xdim, 0.001, name=f'{name}_scale')(z)
        )
        transf = tf.exp(transf_coeff) * tf.keras.activations.tanh(
            custom_dense(xdim, 0.001, name=f'{name}_transformation')(z)
        )

        #  model = keras.Model(
        #      name=name,
        #      inputs=[x_input, v_input, t_input],
        #      outputs=[scale, transl, transf]
        #  )
        model = keras.Model(
            name=name,
            inputs=[x_input, v_input],
            outputs=[scale, transl, transf]
        )

    return model


# pylint:disable=too-many-locals, too-many-arguments, too-many-statements
def get_gauge_network(
        x_shape: Tuple,
        net_config: NetworkConfig,
        conv_config: ConvolutionConfig = None,
        kernel_initializer: str = None,
        input_shapes: dict = None,
        factor: float = 1.,
        batch_size: Tuple = None,
        name: str = None,
):
    """Returns a (functional) `tf.keras.Model`."""
    if len(x_shape) == 4:
        batch_size, T, X, d = x_shape
    elif len(x_shape) == 3:
        T, X, d = x_shape
    else:
        raise ValueError('Incorrect shape passed for `x_shape`.')

    xdim = T * X * d
    #  if len(x_shape) == 4:
    #      batch_size, T, X, d = x_shape
    #  elif len(x_shape) == 3:
    #      T, X, d = x_shape
    #
    #  xdim = T * X * d
    activation_fn = net_config.activation_fn
    if isinstance(activation_fn, str):
        activation_fn = ACTIVATION_FNS.get(activation_fn, None)
        if activation_fn is None:
            raise KeyError(
                f'Bad activation fn specified: {activation_fn}. '
                f'Expected one of: {tuple(ACTIVATION_FNS.keys())}.'
            )

    if input_shapes is None:
        input_shapes = {
            'x': (xdim, 2), 'v': (xdim,), # 't': (2,)
        }

    if name is None:
        name = 'GaugeNetwork'

    def s_(x):
        return f'{name}_{x}'

    def get_input(s):
        return keras.Input(input_shapes[s], name=s_(s), batch_size=batch_size)

    # +-----------------------+
    # |     BUILD NETWORK     |
    # +-----------------------+
    with tf.name_scope(name):
        x_input = get_input('x')
        v_input = get_input('v')
        #  t_input = get_input('t')

        coeff_kwargs = {
            'trainable': True, 'initial_value': tf.zeros([1, xdim]),
        }
        scale_coeff = tf.Variable(name=f'{name}_scale_coeff', **coeff_kwargs)
        transf_coeff = tf.Variable(name=f'{name}_transf_coeff', **coeff_kwargs)

        if conv_config is not None:
            n1 = conv_config.filters[0]
            n2 = conv_config.filters[1]
            f1 = conv_config.sizes[0]
            f2 = conv_config.sizes[1]
            p1 = conv_config.pool_sizes[0]

            if 'xnet' in name.lower():
                x = tf.reshape(x_input, shape=(batch_size, T, X, d + 2))
            else:
                x = tf.reshape(x_input, shape=(batch_size, T, X, d))

            x = PeriodicPadding(f1 - 1)(x)
            x = layers.Conv2D(n1, f1, activation='relu',
                              name=f'{name}_xconv1')(x)
            x = layers.Conv2D(n2, f2, activation='relu',
                              name=f'{name}_xconv2')(x)
            x = layers.MaxPooling2D(p1, name=f'{name}_xpool')(x)
            x = layers.Conv2D(n2, f2, activation='relu',
                              name=f'{name}_xconv3')(x)
            x = layers.Conv2D(n1, f1, activation='relu',
                              name=f'{name}_xconv4')(x)
            x = layers.Flatten()(x)
            if conv_config.use_batch_norm:
                x = layers.BatchNormalization(-1, name=f'{name}_batch_norm')(x)
        else:
            x = layers.Flatten()(x_input)

        args = {
            'x': (net_config.units[0], factor / 2., f'{name}_x'),
            'v': (net_config.units[0], 1. / 2., f'{name}_v'),
            #  't': (net_config.units[0], 1. / 3., f'{name}_t'),
            'scale': (xdim, 0.001, f'{name}_scale'),
            'transl': (xdim, 0.001, f'{name}_transl'),
            'transf': (xdim, 0.001, f'{name}_transf'),
        }

        x = custom_dense(*args['x'])(x)
        v = custom_dense(*args['v'])(v_input)
        #  t = custom_dense(*args['t'])(t_input)

        z = layers.Add()([x, v])
        #  z = layers.Add()([x, v, t])
        z = activation_fn(z)
        #  z = keras.activations.relu(z)
        for idx, units in enumerate(net_config.units[1:]):
            z = custom_dense(units, 1./2., f'{name}_h{idx}',
                             activation=activation_fn)(z)

        #  z = custom_dense(*args['h1'])(z)
        #  z = custom_dense(*args['h2'])(z)

        if net_config.dropout_prob > 0:
            z = layers.Dropout(net_config.dropout_prob)(z)

        #  if net_config.get('use_batch_norm', False):
        if net_config.use_batch_norm:
            z = layers.BatchNormalization(-1, name=f'{name}_batch_norm1')(z)

        scale = custom_dense(*args['scale'], activation='tanh')(z)
        transl = custom_dense(*args['transl'])(z)
        transf = custom_dense(*args['transf'], activation='tanh')(z)

        scale *= tf.exp(scale_coeff)
        transf *= tf.exp(transf_coeff)

        model = keras.Model(name=name,
                            inputs=[x_input, v_input], #, t_input],
                            outputs=[scale, transl, transf])

    return model
