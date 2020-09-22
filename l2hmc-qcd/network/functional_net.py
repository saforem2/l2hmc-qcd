"""
functional_net.py

Contains a functional `tf.keras.Model` that implements the network architecture
from `GaugeeNetwork` in `network/gauge_network.py`.

Author: Sam Foreman (github: @saforem2)
Date: 09/14/2020
"""
from typing import Tuple

import tensorflow as tf

from tensorflow import keras

from config import NetworkConfig
from network.gauge_network import ScaledTanhLayer
from network.gauge_conv_network import ConvolutionConfig, periodic_image

layers = tf.keras.layers


def custom_dense(units, kernel_initializer, name=None):
    """Implements a `layers.Dense` object with custom kernel initializer."""
    return layers.Dense(units, name=name,
                        kernel_initializer=kernel_initializer)


def vs_init(factor, kernel_initializer=None):
    if kernel_initializer == 'zeros':
        return 'zeros'
    return tf.keras.initializers.VarianceScaling(
        mode='fan_in',
        scale=2.*factor,
        distribution='truncated_normal'
    )


def get_kernel_initializers(factor=1., kernel_initializer=None):
    """Get kernel initializers, layer by layer."""
    names = ['x_layer', 'v_layer', 't_layer', 'h_layer1', 'h_layer2',
             'scale_layer', 'transl_layer', 'transf_layer']
    if kernel_initializer == 'zeros':
        kinits = len(names) * ['zeros']
        return dict(zip(names, kinits))

    return {
        'x_layer': vs_init(factor/3.),
        'v_layer': vs_init(1./3.),
        't_layer': vs_init(1./3.),
        'h_layer1': vs_init(1./2.),
        'h_layer2': vs_init(1./2.),
        'scale_layer': vs_init(0.001),
        'transl_layer': vs_init(0.001),
        'transf_layer': vs_init(0.001),
    }


class PeriodicPadding(layers.Layer):
    """Implements PeriodicPadding as a `tf.keras.layers.Layer`."""
    def __init__(self, size, **kwargs):
        super(PeriodicPadding, self).__init__(**kwargs)
        self._size = size

    def call(self, inputs, **kwargs):
        """Call the network (forward-pass)."""
        z1 = inputs[:, -self._size:, :, ...]
        z2 = inputs[:, 0:self._size, :, ...]

        inputs = tf.concat([z1, inputs, z2], 1)

        z3 = inputs[:, :, -self._size:, ...]
        z4 = inputs[:, :, 0:self._size, ...]

        inputs = tf.concat([z3, inputs, z4], 2)

        return inputs


# pylint:disable=too-many-locals, invalid-name
def get_gauge_network(
        lattice_shape: Tuple, net_config: NetworkConfig,
        conv_config: ConvolutionConfig = None,
        kernel_initializer: str = None,
        input_shapes: dict = None,
        factor: float = 1.,
        batch_size: Tuple = None, name: str = None,
):
    """Returns a (functional) `tf.keras.Model`."""
    if len(lattice_shape) == 4:
        batch_size, T, X, d = lattice_shape
    elif len(lattice_shape) == 3:
        T, X, d = lattice_shape

    xdim = T * X * d

    if input_shapes is None:
        input_shapes = {
            'x': (xdim, 2), 'v': (xdim,), 't': (2,)
        }

    h1 = net_config.units[0]
    h2 = net_config.units[1]
    kinits = get_kernel_initializers(factor, kernel_initializer)
    if name is None:
        name = 'GaugeNetwork'

    def s_(x):
        return f'{name}/{x}'

    def get_input(s):
        return keras.Input(input_shapes[s], name=s_(s), batch_size=batch_size)

    #  x_input = keras.Input(shape=(T, X, d), batch_size=batch_size, name='x')
    #  v_input = keras.Input(shape=(T, X, d), batch_size=batch_size, name='v')
    with tf.name_scope(name):
        #  if conv_config is None:
        #      x_input = keras.Input(shape=(xdim, 2), batch_size=batch_size)
        #  else:
        #      x_input = keras.Input(shape=(T, X, d, 2), batch_size=batch_size)
        x_input = get_input('x')
        v_input = get_input('v')
        t_input = get_input('t')

        #  x_input = keras.Input(input_shapes['x'],
        #                        name=s_('x'), batch_size=batch_size)
        #  v_input = keras.Input(input_shapes['v'],
        #                        name=s_('v'), batch_size=batch_size)
        #  t_input = keras.Input(input_shapes['t'],
        #                        name=s_('t'), batch_size=batch_size)
        #  x = tf.concat([tf.math.cos(x_input), tf.math.sin(x_input)], axis=-1)

        if conv_config is not None:
            n1 = conv_config.filters[0]
            n2 = conv_config.filters[1]
            f1 = conv_config.sizes[0]
            f2 = conv_config.sizes[1]
            p1 = conv_config.pool_sizes[0]

            x = tf.reshape(x_input, shape=(batch_size, T, X, d + 2))
            #  x = tf.transpose(x, (0, 1, 2, 4, 3))
            #  x = periodic_image(x, f1 - 1)
            x = PeriodicPadding(f1 - 1)(x)
            x = layers.Conv2D(n1, f1, activation='relu',
                              name=s_('xConv1'))(x)
            x = layers.Conv2D(n2, f2, activation='relu',
                              name=s_('xConv2'))(x)
            x = layers.MaxPooling2D(p1, name=s_('xPool'))(x)
            x = layers.Conv2D(n2, f2, activation='relu',
                              name=s_('xConv3'))(x)
            x = layers.Conv2D(n1, f1, activation='relu',
                              name=s_('xConv4'))(x)
            x = layers.Flatten()(x)
            if conv_config.use_batch_norm:
                x = layers.BatchNormalization(axis=-1,
                                              name=s_('batch_norm'))(x)
            #  v = layers.Flatten()(v_input)
        else:
            x = layers.Flatten()(x_input)
            #  x = layers.Flatten()(x_input)
            #  v = layers.Flatten()(v_input)

        #  v = layers.Dense(h1, name='v_layer')(v_input)
        #  t = layers.Dense(h1, name='t_layer')(t_input)
        x = custom_dense(h1, kinits['x_layer'], s_('x_layer'))(x)
        v = custom_dense(h1, kinits['v_layer'], s_('v_layer'))(v_input)
        t = custom_dense(h1, kinits['t_layer'], s_('t_layer'))(t_input)
        z = layers.Add()([x, v, t])
        z = keras.activations.relu(z)
        z = custom_dense(h2, kinits['h_layer1'], s_('h_layer1'))(z)
        z = custom_dense(h2, kinits['h_layer2'], s_('h_layer2'))(z)
        #  z = layers.Dense(h2, name='h_layer1')(z)
        #  z = layers.Dense(h2, name='h_layer2')(z)
        if net_config.dropout_prob > 0:
            z = layers.Dropout(net_config.dropout_prob)(z)

        args = (xdim, kinits['scale_layer'])
        scale = ScaledTanhLayer(*args, name=s_('scale_layer'))(z)
        transl = custom_dense(xdim, kinits['transl_layer'],
                              s_('transl_layer'))(z)

        kwargs = {
            'kernel_initializer': kinits['transf_layer'],
            'name': s_('transformation_layer')
        }
        transf = ScaledTanhLayer(xdim, **kwargs)(z)

        model = keras.Model(
            name=name,
            inputs=[x_input, v_input, t_input],
            outputs=[scale, transl, transf]
        )

    return model
