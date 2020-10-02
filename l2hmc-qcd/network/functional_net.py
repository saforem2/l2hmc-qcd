"""
functional_net.py

Contains a functional `tf.keras.Model` that implements the network architecture
from `GaugeeNetwork` in `network/gauge_network.py`.

Author: Sam Foreman (github: @saforem2)
Date: 09/14/2020

"""
# pylint:disable=invalid-name
from typing import Tuple

import tensorflow as tf

from tensorflow import keras

from network.config import ConvolutionConfig, NetworkConfig
#  from network.layers import ScaledTanhLayer

layers = tf.keras.layers


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


def get_kernel_initializers(factor=1., kernel_initializer=None):
    """Get kernel initializers, layer by layer."""
    names = ['x', 'v', 't', 'h1', 'h2', 'scale', 'transl', 'transf']
    if kernel_initializer == 'zeros':
        kinits = len(names) * ['zeros']
        return dict(zip(names, kinits))

    return {
        'x': vs_init(factor/3.),
        'v': vs_init(1./3.),
        't': vs_init(1./3.),
        'h1': vs_init(1./2.),
        'h2': vs_init(1./2.),
        'scale': vs_init(0.001),
        'transl': vs_init(0.001),
        'transf': vs_init(0.001),
    }


# pylint:disable=unused-argument
class PeriodicPadding(layers.Layer):
    """Implements PeriodicPadding as a `tf.keras.layers.Layer`."""
    def __init__(self, size, **kwargs):
        super(PeriodicPadding, self).__init__(**kwargs)
        self._size = size

    def call(self, inputs, training=None):
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
                              name='scale/coeff', trainable=True)
    transf_coeff = tf.Variable(initial_value=tf.zeros([1, xdim]),
                               name='transf/coeff', trainable=True)
    if name is None:
        name = 'GenericNetwork'

    def s_(x):
        return f'{name}/{x}'

    if input_shapes is None:
        input_shapes = {
            'x': (input_shape[1],),
            'v': (input_shape[1],),
            't': (2,),
        }

    def get_input(s):
        return keras.Input(input_shapes[s], name=s_(s),
                           batch_size=batch_size)

    with tf.name_scope(name):
        x_input = get_input('x')
        v_input = get_input('v')
        t_input = get_input('t')

        x = custom_dense(h1, factor/3., f'{name}/x')(x_input)
        v = custom_dense(h1, 1./3., f'{name}/v')(v_input)
        t = custom_dense(h1, 1./3., f'{name}/t')(t_input)
        z = layers.Add()([x, v, t])
        z = keras.activations.relu(z)
        z = custom_dense(h2, 1., f'{name}/h1')(z)
        z = custom_dense(h2, 1., f'{name}/h2')(z)
        #  z = layers.Dense(h2, name='h_layer1')(z)
        #  z = layers.Dense(h2, name='h_layer2')(z)
        if net_config.dropout_prob > 0:
            z = layers.Dropout(net_config.dropout_prob)(z)

        #  args = (xdim, )
        #  scale = ScaledTanhLayer(xdim, 0.001, name=f'{name}/scale')(z)
        transl = custom_dense(xdim, 0.001, name=f'{name}/transl')(z)
        scale = tf.exp(scale_coeff) * tf.keras.activations.tanh(
            custom_dense(xdim, 0.001, name=f'{name}/scale')(z)
        )
        transf = tf.exp(transf_coeff) * tf.keras.activations.tanh(
            custom_dense(xdim, 0.001, name=f'{name}/transformation')(z)
        )
        #  transf = ScaledTanhLayer(
        #      xdim, 0.001, name=f'{name}/transformation'
        #  )(z)

        model = keras.Model(
            name=name,
            inputs=[x_input, v_input, t_input],
            outputs=[scale, transl, transf]
        )

    return model


# pylint:disable=too-many-locals, too-many-arguments
def get_gauge_network(
        lattice_shape: Tuple,
        net_config: NetworkConfig,
        conv_config: ConvolutionConfig = None,
        kernel_initializer: str = None,
        input_shapes: dict = None,
        factor: float = 1.,
        batch_size: Tuple = None,
        name: str = None,
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
    if len(net_config.units) == 3:
        h1, h2, h3 = net_config.units
    elif len(net_config.units) == 2:
        h1, h2 = net_config.units
        h3 = h2
    else:
        h1 = h2 = h3 = net_config.units
    #  kinits = get_kernel_initializers(factor, kernel_initializer)
    if name is None:
        name = 'GaugeNetwork'

    def s_(x):
        return f'{name}/{x}'

    def get_input(s):
        return keras.Input(input_shapes[s], name=s_(s), batch_size=batch_size)

    # +-----------------------+
    # |     BUILD NETWORK     |
    # +-----------------------+
    with tf.name_scope(name):
        x_input = get_input('x')
        v_input = get_input('v')
        t_input = get_input('t')

        coeff_kwargs = {
            'trainable': True, 'initial_value': tf.zeros([1, xdim]),
        }
        scale_coeff = tf.Variable(name=f'{name}/scale/coeff', **coeff_kwargs)
        transf_coeff = tf.Variable(name=f'{name}/transf/coeff', **coeff_kwargs)

        if conv_config is not None:
            n1 = conv_config.filters[0]
            n2 = conv_config.filters[1]
            f1 = conv_config.sizes[0]
            f2 = conv_config.sizes[1]
            p1 = conv_config.pool_sizes[0]

            x = tf.reshape(x_input, shape=(batch_size, T, X, d + 2))
            #  x = tf.transpose(x, (0, 1, 2, 4, 3))
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
        else:
            x = layers.Flatten()(x_input)

        args = {
            'x': (h1, factor / 3., f'{name}/x'),
            'v': (h1, 1. / 3., f'{name}/v'),
            't': (h1, 1. / 3., f'{name}/t'),
            'h1': (h2, 1. / 2., f'{name}/h1'),
            'h2': (h2, 1. / 2., f'{name}/h2'),
            'h3': (h3, 1. / 2., f'{name}/h3'),
            'scale': (xdim, 0.001, f'{name}/scale'),
            'transl': (xdim, 0.001, f'{name}/transl'),
            'transf': (xdim, 0.001, f'{name}/transf'),
        }

        x = custom_dense(*args['x'])(x)
        v = custom_dense(*args['v'])(v_input)
        t = custom_dense(*args['t'])(t_input)

        z = layers.Add()([x, v, t])
        z = keras.activations.relu(z)

        #  for n, units in enumerate(net_config.units):
        #      z = custom_dense(units, 1./2., name=f'{name}/h{n}')(z)

        z = custom_dense(*args['h1'])(z)
        z = custom_dense(*args['h2'])(z)
        z = custom_dense(*args['h3'])(z)

        if net_config.dropout_prob > 0:
            z = layers.Dropout(net_config.dropout_prob)(z)

        scale = custom_dense(*args['scale'], activation='tanh')(z)
        transl = custom_dense(*args['transl'])(z)
        transf = custom_dense(*args['transf'], activation='tanh')(z)

        scale *= tf.exp(scale_coeff)
        transf *= tf.exp(transf_coeff)

        #  scale = tf.exp(scale_coeff) * tf.keras.activations.tanh(
        #      custom_dense(xdim, factors['S'], name=f'{name}/scale')(z)
        #  )
        #
        #  transf = tf.exp(transf_coeff) * tf.keras.activations.tanh(
        #      custom_dense(
        #          xdim, factors['Q'], name=f'{name}/transformation'
        #      )(z)
        #  )

        #  scale = ScaledTanhLayer(xdim, 0.001, name=f'{name}/scale')(z)
        #  transl = custom_dense(xdim, 0.001, name=f'{name}/transl')(z)
        #  transf = ScaledTanhLayer(
        #      xdim, 0.001, name=f'{name}/transformation')(z)

        model = keras.Model(
            name=name,
            inputs=[x_input, v_input, t_input],
            outputs=[scale, transl, transf]
        )

    return model
