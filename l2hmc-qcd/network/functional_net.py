"""
functional_net.py

Contains a functional `tf.keras.Model` that implements the network architecture
from `GaueNetwork` in `network/gauge_network.py`.

Author: Sam Foreman (github: @saforem2)
Date: 09/14/2020
"""
from __future__ import absolute_import, print_function, division, annotations
#  import numpy as np
# pylint:disable=invalid-name
#  from typing import Tuple

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import (
    Layer, Add, Dropout, Dense, Flatten, Conv2D, MaxPooling2D,
    BatchNormalization, Input
)
from tensorflow.keras import Model
from tensorflow.keras.initializers import VarianceScaling

from network.config import ConvolutionConfig, NetworkConfig
#  from network.layers import ScaledTanhLayer

#  layers = tf.keras.layers

ACTIVATION_FNS = {
    'relu': tf.keras.activations.relu,
    'tanh': tf.keras.activations.tanh,
    'swish': tf.keras.activations.swish,
}

def get_kinit(scale: float = 1., seed: int = None):
    return VarianceScaling(scale=2. * scale,
                           mode='fan_in', seed=seed,
                           distribution='truncated_normal')



def custom_dense(units, scale=1., name=None, activation=None):
    """Implements a `layers.Dense` object with custom kernel initializer."""
    kinit = tf.keras.initializers.VarianceScaling(
        mode='fan_in', scale=2.*scale,
        distribution='truncated_normal',
    )

    return Dense(units, name=f'{name}_layer',
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


class CustomDense(Layer):
    def __init__(
            self,
            units: int,
            scale: float = 1.,
            activation: str = None,
            **kwargs,
    ):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.scale = scale
        kinit = tf.keras.initializers.VarianceScaling(
            mode='fan_in', scale=2.*self.scale,
            distribution='truncated_normal',
        )
        self.layer = Dense(self.units,
                           name=self.name,
                           activation=activation,
                           kernel_initializer=kinit)

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({'units': self.units, 'scale': self.scale})
        return config

    def call(self, x: tf.Tensor):
        return self.layer(x)



# pylint:disable=unused-argument
class PeriodicPadding(Layer):
    """Implements a PeriodicPadding as a `tf.keras.layers.Layer` object."""
    def __init__(self, size: int, **kwargs):
        super(PeriodicPadding, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs: tf.Tensor):
        """Call the layer in the foreward direction."""
        x0 = inputs[:, -self.size:, :, ...]
        x1 = inputs[:, 0:self.size, :, ...]

        inputs = tf.concat([x0, inputs, x1], 1)

        y0 = inputs[:, :, -self.size:, ...]
        y1 = inputs[:, :, 0:self.size, ...]

        inputs = tf.concat([y0, inputs, y1], 2)

        return inputs

    def get_config(self):
        config = super(PeriodicPadding, self).get_config()
        config.update({'size': self.size})
        return config


# pylint:disable=too-many-locals, too-many-arguments
def get_generic_network(
        input_shape: tuple,
        net_config: NetworkConfig,
        input_shapes: dict[str, tuple],
        factor: float = 1.,
        name: str = None,
        #  kernel_initializer: str = None,
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
        }

    def get_input(s):
        return keras.Input(input_shapes[s], name=s_(s),
                           batch_size=batch_size)

    with tf.name_scope(name):
        x_input = get_input('x')
        v_input = get_input('v')
        #  t_input = get_input('t')

        #  x = Dense(h1,
        x = CustomDense(h1, factor/2., name=f'{name}_x')(x_input)
        #  x = custom_dense(h1, factor/2., f'{name}_x')(x_input)
        v = CustomDense(h1, 1./2., name=f'{name}_v')(v_input)
        #  t = custom_dense(h1, 1./3., f'{name}_t')(t_input)
        z = Add()([x, v])
        #  z = layers.Add()([x, v, t])
        z = keras.activations.relu(z)
        z = CustomDense(h2, 1., name=f'{name}_h1')(z)
        z = CustomDense(h2, 1., name=f'{name}_h2')(z)
        #  z = custom_dense(h2, 1., name=f'{name}_h1')(z)
        #  z = custom_dense(h2, 1., name=f'{name}_h2')(z)
        if net_config.dropout_prob > 0:
            z = Dropout(net_config.dropout_prob)(z)

        transl = CustomDense(xdim, 0.001, name=f'{name}_transl')(z)
        scale = tf.exp(scale_coeff) * tf.keras.activations.tanh(
            CustomDense(xdim, 0.001, name=f'{name}_scale')(z)
        )
        transf = tf.exp(transf_coeff) * tf.keras.activations.tanh(
            CustomDense(xdim, 0.001, name=f'{name}_transformation')(z)
        )

        #  model = keras.Model(
        #      name=name,
        #      inputs=[x_input, v_input, t_input],
        #      outputs=[scale, transl, transf]
        #  )
        model = Model(name=name,
                      inputs=[x_input, v_input],
                      outputs=[scale, transl, transf])

    return model


# pylint:disable=too-many-locals, too-many-arguments, too-many-statements
def get_gauge_network(
        x_shape: tuple,
        net_config: NetworkConfig,
        #  kernel_initializer: str = None,
        conv_config: ConvolutionConfig = None,
        input_shapes: dict[str, tuple] = None,
        factor: float = 1.,
        batch_size: tuple = None,
        name: str = 'GaugeNetwork',
):
    """Returns a (functional) `tf.keras.Model`."""
    # -- Check shapes ---------------------------------------------
    if len(x_shape) == 4:
        batch_size, T, X, d = x_shape
    elif len(x_shape) == 3:
        T, X, d = x_shape
    else:
        raise ValueError('Incorrect shape passed for `x_shape`.')

    xdim = T * X * d
    if input_shapes is None:
        input_shapes = {
            'x': (xdim, 2), 'v': (xdim,),
        }

    # -- Define activation function to use -------------------------
    activation_fn = net_config.activation_fn
    if isinstance(activation_fn, str):
        activation_fn = ACTIVATION_FNS.get(activation_fn, None)
        if activation_fn is None:
            raise KeyError(
                f'Bad activation fn specified: {activation_fn}. '
                f'Expected one of: {tuple(ACTIVATION_FNS.keys())}.'
            )

    def sub_s(s):
        return f'{name}/{s}'

    def get_input(i):
        #  return keras.Input(input_shapes[s], sub_s(s), batch_size)
        #  return keras.Input(input_shapes[s], name=sub_s(s),
        #                     batch_size=batch_size)
        return Input(input_shapes[i], name=sub_s(i))#

    # -- Build network --------------------------------------------
    with tf.name_scope(name):
        x_input = get_input('x')
        v_input = get_input('v')

        coeff_kwargs = {
            'trainable': True, 'initial_value': tf.zeros([1, xdim]),
        }
        scale_coeff = tf.Variable(name=f'{name}/scale_coeff', **coeff_kwargs)
        transf_coeff = tf.Variable(name=f'{name}/transf_coeff', **coeff_kwargs)

        if conv_config is not None:
            n1 = conv_config.filters[0]
            n2 = conv_config.filters[1]
            f1 = conv_config.sizes[0]
            f2 = conv_config.sizes[1]
            p1 = conv_config.pool_sizes[0]

            if 'xnet' in name.lower():
                x = tf.reshape(x_input, shape=(-1, T, X, d + 2))
            else:
                x = tf.reshape(x_input, shape=(-1, T, X, d))

            x = PeriodicPadding(f1 - 1)(x)
            x = Conv2D(n1, f1, activation='relu', name=f'{name}/xconv1')(x)
            x = Conv2D(n2, f2, activation='relu', name=f'{name}/xconv2')(x)
            x = MaxPooling2D(p1, name=f'{name}/xpool')(x)
            x = Conv2D(n2, f2, activation='relu', name=f'{name}/xconv3')(x)
            x = Conv2D(n1, f1, activation='relu', name=f'{name}/xconv4')(x)
            x = Flatten()(x)
            if conv_config.use_batch_norm:
                x = BatchNormalization(-1, name=f'{name}/batch_norm')(x)
        else:
            x = Flatten()(x_input)

        #  args = {
        #      'x': (net_config.units[0], factor / 2., f'{name}/x'),
        #      'v': (net_config.units[0], 1. / 2., f'{name}/v'),
        #      'scale': (xdim, 0.001, f'{name}/scale'),
        #      'transl': (xdim, 0.001, f'{name}/transl'),
        #      'transf': (xdim, 0.001, f'{name}/transf'),
        #  }
        args = {
            'x': {
                'units': net_config.units[0],
                'scale': factor / 2.,
                'name': f'{name}/xlayer',
                #  'kernel_initializer': get_kinit(factor/2.)
            },
            'v': {
                'units': net_config.units[0],
                'scale': 1. / 2.,
                'name': f'{name}/vlayer',
                #  'kernel_initializer': get_kinit(1./2.)
            },
            'scale': {
                'units': xdim,
                'scale': 0.001 / 2.,
                'name': f'{name}/scale',
                #  'kernel_initializer': get_kinit(0.001/2.),
            },
            'transl': {
                'units': xdim,
                'scale': 0.001 / 2.,
                'name': f'{name}/transl',
                #  'kernel_initializer': get_kinit(0.001/2.),
            },
            'transf': {
                'units': xdim,
                'scale': 0.001 / 2.,
                'name': f'{name}/transf',
                #  'kernel_initializer': get_kinit(0.001/2.),
            },
        }

        v = CustomDense(**args['v'])(v_input)
        x = CustomDense(**args['x'])(x)
        #  v = Dense(**args['v'])(v_input)
        #  x = Dense(**args['x'])(x)

        #  x = custom_dense(*args['x'])(x)
        #  v = custom_dense(*args['v'])(v_input)

        z = Add()([x, v])
        z = activation_fn(z)
        #  z = keras.activations.relu(z)
        # ------------------------------------------
        # TODO: Replace 1./2. in custom_dense with:
        # 1 / len(net_config.units[1:])
        # ------------------------------------------
        nlayers = len(net_config.units[1:])
        scale = 1. / (2. * nlayers)
        for idx, units in enumerate(net_config.units[1:]):
            z = CustomDense(units, scale=scale,
                            name=f'{name}/h{idx}',
                            activation=activation_fn)(z)
            #  z = Dense(units, name=f'{name}/h{idx}',
            #            activation=activation_fn,
            #            kernel_initializer=get_kinit(1./2.))(z)
            #  z = custom_dense(units, 1./2., f'{name}_h{idx}',
            #                   activation=activation_fn)(z)

        if net_config.dropout_prob > 0:
            z = Dropout(net_config.dropout_prob)(z)

        if net_config.use_batch_norm:
            z = BatchNormalization(-1, name=f'{name}/batch_norm1')(z)

        scale = CustomDense(**args['scale'], activation='tanh')(z)
        transl = CustomDense(**args['transl'])(z)
        transf = CustomDense(**args['transf'], activation='tanh')(z)
        #  scale = custom_dense(*args['scale'], activation='tanh')(z)
        #  transl = custom_dense(*args['transl'])(z)
        #  transf = custom_dense(*args['transf'], activation='tanh')(z)

        scale *= tf.exp(scale_coeff)
        transf *= tf.exp(transf_coeff)

        model = Model(name=name,
                      inputs=[x_input, v_input], #, t_input],
                      outputs=[scale, transl, transf])
    return model
