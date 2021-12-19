"""
network.py

Contains the TensorFlow implementation of Network.
"""
from __future__ import absolute_import, division, print_function, annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Add, Dropout, Dense, Flatten, Conv2D, MaxPooling2D,
    BatchNormalization, Input
)

from tensorflow.keras import Model

from collections import namedtuple

from ..config import NetworkConfig, ConvolutionConfig

ACTIVATION_FNS = {
    'relu': tf.keras.activations.relu,
    'tanh': tf.keras.activations.tanh,
    'swish': tf.keras.activations.swish,
}


State = namedtuple('State', ['x', 'v', 'beta'])


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


def get_gauge_network(
        x_shape: tuple,
        net_config: NetworkConfig,
        conv_config: ConvolutionConfig,
        input_shapes: dict[str, tuple] = None,
        factor: float = 1.,
):
    """"Returns a functional `tf.keras.Model`."""
    if len(x_shape) == 4:
        _, nt, nx, d = x_shape
    elif len(x_shape) == 3:
        nt, nx, d = x_shape
    else:
        raise ValueError(f'Incorrect shape passed for `x_shape`')

    xdim = nt * nx * d
    if input_shapes is None:
        input_shapes = {
            'v': (xdim,),
            'x': (xdim, 2),
        }

    x_input = Input(input_shapes['x'], name='x_input')
    v_input = Input(input_shapes['v'], name='v_input')
    coeff_kwargs = {
        'trainable': True,
        'initial_value': tf.zeros([1, xdim]),
    }
    scale_coeff = tf.Variable(name='scale_coeff', **coeff_kwargs)
    transf_coeff = tf.Variable(name='transf_coeff', **coeff_kwargs)

    if conv_config is not None:
        try:
            x = tf.reshape(x_input, shape=(-1, nt, nx, d + 2))
        except ValueError:
            x = tf.reshape(x_input, shape=(-1, nt, nx, d))

        iterable = zip(conv_config.filters,
                       conv_config.sizes)
        for idx, (f, n) in enumerate(iterable):
            x = PeriodicPadding(n - 1)(x)
            x = Conv2D(f, n, activation=conv_config.activation)(x)
            if (idx + 1) % 2 == 0:
                x = MaxPooling2D(conv_config.pool)
        x = Flatten()(x)
        if conv_config.use_batch_norm:
            x = BatchNormalization(-1)(x)
    else:
        x = Flatten()(x_input)

    args = {
        'x': {
            'units': net_config.units[0],
            'scale': factor / 2.,
        },
        'v': {
            'units': net_config.units[0],
            'scale': 1. / 2.,
        },
        'scale': {
            'units': xdim,
            'scale': 0.001 / 2.,
        },
        'translation': {
            'units': xdim,
            'scale': 0.001 / 2.,

        },
        'transformation': {
            'units': xdim,
            'scale': 0.001 / 2.,

        },
    }

    v = CustomDense(**args['v'])(v_input)
    x = CustomDense(**args['x'])(x)
    z = Add()([x, v])
    z = net_config.activation_fn(z)
    for units in net_config.units[1:]:
        z = Dense(units, activation=net_config.activation_fn)(z)

    if net_config.dropout_prob > 0:
        z = Dropout(net_config.dropout_prob)(z)

    if net_config.use_batch_norm:
        z = BatchNormalization(-1)(z)

    scale = CustomDense(**args['scale'], activation='tanh')(z)
    transl = CustomDense(**args['translation'], activation='tanh')(z)
    transf = CustomDense(**args['transformation'], activation='tanh')(z)

    scale *= tf.exp(scale_coeff)
    transf *= tf.exp(transf_coeff)

    model = Model(name='GaugeModel',
                  inputs=[x_input, v_input],
                  outputs=[scale, transl, transf])

    return model




