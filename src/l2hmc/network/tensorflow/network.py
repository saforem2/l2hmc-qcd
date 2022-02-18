"""
tensorflow/network.py

Tensorflow implementation of the network used to train the L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from typing import Optional

import numpy as np
import tensorflow as tf

from tensorflow.python.types.core import Callable

from l2hmc.configs import (
    ConvolutionConfig,
    NetWeight,
    NetworkConfig,
)
from l2hmc.network.factory import BaseNetworkFactory
from l2hmc.network.tensorflow.utils import PeriodicPadding

Tensor = tf.Tensor
Model = tf.keras.Model
Add = tf.keras.layers.Add
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Conv2D = tf.keras.layers.Conv2D
Dropout = tf.keras.layers.Dropout
Flatten = tf.keras.layers.Flatten
Reshape = tf.keras.layers.Reshape
Multiply = tf.keras.layers.Multiply
Activation = tf.keras.layers.Activation
MaxPooling2D = tf.keras.layers.MaxPooling2D
BatchNormalization = tf.keras.layers.BatchNormalization

PI = np.pi
TWO_PI = 2. * PI

TF_FLOAT = tf.keras.backend.floatx()


def to_u1(x: Tensor) -> Tensor:
    return (tf.add(x, PI) % TWO_PI) - PI


def linear_activation(x: Tensor) -> Tensor:
    return x


ACTIVATIONS = {
    'relu': tf.keras.activations.relu,
    'tanh': tf.keras.activations.tanh,
    'swish': tf.keras.activations.swish,
    'linear': linear_activation,
}

# FUNCTIONAL_ACTIVATIONS = {
#     'relu': tf.keras.layers.ReLU,
#     'tanh': tf.keras.layers.T
# }


class NetworkFactory(BaseNetworkFactory):
    def build_networks(self, n: int, split_xnets: bool) -> dict:
        """Build LeapfrogNetwork."""
        # TODO: if n == 0: build hmcNetwork (return zeros)
        assert n >= 1, 'Must build at least one network'

        cfg = self.get_build_configs()
        if n == 1:
            return {
                'xnet': get_network(**cfg['xnet'], name='xnet'),
                'vnet': get_network(**cfg['vnet'], name='vnet'),
            }

        vnet = {
            str(i): get_network(**cfg['vnet'], name=f'vnet_{i}')
            for i in range(n)
        }
        if split_xnets:
            # xstr = 'xnet'
            labels = ['first', 'second']
            xnet = {}
            for i in range(n):
                nets = [
                    get_network(**cfg['xnet'], name=f'xnet_{i}_first'),
                    get_network(**cfg['xnet'], name=f'xnet_{i}_second')
                ]
                xnet[str(i)] = dict(zip(labels, nets))
        else:
            xnet = {
                str(i): get_network(**cfg['xnet'], name=f'xnet_{i}')
                for i in range(n)
            }

        return {'xnet': xnet, 'vnet': vnet}


def get_network_configs(
        xdim: int,
        network_config: NetworkConfig,
        # factor: float = 1.,
        activation_fn: str | Callable = None,
        name: str = 'network',
) -> dict:
    """Returns network configs."""
    if isinstance(activation_fn, str):
        activation_fn = Activation(activation_fn)
        assert callable(activation_fn)
        # activation_fn = ACTIVATIONS.get(activation_fn, ACTIVATIONS['relu'])

    assert callable(activation_fn)
    names = {
        'x_input': f'{name}_xinput',
        'v_input': f'{name}_vinput',
        'x_layer': f'{name}_xLayer',
        'v_layer': f'{name}_vLayer',
        'scale': f'{name}_scaleLayer',
        'transf': f'{name}_transformationLayer',
        'transl': f'{name}_translationLayer',
        's_coeff': f'{name}_scaleCoeff',
        'q_coeff': f'{name}_transformationCoeff',
    }
    coeff_kwargs = {
        'trainable': True,
        'initial_value': tf.zeros([1, xdim], dtype=TF_FLOAT),
        'dtype': TF_FLOAT,
    }

    args = {
        'x': {
            # 'scale': factor / 2.,
            'name': names['x_layer'],
            'units': network_config.units[0],
            'activation': linear_activation,
        },
        'v': {
            # 'scale': 1. / 2.,
            'name': names['v_layer'],
            'units': network_config.units[0],
            'activation': linear_activation,
        },
        'scale': {
            # 'scale': 0.001 / 2.,
            'name': names['scale'],
            'units': xdim,
            'activation': linear_activation,
        },
        'transl': {
            # 'scale': 0.001 / 2.,
            'name': names['transl'],
            'units': xdim,
            'activation': linear_activation,
        },
        'transf': {
            # 'scale': 0.001 / 2.,
            'name': names['transf'],
            'units': xdim,
            'activation': linear_activation,
        },
    }

    return {
        'args': args,
        'names': names,
        'activation': activation_fn,
        'coeff_kwargs': coeff_kwargs,
    }


# pylint:disable=too-many-locals, too-many-arguments
def get_network(
        xshape: tuple,
        network_config: NetworkConfig,
        input_shapes: dict[str, tuple[int]] = None,
        net_weight: NetWeight = None,
        conv_config: Optional[ConvolutionConfig] = None,
        # factor: float = 1.,
        name: str = None,
) -> Model:
    """Returns a functional `tf.keras.Model`."""
    xdim = np.cumprod(xshape[1:])[-1]
    name = 'GaugeNetwork' if name is None else name

    if net_weight is None:
        net_weight = NetWeight(1., 1., 1.)

    if input_shapes is None:
        input_shapes = {
            'x': (xdim,), 'v': (xdim,),
        }

    cfg = get_network_configs(name=name,
                              xdim=xdim,
                              network_config=network_config,
                              activation_fn=network_config.activation_fn)
    args = cfg['args']
    names = cfg['names']
    act_fn = cfg['activation']

    x_input = Input(input_shapes['x'], name=names['x_input'], dtype=TF_FLOAT)
    v_input = Input(input_shapes['v'], name=names['v_input'], dtype=TF_FLOAT)

    s_coeff = tf.Variable(**cfg['coeff_kwargs'], name=names['s_coeff'])
    q_coeff = tf.Variable(**cfg['coeff_kwargs'], name=names['q_coeff'])

    if conv_config is None:
        x = Flatten()(x_input)

    else:
        if len(xshape) == 3:
            nt, nx, d = xshape
        elif len(xshape) == 4:
            _, nt, nx, d = xshape
        else:
            raise ValueError(f'Invalid value for `xshape`: {xshape}')

        try:
            x = Reshape((-1, nt, nx, d + 2))(x_input)
        except ValueError:
            x = Reshape((-1, nt, nx, d))(x_input)

        iterable = zip(conv_config.filters, conv_config.sizes)
        for idx, (f, n) in enumerate(iterable):
            x = PeriodicPadding(n - 1)(x)
            # Pass network_config.activation_fn (str) to Conv2D
            x = Conv2D(f, n, name=f'{name}/xConv{idx}',
                       activation=network_config.activation_fn)(x)
            if (idx + 1) % 2 == 0:
                p = conv_config.pool[idx]
                x = MaxPooling2D((p, p), name=f'{name}/xPool{idx}')(x)

        x = Flatten()(x)
        if network_config.use_batch_norm:
            x = BatchNormalization(-1)(x)

    x = Dense(**args['x'], dtype=TF_FLOAT)(x)
    v = Dense(**args['v'], dtype=TF_FLOAT)(v_input)
    z = act_fn(Add()([x, v]))
    for idx, units in enumerate(network_config.units[1:]):
        z = Dense(units,
                  activation=act_fn,
                  dtype=TF_FLOAT,
                  name=f'{name}_hLayer{idx}')(z)

    if network_config.dropout_prob > 0:
        z = Dropout(network_config.dropout_prob)(z)

    if network_config.use_batch_norm:
        z = BatchNormalization(-1, name=f'{name}_batchnorm')(z)

    # Scaling
    s = Multiply()([
        net_weight.s * tf.math.exp(s_coeff),
        tf.math.tanh(Dense(**args['scale'])(z))
    ])

    # Translation
    t = Dense(**args['transl'])(z)

    # Transformation
    q = Multiply()([
        net_weight.q * tf.math.exp(q_coeff),
        tf.math.tanh(Dense(**args['transf'])(z))
    ])

    model = Model(name=name,
                  inputs=[x_input, v_input],
                  outputs=[s, t, q])

    return model
