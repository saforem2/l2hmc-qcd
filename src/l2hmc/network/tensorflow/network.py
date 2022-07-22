"""
tensorflow/network.py

Tensorflow implementation of the network used to train the L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from typing import Optional

import numpy as np
import tensorflow as tf
import logging

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

log = logging.getLogger(__name__)

PI = np.pi
TWO_PI = 2. * PI

TF_FLOAT = tf.keras.backend.floatx()

ACTIVATIONS = {
    'relu': tf.keras.activations.relu,
    'tanh': tf.keras.activations.tanh,
    'swish': tf.keras.activations.swish,
    'linear': lambda x: x,
}


def linear_activation(x: Tensor) -> Tensor:
    return x

# FUNCTIONAL_ACTIVATIONS = {
#     'relu': tf.keras.layers.ReLU,
#     'tanh': tf.keras.layers.T
# }


def zero_weights(model: Model) -> Model:
    for layer in model.layers:
        if isinstance(layer, Model):
            zero_weights(layer)
        else:
            weights = layer.get_weights()
            zeros = []
            for w in weights:
                log.info(f'Zeroing layer: {layer}')
                zeros.append(np.zeros_like(w))

            layer.set_weights(zeros)

    return model


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
        activation_fn: Optional[str | Callable] = None,
        name: Optional[str] = 'network',
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


def setup(
        xdim: int,
        network_config: NetworkConfig,
        name: Optional[str] = 'network',
) -> dict:
    """Setup for building network."""
    layer_kwargs = {
        'x': {
            'units': network_config.units[0],
            'name': f'{name}_xLayer',
            'activation': linear_activation,
        },
        'v': {
            'units': network_config.units[0],
            'name': f'{name}_vLayer',
            'activation': linear_activation,
        },
        'scale': {
            'units': xdim,
            'name': f'{name}_scaleLayer',
            'activation': linear_activation,
        },
        'transl': {
            'units': xdim,
            'name': f'{name}_translationLayer',
            'activation': linear_activation,
        },
        'transf': {
            'units': xdim,
            'name': f'{name}_transformationLayer',
            'activation': linear_activation,
        },
    }

    coeff_defaults = {
        'dtype': TF_FLOAT,
        'trainable': True,
        'initial_value': tf.zeros([1, xdim], dtype=TF_FLOAT),
    }
    coeff_kwargs = {
        'scale': {
            'name': f'{name}_scaleCoeff',
            **coeff_defaults,
        },
        'transf': {
            'name': f'{name}_transformationCoeff',
            **coeff_defaults,
        }
    }

    return {'layer': layer_kwargs, 'coeff': coeff_kwargs}


# pylint:disable=too-many-locals, too-many-arguments
def get_network(
        xshape: tuple,
        network_config: NetworkConfig,
        input_shapes: Optional[dict[str, tuple[int, int]]] = None,
        net_weight: Optional[NetWeight] = None,
        conv_config: Optional[ConvolutionConfig] = None,
        # factor: float = 1.,
        name: Optional[str] = None,
) -> Model:
    """Returns a functional `tf.keras.Model`."""
    xdim = np.cumprod(xshape[1:])[-1]
    name = 'GaugeNetwork' if name is None else name

    if isinstance(network_config.activation_fn, str):
        act_fn = Activation(network_config.activation_fn)
        assert callable(act_fn)
    elif callable(network_config.activation_fn):
        act_fn = network_config.activation_fn
    else:
        raise ValueError(
            'Unexpected value encountered in '
            f'`NetworkConfig.activation_fn`: {network_config.activation_fn}'
        )

    if net_weight is None:
        net_weight = NetWeight(1., 1., 1.)

    if input_shapes is None:
        input_shapes = {
            'x': (int(xdim), int(2)),
            'v': (int(xdim), int(2)),
        }

    kwargs = setup(xdim=xdim, name=name, network_config=network_config)
    # coeff_kwargs = kwargs['coeff']
    layer_kwargs = kwargs['layer']

    x_input = Input(input_shapes['x'], name=f'{name}_xinput')
    v_input = Input(input_shapes['v'], name=f'{name}_vinput')
    # log.info(f'xinput: {x_input}')
    # log.info(f'vinput: {v_input}')

    s_coeff = tf.Variable(
        trainable=True,
        dtype=TF_FLOAT,
        initial_value=tf.zeros([1, xdim], dtype=TF_FLOAT),
    )
    q_coeff = tf.Variable(
        trainable=True,
        dtype=TF_FLOAT,
        initial_value=tf.zeros([1, xdim], dtype=TF_FLOAT),
    )

    v = Flatten()(v_input)
    if conv_config is None or len(conv_config.filters) == 0:
        x = Flatten()(x_input)

    # if conv_config is not None and len(conv_config.filters) > 0:
    elif conv_config is not None and len(conv_config.filters) > 0:
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

    else:
        raise ValueError('Unable to build network.')

    x = Dense(**layer_kwargs['x'])(x)
    v = Dense(**layer_kwargs['v'])(v)
    z = act_fn(Add()([x, v]))
    for idx, units in enumerate(network_config.units[1:]):
        z = Dense(units,
                  # dtype=TF_FLOAT,
                  activation=act_fn,
                  name=f'{name}_hLayer{idx}')(z)

    if network_config.dropout_prob > 0:
        z = Dropout(network_config.dropout_prob)(z)

    if network_config.use_batch_norm:
        z = BatchNormalization(-1, name=f'{name}_batchnorm')(z)

    # -------------------------------------------------------------
    # NETWORK OUTPUTS
    #  1. s: Scale function
    #  2. t: Translation function
    #  3. q: Transformation function
    # -------------------------------------------------------------
    # 1. Scaling
    s = Multiply()([
        net_weight.s * tf.math.exp(s_coeff),
        tf.math.tanh(Dense(**layer_kwargs['scale'])(z))
    ])

    # 2. Translation
    t = Dense(**layer_kwargs['transl'])(z)

    # 3. Transformation
    q = Multiply()([
        net_weight.q * tf.math.exp(q_coeff),
        tf.math.tanh(Dense(**layer_kwargs['transf'])(z))
    ])

    model = Model(name=name, inputs=[x_input, v_input], outputs=[s, t, q])

    return model
