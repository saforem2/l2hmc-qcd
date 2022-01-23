"""
tensorflow/network.py

Tensorflow implementation of the network used to train the L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Layer,
    MaxPooling2D,
)
from tensorflow.python.types.core import Callable

from src.l2hmc.configs import (
    ConvolutionConfig,
    NetWeight,
    NetworkConfig,
    Shape,
)
from src.l2hmc.network.factory import BaseNetworkFactory
from src.l2hmc.network.tensorflow.utils import PeriodicPadding


Tensor = tf.Tensor
Model = tf.keras.Model
PI = np.pi
TWO_PI = 2. * PI

TF_FLOAT = tf.keras.backend.floatx()

NetworkInputs = "tuple[Tensor, Tensor]"
NetworkOutputs = "tuple[Tensor, Tensor, Tensor]"


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


class NetworkFactory(BaseNetworkFactory):
    def build_networks(self, n: int, split_xnets: bool) -> dict:
        """Build LeapfrogNetwork."""
        # TODO: if n == 0: build hmcNetwork (return zeros)
        assert n >= 1, 'Must build at least one network'

        cfg = self.get_build_configs()
        if n == 1:
            return {
                'xnet': get_network(**cfg['xnet'], name='xNet'),
                'vnet': get_network(**cfg['vnet'], name='vNet'),
            }

        vnet = {
            str(i): get_network(**cfg['vnet'], name=f'vNet/lfLayer{i}')
            for i in range(n)
        }
        if split_xnets:
            xstr = 'xNet/lfLayer'
            labels = ['first', 'second']
            xnet = {}
            for i in range(n):
                nets = [
                    get_network(**cfg['xnet'], name=f'{xstr}{i}/first'),
                    get_network(**cfg['xnet'], name=f'{xstr}{i}/second')
                ]
                xnet[str(i)] = dict(zip(labels, nets))
        else:
            xnet = {
                str(i): get_network(**cfg['xnet'], name=f'xNet/lfLayer{i}')
                for i in range(n)
            }

        return {'xnet': xnet, 'vnet': vnet}


def get_kinit(scale: float = 1., seed: int = None):
    return VarianceScaling(scale=2. * scale,
                           mode='fan_in', seed=seed,
                           distribution='truncated_normal')


class CustomDense(Layer):
    def __init__(
            self,
            units: int,
            scale: float = 1.,
            activation: str | Callable = None,
            **kwargs,
    ):
        super(CustomDense, self).__init__(**kwargs)
        if isinstance(activation, str):
            activation = ACTIVATIONS.get(activation, ACTIVATIONS['relu'])

        self.units = units
        self.scale = scale
        self.activation = activation
        kinit = tf.keras.initializers.VarianceScaling(
            mode='fan_in',
            scale=2.*self.scale,
            distribution='truncated_normal',
        )
        self.layer = Dense(self.units,
                           name=self.name,
                           activation=activation,
                           kernel_initializer=kinit)

    def get_config(self) -> dict:
        config = super(CustomDense, self).get_config()
        config.update({
            'units': self.units,
            'scale': self.scale,
            'activation': self.activation,
        })
        return config

    def call(self, x: Tensor) -> Tensor:
        return self.layer(x)


def get_network_configs(
        xdim: int,
        network_config: NetworkConfig,
        # factor: float = 1.,
        activation_fn: str | Callable = None,
        name: str = 'Network',
) -> dict:
    """Returns network configs."""
    if isinstance(activation_fn, str):
        activation_fn = ACTIVATIONS.get(activation_fn, ACTIVATIONS['relu'])

    assert callable(activation_fn)
    names = {
        'x_input': f'{name}/xInput',
        'v_input': f'{name}/vInput',
        'x_layer': f'{name}/xLayer',
        'v_layer': f'{name}/vLayer',
        'scale': f'{name}/scaleLayer',
        'transl': f'{name}/translationLayer',
        'transf': f'{name}/transformationLayer',
        's_coeff': f'{name}/scaleCoeff',
        'q_coeff': f'{name}/transformationCoeff',
    }
    coeff_kwargs = {
        'trainable': True,
        'initial_value': tf.zeros([1, xdim], dtype=TF_FLOAT),
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
        xshape: Shape,
        network_config: NetworkConfig,
        input_shapes: dict[str, tuple[int]] = None,
        net_weight: NetWeight = None,
        conv_config: ConvolutionConfig = None,
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

    s_coeff = tf.Variable(**cfg['coeff_kwargs'],
                          name=names['s_coeff'],
                          dtype=TF_FLOAT)
    q_coeff = tf.Variable(**cfg['coeff_kwargs'],
                          name=names['q_coeff'],
                          dtype=TF_FLOAT)

    if conv_config is not None:
        if len(xshape) == 3:
            nt, nx, d = xshape
        elif len(xshape) == 4:
            _, nt, nx, d = xshape
        else:
            raise ValueError(f'Invalid value for `xshape`: {xshape}')

        n1 = conv_config.filters[0]
        n2 = conv_config.filters[1]
        f1 = conv_config.sizes[0]
        f2 = conv_config.sizes[1]
        p1 = conv_config.pool[0]

        if 'xnet' in name.lower():
            x = tf.reshape(x_input, shape=(-1, nt, nx, d + 2))
        else:
            x = tf.reshape(x_input, shape=(-1, nt, nx, d))

        x = PeriodicPadding(f1 - 1)(x)
        x = Conv2D(n1, f1, activation='relu', name=f'{name}/xConv1')(x)
        x = Conv2D(n2, f2, activation='relu', name=f'{name}/xConv2')(x)
        x = MaxPooling2D(p1, name=f'{name}/xPool')(x)
        x = Conv2D(n2, f2, activation='relu', name=f'{name}/xConv3')(x)
        x = Conv2D(n1, f1, activation='relu', name=f'{name}/xConv4')(x)
        x = Flatten()(x)
        if network_config.use_batch_norm:
            x = BatchNormalization(-1, name=f'{name}/batch_norm')(x)
    else:
        x = Flatten()(x_input)

    x = Dense(**args['x'], dtype=TF_FLOAT)(x)
    v = Dense(**args['v'], dtype=TF_FLOAT)(v_input)
    z = act_fn(Add()([x, v]))
    for idx, units in enumerate(network_config.units[1:]):
        z = Dense(units,
                  activation=act_fn,
                  dtype=TF_FLOAT,
                  name=f'{name}/hLayer{idx}')(z)

    if network_config.dropout_prob > 0:
        z = Dropout(network_config.dropout_prob)(z)

    if network_config.use_batch_norm:
        z = BatchNormalization(-1, name=f'{name}/BatchNorm')(z)

    s_layer = Dense(**args['scale'])
    t_layer = Dense(**args['transl'])
    q_layer = Dense(**args['transf'])

    s = tf.math.exp(s_coeff) * tf.math.tanh(s_layer(z))
    t = t_layer(z)
    q = tf.math.exp(q_coeff) * tf.math.tanh(q_layer(z))

    scale = net_weight.s * s
    transl = net_weight.t * t
    transf = net_weight.q * q

    model = Model(name=name,
                  inputs=[x_input, v_input],
                  outputs=[scale, transl, transf])

    return model