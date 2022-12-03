"""
tensorflow/network.py

Tensorflow implementation of the network used to train the L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from typing import Optional, Callable

import numpy as np
import tensorflow as tf
import logging

# from tensorflow.python.types.core import Callable

from l2hmc.configs import (
    ConvolutionConfig,
    NetWeight,
    NetworkConfig,
)
from l2hmc.group.su3.tensorflow.group import SU3
from l2hmc.group.u1.tensorflow.group import U1Phase
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

TF_FLOAT = tf.dtypes.as_dtype(tf.keras.backend.floatx())
tf.dtypes.as_dtype

ACTIVATIONS = {
    'relu': tf.keras.activations.relu,
    'tanh': tf.keras.activations.tanh,
    'swish': tf.keras.activations.swish,
    'linear': lambda x: x,
}


def linear_activation(x: Tensor) -> Tensor:
    return x


def get_activation(act_fn: str | Callable) -> Callable:
    if isinstance(act_fn, Callable):
        return act_fn
    act_fn = Activation(act_fn)
    assert callable(act_fn)
    return act_fn


def dummy_network(
        inputs: tuple[Tensor, Tensor],
        training: Optional[bool] = None,  # pyright:ignore
) -> tuple[Tensor, Tensor, Tensor]:
    _, v = inputs
    return (
        tf.zeros_like(v),
        tf.zeros_like(v),
        tf.zeros_like(v)
    )


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


class ScaledTanh(tf.keras.Model):
    def __init__(self, out_features: int, name: Optional[str]) -> None:
        super(ScaledTanh, self).__init__(name=name)
        self.coeff = tf.math.exp(
            tf.Variable(
                trainable=True,
                dtype=TF_FLOAT,
                initial_value=tf.zeros(
                    [1, out_features],
                    dtype=TF_FLOAT
                ),
            )
        )
        self.layer = Dense(out_features, use_bias=False)

    def call(self, x):
        return self.coeff * tf.math.tanh(self.layer(x))


class ConvStack(tf.keras.Model):
    def __init__(
            self,
            xshape: list[int] | tuple[int],
            conv_config: ConvolutionConfig,
            activation_fn: str | Callable,
            use_batch_norm: bool = False,
            name: Optional[str] = None,
    ) -> None:
        super(ConvStack, self).__init__(name=name)
        # xshape will be of the form
        # [batch, dim, *lattice_shape, *link_shape]
        # -- Cases ------------------------------------------------------
        # 1. xshape: [dim, nt, nx]
        if len(xshape) == 3:
            # lattice_shape: [nt, nx]
            ny, nz = 0, 0
            d, nt, nx = xshape[0], xshape[1], xshape[2]

        # 2. xshape: [batch, dim, nt, nx]
        elif len(xshape) == 4:
            # lattice_shape: [nt, nx]
            ny, nz = 0, 0
            _, d, nt, nx = xshape[0], xshape[1], xshape[2], xshape[3]

        # 3. xshape: [batch, dim, nt, nx, ny, nz, 3, 3]
        elif len(xshape) == 8:
            # link_shape: [3, 3]
            # lattice_shape = [nt, nx, ny, nz]
            d = xshape[1]
            nt, nx, ny, nz = xshape[2], xshape[3], xshape[4], xshape[5]
        else:
            raise ValueError(f'Invalid value for xshape: {xshape}')
        # ---------------------------------------------------------------
        self.d = d
        self.nt = nt
        self.ny = ny
        self.nz = nz
        self.nx = nx
        self.xshape = xshape
        self.xdim = d * nt * nx
        self.activation_fn = get_activation(activation_fn)
        self.flatten = Flatten()
        self.conv_layers = []
        if (nfilters := len(conv_config.filters)) > 0:
            if (
                    conv_config.sizes is not None
                    and nfilters == len(conv_config.sizes)
            ):
                for idx, (f, n) in enumerate(
                        zip(conv_config.filters,
                            conv_config.sizes)
                ):
                    self.conv_layers.append(
                        PeriodicPadding(n - 1)
                    )
                    self.conv_layers.append(
                        Conv2D(
                            filters=f,
                            kernel_size=n,
                            activation=self.activation_fn
                        )
                    )
                    if (idx + 1) % 2 == 0:
                        p = (
                            2 if conv_config.pool is None
                            else conv_config.pool[idx]
                        )
                        self.conv_layers.append(
                            MaxPooling2D((p, p), name=f'{name}/xPool{idx}')
                        )

        self.conv_layers.append(Flatten())
        if use_batch_norm:
            self.conv_layers.append(BatchNormalization(-1))

        self.conv_layers.append(
            Dense(self.xdim, activation=self.activation_fn)
        )

    def call(self, x: Tensor) -> Tensor:
        if x.shape != self.xshape:
            if len(x.shape) == 2:
                try:
                    x = tf.reshape(
                        x,
                        [x.shape[0], self.d + 2, self.nt, self.nx]
                    )
                except ValueError:
                    x = tf.reshape(
                        x,
                        [x.shape[0], self.d, self.nt, self.nx]
                    )

        # if tf.argmin(x.shape) == 1:
        if x.shape[1] in [self.d, self.d + 2]:
            # NOTE: lattice assumes:
            # NOTE:   x.shape = [N, C, H, W]  = [0, 1, 2, 3], but
            # NOTE: tf wants:   [N, H, W, C] -> [0, 2, 3, 1] (transpose)
            x = tf.transpose(x, (0, 2, 3, 1))

        # if x.shape[1] in [self.d, self.d + 2]:
        #     # [N, C, H, W] --> [N, H, W, C] for TensorFlow
        #     x = tf.transpose(x, (0, 2, 3, 1))
        # if x.shape != self.xshape:
        #     try:
        #         x = tf.reshape(x, [x.shape[0], self.nt, self.nx, self.d + 2])
        #     except ValueError:
        #         x = tf.reshape(x, [x.shape[0], self.nt, self.nx, self.d])

        for layer in self.conv_layers:
            x = layer(x)

        return x


class InputLayer(tf.keras.Model):
    def __init__(
            self,
            xshape: list[int],
            network_config: NetworkConfig,
            activation_fn: str | Callable[[Tensor], Tensor],
            conv_config: Optional[ConvolutionConfig] = None,
            input_shapes: Optional[dict[str, int]] = None,
            name: Optional[str] = None,
    ) -> None:
        super(InputLayer, self).__init__(name=name)
        self.xshape = xshape
        self.activation_fn = get_activation(activation_fn)
        self.net_config = network_config
        self.units = self.net_config.units
        self.xdim = np.cumprod(xshape[1:])[-1]
        self.flatten = Flatten()

        if input_shapes is None:
            input_shapes = {'x': self.xdim, 'v': self.xdim}
        self.input_shapes = {}
        for key, val in input_shapes.items():
            if isinstance(val, (list, tuple)):
                self.input_shapes[key] = np.cumprod(val)[-1]
            elif isinstance(val, int):
                self.input_shapes[key] = val
            else:
                raise ValueError(
                    'Unexpected value in input_shapes!\n'
                    f'\tinput_shapes: {input_shapes}\n'
                    f'\t  val: {val}'
                )

        self.conv_stack = None
        self.conv_config = conv_config
        if conv_config is not None and len(conv_config.filters) > 0:
            self.conv_stack = ConvStack(
                xshape=xshape,
                conv_config=conv_config,
                activation_fn=self.activation_fn,
            )

        self.xlayer = Dense(self.net_config.units[0])
        self.vlayer = Dense(self.net_config.units[0])

    def call(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> Tensor:
        x, v = inputs
        if self.conv_stack is not None:
            x = self.conv_stack(x)

        v = self.vlayer(self.flatten(v))
        x = self.xlayer(self.flatten(x))
        assert x is not None and v is not None
        # v = self.vlayer(tf.reshape(v, [v.shape[0], -1]))
        # x = self.xlayer(tf.reshape(x, [x.shape[0], -1]))
        return self.activation_fn(x + v)


class LeapfrogLayer(tf.keras.Model):
    def __init__(
            self,
            xshape: list[int],
            network_config: NetworkConfig,
            input_shapes: Optional[dict[str, int]] = None,
            net_weight: Optional[NetWeight] = None,
            conv_config: Optional[ConvolutionConfig] = None,
            group: Optional[U1Phase | SU3] = None,
            name: Optional[str] = None,
    ):
        super(LeapfrogLayer, self).__init__(name=name)
        if net_weight is None:
            net_weight = NetWeight(1., 1., 1.)

        self.xshape = xshape
        self.g = group
        self.net_config = network_config
        self.nw = net_weight
        self.xdim = np.cumprod(xshape[1:])[-1]
        act_fn = get_activation(self.net_config.activation_fn)
        self.activation_fn = act_fn

        self.input_layer = InputLayer(
            xshape=xshape,
            network_config=network_config,
            activation_fn=self.activation_fn,
            conv_config=conv_config,
            input_shapes=input_shapes,
        )

        self.units = self.net_config.units
        self.hidden_layers = []
        for idx, units in enumerate(self.units[1:]):
            h = Dense(units, name=f'{name}_hLayer{idx}')
            self.hidden_layers.append(h)

        self.scale = ScaledTanh(self.xdim, name=f'{name}_ScaledTanh')
        self.transf = ScaledTanh(self.xdim, name=f'{name}_ScaledTanh')
        self.transl = Dense(self.xdim)

        self.dropout = None
        if self.net_config.dropout_prob > 0.:
            self.dropout = Dropout(self.net_config.dropout_prob)

        self.batch_norm = None
        if self.net_config.use_batch_norm:
            self.batch_norm = BatchNormalization(-1, name=f'{name}_batchnorm')

    def set_net_weight(self, net_weight: NetWeight):
        self.nw = net_weight

    def call(
            self,
            inputs: tuple[Tensor, Tensor],
            training: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = self.activation_fn(layer(z))

        if self.net_config.dropout_prob > 0. and self.dropout is not None:
            z = self.dropout(z, training=training)

        if self.net_config.use_batch_norm and self.batch_norm is not None:
            z = self.batch_norm(z, training=training)

        dt = inputs[0].dtype
        s = tf.cast(tf.scalar_mul(self.nw.s, self.scale(z)), dt)
        t = tf.cast(tf.scalar_mul(self.nw.t, self.transl(z)), dt)
        q = tf.cast(tf.scalar_mul(self.nw.q, self.transf(z)), dt)
        assert (
            isinstance(s, Tensor)
            and isinstance(t, Tensor)
            and isinstance(q, Tensor)
        )

        # return (
        #     tf.cast(self.nw.s * scale, inputs[0].dtype),
        #     tf.cast(self.nw.t * transl, inputs[0].dtype),
        #     tf.cast(self.nw.q * transf, inputs[0].dtype)
        # )
        return s, t, q


class NetworkFactory(BaseNetworkFactory):
    def build_networks(
            self,
            n: int,
            split_xnets: bool,
            group: U1Phase | SU3
    ) -> dict:
        """Build LeapfrogNetwork."""
        # TODO: if n == 0: build hmcNetwork (return zeros)
        assert n >= 1, 'Must build at least one network'

        cfg = self.get_build_configs()
        if n == 1:
            return {
                'xnet': LeapfrogLayer(
                    **cfg['xnet'],
                    group=group
                ),
                'vnet': LeapfrogLayer(
                    **cfg['vnet'],
                    group=group
                ),
            }
        xnet = {}
        vnet = {}
        for lf in range(n):
            vnet[f'{lf}'] = LeapfrogLayer(
                **cfg['vnet'],
                name=f'vnet/{lf}',
                group=group,
            )
            if split_xnets:
                xnet[f'{lf}'] = {
                    'first': LeapfrogLayer(
                        **cfg['xnet'],
                        name=f'xnet/{lf}/first',
                        group=group,
                    ),
                    'second': LeapfrogLayer(
                        **cfg['xnet'],
                        name=f'xnet/{lf}/second',
                        group=group,
                    ),
                }
            else:
                xnet[f'{lf}'] = LeapfrogLayer(
                    **cfg['xnet'],
                    name=f'xnet/{lf}',
                    group=group,
                )

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
