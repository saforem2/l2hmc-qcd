"""
tensorflow/network.py

Tensorflow implementation of the network used to train the L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from typing import Optional, Callable, Sequence

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
# Layers = tf.keras.layers.Layers
Layer = tf.keras.layers.Layer
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


class ScaledTanh(Layer):
    def __init__(
            self,
            out_features: int,
            name: Optional[str],
            **kwargs,
    ) -> None:
        super(ScaledTanh, self).__init__(name=name, **kwargs)
        self.out_features = out_features
        # self.coeff = tf.math.exp(
        # self.coeff = tf.Variable(
        #     trainable=True,
        #     dtype=TF_FLOAT,
        #     initializer='random_normal',
        #     trainable=True,
        #     shape=(1, self.out_features),
        #     # initial_value=tf.zeros(
        #     #     [1, out_features],
        #     #     dtype=TF_FLOAT
        #     # ),
        # )
        self.dense = Dense(out_features)  # , use_bias=False)

    def get_layer_weights(self) -> dict:
        return {
            'coeff': self.coeff,
            'dense/weight': self.dense.weights,
        }

    def get_weights_dict(
            self,
            sep: str = '/',
            name: Optional[str] = None,
    ) -> dict:
        name = self.name if name is None else name
        weights = self.get_layer_weights()
        return {
            sep.join([name, k]): v
            for k, v in weights.items()
        }

    def get_config(self):
        config = super(ScaledTanh, self).get_config()
        config.update({
            'out_features': self.out_features,
        })

    def build(self, input_shape):  # pyright:ignore
        """Create the state of the layer (weights)."""
        self.coeff = self.add_weight(
            name='coeff',
            shape=(1, self.out_features),
            initializer='zeros',
            trainable=True,
        )

    def call(self, x):
        return (
            tf.math.exp(self.coeff)
            * tf.math.tanh(self.dense(x))
        )


class ConvStack(Layer):
    def __init__(
            self,
            xshape: Sequence[int],
            conv_config: ConvolutionConfig,
            activation_fn: str | Callable,
            use_batch_norm: bool = False,
            name: Optional[str] = None,
            **kwargs,
    ) -> None:
        super(ConvStack, self).__init__(name=name, **kwargs)
        # xshape will be of the form
        # [batch, dim, *lattice_shape, *link_shape]
        # -- Cases ------------------------------------------------------
        # 1. xshape: [dim, nt, nx]
        self.conv_config = conv_config
        self.use_batch_norm = use_batch_norm
        if len(xshape) == 3:
            # lattice_shape: [nt, nx]
            # ny, nz = 0, 0
            # d, nt, nx = xshape[0], xshape[1], xshape[2]
            d, *latvol = xshape[0], *xshape[1:]

        # 2. xshape: [batch, dim, nt, nx]
        elif len(xshape) == 4:
            # lattice_shape: [nt, nx]
            # ny, nz = 0, 0
            # _, d, nt, nx = xshape[0], xshape[1], xshape[2], xshape[3]
            _, d, *latvol = xshape[0], xshape[1], *xshape[2:]

        # 3. xshape: [batch, dim, nt, nx, ny, nz, 3, 3]
        elif len(xshape) == 8:
            # link_shape: [3, 3]
            # lattice_shape = [nt, nx, ny, nz]
            # d, *latvol, *lshape = (
            #     xshape[1],
            #     *xshape[2:-2],
            #     *(xshape[-2], xshape[-1]),
            # )
            d = xshape[1]
            latvol = xshape[2:-2]
            # nt, nx, ny, nz = xshape[2], xshape[3], xshape[4], xshape[5]
        else:
            raise ValueError(f'Invalid value for xshape: {xshape}')
        # ---------------------------------------------------------------
        self.d = d
        self.latvol = latvol
        self.xdim = d * np.cumprod(latvol)[-1]
        # self.nt = nt
        # self.ny = ny
        # self.nz = nz
        # self.nx = nx
        self.xshape = xshape
        # self.xdim = d * nt * nx
        self.activation_fn = get_activation(activation_fn)
        self.flatten = Flatten()
        self.conv_layers = []
        # self._layers_dict = {}
        self._layers_names = []
        idx = 0
        if (filters := getattr(conv_config, 'filters', None)) is not None:
            if (nfilters := len(filters)) > 0:
                if (
                        conv_config.sizes is not None
                        and nfilters == len(conv_config.sizes)
                ):
                    for idx, (f, n) in enumerate(
                            zip(filters, conv_config.sizes)
                    ):
                        self.conv_layers.append(
                            PeriodicPadding(n - 1)
                        )
                        cname = f'Conv2D-{idx}'
                        self._layers_names.append(cname)
                        self.conv_layers.append(
                            Conv2D(
                                filters=f,
                                kernel_size=n,
                                activation=self.activation_fn,
                                name=cname,
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
        self.batch_norm = None
        if use_batch_norm:
            self.batch_norm = BatchNormalization(-1)

        self.output_layer = Dense(
                self.xdim,
                activation=self.activation_fn
        )

    def get_layer_weights(self) -> dict:
        weights = {}
        # for idx, layer in self.conv_layers:
        for name, layer in zip(self._layers_names, self.conv_layers):
            lweights = getattr(layer, 'weights', [])
            if len(lweights) > 0:
                weights.update({
                    f'{name}/weight': lweights
                })
        w, b = self.output_layer.weights
        if self.batch_norm is not None:
            assert isinstance(self.batch_norm, tf.keras.layers.BatchNorm)
            g, b, m, s = self.batch_norm.weights
            pre = 'batch_norm'
            weights.update({
                f'{pre}/gamma': g,
                f'{pre}/beta': b,
                f'{pre}/moving_avg': m,
                f'{pre}/moving_std': s,
            })
        weights.update({
            'DenseOutput.weight': w,
            'DenseOutput.bias': b,
        })

        return weights

    def get_weights_dict(
            self,
            sep: str = '/',
            name: Optional[str] = None,
    ) -> dict:
        name = self.name if name is None else name
        weights = self.get_layer_weights()
        return {
            sep.join([name, k]): v
            for k, v in weights.items()
        }

    def get_config(self):
        config = super(ConvStack, self).get_config()
        config.update({
            'xshape': self.xshape,
            'conv_config': self.conv_config,
            'activation_fn': self.activation_fn,
            'use_batch_norm': self.use_batch_norm,
        })

    def call(self, x: Tensor, training: Optional[bool] = None) -> Tensor:
        if x.shape != self.xshape:
            if len(x.shape) == 2:
                try:
                    x = tf.reshape(
                        x,
                        (x.shape[0], self.d + 2, *self.latvol),
                        # [x.shape[0], self.d + 2, self.nt, self.nx]
                    )
                except ValueError:
                    x = tf.reshape(
                        x,
                        (x.shape[0], self.d, *self.latvol),
                        # [x.shape[0], self.d, self.nt, self.nx]
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

        x = self.flatten(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x, training=training)

        x = self.output_layer(x)

        return x


class InputLayer(Model):
    def __init__(
            self,
            xshape: Sequence[int],
            network_config: NetworkConfig,
            activation_fn: str | Callable[[Tensor], Tensor],
            conv_config: Optional[ConvolutionConfig] = None,
            input_shapes: Optional[dict[str, int | Sequence[int]]] = None,
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
        filters = getattr(conv_config, 'filters', [])
        if conv_config is not None and len(filters) > 0:
            self.conv_stack = ConvStack(
                xshape=xshape,
                conv_config=conv_config,
                activation_fn=self.activation_fn,
            )

        self.xlayer = Dense(self.net_config.units[0])
        self.vlayer = Dense(self.net_config.units[0])

    def get_weights_dict(self) -> dict:
        weights = {}
        filters = getattr(self.conv_config, 'filters', [])
        if (
                self.conv_config is not None
                and len(filters) > 0
                and self.conv_stack is not None
        ):
            if self.conv_stack is not None:
                wd = self.conv_stack.get_weights_dict()
                weights.update({
                    f'{self.name}/ConvStack': wd,
                })
        # if self.conv_stack is not None:
        #     weights.update({
        #         f'{self.name}.ConvStack': self.conv_stack.get_weights_dict()
        #     })

        xw, xb = self.xlayer.weights
        vw, vb = self.vlayer.weights
        weights.update({
            f'{self.name}/xlayer/w': xw,
            f'{self.name}/xlayer/b': xb,
            f'{self.name}/vlayer/w': vw,
            f'{self.name}/vlayer/b': vb,
        })

        return weights

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


class LeapfrogLayer(Model):
    def __init__(
            self,
            xshape: Sequence[int],
            network_config: NetworkConfig,
            group: Optional[U1Phase | SU3 | str] = None,
            input_shapes: Optional[dict[str, int | Sequence[int]]] = None,
            net_weight: Optional[NetWeight] = None,
            conv_config: Optional[ConvolutionConfig] = None,
            name: Optional[str] = None,
    ):
        super(LeapfrogLayer, self).__init__(name=name)
        if net_weight is None:
            net_weight = NetWeight(1., 1., 1.)

        self.xshape = xshape
        self.nw = net_weight
        self.net_config = network_config
        # self._name = name if name is not None else 'network'
        self.xdim = np.cumprod(xshape[1:])[-1]
        act_fn = get_activation(self.net_config.activation_fn)
        self.activation_fn = act_fn
        self.g = group
        if group is not None:
            if isinstance(group, str):
                if group.lower() in ['u1', 'u1phase']:
                    self.g = U1Phase()
                elif group.lower() == 'su3':
                    self.g = SU3()
                else:
                    raise ValueError(f'Unexpected str {group} for group')

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
            h = Dense(units, name=f'hidden.{idx}')
            self.hidden_layers.append(h)

        self.scale = ScaledTanh(self.xdim, name='scale')
        self.transf = ScaledTanh(self.xdim, name='transf')
        self.transl = Dense(self.xdim, name='transl')

        self.dropout = None
        if self.net_config.dropout_prob > 0.:
            self.dropout = Dropout(self.net_config.dropout_prob)

        self.batch_norm = None
        if self.net_config.use_batch_norm:
            self.batch_norm = BatchNormalization(-1, name=f'{name}_batchnorm')

    def get_layer_weights(self) -> dict:
        weights = {}
        # iweights = self.input_layer.get_weights_dict()
        weights.update({
            'input_layer': self.input_layer.get_weights_dict()
        })
        for idx, layer in enumerate(self.hidden_layers):
            w, b = layer.weights
            weights.update({})
            weights[f'hidden_layers.{idx}.weight'] = w
            weights[f'hidden_layers.{idx}.bias'] = b

        weights.update({
            'scale': self.scale.get_weights_dict()
        })
        tw, tb = self.transl.weights
        weights.update({
            'transl.weight': tw,
            'transl.bias': tb,
        })
        weights.update({
            'transf': self.transf.get_weights_dict()
        })

        return weights

    def get_weights_dict(
            self,
            sep: str = '/',
            name: Optional[str] = None,
    ) -> dict:
        name = self.name if name is None else name
        weights = self.get_layer_weights()
        return {
            sep.join([name, k]): v for k, v in weights.items()
        }

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

        # dt = inputs[0].dtype
        # s = tf.cast(tf.scalar_mul(self.nw.s, self.scale(z)), dt)
        # t = tf.cast(tf.scalar_mul(self.nw.t, self.transl(z)), dt)
        # q = tf.cast(tf.scalar_mul(self.nw.q, self.transf(z)), dt)
        s = tf.scalar_mul(self.nw.s, self.scale(z))
        t = tf.scalar_mul(self.nw.t, self.transl(z))
        q = tf.scalar_mul(self.nw.q, self.transf(z))
        # s = self.nw.s * self.scale(z)
        # t = self.nw.t * self.transl(z)
        # q = self.nw.q * self.transf(z)
        # assert (
        #     isinstance(s, Tensor)
        #     and isinstance(t, Tensor)
        #     and isinstance(q, Tensor)
        # )
        # s = self.nw.s * self.scale(z)
        # t = self.nw.t * self.transl(z)
        # q = self.nw.q * self.transf(z)

        # return (
        #     tf.cast(tf.multiply(self.nw.s, s), x.dtype),
        #     tf.cast(tf.multiply(self.nw.t, t), x.dtype),
        #     tf.cast(tf.multiply(self.nw.q, q), x.dtype)
        # )
        # return (
        #     tf.cast(s, x.dtype),
        #     tf.cast(t, x.dtype),
        #     tf.cast(q, x.dtype)
        # )
        # return (s, t, q)  # type:ignore
        return (
            tf.cast(s, TF_FLOAT),
            tf.cast(t, TF_FLOAT),
            tf.cast(q, TF_FLOAT)
        )


class NetworkFactory(BaseNetworkFactory):
    def build_xnet(
            self,
            group: SU3 | U1Phase,
            name: Optional[str] = None,
    ) -> LeapfrogLayer:
        xname = 'xnet' if name is None else f'xnet/{name}'
        return LeapfrogLayer(
            xshape=self.input_spec.xshape,
            network_config=self.network_config,
            group=group,
            input_shapes=self.input_spec.xnet,
            net_weight=self.nw.x,
            conv_config=self.conv_config,
            name=xname,
        )

    def build_vnet(
            self,
            group: SU3 | U1Phase,
            name: Optional[str] = None,
    ) -> LeapfrogLayer:
        vname = 'vnet' if name is None else f'vnet/{name}'
        return LeapfrogLayer(
            xshape=self.input_spec.xshape,
            network_config=self.network_config,
            group=group,
            input_shapes=self.input_spec.vnet,
            net_weight=self.nw.v,
            conv_config=self.conv_config,
            name=vname,
        )

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
                    name='xnet',
                    group=group
                ),
                'vnet': LeapfrogLayer(
                    **cfg['vnet'],
                    name='vnet',
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
