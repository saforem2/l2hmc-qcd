"""
tensorflow/utils.py

Contains helper functions for dealing with tensorflow Networks.
"""
from __future__ import absolute_import, print_function, annotations, division
import tensorflow as tf
import numpy as np

Tensor = tf.Tensor
Layer = tf.keras.layers.Layer


def zero_weights(model: tf.keras.Model):
    for layer in model.layers:
        weights = layer.get_weights()
        zeros = []
        for w in weights:
            zeros.append(np.zeros_like(w))

        layer.set_weights(zeros)

    return model


def periodic_padding(x: Tensor, padding: int = 1) -> Tensor:
    upper_pad = x[-padding:, :]  # type: ignore
    lower_pad = x[:padding, :]   # type: ignore
    partial_x = tf.concat([upper_pad, x, lower_pad], axis=0)
    left_pad = partial_x[:, -padding:]
    right_pad = partial_x[:, :padding]
    x_padded = tf.concat([left_pad, partial_x, right_pad], axis=1)

    return x_padded


# pylint:disable=unused-argument
class PeriodicPadding(Layer):
    """Implements a PeriodicPadding as a `tf.keras.layers.Layer` object."""
    def __init__(self, size: int, **kwargs):
        super(PeriodicPadding, self).__init__(**kwargs)
        self.size = size

    def call(self, x: Tensor) -> Tensor:
        """Call the layer in the foreward direction.
        NOTE: We assume inputs.shape = (batch, Nx, Ny, *)
        """
        assert len(x.shape) >= 3, 'Expected len(v.shape) >= 3'
        assert tf.is_tensor(x)
        assert isinstance(x, Tensor)
        # 1. pad along x axis
        # inputs = tf.concat([v[:, -self.size:], v, v[:, 0:self.size]], axis=1)
        x0 = x[:, -self.size:, ...]  # type: ignore
        x1 = x[:, 0:self.size, ...]  # type: ignore
        x = tf.concat([x0, x, x1], 1)

        # 2. pad along y axis
        y0 = x[:, :, -self.size:, ...]  # type: ignore
        y1 = x[:, :, 0:self.size, ...]  # type: ignore

        x = tf.concat([y0, x, y1], 2)

        return x

    def get_config(self):
        config = super(PeriodicPadding, self).get_config()
        config.update({'size': self.size})
        return config
