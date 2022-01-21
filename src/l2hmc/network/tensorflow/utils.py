"""
tensorflow/utils.py

Contains helper functions for dealing with tensorflow Networks.
"""
from __future__ import absolute_import, print_function, annotations, division
import tensorflow as tf

Tensor = tf.Tensor
Layer = tf.keras.layers.Layer


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

    def call(self, v: Tensor) -> Tensor:
        """Call the layer in the foreward direction.
        NOTE: We assume inputs.shape = (batch, Nx, Ny, *)
        """
        assert len(v.shape) >= 3, 'Expected len(v.shape) >= 3'
        assert tf.is_tensor(v)
        assert isinstance(v, Tensor)
        # 1. pad along x axis
        # inputs = tf.concat([v[:, -self.size:], v, v[:, 0:self.size]], axis=1)
        x0 = v[:, -self.size:, :, ...]  # pyright: ignore
        x1 = v[:, 0:self.size, :, ...]  # pyright: ignore
        inputs = tf.concat([x0, v, x1], 1)

        # 2. pad along y axis
        y0 = v[:, :, -self.size:, ...]
        y1 = v[:, :, 0:self.size, ...]

        inputs = tf.concat([y0, inputs, y1], 2)

        return inputs

    def get_config(self):
        config = super(PeriodicPadding, self).get_config()
        config.update({'size': self.size})
        return config
