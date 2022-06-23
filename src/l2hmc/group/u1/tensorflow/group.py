"""
group/u1/tensorflow/group.py

Implements U1Phase for TensorFlow
"""
from __future__ import absolute_import, division, annotations, print_function

import numpy as np
import tensorflow as tf

from l2hmc.group.group import Group

Tensor = tf.Tensor
TF_FLOAT = tf.keras.backend.floatx()
PI = np.pi
# PI = tf.cast(np.pi, TF_FLOAT)
TWO_PI = 2. * PI


class U1Phase(Group):
    def __init__(self):
        super().__init__(dim=2, shape=[1], dtype=TF_FLOAT)

    def exp(self, x: Tensor) -> Tensor:
        return tf.complex(tf.math.cos(x), tf.math.sin(x))

    def update_gauge(
            self,
            x: Tensor,
            p: Tensor,
    ) -> Tensor:
        return tf.add(x, p)

    def mul(
            self,
            a: Tensor,
            b: Tensor,
            adjoint_a: bool = False,
            adjoint_b: bool = False
    ) -> Tensor:
        if adjoint_a and adjoint_b:
            return tf.subtract(tf.math.negative(a), b)
        elif adjoint_a:
            return tf.add(tf.math.negative(a), b)
        elif adjoint_b:
            return tf.subtract(a, b)
        else:
            return tf.add(a, b)

    def adjoint(self, x: Tensor) -> Tensor:
        return tf.math.negative(x)

    def trace(self, x: Tensor) -> Tensor:
        return tf.math.cos(x)

    def diff_trace(self, x: Tensor) -> Tensor:
        return tf.math.negative(tf.math.sin(x))

    def diff2trace(self, x: Tensor) -> Tensor:
        return tf.math.negative(tf.math.cos(x))

    def floormod(self, x: Tensor, y: Tensor) -> Tensor:
        return (x - tf.math.floordiv(x, y) * y)

    def compat_proj(self, x: Tensor) -> Tensor:
        # return tf.math.floormod(x + PI, 2. * PI) - PI
        # return (x + PI % (2 * PI)) - PI
        # return self.floormod(x + PI, (2. * PI)) - PI
        pi = tf.constant(PI, dtype=x.dtype)
        return ((x + pi) % TWO_PI) - PI

    @staticmethod
    def group_to_vec(x):
        return tf.stack([
            tf.math.cos(x),
            tf.math.sin(x)
        ], axis=-1)

    def random(self, shape: list[int]):
        return self.compat_proj(
            # tf.random.uniform(shape, *(-4, 4), dtype=TF_FLOAT)
            TWO_PI * tf.random.uniform(shape, dtype=TF_FLOAT)
        )

    def random_momentum(self, shape: list[int]) -> Tensor:
        return tf.random.normal(shape, dtype=TF_FLOAT)

    def kinetic_energy(self, p: Tensor) -> Tensor:
        return 0.5 * tf.reduce_sum(
            tf.square(tf.reshape(p, [p.shape[0], -1])),
            axis=1
        )
