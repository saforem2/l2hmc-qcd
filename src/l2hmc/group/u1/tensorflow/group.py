"""
group/u1/tensorflow/group.py

Implements U1Phase for TensorFlow
"""
from __future__ import absolute_import, division, annotations, print_function
from typing import Any

import numpy as np
import tensorflow as tf

from l2hmc.group.group import Group

Tensor = tf.Tensor
TF_FLOAT = tf.dtypes.as_dtype(tf.keras.backend.floatx())
PI = np.pi
# PI = tf.cast(np.pi, TF_FLOAT)
TWO_PI = 2. * PI


class U1Phase(Group):
    def __init__(self):
        super(U1Phase, self).__init__(dim=2, shape=[1], dtype=TF_FLOAT)

    def phase_to_coords(self, phi: Tensor) -> Tensor:
        """Convert complex to Cartesian.

        exp(i φ) --> [cos φ, sin φ]
        """
        coords = [
            tf.math.cos(phi),
            tf.math.sin(phi)
        ]
        return tf.convert_to_tensor(
            tf.concat(coords, axis=-1)
        )

    def coords_to_phase(self, x: Tensor) -> Tensor:
        """Convert Cartesian to phase.

        [cos φ, sin φ] --> atan(sin φ / cos φ)
        """
        # xT = tf.transpose(x)
        # *_, x1T, x2T = xT
        # x1 = tf.transpose(x1T)
        # x2 = tf.transpose(x2T)
        # return tf.convert_to_tensor(
        #     tf.math.atan2(x1, x2)
        # )
        assert x.shape[-1] == 2
        return tf.convert_to_tensor(
            tf.math.atan2(
                x[..., -1],  # type:ignore
                x[..., -2]   # type:ignore
            )
        )

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

    @staticmethod
    def to_tensor(x: Any) -> Tensor:
        return tf.convert_to_tensor(x)

    def adjoint(self, x: Tensor) -> Tensor:
        return self.to_tensor(tf.math.negative(x))

    def trace(self, x: Tensor) -> Tensor:
        return self.to_tensor(tf.math.cos(x))

    def diff_trace(self, x: Tensor) -> Tensor:
        return self.to_tensor(tf.math.negative(tf.math.sin(x)))

    def diff2trace(self, x: Tensor) -> Tensor:
        return self.to_tensor(tf.math.negative(tf.math.cos(x)))

    def floormod(self, x: Tensor, y: Tensor) -> Tensor:
        # return (x - tf.math.floordiv(x, y) * y)
        # z = self.to_tensor(
        #     tf.math.multiply(
        #         y,
        #         tf.math.floordiv(x, y)
        #     )
        # )
        # return self.to_tensor(tf.subtract(x, z))
        return self.to_tensor(
            tf.subtract(
                x,
                tf.multiply(
                    y,
                    tf.math.floordiv(x, y)
                )
            )
        )

    @staticmethod
    def group_to_vec(x: Tensor) -> Tensor:
        return tf.convert_to_tensor(
            tf.concat([tf.math.cos(x), tf.math.sin(x)], axis=1)
        )

    @staticmethod
    def vec_to_group(x: Tensor) -> Tensor:
        if x.dtype in [tf.complex64, tf.complex128]:
            return tf.convert_to_tensor(
                tf.math.atan2(
                    tf.math.imag(x),
                    tf.math.real(x)
                )
            )
        return tf.math.atan2(
            x[..., -1],  # type:ignore
            x[..., -2],  # type:ignore
        )

    def compat_proj(self, x: Tensor) -> Tensor:
        pi = tf.constant(PI, dtype=x.dtype)
        return ((x + pi) % TWO_PI) - PI

    def random(self, shape: list[int]) -> Tensor:
        return self.compat_proj(
            TWO_PI * tf.random.uniform(shape, dtype=TF_FLOAT)
        )

    def random_momentum(self, shape: list[int]) -> Tensor:
        return tf.random.normal(shape, dtype=TF_FLOAT)

    def kinetic_energy(self, p: Tensor) -> Tensor:
        return 0.5 * tf.reduce_sum(
            tf.square(tf.reshape(p, [p.shape[0], -1])),
            axis=-1
        )
