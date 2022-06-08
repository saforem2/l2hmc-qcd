"""
group.py

Contains implementations of various (special) unitary groups.

Date: 2022/03/25

Original version from:

  https://github.com/nftqcd/nthmc/blob/master/lib/group.py

written by Xiao-Yong Jin
"""
from __future__ import absolute_import, division, print_function, annotations


import numpy as np
import tensorflow as tf
import math
from typing import Callable

from l2hmc.group.tensorflow.utils import (
    projectTAH,
    projectSU,
    randTAH3,
    norm2,
    vec_to_su3,
    su3_to_vec
)

# from tensorflow.types.experimental import TensorLike  # type:ignore

Array = np.array
Tensor = tf.Tensor
PI = tf.convert_to_tensor(math.pi)
SQRT1by3 = tf.math.sqrt(1. / 3.)

TF_FLOAT = tf.keras.backend.floatx()

# def adj(x: Array):
#     return x.conj().T


class Group:
    """Gauge group represented as matrices in the last two dims in tensors."""
    def mul(
            self,
            a: Tensor,
            b: Tensor,
            adjoint_a: bool = False,
            adjoint_b: bool = False
    ) -> Tensor:
        return tf.linalg.matmul(a, b, adjoint_a=adjoint_a, adjoint_b=adjoint_b)


class U1Phase(Group):
    dtype = TF_FLOAT
    size = [1]
    shape = (1)

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

    def compat_proj(self, x: Tensor) -> Tensor:
        return tf.math.floormod(x + PI, 2 * PI) - PI

    def random(self, shape: list[int]):
        return self.compat_proj(tf.random.uniform(shape, *(-4, 4)))

    def random_momentum(self, shape: list[int]) -> Tensor:
        return tf.random.normal(shape)

    def kinetic_energy(self, p: Tensor) -> Tensor:
        return 0.5 * tf.reduce_sum(
            tf.square(tf.reshape(p, [p.shape[0], -1])),
            axis=1
        )


class SU3(Group):
    dtype = tf.complex128
    size = [3, 3]
    shape = (3, 3)

    def update_gauge(
            self,
            x: Tensor,
            p: Tensor,
    ) -> Tensor:
        return self.mul(self.exp(p), x)

    def mul(
            self,
            a: Tensor,
            b: Tensor,
            adjoint_a: bool = False,
            adjoint_b: bool = False,
    ) -> Tensor:
        return tf.linalg.matmul(a, b, adjoint_a=adjoint_a, adjoint_b=adjoint_b)

    def adjoint(self, x: Tensor) -> Tensor:
        return tf.linalg.adjoint(x)

    def trace(self, x: Tensor) -> Tensor:
        return tf.linalg.trace(x)

    def diff_trace(self, x: Tensor):  # type: ignore
        print('TODO')

    def diff2Trace(self, x: Tensor):  # -> Tensor:
        print('TODO')

    def exp(self, x: Tensor) -> Tensor:
        # return exp(x)
        return tf.linalg.expm(x)

    def projectTAH(self, x: Tensor) -> Tensor:
        return projectTAH(x)

    def compat_proj(self, x: Tensor) -> Tensor:
        """Arbitrary matrix C projects to skew-hermitian B := (C - C^H) / 2

        Make traceless with tr(B - (tr(B) / N) * I) = tr(B) - tr(B) = 0
        """
        return projectSU(x)

    def random(self, shape: list[int]) -> Tensor:
        r = tf.random.normal(shape, dtype=TF_FLOAT)
        i = tf.random.normal(shape, dtype=TF_FLOAT)
        return projectSU(tf.dtypes.complex(r, i))

    def random_momentum(self, shape: list[int]) -> Tensor:
        return randTAH3(shape[:-2])

    def kinetic_energy(self, p: Tensor) -> Tensor:
        p2 = norm2(p) - tf.constant(8.0)  # - 8.0 ??
        return (
            0.5 * tf.math.reduce_sum(
                tf.reshape(p2, [p.shape[0], -1]), axis=1
            )
        )
        # return 0.5 * tf.reduce_sum(
        #     tf.math.real(tf.linalg.matmul(tf.math.conj(p), p)),
        #     axis=1
        # )
        # return 0.5 * (norm2(p) - 8.0)

    def vec_to_group(self, x: Tensor) -> Tensor:
        return vec_to_su3(x)

    def group_to_vec(self, x: Tensor) -> Tensor:
        return su3_to_vec(x)


def SU3GradientTF(
        f: Callable[[Tensor], Tensor],
        x: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute gradient using TensorFlow GradientTape.

    y = f(x) must be a real scalar value.

    Returns:
      - (f(x), D), where D = T^a D^a = T^a ∂_a f(x)

    NOTE: Use real vector derivatives, e.g.
      D^a = ∂_a f(x)
          = ∂_t f(exp(T^a) x) |_{t=0}
    """
    zeros = tf.zeros(8)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(zeros)
        y = f(tf.linalg.matmul(tf.linalg.expm(vec_to_su3(zeros)), x))
        # y = f(tf.linalg.matmul(exp(vec_to_su3(zeros)), x))
        # y = f(tf.linalg.matmul(exp(su3fromvec(zeros)), x))
    d = tape.gradient(y, zeros)

    return y, d
