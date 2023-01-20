"""
group/su3/tensorflow/group.py

Contains TensorFlow implementation of SU3 object.
"""
from __future__ import absolute_import, division, print_function, annotations

import tensorflow as tf
import logging

from l2hmc.group.group import Group

from l2hmc.group.su3.tensorflow.utils import (
    projectTAH,
    projectSU,
    checkU,
    checkSU,
    randTAH3,
    norm2,
    vec_to_su3,
    su3_to_vec
)


log = logging.getLogger(__name__)

Tensor = tf.Tensor
TF_FLOAT = tf.dtypes.as_dtype(tf.keras.backend.floatx())


class SU3(Group):
    def __init__(self):
        self._nc = 3
        self._free_params = 8
        super().__init__(
            dim=4,
            shape=[3, 3],
            dtype=tf.complex128
        )

    def update_gauge(
            self,
            x: Tensor,
            p: Tensor,
    ) -> Tensor:
        return tf.matmul(tf.linalg.expm(p), x)

    def checkSU(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns the average and maximum of the sumf of deviations of:
             - X† X
             - det(x)
        from unitarity
        """
        return checkSU(x)

    def checkU(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns the average and maximum of
        the sum of the deviations of X†X
        """
        return checkU(x)

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
        log.error('TODO')
        return x

    def diff2Trace(self, x: Tensor):  # -> Tensor:
        log.error('TODO')
        return x

    def exp(self, x: Tensor) -> Tensor:
        return tf.linalg.expm(x)

    def projectTAH(self, x: Tensor) -> Tensor:
        """Returns R = 1/2 (X - X†) - 1/(2 N) tr(X - X†)
        R = - T^a tr[T^a (X - X†)]
          = T^a ∂_a (- tr[X + X†])
        """
        return projectTAH(x)

    def compat_proj(self, x: Tensor) -> Tensor:
        """Arbitrary matrix C projects to skew-hermitian B := (C - C^H) / 2

        Make traceless with tr(B - (tr(B) / N) * I) = tr(B) - tr(B) = 0
        """
        return projectSU(x)

    def random(self, shape: list[int]) -> Tensor:
        r = tf.random.normal(shape, dtype=TF_FLOAT)
        i = tf.random.normal(shape, dtype=TF_FLOAT)
        # return self.compat_proj(tf.dtypes.complex(r, i))
        return projectSU(tf.complex(r, i))

    def random_momentum(self, shape: list[int]) -> Tensor:
        return randTAH3(shape[:-2])

    def kinetic_energy(self, p: Tensor) -> Tensor:
        p2 = norm2(p) - tf.constant(8.0, dtype=TF_FLOAT)
        return (
            0.5 * tf.math.reduce_sum(
                tf.reshape(p2, [p.shape[0], -1]), axis=1
            )
        )

    def vec_to_group(self, x: Tensor) -> Tensor:
        """
        Returns batched SU(3) matrices.

        X = X^a T^a
        tr{X T^b} = X^a tr{T^a T^b} = X^a (-1/2) 𝛅^{ab} = -1/2 X^b
        X^a = -2 X_ij T^a_ji
        """
        return self.compat_proj(vec_to_su3(x))

    def group_to_vec(self, x: Tensor) -> Tensor:
        """
        Returns (batched) 8 real numbers,
        X^a T^a = X - 1/3 tr(X)

        Convention:
            tr{T^a T^a} = -1/2
            X^a = - 2 tr[T^a X]
        """
        return su3_to_vec(self.compat_proj(x))
