"""
sun.py

Contains implementation of SUN class.
"""
from __future__ import absolute_import, division, print_function, annotations

import tensorflow as tf

from l2hmc.group.tensorflow.logm import log3x3

from tensorflow.types.experimental import TensorLike  # type: ignore


transpose = tf.transpose
conj = tf.math.conj
expm = tf.linalg.expm


def conjT(x: TensorLike) -> TensorLike:
    return transpose(conj(x), (-2, -1))


class SUN:
    def __init__(self) -> None:
        super(SUN, self).__init__()

    def exp(self, x: TensorLike, u: TensorLike) -> TensorLike:
        return x @ expm(conjT(x) @ u)

    def log(self, x: TensorLike, y: TensorLike) -> TensorLike:
        _, n, _ = x.shape
        assert n == 3, 'Operation only supported for SU(3)'
        return x @ log3x3(conjT(x) @ y)

    def proju(
            self,
            x: TensorLike,
            u: TensorLike,
    ) -> TensorLike:
        """Arbitrary matrix C projects to skew-Hermitian
                        B := (C - C†) / 2
        then make traceless with
                tr{B - [tr{B} / N] * I} = tr{B} - tr{B} = 0
        """
        _, n, _ = x.shape
        algebra_elem = tf.linalg.solve(u, x)[0]  # X^{-1} u
        # do projection in Lie algebra
        B = (algebra_elem - conjT(algebra_elem)) / 2
        trace = tf.linalg.einsum('bii->b', B)
        B = B - (
            (1 / n) * tf.expand_dims(tf.expand_dims(trace, -1), -1)
            * tf.repeat(tf.eye(n), (x.shape[0], 1, 1))
        )
        assert (
            tf.math.abs(tf.reduce_mean(tf.linalg.einsum('bii->b', B))) < 1e-6
        )

        return B
