"""
logm.py
"""
from __future__ import absolute_import, print_function, division, annotations


import tensorflow as tf
from math import pi as PI
from tensorflow.types.experimental import TensorLike  # type:ignore

TWO_PI = 2. * PI


def charpoly3x3(x: TensorLike) -> tuple[TensorLike, TensorLike, TensorLike]:
    """ det(λ * I - A) = λ³ + λ² * c[3] + λ * c[2] + c[1] """
    # assert isinstance(x, tf.Tensor)
    # assert len(x.shape) >= 3
    elem1 = -(
        x[:, 0, 0] * (x[:, 1, 1] * x[:, 2, 2] - x[:, 1, 2] * x[:, 2, 1])
        - x[:, 1, 0] * (x[:, 0, 1] * x[:, 2, 2] - x[:, 0, 2] * x[:, 2, 1])
        + x[:, 2, 0] * (x[:, 0, 1] * x[:, 1, 2] - x[:, 0, 2] * x[:, 1, 1])
    )
    elem2 = (
        x[:, 0, 0] * x[:, 1, ]
        + x[:, 0, 0] * x[:, 2, 2]
        + x[:, 1, 1] * x[:, 2, 2]
        - x[:, 1, 0] * x[:, 0, 1]
        - x[:, 2, 0] * x[:, 0, 2]
        - x[:, 2, 1] * x[:, 1, 2]
    )
    elem3 = -(x[:, 0, 0] + x[:, 1, 1] + x[:, 2, 2])

    return (elem1, elem2, elem3)


def cmax(x: TensorLike, y: TensorLike) -> TensorLike:
    """Returns the largest-magnitude complex number"""
    return tf.where(tf.math.abs(x) > tf.math.abs(y), x, y)


def cubic_zeros(p: tuple[TensorLike, TensorLike, TensorLike]) -> TensorLike:
    a = 1
    b = p[2]
    c = p[1]
    d = p[0]
    d0 = (b ** 2) - (3 * a * c)
    d1 = (2 * b ** 3) - (9 * a * b * c + 27 * (a ** 2) * d)
    sqrt = tf.math.sqrt(1e-3 + (d1 ** 2) - 4 * (d0 ** 3))
    v = cmax((d1 + sqrt) / 2, (d1 - sqrt) / 2)
    c = (v ** 1/3)
    w = tf.math.exp(TWO_PI * 1j / 3)

    return [
        -(b + ((w ** k) * c) + d0 / ((w ** k) * c)) / (3 * a) for k in range(3)
    ]


def su3_to_eigs(x: TensorLike) -> TensorLike:
    p = charpoly3x3(x)
    zs = cubic_zeros(p)
    return tf.concat([tf.expand_dims(x, -1) for x in zs], axis=-1)


def log3x3(x: TensorLike):
    eigs = su3_to_eigs(x)
    q, _ = tf.linalg.solve(
        tf.expand_dims(tf.math.log(eigs), -1), (
            1e-6 * tf.expand_dims(tf.eye(3), 0)
            + tf.expand_dims(eigs, -1) ** (
                tf.expand_dims(tf.expand_dims(tf.constant([0, 1, 2]), 0), 0)
            )
        )
    )
    q = tf.expand_dims(q, -1)
    eye = tf.reshape(tf.eye(x.shape[-1]), (1, x.shape[-1], x.shape[-1]))
    eye = tf.repeat(x.shape[0], 1, 1)

    return q[:, 0] * eye + q[:, 1] * x + q[:, 2] * x @ x
