"""
group/tensorflow/utils.py

"""
from __future__ import absolute_import, annotations, division, print_function
from math import pi as PI
from typing import Callable, Optional

import numpy as np
import tensorflow as tf


# from l2hmc.group.pytorch.logm import charpoly3x3, su3_to_eigs, log3x3


Array = np.array
Tensor = tf.Tensor

TF_FLOAT = tf.keras.backend.floatx()
TF_COMPLEX = tf.complex64 if TF_FLOAT == tf.float32 else tf.complex128

ONE_HALF = 1. / 2.
ONE_THIRD = 1. / 3.
TWO_PI = tf.constant(2. * PI, dtype=TF_FLOAT)

# SQRT1by2 = tf.constant(np.sqrt(1. / 2.), dtype=TF_FLOAT)
# SQRT1by3 = tf.constant(np.sqrt(1. / 3.), dtype=TF_FLOAT)
SQRT1by2 = tf.cast(tf.math.sqrt(1. / 2.), dtype=TF_FLOAT)
SQRT1by3 = tf.cast(tf.math.sqrt(1. / 3.), dtype=TF_FLOAT)


# Array = np.array
# Tensor = tf.Tensor
PI = tf.convert_to_tensor(PI)
# SQRT1by3 = tf.math.sqrt(1. / 3.)

# --------------------------------------------
# For SU(3) f^abc v[..., c]
# [T^a, T^b] = f^abc T^c
# --------------------------------------------
f012 = +1.0
f036 = +0.5
f045 = -0.5
f135 = +0.5
f146 = +0.5
f234 = +0.5
f256 = -0.5
f347 = +0.86602540378443864676    # SQRT(3/4)
f567 = +0.86602540378443864676


# --------------------------------------------
# For d^abc v[..., c]
# {T^a, T^b} = (-1 / 36) δ^ab + i d^abc T^c
# --------------------------------------------
d007 = -0.57735026918962576451    # -sqrt(1/3)
d035 = -0.5
d046 = -0.5
d117 = -0.57735026918962576451
d136 = 0.5
d145 = -0.5
d227 = -0.57735026918962576451
d233 = -0.5
d244 = -0.5
d255 = 0.5
d266 = 0.5
d337 = 0.28867513459481288225    # sqrt(1/3)/2
d447 = 0.28867513459481288225
d557 = 0.28867513459481288225
d667 = 0.28867513459481288225
d777 = 0.57735026918962576451


def norm2(x: Tensor, axis=[-2, -1]) -> Tensor:
    """No reduction if axis is empty"""
    n = tf.math.real(tf.math.multiply(tf.math.conj(x), x))
    if len(axis) == 0:
        return n

    return tf.math.reduce_sum(n, axis=axis)


def norm2_new(
        x: Tensor,
        axis: Optional[list[int]] = None,
        # allreduce: Optional[bool] = True,
        exclude: Optional[list[int]] = None,
) -> Tensor:
    """No reduction if axis is empty"""
    axis = [-2, -1] if axis is None else axis
    if x.dtype in [tf.complex64, tf.complex128]:
        x = tf.abs(x)
    n = tf.math.square(x)
    if exclude is None:
        if len(axis) == 0:
            return n
        return tf.math.reduce_sum(n, axis=axis)
    return tf.math.reduce_sum(
        n,
        axis=[i for i in range(len(n.shape)) if i not in exclude]
    )


# Converted from qex/src/maths/matrixFunctions.nim
# Last two dims in a tensor contain matrices.
# WARNING: below only works for SU3 for now
def randTAH3(shape: list[int]):
    s2 = 0.70710678118654752440    # sqrt(1/2)
    s3 = 0.577350269189625750    # sqrt(1/3)
    r3 = s2 * tf.random.normal(shape, dtype=TF_FLOAT)
    r8 = s2 * s3 * tf.random.normal(shape, dtype=TF_FLOAT)
    m00 = tf.dtypes.complex(tf.cast(0.0,TF_FLOAT), r8+r3)
    m11 = tf.dtypes.complex(tf.cast(0.0,TF_FLOAT), r8-r3)
    m22 = tf.dtypes.complex(tf.cast(0.0,TF_FLOAT), -2*r8)
    # m00 = tf.dtypes.complex(0.0, r8+r3)
    # m11 = tf.dtypes.complex(0.0, r8-r3)
    # m22 = tf.dtypes.complex(0.0, -2*r8)
    r01 = s2 * tf.random.normal(shape, dtype=TF_FLOAT)
    r02 = s2 * tf.random.normal(shape, dtype=TF_FLOAT)
    r12 = s2 * tf.random.normal(shape, dtype=TF_FLOAT)
    i01 = s2 * tf.random.normal(shape, dtype=TF_FLOAT)
    i02 = s2 * tf.random.normal(shape, dtype=TF_FLOAT)
    i12 = s2 * tf.random.normal(shape, dtype=TF_FLOAT)
    m01 = tf.dtypes.complex(r01, i01)
    m10 = tf.dtypes.complex(-r01, i01)
    m02 = tf.dtypes.complex(r02, i02)
    m20 = tf.dtypes.complex(-r02, i02)
    m12 = tf.dtypes.complex(r12, i12)
    m21 = tf.dtypes.complex(-r12, i12)
    return tf.stack([
        tf.stack([m00, m10, m20], axis=-1),
        tf.stack([m01, m11, m21], axis=-1),
        tf.stack([m02, m12, m22], axis=-1),
    ], axis=-1)


def eigs3(tr, p2, det):
    tr3 = (1.0/3.0)*tr
    p23 = (1.0/3.0)*p2
    tr32 = tr3*tr3
    q = tf.math.abs(0.5*(p23-tr32))
    r = 0.25*tr3*(5*tr32-p2) - 0.5*det
    sq = tf.math.sqrt(q)
    sq3 = q*sq
    isq3 = 1.0/sq3
    maxv = tf.constant(3e38, shape=isq3.shape, dtype=isq3.dtype)
    minv = tf.constant(-3e38, shape=isq3.shape, dtype=isq3.dtype)
    isq3c = tf.math.minimum(maxv, tf.math.maximum(minv, isq3))
    rsq3c = r * isq3c
    maxv = tf.constant(1, shape=isq3.shape, dtype=isq3.dtype)
    minv = tf.constant(-1, shape=isq3.shape, dtype=isq3.dtype)
    rsq3 = tf.math.minimum(maxv, tf.math.maximum(minv, rsq3c))
    t = (1.0/3.0)*tf.math.acos(rsq3)
    st = tf.math.sin(t)
    ct = tf.math.cos(t)
    sqc = sq*ct
    sqs = 1.73205080756887729352*sq*st  # sqrt(3)
    ll = tr3 + sqc
    e0 = tr3 - 2*sqc
    e1 = ll + sqs
    e2 = ll - sqs
    return e0, e1, e2


def rsqrtPHM3f(
        tr: Tensor,
        p2: Tensor,
        det: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    l0, l1, l2 = eigs3(tr, p2, det)
    sl0 = tf.math.sqrt(tf.math.abs(l0))
    sl1 = tf.math.sqrt(tf.math.abs(l1))
    sl2 = tf.math.sqrt(tf.math.abs(l2))
    u = sl0 + sl1 + sl2
    w = sl0 * sl1 * sl2
    d = w*(sl0+sl1)*(sl0+sl2)*(sl1+sl2)
    di = 1.0/d
    c0 = (w*u*u+l0*sl0*(l1+l2)+l1*sl1*(l0+l2)+l2*sl2*(l0+l1))*di
    c1 = -(tr*u+w)*di
    c2 = u*di
    return c0, c1, c2


def rsqrtPHM3(x: Tensor) -> Tensor:
    tr = tf.math.real(tf.linalg.trace(x))
    x2 = tf.linalg.matmul(x, x)
    p2 = tf.math.real(tf.linalg.trace(x2))
    det = tf.math.real(tf.linalg.det(x))
    c0_, c1_, c2_ = rsqrtPHM3f(tr, p2, det)
    c0 = tf.cast(tf.reshape(c0_, c0_.shape + [1, 1]), x.dtype)
    c1 = tf.cast(tf.reshape(c1_, c1_.shape + [1, 1]), x.dtype)
    c2 = tf.cast(tf.reshape(c2_, c2_.shape + [1, 1]), x.dtype)
    # term0 = c0 * eyeOf(x)
    # term1 = tf.math.multiply(c1, x)
    # term2 = tf.math.multiply(c2, x2)
    # return term0 + term1 + term2
    # e3 = tf.eye(3, batch_shape=[1] * len(c0.shape), dtype=c0_.dtype)
    # term0 = tf.cast(c0_ * e3, x.dtype)
    # # term0 = tf.cast(
    # #     c0_ * tf.eye(3, batch_shape=[1] * len(c0.shape)),
    # #     x.dtype
    # # )
    # term1 = tf.multiply(x, tf.cast(c1_, x.dtype))
    # term2 = tf.multiply(x, tf.cast(c2_, x.dtype))
    # return term0 + term1 + term2
    return (
        c0 * eyeOf(x)
        + tf.math.multiply(c1, x)
        + c2 * x2
    )


def projectU(x: Tensor) -> Tensor:
    """x (x'x)^{-1/2}"""
    t = tf.linalg.adjoint(x) @ x
    t2 = rsqrtPHM3(t)
    return tf.linalg.matmul(x, t2)


def projectSU(x: Tensor) -> Tensor:
    nc = tf.constant(x.shape[-1], TF_FLOAT)
    m = projectU(x)
    d = tf.linalg.det(m)
    const = (1.0 / (-nc))
    at2 = tf.cast(
        tf.math.atan2(
            tf.math.imag(d),
            tf.math.real(d),
        ),
        TF_FLOAT
    )
    p = const * at2
    # p = (1.0 / (-nc)) * tf.math.atan2(tf.math.imag(d), tf.math.real(d))
    # y = m * tf.cast(
    y = tf.math.multiply(m, tf.cast(
        tf.reshape(
            tf.complex(
                tf.math.cos(p),
                tf.math.sin(p)
            ),
            p.shape + [1, 1]
        ),
        m.dtype
    ))

    # return tf.cast(y, TF_COMPLEX)
    return y


def projectTAH(x: Tensor) -> Tensor:
    """Returns R = 1/2 (X - X†) - 1/(2 N) tr(X - X†)
    R = - T^a tr[T^a (X - X†)]
      = T^a ∂_a (- tr[X + X†])
    """
    nc = tf.constant(x.shape[-1], dtype=x.dtype)
    r = 0.5 * (x - tf.linalg.adjoint(x))
    d = tf.linalg.trace(r) / nc
    r -= tf.reshape(d, d.shape + [1, 1]) * eyeOf(x)

    return r


def checkU(x: Tensor) -> tuple[Tensor, Tensor]:
    """Returns the average and maximum of the sum of the deviations of X†X"""
    nc = tf.constant(x.shape[-1], dtype=x.dtype)
    d = norm2(tf.linalg.matmul(x, x, adjoint_a=True) - eyeOf(x))
    a = tf.math.reduce_mean(d, axis=range(1, len(d.shape)))
    b = tf.math.reduce_max(d, axis=range(1, len(d.shape)))
    c = 2 * (nc * nc + 1)

    return tf.math.sqrt(a / c), tf.math.sqrt(b / c)


def checkSU(x: Tensor) -> tuple[Tensor, Tensor]:
    """Returns the average and maximum of the sumf of deviations of:
         - X† X
         - det(x)
    from unitarity
    """
    nc = tf.constant(x.shape[-1], dtype=TF_FLOAT)
    d = norm2(tf.linalg.matmul(x, x, adjoint_a=True) - eyeOf(x))
    d += norm2(-1 + tf.linalg.det(x), axis=[])  # type: ignore
    a = tf.cast(
        tf.math.reduce_mean(d, axis=range(1, len(d.shape))),
        TF_FLOAT
    )
    b = tf.cast(
        tf.math.reduce_max(d, axis=range(1, len(d.shape))),
        TF_FLOAT
    )
    c = tf.cast(
        2.0 * (nc * nc + 1),
        TF_FLOAT
    )
    # return tf.math.sqrt(a / c), tf.math.sqrt(b / c)
    return (
        tf.math.sqrt(tf.math.divide(a, c)),
        tf.math.sqrt(tf.math.divide(b, c))
    )


def su3_to_vec(x: Tensor) -> Tensor:
    """Only for x in 3x3 anti-Hermitian.

    Return 8 real numbers, X^a T^a = X - 1/3 tr(X)

    Convention: tr{T^a T^a} = -1/2
    X^a = - 2 tr[T^a X]
    """
    c = -2
    x00 = x[..., 0, 0]  # type:ignore
    x01 = x[..., 0, 1]  # type:ignore
    x11 = x[..., 1, 1]  # type:ignore
    x02 = x[..., 0, 2]  # type:ignore
    x12 = x[..., 1, 2]  # type:ignore
    x22 = x[..., 2, 2]  # type:ignore
    return tf.stack([
        c * tf.math.imag(x01),
        c * tf.math.real(x01),
        tf.math.imag(x11) - tf.math.imag(x00),
        c * tf.math.imag(x02),
        c * tf.math.real(x02),
        c * tf.math.imag(x12),
        c * tf.math.real(x12),
        SQRT1by3 * (
            2 * tf.math.imag(x22) - tf.math.imag(x11) - tf.math.imag(x00)
        ),
    ], axis=-1)


def vec_to_su3(v: Tensor) -> Tensor:
    """
    X = X^a T^a
    tr{X T^b} = X^a tr{T^a T^b} = X^a (-1/2) 𝛅^ab = -1/2 X^b
    X^a = -2 X_{ij} T^a_{ji}
    """
    s3 = 0.577350269189625751  # sqrt(1/3)
    c = -0.5
    # vT = v.T
    # assert vT is not None and len(vT.shape) > 0
    # vT = {str(idx): v.T for idx, v in enumerate(vT)}
    # zero = tf.zeros(vT['0'].shape, dtype=vt['0'].dtype)
    # zero = tf.zeros(vT['0'].shape, dtype=vT[0].dtype)
    # x01 = c * tf.dtypes.complex(vT[1].T, vT[0])
    # x02 = c * tf.dtypes.complex()
    assert len(v.shape) > 1
    vT = tf.transpose(v)
    v0 = vT[0].T
    v1 = vT[1].T
    v2 = vT[2].T
    v3 = vT[3].T
    v4 = vT[4].T
    v5 = vT[5].T
    v6 = vT[6].T
    v7 = vT[7].T
    zero = tf.zeros(v0.shape, dtype=v0.dtype)
    x01 = c * tf.dtypes.complex(v1, v0)
    x02 = c * tf.dtypes.complex(v4, v3)
    x12 = c * tf.dtypes.complex(v6, v5)
    x2i = s3 * v7
    x0i = c * (x2i + v2)
    x1i = c * (x2i - v2)

    # zero = tf.zeros(v[..., 0].shape, dtype=v[..., 0].dtype)
    # x01 = c * tf.dtypes.complex(v[..., 1], v[..., 0])        # type:ignore
    # x02 = c * tf.dtypes.complex(v[..., 4], v[..., 3])        # type:ignore
    # x12 = c * tf.dtypes.complex(v[..., 6], v[..., 5])        # type:ignore
    # x2i = s3 * v[..., 7]                                     # type:ignore
    # x0i = c * (x2i + v[..., 2])                              # type:ignore
    # x1i = c * (x2i - v[..., 2])                              # type:ignore

    def neg_conj(x: Tensor) -> Tensor:
        return tf.math.negative(tf.math.conj(x))

    # ----------------------------------------------------
    # NOTE: Returns matrix of the form:
    #
    #  M = [[ cmplx(x0i),        -x01*,       -x02*],
    #       [        x01,   cmplx(x1i),       -x12*],
    #       [        x02,          x12,  cmplx(x2i)]]
    # ----------------------------------------------------
    v1 = tf.stack([
        tf.dtypes.complex(zero, x0i), neg_conj(x01), neg_conj(x02)
    ], axis=-1)
    v2 = tf.stack([
        x01, tf.dtypes.complex(zero, x1i), neg_conj(x12)
    ], axis=-1)
    v3 = tf.stack([
        x02, x12, tf.dtypes.complex(zero, x2i)
    ], axis=-1)

    return tf.stack([v1, v2, v3])


def eyeOf(m):
    batch_shape = [1] * (len(m.shape) - 2)
    return tf.eye(*m.shape[-2:], batch_shape=batch_shape, dtype=m.dtype)


def exp(m: Tensor, order: int = 12):
    eye = eyeOf(m)
    x = eye + m / tf.constant(order)
    for i in tf.range(order-1, 0, -1):
        x = eye + tf.linalg.matmul(m, x) / tf.constant(tf.cast(i, m.dtype))

    return x


def su3fabc(v: tf.Tensor) -> Tensor:
    """
    returns f^{abc} v[..., c]
    [T^a, T^b] = f^abc T^c
    """
    vT = tf.transpose(v)
    a01 = (+f012) * vT[2]
    a01 = (+f012) * vT[2]
    a02 = (-f012) * vT[1]
    a03 = (+f036) * vT[6]
    a04 = (+f045) * vT[5]
    a05 = (-f045) * vT[4]
    a06 = (-f036) * vT[3]
    a12 = (+f012) * vT[0]
    a13 = (+f135) * vT[5]
    a14 = (+f146) * vT[6]
    a15 = (-f135) * vT[3]
    a16 = (-f146) * vT[4]
    a23 = (+f234) * vT[4]
    a24 = (-f234) * vT[3]
    a25 = (+f256) * vT[6]
    a26 = (-f256) * vT[5]
    a34 = (+f347) * vT[7] + f234 * vT[2]
    a35 = (+f135) * vT[1]
    a36 = (+f036) * vT[0]
    a37 = (-f347) * vT[4]
    a45 = (+f045) * vT[0]
    a46 = (+f146) * vT[1]
    a47 = (+f347) * vT[3]
    a56 = (+f567) * vT[7] + f256 * vT[2]
    a57 = (-f567) * vT[6]
    a67 = (+f567) * vT[5]
    zii = tf.zeros(vT[0].shape, dtype=vT[0].dtype)
    return tf.stack([
        tf.stack([+zii, -a01, -a02, -a03, -a04, -a05, -a06, +zii], -1),
        tf.stack([+a01, +zii, -a12, -a13, -a14, -a15, -a16, +zii], -1),
        tf.stack([+a02, +a12, +zii, -a23, -a24, -a25, -a26, +zii], -1),
        tf.stack([+a03, +a13, +a23, +zii, -a34, -a35, -a36, -a37], -1),
        tf.stack([+a04, +a14, +a24, +a34, +zii, -a45, -a46, -a47], -1),
        tf.stack([+a05, +a15, +a25, +a35, +a45, +zii, -a56, -a57], -1),
        tf.stack([+a06, +a16, +a26, +a36, +a46, +a56, +zii, -a67], -1),
        tf.stack([+zii, +zii, +zii, +a37, +a47, +a57, +a67, +zii], -1),
    ], -1).T


def su3dabc(v: Tensor) -> Tensor:
    """
    returns d^abc v[...,c]
    {T^a,T^b} = -1/3δ^ab + i d^abc T^c
    """
    # NOTE: negative sign of what's on wikipedia
    vT = tf.transpose(v)
    a00 = d007 * vT[7]
    a03 = d035 * vT[5]
    a04 = d046 * vT[6]
    a05 = d035 * vT[3]
    a06 = d046 * vT[4]
    a07 = d007 * vT[0]
    a11 = d117 * vT[7]
    a13 = d136 * vT[6]
    a14 = d145 * vT[5]
    a15 = d145 * vT[4]
    a16 = d136 * vT[3]
    a17 = d117 * vT[1]
    a22 = d227 * vT[7]
    a23 = d233 * vT[3]
    a24 = d244 * vT[4]
    a25 = d255 * vT[5]
    a26 = d266 * vT[6]
    a27 = d227 * vT[2]
    a33 = d337 * vT[7] + d233 * vT[2]
    a35 = d035 * vT[0]
    a36 = d136 * vT[1]
    a37 = d337 * vT[3]
    a44 = d447 * vT[7] + d244 * vT[2]
    a45 = d145 * vT[1]
    a46 = d046 * vT[0]
    a47 = d447 * vT[4]
    a55 = d557 * vT[7] + d255 * vT[2]
    a57 = d557 * vT[5]
    a66 = d667 * vT[7] + d266 * vT[2]
    a67 = d667 * vT[6]
    a77 = d777 * vT[7]
    zii = tf.zeros(vT[0].shape, dtype=vT[0].dtype)
    return tf.stack([
        tf.stack([a00, zii, zii, a03, a04, a05, a06, a07], -1),
        tf.stack([zii, a11, zii, a13, a14, a15, a16, a17], -1),
        tf.stack([zii, zii, a22, a23, a24, a25, a26, a27], -1),
        tf.stack([a03, a13, a23, a33, zii, a35, a36, a37], -1),
        tf.stack([a04, a14, a24, zii, a44, a45, a46, a47], -1),
        tf.stack([a05, a15, a25, a35, a45, a55, zii, a57], -1),
        tf.stack([a06, a16, a26, a36, a46, zii, a66, a67], -1),
        tf.stack([a07, a17, a27, a37, a47, a57, a67, a77], -1),
    ], axis=-1)


def SU3Ad(x: Tensor) -> Tensor:
    """
    X T^c X† = AdX T^c = T^b AdX^bc
    Input x must be in SU(3) group.
    AdX^bc = - 2 tr[T^b X T^c X†] = - 2 tr[T^c X† T^b X]
    """
    y = tf.expand_dims(x, -3)
    return su3_to_vec(
        tf.linalg.matmul(
            y, tf.linalg.matmul(su3gen(), y),
            adjoint_a=True
        )
    )


def su3ad(x: Tensor) -> Tensor:
    """
    adX^{ab} = - f^{abc} X^c = f^{abc} 2 tr(X T^c) = 2 tr(X [T^a, T^b])
    Input x must be in su(3) algebra.
    """
    return su3fabc(tf.negative(su3_to_vec(x)))


def su3adapply(adx: Tensor, y: Tensor) -> Tensor:
    """
    Note:
        adX(Y) = [X, Y]
        adX(T^b) = T^a adX^{ab}
                 = - T^a f^{abc} X^c
                 = X^c f^{cba} T^a
                 = X^c [T^c, T^b]
                 = [X, T^b]
    and
        adX(Y) = T^a adX^{ab} Y^b
               = T^a adX^{ab} (-2) tr{T^b Y}
    """
    return vec_to_su3(tf.linalg.matvec(adx, su3_to_vec(y)))


def gellMann() -> Tensor:
    s3 = 0.57735026918962576451    # sqrt(1/3)
    zero3 = tf.zeros([3, 3], dtype=tf.float64)
    return tf.stack([
        tf.dtypes.complex(
            tf.reshape(
                tf.constant([0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=tf.float64),
                [3, 3]
            ),
            zero3
        ),
        tf.dtypes.complex(
            zero3,
            tf.reshape(
                tf.constant([0, -1, 0, 1, 0, 0, 0, 0, 0], dtype=tf.float64),
                [3, 3]
            )
        ),
        tf.dtypes.complex(
            tf.reshape(
                tf.constant([1, 0, 0, 0, -1, 0, 0, 0, 0], dtype=tf.float64),
                [3, 3]
            ),
            zero3
        ),
        tf.dtypes.complex(
            tf.reshape(
                tf.constant([0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=tf.float64),
                [3, 3]
            ),
            zero3
        ),
        tf.dtypes.complex(
            zero3,
            tf.reshape(
                tf.constant([0, 0, -1, 0, 0, 0, 1, 0, 0], dtype=tf.float64),
                [3, 3]
            )
        ),
        tf.dtypes.complex(
            tf.reshape(
                tf.constant([0, 0, 0, 0, 0, 1, 0, 1, 0], dtype=tf.float64),
                [3, 3]
            ),
            zero3
        ),
        tf.dtypes.complex(
            zero3,
            tf.reshape(
                tf.constant([0, 0, 0, 0, 0, -1, 0, 1, 0], dtype=tf.float64),
                [3, 3]
            )
        ),
        s3*tf.dtypes.complex(
            tf.reshape(
                tf.constant([1, 0, 0, 0, 1, 0, 0, 0, -2], dtype=tf.float64),
                [3, 3]
            ),
            zero3
        )
    ])


_su3gen_private_global_cache_ = None


def su3gen() -> Tensor:
    """
    T[a,i,j] = T^a_ij
    Traceless Anti-Hermitian basis.  tr{T^a T^a} = -1/2
    """
    global _su3gen_private_global_cache_
    if _su3gen_private_global_cache_ is None:
        _su3gen_private_global_cache_ = tf.dtypes.complex(
            tf.constant(0, dtype=tf.float64),
            tf.constant(-0.5, dtype=tf.float64)
        ) * gellMann()
    return _su3gen_private_global_cache_


def diffprojectTAH(m: Tensor, p: Optional[Tensor] = None) -> Tensor:
    """
    returns ∂_c p^a = ∂_c projectTAH(m)^a = - tr[T^a (T^c M + M† T^c)]
    P^a = -2 tr[T^a {- T^d tr[T^d (M - M†)]}]
        = - tr[T^a (M - M†)]
        = - ∂_a tr[M + M†]
    ∂_c P^a = - tr[T^a (T^c M + M† T^c)]
            = - 1/2 tr[{T^a,T^c} (M+M†) + [T^a,T^c] (M-M†)]
            = - 1/2 tr[d^acb T^b i (M+M†) - 1/3 δ^ac (M+M†) + f^acb T^b (M-M†)]
            = - 1/2 { d^acb tr[T^b i(M+M†)] - 1/3 δ^ac tr(M+M†) - f^acb F^b }
            = - 1/2 { d^acb tr[T^b i(M+M†)] - 1/3 δ^ac tr(M+M†) + adF^ac }
    Note:
        T^a T^b = 1/2 {(f^abc + i d^abc) T^c - 1/3 δ^ab}
    """
    if p is None:
        p = projectTAH(m)
    mhalfadP = su3ad(tf.constant(-0.5) * p)
    Ms = m+tf.linalg.adjoint(m)
    trMs = tf.math.real(tf.linalg.trace(Ms))/6.0
    eye = tf.dtypes.complex(
        tf.constant(0, dtype=tf.float64),
        tf.constant(1, dtype=tf.float64)
    )
    # return (
    #     su3dabc(0.25*su3_to_vec(I*Ms))
    #     + tf.reshape(trMs,trMs.shape+[1,1])*eyeOf(mhalfadP)
    #     + mhalfadP
    # )
    return (
        su3dabc(tf.constant(0.25) * su3_to_vec(eye * Ms))
        + tf.reshape(trMs, trMs.shape + [1, 1]) * eyeOf(mhalfadP)
        + mhalfadP
    )


def diffprojectTAHCross(
        m: Tensor,
        x: Optional[Tensor] = None,
        Adx: Optional[Tensor] = None,
        p: Optional[Tensor] = None
) -> Tensor:
    """
    returns
        R^ac = ∇_c p^a
             = ∇_c projectTAH(X Y)^a
             = - ∇_c ∂_a tr[X Y + Y† X†],
    where M = X Y

    The derivatives ∂ is on X and ∇ is on Y.
    ∇_c P^a = - 2 ReTr[T^a X T^c Y]
            = - tr[T^a (X T^c X† X Y + Y† X† X T^c X†)]
            = - tr[T^a (T^b M + M† T^b)] AdX^bc
    """
    if Adx is None:
        if x is None:
            raise ValueError(
                'diffprojectTAHCross must either provide x or Adx.'
            )
        Adx = SU3Ad(x)
    return tf.linalg.matmul(diffprojectTAH(m, p), Adx)


def diffexp(adX: Tensor, order: int = 13) -> Tensor:
    """
    return
        J(X) = (1-exp{-adX})/adX
             = Σ_{k=0}^{∞} 1/(k+1)! (-adX)^k
    up to k=order

    [exp{-X(t)} d/dt exp{X(t)}]_ij
        = [J(X) d/dt X(t)]_ij
        = T^a_ij J(X)^ab (-2) T^b_kl [d/dt X(t)]_lk

    J(X) = 1 + 1/2 (-adX) (1 + 1/3 (-adX) (1 + 1/4 (-adX) (1 + ...)))
    J(x) ∂_t x
        = T^a J(x)^ab (-2) tr[T^b ∂_t x]
        = exp(-x) ∂_t exp(x)
    J(s x) ∂_t x = exp(-s x) ∂_t exp(s x)
    ∂_s J(s x) ∂_t x
        = - exp(-s x) x ∂_t exp(s x) + exp(-s x) ∂_t x exp(s x)
        = (- exp(-s x) x ∂_t exp(s x) + exp(-s x) [∂_t x] exp(s x)
           + exp(-s x) x ∂_t exp(s x))
        = exp(-s x) [∂_t x] exp(s x)
        = exp(-s adx) ∂_t x
        = Σ_k 1/k! (-1)^k s^k (adx)^k ∂_t x
    J(0) = 0
    J(x) ∂_t x
        = ∫_0^1 ds Σ_{k=0} 1/k! (-1)^k s^k (adx)^k ∂_t x
        = Σ_{k=0} 1/(k+1)! (-1)^k (adx)^k ∂_t x
    """
    m = tf.negative(adX)
    eye = eyeOf(m)
    x = eye + 1.0 / (order + 1.0) * m
    for i in tf.range(order, 1, -1):
        x = eye + 1.0 / tf.cast(i, m.dtype)*tf.linalg.matmul(m, x)
    return x


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
    # v = tf.zeros(8, dtype=tf.float64)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(zeros)
        y = f(tf.linalg.matmul(tf.linalg.expm(vec_to_su3(zeros)), x))
        # r = f(tf.linalg.matmul(exp(vec_to_su3(v)),x))
    d = tape.gradient(y, zeros)

    return y, d


def SU3GradientTFMat(f: Callable, x: Tensor) -> tuple[Tensor, Tensor]:
    """
    Compute gradient using TensorFlow GradientTape.
    f(x) must be a real scalar value.
    Returns (f(x),D), where D = T^a D^a = T^a ∂_a f(x)
    Use Matrix derivatives.
    D^a = ∂_a f(x)
        = [∂_a x_ij] [d/dx_ij f(x)]
        = [T^a_ik x_kj] [d/dx_ij f(x)]
        = [T^a_ik x†_kj] [d/dx†_ij f(x)]
    Note for TensorFlow,
        ∇_z f = (∂_z f + ∂_z f†)† = 2 [∂_z Re(f)]† = 2 ∂_z† Re(f)
    """
    with tf.GradientTape(watch_accessed_variables=False) as t:
        t.watch(x)
        r = f(x)
    d = tf.constant(0.5) * projectTAH(t.gradient(r, x) @ tf.linalg.adjoint(x))
    return r, d


def SU3JacobianTF(
        f: Callable,
        x: Tensor,
        is_SU3: bool = True
) -> tuple[Tensor, Tensor]:
    """
    Compute Jacobian using TensorFlow GradientTape with real vector
    derivatives.
    Note for TensorFlow,
        ∇_z f = (∂_z f + ∂_z f†)†

    In order to have the proper gradient info, we always project the result to
    su(3).
    If is_SU3 is True, we multiply the result by its adjoint before projecting.
    Otherwise we assume the result is su3 and project it directly.
    The input x must be in SU(3).
    Returns f(x) and its Jacobian in ad.
    [d/dSU(3)] SU(3)
        T^c_km X_ml (-2) (∂_X_kl F(X)_in) F(X)†_nj T^a_ji
          = T^c_km X_ml (-2) F'(X)_{kl,in} F(X)†_nj T^a_ji
    [d/dSU(3)] su(3)
        (T^c X)_kl (∂_X_kl F(X)^a)
        T^c_km X_ml (∂_X_kl F(X)^a)
          = T^c_km X_ml F'(X)^a_{kl}
    """
    v = tf.zeros(8, dtype=tf.float64)
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t:
        t.watch(v)
        Z = f(tf.linalg.expm(vec_to_su3(v)) @ x)
        # Z = f(tf.linalg.matmul(exp(vec_to_su3(v)),x))
        if is_SU3:
            z = Z @ tf.linalg.adjoint(tf.stop_gradient(Z))
            # z = tf.linalg.matmul(Z,tf.stop_gradient(Z),adjoint_b=True)
        else:
            z = Z
        z = su3_to_vec(z)
        # z = su3vec(z)
    tj = t.jacobian(z, v, experimental_use_pfor=False)
    return Z, tj


def SU3JacobianTFMat(f, x, is_SU3=True):
    """
    Compute Jacobian using TensorFlow GradientTape with matrix derivatives.
    Note for TensorFlow,
        ∇_z f = (∂_z f + ∂_z f†)†

    In order to have the proper gradient info,
    we always project the result to su(3).

    If is_SU3 is True, we multiply the result by its adjoint before projecting.
    Otherwise we assume the result is su3 and project it directly.
    The input x must be in SU(3).
    Returns f(x) and its Jacobian in ad.
    [d/dSU(3)] SU(3)
        T^c_km X_ml (-2) (∂_X_kl F(X)_in) F(X)†_nj T^a_ji
          = T^c_km X_ml (-2) F'(X)_{kl,in} F(X)†_nj T^a_ji
    [d/dSU(3)] su(3)
        (T^c X)_kl (∂_X_kl F(X)^a)
        T^c_km X_ml (∂_X_kl F(X)^a)
          = T^c_km X_ml F'(X)^a_{kl}
    """
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t:
        t.watch(x)
        Z = f(x)
        if is_SU3:
            z = tf.linalg.matmul(
                Z, tf.stop_gradient(Z), adjoint_b=True
            )
        else:
            z = Z
        z = tf.cast(su3_to_vec(z), TF_COMPLEX)
        # z = tf.cast(su3vec(z), tf.complex128)
    jzx = t.jacobian(z, x, experimental_use_pfor=False)
    tj = tf.math.real(
        tf.einsum(
            'aik,kj,bij->ba', su3gen(),
            x, tf.math.conj(jzx)
        )
    )
    return Z, tj
