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
        return tf.reduce_sum(
            tf.square(tf.reshape(p, [p.shape[0], -1])),
            axis=1
        )


class SU3(Group):
    dtype = tf.complex128
    size = [3, 3]
    shape = (3, 3)

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
        return exp(x)

    def projectTAH(self, x: Tensor) -> Tensor:
        return projectTAH(x)

    def random(self, shape: tuple) -> Tensor:
        r = tf.random.normal(shape, dtype=TF_FLOAT)
        i = tf.random.normal(shape, dtype=TF_FLOAT)
        return projectSU(tf.dtypes.complex(r, i))

    def random_momentum(self, shape: tuple) -> Tensor:
        return randTAH3(shape[:-2])

    def kinetic_energy(self, p: Tensor) -> Tensor:
        p2 = norm2(p) - tf.constant(8.0)  # - 8.0 ??
        return (
            0.5 * tf.math.reduce_sum(tf.reshape(p2, [p.shape[0], -1]), axis=1)
        )


def norm2(x: Tensor, axis=[-2, -1]) -> Tensor:
    """No reduction if axis is empty"""
    n = tf.math.real(tf.math.multiply(tf.math.conj(x), x))
    if len(axis) == 0:
        return n

    return tf.math.reduce_sum(n, axis=axis)


# Converted from qex/src/maths/matrixFunctions.nim
# Last two dims in a tensor contain matrices.
# WARNING: below only works for SU3 for now
def randTAH3(shape, s):
    s2 = 0.70710678118654752440    # sqrt(1/2)
    s3 = 0.577350269189625750    # sqrt(1/3)
    r3 = s2 * s.normal(shape, dtype=TF_FLOAT)
    r8 = s2 * s3 * s.normal(shape, dtype=TF_FLOAT)
    # m00 = tf.dtypes.complex(tf.cast(0.0,TF_FLOAT), r8+r3)
    # m11 = tf.dtypes.complex(tf.cast(0.0,TF_FLOAT), r8-r3)
    # m22 = tf.dtypes.complex(tf.cast(0.0,TF_FLOAT), -2*r8)
    m00 = tf.dtypes.complex(0.0, r8+r3)
    m11 = tf.dtypes.complex(0.0, r8-r3)
    m22 = tf.dtypes.complex(0.0, -2*r8)
    r01 = s2 * s.normal(shape, dtype=TF_FLOAT)
    r02 = s2 * s.normal(shape, dtype=TF_FLOAT)
    r12 = s2 * s.normal(shape, dtype=TF_FLOAT)
    i01 = s2 * s.normal(shape, dtype=TF_FLOAT)
    i02 = s2 * s.normal(shape, dtype=TF_FLOAT)
    i12 = s2 * s.normal(shape, dtype=TF_FLOAT)
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


def rsqrtPHM3f(tr, p2, det):
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
    c0, c1, c2 = rsqrtPHM3f(tr, p2, det)
    c0_ = tf.reshape(c0, c0.shape + [1, 1])
    c1_ = tf.reshape(c1, c1.shape + [1, 1])
    c2_ = tf.reshape(c2, c2.shape + [1, 1])
    term0 = tf.cast(c0_ * tf.eye(3, batch_shape=[1] * len(c0.shape)), x.dtype)
    term1 = tf.multiply(x, tf.cast(c1_, x.dtype))
    term2 = tf.multiply(x, tf.cast(c2_, x.dtype))
    return term0 + term1 + term2


def projectU(x: Tensor) -> Tensor:
    """x (x'x)^{-1/2}"""
    # nc = x.shape[-1]
    t = tf.linalg.matmul(x, x, adjoint_a=True)
    t2 = rsqrtPHM3(t)
    return tf.linalg.matmul(x, t2)


def projectSU(x: Tensor) -> Tensor:
    nc = tf.constant(x.shape[-1], TF_FLOAT)
    m = projectU(x)
    d = tf.linalg.det(m)
    tmp = tf.math.atan2(tf.math.imag(d), tf.math.real(d))
    p = tmp * tf.constant(-1.0) / nc
    # p = -(1.0 / nc) * tf.math.atan2(tf.math.imag(d), tf.math.real(d))
    # p = tf.math.multiply(tf.math.negative(tf.constant(1.0) / nc),
    #                      tf.math.atan2(tf.math.imag(d), tf.math.real(d)))
    p_ = tf.reshape(tf.dtypes.complex(tf.math.cos(p), tf.math.sin(p)),
                    p.shape + [1, 1])
    return p_ * m


def projectTAH(x: Tensor) -> Tensor:
    """Returns R = 1/2 (X - X†) - 1/(2 N) tr(X - X†)
    R = - T^a tr[T^a (X - X†)]
      = T^a ∂_a (- tr[X + X†])
    """
    nc = tf.constant(x.shape[-1])
    r = 0.5 * (x - tf.linalg.adjoint(x))
    d = tf.linalg.trace(r) / nc
    r -= tf.reshape(d, d.shape + [1, 1]) * eyeOf(x)

    return r

def checkU(x: Tensor) -> tuple[Tensor, Tensor]:
    """Returns the average and maximum of the sum of the deviations of X†X"""
    nc = tf.constant(x.shape[-1])
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
    nc = tf.constant(x.shape[-1])
    d = norm2(tf.linalg.matmul(x, x, adjoint_a=True) - eyeOf(x))
    det = tf.linalg.det(x)
    d = tf.math.add(d, norm2(tf.constant(1., dtype=det.dtype) + det, axis=[]))
    # d = tf.math.add(d, norm2(tf.constant(1.) + tf.linalg.det(x), axis=[]))
    # d += norm2(-1 + tf.linalg.det(x), axis=[])
    a = tf.math.reduce_mean(d, axis=range(1, len(d.shape)))
    b = tf.math.reduce_max(d, axis=range(1, len(d.shape)))
    c = tf.cast(2 * (nc * nc + 1), TF_FLOAT)
    return tf.math.sqrt(a / c), tf.math.sqrt(b / c)


def su3vec(x: Tensor) -> Tensor:
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


def su3fromvec(v: Tensor) -> Tensor:
    """
    X = X^a T^a
    tr{X T^b} = X^a tr{T^a T^b} = X^a (-1/2) 𝛅^ab = -1/2 X^b
    X^a = -2 X_{ij} T^a_{ji}
    """
    s3 = 0.577350269189625751  # sqrt(1/3)
    c = -0.5
    zero = tf.zeros(v[..., 0].shape, dtype=v[..., 0].dtype)  # type:ignore
    x01 = c * tf.dtypes.complex(v[..., 1], v[..., 0])        # type:ignore
    x02 = c * tf.dtypes.complex(v[..., 4], v[..., 3])        # type:ignore
    x12 = c * tf.dtypes.complex(v[..., 6], v[..., 5])        # type:ignore
    x2i = s3 * v[..., 7]                                     # type:ignore
    x0i = c * (x2i + v[..., 2])                              # type:ignore   
    x1i = c * (x2i - v[..., 2])                              # type:ignore  

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
        y = f(tf.linalg.matmul(exp(su3fromvec(zeros)), x))
    d = tape.gradient(y, zeros)

    return y, d
