"""
group.py

Contains implementations of various (special) unitary groups.
"""
from __future__ import absolute_import, division, print_function, annotations

import numpy as np
import torch

from math import pi as PI


from typing import Callable


Array = np.array
Tensor = torch.Tensor

ONE_HALF = 1. / 2.
ONE_THIRD = 1. / 3.
TWO_PI = torch.tensor(2. * PI)

SQRT1by2 = torch.tensor(np.sqrt(1. / 2.))
SQRT1by3 = torch.tensor(np.sqrt(1. / 3.))


def eyeOf(m):
    batch_shape = [1] * (len(m.shape) - 2)
    eye = torch.zeros(batch_shape + [*m.shape[-2:]])
    eye[-2:] = torch.eye(m.shape[-1])
    # return torch.stack([torch.eye(m.shape[-1]) for _ in batch_shape])
    # return torch.tensor([torch.eye([*m.shape[-2:]]) for _ in batch_shape])
    return eye


def expm(m: Tensor, order: int = 12) -> Tensor:
    eye = eyeOf(m)
    x = eye + m / torch.tensor(order)
    for i in range(order - 1, 0, -1):
        x = eye + (torch.matmul(m, x) / torch.tensor(i).type_as(m))

    return x


def norm2(x: Tensor, axis: list[int] = [-2, -1]) -> Tensor:
    """No reduction if axis is empty"""
    n = torch.real(torch.multiply(x.conj(), x))
    if len(axis) == 0:
        return n

    return n.sum(axis)


def randTAH3(shape: list[int]):
    r3 = SQRT1by2 * torch.randn(shape)
    r8 = SQRT1by2 * SQRT1by3 * torch.randn(shape)
    m00 = torch.complex(torch.zeros_like(r3), r8 + r3)
    m11 = torch.complex(torch.zeros_like(r3), r8 - r3)
    m22 = torch.complex(torch.zeros_like(r3), -2 * r8)
    r01 = SQRT1by2 * torch.randn(shape)
    r02 = SQRT1by2 * torch.randn(shape)
    r12 = SQRT1by2 * torch.randn(shape)
    i01 = SQRT1by2 * torch.randn(shape)
    i02 = SQRT1by2 * torch.randn(shape)
    i12 = SQRT1by2 * torch.randn(shape)
    m01 = torch.complex(r01, i01)
    m10 = torch.complex(-r01, i01)
    m02 = torch.complex(r02, i02)
    m20 = torch.complex(-r02, i02)
    m12 = torch.complex(r12, i12)
    m21 = torch.complex(-r12, i12)

    return torch.stack([
        torch.stack([m00, m10, m20], dim=-1),
        torch.stack([m01, m11, m21], dim=-1),
        torch.stack([m02, m12, m22], dim=-1),
    ], dim=-1)


def eigs3x3(
        tr: Tensor,
        p2: Tensor,
        det: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    tr3 = ONE_THIRD * tr
    p23 = ONE_THIRD * p2
    tr32 = tr3 * tr3
    q = (0.5 * (p23 - tr32)).abs()
    r = 0.25 * tr3 * (5 * tr32 - p2) - 0.5 * det
    sq = torch.sqrt(q)
    sq3 = q * sq
    isq3 = 1.0 / sq3
    maxv = 3e38 * torch.ones_like(isq3)
    minv = -3e38 * torch.ones_like(isq3)
    # isq3c = torch.from_numpy()
    isq3c = torch.tensor(
        np.minimum(maxv.numpy(), np.maximum(minv.numpy(), isq3.numpy()))
    )
    rsq3c = r * isq3c
    maxv = 1. * torch.ones_like(isq3)
    minv = -1. * torch.ones_like(isq3)
    rsq3 = torch.tensor(
        np.minimum(maxv.numpy(), np.maximum(minv.numpy(), rsq3c.numpy()))
    )
    t = (1.0 / 3.0) * torch.acos(rsq3)
    st = torch.sin(t)
    ct = torch.cos(t)
    sqc = sq * ct
    sqs = torch.tensor(np.sqrt(3)) * sq * st
    ll = tr3 + sqc
    e0 = tr3 - 2 * sqc
    e1 = ll + sqs
    e2 = ll - sqs

    return e0, e1, e2


def rsqrtPHM3f(tr, p2, det):
    e0, e1, e2 = eigs3x3(tr, p2, det)
    se0 = e0.abs().sqrt()
    se1 = e1.abs().sqrt()
    se2 = e2.abs().sqrt()
    u = se0 + se1 + se2
    w = se0 * se1 * se2
    d = w * (se0 + se1) * (se0 + se2) * (se1 + se2)
    di = 1.0 / d
    c0 = di * (
        w * u * u
        + e0 * se0 * (e1 + e2)
        + e1 * se1 * (e0 + e2)
        + e2 * se2 * (e0 + e1)
    )
    c1 = -(tr * u + w) * di
    c2 = u * di

    return c0, c1, c2


def rsqrtPHM3(x: Tensor) -> Tensor:
    # tr = x.trace().real
    tr = torch.diagonal(x, dim1=-2, dim2=-1).sum(-1)
    x2 = torch.matmul(x, x)
    p2 = torch.diagonal(x2, dim1=-2, dim2=-1).sum(-1)
    # p2 = x2.trace().real
    det = x.det().real
    c0, c1, c2 = rsqrtPHM3f(tr, p2, det)
    c0_ = c0.reshape(c0.shape + (1, 1))
    c1_ = c1.reshape(c1.shape + (1, 1))
    c2_ = c2.reshape(c2.shape + (1, 1))
    # eyes = torch.stack([torch.eye(3) for _ in range((1) * len(c0.shape))])
    # eye = torch.stack([tf.eye(3)
    # term0 = (c0_ * eyes).type_as(x)
    term0 = c0_ * eyeOf(x)
    # term0 = c0_ * torch.eye(3).type_as(x)
    term1 = x * c1_.type_as(x)
    term2 = x * c2_.type_as(x)
    # term1 = x * c1_.type_as(x)
    # term2 = x * c2_.type_as(x)

    return term0 + term1 + term2


def projectU(x: Tensor) -> Tensor:
    """x (x'x)^{1/2}"""
    t = torch.matmul(x.adjoint(), x)
    t2 = rsqrtPHM3(t)
    return torch.matmul(x, t2)


def projectSU(x: Tensor) -> Tensor:
    nc = torch.tensor(x.shape[-1]).to(x.dtype)
    m = projectU(x)
    d = m.det()
    tmp = torch.atan2(d.imag, d.real)
    p = tmp * (-1.0 / nc)
    # pcos = torch.cos(p)
    # psin = torch.sin(p)
    p_ = torch.complex(p.real, p.imag).reshape(p.shape + (1, 1))

    return p_ * m


def projectTAH(x: Tensor) -> Tensor:
    """Returns R = 1/2 (X - X†) - 1/(2 N) tr(X - X†)
    R = - T^a tr[T^a (X - X†)]
      = T^a ∂_a (- tr[X + X†])
    """
    nc = torch.tensor(x.shape[-1]).to(x.dtype)
    r = 0.5 * (x - x.adjoint())
    d = r.trace() / nc
    r = r - d.reshape(d.shape + (1, 1)) * eyeOf(x)

    return r


def checkU(x: Tensor) -> tuple[Tensor, Tensor]:
    """Returns the average and maximum of the sum of the deviations of X†X"""
    nc = torch.tensor(x.shape[-1]).to(x.dtype)
    d = norm2(torch.matmul(x.adjoint(), x) - eyeOf(x))
    a = d.mean(*range(1, len(d.shape)))
    b = d.max(*range(1, len(d.shape)))
    c = 2 * (nc * nc + 1).to(x.dtype)

    return torch.sqrt(a / c), torch.sqrt(b / c)


def checkSU(x: Tensor) -> tuple[Tensor, Tensor]:
    """Returns the average and maximum of the sum of deviations of:
         1. X† X
         2. det(x)
    from unitarity
    """
    nc = torch.tensor(x.shape[-1]).to(x.dtype)
    d = norm2(torch.matmul(x.adjoint(), x) - eyeOf(x))
    det = x.det()
    d = d + norm2(torch.ones_like(det) + det, axis=[])
    range_ = tuple(range(1, len(d.shape)))
    a = d.mean(range_)
    b = d.amax(range_)
    # b = d.max(tuple(range(1, len(d.shape))))
    c = (2 * (nc * nc + 1)).real  # .to(x.real.dtype)

    return torch.sqrt(a / c), torch.sqrt(b / c)


def su3vec(x: Tensor) -> Tensor:
    """Only for x in 3x3 anti-Hermitian.

    Returns 8 real numbers, X^a T^a = X - 1/3 tr(X)

    Convention: tr{T^a T^a} = -1/2
    X^a = - 2 tr[T^a X]
    """
    c = -2
    x00 = x[..., 0, 0]
    x01 = x[..., 0, 1]
    x11 = x[..., 1, 1]
    x02 = x[..., 0, 2]
    x12 = x[..., 1, 2]
    x22 = x[..., 2, 2]
    return torch.stack([
        c * x01.imag,
        c * x01.real,
        x11.imag - x00.imag,
        c * x02.imag,
        c * x02.real,
        c * x12.imag,
        c * x12.real,
        SQRT1by3 * (
            2 * x22.imag - x11.imag - x00.imag
        ),
    ], dim=-1)


def su3fromvec(v: Tensor) -> Tensor:
    """
    X = X^a T^a
    tr{X T^b} = X^a tr{T^a T^b} = X^a (-1/2) 𝛅^ab = -1/2 X^b
    X^a = -2 X_{ij} T^a_{ji}
    """
    s3 = SQRT1by3
    c = -0.5
    zero = torch.zeros_like(v[..., 0])
    x01 = c * torch.complex(v[..., 1], v[..., 0])
    x02 = c * torch.complex(v[..., 4], v[..., 3])
    x12 = c * torch.complex(v[..., 6], v[..., 5])
    x2i = s3 * v[..., 7]
    x0i = c * (x2i + v[..., 2])
    x1i = c * (x2i - v[..., 2])
    return torch.stack([
        torch.stack([torch.complex(zero, x0i), -x01.conj(), -x02.conj()], -1),
        torch.stack([x01, torch.complex(zero, x1i), -x12.conj()], -1),
        torch.stack([x02, x12, torch.complex(zero, x2i)], -1),
    ])


def SU3Gradient(
        f: Callable[[Tensor], Tensor],
        x: Tensor,
        create_graph: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute gradient using autograd.

    y = f(x) must be a real scalar value.

    Returns:
     - (f(x), D), where D = T^a D^a = T^a ∂_a f(x)

    NOTE: Use real vector derivatives, e.g.
      D^a = ∂_a f(x)
          = ∂_t f(exp(T^a) x) |_{t=0}
    """
    x.requires_grad_(True)
    y = f(x)
    identity = torch.ones(x.shape[0], device=x.device)
    dydx, = torch.autograd.grad(y, x,
                                create_graph=create_graph,
                                retain_graph=True,
                                grad_outputs=identity)
    return y, dydx


class Group:
    """Gauge group represented as matrices in the last two dimensions."""
    def mul(
            self,
            a: Tensor,
            b: Tensor,
            adjoint_a: bool = False,
            adjoint_b: bool = False,
    ) -> Tensor:
        if adjoint_a and adjoint_b:
            return a.adjoint() @ b.adjoint()
        if adjoint_a:
            return a.adjoint() @ b
        if adjoint_b:
            return a @ b.adjoint()
        return a @ b


def rand_unif(
        shape: list[int],
        a: float,
        b: float,
        requires_grad: bool = True
):
    rand = (a - b) * torch.rand(tuple(shape)) + b
    return rand.clone().detach().requires_grad_(requires_grad)


def random_angle(shape: list[int], requires_grad: bool = True) -> Tensor:
    """Returns random angle with `shape` and values in [-pi, pi)."""
    return rand_unif(shape, -PI, PI, requires_grad=requires_grad)


class U1Phase(Group):
    # dtype = torch.complex128
    size = [1]
    shape = (1)

    def mul(
        self,
        a: Tensor,
        b: Tensor,
        adjoint_a: bool = False,
        adjoint_b: bool = False,
    ) -> Tensor:
        if adjoint_a and adjoint_b:
            return -a - b
        if adjoint_a:
            return -a + b
        if adjoint_b:
            return a - b
        return a + b

    def adjoint(self, x: Tensor) -> Tensor:
        return -x

    def trace(self, x: Tensor) -> Tensor:
        return torch.cos(x)

    def diff_trace(self, x: Tensor) -> Tensor:
        return (-torch.sin(x))

    def diff2trace(self, x: Tensor) -> Tensor:
        return (-torch.cos(x))

    def compat_proj(self, x: Tensor) -> Tensor:
        return (x + PI % TWO_PI) - PI

    def random(self, shape: list[int]) -> Tensor:
        return self.compat_proj(random_angle(shape))
        # return self.compat_proj(torch.rand(shape, *(-4, 4)))

    def random_momentum(self, shape: list[int]) -> Tensor:
        return torch.randn(shape).reshape(shape[0], -1)

    def kinetic_energy(self, p: Tensor) -> Tensor:
        return 0.5 * p.reshape(p.shape[0], -1).square().sum(-1)
        # return p.reshape(p.shape[0], -1).square().sum(1)


class SU3(Group):
    dtype = torch.complex128
    size = [3, 3]
    shape = (3, 3)

    def mul(
        self,
        a: Tensor,
        b: Tensor,
        adjoint_a: bool = False,
        adjoint_b: bool = False,
    ) -> Tensor:
        if adjoint_a and adjoint_b:
            return a.adjoint() @ b.adjoint()
        if adjoint_a:
            return a.adjoint() @ b
        if adjoint_b:
            return a @ b.adjoint()
        return a @ b

    def adjoint(self, x: Tensor) -> Tensor:
        return x.adjoint()

    def trace(self, x: Tensor) -> Tensor:
        return torch.diagonal(x, dim1=-2, dim2=-1).sum(-1)
        # return torch.trace(x)

    def exp(self, x: Tensor) -> Tensor:
        return expm(x)

    def projectTAH(self, x: Tensor) -> Tensor:
        return projectTAH(x)

    def compat_proj(self, x: Tensor) -> Tensor:
        """Arbitrary matrix C projects to skew-hermitian B := (C - C^H) / 2

        Make traceless with tr(B - (tr(B) / N) * I) = tr(B) - tr(B) = 0
        """
        _, n, _ = x.shape
        algebra_elem = torch.solve(u, x)[0]  # X^{-1} u
        # do projection in lie algebra
        B = (algebra_elem - algebra_elem.conj().transpose(-2, -1)) / 2.
        trace = torch.einsum('bii->b', B)
        B = B - (
            (1 / n) * trace.unsqueeze(-1).unsqueeze(-1)
            * torch.eye(n).repeat(x.shape[0], 1, 1)
        )
        assert torch.abs(torch.mean(torch.einsum('bii->b', B))) < 1e-6

        return B

    def random(self, shape: list[int]) -> Tensor:
        r = torch.randn(shape)
        i = torch.randn(shape)
        return projectSU(torch.complex(r, i))

    def random_momentum(self, shape: list[int]) -> Tensor:
        return randTAH3(shape[:-2])

    def kinetic_energy(self, p: Tensor) -> Tensor:
        return 0.5 * (p.conj() @ p).real.sum(1)
