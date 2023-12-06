"""
group/pytorch/utils.py

"""
from __future__ import absolute_import, annotations, division, print_function
from math import pi as PI
from re import S
from typing import Optional, Sequence
from typing import Callable
import logging

import numpy as np
import torch
# from l2hmc.group.pytorch.logm import charpoly3x3, su3_to_eigs, log3x3

from l2hmc import DEVICE

log = logging.getLogger(__name__)
# log = get_logger(__name__)

Array = np.array
Tensor = torch.Tensor


# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NP_SQRT1by2 = np.sqrt(1. / 2.)
NP_SQRT1by3 = np.sqrt(1. / 3.)
TORCH_DTYPE = torch.get_default_dtype()

SQRT1by2 = torch.tensor(NP_SQRT1by2, dtype=TORCH_DTYPE, device=DEVICE)
SQRT1by3 = torch.tensor(NP_SQRT1by3, dtype=TORCH_DTYPE, device=DEVICE)
SQRT3 = torch.tensor(3., dtype=TORCH_DTYPE, device=DEVICE).sqrt_()
ONE_HALF = 1. / 2.
ONE_THIRD = 1. / 3.
TWO_PI = torch.tensor(2. * PI, dtype=TORCH_DTYPE, device=DEVICE)
EPS = torch.tensor(1e-12, dtype=TORCH_DTYPE, device=DEVICE)


f012 = +1.0
f036 = +0.5
f045 = -0.5
f135 = +0.5
f146 = +0.5
f234 = +0.5
f256 = -0.5
f347 = +0.86602540378443864676    # SQRT(3/4)
f567 = +0.86602540378443864676


def cmax(x: Tensor, y: Tensor) -> Tensor:
    """Returns the largets-magnitude complex number"""
    return torch.where(torch.abs(x) > torch.abs(y), x, y)


def unit(
        shape: Sequence[int],
        dtype: Optional[torch.dtype] = torch.complex128,
):
    batch_shape = list([1] * (len(shape) - 2))
    eye = torch.zeros(list(batch_shape + [*shape[-2:]])).to(dtype)
    eye[-2:] = torch.eye(shape[-1])
    return eye


def charpoly3x3(A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """ det(λ * I - A) = λ³ + λ² * c[3] + λ * c[2] + c[1] """
    elem1 = -(
        A[:, 0, 0] * (A[:, 1, 1] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 1])
        - A[:, 1, 0] * (A[:, 0, 1] * A[:, 2, 2] - A[:, 0, 2] * A[:, 2, 1])
        + A[:, 2, 0] * (A[:, 0, 1] * A[:, 1, 2] - A[:, 0, 2] * A[:, 1, 1])
    )
    elem2 = (
        A[:, 0, 0] * A[:, 1, 1]
        + A[:, 0, 0] * A[:, 2, 2]
        + A[:, 1, 1] * A[:, 2, 2]
        - A[:, 1, 0] * A[:, 0, 1]
        - A[:, 2, 0] * A[:, 0, 2]
        - A[:, 2, 1] * A[:, 1, 2]
    )
    elem3 = -(A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2])

    return (elem1, elem2, elem3)


def cubic_zeros(p):
    a = 1
    b = p[2]
    c = p[1]
    d = p[0]
    D0 = b ** 2 - 3 * a * c
    D1 = 2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d
    L = torch.pow(1e-3 + D1 ** 2 - 4 * D0 ** 3, 0.5)
    V = cmax((D1 + L) / 2, (D1 - L) / 2)
    C = V ** (1 / 3)
    w = np.exp(2 * np.pi * 1j / 3)

    return [
        -(b + (w ** k * C) + D0 / (w ** k * C)) / (3 * a) for k in range(3)
    ]


def su3_to_eigs(x: Tensor) -> Tensor:
    p = charpoly3x3(x)
    zs = cubic_zeros(p)
    return torch.cat([x.unsqueeze(-1) for x in zs], dim=-1)


def log3x3(x: Tensor):
    eigs = su3_to_eigs(x)
    q, _ = torch.linalg.solve(
        torch.log(eigs).unsqueeze(-1), (
            1e-6 * torch.eye(3).unsqueeze(0)
            + eigs.unsqueeze(-1) ** (
                torch.tensor([0, 1, 2]).unsqueeze(0).unsqueeze(0)
            )
        )
    )
    q = q.unsqueeze(-1)
    eye = torch.eye(x.shape[-1]).reshape(1, x.shape[-1], x.shape[-1])
    eye = eye.repeat(x.shape[0], 1, 1)

    return q[:, 0] * eye + q[:, 1] * x + q[:, 2] * x @ x


def eyeOf1(m: Tensor):
    batch_shape = [1] * (len(m.shape) - 2)
    eye = torch.zeros(batch_shape + [*m.shape[-2:]], device=DEVICE)
    eye[-2:] = torch.eye(m.shape[-1], device=DEVICE)
    # return torch.stack([torch.eye(m.shape[-1]) for _ in batch_shape])
    # return torch.tensor([torch.eye([*m.shape[-2:]]) for _ in batch_shape])
    return eye


def eyeOf(x: torch.Tensor) -> torch.Tensor:
    # NOTE:
    #  batch_dims = [[1], [1], [1], ..., [1]]
    #  len(batch_dims) = len(x.shape) - 2
    batch_dims = [1] * (len(x.shape) - 2)
    eye = torch.zeros(batch_dims + [*x.shape[-2:]], device=DEVICE)
    eye[-2:] = torch.eye(x.shape[-1], device=DEVICE)
    return eye


def eye_like(x: Tensor):
    return torch.eye(*x.size(), out=torch.empty_like(x)).to(DEVICE)


def expm(m: Tensor, order: int = 12) -> Tensor:
    eye = eyeOf(m)
    x = eye + m / torch.tensor(order, device=DEVICE)
    for i in range(order - 1, 0, -1):
        x = eye + (torch.matmul(m, x) / torch.tensor(i).type_as(m))

    return x


def norm2(
        x: Tensor,
        axis: Sequence[int] = [-2, -1],
        exclude: Optional[Sequence[int]] = None,
) -> Tensor:
    """No reduction if axis is empty"""
    if x.is_complex():
        x = x.abs()
    n = x.square()
    if exclude is None:
        return n if len(axis) == 0 else n.sum(tuple(axis))
    return n.sum([i for i in range(len(n.shape)) if i not in exclude])


def randTAH3(shape: Sequence[int]):
    shape = tuple(shape)
    r3 = SQRT1by2 * torch.randn(shape, device=DEVICE)
    r8 = SQRT1by2 * SQRT1by3 * torch.randn(shape, device=DEVICE)
    m00 = torch.complex(torch.zeros_like(r3), r8 + r3)
    m11 = torch.complex(torch.zeros_like(r3), r8 - r3)
    m22 = torch.complex(torch.zeros_like(r3), -2 * r8)
    r01 = SQRT1by2 * torch.randn(shape, device=DEVICE)
    r02 = SQRT1by2 * torch.randn(shape, device=DEVICE)
    r12 = SQRT1by2 * torch.randn(shape, device=DEVICE)
    i01 = SQRT1by2 * torch.randn(shape, device=DEVICE)
    i02 = SQRT1by2 * torch.randn(shape, device=DEVICE)
    i12 = SQRT1by2 * torch.randn(shape, device=DEVICE)
    m01 = torch.complex(r01, i01).to(DEVICE)
    m10 = torch.complex(-r01, i01).to(DEVICE)
    m02 = torch.complex(r02, i02).to(DEVICE)
    m20 = torch.complex(-r02, i02).to(DEVICE)
    m12 = torch.complex(r12, i12).to(DEVICE)
    m21 = torch.complex(-r12, i12).to(DEVICE)

    return torch.stack([
        torch.stack([m00, m10, m20], dim=-1),
        torch.stack([m01, m11, m21], dim=-1),
        torch.stack([m02, m12, m22], dim=-1),
    ], dim=-1)


def acos_safe1(x, eps=1e-4):
    sign = torch.sign(x)
    slope = np.arccos(1-eps) / eps
    return torch.where(
        abs(x) <= 1-eps,
        torch.acos(x),
        (
            torch.acos(sign * (1 - eps))
            - slope * sign * (abs(x) - 1 + eps)
        )
    )


def acos_safe(x, eps=1e-4):
    slope = np.arccos(1-eps) / eps
    # TODO: stop doing this allocation once sparse gradients with NaNs (like in
    # th.where) are handled differently.
    buf = torch.empty_like(x)
    good = abs(x) <= 1-eps
    bad = ~good
    sign = torch.sign(x[bad])
    buf[good] = torch.acos(x[good])
    buf[bad] = (
        torch.acos(sign * (1 - eps))
        - slope*sign*(abs(x[bad]) - 1 + eps)
    )
    return buf


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
    maxv = 3e38 * torch.ones(isq3.shape).to(isq3.device)
    minv = -3e38 * torch.ones(isq3.shape).to(isq3.device)
    # isq3c = torch.from_numpy()

    isq3c = maxv.minimum(minv.maximum(isq3))
    rsq3c = r * isq3c
    maxv = 1.0 * torch.ones(isq3.shape).to(isq3.device)
    minv = -1.0 * torch.ones(isq3.shape).to(isq3.device)

    rsq3 = maxv.minimum(minv.maximum(rsq3c.real))
    rsq3 = rsq3.clamp_(-1.0 + EPS, 1.0 - EPS)
    # rsq3 = rsq3.clamp_(-0.99, 0.99)
    # csq3 = macvc.minimum(minvc.maximum(isq3c.imag))
    # rsq3_n2 = norm2(rsq3)
    # if any(rsq3_n2.max() > 1.0):
    # log.info(f"rsq3: {rsq3}")
    # rsq3_ = rsq3.clamp(1.0 - 1.0e-32)
    # t = (1.0 / 3.0) * torch.acos(rsq3.clamp_(-0.999999999999, 0.9999999999999))
    t = (1.0 / 3.0) * torch.acos(rsq3)
    # # if t.requires_grad:
    # t.register_hook(
    #     lambda z: log.info(
    #         f"[hook] torch.acos(rsq3):\n {z}"
    #     )
    # )
    # t.register_hook(
    #     lambda z: log.info(
    #         f"[hook] torch.acos(rsq3).grad:\n {z.grad}"
    #     )
    # )
    # t.register_hook(lambda grad: grad.clamp(max=1.0))  # grad.clamp(max=1.0))
    # t = (1.0 / 3.0) * torch.acos(rsq3.clamp_(-0.999999, 0.999999))

    # t = (1.0 / 3.0) * acos_safe(rsq3)
    st = torch.sin(t)
    ct = torch.cos(t)
    sqc = sq * ct
    sqs = SQRT3 * sq * st
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
    # di = torch.tensor(1.0) / d
    # di.register_hook(lambda grad: grad.clamp_(max=1.0))

    # d.register_hook(lambda z: log.info(f"[hook] d:\n {z}"))
    # d.register_hook(lambda grad: log.info(f"[hook] d.grad:\n {grad}"))
    # d.register_hook(lambda grad: grad + 1e-6)
    # d = torch.where(d == 0., 0.1, d)
    # di = 1.0 / torch.where(d == 0, 1e-6, d)
    di = 1.0 / d
    # di = di.clamp_(-3e38, 3e38)
    # return torch.where(torch.abs(x) > torch.abs(y), x, y)
    # di.register_hook(lambda z: log.info(f"[hook] 1/d:\n {z}"))
    # di.register_hook(lambda grad: grad.clamp(max=1.0))

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
    tr = torch.diagonal(x, dim1=-2, dim2=-1).sum(-1).real
    x2 = torch.matmul(x, x)
    p2 = torch.diagonal(x2, dim1=-2, dim2=-1).sum(-1).real
    det = x.det().real
    c0_, c1_, c2_ = rsqrtPHM3f(tr, p2, det)
    c0 = c0_.reshape(c0_.shape + (1, 1)).type_as(x)
    c1 = c1_.reshape(c1_.shape + (1, 1)).type_as(x)
    c2 = c2_.reshape(c2_.shape + (1, 1)).type_as(x)
    return c0 * eyeOf(x) + c1 * x + c2 * x2


def projectU(x: Tensor) -> Tensor:
    """x (x'x)^{1/2}"""
    # t = x.adjoint() @ x
    t = torch.matmul(x.adjoint(), x)
    t2 = rsqrtPHM3(t)
    # return torch.matmul(x, t2)
    return x @ t2


def projectSU(x: Tensor) -> Tensor:
    nc = x.shape[-1]
    m = projectU(x)
    d = m.det().to(torch.complex128)
    p = (1.0 / (-nc)) * torch.atan2(d.imag, d.real)
    return m * torch.complex(p.cos(), p.sin()).reshape(list(p.shape) + [1, 1])


def projectTAH(x: Tensor) -> Tensor:
    """Returns R = 1/2 (X - X†) - 1/(2 N) tr(X - X†)
    R = - T^a tr[T^a (X - X†)]
      = T^a ∂_a (- tr[X + X†])
    """
    nc = torch.tensor(x.shape[-1]).to(x.dtype)
    r = 0.5 * (x - x.adjoint())
    d = torch.diagonal(r, dim1=-2, dim2=-1).sum(-1) / nc
    r = r - d.reshape(d.shape + (1, 1)) * eyeOf(x)

    return r


def checkU(x: Tensor) -> tuple[Tensor, Tensor]:
    """Returns the average and maximum of the sum of the deviations of X†X"""
    nc = torch.tensor(x.shape[-1]).to(x.dtype)
    d = norm2(torch.matmul(x.adjoint(), x) - eyeOf(x))
    d_ = d.flatten(1)
    a = d_.mean(-1)
    b, _ = d_.max(-1)
    # a = d.mean(*range(1, len(d.shape)))
    # b, _ = d.
    # b = d.max(*range(1, len(d.shape)))
    c = 2 * (nc * nc + 1)

    return (a / c).sqrt(), (b / c).sqrt()


def checkSU(x: Tensor) -> tuple[Tensor, Tensor]:
    """Returns the average and maximum of the sum of deviations of:
         1. X† X
         2. det(x)
    from unitarity
    """
    nc = x.shape[-1]
    d = norm2(x.adjoint() @ x - eyeOf(x))
    d += norm2(-1 + x.det(), axis=[])
    d = d.unsqueeze(1)
    a = d.mean(tuple(range(1, len(d.shape))))
    b = d.amax(tuple(range(1, len(d.shape))))
    c = float(2 * (nc * nc + 1))

    return (a / c).sqrt(), (b / c).sqrt()


def su3_to_vec(x: Tensor) -> Tensor:
    """Only for x in 3x3 anti-Hermitian.

    Returns 8 real numbers, X^a T^a = X - 1/3 tr(X)

    Convention: tr{T^a T^a} = -1/2
    X^a = - 2 tr[T^a X]
    """
    c = -2
    x00 = x[..., 0, 0]
    x01 = x[..., 0, 1]
    x02 = x[..., 0, 2]
    x11 = x[..., 1, 1]
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
            (2 * x22.imag) - x11.imag - x00.imag
        ),
    ], dim=-1)


def vec_to_su3(v: Tensor) -> Tensor:
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
    x2i = (s3 * v[..., 7]).to(v[..., 0].dtype)
    x0i = c * (x2i + v[..., 2]).to(v[..., 0].dtype)
    x1i = c * (x2i - v[..., 2]).to(v[..., 0].dtype)
    v00 = torch.complex(zero, x0i)
    v11 = torch.complex(zero, x1i)
    v22 = torch.complex(zero, x2i)
    return torch.stack([
        torch.stack([v00, -x01.conj(), -x02.conj()], -1),
        torch.stack([x01, v11, -x12.conj()], -1),
        torch.stack([x02, x12, v22], -1),
    ], -1)


def su3fabc(v: Tensor):
    """
    returns f^{abc} v[..., c]
    [T^a, T^b] = f^abc T^c
    """
    a01 = +f012 * v[..., 2]
    a02 = -f012 * v[..., 1]
    a03 = +f036 * v[..., 6]
    a04 = +f045 * v[..., 5]
    a05 = -f045 * v[..., 4]
    a06 = -f036 * v[..., 3]
    a12 = +f012 * v[..., 0]
    a13 = +f135 * v[..., 5]
    a14 = +f146 * v[..., 6]
    a15 = -f135 * v[..., 3]
    a16 = -f146 * v[..., 4]
    a23 = +f234 * v[..., 4]
    a24 = -f234 * v[..., 3]
    a25 = +f256 * v[..., 6]
    a26 = -f256 * v[..., 5]
    a34 = +f347 * v[..., 7] + f234 * v[..., 2]
    a35 = +f135 * v[..., 1]
    a36 = +f036 * v[..., 0]
    a37 = -f347 * v[..., 4]
    a45 = +f045 * v[..., 0]
    a46 = +f146 * v[..., 1]
    a47 = +f347 * v[..., 3]
    a56 = +f567 * v[..., 7] + f256 * v[..., 2]
    a57 = -f567 * v[..., 6]
    a67 = +f567 * v[..., 5]
    zii = torch.zeros(v[..., 0].shape, dtype=v[..., 0].dtype)
    return torch.stack([
        torch.stack([+zii, -a01, -a02, -a03, -a04, -a05, -a06, +zii], -1),
        torch.stack([+a01, +zii, -a12, -a13, -a14, -a15, -a16, +zii], -1),
        torch.stack([+a02, +a12, +zii, -a23, -a24, -a25, -a26, +zii], -1),
        torch.stack([+a03, +a13, +a23, +zii, -a34, -a35, -a36, -a37], -1),
        torch.stack([+a04, +a14, +a24, +a34, +zii, -a45, -a46, -a47], -1),
        torch.stack([+a05, +a15, +a25, +a35, +a45, +zii, -a56, -a57], -1),
        torch.stack([+a06, +a16, +a26, +a36, +a46, +a56, +zii, -a67], -1),
        torch.stack([+zii, +zii, +zii, +a37, +a47, +a57, +a67, +zii], -1),
    ], dim=-1)


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
