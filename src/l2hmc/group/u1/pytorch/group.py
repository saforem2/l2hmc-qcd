"""
group/u1/pytorch/group.py

Contains pytorch implementation of `U1Phase`
"""
from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Sequence

import torch

from l2hmc.group.group import Group

import logging

log = logging.getLogger(__name__)

PI = torch.pi
TWO_PI = torch.pi * 2.
Tensor = torch.Tensor
PT_FLOAT = torch.get_default_dtype()


def rand_unif(
        shape: Sequence[int],
        a: float,
        b: float,
        requires_grad: bool = True
):
    """Return tensor x ~ U(a, b), where a <= x <= b with shape `shape`

    >>> import numpy as np
    >>> x = rand_unif([1, 2, 3], -1, 1)
    >>> x.shape
    torch.Size([1, 2, 3])
    >>> (-1. <= x.min()).item()
    True
    >>> (x.max() <= 1.).item()
    True
    """
    rand = (a - b) * torch.rand(tuple(shape)) + b
    return rand.clone().detach().requires_grad_(requires_grad)


def random_angle(shape: Sequence[int], requires_grad: bool = True) -> Tensor:
    """Returns random angle with `shape` and values in (-pi, pi).
    """
    return rand_unif(shape, -PI, PI, requires_grad=requires_grad)


def eyeOf(x: torch.Tensor) -> torch.Tensor:
    # NOTE:
    #  batch_dims = [[1], [1], [1], ..., [1]]
    #  len(batch_dims) = len(x.shape) - 1
    batch_dims = [1] * (len(x.shape) - 1)
    eye = torch.zeros(batch_dims + [*x.shape[-1:]]).to(x.device)
    eye[-1:] = torch.eye(x.shape[-1])  # , device=DEVICE)
    return eye


class U1Phase(Group):
    def __init__(self) -> None:
        dim = 2
        shape = [1]
        dtype = PT_FLOAT
        super().__init__(dim=dim, shape=shape, dtype=dtype)

    def phase_to_coords(self, phi: Tensor) -> Tensor:
        """Convert complex to Cartesian.

        exp(i φ) --> [cos φ, sin φ]

        >>> g = U1Phase()
        >>> g.phase_to_coords(torch.tensor([1.0, 0.0]))
        tensor([0.5403, 1.0000, 0.8415, 0.0000])
        """
        return torch.cat([phi.cos(), phi.sin()], -1)

    def coords_to_phase(self, x: Tensor) -> Tensor:
        """Convert Cartesian to phase.

        [cos φ, sin φ] --> atan(sin φ / cos φ)
        """
        assert x.shape[-1] == 2
        return torch.atan2(x[..., -1], x[..., -2])

    @staticmethod
    def group_to_vec(x: Tensor) -> Tensor:
        # exp(i φ) --> [cos φ, sin φ]
        return torch.cat([x.cos(), x.sin()], dim=1)

    @staticmethod
    def vec_to_group(x: Tensor) -> Tensor:
        # return torch.complex(x.cos(), x.sin())
        if x.is_complex():
            return torch.atan2(x.imag, x.real)

        return torch.atan2(x[..., -1], x[..., -2])

    def exp(self, x: Tensor) -> Tensor:
        return torch.complex(x.cos(), x.sin())

    def update_gauge(self, x: Tensor, p: Tensor) -> Tensor:
        return x + p

    def mul(
            self,
            a: Tensor,
            b: Tensor,
            adjoint_a: Optional[bool] = None,
            adjoint_b: Optional[bool] = None,
    ) -> Tensor:
        if adjoint_a and adjoint_b:
            return -a - b
        elif adjoint_a:
            return -a + b
        elif adjoint_b:
            return a - b
        else:
            return a + b

    def adjoint(self, x: Tensor) -> Tensor:
        return -x

    def trace(self, x: Tensor) -> Tensor:
        return torch.cos(x)

    def diff_trace(self, x: Tensor) -> Tensor:
        return (-torch.sin(x))

    def diff2trace(self, x: Tensor) -> Tensor:
        return (-torch.cos(x))

    @torch.no_grad()
    def floormod(self, x: Tensor | float, y: Tensor | float) -> Tensor:
        return (x - torch.floor_divide(x, y) * y)

    def compat_proj(self, x: Tensor) -> Tensor:
        return ((x + PI) % TWO_PI) - PI

    def projectTAH(
        self,
        x: Tensor
    ) -> Tensor:
        """Returns
        r = (1/2) * (x - x.H) - j Im[ Tr(x) ] / Nc
        """
        # nc = torch.tensor(x.shape[-1]).to(x.dtype)
        # r = 0.5 * (x - x.adjoint())
        # d = tor
        # d = torch.diagonal(r, dim1=-2, dim2=-1).sum(-1) / nc
        # r = r - d.reshape(d.shape + (1, 1)) * eyeOf(x)
        # TODO: Fix for U1
        return x

    def projectSU(self, x):
        return self.compat_proj(x)

    def random(self, shape: Sequence[int]) -> Tensor:
        return self.compat_proj(TWO_PI * torch.rand(*shape))

    def random_momentum(self, shape: Sequence[int]) -> Tensor:
        return torch.randn(*shape).reshape(shape[0], -1)

    def kinetic_energy(self, p: Tensor) -> Tensor:
        return 0.5 * p.flatten(1).square().sum(-1)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
