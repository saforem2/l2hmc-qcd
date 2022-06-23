"""
group/u1/pytorch/group.py

Contains pytorch implementation of `U1Phase`
"""
from __future__ import absolute_import, division, print_function, annotations
from typing import Optional

import torch

from l2hmc.group.group import Group

import logging

log = logging.getLogger(__name__)

PI = torch.pi
TWO_PI = torch.pi * 2.
Tensor = torch.Tensor
PT_FLOAT = torch.get_default_dtype()


def rand_unif(
        shape: list[int],
        a: float,
        b: float,
        requires_grad: bool = True
):
    rand = (a - b) * torch.rand(tuple(shape)) + b
    return rand.clone().detach().requires_grad_(requires_grad)


def random_angle(shape: list[int], requires_grad: bool = True) -> Tensor:
    """Returns random angle with `shape` and values in (-pi, pi)."""
    return rand_unif(shape, -PI, PI, requires_grad=requires_grad)


class U1Phase(Group):
    def __init__(self):
        dim = 2
        shape = [1]
        dtype = PT_FLOAT
        super().__init__(dim=dim, shape=shape, dtype=dtype)

    def exp(self, x: Tensor) -> Tensor:
        return torch.complex(x.cos(), x.sin())

    def update_gauge(
            self,
            x: Tensor,
            p: Tensor,
    ):
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

    @staticmethod
    def group_to_vec(x):
        return torch.stack([x.cos(), x.sin()], dim=-1)

    def compat_proj(self, x: Tensor) -> Tensor:
        # return (x + PI % TWO_PI) - PI
        # return self.floormod(x + PI, TWO_PI) - PI
        return ((x + PI) % TWO_PI) - PI

    def random(self, shape: list[int]) -> Tensor:
        # return self.compat_proj(random_angle(shape, requires_grad=True))
        return self.compat_proj(TWO_PI * torch.rand(shape))
        # return self.compat_proj(torch.rand(shape, *(-4, 4)))

    def random_momentum(self, shape: list[int]) -> Tensor:
        return torch.randn(shape).reshape(shape[0], -1)

    def kinetic_energy(self, p: Tensor) -> Tensor:
        return 0.5 * p.flatten(1).square().sum(-1)
        # return p.reshape(p.shape[0], -1).square().sum(1)
