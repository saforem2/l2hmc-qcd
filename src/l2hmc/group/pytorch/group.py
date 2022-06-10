"""
group.py

Contains implementations of various (special) unitary groups.
"""
from __future__ import absolute_import, division, print_function, annotations

import numpy as np
import torch

from math import pi as PI

from l2hmc.group.pytorch.utils import (
    projectTAH,
    projectSU,
    randTAH3,
    norm2,
    vec_to_su3,
    su3_to_vec,
    checkU,
    checkSU,
)

from typing import Callable

# from l2hmc.group.pytorch.logm import charpoly3x3, su3_to_eigs, log3x3


Array = np.array
Tensor = torch.Tensor

ONE_HALF = 1. / 2.
ONE_THIRD = 1. / 3.
TWO_PI = torch.tensor(2. * PI)

SQRT1by2 = torch.tensor(np.sqrt(1. / 2.))
SQRT1by3 = torch.tensor(np.sqrt(1. / 3.))


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

    def update_gauge(
        self,
        x: Tensor,
        p: Tensor,
    ) -> Tensor:
        return x + p

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
        return 0.5 * p.flatten(1).square().sum(-1)
        # return p.reshape(p.shape[0], -1).square().sum(1)


class SU3(Group):
    dtype = torch.complex128
    size = [3, 3]
    shape = (3, 3)

    def update_gauge(
            self,
            x: Tensor,
            p: Tensor,
    ) -> Tensor:
        return self.mul(self.exp(p), x)

    def checkSU(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return checkSU(x)

    def checkU(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return checkU(x)

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
        # return expm(x)
        return torch.linalg.matrix_exp(x)

    def projectTAH(self, x: Tensor) -> Tensor:
        return projectTAH(x)

    def compat_proju(self, u: Tensor, x: Tensor) -> Tensor:
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

    def compat_proj(self, x: Tensor) -> Tensor:
        """Arbitrary matrix C projects to skew-hermitian B := (C - C^H) / 2

        Make traceless with tr(B - (tr(B) / N) * I) = tr(B) - tr(B) = 0
        """
        return projectSU(x)

    def random(self, shape: list[int]) -> Tensor:
        """Returns (batched) random SU(3) matrices."""
        r = torch.randn(shape)
        i = torch.randn(shape)
        return projectSU(torch.complex(r, i))

    def random_momentum(self, shape: list[int]) -> Tensor:
        """Returns (batched) Traceless Anti-Hermitian matrices"""
        return randTAH3(shape[:-2])

    def kinetic_energy(self, p: Tensor) -> Tensor:
        return 0.5 * (norm2(p) - 8.0).flatten(1).sum(1)

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
