"""
group/su3/pytorch/group.py

Contains PyTorch implementation of `SU3` object
"""
from __future__ import absolute_import, division, print_function, annotations

import torch

from l2hmc.group.group import Group
from l2hmc.group.su3.pytorch.utils import (
    checkU,
    checkSU,
    su3_to_vec,
    vec_to_su3,
    norm2,
    randTAH3,
    projectSU,
    projectTAH,
)

import logging

log = logging.getLogger(__name__)

Tensor = torch.Tensor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SU3(Group):
    def __init__(self) -> None:
        super().__init__(dim=4, shape=[3, 3], dtype=torch.complex128)

    def update_gauge(
            self,
            x: Tensor,
            p: Tensor,
    ) -> Tensor:
        return torch.matrix_exp(p) @ x

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
        # return torch.trace(x)
        return torch.diagonal(x, dim1=-2, dim2=-1).sum(-1)

    def diff_trace(self, x: Tensor) -> Tensor:
        log.error('TODO')
        return x

    def diff2trace(self, x: Tensor) -> Tensor:
        log.error('TODO')
        return x

    def exp(self, x: Tensor) -> Tensor:
        # return expm(x)
        return torch.linalg.matrix_exp(x)

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
        """Returns (batched) random SU(3) matrices."""
        r = torch.randn(shape, requires_grad=True, device=DEVICE)
        i = torch.randn(shape, requires_grad=True, device=DEVICE)
        return projectSU(torch.complex(r, i)).to(DEVICE)

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

    def compat_proju(self, u: Tensor, x: Tensor) -> Tensor:
        """Arbitrary matrix C projects to skew-hermitian B := (C - C^H) / 2

        Make traceless with tr(B - (tr(B) / N) * I) = tr(B) - tr(B) = 0
        """
        _, n, _ = x.shape
        algebra_elem = torch.linalg.solve(u, x)[0]  # X^{-1} u
        # do projection in lie algebra
        B = (algebra_elem - algebra_elem.conj().transpose(-2, -1)) / 2.
        trace = torch.einsum('bii->b', B)
        B = B - (
            (1 / n) * trace.unsqueeze(-1).unsqueeze(-1)
            * torch.eye(n).repeat(x.shape[0], 1, 1)
        )
        assert torch.abs(torch.mean(torch.einsum('bii->b', B))) < 1e-6

        return B
