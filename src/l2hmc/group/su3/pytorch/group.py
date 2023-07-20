"""
group/su3/pytorch/group.py

Contains PyTorch implementation of `SU3` object
"""
from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Sequence

import torch

from l2hmc import DEVICE
from l2hmc.group.group import Group
from l2hmc.group.su3.pytorch.utils import (
    checkU,
    checkSU,
    eigs3x3,
    su3_to_vec,
    vec_to_su3,
    norm2,
    randTAH3,
    projectSU,
    # projectTAH,
    eyeOf
)

import logging

log = logging.getLogger(__name__)

Tensor = torch.Tensor

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



class SU3(Group):
    def __init__(self) -> None:
        super().__init__(
            dim=4,
            shape=[3, 3],
            dtype=torch.complex128,
            name='SU3',
        )

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
        nc = torch.tensor(x.shape[-1])
        assert nc == 3  # SU3 for the time being
        r = 0.5 * (x - x.adjoint())
        d = torch.diagonal(r, dim1=-2, dim2=-1).sum(-1) / nc
        r = r - d.reshape(d.shape + (1, 1)) * eyeOf(x)
        # return projectTAH(x)
        return r

    def compat_proj(self, x: Tensor) -> Tensor:
        """Arbitrary matrix C projects to skew-hermitian B := (C - C^H) / 2

        Make traceless with tr(B - (tr(B) / N) * I) = tr(B) - tr(B) = 0
        """
        # with torch.no_grad():
        return self.projectSU(x)

    def random(self, shape: Sequence[int]) -> Tensor:
        """Returns (batched) random SU(3) matrices."""
        r = torch.randn(*shape, requires_grad=True, device=DEVICE)
        i = torch.randn(*shape, requires_grad=True, device=DEVICE)
        with torch.no_grad():
            x = projectSU(torch.complex(r, i)).to(DEVICE)
        return x

    def random_momentum(self, shape: Sequence[int]) -> Tensor:
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

    @staticmethod
    def rsqrtPHM3f(
            tr: Tensor,
            p2: Tensor,
            det: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
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

    def rsqrtPHM3(self, x: Tensor) -> Tensor:
        tr = torch.diagonal(x, dim1=-2, dim2=-1).sum(-1).real
        x2 = x @ x
        p2 = torch.diagonal(x2, dim1=-2, dim2=-1).sum(-1).real
        det = x.det().real
        c0_, c1_, c2_ = self.rsqrtPHM3f(tr, p2, det)
        c0 = c0_.reshape(c0_.shape + (1, 1)).to(x.dtype)
        c1 = c1_.reshape(c1_.shape + (1, 1)).to(x.dtype)
        c2 = c2_.reshape(c2_.shape + (1, 1)).to(x.dtype)
        return c0 * eyeOf(x) + c1 * x + c2 * x2

    def projectU(self, x: Tensor) -> Tensor:
        t = x.mH @ x
        t2 = self.rsqrtPHM3(t)
        return x @ t2

    def projectSU(self, x: Tensor) -> Tensor:
        return projectSU(x)

    def norm2(
            self,
            x: Tensor,
            axis: Sequence[int] = [-2, -1],
            exclude: Optional[Sequence[int]] = None,
    ) -> Tensor:
        """No reduction if axis is empty"""
        if x.dtype in [torch.complex64, torch.complex128]:
            x = x.abs()
        n = x.square()
        if exclude is None:
            if len(axis) == 0:
                return n
            return n.sum(*axis)
        return n.sum([
            i for i in range(len(n.shape)) if i not in exclude
        ])
