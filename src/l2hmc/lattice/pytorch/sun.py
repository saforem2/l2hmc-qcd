"""
sun.py

Contains implementation of SUN class.

Modified from original implementation at:
  https://github.com/CUAI/Equivariant-Manifold-Flows/blob/main/flows/sun.py
"""
from __future__ import absolute_import, print_function, division, annotations

import numpy as np
import torch

from torch import matrix_exp as expm
from l2hmc.lattice.pytorch.logm import log3x3_cdesa

Tensor = torch.Tensor


class TimePotentialSU3(nn.MOdule):
    def __init__(self) -> None:
        super(TimePotentialSU3, self).__init__()
        self.full_eigdecomp = su3_to_eigs_cdesa
        self.deepset = ComplexDeepTimeSet(1, 1, hidden_channels=64)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        x = self.full_eigdecomp(x)
        x = x.unsqueeze(-1)
        x = self.deepset(t, x)

        return x


class SU3TimeEquivariantVectorField(nn.Module):
    def __init__(self, func):
        super(SU3TimeEquivariantVectorField, self).__init__()
        self.func = func

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.autograd.grad(  # vector field
            self.func(t, x).squeeze().sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0]


class AmbientProjNN(nn.Module):
    def __init__(self, func):
        super(AmbientProjNN, self).__init__()
        self.func = func
        self.man = SUN()

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        self.man.proju(x, self.func(t, x))



class SUN:
    def __init__(self) -> None:
        super(SUN, self).__init__()

    def exp(self, x: Tensor, u: Tensor) -> Tensor:
        return x @ expm(x.conj().transpose(-2, -1) @ u)

    def log(self, x: Tensor, y: Tensor) -> Tensor:
        _, n, _ = x.shape
        assert n == 3, "Operation supported only for SU(3)"
        return x @ log3x3_cdesa(x.conj().transpose(-2, -1) @ y)

    def proju(self, x: Tensor, u: Tensor, inplace: bool = False) -> Tensor:
        """Arbitrary matrix C projects to skew-Hermitian 
        
                    B := (C - C^†) / 2

        then make traceless with

                    tr{ B - tr{B / N} * I) = tr{B} - tr{B} = 0
        """
        _, n, _ = x.shape
        algebra_elem = torch.solve(u, x)[0]  # X^{-1} u

        # do projection in lie algebra
        B = (algebra_elem - algebra_elem.conj().transpose(-2, -1)) / 2
        trace = torch.einsum('bii->b', B)
        B = B - (
            (1 / n) * trace.unsqueeze(-1).unsqueeze(-1)
            * torch.eye(n).repeat(x.shape[0], 1, 1)
        )

        assert torch.abs(torch.mean(torch.einsum('bii->b', B))) < 1e-6

        return B




