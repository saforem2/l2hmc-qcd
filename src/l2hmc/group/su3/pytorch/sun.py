"""
sun.py

Contains implementation of SUN class.

Modified from original implementation at:
  https://github.com/CUAI/Equivariant-Manifold-Flows/blob/main/flows/sun.py
"""
from __future__ import absolute_import, print_function, division, annotations

# import numpy as np
import torch

from torch import matrix_exp as expm
from l2hmc.group.pytorch.logm import log3x3

nn = torch.nn

Tensor = torch.Tensor


class SUN:
    def __init__(self) -> None:
        super(SUN, self).__init__()

    def exp(self, x: Tensor, u: Tensor) -> Tensor:
        return x @ expm(x.conj().transpose(-2, -1) @ u)

    def log(self, x: Tensor, y: Tensor) -> Tensor:
        _, n, _ = x.shape
        assert n == 3, "Operation supported only for SU(3)"
        return x @ log3x3(x.conj().transpose(-2, -1) @ y)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Arbitrary matrix C projects to skew-Hermitian

                    B := (C - C^†) / 2

        then make traceless with

                    tr{ B - {tr{B} / N} * I) = tr{B} - tr{B} = 0
        """
        _, n, _ = x.shape
        algebra_elem = torch.linalg.solve(u, x)[0]  # X^{-1} u

        # do projection in lie algebra
        B = (algebra_elem - algebra_elem.conj().transpose(-2, -1)) / 2
        trace = torch.einsum('bii->b', B)
        B = B - (
            (1 / n) * trace.unsqueeze(-1).unsqueeze(-1)
            * torch.eye(n).repeat(x.shape[0], 1, 1)
        )

        assert torch.abs(torch.mean(torch.einsum('bii->b', B))) < 1e-6

        return B
