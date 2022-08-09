"""
haarSUN.py

Inspired by implementation from:
    https://github.com/CUAI/Equivariant-Manifold-Flows
"""
from __future__ import absolute_import, print_function, division, annotations
from typing import Optional

import torch
import torch.distributions as distributions
import scipy.linalg
import numpy as np

from l2hmc.group.su3.pytorch.logm import su3_to_eigs

# from l2hmc.group.pytorch.logm import su3_to_eigs

Tensor = torch.Tensor


class HaarSUN(torch.distributions.Distribution):
    support = torch.distributions.constraints.real  # type:ignore
    has_rsample = True

    def __init__(self, dim: int = 3):
        self.dim = dim
        super().__init__()

    def rsample(self, nsamples: int) -> Tensor:
        """Produces n uniform samples over SU(self.dim)."""
        final = torch.zeros(nsamples, self.dim, self.dim).to(torch.complex64)
        for k in range(nsamples):
            z = torch.complex(
                torch.randn(self.dim, self.dim),
                torch.randn(self.dim, self.dim),
            )
            q, r = torch.linalg.qr(z / np.sqrt(2.0))
            diag = r.diag()
            ph = diag / diag.abs()
            q = q * ph * q
            q = q / (q.det() ** (1 / self.dim))
            final[k] = q
        return final

    def log_prob(self, z: Tensor) -> Tensor:
        """ log(z) = log p(v) - log det [ ∂_v proj_{µ} v ]"""
        _, n, _ = z.shape

        assert n == 3, 'Operation supported only for SU(3)'
        v = su3_to_eigs(z)

        # recall that eigdecomp returns [real1, real2, ..., imag1, imag2, ...]
        # use haar formula from boyda, Π_{i < j} | λ_i - λ_j | ²
        log_prob = torch.tensor(0.)
        for j in range(n):
            for i in range(j):
                log_prob += torch.log(torch.abs(v[:, i] - v[:, j]) ** 2)

        return log_prob

    def rsample_log_prob(self, nsamples: int):
        z = self.rsample(nsamples)
        return z, self.log_prob(z)
