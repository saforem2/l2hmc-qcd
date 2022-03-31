"""
haarSUN.py

Modified from original implementation at:
    https://github.com/CUAI/Equivariant-Manifold-Flows
"""
from __future__ import absolute_import, print_function, division, annotations

import torch
import torch.distributions
import scipy
import scipy.linalg
import numpy as np

from l2hmc.lattice.pytorch.logm import su3_to_eigs_cdesa

Tensor = torch.Tensor


class HaarSUN(torch.distributions.Distribution):
    support = torch.distributions.constraints.real
    has_rsample = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def rsample(self, n: int, dim: int):
        """Produces n uniform samples over SU(dim)."""
        final = torch.zeros(n, dim, dim).to(torch.complex64)
        for k in range(n):
            z = scipy.randn(dim, dim) + 1j * scipy.randn(dim, dim)
            q, r = scipy.linalg.qr(z / np.sqrt(2.0))
            d = np.diag(r)
            ph = d / np.abs(d)
            q = np.multiply(q, ph, q)
            q = q / scipy.linalg.det(q) ** (1 / dim)
            final[k] = torch.tensor(q)

        return final

    def log_prob(self, z: Tensor) -> Tensor:
        """ log(z) = log p(v) - log det [ ∂_v proj_{µ} v ]"""
        _, n, _ = z.shape

        assert n == 3, 'Operation supported only for SU(3)'
        v = su3_to_eigs_cdesa(z)

        # recall that eigdecomp returns [real1, real2, ..., imag1, imag2, ...]
        # use haar formula from boyda, Π_{i < j} | λ_i - λ_j | ²
        log_prob = 0.
        for j in range(n):
            for i in range(j):
                log_prob += torch.log(torch.abs(v[:, i] - v[:, j]) ** 2)
        return log_prob

    def rsample_log_prob(self, shape=torch.Size()):
        z = self.rsample(shape)
        return z, self.log_prob(z)
