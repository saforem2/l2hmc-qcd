"""
lattice.py

Contains pytorch implementation of Lattice object.

Author: Sam Foreman
Date: 06/02/2021
"""
from __future__ import absolute_import, division, print_function, annotations

import torch
import numpy as np
from dataclasses import dataclass
from math import factorial
from math import pi as PI
from scipy.special import i1, i0

TWO_PI = 2. * PI


@dataclass
class Charges:
    intQ: torch.Tensor
    sinQ: torch.Tensor

    def asdict(self):
        return {'intQ': self.intQ, 'sinQ': self.sinQ}


@dataclass
class LatticeMetrics:
    plaqs: torch.Tensor
    charges: Charges
    p4x4: torch.Tensor

    def asdict(self):
        return {
            'plaqs': self.plaqs,
            'p4x4': self.p4x4,
            'sinQ': self.charges.sinQ,
            'intQ': self.charges.intQ,
        }


# TODO: Deal with bessel functions in `area_law` and `plaq_exact`
def bessel_i1(x: torch.Tensor, N=100):
    def numerator(x, k):
        return ((x ** 2 / 4) ** k)

    def denominator(k):
        return factorial(k) * factorial(k + 1)

    return 0.5 * x * torch.from_numpy(np.cumsum(np.array([
        numerator(x, k) / denominator(k) for k in range(N)
    ])))


def area_law(beta: float, nplaqs: int):
    return (i1(beta) / i0(beta)) ** nplaqs


def plaq_exact(beta: float):
    return i1(beta) / i0(beta)


# def project_angle(x: torch.Tensor):
#     return x - TWO_PI * torch.floor(x + PI / TWO_PI)

def project_angle(x: torch.Tensor) -> torch.Tensor:
    """For x in [-4pi, 4pi], returns x in [-pi, pi]."""
    return x - TWO_PI * torch.floor((x + PI) / TWO_PI)


class Lattice:
    def __init__(self, shape: tuple):
        self._shape = shape
        self.batch_size, self.x_shape = shape[0], shape[1:]
        self.nt, self.nx, self._dim = self.x_shape
        self.nplaqs = self.nt * self.nx
        self.nlinks = self.nplaqs * self._dim

    def draw_uniform_batch(self, requires_grad=True):
        """Draw batch of samples, uniformly from [-pi, pi)."""
        unif = torch.rand(self._shape, requires_grad=requires_grad)
        return TWO_PI * unif - PI

    def unnormalized_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.action(x=x)

    def observables(self, x: torch.Tensor) -> LatticeMetrics:
        wloops = self.wilson_loops(x)
        return LatticeMetrics(p4x4=self.plaqs4x4(x=x),
                              plaqs=self.plaqs(wloops=wloops),
                              charges=self.charges(wloops=wloops))

    def wilson_loops(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the Wilson loops by summing links in CCW direction."""
        # --------------------------
        # NOTE: Watch your shapes!
        # --------------------------
        # * First, x.shape = [-1, Lt, Lx, 2], so
        #       (x_reshaped).T.shape = [2, Lx, Lt, -1]
        #   and,
        #       x0.shape = x1.shape = [Lx, Lt, -1]
        #   where x0 and x1 are the links along the 2 (t, x) dimensions.
        #
        # * The Wilson loop is then:
        #       wloop = U0(x, y) +  U1(x+1, y) - U0(x, y+1) - U(1)(x, y)
        #   and so output = wloop.T, with output.shape = [-1, Lt, Lx]
        # --------------------------
        x0, x1 = x.reshape(-1, *self.x_shape).T
        return (x0 + x1.roll(-1, dims=0) - x0.roll(-1, dims=1) - x1).T

    def wilson_loops4x4(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the 4x4 Wilson loops"""
        x0, x1 = x.reshape(-1, *self.x_shape).T
        return (
            x0                                  # Ux  [x, y]
            + x0.roll(-1, dims=2)               # Ux  [x+1, y]
            + x0.roll(-2, dims=2)               # Ux  [x+2, y]
            + x0.roll(-3, dims=2)               # Ux  [x+3, y]
            + x0.roll(-4, dims=2)               # Ux  [x+4, y]
            + x1.roll((-4, -1), dims=(2, 1))    # Uy  [x+4, y+1]
            + x1.roll((-4, -2), dims=(2, 1))    # Uy  [x+4, y+2]
            + x1.roll((-4, -3), dims=(2, 1))    # Uy  [x+4, y+3]
            - x0.roll((-3, -4), dims=(2, 1))    # -Ux [x+3, y+4]
            - x0.roll((-2, -4), dims=(2, 1))    # -Ux [x+2, y+4]
            - x0.roll((-1, -4), dims=(2, 1))    # -Ux [x+1, y+4]
            - x1.roll(-4, dims=1)               # -Uy [x, y+4]
            - x1.roll(-3, dims=1)               # -Uy [x, y+3]
            - x1.roll(-2, dims=1)               # -Uy [x, y+2]
            - x1.roll(-1, dims=1)               # -Uy [x, y+1]
            - x1                                # -Uy [x, y]
        ).T

    def plaqs(
            self,
            x: torch.Tensor = None,
            wloops: torch.Tensor = None,
            # beta: float = None
    ) -> torch.Tensor:
        """Calculate the average plaquettes for a batch of lattices."""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops = self.wilson_loops(x)

        return torch.cos(wloops).mean((1, 2))

    def _plaqs4x4(self, wloops4x4: torch.Tensor) -> torch.Tensor:
        return torch.cos(wloops4x4).mean((1, 2))

    def plaqs4x4(
            self,
            x: torch.Tensor = None,
            wloops4x4: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculate the 4x4 Wilson loops for a batch of lattices."""
        if wloops4x4 is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops4x4 = self.wilson_loops4x4(x)

        return self._plaqs4x4(wloops4x4)

    def _sin_charges(self, wloops: torch.Tensor) -> torch.Tensor:
        """Calculate sinQ from Wilson loops."""
        return torch.sin(wloops).sum((1, 2)) / TWO_PI

    def _int_charges(self, wloops: torch.Tensor) -> torch.Tensor:
        """Calculate intQ from Wilson loops."""
        return project_angle(wloops).sum((1, 2)) / TWO_PI

    def sin_charges(
            self,
            x: torch.Tensor = None,
            wloops: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculate the real-valued charge approximation, sin(Q)."""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops = self.wilson_loops(x)

        return self._sin_charges(wloops)

    def int_charges(
            self,
            x: torch.Tensor = None,
            wloops: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculate the integer valued topological charge, int(Q)."""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops = self.wilson_loops(x)

        return self._int_charges(wloops)

    def charges(
            self,
            x: torch.Tensor = None,
            wloops: torch.Tensor = None,
    ) -> Charges:
        """Calculate both charge representations and return as single object"""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops = self.wilson_loops(x)

        sinQ = self._sin_charges(wloops)
        intQ = self._int_charges(wloops)
        return Charges(intQ=intQ, sinQ=sinQ)

    def action(
            self,
            x: torch.Tensor = None,
            wloops: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculate the Wilson gauge action for a batch of lattices."""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops = self.wilson_loops(x)

        return (1. - torch.cos(wloops)).sum((1, 2))
