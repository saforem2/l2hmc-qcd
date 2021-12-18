"""
lattice.py

Contains pytorch implementation of Lattice object.

Author: Sam Foreman
Date: 06/02/2021
"""
from __future__ import absolute_import, division, print_function, annotations
from collections import namedtuple

import torch
import numpy as np
from dataclasses import dataclass, asdict
from math import factorial
from math import pi as PI
from scipy.special import i1, i0

TWO_PI = 2. * PI


LatticeMetrics = namedtuple('LatticeMetrics', ['plaqs', 'p4x4', 'Qs', 'Qi'])


@dataclass
class Charges:
    Qs: torch.Tensor
    Qi: torch.Tensor

    def asdict(self):
        return asdict(self)


# TODO: Deal with bessel functions in `area_law` and `plaq_exact`
def bessel_i1(x: torch.Tensor, N=100):
    def numerator(x, k):
        return ((x ** 2 / 4) ** k)

    def denominator(k):
        return factorial(k) * factorial(k + 1)

    return 0.5 * x * torch.from_numpy(np.cumsum(np.array([
        numerator(x, k) / denominator(k) for k in range(N)
    ])))


def area_law(beta: float, num_plaqs: int):
    return (i1(beta) / i0(beta)) ** num_plaqs


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
        self._nt, self._nx, self._dim = self.x_shape
        self.num_plaqs = self._nt * self._nx
        self.num_links = self.num_plaqs * self._dim

    def unnormalized_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.calc_actions(x=x)

    def calc_observables(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        wloops = self.calc_wilson_loops(x)
        #  actions = self.calc_actions(wloops=wloops)
        charges = self.calc_both_charges(wloops=wloops)
        plaqs = self.calc_plaqs(wloops=wloops)  # , beta=beta)
        p4x4 = self.calc_plaqs4x4(x=x)
        metrics = {
            'p4': p4x4,
            'plaqs': plaqs,
            'Qi': charges.Qi,
            'Qs': charges.Qs,
        }
        # return Metrics(**metrics)
        return metrics

    def calc_wilson_loops(self, x: torch.Tensor) -> torch.Tensor:
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

    def calc_wilson_loops4x4(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1 = x.reshape(-1, *self.x_shape).T
        return (x0                                  # U0(x, y)
                + x0.roll(-1, dims=2)               # U0(x+1, y)
                + x0.roll(-2, dims=2)               # U0(x+2, y)
                + x0.roll(-3, dims=2)               # U0(x+3, y)
                + x0.roll(-4, dims=2)               # U0(x+4, y)
                + x1.roll((-4, -1), dims=(2, 1))    # U1(x+4, y+1)
                + x1.roll((-4, -2), dims=(2, 1))    # U1(x+4, y+2)
                + x1.roll((-4, -3), dims=(2, 1))    # U1(x+4, y+3)
                - x0.roll((-3, -4), dims=(2, 1))    # U0*(x+3, y+4)
                - x0.roll((-2, -4), dims=(2, 1))    # U0*(x+2, y+4)
                - x0.roll((-1, -4), dims=(2, 1))    # U0*(x+1, y+4)
                - x1.roll(-4, dims=1)               # U0*(x, y+4)
                - x1.roll(-3, dims=1)               # U1*(x, y+3)
                - x1.roll(-2, dims=1)               # U1*(x, y+2)
                - x1.roll(-1, dims=1)               # U1*(x, y+1)
                - x1)                               # U1*(x, y)

    def calc_plaqs4x4(
            self,
            x: torch.Tensor = None,
            wloops4x4: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculate the 4x4 Wilson loops for a batch of lattices."""
        if wloops4x4 is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops4x4 = self.calc_wilson_loops4x4(x)

        return torch.cos(wloops4x4).mean((1, 2))

    def calc_plaqs(
            self,
            x: torch.Tensor = None,
            wloops: torch.Tensor = None,
            # beta: float = None
    ) -> torch.Tensor:
        """Calculate the average plaquettes for a batch of lattices."""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops = self.calc_wilson_loops(x)

        return torch.cos(wloops).mean((1, 2))

    def calc_actions(
            self,
            x: torch.Tensor = None,
            wloops: torch.Tensor = None,
    ):
        """Calculate the Wilson gauge action for a batch of lattices."""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops = self.calc_wilson_loops(x)

        return (1. - torch.cos(wloops)).sum((1, 2))

    def calc_charges(
            self,
            x: torch.Tensor = None,
            wloops: torch.Tensor = None,
            use_sin: bool = False
    ):
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops = self.calc_wilson_loops(x)

        q = torch.sin(wloops) if use_sin else project_angle(wloops)

        return q.sum((1, 2)) / TWO_PI

    def calc_both_charges(
            self,
            x: torch.Tensor = None,
            wloops: torch.Tensor = None
    ) -> Charges:
        """Calculate the charges using both integer and sin representations."""
        if wloops is None:
            if x is not None:
                wloops = self.calc_wilson_loops(x)
            else:
                raise ValueError('One of `x` or `wloops` must be specified.')

        qs = torch.sin(wloops).sum((1, 2)) / TWO_PI
        qi = project_angle(wloops).sum((1, 2)) / TWO_PI

        return Charges(Qs=qs, Qi=qi)
