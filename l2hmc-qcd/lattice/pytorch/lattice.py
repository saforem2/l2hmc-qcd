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

@dataclass
class Charges:
    sinQ: torch.Tensor
    intQ: torch.Tensor

    def asdict(self):
        return asdict(self)


LatticeMetrics = namedtuple('LatticeMetrics', ['plaqs', 'p4x4',
                                               'sinQ', 'intQ'])


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

    def unnormalized_log_prob(self, x: torch.Tensor):
        return self.calc_actions(x=x)

    def calc_observables(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        wloops = self.calc_wilson_loops(x)
        #  actions = self.calc_actions(wloops=wloops)
        charges = self.calc_both_charges(wloops=wloops)
        plaqs = self.calc_plaqs(wloops=wloops) # , beta=beta)
        p4x4 = self.calc_plaqs4x4(x=x)
        metrics = {
            'p4x4': p4x4,
            'plaqs': plaqs, 
            'intQ': charges.intQ,
            'sinQ': charges.sinQ,
        }
        # return Metrics(**metrics)
        return metrics

    def calc_wilson_loops(self, x: torch.Tensor):
        """Calculate the Wilson loops by summing links in CCW direction."""
        x0, x1 = x.reshape(-1, *self.x_shape).T
        return (x0 + x1.roll(-1, dims=0) - x0.roll(-1, dims=1) - x1).T

    def calc_wilson_loops4x4(self, x: torch.Tensor):
        x = x.reshape(-1, *self.x_shape).T
        # x0 = x[0]
        # x1 = x[..., 1]
        wl4x4 = (x[0]                                 # U0(x, y)
                + x[0].roll(-1, dims=2)               # U0(x+1, y)
                + x[0].roll(-2, dims=2)               # U0(x+2, y)
                + x[0].roll(-3, dims=2)               # U0(x+3, y)
                + x[0].roll(-4, dims=2)               # U0(x+4, y)
                + x[1].roll((-4, -1), dims=(2, 1))    # U1(x+4, y+1)
                + x[1].roll((-4, -2), dims=(2, 1))    # U1(x+4, y+2)
                + x[1].roll((-4, -3), dims=(2, 1))    # U1(x+4, y+3)
                - x[0].roll((-3, -4), dims=(2, 1))    # U0*(x+3, y+4)
                - x[0].roll((-2, -4), dims=(2, 1))    # U0*(x+2, y+4)
                - x[0].roll((-1, -4), dims=(2, 1))    # U0*(x+1, y+4)
                - x[1].roll(-4, dims=1)               # U0*(x, y+4)
                - x[1].roll(-3, dims=1)               # U1*(x, y+3)
                - x[1].roll(-2, dims=1)               # U1*(x, y+2)
                - x[1].roll(-1, dims=1)               # U1*(x, y+1)
                - x[1])                               # U1*(x, y)

        return wl4x4

    def calc_plaqs4x4(
            self,
            x: torch.Tensor = None,
            wloops4x4: torch.Tensor = None,
            # beta: float = None
    ):
        """Calculate the 4x4 Wilson loops for a batch of lattices."""
        if x is None and wloops4x4 is None:
            raise ValueError('One of `x` or `wloops` must be specified.')

        if wloops4x4 is None:
            assert isinstance(x, (torch.Tensor, np.ndarray))
            wloops4x4 = self.calc_wilson_loops4x4(x)

        # if beta is not None:
        #     return area_law(beta, 16) - p4x4

        return torch.cos(wloops4x4).mean((1, 2))

    def calc_plaqs(
            self,
            x: torch.Tensor = None,
            wloops: torch.Tensor = None,
            # beta: float = None
    ):
        """Calculate the average plaquettes for a batch of lattices."""
        if wloops is None:
            try:
                assert isinstance(x, (torch.Tensor, np.ndarray))
                wloops = self.calc_wilson_loops(x)
            except ValueError:
                raise

        # plaqs = torch.mean(torch.cos(wloops), (1, 2))

        # if beta is not None:
        #     return plaq_exact(beta) - plaqs

        return torch.cos(wloops).mean((1, 2))


    def calc_actions(
            self,
            x: torch.Tensor = None,
            wloops: torch.Tensor = None,
    ):
        """Calculate the Wilson gauge action for a batch of lattices."""
        if wloops is None:
            try:
                assert isinstance(x, (torch.Tensor, np.ndarray))
                wloops = self.calc_wilson_loops(x)
            except ValueError:
                raise

        return (1. - torch.cos(wloops)).sum((1, 2))

    def calc_charges(
            self,
            x: torch.Tensor = None,
            wloops: torch.Tensor = None,
            use_sin: bool = False
    ):
        if wloops is None:
            try:
                assert isinstance(x, (torch.Tensor, np.ndarray))
                wloops = self.calc_wilson_loops(x)
            except ValueError:
                raise

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

        sinq = torch.sin(wloops).sum((1, 2)) / TWO_PI
        intq = project_angle(wloops).sum((1, 2)) / TWO_PI

        return Charges(sinQ=sinq, intQ=intq)
