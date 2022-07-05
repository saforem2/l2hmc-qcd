"""
lattice.py

Implements BaseLatticeU1 class in numpy
"""
from __future__ import absolute_import, division, print_function, annotations
from dataclasses import dataclass
from math import pi as PI
from typing import Optional
from scipy.special import i1, i0

import numpy as np

TWO_PI = 2. * PI
Array = np.ndarray


@dataclass
class Charges:
    intQ: Array
    sinQ: Array

    def asdict(self):
        return {'intQ': self.intQ, 'sinQ': self.sinQ}


@dataclass
class LatticeMetrics:
    plaqs: Array
    charges: Charges
    p4x4: Array

    def asdict(self):
        return {
            'plaqs': self.plaqs,
            'p4x4': self.p4x4,
            'sinQ': self.charges.sinQ,
            'intQ': self.charges.intQ,
        }


def area_law(beta: float, nplaqs: int):
    return (i1(beta) / i0(beta)) ** nplaqs


def plaq_exact(beta: float):
    return area_law(beta, nplaqs=1)


def project_angle(x: Array) -> Array:
    return x - TWO_PI * np.floor((x + PI) / TWO_PI)


class BaseLatticeU1:
    def __init__(self, nchains: int, shape: tuple[int, int]):
        self.nchains = nchains
        self._dim = 2
        assert len(shape) == 2
        self.nt, self.nx = shape
        self.xshape = (self._dim, *shape)
        self._shape = (nchains, *self.xshape)

        self.nplaqs = self.nt * self.nx
        self.nlinks = self.nplaqs * self._dim

    def draw_uniform_batch(self):
        unif = np.random.uniform(self.xshape)
        return TWO_PI * unif - PI

    def unnormalized_log_prob(self, x: Array) -> Array:
        return self.action(x=x)

    def action(
            self,
            x: Optional[Array] = None,
            wloops: Optional[Array] = None
    ) -> Array:
        """Calculate the Wilson gauge action for a batch of lattices."""
        wloops = self._get_wloops(x) if wloops is None else wloops
        return (1. - np.cos(wloops)).sum((1, 2))

    def calc_metrics(self, x: Array) -> dict[str, Array]:
        wloops = self.wilson_loops(x)
        plaqs = self.plaqs(wloops=wloops)
        charges = self.charges(wloops=wloops)
        return {'plaqs': plaqs, 'intQ': charges.intQ, 'sinQ': charges.sinQ}

    def observables(self, x: Array) -> LatticeMetrics:
        wloops = self.wilson_loops(x)
        return LatticeMetrics(p4x4=self.plaqs4x4(x=x),
                              plaqs=self.plaqs(wloops=wloops),
                              charges=self.charges(wloops=wloops))

    def wilson_loops(self, x: Array) -> Array:
        """Calculate the Wilson loops by summing links in CCW direction."""
        # --------------------------
        # NOTE: Watch your shapes!
        # --------------------------
        # * First, x.shape = [-1, 2, Lt, Lx], so
        #       (x_reshaped).T.shape = [2, Lx, Lt, -1]
        #   and,
        #       x0.shape = x1.shape = [Lx, Lt, -1]
        #   where x0 and x1 are the links along the 2 (t, x) dimensions.
        #
        # * The Wilson loop is then:
        #       wloop = U0(x, y) +  U1(x+1, y) - U0(x, y+1) - U(1)(x, y)
        #   and so output = wloop.T, with output.shape = [-1, Lt, Lx]
        # --------------------------
        x0, x1 = x.reshape(-1, *self.xshape).transpose(1, 2, 3, 0)
        return (x0 + np.roll(x1, -1, axis=0) - np.roll(x0, -1, axis=1) - x1).T

    def wilson_loops4x4(self, x: Array) -> Array:
        """Calculate the 4x4 Wilson loops"""
        x0, x1 = x.reshape(-1, *self.xshape).transpose(1, 0, 2, 3)
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
            x: Optional[Array] = None,
            wloops: Optional[Array] = None,
            # beta: float = None
    ) -> Array:
        """Calculate the average plaquettes for a batch of lattices."""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops = self.wilson_loops(x)

        return np.cos(wloops).mean((1, 2))

    def _plaqs4x4(self, wloops4x4: Array) -> Array:
        return np.cos(wloops4x4).mean((1, 2))

    def plaqs4x4(
            self,
            x: Optional[Array] = None,
            wloops4x4: Optional[Array] = None
    ) -> Array:
        """Calculate the 4x4 Wilson loops for a batch of lattices."""
        if wloops4x4 is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops4x4 = self.wilson_loops4x4(x)

        return self._plaqs4x4(wloops4x4)

    def _sin_charges(self, wloops: Array) -> Array:
        """Calculate sinQ from Wilson loops."""
        return np.sin(wloops).sum((1, 2)) / TWO_PI

    def _int_charges(self, wloops: Array) -> Array:
        """Calculate intQ from Wilson loops."""
        return project_angle(wloops).sum((1, 2)) / TWO_PI

    def _get_wloops(self, x: Optional[Array] = None) -> Array:
        if x is None:
            raise ValueError('One of `x` or `wloops` must be specified.')
        return self.wilson_loops(x)

    def sin_charges(
            self,
            x: Optional[Array] = None,
            wloops: Optional[Array] = None
    ) -> Array:
        """Calculate the real-valued charge approximation, sin(Q)."""
        wloops = self._get_wloops(x) if wloops is None else wloops
        return self._sin_charges(wloops)

    def int_charges(
            self,
            x: Optional[Array] = None,
            wloops: Optional[Array] = None
    ) -> Array:
        """Calculate the integer valued topological charge, int(Q)."""
        wloops = self._get_wloops(x) if wloops is None else wloops
        return self._int_charges(wloops)

    def charges(
            self,
            x: Optional[Array] = None,
            wloops: Optional[Array] = None
    ) -> Charges:
        """Calculate both charge representations and return as single object"""
        wloops = self._get_wloops(x) if wloops is None else wloops
        sinQ = self._sin_charges(wloops)
        intQ = self._int_charges(wloops)
        return Charges(intQ=intQ, sinQ=sinQ)
