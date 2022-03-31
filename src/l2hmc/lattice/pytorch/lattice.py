"""
lattice.py

Contains pytorch implementation of Lattice object.

Author: Sam Foreman
Date: 06/02/2021
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass
from typing import Optional
from math import pi as PI

from scipy.special import i0, i1
import torch

from lgt.lattice.u1.numpy.lattice import BaseLatticeU1

TWO_PI = 2. * PI
Tensor = torch.Tensor


@dataclass
class Charges:
    intQ: Tensor
    sinQ: Tensor


@dataclass
class LatticeMetrics:
    plaqs: Tensor
    charges: Charges
    p4x4: Tensor


def area_law(beta: float, nplaqs: int):
    return (i1(beta) / i0(beta)) ** nplaqs


def plaq_exact(beta: float | Tensor):
    return i1(beta) / i0(beta)


def project_angle(x: Tensor) -> Tensor:
    """For x in [-4pi, 4pi], returns x in [-pi, pi]."""
    return x - TWO_PI * torch.floor((x + PI) / TWO_PI)


class LatticeU1(BaseLatticeU1):
    def __init__(self, nb: int, shape: tuple[int, int]):
        super().__init__(nb, shape=shape)

    def draw_uniform_batch(self, requires_grad=True) -> Tensor:
        """Draw batch of samples, uniformly from [-pi, pi)."""
        unif = torch.rand(self._shape, requires_grad=requires_grad)
        return TWO_PI * unif - PI

    def unnormalized_log_prob(self, x: Tensor) -> Tensor:
        return self.action(x=x)

    def action(
            self,
            x: Optional[Tensor] = None,
            wloops: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate the Wilson gauge action for a batch of lattices."""
        wloops = self._get_wloops(x) if wloops is None else wloops
        return (1. - torch.cos(wloops)).sum((1, 2))

    def plaqs_diff(
            self,
            beta: float,
            x: Optional[Tensor] = None,
            wloops: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate the difference between plaquettes and expected value"""
        wloops = self._get_wloops(x) if wloops is None else wloops
        plaqs = self.plaqs(wloops=wloops)
        pexact = plaq_exact(beta) * torch.ones_like(plaqs)
        return pexact - self.plaqs(wloops=wloops)

    def calc_metrics(
            self,
            x: Tensor,
            beta: Optional[float] = None
    ) -> dict[str, Tensor]:
        """Calculate various metrics and return as dict"""
        wloops = self.wilson_loops(x)
        plaqs = self.plaqs(wloops=wloops)
        charges = self.charges(wloops=wloops)
        metrics = {'plaqs': plaqs}
        if beta is not None:
            metrics.update({
               'plaqs_err': plaq_exact(torch.from_numpy(beta)) - plaqs
            })
        metrics.update({
            'intQ': charges.intQ, 'sinQ': charges.sinQ
        })

        return metrics

    def observables(self, x: Tensor) -> LatticeMetrics:
        """Calculate observables and return as LatticeMetrics object"""
        wloops = self.wilson_loops(x)
        return LatticeMetrics(p4x4=self.plaqs4x4(x=x),
                              plaqs=self.plaqs(wloops=wloops),
                              charges=self.charges(wloops=wloops))

    def wilson_loops(self, x: Tensor) -> Tensor:
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
        x0, x1 = x.reshape(-1, *self.xshape).transpose(0, 1)
        x0 = x0.T
        x1 = x1.T

        return (x0 + x1.roll(-1, dims=0) - x0.roll(-1, dims=1) - x1).T

    def wilson_loops4x4(self, x: Tensor) -> Tensor:
        """Calculate the 4x4 Wilson loops"""
        x0, x1 = x.reshape(-1, *self.xshape).transpose(0, 1)
        x0 = x0.T
        x1 = x1.T
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
            x: Optional[Tensor] = None,
            wloops: Optional[Tensor] = None,
            # beta: float = None
    ) -> Tensor:
        """Calculate the average plaquettes for a batch of lattices."""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops = self.wilson_loops(x)

        return torch.cos(wloops).mean((1, 2))

    def _plaqs4x4(self, wloops4x4: Tensor) -> Tensor:
        return torch.cos(wloops4x4).mean((1, 2))

    def plaqs4x4(
            self,
            x: Optional[Tensor] = None,
            wloops4x4: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate the 4x4 Wilson loops for a batch of lattices."""
        if wloops4x4 is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops4x4 = self.wilson_loops4x4(x)

        return self._plaqs4x4(wloops4x4)

    def _sin_charges(self, wloops: Tensor) -> Tensor:
        """Calculate sinQ from Wilson loops."""
        return torch.sin(wloops).sum((1, 2)) / TWO_PI

    def _int_charges(self, wloops: Tensor) -> Tensor:
        """Calculate intQ from Wilson loops."""
        return project_angle(wloops).sum((1, 2)) / TWO_PI

    def _get_wloops(
            self,
            x: Optional[Tensor] = None
    ) -> Tensor:
        if x is None:
            raise ValueError('One of `x` or `wloops` must be specified.')
        return self.wilson_loops(x)

    def sin_charges(
            self,
            x: Optional[Tensor] = None,
            wloops: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate the real-valued charge approximation, sin(Q)."""
        wloops = self._get_wloops(x) if wloops is None else wloops
        return self._sin_charges(wloops)

    def int_charges(
            self,
            x: Optional[Tensor] = None,
            wloops: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate the integer valued topological charge, int(Q)."""
        wloops = self._get_wloops(x) if wloops is None else wloops
        return self._int_charges(wloops)

    def charges(
            self,
            x: Optional[Tensor] = None,
            wloops: Optional[Tensor] = None
    ) -> Charges:
        """Calculate both charge representations and return as single object"""
        wloops = self._get_wloops(x) if wloops is None else wloops
        sinQ = self._sin_charges(wloops)
        intQ = self._int_charges(wloops)
        return Charges(intQ=intQ, sinQ=sinQ)

    def plaq_loss(
            self,
            acc: Tensor,
            x1: Optional[Tensor] = None,
            x2: Optional[Tensor] = None,
            wl1: Optional[Tensor] = None,
            wl2: Optional[Tensor] = None,
    ) -> float:
        wloops1 = self._get_wloops(x1) if wl1 is None else wl1
        wloops2 = self._get_wloops(x2) if wl2 is None else wl2
        dwloops = 2. * (1. - torch.cos(wloops2 - wloops1))
        ploss = acc * dwloops.sum((1, 2)) + 1e-4

        return -ploss.mean(0)

    def charge_loss(
            self,
            acc: Tensor,
            x1: Optional[Tensor] = None,
            x2: Optional[Tensor] = None,
            wl1: Optional[Tensor] = None,
            wl2: Optional[Tensor] = None,
    ):
        wloops1 = self._get_wloops(x1) if wl1 is None else wl1
        wloops2 = self._get_wloops(x2) if wl2 is None else wl2
        q1 = self._sin_charges(wloops=wloops1)
        q2 = self._sin_charges(wloops=wloops2)
        dq = (q2 - q1) ** 2
        qloss = acc * dq + 1e-4
        return -qloss.mean(0)
