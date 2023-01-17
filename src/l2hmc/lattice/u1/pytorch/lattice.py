"""
lattice.py

Contains pytorch implementation of Lattice object.

Author: Sam Foreman
Date: 06/02/2021
"""
from __future__ import absolute_import, annotations, division, print_function
# from dataclasses import dataclass
from typing import Optional
from math import pi as PI

# from scipy.special import i0, i1
from torch.special import i0, i1
import torch

from l2hmc.lattice.lattice import Lattice
# from l2hmc.lattice.u1.numpy.lattice import BaseLatticeU1
import l2hmc.group.u1.pytorch.group as g
# import l2hmc.group.pytorch.group as g
from l2hmc.configs import Charges, LatticeMetrics

TWOPI = 2. * PI
Tensor = torch.Tensor

DTYPE = torch.get_default_dtype()

if torch.cuda.is_available():
    DTYPE = torch.get_autocast_gpu_dtype()


def area_law(beta: float, nplaqs: int):
    return (i1(beta) / i0(beta)) ** nplaqs


def plaq_exact(beta: float | Tensor):
    if beta.dtype != torch.float32:
        beta = beta.to(torch.float32)
    # return i1(beta) / i0(beta)
    ratio = (i1(beta) / i0(beta))
    return ratio.to(DTYPE)


def project_angle(x: Tensor) -> Tensor:
    """For x in [-4pi, 4pi], returns x in [-pi, pi]."""
    return x - TWOPI * torch.floor((x + PI) / TWOPI)


class LatticeU1(Lattice):
    def __init__(self, nchains: int, shape: list[int]):
        assert len(shape) == 2
        self.g = g.U1Phase()
        self.nt, self.nx, = shape
        self.volume = self.nt * self.nx
        self.nplaqs = self.nt * self.nx
        # self.nsites = np.cumprod(shape)[-1]
        # self.nlinks = self.nsites * self.g._dim
        # self.site_idxs = tuple(
        #     [self.nt]
        #     + [self.nx for _ in range(self.g._dim - 1)]
        # )
        # self.link_idxs = tuple(list(self.site_idxs) + [self.g._dim])

        super().__init__(group=self.g, nchains=nchains, shape=shape)

    def draw_uniform_batch(self, requires_grad=True) -> Tensor:
        """Draw batch of samples, uniformly from (-pi, pi)."""
        return TWOPI * (
            torch.rand(self._shape, requires_grad=requires_grad)
        ) - PI

    def kinetic_energy(self, v: Tensor) -> Tensor:
        return 0.5 * v.flatten(1) ** 2

    def action(self, x: Tensor, beta: Tensor) -> Tensor:
        wloops = self._get_wloops(x)
        return self._action(wloops, beta)

    def _action(
            self,
            wloops: Tensor,
            beta: Tensor
    ) -> Tensor:
        """Calculate the Wilson gauge action for a batch of lattices."""
        return beta * (1. - wloops.cos()).sum((1, 2))

    def action_with_grad(
            self,
            x: Tensor,
            beta: Tensor
    ) -> tuple[Tensor, Tensor]:
        x.requires_grad_(True)
        s = self.action(x, beta)
        identity = torch.ones(x.shape[0], device=x.device)
        assert isinstance(s, Tensor)
        dsdx, = torch.autograd.grad(s, x,
                                    retain_graph=True,
                                    grad_outputs=identity)
        return s, dsdx

    def grad_action(
            self,
            x: Tensor,
            beta: Tensor,
            create_graph: bool = True,
    ) -> Tensor:
        """Compute the gradient of the potential function."""
        x.requires_grad_(True)
        s = self.action(x, beta)
        identity = torch.ones(x.shape[0], device=x.device)
        assert isinstance(s, Tensor)
        dsdx, = torch.autograd.grad(s, x,
                                    create_graph=create_graph,
                                    retain_graph=True,
                                    grad_outputs=identity)
        return dsdx

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
    ) -> dict[str, Tensor]:
        """Calculate various metrics and return as dict"""
        wloops = self.wilson_loops(x)
        plaqs = self.plaqs(wloops=wloops)
        charges = self.charges(wloops=wloops)
        return {
            'plaqs': plaqs,
            'intQ': charges.intQ,
            'sinQ': charges.sinQ
        }

    def observables(self, x: Tensor) -> LatticeMetrics:
        """Calculate observables and return as LatticeMetrics object"""
        wloops = self.wilson_loops(x)
        return LatticeMetrics(
            p4x4=self.plaqs4x4(x=x),
            plaqs=self.plaqs(wloops=wloops),
            charges=self.charges(wloops=wloops)
        )

    def wilson_loops(self, x: Tensor) -> Tensor:
        """Calculate the Wilson loops by summing links in CCW direction."""
        x = x.view((-1, *self.xshape))  # shape: [nchains, 2, Nt, Nx]
        xu = x[:, 0]  # shape: [nchains, Nt, Nx]
        xv = x[:, 1]  # shape: [nchains, Nt, Nx]
        return xu + xv.roll(-1, dims=1) - xu.roll(-1, dims=2) - xv

    def wilson_loops4x4(self, x: Tensor) -> Tensor:
        """Calculate the 4x4 Wilson loops"""
        x = x.reshape(-1, *self.xshape)
        xu = x[:, 0]
        xv = x[:, 1]
        # xu, xv = x.reshape(-1, *self.xshape).transpose(0, 1)
        # xv = xv.T
        # xu = xu.T
        return (
            xu                                  # Ux  [x, y]
            + xu.roll(-1, dims=2)               # Ux  [x+1, y]
            + xu.roll(-2, dims=2)               # Ux  [x+2, y]
            + xu.roll(-3, dims=2)               # Ux  [x+3, y]
            + xu.roll(-4, dims=2)               # Ux  [x+4, y]
            + xv.roll((-4, -1), dims=(2, 1))    # Uy  [x+4, y+1]
            + xv.roll((-4, -2), dims=(2, 1))    # Uy  [x+4, y+2]
            + xv.roll((-4, -3), dims=(2, 1))    # Uy  [x+4, y+3]
            - xu.roll((-3, -4), dims=(2, 1))    # -Ux [x+3, y+4]
            - xu.roll((-2, -4), dims=(2, 1))    # -Ux [x+2, y+4]
            - xu.roll((-1, -4), dims=(2, 1))    # -Ux [x+1, y+4]
            - xv.roll(-4, dims=1)               # -Uy [x, y+4]
            - xv.roll(-3, dims=1)               # -Uy [x, y+3]
            - xv.roll(-2, dims=1)               # -Uy [x, y+2]
            - xv.roll(-1, dims=1)               # -Uy [x, y+1]
            - xv                                # -Uy [x, y]
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

        return wloops.cos().mean((1, 2))

    def _plaqs(self, wloops: Tensor) -> Tensor:
        return wloops.cos().mean((1, 2))

    def _plaqs4x4(self, wloops4x4: Tensor) -> Tensor:
        return wloops4x4.cos().mean((1, 2))

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
        # return torch.sin(wloops).sum((1, 2)) / TWO_PI
        return wloops.sin().sum((1, 2)) / TWOPI

    def _int_charges(self, wloops: Tensor) -> Tensor:
        """Calculate intQ from Wilson loops."""
        return (project_angle(wloops)).sum((1, 2)) / TWOPI

    def _get_wloops(
            self,
            x: Optional[Tensor] = None
    ) -> Tensor:
        if x is None:
            raise ValueError('One of `x` or `wloops` must be specified.')
        return self.wilson_loops(x)

    def random(self) -> Tensor:
        return self.g.random(list(self._shape))

    def random_momentum(self) -> Tensor:
        return self.g.random_momentum(list(self._shape))

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

    def _charges(self, wloops: Tensor) -> Charges:
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
    ) -> Tensor:
        wloops1 = self._get_wloops(x1) if wl1 is None else wl1
        wloops2 = self._get_wloops(x2) if wl2 is None else wl2
        dwloops = 2. * (1. - (wloops2 - wloops1).cos())
        ploss = acc * dwloops.sum((1, 2)) + 1e-4
        assert isinstance(ploss, Tensor)

        return -ploss.mean(0)

    def charge_loss(
            self,
            acc: Tensor,
            x1: Optional[Tensor] = None,
            x2: Optional[Tensor] = None,
            wl1: Optional[Tensor] = None,
            wl2: Optional[Tensor] = None,
    ) -> Tensor:
        wloops1 = self._get_wloops(x1) if wl1 is None else wl1
        wloops2 = self._get_wloops(x2) if wl2 is None else wl2
        q1 = self._sin_charges(wloops=wloops1)
        q2 = self._sin_charges(wloops=wloops2)
        dq = (q2 - q1) ** 2
        qloss = acc * dq + 1e-4
        return -qloss.mean(0)


if __name__ == '__main__':
    lattice = LatticeU1(3, [8, 8])
    beta = torch.tensor(1.0)
    x = lattice.random()
    v = lattice.random_momentum()
    action = lattice.action(x, beta)
    kinetic = lattice.kinetic_energy(v)
