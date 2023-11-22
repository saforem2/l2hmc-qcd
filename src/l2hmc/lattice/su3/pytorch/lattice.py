"""
lattice.py

Contains implementation of generic GaugeLattice object.
"""
from __future__ import absolute_import, print_function, division, annotations

import numpy as np

from typing import Optional, Tuple
import logging

import torch
import l2hmc.group.su3.pytorch.group as g
from l2hmc.lattice.lattice import Lattice
from l2hmc.configs import Charges

log = logging.getLogger(__name__)
# from l2hmc import get_logger
# log = get_logger(__name__)

Array = np.ndarray
PI = np.pi
TWO_PI = 2. * np.pi

Tensor = torch.Tensor

Site: Tuple[int, int, int, int]                # t, x, y, z
Link: Tuple[int, int, int, int, int]           # t, x, y, z, dim
Buffer: Tuple[int, int, int, int, int, int]    # b, t, x, y, z, dim


def pbc(tup: tuple[int], shape: tuple[int]) -> list:
    return np.mod(tup, shape).tolist()


def mat_adj(mat: Array) -> Array:
    return mat.conj().T


class LatticeSU3(Lattice):
    """4D Lattice with SU(3) links."""
    dim = 4

    def __init__(
            self,
            nchains: int,
            shape: list[int],
            c1: float = 0.0,
    ) -> None:
        """4D SU(3) Lattice object for dealing with lattice quantities.

        NOTE:
         - `x.shape = [nb, 4, nt, nx, ny, nz, 3, 3]`
         - `x.dtype = complex128`

        Args:
         - nb (int): Batch dimension, i.e. number of chains to run in parallel
         - shape: Lattice shape, list or tuple of 4 ints
         - c1 (float): Factor multiplying the rectangular terms' contribution in
            the DBW2 action
        """
        assert len(shape) == 4  # (nb, nt, nx, dim)
        self.g = g.SU3()
        self.nt, self.nx, self.ny, self.nz = shape
        self.volume = self.nt * self.nx * self.ny * self.nz
        self.c1 = c1
        super().__init__(group=self.g, nchains=nchains, shape=shape)

    def random(self) -> Tensor:
        return self.g.random(list(self._shape))

    def random_momentum(self) -> Tensor:
        return self.g.random_momentum(list(self._shape))

    def update_link(
            self,
            x: Tensor,
            p: Tensor,
    ) -> Tensor:
        return self.g.mul(self.g.exp(p), x)

    def coeffs(self, beta: Tensor) -> dict[str, Tensor]:
        """Coefficients for the plaquette and rectangle terms.

        Reference: https://arxiv.org/pdf/hep-lat/0512017.pdf
        """
        rect_coeff = beta * self.c1
        plaq_coeff = beta * (torch.tensor(1.0) - torch.tensor(8.0) * self.c1)

        return {'plaq': plaq_coeff, 'rect': rect_coeff}

    def _link_staple_op(self, link: Tensor, staple: Tensor) -> Tensor:
        return self.g.mul(link, staple)

    def _rectangles(self, x: Tensor, u: int, v: int) -> tuple[Tensor, Tensor]:
        xu = x[:, u]
        xv = x[:, v]
        xuv = xu @ xv.roll(shifts=-1, dims=(u + 1))
        xvu = xv @ xu.roll(shifts=-1, dims=(v + 1))
        yu = xu.roll(-1, dims=v+1)
        yv = xv.roll(-1, dims=u+1)
        uu = xv.adjoint() @ xuv
        ur = xu.adjoint() @ xvu
        ul = xuv @ yu.adjoint()
        ud = xvu @ yv.adjoint()
        ul_ = ul.roll(-1, dims=u+1)
        ud_ = ud.roll(-1, dims=v+1)
        urul_ = ur @ ul_.adjoint()
        uuud_ = uu @ ud_.adjoint()

        return urul_, uuud_

    def _plaquette(
            self,
            x: Tensor,
            u: int,
            v: int,
    ):
        """U[μ](x) * U[ν](x+μ) * U†[μ](x+ν) * U†[ν](x)"""
        assert isinstance(x, Tensor)  # and len(x.shape.as_list > 1)
        xu = x[:, u]
        xv = x[:, v]
        xuv = xu @ xv.roll(shifts=-1, dims=(u + 1))
        xvu = xv @ xu.roll(shifts=-1, dims=(v + 1))
        return xuv @ xvu.adjoint()

    def _trace_plaquette(self, x: Tensor, u: int, v: int):
        """tr[Up] = tr{ U[μ](x) * U[ν](x+μ) * U†[μ](x+ν) * U†[ν](x) }"""
        return self.g.trace(self._plaquette(x, u, v))

    def _plaquette_field(self, x: Tensor, needs_rect: bool = False):
        # y.shape = [nb, d, nt, nx, nx, nx, 3, 3]
        x = x.view((x.shape[0], *self._shape[1:]))
        assert isinstance(x, Tensor)
        assert len(x.shape) == 8
        pcount = 0
        rcount = 0
        plaqs = []
        rects = []
        for u in range(1, self.dim):
            for v in range(0, u):
                plaq = self._plaquette(x, u, v)
                plaqs.append(plaq)
                # plaq = self.g.trace(self.g.mul(yuv, yvu, adjoint_b=True))
                pcount += 1

                # plaqs.append(plaq)
                if needs_rect:
                    urul_, uuud_ = self._rectangles(x, u, v)
                    rects.extend((urul_, uuud_))
                    rcount += 1
                else:
                    rects.extend((torch.zeros_like(plaq), torch.zeros_like(plaq)))
        return plaqs, rects

    def _wilson_loops(
            self,
            x: Tensor,
            needs_rect: bool = False
    ) -> tuple[Tensor, Tensor]:
        # y.shape = [nb, d, nt, nx, nx, nx, 3, 3]
        x = x.view((x.shape[0], *self._shape[1:]))
        assert isinstance(x, Tensor)
        assert len(x.shape) == 8
        pcount = 0
        rcount = 0
        plaqs = []
        rects = []
        for u in range(1, self.dim):
            for v in range(0, u):
                xu = x[:, u]
                xv = x[:, v]
                yuv = self.g.mul(xu, xv.roll(-1, dims=u+1))
                yvu = self.g.mul(xv, xu.roll(-1, dims=v+1))
                plaq = self.g.trace(self.g.mul(yuv, yvu, adjoint_b=True))
                plaqs.append(plaq)
                pcount += 1

                if needs_rect:
                    yu = xu.roll(-1, dims=v+1)
                    yv = xv.roll(-1, dims=u+1)
                    uu = self.g.mul(xv, yuv, adjoint_a=True)
                    ur = self.g.mul(xu, yvu, adjoint_a=True)
                    ul = self.g.mul(yuv, yu, adjoint_b=True)
                    ud = self.g.mul(yvu, yv, adjoint_b=True)
                    ul_ = ul.roll(-1, dims=u+1)
                    ud_ = ud.roll(-1, dims=v+1)
                    tr_urul_ = (
                        self.g.trace(self.g.mul(ur, ul_, adjoint_b=True))
                    )
                    tr_uuud_ = (
                        self.g.trace(self.g.mul(uu, ud_, adjoint_b=True))
                    )
                    rects.extend((tr_urul_, tr_uuud_))
                    rcount += 1
                else:
                    rects.extend((torch.zeros_like(plaq), torch.zeros_like(plaq)))
        return torch.stack(plaqs), torch.stack(rects)

    def _plaquettes(self, x: Tensor) -> Tensor:
        ps, _ = self._wilson_loops(x)
        psum = ps.real.sum(tuple(range(2, len(ps.shape)))).sum(0)

        # NOTE: return psum / (len(ps) * dim(link) * volume)
        return psum / (6 * 3 * self.volume)

    def _plaqs(self, wloops: Tensor) -> Tensor:
        psum = wloops.real.sum(tuple(range(2, len(wloops.shape)))).sum(0)

        return psum / (6 * 3 * self.volume)

    def charges(self, x: Tensor) -> Charges:
        ps, _ = self._wilson_loops(x)
        return Charges(intQ=self._int_charges(wloops=ps),
                       sinQ=self._sin_charges(wloops=ps))

    def int_charges(self, x: Tensor) -> Tensor:
        ps, _ = self._wilson_loops(x)
        return self._int_charges(ps)

    def sin_charges(self, x: Tensor) -> Tensor:
        ps, _ = self._wilson_loops(x)
        return self._sin_charges(ps)

    def _charges(self, wloops: Tensor) -> Charges:
        qsum = wloops.imag.sum(tuple(range(2, len(wloops.shape)))).sum(0)
        qint = qsum / (32 * (np.pi ** 2))
        qsin = qsum / (6 * 3 * self.volume)
        return Charges(intQ=qint, sinQ=qsin)

    def _int_charges(self, wloops: Tensor) -> Tensor:
        # TODO: IMPLEMENT
        qsum = wloops.imag.sum(tuple(range(2, len(wloops.shape)))).sum(0)
        return qsum / (32 * (np.pi ** 2))

    def _sin_charges(self, wloops: Tensor) -> Tensor:
        qsum = wloops.imag.sum(tuple(range(2, len(wloops.shape)))).sum(0)

        return qsum / (6 * 3 * self.volume)

    def wilson_loops(self, x: Tensor) -> Tensor:
        ps, _ = self._wilson_loops(x=x, needs_rect=False)
        return ps

    def kinetic_energy(self, v: Tensor) -> Tensor:
        return self.g.kinetic_energy(v)

    def potential_energy(self, x: Tensor, beta: Tensor) -> Tensor:
        return self.action(x, beta)

    def action(
            self,
            x: Tensor,
            beta: Tensor,
    ) -> Tensor:
        """Returns the action"""
        coeffs = self.coeffs(beta)
        ps, rs = self._wilson_loops(x, needs_rect=(self.c1 != 0))
        assert isinstance(x, Tensor)
        psum = ps.real.sum(tuple(range(2, len(ps.shape)))).sum(0)

        action = coeffs['plaq'] * psum

        if self.c1 != 0:
            rsum = rs.real.sum(tuple(range(2, len(rs.shape)))).sum(0)
            action = action + coeffs['rect'] * rsum

        return action * torch.tensor(-1.0 / 3.0)

    def _action(
            self,
            wloops: tuple[Tensor, Tensor],
            beta: Tensor,
    ) -> Tensor:
        coeffs = self.coeffs(beta)
        ps, rs = wloops
        # assert isinstance(x, Tensor)
        psum = ps.real.sum(tuple(range(2, len(ps.shape)))).sum(0)
        action = coeffs['plaq'] * psum
        if self.c1 != 0:
            rsum = rs.real.sum(tuple(range(2, len(rs.shape)))).sum(0)
            action = action + coeffs['rect'] * rsum

        return action / 3.0

    def action_with_grad(
            self,
            x: Tensor,
            beta: Tensor,
    ) -> tuple[Tensor, Tensor]:
        x.requires_grad_(True)
        s = self.action(x, beta)
        sc = torch.complex(s, torch.zeros_like(s))
        identity = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        dsdx, = torch.autograd.grad(sc, x, grad_outputs=identity)
        return s, self.g.projectTAH(dsdx @ x.adjoint())

    def grad_action(self, x: Tensor, beta: Tensor) -> Tensor:
        """Returns the derivative of the action"""
        x.requires_grad_(True)
        if x.shape != self._shape:
            x = x.reshape(x.shape[0], *self._shape[1:])
        s = self.action(x, beta)
        identity = torch.ones(x.shape[0], device=x.device)
        dsdx, = torch.autograd.grad(s, x, grad_outputs=identity)
        # return self.g.projectTAH(self.g.mul(dsdx, x, adjoint_b=True))
        return self.g.projectTAH(dsdx @ x.adjoint())

    def calc_metrics(
            self,
            x: Tensor,
            beta: Optional[Tensor] = None,
            xinit: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        plaqs = self._plaquettes(x)
        q = self.charges(x)
        metrics = {
            'plaqs': plaqs,
            'sinQ': q.sinQ,
            'intQ': q.intQ,
        }

        if beta is not None:
            # s = self.action(x, beta)
            s, dsdx = self.action_with_grad(x, beta)
            metrics.update({
                'action': s,
                'dsdx': dsdx,
            })
            if xinit is not None:
                # action_ = self.action(xinit, beta)
                s_, dsdx_ = self.action_with_grad(xinit, beta)
                metrics.update({
                    'daction': (s - s_).abs(),
                    'dsdx': (dsdx - dsdx_).abs()
                })

        if xinit is not None:
            wloops_ = self.wilson_loops(xinit)
            plaqs_ = self.plaqs(wloops=wloops_)
            q_ = self._charges(wloops=wloops_)
            metrics.update({
                'dplaqs': (plaqs - plaqs_).abs(),
                'dQint': (q.intQ - q_.intQ).abs(),
                'dQsin': (q.sinQ - q_.sinQ).abs(),
            })

        return metrics

    def plaq_loss(
            self,
            acc: Tensor,
            x1: Optional[Tensor] = None,
            x2: Optional[Tensor] = None,
            wloops1: Optional[Tensor] = None,
            wloops2: Optional[Tensor] = None
    ):
        log.error('TODO')

    def charge_loss(
            self,
            acc: Tensor,
            x1: Optional[Tensor] = None,
            x2: Optional[Tensor] = None,
            wloops1: Optional[Tensor] = None,
            wloops2: Optional[Tensor] = None
    ):
        log.error('TODO')


if __name__ == '__main__':
    lattice = LatticeSU3(3, [4, 4, 4, 8])
    beta = torch.tensor(1.0)
    x = lattice.random()
    v = lattice.random_momentum()
    action = lattice.action(x, beta)
    kinetic = lattice.kinetic_energy(v)
