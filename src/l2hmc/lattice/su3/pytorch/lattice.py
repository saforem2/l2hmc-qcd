"""
lattice.py

Contains implementation of generic GaugeLattice object.
"""
from __future__ import absolute_import, print_function, division, annotations

import numpy as np

from typing import Optional, Tuple
import logging

import torch
from l2hmc.group.pytorch import group as g
from l2hmc.configs import Charges

log = logging.getLogger(__name__)
# from l2hmc.lattice.su3.lattice.

# from typing import Generator
# from l2hmc.lattice.generators import generate_SU3

Array = np.ndarray
PI = np.pi
TWO_PI = 2. * np.pi

Tensor = torch.Tensor
# Tensor = .types.experimental.Tensor


def pbc(tup: tuple[int], shape: tuple[int]) -> list:
    return np.mod(tup, shape).tolist()


def mat_adj(mat: Array) -> Array:
    return mat.conj().T


Site: Tuple[int, int, int, int]                # t, x, y, z
Link: Tuple[int, int, int, int, int]           # t, x, y, z, dim
Buffer: Tuple[int, int, int, int, int, int]    # b, t, x, y, z, dim


# ---------------------------------------------------------------
# TODO:
# - Finish implementation of BaseLatticeSU3
# - Write tensorflow and torch implementations as subclasses
#
# class LatticeSU3(BaseLatticeSU3):
# ---------------------------------------------------------------
class LatticeSU3:
    """4D Lattice with SU(3) links."""
    dim = 4

    def __init__(
            self,
            nb: int,
            shape: tuple[int, int, int, int],
            c1: float = 0.0,
    ) -> None:
        """4D SU(3) Lattice object for dealing with lattice quantities.

        NOTE:
            `x.shape = [nb, 4, nt, nx, ny, nz, 3, 3]`.

        Args:
         - nb (int): Batch dimension, i.e. number of chains to run in parallel
         - shape: Lattice shape, list or tuple of 4 ints
         - c1: Constant indicating whether or not to use rectangle terms ?
        """
        self.dim = 4
        self.g = g.SU3()
        assert len(shape) == 4  # (nb, nt, nx, dim)
        self.c1 = c1
        # self.c1 = torch.tensor(c1)
        # self.c1 = torch.tensor(c1)
        self.link_shape = self.g.shape
        self.nt, self.nx, self.ny, self.nz = shape
        self._shape = (nb, self.dim, *shape, *self.g.shape)
        self.volume = self.nt * self.nx * self.ny * self.nz
        self.site_idxs = tuple(
            [self.nt] + [self.nx for _ in range(self.dim - 1)]
        )
        self.nplaqs = self.nt * self.nx
        self._lattice_shape = shape
        self.nsites = np.cumprod(shape)[-1]
        self.nlinks = self.nsites * self.dim
        self.link_idxs = tuple(list(self.site_idxs) + [self.dim])

    def coeffs(self, beta: Tensor) -> dict[str, Tensor]:
        """Coefficients for the plaquette and rectangle terms."""
        rect_coeff = beta * self.c1
        plaq_coeff = beta * (torch.tensor(1.0) - torch.tensor(8.0) * self.c1)

        return {'plaq': plaq_coeff, 'rect': rect_coeff}

    def _link_staple_op(self, link: Tensor, staple: Tensor) -> Tensor:
        return self.g.mul(link, staple)

    def _plaquette(self, x: Tensor, u: int, v: int):
        """U[μ](x) * U[ν](x+μ) * U†[μ](x+ν) * U†[ν](x)"""
        assert isinstance(x, Tensor)  # and len(x.shape.as_list > 1)
        xuv = self.g.mul(x[:, u], x[:, v].roll(shifts=-1, dims=(u + 1)))
        xvu = self.g.mul(x[:, v], x[:, u].roll(shifts=-1, dims=(v + 1)))
        # xuv = self.g.mul(x[:, u], .roll(x[:, v], shift=-1, axis=u + 1))
        # xvu = self.g.mul(x[:, v], tf.roll(x[:, u], shift=-1, axis=v + 1))
        return self.g.trace(self.g.mul(xuv, xvu, adjoint_b=True))

    def _wilson_loops(
            self,
            x: Tensor,
            needs_rect: bool = False
    ) -> tuple[Tensor, Tensor]:
        # y.shape = [nb, d, nt, nx, nx, nx, 3, 3]
        x = x.view(self._shape)
        # x = tf.reshape(x, self._shape)
        assert isinstance(x, Tensor)
        assert len(x.shape) == 8
        # assert isinstance(x, Tensor)
        pcount = 0
        rcount = 0
        # plaqs = tf.TensorArray(x.dtype, size=0, dynamic_size=True)
        # rects = tf.TensorArray(x.dtype, size=0, dynamic_size=True)
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
                # plaqs = plaqs.write(pcount, plaq)
                pcount += 1

                # plaqs.append(plaq)
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
                    rects.append(tr_urul_)
                    rects.append(tr_uuud_)
                    rcount += 1
                else:
                    rects.append(torch.zeros_like(plaq))
                    rects.append(torch.zeros_like(plaq))

        return torch.stack(plaqs), torch.stack(rects)

    def _plaquettes(self, x: Tensor) -> Tensor:
        ps, _ = self._wilson_loops(x)
        psum = ps.real.sum(tuple(range(2, len(ps.shape)))).sum(0)
        # psum = torch.zeros_like(ps[0].real)
        # # psum = tf.zeros_like(tf.math.real(ps[0]))  # type: ignore
        # for p in ps:  # NOTE: len(ps) == 6
        #     psum = psum + p.real.sum(tuple(range(1, len(p.shape))))

        # NOTE: return psum / (len(ps) * dim(link) * volume)
        return psum / (6 * 3 * self.volume)

    def plaqs(self, wloops: Tensor) -> Tensor:
        # psum = tf.zeros_like(tf.math.real(wloops[0]))  # type:ignore
        # psum = torch.zeros_like(wloops[0].real)
        psum = wloops.real.sum(tuple(range(2, len(wloops.shape)))).sum(0)
        # p = wloops[0]
        # psum = p.real.sum(tuple(range(1, len(p.shape))))
        # for p in wloops[1:]:
        #     psum = psum + p.real.sum(tuple(range(1, len(p.shape))))

        return psum / (6 * 3 * self.volume)

    def charges(self, x: Tensor) -> Charges:
        ps, _ = self._wilson_loops(x)
        return Charges(intQ=self._int_charges(wloops=ps),
                       sinQ=self._sin_charges(wloops=ps))

    def int_charges(self, x: Tensor) -> Tensor:
        ps, _ = self._wilson_loops(x)
        return self._int_charges(ps)

    def _int_charges(self, wloops: Tensor) -> Tensor:
        # TODO: IMPLEMENT
        # qsum = torch.zeros_like(wloops[0].real)
        qsum = wloops.imag.sum(tuple(range(2, len(wloops.shape)))).sum(0)
        # p = wloops[0]
        # qsum = p.imag.sum(tuple(range(1, len(p.shape))))
        # for p in wloops[1:]:
        #     qsum = qsum + p.imag.sum(tuple(range(1, len(p.shape))))

        return qsum / (32 * (np.pi ** 2))

    def sin_charges(self, x: Tensor) -> Tensor:
        ps, _ = self._wilson_loops(x)
        return self._sin_charges(ps)

    def _sin_charges(self, wloops: Tensor) -> Tensor:
        qsum = wloops.imag.sum(tuple(range(2, len(wloops.shape)))).sum(0)

        return qsum / (6 * 3 * self.volume)

    def wilson_loops(self, x: Tensor) -> Tensor:
        ps, _ = self._wilson_loops(x=x, needs_rect=False)
        return ps

    def action(
            self,
            x: Tensor,
            beta: Tensor,
    ):
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

    def random(self):
        return self.g.random(list(self._shape))

    def grad_action(self, x: Tensor, beta: Tensor) -> Tensor:
        """Returns the derivative of the action"""
        x.requires_grad_(True)
        s = self.action(x, beta)
        identity = torch.ones(x.shape[0], device=x.device)
        dsdx, = torch.autograd.grad(s, x, grad_outputs=identity)

        return self.g.projectTAH(self.g.mul(dsdx, x, adjoint_b=True))

    def calc_metrics(
            self,
            x: Tensor,
            beta: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        wloops = self.wilson_loops(x)
        # ps, rs = self._wilson_loops(x, needs_rect=(self.c1 != 0))
        # plaqs = self.plaqs(wloops=wloops)

        # charges = self.charges(wloops=wloops)
        plaqs = self.plaqs(wloops)
        qsin = self._sin_charges(wloops)
        qint = self._int_charges(wloops)
        # qint = tf.zeros_like(qsin)
        metrics = {
            'sinQ': qsin,
            'intQ': qint,
            'plaqs': plaqs,
        }
        if beta is not None:
            action = self.action(x, beta)
            metrics['action'] = action

        # TODO: FIX ME
        # qsin = self._sin_charges(wloops=ps)
        # if beta is not None:
        #     pexact = plaq_exact(beta) * tf.ones_like(plaqs)
        #     metrics.update({
        #        'plaqs_err': pexact - plaqs
        #     })

        # metrics.update({
        #     'intQ': charges.intQ, 'sinQ': charges.sinQ
        # })
        return metrics
