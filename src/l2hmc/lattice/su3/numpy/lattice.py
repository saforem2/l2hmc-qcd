"""
su3/numpy/lattice.py

Implements SU3BaseLattice Class
"""
from __future__ import absolute_import, print_function, division, annotations

# import jax.numpy as jnp
import numpy as np

from l2hmc.group.tensorflow import group as g
# from lgt.group import group as g
# from typing import Generator
# from l2hmc.lattice.generators import generate_SU3

Array = np.ndarray
PI = np.pi
TWO_PI = 2. * np.pi


def pbc(tup: tuple[int], shape: tuple[int]) -> list:
    return np.mod(tup, shape).tolist()


def mat_adj(mat: Array) -> Array:
    return mat.conj().T


Site = tuple[int, int, int, int]                # t, x, y, z
Link = tuple[int, int, int, int, int]           # t, x, y, z, dim
Buffer = tuple[int, int, int, int, int, int]    # b, t, x, y, z, dim


class BaseLatticeSU3:
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
            `x.shape = [nb, 4, nt, nx, nx, nx, 3, 3]`.

        Args:
         - nb (int): Batch dimension, i.e. number of chains to run in parallel
         - nt (int): Temporal extent of lattice
         - nx (int): Spatial extent of lattice
         - dim (int): Number of dimensions, 4 by default (t, x, y, z)
        """
        self.dim = 4
        self.g = g.SU3()
        assert len(shape) == 4  # (nb, nt, nx, dim)
        self.c1 = c1
        self.link_shape = self.g.shape
        self.nt, self.nx, self.ny, self.nz = shape
        self._shape = (nb, 4, *shape, *self.g.shape)
        self.volume = self.nt * self.nx * self.ny * self.nz
        # ----------------------------------------------------------------
        # NOTE:
        #   - self.link_shape:             [*e] = [3, 3]
        #   - self.site_idxs:          [nb, *n] = [nb, t, x, y, z]
        #   - self.link_idxs:       [nb, *n, d] = [nb, t, x, y, z, d]
        #   - self.shape:       [nb, *n, d, *e] = [nb, t, x, y, z, d, 3, 3]
        #
        #   - where:                        t  in [0, 1, ..., (nt - 1)]
        #                            (x, y, z) in [0, 1, ..., (nx - 1)]
        #                                   d  in [0, 1, ..., (dim - 1)]
        # ----------------------------------------------------------------
        self.site_idxs = tuple(
            [self.nt] + [self.nx for _ in range(self.dim - 1)]
        )
        self.nplaqs = self.nt * self.nx
        self._lattice_shape = shape
        self.nsites = np.cumprod(shape)[-1]
        self.nlinks = self.nsites * self.dim
        self.link_idxs = tuple(list(self.site_idxs) + [self.dim])

    def coeffs(self, beta: Array) -> dict[str, Array]:
        """Coefficients for the plaquette and rectangle terms."""
        rect_coeff = beta * self.c1
        plaq_coeff = beta * (1.0 - 8.0 * self.c1)

        return {'plaq': plaq_coeff, 'rect': rect_coeff}

    def _link_staple_op(self, link: Array, staple: Array) -> Array:
        return self.g.mul(link, staple)

    def _plaquette(self, x: Array, u: int, v: int):
        xuv = self.g.mul(x[:, u], np.roll(x[:, v], shift=-1, axis=u + 1))
        xvu = self.g.mul(x[:, v], np.roll(x[:, u], shift=-1, axis=v + 1))
        return self.g.trace(self.g.mul(xuv, xvu, adjoint_b=True))

    def _wilson_loops(
            self,
            x: Array,
            needs_rect: bool = False
    ) -> tuple[list[Array], list[Array]]:
        # x.shape = [nb, nt, nx, nx, nx, d, 3, 3]
        # y.shape = [nb, d, nt, nx, nx, nx, 3, 3]
        assert len(x.shape) == 8
        plaqs = []
        rects = []
        for u in range(1, 4):
            for v in range(0, u):
                yuv = self.g.mul(x[:, u], np.roll(x[:, v], shift=-1, axis=u+1))
                yvu = self.g.mul(x[:, v], np.roll(x[:, u], shift=-1, axis=v+1))
                plaq = self.g.trace(self.g.mul(yuv, yvu, adjoint_b=True))
                plaqs.append(plaq)
                if needs_rect:
                    yu = np.roll(x[:, u], shift=-1, axis=v+1)
                    yv = np.roll(x[:, v], shift=-1, axis=u+1)
                    uu = self.g.mul(x[:, v], yuv, adjoint_a=True)
                    ur = self.g.mul(x[:, u], yvu, adjoint_a=True)
                    ul = self.g.mul(yuv, yu, adjoint_b=True)
                    ud = self.g.mul(yvu, yv, adjoint_b=True)
                    ul_ = np.roll(ul, shift=-1, axis=u+1)
                    ud_ = np.roll(ud, shift=-1, axis=v+1)
                    rects.append(
                        self.g.trace(self.g.mul(ur, ul_, adjoint_b=True))
                    )
                    rects.append(
                        self.g.trace(self.g.mul(uu, ud_, adjoint_b=True))
                    )

        return plaqs, rects

    def _plaquettes(self, x: Array) -> Array:
        ps, _ = self._wilson_loops(x)
        psum = 0.0
        for p in ps:  # NOTE: len(ps) == 6
            psum += np.sum(p.real, axis=range(1, len(p.shape)))

        # NOTE: return psum / (len(ps) * dim(link) * volume)
        return psum / (6 * 3 * self.volume)

    def action(self, x: Array, beta: Array):
        """Returns the action"""
        coeffs = self.coeffs(beta)
        ps, rs = self._wilson_loops(x, needs_rect=self.c1 != 0)
        psum = 0.0
        for p in ps:
            psum += np.sum(
                p.real,
                axis=range(1, len(p.shape))
            )

        action = coeffs['plaq'] * psum

        if self.c1 != 0:
            rsum = 0.0
            for r in rs:
                rsum += np.sum(
                    r.real,
                    axis=range(1, len(r.shape))
                )
            action += coeffs['rect'] * rsum
            # action += tf.math.multiply(coeffs['rect'], rsum)

        return action * (-1.0 / 3.0)

    def random(self):
        return self.g.random(self._shape)

    # def grad_action(self, x: Array, beta: Array) -> Array:
    #     """Returns the derivative of the action"""
    #     if tf.executing_eagerly():
    #         with tf.GradientTape(watch_accessed_variables=False) as tape:
    #             tape.watch(x)
    #             action = self.action(x, beta)
    #         g = tape.gradient(action, x)
    #     else:
    #         g = tf.gradients(self.action(x, beta), [x])[0]

    #     return self.g.projectTAH(self.g.mul(g, x, adjoint_b=True))
