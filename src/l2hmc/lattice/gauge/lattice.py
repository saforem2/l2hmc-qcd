"""
lattice.py

Contains implementation of generic GaugeLattice object.
"""
from __future__ import absolute_import, print_function, division, annotations
from typing import Generator

import numpy as np
from src.l2hmc.lattice.gauge.generators import generate_SU3

Array = np.ndarray
PI = np.pi
TWO_PI = 2. * np.pi


def pbc(tup: tuple[int], shape: tuple[int]) -> list[Array]:
    return [np.mod(tup, shape)]


def mat_adj(mat: Array) -> Array:
    return mat.conj().T


def project_angle(x: Array) -> Array:
    return x - TWO_PI * np.floor((x + PI) / TWO_PI)


class LatticeSU3:
    """4D Lattice with SU(3) links."""
    # Site = tuple[int, int, int, int, int]
    # Link = tuple[int, int, int, int, int, int]
    def __init__(self, nb: int, nt: int, nx: int, dim: int = 4) -> None:
        self.nb = nb    # batch size
        self.nt = nt    # temporal extent
        self.nx = nx    # spatial extent
        self.dim = dim  # dimensionality
        self.link_shape = (3, 3)
        self.link_dtype = np.complexfloating
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
        self.site_idxs = tuple([nt] + [nx for _ in range(dim - 1)])
        self.link_idxs = tuple(list(self.site_idxs) + [dim])
        self.shape = tuple(list(self.link_idxs) + list(self.link_shape))
        self.nsites = np.cumprod(self.site_idxs)[-1]
        self.nlinks = self.nsites * self.dim
        self.nplaqs = self.nt * self.nx
        self.bases = np.eye(self.dim, dtype=np.integer)
        self.plaq_idxs, self.plaqs_dict = self._create_plaq_lookup_table()

    def _create_plaq_lookup_table(self) -> tuple[list, dict]:
        """Create dictionary of (site, plaq_idsx) to improve efficiency."""
        t = np.arange(self.nt)
        x = np.arange(self.nx)
        d = np.arange(self.dim)
        sites = [(i, a, b, c) for i in t for a in x for b in x for c in x]
        plaq_idxs = [
            list(s) + [i] + [j] for s in sites for i in d for j in d if j > i
        ]
        shape = self.site_idxs
        plaqs_dict = {}
        for p in plaq_idxs:
            *site, u, v = p
            idx1 = tuple(site + [u])
            idx2 = tuple(pbc(site + self.bases[u], shape) + [v])
            idx3 = tuple(pbc(site + self.bases[v], shape) + [u])
            idx4 = tuple(site + [v])
            plaqs_dict[tuple(site)] = [idx1, idx2, idx3, idx4]

        return plaq_idxs, plaqs_dict

    def _generate_links(self, eps: float = 0.) -> Array:
        """Generate array of randomly initialized link variables"""
        links = np.zeros(self.shape, dtype=np.complexfloating)
        if eps > 0:
            for link in self.iter_links():
                links[link] = generate_SU3(eps)

        return links

    def iter_sites(self) -> Generator:
        for i in range(self.nsites):
            indices = []
            for dim in self.site_idxs:
                indices.append(i % dim)
                i = i // dim
            yield tuple(indices)

    def iter_links(self) -> Generator:
        for site in self.iter_sites():
            for u in range(self.dim):
                yield tuple(list(site) + [u])

    def get_link(self, site, direction, shape, links):
        return links[tuple(pbc(site, shape) + [direction])]

    def _plaq_observables(self, links: Array, beta: float = 1.) -> dict:
        q = 0.
        plaqs = 0.
        action = 0.
        for plaq in self.plaq_idxs:
            *site, u, v = plaq
            plaq_sum = self._plaq_op(site, u, v, links)
            local_action = self._action_op(plaq_sum)
            action += 1. - local_action
            plaqs += local_action
            q += project_angle(plaq_sum)

        return {
            'action': beta * action,
            'plaqs': plaqs / self.nplaqs,
            'charge': int(q / TWO_PI),
        }

    def local_action(self, *local_links: Array, links: Array) -> float:
        """Compute local action (internal energy) of a collection of `links`"""
        action = 0.0
        for link in local_links:
            s1 = link[:-1]
            u = link[-1]
            for v in range(self.dim):
                if v != u:
                    s2 = np.array(s1) - self.bases[v]
                    p1 = self._plaq_op(s1, u, v, links)
                    p2 = self._plaq_op(s2, u, v, links)
                    action += (p1 + p2)
        return action

    def _total_action(self, links: Array, beta: float = 1.) -> float:
        """Computes total action for a single lattice of gauge links.

        Explicitly, this is done by summing the internal energy of each plaq
        over all plaqs.

        For SU(N), the action of a single plaquette is:

                            S = 1 - Re(Tr(P)),

        where P is the plaquette operator.
        """
        if links.shape != self.shape:
            links = links.reshape(self.shape)

        action = 0.
        for plaq in self.plaq_idxs:
            *site, u, v = plaq
            plaq_sum = self._plaq_op(site, u, v, links)
            local_action = self._action_op(plaq_sum)
            action += 1. - local_action

        return beta * action

    def total_action(self, inputs: Array, beta: float = 1.) -> Array:
        return np.array([self._total_action(x, beta) for x in inputs])

    def _grad_action(
            self,
            links: Array,
            beta: float = 1.,
            flatten: bool = True,
    ) -> Array:
        """Compute the gradient of the action for the input lattice links."""
        if len(links.shape) == 1:
            links = links.reshape(self.shape)

        grad_arr = np.zeros(links.shape)
        for n in self.iter_sites():
            for u in range(self.dim):
                grad = np.floating(0.0)
                for v in range(self.dim):
                    if v != u:
                        m = tuple(np.mod((n - self.bases[v]), self.site_idxs))
                        p1 = self._plaq_op(n, u, v, links)
                        p2 = self._plaq_op(m, u, v, links)
                        staple = (np.trace(p1).imag() - np.trace(p2).imag())
                        grad += staple / self.link_shape[0]
                        # dsdp1 = self._grad_action_op(p1)
                        # dsdp2 = self._grad_action_op(p2)

                grad_arr[n][u] = beta * grad
        if flatten:
            return grad_arr.flatten()

        return grad_arr

    def grad_action(
            self,
            inputs: Array,
            beta: float = 1.,
            flatten: bool = True
    ) -> Array:
        return np.array([self._grad_action(x, beta, flatten) for x in inputs])

    def _grad_action_op(self, plaq: Array) -> Array:
        return np.trace(plaq).imag() / self.link_shape[0]

    def _link_staple_op(self, link: Array, staple: Array) -> Array:
        return link @ staple

    def _plaq_op(
            self,
            site: tuple[int],
            u: int,
            v: int, links: Array
    ) -> Array:
        shape = self.site_idxs
        l1 = self.get_link(site, u, shape, links)
        l2 = self.get_link(site + self.bases[u], v, shape, links)
        l3 = self.get_link(site + self.bases[v], u, shape, links)
        l4 = self.get_link(site, v, shape, links)

        prod = l1 @ l2 @ l3.conj().T @ l4.conj().T

        return prod

    def _action_op(self, plaq: Array) -> Array:
        return 1.0 * np.trace(plaq).real() / self.link_shape[0]

    def _get_staples(self, site: Array, u: int, links: Array) -> list[Array]:
        if len(links.shape) == 2:  # (nbatch, *)
            links = links.reshape((links.shape[0], *self.shape))

        sites = self.site_idxs
        staples = []
        args = (self.site_idxs, links)
        for v in range(self.dim):
            staple = 0
            if v != u:
                ub = self.bases[u]
                vb = self.bases[v]
                link1 = self.get_link(site + ub, v, *args)
                link2 = self.get_link(site + vb, u, *args)
                link3 = self.get_link(site, v, *args)
                link4 = self.get_link(site + ub - vb, v, *args)
                link5 = self.get_link(site - vb, u, *args)
                link6 = self.get_link(site - ub, v, *args)

                prod1 = link1 @ link2.conj().T @ link3.conj().T
                prod2 = link4.conj().T @ link5.conj().T @ link6
                staple = prod1 + prod2

            staples.append(staple)

        return staples
