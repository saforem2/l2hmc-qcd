import numpy as np
from functools import reduce


class Lattice(object):
    """Lattice object."""
    def __init__(self, time_size, space_size, dim, beta, link_type, a=1):
        self.sites = np.zeros(
            [time_size] + [space_size for _ in range(dim-1)],
            dtype=link_type
        )
        self.links = np.zeros(
            list(self.sites.shape) + [dim],
            dtype=link_type
        )
        self.dim = dim  # dimensionality
        self.beta = beta
        self.link_type = link_type
        self.a = a  # lattice spacing
        self.bases = np.array([
            np.array([1, 0, 0, 0]),
            np.array([0, 1, 0, 0]),
            np.array([0, 0, 1, 0]),
            np.array([0, 0, 0, 1]),
        ])
        self.num_sites = reduce(lambda a, b: a*b, self.links.shape)

        for link in self.iter_links():
            self.links[link] = link_type.get_random_element()

    def iter_links(self):
        for site in self.iter_sites():
            for mu in range(self.dim):
                yield tuple(list(site) + [mu])

    def iter_sites(self):
        for i in range(self.num_sites):
            indices = list()
            for dim in self.sites.shape:
                indices.append(i % dim)
                i = i // dim
            yield tuple(indices)

    def get_neighbors(self, site):
        shape = self.sites.shape
        neighbors = list()
        for i, dim in enumerate(shape):
            e = list(site)
            if site[i] > 0:
                e[i] = e[i] - 1
            else:
                e[i] = dim - 1
            neighbors.append(tuple(e))

            e = list(site)
            if site[i] < dim - 1:
                e[i] = e[i] + 1
            else:
                e[i] = 0
            neighbors.append(tuple(e))
        return neighbors

    def get_random_site(self):
        return tuple([random.randint(0, d-1) for d in self.sites.shape])

    def get_random_link(self):
        return tuple([random.randint(0, d-1) for d in self.links.shape])

    def local_action(self, *links):
        S = 0.0
        for link in links:
            site1 = link[:-1]
            mu = link[-1]
            for nu in range(self.dim):
                if nu != mu:
                    site2 = np.array(site1) - self.bases[nu]
                    S += 5.0 / 3.0 * (self.plaquette_operator(site1, mu, nu)
                                      + self.plaquette_operator(site2, mu, nu))
        return S

    def total_action(self):
        S = 0.0
        for site in self.iter_sites():
            for mu in range(self.dim):
                for nu in range(self.dim):
                    if nu > mu:
                        S += 5.0 / 3.0 * self.plaquette_operator(site, mu, nu)
        return S

    def plaquette_operator(self, c, mu, nu):
        c = np.array(c)
        return 1.0 / 3.0 * np.trace(
            self.links[tuple(list(c % 5) + [mu])]
            * self.links[tuple(list((c + self.bases[mu]) % 5) + [nu])]
            * self.links[tuple(list((c + self.bases[nu]) % 5) 
                               + [mu])].conjugate().T
            * self.links[tuple(list(c % 5) + [nu])].conjugate().T
        ).real

    def rect_operator(self, c, mu, nu):
        c = np.array(c)
        return 1.0 / 3.0 * np.trace(
            self.links[tuple(list(c % 5) + [mu])]
            * self.links[tuple(list((c + self.bases[mu]) % 5) + [mu])]
            * self.links[tuple(list((c + 2 * self.bases[mu]) % 5) + [nu])]
            * self.links[tuple(list((c + self.bases[mu] + self.bases[nu]) % 5)
                               + [mu])].conjugate().T
            * self.links[tuple(list((c + self.bases[nu]) % 5)
                               + [mu])].conjugate().T
            * self.links[tuple(list(c % 5) + [nu])].conjugate().T
        ).real

    def hmc_update(self):
        """HMC update.

        TODO: Complete method.
        """
        pass
