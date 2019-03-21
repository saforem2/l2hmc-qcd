import numpy as np
import tensorflow as tf
import random


class IsingLattice(object):
    """Lattice class."""
    def __init__(self, dim, num_sites, coupling=1., randomize=True):
        self._idxs = dim * (num_sites,)
        self.dim = dim
        self.sites = np.zeros(self._idxs)
        self.sites_shape = self.sites.shape
        self.sites_flat = self.sites.flatten()
        self.num_sites = np.cumprod(self.sites.shape)[-1]
        if randomize:
            self.randomize_sites()
        else:
            self.fill_sites(1)

    def iter_sites(self):
        """Method for createring iterable over all sites."""
        for i in range(self.num_sites):
            indices = list()
            for dim in self.sites.shape:
                indices.append(i % dim)
                i = i // dim
            yield tuple(indices)

    def _randomize(self):
        """Fill sites with random values drawn from {-1, +1}."""
        return 2 * np.random.randint(2, size=self.sites.shape) - 1
        #  self.sites = 2 * np.random.randint(2, size=self.sites.shape) - 1

    def randomize_sites(self):
        self.sites = self._randomize()

    def fill_sites(self, val):
        """Fill sites with `val`."""
        self.sites = val * np.ones(self._idxs)

    def _flatten_sites(self):
        """Return self.sites as flattened array."""
        return self.sites.flatten()

    def _reshape_sites(self, sites):
        """Reshape sites into array of shape dim * (num_sites,)."""
        return sites.reshape(self.sites_shape)

    def get_neighbors(self, site):
        """Get all nearest neighbors of `site`."""
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
        """Get random site from sites."""
        return tuple([random.randint(0, d-1) for d in self.sites.shape])

    def get_energy_function(self):
        def fn(batch, batch_size):
            return self.calc_energy(batch, batch_size)
        return fn

    def _calc_energy(self, sites=None):
        """Calculate total energy."""
        if sites is None:
            sites = self.sites

        energy = 0
        for site in self.iter_sites():
            S = sites[site]
            S_nbrs = [S * sites[nbr] for nbr in self.get_neighbors(site)]
            energy += sum(S_nbrs)
        return energy

    def calc_energy(self, batch, batch_size):
        """Calculate the energy for each element in batch.

        Args:
            batch (np.ndarray or list):
                Array containing `batch_size` samples of lattice.sites arrays.
                For example, each sample in batch would be a unique random
                configuration of sites.
            batch_size (int):
                Number of samples in batch
        Returns:
            energies (list):
                List containing the energy of each sample in batch.
        """
        energies = []
        try:
            if tf.executing_eagerly():
                #  if len(batch.numpy().shape) == 1:
                #      return self._calc_energy(batch)
                #  else:
                num_samples = batch.numpy().shape[0]
        except AttributeError:
            num_samples = batch_size

        for idx in range(min(num_samples, batch_size)):
            sites = batch[idx]
            energy = 0
            for site in self.iter_sites():
                S = sites[site]
                S_nbrs = [S * sites[nbr] for nbr in self.get_neighbors(site)]
                energy += sum(S_nbrs)

            energies.append(energy)

        return energies

    def calc_magnetization(self):
        """Calcaulte total magnetization."""
        return np.sum(self.sites)

    def get_probability(self, energy1, energy2, temperature):
        return np.exp((energy1 - energy2) / temperature)

    def _mcmove(self, temperature):
        """Method for implementing a single MCMC move."""
        for site in self.iter_sites():
            rand_site = self.get_random_site()
            S = self.sites[rand_site]
            S_nbrs = [self.sites[nbr] for nbr in self.get_neighbors(rand_site)]
            cost = 2 * S * sum(S_nbrs)
            if cost < 0:
                self.sites[rand_site] *= -1
            elif np.random.random() < np.exp(-cost / temperature):
                self.sites[rand_site] *= -1
        #  sites_new = np.copy(self.sites)
        #  rand_site = self.get_random_site()
        #  sites_new[rand_site] *= -1
        #  current_energy = self.calc_energy(self.sites)
        #  new_energy = self.calc_energy(sites_new)
        #  prob = self.get_probability(current_energy, new_energy, temperature)
        #  if prob > np.random.random():
        #      self.sites = sites_new

    def run_mcmc(self, temp_arr, mc_steps, eq_steps=None):
        #  energy_arr = np.zeros(temp_arr.shape)
        if eq_steps is None:
            eq_steps = .1 * mc_steps
        energy_arr = []
        for temp in temp_arr:
            energy = 0
            for step in range(eq_steps):
                self._mcmove(temp)
            for step in range(mc_steps):
                self._mcmove(temp)
                energy += self.calc_energy(self.sites)
            energy_val = energy / (self.num_sites * mc_steps)
            energy_arr.append(energy_val)
            print(f'Temp: {temp:.4g}, Energy: {energy_val}')
        return energy_arr

