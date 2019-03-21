"""
lattice.py

Contains implementation of GaugeLattice class.

Author: Sam Foreman (github: @saforem2)
Date: 01/15/2019
"""
import os
import random
import pickle

import numpy as np
import tensorflow as tf
from functools import reduce
from scipy.linalg import expm
from scipy.special import i0, i1
from .gauge_generators import generate_SU2, generate_SU3, generate_SU3_array


EPS = 0.1

NUM_SAMPLES = 500
PHASE_MEAN = 0
PHASE_SIGMA = 0.5  # for phases within +/- π / 6 ~ 0.5
PHASE_SAMPLES = np.random.normal(PHASE_MEAN, PHASE_SIGMA, NUM_SAMPLES // 2)

# the random phases must come in +/- pairs to ensure ergodicity
RANDOM_PHASES = np.append(PHASE_SAMPLES, -PHASE_SAMPLES)

def u1_plaq_exact(beta):
    """Computes the expected value of the `average` plaquette for U(1)."""
    return i1(beta) / i0(beta)

def pbc(tup, shape):
    """Returns tup % shape for implementing periodic boundary conditions."""
    return list(np.mod(tup, shape))

def pbc_tf(tup, shape):
    """Tensorflow implementation of `pbc` defined above."""
    return list(tf.mod(tup, shape))

def mat_adj(mat):
    """Returns the adjoint (i.e. conjugate transpose) of a matrix `mat`."""
    return tf.transpose(tf.conj(mat))  # conjugate transpose

def project_angle(x):
    """Returns the projection of an angle `x` from [-4pi, 4pi] to [-pi, pi]."""
    return x - 2 * np.pi * tf.math.floor((x + np.pi) / (2 * np.pi))

def save_params_to_pkl_file(params, out_dir):
    if not os.path.isdir(out_dir):
        print(f'Creating directory: {out_dir}.')
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir)
    with open(out_file, 'wb') as f:
        pickle.dump(params, f)


class GaugeLattice(object):
    """Lattice with Gauge field existing on links."""
    def __init__(self, 
                 time_size, 
                 space_size, 
                 dim, 
                 link_type,
                 num_samples=None, 
                 rand=False):
        """Initialization for GaugeLattice object.

        Args:
            time_size (int): Temporal extent of lattice.
            space_size (int): Spatial extent of lattice.
            dim (int): Dimensionality
            link_type (str): 
                String representing the type of gauge group for the link
                variables. Must be either 'U1', 'SU2', or 'SU3'
        """
        assert link_type.upper() in ['U1', 'SU2', 'SU3'], (
            "Invalid link_type. Possible values: U1', 'SU2', 'SU3'"
        )

        self.time_size = time_size
        self.space_size = space_size
        self.dim = dim
        self.link_type = link_type
        #  self.data_format = data_format

        self.link_shape = None

        self._init_lattice(link_type, num_samples, rand)

    # pylint:disable=invalid-name
    def _init_lattice(self, link_type, num_samples, rand):
        """Initialize lattice, create self.links variables."""
        if link_type == 'SU2':
            self.link_shape = (2, 2)
            link_dtype = np.complex64
            self.plaquette_operator = self._plaquette_operator_SUN
            self.action_operator = self._action_operator_SUN

        if link_type == 'SU3':
            self.link_shape = (3, 3)
            link_dtype = np.complex64
            self.plaquette_operator = self._plaquette_operator_SUN
            self.action_operator = self._action_operator_SUN

        if link_type == 'U1':
            self.link_shape = ()
            link_dtype = np.float32
            self.plaquette_operator = self._plaquette_operator_U1
            self.action_operator = self._action_operator_U1

        sites_shape = tuple(
            [self.time_size]
            + [self.space_size for _ in range(self.dim-1)]
            + list(self.link_shape)
        )

        links_shape = tuple(
            [self.time_size]
            + [self.space_size for _ in range(self.dim-1)]
            + [self.dim]
            + list(self.link_shape)
        )

        self.sites = np.zeros(sites_shape, dtype=link_dtype)
        self.links = np.zeros(links_shape, dtype=link_dtype)

        if rand:
            self.links = np.array(
                np.random.uniform(0, 2*np.pi, links_shape),
                dtype=np.float32
            )

        if self.link_type != 'U1':
            self.site_idxs = self.sites.shape[:-2] # idxs for individ. sites
            self.link_idxs = self.links.shape[:-2] # idx for individ. links
        else:
            self.site_idxs = self.sites.shape  # idxs for individ. sites
            self.link_idxs = self.links.shape  # idxs for individ. links


        self.num_sites = np.cumprod(self.sites.shape)[-1]
        self.num_links = self.num_sites * self.dim
        self.num_plaquettes = self.time_size * self.space_size
        self.bases = np.eye(self.dim, dtype=np.int)

        self._create_plaq_lookup_table()

        if num_samples:
            self.num_samples = num_samples
            self.samples = self.get_links_samples(num_samples,
                                                  rand=rand,
                                                  link_type=self.link_type)
            self.samples[0] = self.links

        #  if self.data_format == 'channels_first':
        #      self.samples = self.samples.transpose(0, 3, 1, 2)
        #      self.links = self.links.transpose(2, 0, 1)
        #      self.link_idxs = self.links.shape

    def _create_plaq_lookup_table(self):
        """Create dictionary of (site, plaquette_idxs) to improve efficiency."""
        # Construct list containing the indices of the link variables for each
        # plaquette in the lattice to use as a lookup table instead of having
        # to perform nested loops
        t = np.arange(self.time_size)
        x = np.arange(self.space_size)
        u = np.arange(self.dim)
        v = np.arange(self.dim)
        s_tups = [(i, j) for i in t for j in x]  # site tuples
        self.plaquette_idxs = [
            list(s) + [i] + [j] for s in s_tups for i in u for j in v if j > i
        ]
        shape = self.site_idxs
        self.plaquettes_dict = {}
        for p in self.plaquette_idxs:
            *site, u, v = p
            #  if self.data_format == 'channels_first':
            #      idx1 = tuple([u] + site)
            #      idx2 = tuple([v] + pbc(site + self.bases[u], shape))
            #      idx3 = tuple([u] + pbc(site + self.bases[v], shape))
            #      idx4 = tuple([v] + site)
            #  elif self.data_format == 'channels_last':
            idx1 = tuple(site + [u])
            idx2 = tuple(pbc(site + self.bases[u], shape) + [v])
            idx3 = tuple(pbc(site + self.bases[v], shape) + [u])
            idx4 = tuple(site + [v])
            #  else:
            #      raise AttributeError(f"self.data_format expected to be "
            #                           f"one of 'channels_first' or "
            #                           f"channels_last.")

            self.plaquettes_dict[tuple(site)] = [idx1, idx2, idx3, idx4]

    def _generate_links(self, rand=False, link_type=None):
        """Method for obtaning an array of randomly initialized link variables.
            Args:
                link_type (str): 
                    Specifies the gauge group to be used on the links.

            Returns:
                _links (np.ndarray):
                    Array of the same shape as self.links.shape, containing
                    randomly initialized link variables.
        """
        if link_type is None:
            link_type = self.link_type

        if link_type == 'SU2':
            links = np.zeros(self.links.shape, dtype=np.complex64)
            if rand:
                for link in self.iter_links():
                    links[link] = generate_SU2(EPS)

        if link_type == 'SU3':
            links = np.zeros(self.links.shape, dtype=np.complex64)
            if rand:
                for link in self.iter_links():
                    links[link] = generate_SU3(EPS)

        if link_type == 'U1':
            if rand:
                links = 2 * np.pi * np.random.rand(*self.links.shape)
            else:
                links = np.zeros(self.links.shape)

        #  if self.data_format == 'channels_last':
        #      links = links.transpose((-1, *np.arange(len(links.shape) - 1)))

        return links

    def get_links_samples(self, num_samples, rand=False, link_type=None):
        """Return `num_samples` randomly initialized links arrays."""
        samples = np.array([
            self._generate_links(rand, link_type) for _ in range(num_samples)
        ])
        return samples

    def iter_sites(self):
        """Iterator for looping over sites."""
        for i in range(self.num_sites):
            indices = list()
            for dim in self.site_idxs:
                indices.append(i % dim)
                i = i // dim
            yield tuple(indices)

    def iter_links(self):
        """Iterator for looping over links."""
        for site in self.iter_sites():
            for u in range(self.dim):
                #  if self.data_format == 'channels_first':
                #      yield tuple([u] + list(site))
                #  else:
                yield tuple(list(site) + [u])

    def get_random_site(self):
        """Return indices of randomly chosen site."""
        return tuple([random.randint(0, d - 1) for d in self.site_idxs])

    def get_random_link(self):
        """Return indices of randomly chosen link."""
        return tuple([random.randint(0, d - 1) for d in self.link_idxs])

    def get_link(self, site, direction, shape, links=None):
        """Returns the value of the link variable located at site + direction."""
        if links is None:
            links = self.links
        return links[tuple(pbc(site, shape) + [direction])]

    def get_energy_function(self, samples=None):
        """Returns function object used for calculating the energy (action)."""
        with tf.name_scope('lattice_energy_fn'):
            #  if samples is None:
            #      def fn(links):
            #          return self._total_action(links)
            #  else:
            def fn(samples):
                return self.total_action(samples)
        return fn

    def calc_plaq_observables(self, samples):
        """Computes plaquette observables for an individual lattice of links.

        Args:
            links: Lattice of link variables.

        Returns:
            total_action: Total action of `links`.
            avg_plaq: Average plaquette value (sum of links around elementary
                plaquette).
            topological_charge: Topological charge of `links`.
        """
        if samples.shape != self.samples.shape:
            samples = tf.reshape(samples, shape=self.samples.shape)

        plaq_sums = (samples[:, :, :, 0]
                     - samples[:, :, :, 1]
                     - tf.roll(samples[:, :, :, 0], shift=-1, axis=2)
                     + tf.roll(samples[:, :, :, 1], shift=-1, axis=1))

        local_actions = tf.cos(plaq_sums)

        total_action = tf.reduce_sum(1. - local_actions, axis=(1, 2))
        avg_plaq = tf.reduce_sum(local_actions, (1, 2)) / self.num_plaquettes
        topological_charge = tf.floor(
            0.1 + tf.reduce_sum(project_angle(plaq_sums), (1, 2)) / (2 * np.pi)
        )

        return total_action, avg_plaq, topological_charge

    #  def calc_plaq_observables(self, samples):
    #      """Calculate plaquette observables for each sample in `samples.`
    #
    #      Args:
    #          samples: Array (a `batch`) of samples of link configurations.
    #          beta: Inverse coupling constant.
    #      """
    #      #  if samples.shape != self.samples.shape:
    #      #      samples = np.reshape(samples, self.samples.shape)
    #      if tf.executing_eagerly():
    #          return np.array([
    #              self._calc_plaq_observables(sample) for sample in samples
    #          ]).reshape((-1, 3)).T
    #
    #      plaq_observables = []
    #      for idx in range(samples.shape[0]):
    #          observables = self._calc_plaq_observables(samples[idx])
    #          plaq_observables.extend(observables)
    #
    #      #  np.array(plaq_observables).reshape((-1, 3)).T
    #      return tf.transpose(tf.reshape(plaq_observables, shape=(-1, 3)))

    def total_action(self, samples):
        """Computes the total action of an individual lattice by summing the
        internal energy of each plaquette over all plaquettes.

        NOTE:
            * For SU(N) (N = 2, 3), the action of a single plaquette is
            calculated as:
                Sp = 1 - Re{Tr(Up)}, where Up is the plaquette_operator defined
                as the product of the gauge fields around an elementary
                plaquette.
            * For U(1), the action of a sinigle plaquette is calculated as:
                Sp = 1 - cos(Qp), where Qp is the plaquette operator defined as
                the sum of the angles (phases) around an elementary plaquette.
        """
        if samples.shape != self.samples.shape:
            samples = tf.reshape(samples, shape=self.samples.shape)

        total_action = tf.reduce_sum(
            (1. - tf.cos(samples[:, :, :, 0] - samples[:, :, :, 1]
                         - tf.roll(samples[:, :, :, 0], shift=-1, axis=2)
                         + tf.roll(samples[:, :, :, 1], shift=-1, axis=1))),
            axis=(1, 2),
            name='total_action'
        )

        return total_action

    #  def _total_action_old(self, links):
    #      total_action = np.sum([
    #          1. - tf.cos(
    #              links[v[0]] + links[v[1]] - links[v[2]] - links[v[3]]
    #          ) for v in list(self.plaquettes_dict.values())
    #      ])
    #
    #      return total_action
    #
    #  def total_action_old(self, samples):
    #      """Return the total action (sum over all plaquettes) for each sample in
    #      samples, at inverse coupling strength `self.beta`.
    #
    #      Args:
    #          samples (array-like):
    #              Array of `links` arrays, each one representing the links of
    #              an individual lattice.  NOTE: If samples is None, only a
    #              single `GaugeLattice` was instantiated during the __init__
    #              method, so this will return the total action of that single
    #              lattice.
    #      Returns:
    #          _ (float or list of floats):
    #              If samples is None, returns action of instantiated lattice
    #              as a float. Otherwise, returns list containing the action
    #              of each sample in samples.
    #      """
    #      if samples.shape != self.samples.shape:
    #          samples = tf.reshape(samples, self.samples.shape)
    #
    #      if tf.executing_eagerly():
    #          return np.array([self._total_action_old(sample) for sample in samples],
    #                          dtype=np.float32)
    #
    #      with tf.name_scope('total_action'):
    #          total_actions = []
    #          for idx in range(samples.shape[0]):
    #              total_actions.append(self._total_action_old(samples[idx]))
    #
    #      return total_actions

    def _action_operator_SUN(self, plaq):
        """Operator used in calculating the action for SU(N) gauge model."""
        return 1.0 * tf.real(tf.trace(plaq)) / self.link_shape[0]

    def _action_operator_U1(self, plaq):
        """Operator used in calculating the Wilson action for U(1) gauge."""
        return tf.math.cos(plaq)

    def _link_staple_operator(self, link, staple):
        """Operator used in calculating invid. link contrib. to action."""
        if self.link_type == 'U1':
            return np.cos(link + staple)
        return tf.matmul(link, staple)

    def _plaquette_operator_U1(self, site, u, v, links):
        """Local (counter-clockwise) plaquette operator calculated at `site`

        Example: 
            Let [u] denote self.bases[u]. We compute the link sum 
            l1 + l2 - l3 - l4 for the Wilson action.

                     site - [v]          site + [u] + [v]
                                   l3
                            +------<<------+
                            |     (-u)     |
                            v              ^ 
                       l4   v (-v)    (+v) ^  l2
                            |      (u)     |
                            +------>>------+
                                   l1
                          site           site + [u]

        Args:
            site (tuple): 
                Starting point (lower left site) of plaquette loop
                calculation.
            u (int): 
                First direction (0 <= u <= self.dim - 1)
            v (int): 
                Second direction (0 <= v <= self.dim - 1)
            links (array-like): 
                Array of link variables (shape = self.links.shape). If none
                is provided, self.links will be used.
        Returns:
            _ (float): Plaquette sum l1 + l2 - l3 - l4
        """
        shape = self.site_idxs

        l1 = links[tuple(pbc(site, shape) + [u])]                   # U(x; u)
        l2 = links[tuple(pbc(site + self.bases[u], shape) + [v])]   # U(x+u; v)
        l3 = links[tuple(pbc(site + self.bases[v], shape) + [u])]   # U(x+v; u)
        l4 = links[tuple(pbc(site, shape) + [v])]                   # U(x, v)

        return l1 + l2 - l3 - l4

    def _plaquette_operator_SUN(self, site, u, v, links):
        """
        Local plaqutte operator at site in directions u, v for SU(N) model.
        """
        shape = self.site_idxs
        l1 = self.get_link(site, u, shape, links)
        l2 = self.get_link(site + self.bases[u], v, shape, links)
        l3 = self.get_link(site + self.bases[v], u, shape, links)
        l4 = self.get_link(site, v, shape, links)

        prod = tf.matmul(l1, l2)
        prod = tf.matmul(prod, mat_adj(l3))
        prod = tf.matmul(prod, mat_adj(l4))
        return prod

    # pylint:disable=too-many-locals
    def _get_staples(self, site, u, links=None):
        """Calculates each of the staples for the link variable at site + u."""
        if links is None:
            links = self.links
        if len(links.shape) == 1:
            links = tf.reshape(links, self.links.shape)

        shape = self.site_idxs

        staples = []
        for v in range(self.dim):  # u, v instead of mu, nu for readability
            if v != u:
                l1 = self.get_link(site + self.bases[u], v, shape, links)
                l2 = self.get_link(site + self.bases[v], u, shape, links)
                l3 = self.get_link(site, v, shape, links)

                l4 = self.get_link(site + self.bases[u] - self.bases[v], v,
                                   shape, links)
                l5 = self.get_link(site - self.bases[v], u, shape, links)
                l6 = self.get_link(site - self.bases[v], v, shape, links)

                if self.link_type == 'U1':
                    _sum1 = l1 - l2 - l3
                    _sum2 = -l4 -l5 + l6

                elif self.link_type in ['SU2', 'SU3']:
                    prod1 = tf.matmul(l1, mat_adj(l2))
                    prod1 = tf.matmul(prod1, mat_adj(l3))

                    prod2 = tf.matmul(mat_adj(l3), mat_adj(l4))
                    prod2 = tf.matmul(prod2, mat_adj(l6))

                    _sum = prod1 + prod2

                #_arr = [_sum1, _sum2]
                staples.append(_sum1)
                staples.append(_sum2)

        return staples

    def _update_link(self, site, d, links, beta):
        """Update the link located at site + d using Metropolis-Hastings
        accept/reject."""
        if links.shape != self.links.shape:
            links = tf.reshape(links, self.links.shape)

        shape = self.site_idxs

        staples = self._get_staples(site, d, links)

        current_link = self.get_link(site, d, shape, links)
        proposed_link = current_link + np.random.choice(RANDOM_PHASES)

        minus_current_action = np.sum(
            [np.cos(current_link + s) for s in staples]
        )
        minus_proposed_action = np.sum(
            [np.cos(proposed_link + s) for s in staples]
        )

        # note that if the proposed action is smaller than the current action,
        # prob > 1 and we accept the new link
        prob = min(1, np.exp(beta * (minus_proposed_action
                                          - minus_current_action)))
        accept = 0
        if np.random.uniform() < prob:
            self.links[tuple(pbc(site, shape) + [d])] = proposed_link
            accept = 1
        return accept

    def run_metropolis(self, beta, links=None):
        """Run the MCMC simulation using Metropolis-Hastings accept/reject. """
        if links is None:
            links = self.links
        if len(links.shape) == 1:
            links = tf.reshape(links, self.links.shape)
        # relax the initial configuration
        eq_steps = 1000
        for _ in range(eq_steps):
            for site in self.iter_sites():
                for d in range(self.dim):
                    _ = self._update_link(self, site, d, beta)

        num_acceptances = 0  # keep track of acceptance rate
        for step in range(10000):
            for site in self.iter_sites():
                for d in range(self.dim):
                    num_acceptances += self._update_link(self, site, d, beta)



