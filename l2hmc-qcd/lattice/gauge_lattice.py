import numpy as np
import tensorflow as tf
import random
from functools import reduce
from scipy.linalg import expm
from scipy.special import i0, i1
#  from matrices import GELLMANN_MATRICES, PAULI_MATRICES
from .gauge_generators import generate_SU2, generate_SU3, generate_SU3_array

#  from HMC.hmc import HMC

import tensorflow.contrib.eager as tfe

EPS = 0.1

NUM_SAMPLES = 500
PHASE_MEAN = 0
PHASE_SIGMA = 0.5  # for phases within +/- π / 6 ~ 0.5
PHASE_SAMPLES = np.random.normal(PHASE_MEAN, PHASE_SIGMA, NUM_SAMPLES // 2)

# the random phases must come in +/- pairs to ensure ergodicity
RANDOM_PHASES = np.append(PHASE_SAMPLES, -PHASE_SAMPLES)

###############################################################################
#                      GLOBAL VARIABLES
# ------------------------------------------------------------------------------
NUM_CONFIGS_PER_SAMPLE = 10000
NUM_SAMPLES = 25
NUM_EQ_CONFIGS = 20000
NUM_CONFIGS = NUM_CONFIGS_PER_SAMPLE * NUM_SAMPLES
###############################################################################

def u1_plaq_exact(beta):
    return i1(beta) / i0(beta)

def pbc(tup, shape):
    return list(np.mod(tup, shape))

def mat_adj(mat):
    return tf.transpose(tf.conj(mat))  # conjugate transpose

def project_angle(x):
    """Function to project an angle from [-4pi, 4pi] to [-pi, pi]."""
    return x - 2 * np.pi * tf.math.floor((x + np.pi) / (2 * np.pi))


# pylint: disable=invalid-name, too-many-instance-attributes
class GaugeLattice(object):
    """Lattice with Gauge field existing on links."""
    def __init__(self,
                 time_size,
                 space_size,
                 dim,
                 beta,
                 link_type,
                 num_samples=None,
                 rand=False):
        """Initialization for GaugeLattice object.

        Args:
            time_size (int): Temporal extent of lattice.
            space_size (int): Spatial extent of lattice.
            dim (int): Dimensionality
            beta (float): Inverse coupling constant.
            link_type (str): String representing the type of gauge group for
                the link variables. Must be either 'U1', 'SU2', or 'SU3'
            num_samples (int): Number of sample lattices to use.
            rand (bool): Flag specifying if lattice should be initialized
                randomly or uniformly.
        """
        assert link_type.upper() in ['U1', 'SU2', 'SU3'], (
            "Invalid link_type. Possible values: U1', 'SU2', 'SU3'"
        )
        self.time_size = time_size
        self.space_size = space_size
        self.dim = dim
        self.beta = beta
        self.link_type = link_type
        self.link_shape = None

        self._init_lattice(link_type, rand)
        self.samples = None

        self.num_sites = np.cumproduct(self.site_idxs)[-1]
        self.num_links = int(self.dim * self.num_sites)
        self.num_plaquettes = self.time_size * self.space_size
        self.bases = np.eye(dim, dtype=np.int)

        if self.link_type == 'U1':
            self.plaquette_operator = self.plaquette_operator_u1
            self._action_op = self._action_op_u1

        else:
            self.plaquette_operator = self.plaquette_operator_suN
            self.action_op = self._action_op_suN

        if num_samples is not None:
            #  Create `num_samples` randomized instances of links array
            self.num_samples = num_samples
            self.samples = self.get_links_samples(num_samples, rand=rand,
                                                  link_type=self.link_type)
            self.samples[0] = self.links

    def _init_lattice(self, link_type, rand):
        """Initialize lattice by creating self.sites and sites.links variables.
        
        Link variables are randomly initialized to elements in their respective
        gauge group.
        """
        if link_type == 'SU2':
            self.link_shape = (2, 2)
            link_dtype = np.complex64

        if link_type == 'SU3':
            self.link_shape = (3, 3)
            link_dtype = np.complex64

        if link_type == 'U1':
            self.link_shape = ()
            link_dtype = np.float32

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

        self.num_sites = np.cumprod(self.sites.shape)[-1]
        self.num_links = self.num_sites * self.dim
        if self.link_type != 'U1':
            # Indices for individual sites and links
            self.site_idxs = self.sites.shape[:-2]
            self.link_idxs = self.links.shape[:-2]
        else:
            self.site_idxs = self.sites.shape
            self.link_idxs = self.links.shape

            if rand:
                self.links = np.array(np.random.uniform(0, 2*np.pi,
                                                        links_shape),
                                      dtype=np.float32)

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
                yield tuple(list(site) + [u])

    def get_neighbors(self, site):
        """Return neighbors of `site`."""
        shape = self.sites.shape
        neighbors = list()
        for i, dim in enumerate(shape):
            nbr = list(site)
            if site[i] > 0:
                nbr[i] = nbr[i] - 1
            else:
                nbr[i] = dim - 1
            neighbors.append(tuple(nbr))

            nbr = list(site)
            if site[i] < dim - 1:
                nbr[i] = nbr[i] + 1
            else:
                nbr[i] = 0
            neighbors.append(tuple(nbr))
        return neighbors

    def get_random_site(self):
        """Return indices of randomly chosen site."""
        return tuple([random.randint(0, d-1) for d in self.site_idxs])

    def get_random_link(self):
        """Return inidices of randomly chosen link."""
        return tuple([random.randint(0, d-1) for d in self.link_idxs])

    def get_link(self, site, direction, shape, links=None):
        """Returns the value of the link variable located at site + direction."""
        if links is None:
            links = self.links
        return links[tuple(pbc(site, shape) + [direction])]

    def get_energy_function(self, samples=None):
        """Returns function object used for calculating the energy (action)."""
        if samples is None:
            def fn(links):
                return self._total_action(links)
        else:
            def fn(samples):
                return self.total_action(samples)
        return fn

    def get_grad_potential_fn(self, samples=None):
        """
        Returns function object used for calculating the gradient of the
        potential.
        """
        if samples is None:
            def fn(links):
                return self._grad_action(links)
        else:
            def fn(samples):
                return self.grad_action(samples)
        return fn

    def _calc_plaq_observables(self, links):
        """Computes the average plaquette of a particular lattice of links."""
        if links.shape != self.links.shape:
            links = tf.reshape(links, shape=self.links.shape)

        plaquettes_sum = 0.
        topological_charge = 0.
        total_action = 0.
        #  for site in self.iter_sites():
        #      for u in range(self.dim):
        #          for v in range(self.dim):
        #              if v > u:
        for plaq in self.plaquette_idxs:
            *site, u, v = plaq
            plaq_sum = self.plaquette_operator(site, u, v, links)
            local_action = self._action_op(plaq_sum)

            total_action += 1 - local_action
            plaquettes_sum += local_action
            topological_charge += project_angle(plaq_sum)

        return [self.beta * total_action,
                plaquettes_sum / self.num_plaquettes,
                int(topological_charge / (2 * np.pi))]

    #  def _calc_plaq_observables_np(self, links):
    #      """Compute the average plaquette of a particular lattice of links."""
    #      if links.shape != self.links.shape:
    #          links = links.reshape(*self.links.shape)
    #
    #      shape = self.site_idxs
    #      plaquettes_sum = 0.
    #      topological_charge = 0.
    #      total_action = 0.
    #
    #      for plaq in self.plaquette_idxs:
    #          *site, u, v = plaq
    #          #  plaq_sum = self.plaquette_operator(site, u, v, links)
    #
    #          l1 = links[tuple(pbc(site, shape) + [u])]
    #          l2 = links[tuple(pbc(site + self.bases[u], shape) + [v])]
    #          l3 = links[tuple(pbc(site + self.bases[v], shape) + [u])]
    #          l4 = links[tuple(pbc(site, shape) + [v])]
    #
    #          plaq_sum = l1 + l2 - l3 - l4
    #
    #          local_action = np.cos(plaq_sum)
    #          total_action += 1 - local_action
    #          plaquettes_sum += local_action
    #          topological_charge += plaq_sum
    #          #  topological_charge += project_angle(plaq_sum)
    #
    #      return [self.beta * total_action,
    #              plaquettes_sum / self.num_plaquettes,
    #              topological_charge / (2 * np.pi)]

    def calc_plaq_observables(self, samples):
        """Calculate the average plaquette for each sample in samples."""
        if tf.executing_eagerly():
            return np.array([
                self._calc_plaq_observables(sample) for sample in samples
            ]).reshape((-1, 3)).T
        else:
            plaq_observables = []
            for idx in range(samples.shape[0]):
                observables = self._calc_plaq_observables(samples[idx])
                plaq_observables.extend(observables)

            return np.array(plaq_observables).reshape((-1, 3)).T

    #  def calc_plaq_observables_np(self, samples):
    #      """Calculate the average plaquette for each sample in samples."""
    #      return [self._calc_plaq_observables_np(sample) for sample in samples]

    def local_action(self, *links, all_links):
        """Compute local action (internal energy) of a collection of `links`
        that belong to lattice.

        Args:
            *links (array-like):
                Collection of links over which to calculate the local action.
            all_links (array-like):
                Links array, shape = self.links.shape 
        """
        S = 0.0
        for link in links:
            site1 = link[:-1]
            u = link[-1]
            for v in range(self.dim):
                if v != u:
                    site2 = np.array(site1) - self.bases[v]
                    plaq1 = self.plaquette_operator(site1, u, v, all_links)
                    plaq2 = self.plaquette_operator(site2, u, v, all_links)
                    S += (plaq1 + plaq2)
        return S

    def _total_action(self, links):
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
        if links.shape != self.links.shape:
            links = tf.reshape(links, self.links.shape)

        total_action = 0.0
        for plaq in self.plaquette_idxs:
            *site, u, v = plaq
            plaq_sum = self.plaquette_operator(site, u, v, links)
            local_action = self._action_op(plaq_sum)

            total_action += 1 - local_action

        return self.beta * total_action

    def total_action(self, samples):
        """
        Return the total action (sum over all plaquettes) for each sample in
        samples, at inverse coupling strength `self.beta`. 

        Args:
            samples (array-like):
                Array of `links` arrays, each one representing the links of
                an individual lattice.  NOTE: If samples is None, only a
                single `GaugeLattice` was instantiated during the __init__
                method, so this will return the total action of that single
                lattice.
        Returns:
            _ (float or list of floats): 
                If samples is None, returns action of instantiated lattice
                as a float. Otherwise, returns list containing the action
                of each sample in samples.
        """
        if tf.executing_eagerly():
            return [self._total_action(sample) for sample in samples]
        else:
            total_actions = []
            for idx in range(samples.shape[0]):
                total_actions.append(self._total_action(samples[idx]))

            return total_actions

    def _grad_action(self, links=None, flatten=True):
        """Compute the gradient of the action for the array of link variables.
        Args:
            links: Array of link variables. If None is provided, self.links is
                used.
        Returns:
            grad_arr: Array containing the gradient of the action. Same shape
                as links.
        """
        if links is None:
            links = self.links

        if len(links.shape) == 1:
            links = tf.reshape(links, self.links.shape)

        grad_arr = np.zeros(links.shape)
        shape = self.site_idxs
        for site in self.iter_sites():
            for u in range(self.dim):
                grad = np.float32(0.0)
                for v in range(self.dim):
                    if v != u:
                        site2 = np.mod((site - self.bases[v]), shape)
                        plaq1 = self.plaquette_operator(site, u, v, links)
                        plaq2 = self.plaquette_operator(site2, u, v, links)
                        grad += (self._grad_action_op(plaq1)
                                 - self._grad_action_op(plaq2))
                grad_arr[site][u] = self.beta * grad
                #  grad_arr[site][u] = self._local_grad_action(site, u)
        if flatten:
            return grad_arr.flatten()

        return grad_arr

    def grad_action(self, samples=None):
        """Return the gradient of the action for each sample in samples, at
        inverse coupling strength `self.beta`.

        NOTE: If samples is None, only a single `GaugeLattice` was instantiated
        during the __init__ method, so this will return the gradient of the
        action of that single lattice instance.
        """
        if samples is None:
            return self._grad_action()
        return [
            self._grad_action(sample) for sample in samples
        ]

    def _action_op_u1(self, plaq):
        """Operator used in calculating the action."""
        #  if self.link_type == 'U1':
            #  return np.cos(plaq)
        return tf.math.cos(plaq)

    def _action_op_suN(self, plaq):
        """Operator used in calculating the action for SU(n) gauge model."""
        return 1.0 * tf.real(tf.trace(plaq)) / self.link_shape[0]

    def _grad_action_op(self, plaq):
        """Operator used in calculating the gradient of the action."""
        if self.link_type == 'U1':
            return tf.math.sin(plaq)
        return tf.imag(tf.trace(plaq)) / self.link_shape[0]

    def _link_staple_op(self, link, staple):
        """Operator used in calculating the change in the action caused by
        updating an individual `link`."""
        if self.link_type == 'U1':
            return np.cos(link + staple)
        return tf.matmul(link, staple)

    def plaquette_operator_u1(self, site, u, v, links):
        """Local (counter-clockwise) plaquette operator calculated at `site`
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
        """
        shape = self.site_idxs

        l1 = links[tuple(pbc(site, shape) + [u])]                   # U(x; u)
        l2 = links[tuple(pbc(site + self.bases[u], shape) + [v])]   # U(x+u; v)
        l3 = links[tuple(pbc(site + self.bases[v], shape) + [u])]   # U(x+v; u)
        l4 = links[tuple(pbc(site, shape) + [v])]                   # U(x, v)

        #  l1 = self.get_link(site, u, shape, links)                  # U(x; u)
        #  l2 = self.get_link(site + self.bases[u], v, shape, links)  # U(x + u; v)
        #  l3 = self.get_link(site + self.bases[v], u, shape, links)  # U(x + v; u)
        #  l4 = self.get_link(site, v, shape, links)                  # U(x; v)
        return l1 + l2 - l3 - l4

    def plaquette_operator_suN(self, site, u, v, links):
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

    # pylint: disable=too-many-locals
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

                l4 = self.get_link(site + self.bases[u] - self.bases[v],
                                   v, shape, links)
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

    def rect_operator(self, site, u, v, links=None):
        """Rectangular plaquette operator."""
        if links is None:
            links = self.links

        shape = self.sites.shape

        #  site = np.array(site)
        l1 = tuple(pbc(site, shape) + [u])  #pylint: ignore invalid-name
        l2 = tuple(pbc(site + self.bases[u], shape) + [u])
        l3 = tuple(pbc(site + 2 * self.bases[u], shape) + [v])
        l4 = tuple(pbc(site + self.bases[u]+self.bases[v], shape) + [u])
        l5 = tuple(pbc(site + self.bases[v], shape) + [u])
        l6 = tuple(pbc(site, shape) + [v])

        if self.link_shape != ():
            return 1.0 * tf.real(tf.trace(links[l1]
                                          * links[l2]
                                          * links[l3]
                                          * tf.transpose(tf.conj(links[l4]))
                                          * tf.transpose(tf.conj(links[l5]))
                                          * tf.transpose(tf.conj(links[l6]))))
        return (links[l1] + links[l2] + links[l3]
                - links[l4] - links[l5] - links[l6])

    def _update_link(self, site, d, links=None):
        """Update the link located at site + d using Metropolis-Hastings
        accept/reject."""
        if links is None:
            links = self.links

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
        prob = min(1, np.exp(self.beta * (minus_proposed_action
                                          - minus_current_action)))
        accept = 0
        if np.random.uniform() < prob:
            self.links[tuple(pbc(site, shape) + [d])] = proposed_link
            accept = 1
        return accept

    def run_metropolis(self, links=None):
        """Run the MCMC simulation using Metropolis-Hastings accept/reject. """
        if links is None:
            links = self.links
        if len(links.shape) == 1:
            links = tf.reshape(links, self.links.shape)
        # relax the initial configuration
        eq_steps = 1000
        for step in range(eq_steps):
            for site in self.iter_sites():
                for d in range(self.dim):
                    _ = self._update_link(self, site, d)

        num_acceptances = 0  # keep track of acceptance rate
        for step in range(10000):
            for site in self.iter_sites():
                for d in range(self.dim):
                    num_acceptances += self._update_link(self, site, d)
