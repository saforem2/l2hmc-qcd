"""
hmc.py

Defines class that implements a wrapper around tfp HMC implementation to make
running experiments easier.

Original code from:
    https://github.com/AVMCMC/AuxiliaryVariationalMCMC/blob/master/LearningToSample/src/HMC/hmc.py
"""
from __future__ import print_function, absolute_import, division
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from lattice.lattice import GaugeLattice

PARAMS = {
    'time_size': 8,
    'space_size': 8,
    'dim': 2,
    'link_type': 'U1',
    'num_samples': 5,
    'rand': False,
}


class HamiltonianMonteCarloSampler(object):
    """
    Wrapper around tfp HMC implementation to make running experiments easier.

    Attributes:
        time_size: Temporal extent of gauge lattice.
        space_size: Spatial extent of gauge lattice.
        dim: Dimensionality of gauge lattice.
        link_type: Gauge group of gauge lattice.
        num_samples: Number of samples / batch of link configurations of gauge
            lattice.
        rand: Whether or not samples start from random state. 
    """
    def __init__(self, params=None, eps=0.1, num_steps=10):
        # create instance attributes using key, value pairs from params
        if params is None:
            self.time_size = 8
            self.space_size = 8
            self.dim = 2
            self.link_type = 'U1'
            self.num_samples = 5
            self.rand = False
        else:
            for key, val in params.items():
                setattr(self, key, val)

        with tf.name_scope('lattice'):
            self.lattice = self._create_lattice()

        self.batch_size = self.lattice.samples.shape[0]
        self.samples = tf.convert_to_tensor(self.lattice.samples.flatten(),
                                            dtype=tf.float32)

        # pylint: disable=invalid-name
        if not tf.executing_eagerly():
            self.x = tf.placeholder(tf.float32, self.samples.shape, name='x')
            self.beta = tf.placeholder(tf.float32, shape=(), name='beta')

        with tf.name_scope('potential_fn'):
            self.potential_fn = self.lattice.get_energy_function(self.samples)

        self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            self.potential_fn, eps, num_steps, state_gradients_are_stopped=True
        )

    def _create_lattice(self):
        """Create and return gauge lattice object."""
        return GaugeLattice(time_size=self.time_size,
                            space_size=self.space_size,
                            dim=self.dim,
                            link_type=self.link_type,
                            num_samples=self.num_samples,
                            rand=self.rand)

    def sample(self, sess, num_samples, beta):
        """Create sample chain from `tfp.mcmc.sample_chain`."""
        samples_np = np.array(self.lattice.samples, dtype=np.float32)

        sample_op, _ = tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=0,
            current_state=self.x,
            kernel=self.hmc_kernel
        )

        time1 = time.time()
        out_samples = sess.run(sample_op, feed_dict={self.x: samples_np,
                                                     self.beta: beta})
        time2 = time.time()
        return out_samples, time2 - time1

    def potential_energy(self, position, beta):
        """Compute potential energy using `self.potential` and beta."""
        #  return beta * self.potential(position)
        with tf.name_scope('potential_energy'):
            potential_energy = tf.multiply(beta, self.potential_fn(position))

        return potential_energy

    def kinetic_energy(self, v):
        """Compute the kinetic energy."""
        with tf.name_scope('kinetic_energy'):
            kinetic_energy = 0.5 * tf.reduce_sum(v**2, axis=1)

        return kinetic_energy

    def hamiltonian(self, position, momentum, beta):
        """Compute the overall Hamiltonian."""
        with tf.name_scope('hamiltonian'):
            hamiltonian = (self.potential_energy(position, beta)
                           + self.kinetic_energy(momentum))
        return hamiltonian

    def metropolis_hastings_accept(energy_prev, energy_next):
        """Run Metropolis-Hastings algorithm for 1 step."""
        dE = energy_prev - energy_next
        return (tf.exp(dE) - tf.random_uniform(tf.shape(energy_prev))) >=0
