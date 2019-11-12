"""
lattice.py

Contains implementation of GaugeLattice class.

Author: Sam Foreman (github: @saforem2)
Date: 01/15/2019
"""
from __future__ import absolute_import, division, print_function

import config as cfg
#  from config import NP_FLOAT, TF_FLOAT

import numpy as np
import tensorflow as tf

from scipy.special import i0, i1

__all__ = ['u1_plaq_exact', 'u1_plaq_exact_tf',
           'project_angle', 'project_angle_fft',
           'GaugeLattice']


NP_FLOAT = cfg.NP_FLOAT
TF_FLOAT = cfg.TF_FLOAT


def u1_plaq_exact(beta):
    """Computes the expected value of the `average` plaquette for U(1)."""
    return i1(beta) / i0(beta)


def u1_plaq_exact_tf(beta):
    return tf.math.bessel_i1(beta) / tf.math.bessel_i0(beta)


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


def project_angle_fft(x, N=10):
    """Use the fourier series representation `x` to approx `project_angle`.
    NOTE: Because `project_angle` suffers a discontinuity, we approximate `x`
    with its Fourier series representation in order to have a differentiable
    function when computing the loss.
    Args:
        x (array-like): Array to be projected.
        N (int): Number of terms to keep in Fourier series.
    """
    y = np.zeros(x.shape, dtype=NP_FLOAT)
    for n in range(1, N):
        y += (-2 / n) * ((-1) ** n) * tf.sin(n * x)
    return y


class GaugeLattice(object):
    """Lattice with Gauge field existing on links."""

    def __init__(self, 
                 time_size=8,
                 space_size=8,
                 dim=2,
                 link_type='U1',
                 batch_size=None,
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
        self.link_shape = ()

        self.num_links = self.time_size * self.space_size * self.dim
        self.num_plaqs = self.time_size * self.space_size
        self.bases = np.eye(self.dim, dtype=np.int)

        # create samples using @property method: `@samples.setter`
        self.samples = (batch_size, rand)
        self._samples_shape = self._samples.shape

        self.links = self.samples[0]
        self.batch_size = self.samples.shape[0]
        self.links_shape = self.samples.shape[1:]
        self.x_dim = self.num_links
        #  self.samples = self._init_samples(batch_size, rand)
        self.samples_tensor = tf.convert_to_tensor(
            self.samples.reshape((self.batch_size, self.x_dim)),
            dtype=TF_FLOAT
        )

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, args):
        """Create samples."""
        batch_size, rand = args
        links_shape = tuple(
            [self.time_size]
            + [self.space_size for _ in range(self.dim-1)]
            + [self.dim]
            + list(self.link_shape)
        )
        samples_shape = (batch_size, *links_shape)
        if rand:
            samples = np.array(
                np.random.uniform(0, 2*np.pi, samples_shape),
                dtype=NP_FLOAT
            )
        else:
            samples = np.zeros(samples_shape, dtype=NP_FLOAT)

        self._samples = samples

    def calc_plaq_sums(self, samples=None):
        """Calculate plaquette sums. 

        Explicitly, calculate the sum of the link variables around each
        plaquette in the lattice for each sample in samples.

        Args:
            samples (tf.Tensor): Tensor of shape (N, D) where N is the batch
                size and D is the number of links on the lattice. If samples is
                None, self.samples will be used.

        Returns:
            plaq_sums (tf operation): Tensorflow operation capable of
                calculating the plaquette sums.

        NOTE: self.samples.shape = (N, L, T, D), where:
            N = batch_size
            L = space_size 
            T = time_size
            D = dimensionality
        """
        if samples is None:
            samples = self.samples

        if isinstance(samples, np.ndarray):
            return self.calc_plaq_sums_np(samples)

        with tf.name_scope('plaq_sums'):
            if samples.shape != self.samples.shape:
                samples = tf.reshape(samples, shape=self.samples.shape)

            # assuming D = 2, plaq_sums will have shape: (N, L, T)
            plaq_sums = (samples[:, :, :, 0]
                         - samples[:, :, :, 1]
                         - tf.roll(samples[:, :, :, 0], shift=-1, axis=2)
                         + tf.roll(samples[:, :, :, 1], shift=-1, axis=1))

        return plaq_sums

    def calc_plaq_sums_np(self, samples):
        """Calculate plaquette sums.

        Same as `self.calc_plaq_sums` defined above, but to be used with
        `numpy.ndarray` objects.
        """
        assert isinstance(samples, (np.ndarray, list))
        if samples.shape != self.samples.shape:
            samples = np.reshape(samples, self.samples.shape)

        plaq_sums = (samples[:, :, :, 0]
                     - samples[:, :, :, 1]
                     - np.roll(samples[:, :, :, 0], shift=-1, axis=2)
                     + np.roll(samples[:, :, :, 1], shift=-1, axis=1))

        return plaq_sums

    def calc_actions(self, samples=None, plaq_sums=None):
        """Calculate the total action for each sample in samples."""
        if plaq_sums is None:
            if samples is None:
                samples = self.samples
            plaq_sums = self.calc_plaq_sums(samples)

        if isinstance(plaq_sums, np.ndarray):
            return self.calc_actions_np(plaq_sums)

        with tf.name_scope('actions'):
            total_actions = tf.reduce_sum(1. - tf.cos(plaq_sums),
                                          axis=(1, 2), name='actions')

        return total_actions

    def calc_actions_np(self, plaq_sums):
        """Calculate actions for `np.ndarray` objcts."""
        total_actions = np.sum(1. - np.cos(plaq_sums), axis=(1, 2))

        return total_actions

    def calc_plaqs(self, samples=None, plaq_sums=None):
        """Calculate the average plaq. values for each sample in samples."""
        if plaq_sums is None:
            if samples is None:
                samples = self.samples
            plaq_sums = self.calc_plaq_sums(samples)

        with tf.name_scope('plaqs'):
            #  plaqs = tf.reduce_mean(tf.cos(self.calc_plaq_sums(samples)),
            #                         axis=(1, 2), name='plaqs')
            plaqs = tf.reduce_sum(tf.cos(plaq_sums),
                                  axis=(1, 2), name='plaqs') / self.num_plaqs
        return plaqs

    def calc_plaqs_np(self, plaq_sums):
        plaqs = np.sum(np.cos(plaq_sums), axis=(1, 2)) / self.num_plaqs

        return plaqs

    def calc_top_charges(self, samples=None, plaq_sums=None):
        """Calculate topological charges for each sample in samples."""
        if plaq_sums is None:
            if samples is None:
                samples = self.samples
            plaq_sums = self.calc_plaq_sums(samples)

        with tf.name_scope('top_charges'):
            ps_proj = tf.sin(plaq_sums)
            #  if fft:
            #      ps_proj = tf.sin(self.calc_plaq_sums(samples))
            #  ps_proj = project_angle_fft(self.calc_plaq_sums(samples), N=1)
            #  else:
            #  ps_proj = tf.sin(self.calc_plaq_sums(samples))
            #  ps_proj = project_angle(self.calc_plaq_sums(samples))

            top_charges = (tf.reduce_sum(ps_proj, axis=(1, 2),
                                         name='top_charges')) / (2 * np.pi)
        return top_charges

    def calc_top_charges_diff(self, x1, x2):
        """Calculate the difference in topological charge between x1 and x2."""
        with tf.name_scope('top_charges_diff'):
            with tf.name_scope('charge1'):
                ps1 = self.calc_plaq_sums(samples=x1)
                q1 = self.calc_top_charges(plaq_sums=ps1)
            with tf.name_scope('charge2'):
                ps2 = self.calc_plaq_sums(samples=x2)
                q2 = self.calc_top_charges(plaq_sums=ps2)

            charge_diff = tf.abs(q1 - q2)

        return charge_diff

    def get_potential_fn(self, samples):
        """Returns callable function used for calculating the energy."""
        def fn(samples):
            return self.calc_actions(samples)
        return fn
