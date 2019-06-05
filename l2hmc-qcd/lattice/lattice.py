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
from globals import TF_FLOAT, NP_FLOAT


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
                 time_size,
                 space_size,
                 dim=2,
                 link_type='U1',
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
        self.link_shape = ()

        self.num_links = self.time_size * self.space_size * self.dim
        self.num_plaqs = self.time_size * self.space_size
        self.bases = np.eye(self.dim, dtype=np.int)

        self.samples = self._init_samples(num_samples, rand)
        self.samples_tensor = tf.convert_to_tensor(self.samples,
                                                   dtype=TF_FLOAT)
        self.links = self.samples[0]
        self.batch_size = self.samples.shape[0]
        self.links_shape = self.samples.shape[1:]

    def _init_samples(self, num_samples, rand):
        """Initialize samples."""
        links_shape = tuple(
            [self.time_size]
            + [self.space_size for _ in range(self.dim-1)]
            + [self.dim]
            + list(self.link_shape)
        )
        samples_shape = (num_samples, *links_shape)
        if rand:
            samples = np.array(
                np.random.uniform(0, 2*np.pi, samples_shape),
                dtype=NP_FLOAT
            )
        else:
            samples = np.zeros(samples_shape, dtype=NP_FLOAT)

        return samples

    def calc_plaq_sums(self, samples=None):
        """Calculate plaquette sums. 

        Explicitly, calculate the sum of the link variables around each
        plaquette in the lattice for each sample in samples.

        Args:
            samples (tf tensor): Tensor of shape (N, D) where N is the batch
                size and D is the number of links on the lattice. If samples is
                None, self.samples will be used.

        Returns:
            plaq_sums (tf operation): Tensorflow operation capable of
                calculating the plaquette sums.
        """
        if samples is None:
            samples = self.samples

        if samples.shape != self.samples.shape:
            samples = tf.reshape(samples, shape=(self.samples.shape))

        with tf.name_scope('calc_plaq_sums'):
            plaq_sums = (samples[:, :, :, 0]
                         - samples[:, :, :, 1]
                         - tf.roll(samples[:, :, :, 0], shift=-1, axis=2)
                         + tf.roll(samples[:, :, :, 1], shift=-1, axis=1))

        return plaq_sums

    def calc_actions(self, samples=None):
        """Calculate the total action for each sample in samples."""
        if samples is None:
            samples = self.samples

        with tf.name_scope('calc_total_actions'):
            total_actions = tf.reduce_sum(
                1. - tf.cos(self.calc_plaq_sums(samples)), axis=(1, 2),
                name='total_actions'
            )

        return total_actions

    def calc_plaqs(self, samples=None):
        """Calculate the average plaq. values for each sample in samples."""
        if samples is None:
            samples = self.samples

        with tf.name_scope('calc_plaqs'):
            plaqs = tf.reduce_sum(tf.cos(self.calc_plaq_sums(samples)),
                                  axis=(1, 2), name='plaqs') / self.num_plaqs
        return plaqs

    def calc_top_charges(self, samples=None, fft=False):
        """Calculate topological charges for each sample in samples."""
        if samples is None:
            samples = self.samples

        with tf.name_scope('calc_top_charges'):
            if fft:
                ps_proj = project_angle_fft(self.calc_plaq_sums(samples), N=1)
            else:
                ps_proj = project_angle(self.calc_plaq_sums(samples))

            top_charges = (tf.reduce_sum(ps_proj, axis=(1, 2),
                                         name='top_charges')) / (2 * np.pi)
        return top_charges

    # pylint: disable=invalid-name
    def calc_top_charges_diff(self, x1, x2, fft=False):
        """Calculate the difference in topological charge between x1 and x2."""
        with tf.name_scope('calc_top_charges_diff'):
            charge_diff = tf.abs(self.calc_top_charges(x1, fft)
                                 - self.calc_top_charges(x2, fft))

        return charge_diff

    def get_potential_fn(self, samples):
        """Returns callable function used for calculating the energy."""
        def fn(samples):
            return self.calc_actions(samples)
        return fn

