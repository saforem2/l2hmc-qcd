"""
utils.py

Collection of functions for calculating various lattice observables.

Notably, this includes numpy-based methods for calculating the action,
plaquettes, and topological charge of a given lattice without needing to
instantiate an entire `GaugeLattice` object.

Author: Sam Foreman (github: @saforem2)
Date: 11/09/2019
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from scipy.special import i0, i1

from config import NP_FLOAT


def u1_plaq_exact(beta):
    """Computes the expected value of the `average` plaquette for U(1)."""
    return i1(beta) / i0(beta)


def u1_plaq_exact_tf(beta):
    """Computes the expected value of the avg. plaquette for 2D U(1)."""
    return tf.math.bessel_i1(beta) / tf.math.bessel_i0(beta)


def calc_plaqs_diffs(plaqs, beta):
    """Calculate the difference between expected and observed plaquettes."""
    return u1_plaq_exact(beta) - plaqs


def pbc(tup, shape):
    """Returns tup % shape for implementing periodic boundary conditions."""
    return list(np.mod(tup, shape))


def pbc_tf(tup, shape):
    """Tensorflow implementation of `pbc` defined above."""
    return list(tf.math.mod(tup, shape))


def mat_adj(mat):
    """Returns the adjoint (i.e. conjugate transpose) of a matrix `mat`."""
    return tf.transpose(tf.math.conj(mat))  # conjugate transpose


def project_angle(x):
    """Returns the projection of an angle `x` from [-4pi, 4pi] to [-pi, pi]."""
    return x - 2 * np.pi * tf.math.floor((x + np.pi) / (2 * np.pi))


def project_angle_np(x):
    """Returns the projection of an angle `x` from [-4pi, 4pi] to [-pi, pi]."""
    return x - 2 * np.pi * np.floor((x + np.pi) / (2 * np.pi))


def project_angle_fft(x, n=10):
    """Use the fourier series representation `x` to approx `project_angle`.
    NOTE: Because `project_angle` suffers a discontinuity, we approximate `x`
    with its Fourier series representation in order to have a differentiable
    function when computing the loss.
    Args:
        x (array-like): Array to be projected.
        N (int): Number of terms to keep in Fourier series.
    """
    y = np.zeros(x.shape, dtype=NP_FLOAT)
    for i in range(1, n):
        y += (-2 / i) * ((-1) ** i) * tf.sin(i * x)
    return y


def calc_plaqs_np(x):
    """Defines the potential energy function using the Wilson action."""
    potential = (x[..., 0]
                 - x[..., 1]
                 - np.roll(x[..., 0], shift=-1, axis=2)
                 + np.roll(x[..., 1], shift=-1, axis=1))
    return potential


def get_potential_fn(lattice_shape):
    """Wrapper method that reshapes `x` to `lattice_shape`."""
    def gauge_potential(x):
        """Defines the potential energy function using the Wilson action."""
        x = tf.reshape(x, lattice_shape)
        potential = (x[..., 0]
                     - x[..., 1]
                     - tf.roll(x[..., 0], shift=-1, axis=2)
                     + tf.roll(x[..., 1], shift=-1, axis=1))
        return potential

    return gauge_potential


def expand_samples(samples):
    """Reshape each sample in samples to the correct lattice shape.

    NOTE: In order to calculate the plaquette sums, we must first reshape
    samples from (N, D) --> (N, L, L, D), where:

        N = batch_size (number of unique lattices)
        L = space_size (extent of lattice in the spatial dimension)
        T = time_size (extent of lattice in the temporal dimension)
        D = dimensionality ( = 2 for current lattice implementation)

    Returns:
        samples (np.ndarray): Reshaped samples.

    Raises:
        ValueError if unable to correctly expand samples.
    """
    if len(samples.shape) == 2:
        bs = samples.shape[0]  # pylint:disable=invalid-name
        x_dim = samples.shape[1]
        size = np.sqrt(x_dim / 2)
        if size - int(size) < 1e-2:
            size = int(size)
            samples = samples.reshape((bs, size, size, 2))

        else:
            raise ValueError('Unable to correctly reshape `samples`.  Exiting')

    return samples


def plaq_sums(samples):
    """Calculate plaquette sums for a collection of lattices.

    Explicitly, calculate the sum of the link variables around each
    plaquette in the lattice for each sample in samples.


    Args:
        samples (np.ndarray): Array of shape (N, D) where N is the batch
            size (number of unique lattices) and D is the number of links on
            the lattice.

    Returns:
        plaq_sums (np.ndarray): The plaquette sums for each of the plaquettes
            on the lattice, for each of the lattices in `samples`.

    """
    if len(samples.shape) == 2:
        samples = expand_samples(samples)

    plaqs = (samples[:, :, :, 0]
             - samples[:, :, :, 1]
             - np.roll(samples[:, :, :, 0], shift=-1, axis=2)
             + np.roll(samples[:, :, :, 1], shift=-1, axis=1))

    return plaqs


def actions(samples):
    """Calculate the actions for a collection of lattices.

    Args:
        samples (np.ndarray): Array of shape (N, D) where N is the batch size
            (number of individual lattices), and D is the number of links on
            each lattice.

    Returns:
        actions: The total action calculated for each lattice in `samples`.
    """
    return np.sum(1. - np.cos(plaq_sums(samples)), axis=(1, 2))


def avg_plaqs(samples, ps=None):
    """Calculate the average plaquette value for each lattice in samples.

    Args:
        samples (np.ndarray): Array of shape (N, D) where N is the batch size
            (number of individual lattices), and D is the number of links on
            each lattice.
        ps (optional): Plaquette sums, if already calculated. If not passed,
            the plaquette sums will be calculated explicitly.

    Returns:
        plaqs_avg: The average plaquette calculated for each lattice in
            `samples`.
    """
    if plaq_sums is None:
        ps = plaq_sums(samples)

    return np.mean(np.cos(ps), axis=(1, 2, 3))


def top_charges(samples, ps=None):
    """Calculate the topological charges for each lattice in samples.

    Args:
        samples (np.ndarray): Array of shape (N, D) where N is the batch size
            (number of individual lattices), and D is the number of links on
            each lattice.
        ps (optional): Plaquette sums, if already calculated. If not passed,
            the plaquette sums will be calculated explicitly.

    Returns:
        tc: The topological charge calculated for each lattice in

            `samples`.
    """
    if ps is None:
        ps = plaq_sums(samples)

    return np.sin(ps)
