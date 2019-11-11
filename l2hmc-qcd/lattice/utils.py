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

import scipy
import numpy as np

from config import NP_FLOAT


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
        bs = samples.shape[0]
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

    plaq_sums = (samples[:, :, :, 0]
                 - samples[:, :, :, 1]
                 - np.roll(samples[:, :, :, 0], shift=-1, axis=2)
                 + np.roll(samples[:, :, :, 1], shift=-1, axis=1))

    return plaq_sums


def actions(samples):
    """Calculate the actions for a collection of lattices.

    Args:
        samples (np.ndarray): Array of shape (N, D) where N is the batch size
            (number of individual lattices), and D is the number of links on
            each lattice.

    Returns:
        actions: The total action calculated for each lattice in `samples`.
    """
    ps = plaq_sums(samples)

    actions = np.sum(1. - np.cos(ps), axis=(1, 2))

    return actions


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

    plaqs_avg = np.mean(np.cos(ps), axis=(1, 2, 3))

    return plaqs_avg


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

    tc = np.sin(ps)

    return tc
