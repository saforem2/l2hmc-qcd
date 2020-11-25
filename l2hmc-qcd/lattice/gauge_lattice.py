"""
GaugeLattice.py

Conatains an implementation of the GaugeLattice object.

Author: Sam Foreman (github: @saforem2)
Date: 07/24/2020
----------------------------------------------------------
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from utils.attr_dict import AttrDict


@dataclass
class Charges:
    sinQ: tf.Tensor
    intQ: tf.Tensor


def plaq_exact(beta):
    """Computes the expected value of the avg. plaquette for 2D U(1)."""
    return tf.math.bessel_i1(beta) / tf.math.bessel_i0(beta)


def project_angle(x):
    """Returns the projection of an angle `x` from [-4 π, 4 π] to [-π, π]."""
    return x - 2 * np.pi * tf.math.floor((x + np.pi) / (2 * np.pi))


# pylint:disable=too-many-instance-attributes
class GaugeLattice:
    """GaugeLattice object."""

    def __init__(self, shape: tuple):
        """
        Args:
            shape (tuple): Shape of batch of lattices.

        NOTE: shape = (batch_size, Lt, Lx, dim) = (B, T, X, D)
        """
        self._shape = shape
        self.batch_size, self.lattice_shape = shape[0], shape[1:]
        self._nt, self._nx, self._dim = self.lattice_shape
        self.num_plaqs = self._nt * self._nx
        self.num_links = self.num_plaqs * self._dim

    def calc_observables(self, x, beta=None):
        """Calculate all observables for a batch of lattices `x`."""
        wloops = self.calc_wilson_loops(x)
        actions = self.calc_actions(wloops=wloops)
        charges = self.calc_charges(wloops=wloops)
        plaqs = self.calc_plaqs(wloops=wloops, beta=beta)

        observables = {
            'plaqs': plaqs,
            'wloops': wloops,
            'actions': actions,
            'charges': charges,
        }

        return observables

    def calc_wilson_loops(self, x):
        """Calculate the plaquettes by summing the links in CCW direction."""
        x = tf.reshape(x, shape=self._shape)
        # NOTE: x.shape = (B, T, X, D) -> plaqs.shape = (B, T, L)
        wilson_loops = (x[..., 0]
                        - x[..., 1]
                        - tf.roll(x[..., 0], shift=-1, axis=2)
                        + tf.roll(x[..., 1], shift=-1, axis=1))

        return wilson_loops

    def calc_plaqs(self, x=None, wloops=None, beta=None):
        """Calculate the plaquettes for a batch of lattices."""
        if wloops is None:
            try:
                wloops = self.calc_wilson_loops(x)
            except ValueError as err:
                print('One of `x` or `wloops` must be specified.')
                raise err

        if beta is not None:
            return plaq_exact(beta) - tf.reduce_mean(tf.cos(wloops), (1, 2))

        return tf.reduce_mean(tf.cos(wloops), axis=(1, 2))

    def calc_actions(self, x=None, wloops=None):
        """Calculate the Wilson gauge action for a batch of lattices."""
        if wloops is None:
            try:
                wloops = self.calc_wilson_loops(x)
            except ValueError as err:
                print('One of `x` or `wloops` must be specified.')
                raise err

        return tf.reduce_sum(1. - tf.cos(wloops), axis=(1, 2), name='actions')

    def calc_charges(self, x=None, wloops=None, use_sin=False):
        """Calculate the topological charges for a batch of lattices."""
        if wloops is None:
            try:
                wloops = self.calc_wilson_loops(x)
            except ValueError as err:
                print('One of `x` or `wloops` must be specified.')
                raise err

        q = tf.sin(wloops) if use_sin else project_angle(wloops)

        return tf.reduce_sum(q, axis=(1, 2), name='charges') / (2 * np.pi)

    def calc_both_charges(self, x=None, wloops=None):
        """Calculate the charges using both integer and sin represntations."""
        if wloops is None:
            try:
                wloops = self.calc_wilson_loops(x)
            except ValueError as err:
                print('One of `x` or `wloops` must be specified.')
                raise err

        sinq = tf.reduce_sum(tf.sin(wloops), axis=(1, 2)) / (2 * np.pi)
        intq = tf.reduce_sum(project_angle(wloops), axis=(1, 2)) / (2 * np.pi)
        return Charges(sinQ=sinq, intQ=intq)


