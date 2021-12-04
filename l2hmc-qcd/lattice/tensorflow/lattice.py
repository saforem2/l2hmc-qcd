"""
lattice.py

TensorFlow implementation of the Lattice object.
"""
from __future__ import absolute_import, division, print_function, annotations

import numpy as np
import tensorflow as tf
from dataclasses import dataclass, asdict

PI = tf.constant(np.pi)
TWO_PI = 2. * PI


@dataclass
class Charges:
    sinQ: tf.Tensor
    intQ: tf.Tensor


@dataclass
class LatticeMetrics:
    charges: Charges
    actions: tf.Tensor
    plaqs: tf.Tensor


def area_law(beta: float, num_plaqs: int) -> float:
    """Returns the expected value of the Wilson loop containing `num_plaqs`."""
    return (tf.math.bessel_i1(beta) / tf.math.bessel_i0(beta)) ** num_plaqs


def plaq_exact(beta: float) -> float:
    """Computes the expected value of the avg. plaquette for 2D U(1)."""
    return tf.math.bessel_i1(beta) / tf.math.bessel_i0(beta)


def project_angle(x):
    """Returns the projection of an angle `x` from [-4 π, 4 π] to [-π, π]."""
    return x - TWO_PI * tf.math.floor((x + PI) / TWO_PI)


Observables = dict[str, tf.Tensor]

class Lattice:
    def __init__(self, shape: tuple):
        self._shape = shape
        self.batch_size, self.x_shape = shape[0], shape[1:]
        self._nt, self._nx, self._dim = self.x_shape
        self.nplaqs = self._nt * self._nx
        self.nlinks = self.num_plaqs * self._dim

    def unnormalized_log_prob(self, x: tf.Tensor) -> tf.Tensor:
        return self.action(x)

    def observables(self, x: tf.Tensor) -> dict[str, tf.Tensor]:
        wloops = self.wilson_loops(x)
        charges = self.charges(wloops=wloops)
        plaqs = self.plaqs(wloops=wloops)
        plaqs4x4 = self.plaqs4x4(x=x)
        return {
            'plaqs': plaqs,
            'intQ': charges.int,
            'sinQ': charges.sin,
            'plaqs4x4': plaqs4x4,
        }

    def wilson_loops(self, x: tf.Tensor) -> tf.Tensor:
        x0, x1 = tf.reshape(x, (-1, *self.x_shape)).T
        return (x0 + tf.roll(x1, -1, 0) - tf.roll(x0, -1, 1) - x1).T

    def plaqs(
            self,
            x: tf.Tensor = None,
            wloops: tf.Tensor = None,
    ) -> tf.Tensor:
        """Calculate the avg plaq for each of the lattices in x."""
        if x is None:
            if wloops is None:
                raise ValueError('One of `x` or `wloops` must be specified')
            return tf.reduce_mean(tf.math.cos(wloops))
        return tf.reduce_mean(tf.math.cos(self.wilson_loops(x)), (1, 2))

    def _wilson_loops4x4(self, x: tf.Tensor = None) -> tf.Tensor:
        """Calculate the 4x4 Wilson loops"""
        x0, x1 = tf.reshape(x, (-1, *self.x_shape)).T
        return (x0                                      # U_x [x, y]
                + tf.roll(x0, -1, 2)                    # U_x [x+1, y]
                + tf.roll(x0, -2, 2)                    # U_x [x+2, y]
                + tf.roll(x0, -3, 2)                    # U_x [x+3, y]
                + tf.roll(x0, -4, 2)                    # U_x [x+4, y]
                + tf.roll(x1, (-4, -1), (2, 1))         # U_y [x+4, y]
                + tf.roll(x1, (-4, -2), (2, 1))         # U_y [x+4, y+1]
                + tf.roll(x1, (-4, -3), (2, 1))         # U_y [x+4, y+3]
                - tf.roll(x0, (-3, -4), (2, 1))         # U_x [x+3, y+4]
                - tf.roll(x1, (-3, -4), (2, 1))         # U_x [x+4, y+4]
                - tf.roll(x0, (-2, -4), (2, 1))         # U_x [x+2, y+4]
                - tf.roll(x0, (-1, -4), (2, 1))         # U_x [x+1, y+4]
                - tf.roll(x1, -4, 1)                    # U_y [x, y+4]
                - tf.roll(x1, -3, 1)                    # U_y [x, y+3]
                - tf.roll(x1, -2, 1)                    # U_y [x, y+2]
                - tf.roll(x1, -1, 1)                    # U_y [x, y+1]
                - x1)                                   # U_y [x, y]


    def _plaqs4x4(self, wloops4x4: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.math.cos(wloops4x4), (1, 2))

    def plaqs4x4(
        self,
        x: tf.Tensor = None,
        wloops4x4: tf.Tensor = None
    ) -> tf.Tensor:
        """Calculate 4x4 wilson loops."""
        if x is None:
            if wloops4x4 is None:
                raise ValueError('One of `x` or `wloops` must be specified')
            wloops4x4 = self._wilson_loops4x4(x)

        return tf.reduce_mean(tf.math.cos(wloops4x4), (1, 2))
 
    def _sin_charges(self, wloops: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(tf.math.sin(wloops), (1, 2)) / TWO_PI

    def _int_charges(self, wloops: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(project_angle(wloops), (1, 2)) / TWO_PI

    def sin_charges(
            self,
            x: tf.Tensor,
            wloops: tf.Tensor
    ) -> tf.Tensor:
        pass

    def int_charges(self,
                    x: tf.Tensor
                    wloops: tf.Tensor) -> tf.Tensor:
        """Calculate the integer valued charges."""
        if x is None:
            if wloops is None:
                raise ValueError(f'One of `x` or `wloops` must be specified')
            wloops = self.wils



    def action(self):
        pass
