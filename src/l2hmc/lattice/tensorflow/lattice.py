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

Tensor = tf.Tensor


@dataclass
class Charges:
    intQ: Tensor
    sinQ: Tensor

    def asdict(self):
        return asdict(self)


@dataclass
class LatticeMetrics:
    plaqs: Tensor
    charges: Charges
    p4x4: Tensor

    def asdict(self):
        return {
            'plaqs': self.plaqs,
            'sinQ': self.charges.sinQ,
            'intQ': self.charges.intQ,
            'p4x4': self.p4x4,
        }


def area_law(beta: float, num_plaqs: int) -> float:
    """Returns the expected value of the Wilson loop containing `num_plaqs`."""
    return (tf.math.bessel_i1(beta) / tf.math.bessel_i0(beta)) ** num_plaqs


def plaq_exact(beta: float) -> float:
    """Computes the expected value of the avg. plaquette for 2D U(1)."""
    return tf.math.bessel_i1(beta) / tf.math.bessel_i0(beta)


def project_angle(x):
    """Returns the projection of an angle `x` from [-4 π, 4 π] to [-π, π]."""
    return x - TWO_PI * tf.math.floor((x + PI) / TWO_PI)


Observables = dict[str, Tensor]


class Lattice:
    def __init__(self, shape: tuple):
        self._shape = shape
        self.batch_size, self.x_shape = shape[0], shape[1:]
        self.nt, self.nx, self._dim = self.x_shape
        self.nplaqs = self.nt * self.nx
        self.nlinks = self.nplaqs * self._dim

    def draw_uniform_batch(self) -> Tensor:
        """Draw batch of samples, uniformly from [-pi, pi)."""
        return tf.random.uniform(self._shape, *(-PI, PI))

    def unnormalized_log_prob(self, x: Tensor) -> Tensor:
        return self.action(x)

    def calc_metrics(self, x: Tensor) -> Observables:
        wloops = self.wilson_loops(x)
        plaqs = self.plaqs(wloops=wloops)
        charges = self.charges(wloops=wloops)
        return {
            'plaqs': plaqs,
            'intQ': charges.intQ,
            'sinQ': charges.sinQ,
        }

    def observables(self, x: Tensor) -> LatticeMetrics:
        """Calculate Lattice observables."""
        wloops = self.wilson_loops(x)
        return LatticeMetrics(p4x4=self.plaqs4x4(x=x),
                              plaqs=self.plaqs(wloops=wloops),
                              charges=self.charges(wloops=wloops))

    def wilson_loops(self, x: Tensor) -> Tensor:
        """Calculate the Wilson loops by summing links in CCW direction."""
        # --------------------------
        # NOTE: Watch your shapes!
        # --------------------------
        # * First, x.shape = [-1, Lt, Lx, 2], so
        #       (x_reshaped).T.shape = [2, Lx, Lt, -1]
        #   and,
        #       x0.shape = x1.shape = [Lx, Lt, -1]
        #   where x0 and x1 are the links along the 2 (t, x) dimensions.
        #
        # * The Wilson loop is then:
        #       wloop = U0(x, y) +  U1(x+1, y) - U0(x, y+1) - U(1)(x, y)
        #   and so output = wloop.T, with output.shape = [-1, Lt, Lx]
        # --------------------------
        x = tf.reshape(x, shape=(-1, *self.x_shape))
        wilson_loops = (x[..., 0]
                        - x[..., 1]
                        - tf.roll(x[..., 0], -1, axis=2)
                        + tf.roll(x[..., 1], -1, axis=1))
        return wilson_loops
        # x0, x1 = tf.transpose(tf.reshape(x, (-1, *self.x_shape)))
        # return tf.transpose(x0 + tf.roll(x1, -1, 0) - tf.roll(x0, -1, 1) - x1)
        # ---
        # x0, x1 = tf.reshape(x, (-1, *self.x_shape)).T
        # return tf.transpose(x[0] + tf.roll(x[1], -1, 0) - tf.roll(x[0], -1, 1) - x[1])

    def wilson_loops4x4(self, x: Tensor) -> Tensor:
        """Calculate the 4x4 Wilson loops"""
        x0, x1 = tf.transpose(tf.reshape(x, (-1, *self.x_shape)))
        return tf.transpose(
            x0                                      # Ux [x, y]
            + tf.roll(x0, -1, 2)                    # Ux [x+1, y]
            + tf.roll(x0, -2, 2)                    # Ux [x+2, y]
            + tf.roll(x0, -3, 2)                    # Ux [x+3, y]
            + tf.roll(x0, -4, 2)                    # Ux [x+4, y]
            + tf.roll(x1, (-4, -1), (2, 1))         # Uy [x+4, y]
            + tf.roll(x1, (-4, -2), (2, 1))         # Uy [x+4, y+1]
            + tf.roll(x1, (-4, -3), (2, 1))         # Uy [x+4, y+3]
            - tf.roll(x0, (-3, -4), (2, 1))         # -Ux [x+3, y+4]
            - tf.roll(x0, (-2, -4), (2, 1))         # -Ux [x+2, y+4]
            - tf.roll(x0, (-1, -4), (2, 1))         # -Ux [x+1, y+4]
            - tf.roll(x1, -4, 1)                    # -Uy [x, y+4]
            - tf.roll(x1, -3, 1)                    # -Uy [x, y+3]
            - tf.roll(x1, -2, 1)                    # -Uy [x, y+2]
            - tf.roll(x1, -1, 1)                    # -Uy [x, y+1]
            - x1                                    # -Uy [x, y]
        )

    def plaqs(
            self,
            x: Tensor = None,
            wloops: Tensor = None,
    ) -> Tensor:
        """Calculate the avg plaq for each of the lattices in x."""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified')
            wloops = self.wilson_loops(x)
        return tf.reduce_mean(tf.math.cos(wloops), (1, 2))

    def _plaqs4x4(self, wloops4x4: Tensor) -> Tensor:
        return tf.reduce_mean(tf.math.cos(wloops4x4), (1, 2))

    def plaqs4x4(
            self,
            x: Tensor = None,
            wloops4x4: Tensor = None
    ) -> Tensor:
        """Calculate 4x4 wilson loops."""
        if wloops4x4 is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified')
            wloops4x4 = self.wilson_loops4x4(x)

        return self._plaqs4x4(wloops4x4)

    def _sin_charges(self, wloops: Tensor) -> Tensor:
        """Calculate sinQ from Wilson loops."""
        return tf.reduce_sum(tf.math.sin(wloops), (1, 2)) / TWO_PI

    def _int_charges(self, wloops: Tensor) -> Tensor:
        return tf.reduce_sum(project_angle(wloops), (1, 2)) / TWO_PI

    def sin_charges(
            self,
            x: Tensor = None,
            wloops: Tensor = None
    ) -> Tensor:
        """Calculate the real-valued charge approximation, sin(Q)"""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified')
            wloops = self.wilson_loops(x)

        return self._sin_charges(wloops)

    def int_charges(
            self,
            x: Tensor = None,
            wloops: Tensor = None
    ) -> Tensor:
        """Calculate the integer valued charges."""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified')
            wloops = self.wilson_loops(x)

        return self._int_charges(wloops)

    def charges(
            self,
            x: Tensor = None,
            wloops: Tensor = None,
    ) -> Charges:
        """Calculate both charge representations and return as single object"""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops = self.wilson_loops(x)

        sinQ = self._sin_charges(wloops)
        intQ = self._int_charges(wloops)
        return Charges(intQ=intQ, sinQ=sinQ)

    def action(
            self,
            x: Tensor = None,
            wloops: Tensor = None,
    ) -> Tensor:
        """Calculate the Wilson gauge action for a batch of lattices."""
        if wloops is None:
            if x is None:
                raise ValueError('One of `x` or `wloops` must be specified.')
            wloops = self.wilson_loops(x)

        return tf.reduce_sum(tf.constant(1.) - tf.math.cos(wloops), (1, 2))
