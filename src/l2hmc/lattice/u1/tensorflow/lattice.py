"""
lattice.py

TensorFlow implementation of the Lattice object.
"""
from __future__ import absolute_import, annotations, division, print_function
# from dataclasses import asdict, dataclass

import numpy as np
import tensorflow as tf
from typing import Optional

# from l2hmc.lattice.u1.numpy.lattice import BaseLatticeU1
from l2hmc.lattice.lattice import Lattice
import l2hmc.group.u1.tensorflow.group as g
from l2hmc.configs import Charges, LatticeMetrics

TF_FLOAT = tf.dtypes.as_dtype(tf.keras.backend.floatx())
PI = tf.constant(np.pi, dtype=TF_FLOAT)
TWO_PI = tf.constant(2. * PI, dtype=TF_FLOAT)

Tensor = tf.Tensor


def area_law(beta: float, num_plaqs: int) -> float:
    """Returns the expected value of the Wilson loop containing `num_plaqs`."""
    return (tf.math.bessel_i1(beta) / tf.math.bessel_i0(beta)) ** num_plaqs


def plaq_exact(beta: float | Tensor) -> Tensor:
    """Computes the expected value of the avg. plaquette for 2D U(1)."""
    if isinstance(beta, float):
        beta = tf.constant(beta, dtype=TF_FLOAT)
    pexact = tf.constant(tf.math.bessel_i1(beta) / tf.math.bessel_i0(beta))
    return pexact


def project_angle(x):
    """Returns the projection of an angle `x` from [-4 π, 4 π] to [-π, π]."""
    return x - TWO_PI * tf.math.floor((x + PI) / TWO_PI)


class LatticeU1(Lattice):
    def __init__(self, nchains: int, shape: list[int]):
        assert len(shape) == 2
        self.g = g.U1Phase()
        self.nt, self.nx = shape
        self.volume = self.nt * self.nx
        self.nplaqs = self.nt * self.nx
        super().__init__(group=self.g, nchains=nchains, shape=shape)

        # self.nt, self.nx, = shape
        # self._shape = (nchains, self.dim, *shape, self.g.shape)
        # self.volume = self.nt * self.nx
        # self.nplaqs = self.nt * self.nx
        # self._lattice_shape = shape
        # self.nsites = np.cumprod(shape)[-1]
        # self.nlinks = self.nsites * self.dim
        # self.site_idxs = tuple(
        #     [self.nt] + [self.nx for _ in range(self.dim - 1)]
        # )
        # self.link_idxs = tuple(list(self.site_idxs) + [self.dim])

    def draw_uniform_batch(self) -> Tensor:
        """Draw batch of samples, uniformly from (-pi, pi)."""
        return tf.random.uniform(self._shape, *(-PI, PI), dtype=TF_FLOAT)

    def kinetic_energy(self, v: Tensor) -> Tensor:
        return 0.5 * tf.reshape(v, (v.shape[0], -1)) ** 2

    def unnormalized_log_prob(
            self,
            x: Tensor,
            beta: Optional[Tensor] = None,
    ) -> Tensor:
        beta = tf.constant(1.) if beta is None else beta
        assert isinstance(beta, Tensor)
        return self.action(x, beta)

    def action(self, x: Tensor, beta: Tensor) -> Tensor:
        """Calculate the Wilson gauge action for a batch of lattices."""
        wloops = self._get_wloops(x)
        return self._action(wloops, beta)

    def _action(self, wloops: Tensor, beta: Tensor) -> Tensor:
        local_action = tf.ones_like(wloops) - tf.math.cos(wloops)
        return (
            tf.cast(beta, wloops.dtype) * tf.reduce_sum(local_action, (1, 2))
        )

    def action_with_grad(
            self,
            x: Tensor,
            beta: Tensor
    ) -> tuple[Tensor, Tensor]:
        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                tape.watch(x)
                s = self.action(x, beta)
            dsdx = tape.gradient(s, x)
        else:
            s = self.action(x, beta)
            dsdx = tf.gradients(s, [x])[0]

        return s, dsdx

    def grad_action(self, x: Tensor, beta: Tensor) -> Tensor:
        """Compute the gradient of the potential function."""
        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                tape.watch(x)
                pe = self.action(x, beta)
            grad = tape.gradient(pe, x)
        else:
            grad = tf.gradients(self.action(x, beta), [x])[0]

        return grad

    def calc_metrics(
            self,
            x: Tensor,
            xinit: Optional[Tensor] = None,
            beta: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        wloops = self.wilson_loops(x)
        plaqs = self.plaqs(wloops=wloops)
        charges = self.charges(wloops=wloops)
        metrics = {
            'plaqs': plaqs,
            'intQ': charges.intQ,
            'sinQ': charges.sinQ
        }

        if beta is not None:
            pexact = plaq_exact(beta) * tf.ones_like(plaqs)
            metrics.update({
               'plaqs_err': pexact - plaqs,
               'action': self.action(x, beta),
            })

        if xinit is not None:
            wloops_ = self.wilson_loops(xinit)
            plaqs_ = self.plaqs(wloops=wloops_)
            charges_ = self.charges(wloops=wloops_)
            metrics.update({
                'dplaqs': tf.abs(tf.subtract(plaqs, plaqs_)),
                'dQint': tf.abs(charges.intQ - charges_.intQ),
                'dQsin': tf.abs(charges.sinQ - charges_.sinQ),
            })
            # if beta is not None:
            #     action_ = self.action(xinit, beta)
            #     metrics.update({
            #         'daction': tf.abs(metrics['action'] - action_),
            #     })

        return metrics

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
        # * First, x.shape = [-1, 2, Lt, Lx], so
        #       (x_reshaped).T.shape = [2, Lx, Lt, -1]
        #   and,
        #       x0.shape = x1.shape = [Lx, Lt, -1]
        #   where x0 and x1 are the links along the 2 (t, x) dimensions.
        #
        # * The Wilson loop is then:
        #       wloop = U0(x, y) +  U1(x+1, y) - U0(x, y+1) - U1(x, y)
        #   and so output = wloop.T, with output.shape = [-1, Lt, Lx]
        # --------------------------
        # x = tf.transpose(tf.reshape(x, (-1, *self.xshape)), (1, 2, 3, 0))
        x = tf.reshape(x, (-1, *self.xshape))
        # x = tf.transpose(tf.reshape(x, self._shape), (1, 2, 3, 0))
        xu = x[:, 0]  # type:ignore  NOTE: x0 = t links
        xv = x[:, 1]  # type:ignore  NOTE: x1 = x links
        # return tf.transpose(
        #     xu + tf.roll(xv, -1, axis=0) - tf.roll(xu, -1, axis=1) - xv
        # )
        return xu + tf.roll(xv, -1, axis=1) - tf.roll(xu, -1, axis=2) - xv

    def wilson_loops4x4(self, x: Tensor) -> Tensor:
        """Calculate the 4x4 Wilson loops"""
        x = tf.transpose(tf.reshape(x, (-1, *self.xshape)), (1, 2, 3, 0))
        x0 = x[0]  # type:ignore
        x1 = x[1]  # type:ignore
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

    def plaqs_diff(
            self,
            beta: float,
            x: Optional[Tensor] = None,
            wloops: Optional[Tensor] = None,
    ) -> Tensor:
        wloops = self._get_wloops(x) if wloops is None else wloops
        plaqs = self.plaqs(wloops=wloops)
        pexact = plaq_exact(beta) * tf.ones_like(plaqs)
        return pexact - self.plaqs(wloops=wloops)

    def plaqs(
            self,
            x: Optional[Tensor] = None,
            wloops: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate the avg plaq for each of the lattices in x."""
        wloops = self._get_wloops(x) if wloops is None else wloops
        return tf.reduce_mean(tf.math.cos(wloops), (1, 2))

    def _plaqs(self, wloops: Tensor) -> Tensor:
        return tf.reduce_mean(tf.math.cos(wloops), (1, 2))

    def _plaqs4x4(self, wloops4x4: Tensor) -> Tensor:
        return tf.reduce_mean(tf.math.cos(wloops4x4), (1, 2))

    def plaqs4x4(
            self,
            x: Optional[Tensor] = None,
            wloops4x4: Optional[Tensor] = None
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

    def _get_wloops(self, x: Optional[Tensor] = None) -> Tensor:
        if x is None:
            raise ValueError('Expected input `x`')
        return self.wilson_loops(x)

    def random(self):
        return self.g.random(list(self._shape))

    def random_momentum(self) -> Tensor:
        return self.g.random_momentum(list(self._shape))

    def sin_charges(
            self,
            x: Optional[Tensor] = None,
            wloops: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate the real-valued charge approximation, sin(Q)"""
        wloops = self._get_wloops(x) if wloops is None else wloops
        return self._sin_charges(wloops)

    def int_charges(
            self,
            x: Optional[Tensor] = None,
            wloops: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate the integer valued charges."""
        wloops = self._get_wloops(x) if wloops is None else wloops
        return self._int_charges(wloops)

    def charges(
            self,
            x: Optional[Tensor] = None,
            wloops: Optional[Tensor] = None
    ) -> Charges:
        """Calculate both charge representations and return as single object"""
        wloops = self._get_wloops(x) if wloops is None else wloops
        sinq = self._sin_charges(wloops)
        intq = self._int_charges(wloops)
        return Charges(intQ=intq, sinQ=sinq)

    def _charges(self, wloops: Tensor) -> Charges:
        sinQ = self._sin_charges(wloops)
        intQ = self._int_charges(wloops)
        return Charges(intQ=intQ, sinQ=sinQ)

    def plaq_loss(
            self,
            acc: Tensor,
            x1: Optional[Tensor] = None,
            x2: Optional[Tensor] = None,
            wl1: Optional[Tensor] = None,
            wl2: Optional[Tensor] = None,
    ) -> float:
        wloops1 = self._get_wloops(x1) if wl1 is None else wl1
        wloops2 = self._get_wloops(x2) if wl2 is None else wl2
        dwl = tf.subtract(wloops2, wloops1)
        dwloops = 2. * (tf.ones_like(wl1) - tf.math.cos(dwl))
        ploss = acc * tf.reduce_sum(dwloops, axis=(1, 2)) + 1e-4

        return tf.reduce_mean(-ploss, axis=0)

    def charge_loss(
            self,
            acc: Tensor,
            x1: Optional[Tensor] = None,
            x2: Optional[Tensor] = None,
            wl1: Optional[Tensor] = None,
            wl2: Optional[Tensor] = None,
    ) -> float:
        wloops1 = self._get_wloops(x1) if wl1 is None else wl1
        wloops2 = self._get_wloops(x2) if wl2 is None else wl2
        q1 = self._sin_charges(wloops=wloops1)
        q2 = self._sin_charges(wloops=wloops2)
        qloss = (acc * tf.math.subtract(q2, q1) ** 2) + 1e-4
        return tf.reduce_mean(-qloss, axis=0)


if __name__ == '__main__':
    lattice = LatticeU1(3, [8, 8])
    beta = tf.constant(1.0)
    x = lattice.random()
    v = lattice.random_momentum()
    action = lattice.action(x, beta)
    kinetic = lattice.kinetic_energy(v)
