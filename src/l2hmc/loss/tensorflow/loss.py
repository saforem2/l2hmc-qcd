"""
loss.py

Contains tensorflow implementation of loss function for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from typing import Optional

import tensorflow as tf

# import l2hmc.group.tensorflow.group as g
from l2hmc.group.u1.tensorflow.group import U1Phase
from l2hmc.group.su3.tensorflow.group import SU3

from l2hmc.configs import LossConfig
from l2hmc.lattice.u1.tensorflow.lattice import LatticeU1
from l2hmc.lattice.su3.tensorflow.lattice import LatticeSU3
# from l2hmc.lattice.tensorflow.lattice import Lattice

TF_FLOAT = tf.keras.backend.floatx()
Tensor = tf.Tensor


class LatticeLoss:
    def __init__(
            self,
            lattice: LatticeU1 | LatticeSU3,
            loss_config: LossConfig
    ):
        self.lattice = lattice
        self.config = loss_config
        self.plaq_weight = tf.constant(self.config.plaq_weight,
                                       dtype=TF_FLOAT)
        self.charge_weight = tf.constant(self.config.charge_weight,
                                         dtype=TF_FLOAT)
        if isinstance(self.lattice, LatticeU1):
            self.g = U1Phase()
        elif isinstance(self.lattice, LatticeSU3):
            self.g = SU3()
        else:
            raise ValueError(f'Unexpected value for `self.g`: {self.g}')

    def __call__(self, x_init: Tensor, x_prop: Tensor, acc: Tensor) -> Tensor:
        return self.calc_loss(x_init, x_prop, acc)

    @staticmethod
    def mixed_loss(loss: Tensor, weight: float) -> Tensor:
        w = tf.constant(weight, dtype=TF_FLOAT)
        return (w / loss) - (loss / w)

    def plaq_loss(self, x1: Tensor, x2: Tensor, acc: Tensor) -> Tensor:
        w1 = self.lattice.wilson_loops(x=x1)
        w2 = self.lattice.wilson_loops(x=x2)
        return self._plaq_loss(w1=w1, w2=w2, acc=acc)

    def charge_loss(self, x1: Tensor, x2: Tensor, acc: Tensor) -> Tensor:
        w1 = self.lattice.wilson_loops(x=x1)
        w2 = self.lattice.wilson_loops(x=x2)
        return self._charge_loss(w1=w1, w2=w2, acc=acc)

    def _plaq_loss(self, w1: Tensor, w2: Tensor, acc: Tensor) -> Tensor:
        dw = tf.subtract(w2, w1)
        dwloops = 2. * (tf.ones_like(w1) - tf.math.cos(dw))
        if isinstance(self.g, U1Phase):
            ploss = acc * tf.reduce_sum(dwloops, axis=(1, 2))
        elif isinstance(self.g, SU3):
            ploss = acc * tf.reduce_sum(
                dwloops, tuple(range(2, 3, len(w1.shape)))
            )
        else:
            raise ValueError(f'Unexpected value for self.g: {self.g}')

        if self.config.use_mixed_loss:
            ploss += 1e-4  # to prevent division by zero in mixed_loss
            tf.reduce_mean(self.mixed_loss(ploss, self.plaq_weight))

        return tf.reduce_mean(-ploss / self.plaq_weight)

    def _charge_loss(self, w1: Tensor, w2: Tensor, acc: Tensor) -> Tensor:
        dq2 = tf.math.square(tf.subtract(
            self.lattice._sin_charges(wloops=w2),
            self.lattice._sin_charges(wloops=w1),
        ))
        qloss = acc * dq2
        # qloss = (acc * (q2 - q1) ** 2)
        if self.config.use_mixed_loss:
            qloss += 1e-4
            return tf.reduce_mean(
                self.mixed_loss(qloss, self.charge_weight)
            )
        return tf.reduce_mean(-qloss / self.charge_weight)

    def lattice_metrics(
            self,
            xinit: Tensor,
            xout: Optional[Tensor] = None,
            # beta: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        metrics = self.lattice.calc_metrics(x=xinit)  # , beta=beta)
        if xout is not None:
            wloops = self.lattice.wilson_loops(x=xout)
            qint = self.lattice._int_charges(wloops=wloops)
            qsin = self.lattice._sin_charges(wloops=wloops)
            metrics.update({
                'dQint': tf.math.abs(tf.subtract(qint, metrics['intQ'])),
                'dQsin': tf.math.abs(tf.subtract(qsin, metrics['sinQ']))
            })

        return metrics

    def calc_loss(self, x_init: Tensor, x_prop: Tensor, acc: Tensor) -> Tensor:
        wl_init = self.lattice.wilson_loops(x=x_init)
        wl_prop = self.lattice.wilson_loops(x=x_prop)

        plaq_loss = tf.constant(0., dtype=TF_FLOAT)
        if self.plaq_weight > 0:
            plaq_loss = self._plaq_loss(w1=wl_init, w2=wl_prop, acc=acc)

        charge_loss = tf.constant(0., dtype=TF_FLOAT)
        if self.charge_weight > 0:
            charge_loss = self._charge_loss(w1=wl_init, w2=wl_prop, acc=acc)

        return tf.add(charge_loss, plaq_loss)
