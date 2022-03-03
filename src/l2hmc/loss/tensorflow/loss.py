"""
loss.py

Contains tensorflow implementation of loss function for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function

import tensorflow as tf

from l2hmc.configs import LossConfig
from l2hmc.lattice.tensorflow.lattice import Lattice

TF_FLOAT = tf.keras.backend.floatx()
Tensor = tf.Tensor


class LatticeLoss:
    def __init__(self, lattice: Lattice, loss_config: LossConfig):
        self.lattice = lattice
        self.config = loss_config
        self.plaq_weight = tf.constant(self.config.plaq_weight,
                                       dtype=TF_FLOAT)
        self.charge_weight = tf.constant(self.config.charge_weight,
                                         dtype=TF_FLOAT)

    def __call__(self, x_init: Tensor, x_prop: Tensor, acc: Tensor) -> Tensor:
        return self.calc_loss(x_init, x_prop, acc)

    @staticmethod
    def mixed_loss(loss: Tensor, weight: float) -> Tensor:
        w = tf.constant(weight, dtype=TF_FLOAT)
        return (w / loss) - (loss / w)

    def _plaq_loss(self, w1: Tensor, w2: Tensor, acc: Tensor) -> Tensor:
        dw = tf.subtract(w2, w1)
        dwloops = 2. * (tf.ones_like(w1) - tf.math.cos(dw))
        ploss = acc * tf.reduce_sum(dwloops, axis=(1, 2)) + 1e-4
        if self.config.use_mixed_loss:
            tf.reduce_mean(self.mixed_loss(ploss, self.plaq_weight), axis=0)

        return tf.reduce_mean(-ploss / self.plaq_weight, axis=0)

    def _charge_loss(self, w1: Tensor, w2: Tensor, acc: Tensor) -> Tensor:
        q1 = self.lattice._sin_charges(wloops=w1)
        q2 = self.lattice._sin_charges(wloops=w2)
        qloss = (acc * (q2 - q1) ** 2) + 1e-4  # type: ignore
        if self.config.use_mixed_loss:
            return tf.reduce_mean(
                self.mixed_loss(qloss, self.charge_weight), axis=0
            )
        return tf.reduce_mean(-qloss / self.charge_weight, axis=0)

    def lattice_metrics(
            self,
            xinit: Tensor,
            xout: Tensor = None,
    ) -> dict[str, Tensor]:
        metrics = self.lattice.calc_metrics(x=xinit)
        if xout is not None:
            wl_out = self.lattice.wilson_loops(x=xout)
            qint_out = self.lattice._int_charges(wloops=wl_out)
            qsin_out = self.lattice._sin_charges(wloops=wl_out)
            metrics.update({
                'dQint': tf.math.abs(tf.subtract(qint_out, metrics['intQ'])),
                'dQsin': tf.math.abs(tf.subtract(qsin_out, metrics['sinQ']))
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
