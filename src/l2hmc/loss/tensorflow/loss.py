"""
loss.py

Contains tensorflow implementation of loss function for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function

import tensorflow as tf
import numpy as np

from src.l2hmc.configs import LossConfig
from src.l2hmc.lattice.tensorflow.lattice import Lattice

TF_FLOAT = tf.keras.backend.floatx()
Tensor = tf.Tensor


class LatticeLoss:
    def __init__(self, lattice: Lattice, loss_config: LossConfig):
        super().__init__()
        self.lattice = lattice
        self.config = loss_config

    def __call__(self, x_init: Tensor, x_prop: Tensor, acc: Tensor) -> Tensor:
        wl_init = self.lattice.wilson_loops(x=x_init)
        wl_prop = self.lattice.wilson_loops(x=x_prop)

        plaq_loss = tf.constant(0., dtype=TF_FLOAT)
        if self.config.plaq_weight > 0:
            plaq_loss = self._plaq_loss(w1=wl_init, w2=wl_prop, acc=acc)

        charge_loss = tf.constant(0., dtype=TF_FLOAT)
        if self.config.charge_weight > 0:
            charge_loss = self._charge_loss(w1=wl_init, w2=wl_prop, acc=acc)

        return tf.add(charge_loss, plaq_loss)

    @staticmethod
    def mixed_loss(loss: Tensor, weight: float) -> Tensor:
        return tf.reduce_mean((weight / loss) - (loss / weight))

    def _plaq_loss(self, w1: Tensor, w2: Tensor, acc: Tensor) -> Tensor:
        # dw = tf.subtract(w2, w1)
        dwloops = 2. * (tf.ones_like(w1) - tf.math.cos(w2 - w1))
        # dwloops = tf.scalar_mul(2., tf.subtract(1., tf.math.cos(dw)))
        # ploss = tf.add(acc * tf.reduce_sum(dwloops, axis=(1, 2)), 1e-4)
        ploss = acc * tf.reduce_sum(dwloops, axis=(1, 2)) + 1e-4
        if self.config.use_mixed_loss:
            return self.mixed_loss(ploss, self.config.plaq_weight)
        return tf.reduce_mean(-ploss / self.config.plaq_weight, axis=0)

    def _charge_loss(self, w1: Tensor, w2: Tensor, acc: Tensor) -> Tensor:
        q1 = tf.reduce_sum(tf.sin(w1), axis=(1, 2)) / (2 * np.pi)
        q2 = tf.reduce_sum(tf.sin(w2), axis=(1, 2)) / (2 * np.pi)
        qloss = (acc * (q2 - q1) ** 2) + 1e-4
        if self.config.use_mixed_loss:
            return self.mixed_loss(qloss, self.config.charge_weight)
        return tf.reduce_mean(-qloss / self.config.charge_weight, axis=0)

    def call(self, x_init: Tensor, x_prop: Tensor, acc: Tensor) -> Tensor:
        wl_init = self.lattice.wilson_loops(x=x_init)
        wl_prop = self.lattice.wilson_loops(x=x_prop)

        plaq_loss = tf.constant(0., dtype=TF_FLOAT)
        if self.config.plaq_weight > 0:
            plaq_loss = self._plaq_loss(w1=wl_init, w2=wl_prop, acc=acc)

        charge_loss = tf.constant(0., dtype=TF_FLOAT)
        if self.config.charge_weight > 0:
            charge_loss = self._charge_loss(w1=wl_init, w2=wl_prop, acc=acc)

        return tf.reduce_sum([plaq_loss, charge_loss])
