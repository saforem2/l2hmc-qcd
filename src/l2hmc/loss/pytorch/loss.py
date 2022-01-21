"""
loss.py

Contains pytorch implementation of loss function for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function

import torch

from src.l2hmc.configs import LossConfig
from src.l2hmc.lattice.pytorch.lattice import Lattice

Tensor = torch.Tensor


class LatticeLoss:
    def __init__(self, lattice: Lattice, loss_config: LossConfig):
        self.lattice = lattice
        self.config = loss_config

    def __call__(self, x_init: Tensor, x_prop: Tensor, acc: Tensor) -> Tensor:
        return self.calc_loss(x_init, x_prop, acc)

    @staticmethod
    def mixed_loss(loss: Tensor, weight: float) -> Tensor:
        return (weight / loss) - (loss / weight)

    def _plaq_loss(self, w1: Tensor, w2: Tensor, acc: Tensor) -> Tensor:
        dwloops = 2. * (1. - torch.cos(w2 - w1))
        ploss = acc * dwloops.sum((1, 2)) + 1e-4
        if self.config.use_mixed_loss:
            loss = self.mixed_loss(ploss, self.config.plaq_weight)
        else:
            loss = (-ploss / self.config.plaq_weight)

        return loss.mean(dim=0)

    def _charge_loss(self, w1: Tensor, w2: Tensor, acc: Tensor) -> Tensor:
        q1 = self.lattice._sin_charges(wloops=w1)
        q2 = self.lattice._sin_charges(wloops=w2)
        qloss = (acc * (q2 - q1) ** 2) + 1e-4
        if self.config.use_mixed_loss:
            loss = self.mixed_loss(qloss, self.config.charge_weight)
        else:
            loss = -qloss / self.config.charge_weight

        return loss.mean(dim=0)

    def calc_loss(self, x_init: Tensor, x_prop: Tensor, acc: Tensor) -> Tensor:
        wl_init = self.lattice.wilson_loops(x=x_init)
        wl_prop = self.lattice.wilson_loops(x=x_prop)

        plaq_loss = torch.tensor(0.)
        if self.config.plaq_weight > 0:
            plaq_loss = self._plaq_loss(w1=wl_init, w2=wl_prop, acc=acc)

        charge_loss = torch.tensor(0.)
        if self.config.charge_weight > 0:
            charge_loss = self._charge_loss(w1=wl_init, w2=wl_prop, acc=acc)

        return plaq_loss + charge_loss
