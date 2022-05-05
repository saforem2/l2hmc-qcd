"""
loss.py

Contains pytorch implementation of loss function for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from typing import Optional

import torch

from l2hmc.configs import LossConfig
from l2hmc.lattice.u1.pytorch.lattice import LatticeU1
from l2hmc.lattice.su3.pytorch.lattice import LatticeSU3

Tensor = torch.Tensor


class LatticeLoss:
    def __init__(
            self,
            lattice: LatticeU1 | LatticeSU3,
            loss_config: LossConfig
    ):
        self.lattice = lattice
        self.config = loss_config

    def __call__(self, x_init: Tensor, x_prop: Tensor, acc: Tensor) -> Tensor:
        return self.calc_loss(x_init, x_prop, acc)

    @staticmethod
    def mixed_loss(loss: Tensor, weight: float) -> Tensor:
        return (weight / loss) - (loss / weight)

    def _plaq_loss(self, w1: Tensor, w2: Tensor, acc: Tensor) -> Tensor:
        dw = w2 - w1
        dwloops = 2. * (torch.ones_like(w1) - torch.cos(dw))
        ploss = acc * dwloops.sum((1, 2)) + 1e-4

        # dwloops = 2. * (1. - torch.cos(w2 - w1))
        if self.config.use_mixed_loss:
            return self.mixed_loss(ploss, self.config.plaq_weight).mean(0)

        return (-ploss / self.config.plaq_weight).mean(0)

    def _charge_loss(self, w1: Tensor, w2: Tensor, acc: Tensor) -> Tensor:
        q1 = self.lattice._sin_charges(wloops=w1)  # type:ignore
        q2 = self.lattice._sin_charges(wloops=w2)  # type:ignore
        qloss = (acc * (q2 - q1) ** 2) + 1e-4
        if self.config.use_mixed_loss:
            return self.mixed_loss(qloss, self.config.charge_weight).mean(0)

        return (-qloss / self.config.charge_weight).mean(0)

    def lattice_metrics(
            self,
            xinit: Tensor,
            xout: Optional[Tensor] = None,
            # beta: Optional[float] = None,
    ) -> dict[str, Tensor]:
        metrics = self.lattice.calc_metrics(x=xinit)  # , beta=beta)
        if xout is not None:
            wloops = self.lattice.wilson_loops(x=xout)
            qint = self.lattice._int_charges(wloops=wloops)
            qsin = self.lattice._sin_charges(wloops=wloops)
            metrics.update({
                'dQint': (qint - metrics['intQ']).abs(),
                'dQsin': (qsin - metrics['sinQ']).abs(),
            })

        return metrics

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
