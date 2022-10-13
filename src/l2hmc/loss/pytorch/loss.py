"""
loss.py

Contains pytorch implementation of loss function for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from typing import Optional

import torch


from l2hmc.configs import LossConfig
from l2hmc.group.u1.pytorch.group import U1Phase
from l2hmc.group.su3.pytorch.group import SU3
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
        # self.xshape = self.lattice._shape
        self.xshape = self.lattice.xshape
        self.plaq_weight = torch.tensor(self.config.plaq_weight,
                                        dtype=torch.float)
        self.charge_weight = torch.tensor(self.config.charge_weight,
                                          dtype=torch.float)
        if isinstance(self.lattice, LatticeU1):
            self.g = U1Phase()
        elif isinstance(self.lattice, LatticeSU3):
            self.g = SU3()
        else:
            raise ValueError(f'Unexpected value for `self.g`: {self.g}')

    def __call__(self, x_init: Tensor, x_prop: Tensor, acc: Tensor) -> Tensor:
        return self.calc_loss(x_init=x_init, x_prop=x_prop, acc=acc)

    @staticmethod
    def mixed_loss(loss: Tensor, weight: Tensor) -> Tensor:
        return (weight / loss) - (loss / weight)

    def _plaq_loss(
            self,
            w1: Tensor,
            w2: Tensor,
            acc: Tensor,
            use_mixed_loss: Optional[bool] = None,
    ) -> Tensor:
        dw = w2 - w1
        dwloops = 2. * (torch.ones_like(w1) - dw.cos())
        if isinstance(self.g, U1Phase):
            ploss = acc * dwloops.sum((1, 2))
        elif isinstance(self.g, SU3):
            ploss = acc * dwloops.sum((tuple(range(2, len(w1.shape)))))
        else:
            raise ValueError(f'Unexpected value for `self.g`: {self.g}')

        # dwloops = 2. * (1. - torch.cos(w2 - w1))
        use_mixed = (
            self.config.use_mixed_loss
            if use_mixed_loss is None else use_mixed_loss
        )
        if use_mixed:
            ploss += 1e-4
            return self.mixed_loss(ploss, self.plaq_weight).mean()

        return (-ploss / self.plaq_weight).mean()

    def _charge_loss(
            self,
            w1: Tensor,
            w2: Tensor,
            acc: Tensor,
            use_mixed_loss: Optional[bool] = None,
    ) -> Tensor:
        q1 = self.lattice._sin_charges(wloops=w1)  # type:ignore
        q2 = self.lattice._sin_charges(wloops=w2)  # type:ignore
        dqsq = (q2 - q1) ** 2
        qloss = acc * dqsq

        use_mixed = (
            self.config.use_mixed_loss
            if use_mixed_loss is None else use_mixed_loss
        )
        if use_mixed:
            qloss += 1e-4
            return self.mixed_loss(qloss, self.charge_weight).mean()

        return (-qloss / self.charge_weight).mean()

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

    def plaq_loss(
            self,
            x_init: Tensor,
            x_prop: Tensor,
            acc: Tensor,
            use_mixed_loss: Optional[bool] = None
    ) -> Tensor:
        wl_init = self.lattice.wilson_loops(x=x_init)
        wl_prop = self.lattice.wilson_loops(x=x_prop)
        return self._plaq_loss(
            w1=wl_init,
            w2=wl_prop,
            acc=acc,
            use_mixed_loss=use_mixed_loss
        )

    def charge_loss(
            self,
            x_init: Tensor,
            x_prop: Tensor,
            acc: Tensor,
            use_mixed_loss: Optional[bool] = None
    ) -> Tensor:
        wl_init = self.lattice.wilson_loops(x=x_init)
        wl_prop = self.lattice.wilson_loops(x=x_prop)
        return self._charge_loss(
            w1=wl_init,
            w2=wl_prop,
            acc=acc,
            use_mixed_loss=use_mixed_loss
        )

    def general_loss(
            self,
            x_init: Tensor,
            x_prop: Tensor,
            acc: Tensor,
            plaq_weight: Optional[float] = None,
            charge_weight: Optional[float] = None,
            use_mixed_loss: Optional[bool] = None
    ) -> Tensor | float:
        wl_init = self.lattice.wilson_loops(x=x_init)
        wl_prop = self.lattice.wilson_loops(x=x_prop)
        pw = self.plaq_weight if plaq_weight is None else plaq_weight
        qw = self.charge_weight if charge_weight is None else charge_weight
        ploss = 0.0
        qloss = 0.0
        loss = 0.0
        # loss = torch.tensor(0.)
        if pw > 0:
            ploss = self._plaq_loss(w1=wl_init, w2=wl_prop, acc=acc,
                                    use_mixed_loss=use_mixed_loss)
            loss += pw * ploss
        if qw > 0:
            qloss = self._charge_loss(w1=wl_init, w2=wl_prop, acc=acc,
                                      use_mixed_loss=use_mixed_loss)
            loss += qw * qloss

        return loss

    def calc_loss(self, x_init: Tensor, x_prop: Tensor, acc: Tensor) -> Tensor:
        wl_init = self.lattice.wilson_loops(x=x_init)
        wl_prop = self.lattice.wilson_loops(x=x_prop)

        plaq_loss = torch.tensor(0.).to(x_init.real.dtype).to(x_init.device)
        if self.plaq_weight > 0:
            plaq_loss = self._plaq_loss(w1=wl_init, w2=wl_prop, acc=acc)

        charge_loss = torch.tensor(0.).to(x_init.real.dtype).to(x_init.device)
        if self.charge_weight > 0:
            charge_loss = self._charge_loss(w1=wl_init, w2=wl_prop, acc=acc)

        return plaq_loss + charge_loss
