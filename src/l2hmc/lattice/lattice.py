"""
lattice/lattice.py

Implements `Lattice`, an ABC for building Lattice subclasses
"""
from __future__ import absolute_import, annotations, division, print_function
from abc import ABC, abstractmethod
from typing import Optional, Any

import numpy as np
# import tensorflow as tf
# import torch

from l2hmc.group.group import Group
from l2hmc.configs import Charges


# TensorLike = Union[torch.Tensor, tf.Tensor, np.ndarray]


class Lattice(ABC):
    def __init__(
            self,
            group: Group,
            nchains: int,
            shape: list[int],
    ) -> None:
        self.g = group
        self.link_shape = self.g._shape
        self.xshape = [self.g._dim, *shape]
        if len(self.g._shape) > 1:
            self.xshape.extend(self.g._shape)

        self.dim = self.g._dim
        self._shape = [nchains, *self.xshape]
        # _shape = [nchains, self.g._dim, *shape, self.g._shape]
        # self._shape = list([int(i) for i in _shape])
        self.volume = np.cumprod(self._shape[1:-len(self.g._shape)])[-1]
        self.nchains = nchains
        self._lattice_shape = shape
        self.volume = np.cumprod(shape)[-1]

    def draw_batch(self) -> Any:
        return self.g.random(list(self._shape[:-2]))

    def update_link(
            self,
            x: Any,
            p: Any
    ) -> Any:
        return self.g.mul(self.g.exp(p), x)

    def random(self) -> Any:
        return self.g.random(list(self._shape))

    def random_momentum(self) -> Any:
        return self.g.random_momentum(list(self._shape))

    @abstractmethod
    def _action(self, wloops: Any, beta: Any) -> Any:
        """Compute the action using wloops, at inverse coupling beta."""
        pass

    def action(
            self,
            x: Any,
            beta: Any,
    ) -> Any:
        """Compute the action directly from x, at inverse coupling beta."""
        wloops = self.wilson_loops(x)
        assert wloops is not None
        return self._action(wloops, beta)

    @abstractmethod
    def kinetic_energy(self, v: Any) -> Any:
        pass

    def potential_energy(self, x: Any, beta: Any) -> Any:
        return self.action(x, beta)

    @abstractmethod
    def wilson_loops(self, x: Any) -> Any:
        pass

    @abstractmethod
    def _plaqs(self, wloops: Any) -> Any:
        pass

    def plaqs(
            self,
            x: Optional[Any] = None,
            wloops: Optional[Any] = None
    ) -> Any:
        if wloops is None:
            assert x is not None
            wloops = self.wilson_loops(x) if wloops is None else wloops

        assert wloops is not None
        return self._plaqs(wloops)

    def charges(
            self,
            x: Optional[Any] = None,
            wloops: Optional[Any] = None
    ) -> Charges:
        if wloops is None:
            assert x is not None
            wloops = self.wilson_loops(x)

        return self._charges(wloops=wloops)

    @abstractmethod
    def _charges(
            self,
            wloops: Any,
    ) -> Charges:
        pass

    def sin_charges(
            self,
            x: Optional[Any] = None,
            wloops: Optional[Any] = None
    ) -> Any:
        if wloops is None:
            assert x is not None
            wloops = self.wilson_loops(x)

        return self._sin_charges(wloops)

    @abstractmethod
    def _sin_charges(self, wloops: Any) -> Any:
        pass

    def int_charges(
            self,
            x: Optional[Any] = None,
            wloops: Optional[Any] = None
    ) -> Any:
        if wloops is None:
            assert x is not None
            wloops = self.wilson_loops(x)

        return self._int_charges(wloops)

    @abstractmethod
    def _int_charges(self, wloops: Any) -> Any:
        pass

    def unnormalized_log_prob(
            self,
            x: Any,
            beta: Any
    ) -> Any:
        return self.action(x=x, beta=beta)

    @abstractmethod
    def grad_action(
            self,
            x: Any,
            beta: Any,
    ) -> Any:
        pass

    @abstractmethod
    def action_with_grad(
            self,
            x: Any,
            beta: Any,
    ) -> tuple[Any, Any]:
        pass

    @abstractmethod
    def calc_metrics(
            self,
            x: Any,
            beta: Optional[Any] = None,
    ) -> dict[str, Any]:
        wloops = self.wilson_loops(x)
        plaqs = self.plaqs(wloops=wloops)
        charges = self.charges(wloops=wloops)
        metrics = {
            'plaqs': plaqs,
            'intQ': charges.intQ,
            'sinQ': charges.sinQ,
        }
        if beta is not None:
            s, ds = self.action_with_grad(x, beta)
            metrics['action'] = s
            metrics['grad_action'] = ds

        return metrics

    @abstractmethod
    def plaq_loss(
            self,
            acc: Any,
            x1: Optional[Any] = None,
            x2: Optional[Any] = None,
            wloops1: Optional[Any] = None,
            wloops2: Optional[Any] = None,
    ) -> Any:
        if wloops1 is None:
            assert x1 is not None
            wloops1 = self.wilson_loops(x1)

        if wloops2 is None:
            assert x2 is not None
            wloops2 = self.wilson_loops(x2)

        pass

    @abstractmethod
    def charge_loss(
            self,
            acc: Any,
            x1: Optional[Any] = None,
            x2: Optional[Any] = None,
            wloops1: Optional[Any] = None,
            wloops2: Optional[Any] = None,
    ) -> Any:
        if wloops1 is None:
            assert x1 is not None
            wloops1 = self.wilson_loops(x1)

        if wloops2 is None:
            assert x2 is not None
            wloops2 = self.wilson_loops(x2)
