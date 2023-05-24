"""
group/group.py

Implements `Group` abstract base class, for subclassing
"""
from __future__ import absolute_import, division, print_function, annotations

from abc import ABC, abstractmethod

import torch
# import numpy as np
# import tensorflow as tf
from typing import Optional, Any, Sequence

# tfTensor = tf.Tensor
ptTensor = torch.Tensor

# DataType = Union[torch.dtype, tf.DType]
# Any = Union[ptTensor, tfTensor, np.ndarray]


class Group(ABC):
    """Gauge group represented as matrices in the last two dimensions."""
    def __init__(
            self,
            dim: int,
            shape: Sequence[int],
            dtype: Any,
            name: Optional[str] = None,
    ) -> None:
        self._dim = dim
        self._shape = shape
        self._dtype = dtype
        if name is not None:
            self._name = name

    @abstractmethod
    def exp(self, x: Any) -> Any:
        pass

    @abstractmethod
    def mul(
            self,
            a: Any,
            b: Any,
            adjoint_a: bool = False,
            adjoint_b: bool = False,
    ) -> Any:
        pass

    @abstractmethod
    def update_gauge(
            self,
            x: Any,
            p: Any
    ) -> Any:
        pass

    @abstractmethod
    def adjoint(self, x: Any) -> Any:
        pass

    @abstractmethod
    def trace(self, x: Any) -> Any:
        pass

    @abstractmethod
    def compat_proj(self, x: Any) -> Any:
        pass

    @abstractmethod
    def random(self, shape: list[int]) -> Any:
        pass

    @abstractmethod
    def random_momentum(self, shape: list[int]) -> Any:
        pass

    @abstractmethod
    def kinetic_energy(self, p: Any) -> Any:
        pass
