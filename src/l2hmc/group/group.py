"""
group/group.py

Implements `Group` abstract base class, for subclassing
"""
from __future__ import absolute_import, division, print_function, annotations

from abc import ABC, abstractmethod

import torch
import numpy as np
import tensorflow as tf
from typing import Union

tfTensor = tf.Tensor
ptTensor = torch.Tensor

DataType = Union[torch.dtype, tf.DType]
TensorLike = Union[ptTensor, tfTensor, np.ndarray]


class Group(ABC):
    """Gauge group represented as matrices in the last two dimensions."""
    def __init__(self, dim: int, shape: list[int], dtype: DataType) -> None:
        self._dim = dim
        self._shape = shape
        self._dtype = dtype

    @abstractmethod
    def exp(self, x: TensorLike) -> TensorLike:
        pass

    @abstractmethod
    def mul(
            self,
            a: TensorLike,
            b: TensorLike,
            adjoint_a: bool = False,
            adjoint_b: bool = False,
    ) -> TensorLike:
        pass

    @abstractmethod
    def update_gauge(
            self,
            x: TensorLike,
            p: TensorLike
    ) -> TensorLike:
        pass

    @abstractmethod
    def adjoint(self, x: TensorLike) -> TensorLike:
        pass

    @abstractmethod
    def trace(self, x: TensorLike) -> TensorLike:
        pass

    @abstractmethod
    def compat_proj(self, x: TensorLike) -> TensorLike:
        pass

    @abstractmethod
    def random(self, shape: list[int]) -> TensorLike:
        pass

    @abstractmethod
    def random_momentum(self, shape: list[int]) -> TensorLike:
        pass

    @abstractmethod
    def kinetic_energy(self, p: TensorLike) -> TensorLike:
        pass
