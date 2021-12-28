"""
dynamics.py

Contains pytorch implementation of Dynamics object for training L2HMC sampler.
"""
from __future__ import absolute_import, division, print_function, annotations

from pathlib import Path
from typing import Union
import torch

Shape = Union[tuple, list]
Tensor = torch.Tensor

def rand_unif(shape: Shape,
              a: float,
              b: float,
              requires_grad: bool) -> Tensor:
    """Draw tensor from random uniform distribution U[a, b]"""
    rand = torch.rand(shape, requires_grad=requires_grad)
    return torch.tensor((a - b) * rand + b, requires_grad=requires_grad)
