"""
lattice.py

Contains implementation of generic GaugeLattice object.
"""
from __future__ import absolute_import, print_function, division, annotations

import numpy as np
import torch

# from l2hmc.group.pytorch import group as g
# from l2hmc.group import group as g
from l2hmc.lattice.su3.numpy.lattice import BaseLatticeSU3


Array = np.ndarray
Tensor = torch.Tensor

PI = np.pi


class LatticeSU3(BaseLatticeSU3):
    """4D Lattice with SU(3) Links"""
    dim = 4
    def __init__(
        self,
        nb: int,
        shape: tuple[int, int, int, int],
        c1: float = 0.0,
    ) -> None:
        pass
