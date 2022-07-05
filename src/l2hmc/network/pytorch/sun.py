"""
network/pytorch/sun.py
"""
from __future__ import absolute_import, division, print_function, annotations
import torch

nn = torch.nn
Tensor = torch.Tensor

class TimePotentialSU3(nn.Module):
    def __init__(self) -> None:
        super(TimePotentialSU3, self).__init__()
        self.full_eigdecomp = su3_to_eigs_cdesa
        self.deepset = ComplexDeepTimeSet(1, 1, hidden_channels=64)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        x = self.full_eigdecomp(x)
        x = x.unsqueeze(-1)
        x = self.deepset(t, x)

        return x


class SU3TimeEquivariantVectorField(nn.Module):
    def __init__(self, func):
        super(SU3TimeEquivariantVectorField, self).__init__()
        self.func = func

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.autograd.grad(  # vector field
            self.func(t, x).squeeze().sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0]


class AmbientProjNN(nn.Module):
    def __init__(self, func):
        super(AmbientProjNN, self).__init__()
        self.func = func
        self.man = SUN()

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        self.man.proju(x, self.func(t, x))



