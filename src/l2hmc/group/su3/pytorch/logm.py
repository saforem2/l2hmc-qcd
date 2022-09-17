"""
logm.py

Modified from original implementation at:
  https://github.com/CUAI/Equivariant-Manifold-Flows/blob/main/flows/logm.py
"""
from __future__ import absolute_import, print_function, division, annotations

import torch
import numpy as np

Tensor = torch.Tensor


def charpoly3x3(A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """ det(λ * I - A) = λ³ + λ² * c[3] + λ * c[2] + c[1] """
    elem1 = -(
        A[:, 0, 0] * (A[:, 1, 1] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 1])
        - A[:, 1, 0] * (A[:, 0, 1] * A[:, 2, 2] - A[:, 0, 2] * A[:, 2, 1])
        + A[:, 2, 0] * (A[:, 0, 1] * A[:, 1, 2] - A[:, 0, 2] * A[:, 1, 1])
    )
    elem2 = (
        A[:, 0, 0] * A[:, 1, 1]
        + A[:, 0, 0] * A[:, 2, 2]
        + A[:, 1, 1] * A[:, 2, 2]
        - A[:, 1, 0] * A[:, 0, 1]
        - A[:, 2, 0] * A[:, 0, 2]
        - A[:, 2, 1] * A[:, 1, 2]
    )
    elem3 = -(A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2])

    return (elem1, elem2, elem3)


def cmax(x: Tensor, y: Tensor) -> Tensor:
    """Returns the largets-magnitude complex number"""
    return torch.where(torch.abs(x) > torch.abs(y), x, y)


def cubic_zeros(p):
    a = 1
    b = p[2]
    c = p[1]
    d = p[0]
    D0 = b ** 2 - 3 * a * c
    D1 = 2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d
    L = torch.pow(1e-3 + D1 ** 2 - 4 * D0 ** 3, 0.5)
    V = cmax((D1 + L) / 2, (D1 - L) / 2)
    C = V ** (1 / 3)
    w = np.exp(2 * np.pi * 1j / 3)

    return [
        -(b + (w ** k * C) + D0 / (w ** k * C)) / (3 * a) for k in range(3)
    ]


def su3_to_eigs(x: Tensor) -> Tensor:
    p = charpoly3x3(x)
    zs = cubic_zeros(p)
    return torch.cat([x.unsqueeze(-1) for x in zs], dim=-1)


def log3x3(x: Tensor):
    eigs = su3_to_eigs(x)
    q, _ = torch.linalg.solve(
        torch.log(eigs).unsqueeze(-1), (
            1e-6 * torch.eye(3).unsqueeze(0)
            + eigs.unsqueeze(-1) ** (
                torch.tensor([0, 1, 2]).unsqueeze(0).unsqueeze(0)
            )
        )
    )
    q = q.unsqueeze(-1)
    eye = torch.eye(x.shape[-1]).reshape(1, x.shape[-1], x.shape[-1])
    eye = eye.repeat(x.shape[0], 1, 1)

    return q[:, 0] * eye + q[:, 1] * x + q[:, 2] * x @ x
