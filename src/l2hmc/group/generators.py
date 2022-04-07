"""
generators.py

Contains methods for generating elements of SU(2), SU(3) gauge groups.

Note:
 - The `eps` (type: float) argument to the below functions controls the
   'distance' from the identity matrix.
"""
from __future__ import absolute_import, print_function, division, annotations

import numpy as np

# from re import split
# from multiprocessing import pool


def generate_SU2(eps: float) -> np.ndarray:
    """Returns a single randomly initialized SU(2) matrix."""
    r_rand_nums = np.random.uniform(0, 0.5, (4))
    r = np.empty((4))

    r[1:] = eps * r_rand_nums[1:] / np.linalg.norm(r_rand_nums[1:])
    r[0] = np.sign(r_rand_nums[0]) * np.sqrt(1 - eps ** 2)

    r11 = +r[0] + 1j * r[3]
    r12 = +r[2] + 1j * r[1]
    r21 = -r[2] + 1j * r[1]
    r22 = +r[0] - 1j * r[3]

    return np.array([[r11, r12], [r21, r22]])


def generate_SU3(eps: float) -> np.ndarray:
    """Returns a single randomly initialized SU(3) mtx."""
    r = np.identity(3, dtype=np.complexfloating)
    s = np.identity(3, dtype=np.complexfloating)
    t = np.identity(3, dtype=np.complexfloating)

    r[:2, :2] = generate_SU2(eps)
    s[0:3:2, 0:3:2] = generate_SU2(eps)
    t[1:, 1:] = generate_SU2(eps)

    return np.dot(np.dot(r, s), t)


def generate_SU3_array(n: int, eps: float) -> np.ndarray:
    """Generates a 2*n array of SU(3) mtxs; eps controls dist from Identity"""
    arr = np.zeros((2 * n, 3, 3), dtype=np.complexfloating)
    for i in range(n):
        mtx = generate_SU3(eps)
        arr[2 * i] = mtx
        arr[2 * i + 1] = mtx.conj().T

    return arr
