import numpy as np
import os
from re import split
from multiprocessing import pool

def generate_SU2(eps):
    """Generates a random SU(2) matrix where eps controls the 'distance' from
    the identity."""
    r_rand_nums = np.random.uniform(0, 0.5, (4))
    r = np.empty((4))

    r[1:] = eps * r_rand_nums[1:] / np.linalg.norm(r_rand_nums[1:])
    r[0] = np.sign(r_rand_nums[0]) * np.sqrt(1 - eps**2)

    r11 = r[0] + r[3] * 1j
    r12 = r[1] * 1j + r[2]
    r21 = r[1] * 1j - r[2]
    r22 = r[0] - r[3] * 1j

    return np.array([[r11, r12], [r21, r22]])

def generate_SU3_array(n, eps):
    """Generates a 2*n array of SU(3) matrices where eps controls the
    distance from the identity."""
    su3_array = np.zeros((2*n, 3, 3), dtype=np.complex64)

    for i in range(n):
        R_su3 = np.identity(3, dtype=np.complex64)
        S_su3 = np.identity(3, dtype=np.complex64)
        T_su3 = np.identity(3, dtype=np.complex64)

        R_su3[:2, :2] = generate_SU2(eps)
        S_su3[0:3:2, 0:3:2] = generate_SU2(eps)
        T_su3[1:, 1:] = generate_SU2(eps)

        X_su3 = np.dot(np.dot(R_su3, S_su3), T_su3)

        su3_array[2*i, :, :] = X_su3
        su3_array[2*i+1, :, :] = X_su3.conj().T

    return su3_array

def generate_site():
    link_value = np.identity(3, dtype=np.complex)
    link = np.tile(link_value, (4, 1, 1))
    return link

def generate_link():
    """Returns a 4-dimensional array of SU(3) matrices initialized to the
    identity which can be assigned to each spatial lattice point."""
    link_value = np.identity(3, dtype=np.complex)
    link = np.tile(link_value, (4, 1, 1))
    return link

def generate_lattice(n_points):
    """Returns a (n_points, n_points, n_points, n_points, 4, 3, 3) numpy array
    as our lattice with four SU(3) links assigned to each point in the
    lattice."""
    grid = np.zeros(tuple(4 * [4] + [4, 3, 3]), dtype=np.complex64)
    for t in range(n_points):
        for x in range(n_points):
            for y  in range(n_points):
                for z in range(n_points):
                    grid[t, x, y, z, :, :, :] = generate_site()
    return grid

def link(lattice, coords, mu):
    """Function to account for periodic boundary conditions."""
    n_points = lattice.shape[0]
    return lattice[coords[0] % n_points,
                   coords[1] % n_points,
                   coords[2] % n_points,
                   coords[3] % n_points,
                   mu, :, :]

def wilson_link_sum(lattice, coords, mu, u0):
    """Staple sum for the wilson plaquette action."""
    dimension = 4
    res = np.zeros((3, 3), dtype=np.complex)

    for nu in range(dimension):
        if nu != mu:
            coords_mu = coords[:]
            coords_mu[mu] += 1

            coords_nu = coords[:]
            coords_nu[nu] += 1

            coords_mu_n_nu = coords[:]
            coords_mu_n_nu[mu] += 1
            coords_mu_n_nu[nu] -= 1

            coords_n_nu = coords[:]
            coords_n_nu[nu] -= 1

            # 1x1 positive
            res += np.dot(np.dot(link(lattice, coords_mu, nu),
                                 link(lattice, coords_nu, mu).conj().T),
                          link(lattice, coords, nu).conj().T)

            # 1x1 negative
            res += np.dot(np.dot(link(lattice, coords_mu_n_nu, nu).conj().T,
                                 link(lattice, coords_n_nu, mu).conj().T),
                          link(lattice, coords_n_nu, nu))

    return (res / u0 ** 4) / 3

def improved_link_sum(lattice, coords, mu, u0):
    """Staple sum for rectangular improved action which includes next nearest
    neighbor links."""
    dimension = 4
    res = np.zeros((3, 3), dtype=np.complex)
    res_rec = np.zeros((3, 3), dtype=np.complex)

    for nu in range(dimension):
        if nu != mu:
            coords_mu = coords[:]
            coords_mu[mu] += 1

            coords_nu = coords[:]
            coords_nu[nu] += 1

            coords_mu_nu = coords[:]
            coords_mu_nu[mu] += 1
            coords_mu_nu[nu] += 1

            coords_mu_mu = coords[:]
            coords_mu_mu[mu] += 2

            coords_nu_nu = coords[:]
            coords_nu_nu[nu] += 2

            coords_mu_n_nu = coords[:]
            coords_mu_n_nu[mu] += 1
            coords_mu_n_nu[nu] -= 1

            coords_n_nu = coords[:]
            coords_n_nu[nu] -= 1

            coords_n_nu_nu = coords[:]
            coords_n_nu_nu[nu] -= 2

            coords_mu_mu_n_nu = coords[:]
            coords_mu_mu_n_nu[mu]

