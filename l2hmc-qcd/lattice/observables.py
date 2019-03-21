"""
Helper methods for calculating lattice observables.

Separated implementation from GaugeLattice object to deal with simulated
annealing schedule.

Author: Sam Foreman (github: @saforem2)
Date: 1/12/2019
"""
import numpy as np
import tensorflow as tf
from scipy.special import i0, i1

def u1_plaq_exact(beta):
    """Exact value of the average plaquette calculated at `beta`."""
    return i1(beta) / i0(beta)

def pbc(tup, shape):
    """Enforce periodic boundary condiditions for tup. Put `tup` in `shape`."""
    return list(np.mod(tup, shape))

def pbc_tf(tup, shape):
    """Same as `pbc` method above, in tensorflow."""
    return list(tf.mod(tup, shape))

def project_angle(x):
    """Function to project an angle from [-4pi, 4pi] to [-pi, pi]."""
    return x - 2 * np.pi * tf.math.floor((x + np.pi) / (2 * np.pi))

# pylint: disable=invalid-name
def _calc_total_action(lattice, beta):
    """Computes the total action of an individual lattice by summing the
    internal energy of each plaquette over all plaquettes.

    NOTE:
        * For SU(N) (N = 2, 3), the action of a single plaquette is
        calculated as:
            Sp = 1 - Re{Tr(Up)}, where Up is the plaquette_operator defined
            as the product of the gauge fields around an elementary
            plaquette.
        * For U(1), the action of a sinigle plaquette is calculated as:
            Sp = 1 - cos(Qp), where Qp is the plaquette operator defined as
            the sum of the angles (phases) around an elementary plaquette.
    """
    total_action = 0.0
    shape = lattice.site_idxs
    links = lattice.links
    bases = lattice.bases
    for plaq in lattice.plaquette_idxs:
        *site, u, v = plaq
        l1 = links[tuple(pbc_tf(site, shape) + [u])]
        l2 = links[tuple(pbc_tf(site + bases[u], shape) + [v])]
        l3 = links[tuple(pbc_tf(site + bases[v], shape) + [u])]
        l4 = links[tuple(pbc_tf(site, shape) + [v])]

        total_action += tf.math.cos(l1 + l2 - l3 -l4)

    return beta * total_action

def calc_total_action(samples, beta):
    """Computes the total action of each sample lattice in samples."""
    if tf.executing_eagerly():
        return np.array([
            _calc_total_action(sample, beta) for sample in samples
        ])

    total_actions = []
    for idx in range(samples.shape[0]):
        total_actions.append(_calc_total_action(samples[idx], beta))
    return total_actions

def _calc_observables(lattice, beta):
    """Calculates all relevant observables for an individual lattice.

    Args:
        lattice: Lattice object.
        beta: Inverse coupling strength.
    Returns:
        total_action: Total (Wilson) action.
        average_plaquette: Average plaquette.
        topological_charge (int): Topological charge of lattice.
    """
    plaquettes_sum = 0.
    topological_charge = 0.
    total_action = 0.

    for plaq in lattice.plaquette_idxs:
        *site, u, v = plaq
        plaq_sum = lattice.plaquette_operator(site, u, v, lattice.links)
        local_action = np.cos(plaq_sum)

        total_action += 1 - local_action
        plaquettes_sum+= local_action
        topological_charge += project_angle(plaq_sum)

    return [beta * total_action,
            plaquettes_sum /  lattice.num_plaquettes,
            int(topological_charge / (2 * np.pi))]

def calc_observables(samples, beta):
    """Calculates all relevant observables for each lattice in `samples`."""
    return [
        _calc_observables(sample, beta) for sample in samples
    ]
