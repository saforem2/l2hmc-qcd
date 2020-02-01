"""
utils.py

Collection of useful functions for running inference on a trained L2HMC sampler

Author: Sam Foreman (github: @saforem2)
Date: 11/06/2019
"""
import tensorflow as tf
import numpy as np

import utils.file_io as io

from lattice.lattice import GaugeLattice


def set_eps(sess, eps):
    """Explicitly sets the step-size (`eps`) when running inference.

    Args:
        sess (tf.Session): Session in which to set eps.
        eps (float): Desired step size.
    """
    graph = tf.get_default_graph()
    eps_setter = graph.get_operation_by_name('init/eps_setter')
    inputs = tf.get_collection('inputs')
    eps_tensor = [i for i in tf.global_variables() if 'eps' in i.name][0]
    eps_ph = [i for i in inputs if 'eps_ph' in i.name][0]

    eps_np = sess.run(eps_tensor)
    io.log(f'INFO: Original value of `eps`: {eps_np}')
    io.log(f'INFO: Setting `eps` to: {eps}.')
    sess.run(eps_setter, feed_dict={eps_ph: eps})
    eps_np = sess.run(eps_tensor)

    io.log(f'INFO: New value of `eps`: {eps_np}')


def init_gauge_samples(params, init_method):
    """Create initial samples to be used at beginning of inference run."""
    x_dim = params['space_size'] * params['time_size'] * params['dim']
    samples_shape = (params['batch_size'], x_dim)
    if 'zero' in init_method:
        samples_init = (np.zeros(samples_shape)
                        + 1e-2 * np.random.randn(*samples_shape))
    if 'ones' in init_method:
        samples_init = (np.ones(samples_shape)
                        + 1e-2 * np.random.randn(*samples_shape))
    else:
        #  tmp = samples_shape[0] * samples_shape[1]
        #  samples_init = np.random.randn(-1, 1, tmp).reshape(*samples_shape)
        samples_init = np.random.randn(*samples_shape)

    return samples_init


def init_gmm_samples(params, init_method):
    samples_shape = (params['batch_size'], params['x_dim'])
    if init_method == 'random':
        tmp = samples_shape[0] * samples_shape[1]
        samples_init = np.random.uniform(-1, 1, tmp).reshape(*samples_shape)
    elif 'zero' in init_method:
        samples_init = np.zeros(samples_shape)
    elif 'ones' in init_method:
        samples_init = np.ones(samples_shape)
        #  samples_init = 2 * np.random.rand(*(params['batch_size'],
        #                                      params['x_dim']))

    return samples_init


def _create_lattice(params):
    """Creates and returns a `GaugeLattice` object from `params`."""
    return GaugeLattice(dim=params['dim'],
                        rand=params['rand'],
                        link_type=params['link_type'],
                        time_size=params['time_size'],
                        space_size=params['space_size'],
                        batch_size=params['batch_size'])


def _gauge_potential(lattice):
    return lattice.get_potential_fn(lattice.samples)


def gauge_potential_energy(state, lattice):
    """Create gauge action (potential, minus-log-likelihood) from GaugeLattice.

    Args:
        lattice (GaugeLattice object): Input lattice.
    """
    return state.beta * _gauge_potential(lattice)


def potential_energy(x, beta, potential_fn):
    return beta * potential_fn(x)


def kinetic_energy(v):
    return 0.5 * np.sum(v ** 2, axis=1)


def hamiltonian(x, beta, v, potential_fn):
    return potential_energy(x, beta, potential_fn) + kinetic_energy(v)


def calc_energies(state_init, state_proposed, state_out):
    """calculate the Hamiltonian and PE/KE for each state above.

    NOTE: 
        - `State` is a namedtuple of the form (x, v, beta).

        - `EnergyData` is a namedtuple of the form ('init', 'proposed', out')

    Args:
        state_init (State object): Initial state at the beginning of the
            trajectory.
        state_proposed (State object): Proposed state at the end of the MD
            trajectory (prior to accept/reject).
        state_out (State object): Output state following accept/reject at the
            end of the trajectory.

    Returns:
        pe_data (EnergyData object): Potential energy data.
        ke_data (EnergyData object): Kinetic energy data.
        h_data (EnergyData object): Hamiltonian data.
    """
    for si, sp, so in zip(state_init, state_proposed, state_out):
        pei = potential_energy(x, beta, potential_fn)
