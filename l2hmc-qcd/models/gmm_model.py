"""
gmm_model.py

Implements the GaussianMixtureModel class responsible for building the
computation graph used in tensorflow.

Author: Sam Foreman (github: @saforem2)
Date: 09/03/2019
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time

import numpy as np
import tensorflow as tf

import utils.file_io as io

from collections import namedtuple
from utils.horovod_utils import warmup_lr  # noqa: 401
from utils.distributions import GMM, gen_ring
from base.base_model import BaseModel
from params.gmm_params import GMM_PARAMS
import config as cfg

if cfg.HAS_HOROVOD:
    import horovod.tensorflow as hvd  # noqa: 401

TF_FLOAT = cfg.TF_FLOAT
NP_FLOAT = cfg.NP_FLOAT

LFdata = namedtuple('LFdata', ['init', 'proposed', 'prob'])
SEP_STR = 80 * '-'
SEP_STRN = 80 * '-' + '\n'


def distribution_arr(x_dim, num_distributions):
    """Create array describing likelihood of drawing from distributions."""
    if num_distributions > x_dim:
        pis = [1. / num_distributions] * num_distributions
        pis[0] += 1 - sum(pis)
    if x_dim == num_distributions:
        big_pi = round(1.0 / num_distributions, x_dim)
        pis = num_distributions * [big_pi]
    else:
        big_pi = (1.0 / num_distributions) - x_dim * 1e-16
        pis = num_distributions * [big_pi]
        small_pi = (1. - sum(pis)) / (x_dim - num_distributions)
        pis.extend((x_dim - num_distributions) * [small_pi])

    return np.array(pis, dtype=NP_FLOAT)


def ring_of_gaussians(num_modes, sigma, r=1.):
    """Create ring of Gaussians for GaussianMixtureModel. 

    Args:
        num_modes (int): Number of modes for distribution.
        sigma (float): Standard deviation of each mode.
        x_dim (int): Spatial dimensionality in which the distribution exists.
        radius (float): Radius from the origin along which the modes are
            located.

    Returns:
        distribution (GMM object): Gaussian mixture distribution.
        mus (np.ndarray): Array of the means of the distribution.
        covs (np.array): Covariance matrices.
        distances (np.ndarray): Array of the differences between different
            modes. 
    """
    covs, distribution = gen_ring(r=r, var=sigma, nb_mixtures=num_modes)
    mus = np.array(distribution.mus)
    diffs = mus[1:] - mus[:-1, :]
    distances = [np.sqrt(np.dot(d, d.T)) for d in diffs]

    return distribution, mus, covs, distances


def lattice_of_gaussians(num_modes, sigma, x_dim=2, size=None):
    """Create lattice of Gaussians for GaussianMixtureModel.

    Args:
        num_modes (int): Number of modes for distribution.
        sigma (float): Standard deviation of each mode.
        x_dim (int): Spatial dimensionality in which the distribution exists.
        size (int): Spatial extent of lattice.

    Returns:
        distribution (GMM object): Gaussian mixture distribution.
        covs (np.array): Covariance matrices.
        mus (np.ndarray): Array of the means of the distribution.
        pis (np.ndarray): Array of relative probabilities for each mode. Must
            sum to 1.
    """
    if size is None:
        size = int(np.sqrt(num_modes))

    mus = np.array([(i, j) for i in range(size) for j in range(size)])
    covs = np.array([sigma * np.eye(x_dim) for _ in range(num_modes)])
    pis = [1. / num_modes] * num_modes
    pis[0] += 1. - sum(pis)

    distribution = GMM(mus, covs, pis)

    return distribution, mus, covs, pis


class GaussianMixtureModel(BaseModel):
    def __init__(self, params=None):
        super(GaussianMixtureModel, self).__init__(params)
        self._model_type = 'GaussianMixtureModel'

        if params is None:
            params = GMM_PARAMS  # default parameters, defined in `config.py`.

        self.build(params)

    def build(self, params=None):
        """Build TensorFlow graph."""
        params = self.params if params is None else params

        self.x_dim = params.get('x_dim', 2)
        self.num_distributions = params.get('num_distributions', 2)

        t0 = time.time()
        io.log(SEP_STRN + 'INFO: Building graph for `GaussianMixtureModel`...')

        # ************************************************
        # Build inputs and their respective placeholders
        # ------------------------------------------------
        self._build_inputs()

        with tf.name_scope('init'):
            # ***************************************************************
            # Create target distribution for Gaussian Mixture Model
            # ---------------------------------------------------------------
            means, covs, pis, distribution = self.create_distribution()
            self.means = means
            self.covs = covs
            self.pis = pis
            self.distribution = distribution

        # ***************************************************************
        # Build operations common to all models (defined in `BaseModel`)
        # ---------------------------------------------------------------
        self._build()

        io.log(f'INFO: Done building graph. '
               f'Took: {time.time() - t0}s\n' + 80 * '-')

    def create_distribution(self):
        """Create distribution."""
        self.sigmas = self._get_sigmas()

        if self.arrangement == 'lattice':
            sigma = np.max(self.sigmas)
            distribution, means, covs, pis = lattice_of_gaussians(
                self.num_distributions, sigma, x_dim=self.x_dim
            )

        elif self.arrangement == 'ring':
            sigma = np.max(self.sigmas)
            r = getattr(self, 'size', 1.)
            distribution, means, covs, pis = ring_of_gaussians(
                self.num_distributions, sigma, r=r
            )

        else:
            means = self._create_means()
            covs = self._create_covs()
            pis = self._get_pis()
            distribution = GMM(means, covs, pis)

        return means, covs, pis, distribution

    def _create_means(self):
        """Create means of target distribution."""
        means = np.zeros((self.x_dim, self.x_dim))

        if self.arrangement == 'diag':
            for i in range(self.x_dim):
                means[i::self.x_dim, i] = self.center

        if self.arrangement == 'yaxis':
            means[::2, 1] = self.center
            means[1::2, 1] = - self.center

        if self.arrangement == 'xaxis':
            means[::2, 0] = self.center
            means[1::2, 0] = - self.center

        else:
            if self.arrangement not in ['xaxis', 'yaxis', 'diag']:
                raise AttributeError(f'Invalid value for `self.arrangement`: '
                                     f'{self.arrangement}. Expected one of: '
                                     f"'xaxis', 'yaxis', 'diag', "
                                     "'lattice', or 'ring'.")

        return means.astype(NP_FLOAT)

    def _get_pis(self):
        pi1 = getattr(self, 'pi1', None)
        pi2 = getattr(self, 'pi2', None)
        if pi1 is None or pi2 is None:
            pis = distribution_arr(self.x_dim, self.num_distributions)
        else:
            pis = np.array([pi1, pi2])

        return pis

    def _get_sigmas(self):
        """Get sigmas."""
        sigmas = getattr(self, 'sigmas', None)
        if sigmas is None:
            sigma = getattr(self, 'sigma', None)
            if sigma is not None:
                sigmas = sigma * np.ones(self.x_dim)
            else:
                sigma1 = getattr(self, 'sigma1', 0.1)
                sigma2 = getattr(self, 'sigma2', 0.1)
                sigmas = np.array([sigma1, sigma2])

        sigmas = np.array(sigmas, dtype=NP_FLOAT)

        return sigmas

    def _create_covs(self):
        """Create covariance matrix from of individual covariance matrices."""
        covs = np.array(
            [s * np.eye(self.x_dim) for s in self.sigmas], dtype=NP_FLOAT
        )

        return covs

    def create_dynamics(self, **params):
        """Create `Dynamics` object."""
        dynamics_params = {
            'eps_trainable': not self.eps_fixed,
            'num_filters': 2,
            'x_dim': self.x_dim,
            'batch_size': self.batch_size,
            '_input_shape': (self.batch_size, self.x_dim)
        }

        dynamics_params.update(params)
        potential_fn = self.distribution.get_energy_function()
        dynamics = self._create_dynamics(potential_fn, **dynamics_params)

        return dynamics
