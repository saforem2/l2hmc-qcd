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
#  from .gauge_model import allclose
from dynamics.dynamics import Dynamics
from params.gmm_params import GMM_PARAMS
from config import NP_FLOAT, TF_FLOAT, HAS_HOROVOD, State

if HAS_HOROVOD:
    import horovod.tensorflow as hvd  # noqa: 401

LFdata = namedtuple('LFdata', ['init', 'proposed', 'prob'])


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
        self.num_distributions = params.get('num_distributions', 2)
        #  aux_weight = getattr(self, 'aux_weight', 1.)
        use_gaussian_loss = getattr(self, 'use_gaussian_loss', False)
        use_nnehmc_loss = getattr(self, 'use_nnehmc_loss', False)
        self.use_gaussian_loss = use_gaussian_loss
        self.use_nnehmc_loss = use_nnehmc_loss

        t0 = time.time()
        io.log(80 * '-')
        io.log(f'INFO: Building graph for `GaussianMixtureModel`...')
        with tf.name_scope('init'):
            # ---------------------------------------------------------------
            # Create target distribution for Gaussian Mixture Model
            # ---------------------------------------------------------------
            means, covs, pis, distribution = self.create_distribution()
            self.means = means
            self.covs = covs
            self.pis = pis
            self.distribution = distribution
            #  self.means = self._create_means(params)
            #  self.sigmas, self.covs = self._create_covs(params)
            #  self.pis = distribution_arr(self.x_dim, self.num_distributions)
            #  self.distribution = GMM(self.means, self.covs, self.pis)

            # ---------------------------------------------------------------
            # Create inputs as to be fed values using `tf.placeholders`
            # ---------------------------------------------------------------
            io.log(f'INFO: Creating input placeholders...')
            inputs = self._create_inputs()
            self.x = inputs['x']
            self.beta = inputs['beta']
            #  nw_keys = ['scale_weight', 'transl_weight', 'transf_weight']
            #  self.net_weights = [inputs[k] for k in nw_keys]
            self.net_weights = inputs['net_weights']
            self.train_phase = inputs['train_phase']
            self.eps_ph = inputs['eps_ph']
            self._inputs = inputs

            # ---------------------------------------------------------------
            # Create dynamics for running augmented L2HMC leapfrog
            # ---------------------------------------------------------------
            io.log(f'INFO: Creating `Dynamics`...')
            self.dynamics = self.create_dynamics()
            self.dynamics_eps = self.dynamics.eps
            # Create operation for assigning to `dynamics.eps` 
            # the value fed into the placeholder `eps_ph`.
            self.eps_setter = self._build_eps_setter()

            # ---------------------------------------------------------------
            # Create metric function for measuring 'distance' between configs
            # ---------------------------------------------------------------
            self.metric_fn = self._create_metric_fn()

        with tf.name_scope('sampler'):
            x_data, z_data = self._build_sampler()

        # *******************************************************************
        # Build energy_ops to calculate energies.
        # -------------------------------------------------------------------
        with tf.name_scope('energy_ops'):
            self.v_ph = tf.placeholder(dtype=TF_FLOAT, shape=self.x.shape,
                                       name='v_placeholder')
            self.sumlogdet_ph = tf.placeholder(dtype=TF_FLOAT,
                                               shape=self.x.shape[0],
                                               name='sumlogdet_placeholder')

            self.state = State(x=self.x, v=self.v_ph, beta=self.beta)

            ph_str = 'energy_placeholders'
            _ = [tf.add_to_collection(ph_str, i) for i in self.state]
            tf.add_to_collection(ph_str, self.sumlogdet_ph)

            self.energy_ops = self._calc_energies(self.state,
                                                  self.sumlogdet_ph)
            #  self.energy_ops = self._build_energy_ops()

        # *******************************************************************
        # Calculate loss_op and train_op to backprop. grads through network
        # -------------------------------------------------------------------
        with tf.name_scope('calc_loss'):
            self.loss_op, self._losses_dict = self.calc_loss(x_data, z_data)
            #  self.loss_op, self.losses_dict = self.calc_loss(x_data, z_data)

        # *******************************************************************
        # Calculate gradients and build training operation
        # -------------------------------------------------------------------
        with tf.name_scope('train'):
            io.log(f'INFO: Calculating gradients for backpropagation...')
            self.grads = self._calc_grads(self.loss_op)
            self.train_op = self._apply_grads(self.loss_op, self.grads)
            self.train_ops = self._build_train_ops()
            t_ops = list(self.train_ops.values())
            _ = [tf.add_to_collection('train_ops', v) for v in t_ops]

        io.log(f'INFO: Done building graph. '
               f'Took: {time.time() - t0}s\n' + 80 * '-')

    def _double_check(self, key, params, default_val=None):
        """Check if key is in params, else, check if `self.key` is defined."""
        return params.get(key, getattr(self, key, default_val))

    def create_distribution(self):
        """Create distribution."""
        self.sigmas = self._get_sigmas()

        if self.arrangement == 'lattice':
            sigma = np.max(self.sigmas)
            #  L = int(np.sqrt(self.num_distributions))
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
            pis = distribution_arr(self.x_dim, self.num_distributions)
            distribution = GMM(means, covs, pis)

        return means, covs, pis, distribution

    def _create_means(self):
        """Create means of target distribution."""
        #  params = self.params if params is None else params
        #  diag = self._double_check('diag', params, False)
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
        #  params = self.params if params is None else params
        #  sigmas = self._double_check('sigmas', params, None)
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

    def _create_metric_fn(self):
        """Create metric fn for measuring the distance between two samples."""
        with tf.name_scope('metric_fn'):
            def metric_fn(x1, x2):
                return tf.square(x1 - x2)

        return metric_fn

    def calc_loss(self, x_data, z_data):
        """Calculate the total loss from all terms."""
        ld = {}
        total_loss = 0.

        if self.use_gaussian_loss:
            gaussian_loss = self.gaussian_loss(x_data, z_data)
            ld['gaussian'] = gaussian_loss
            total_loss += gaussian_loss

        if self.use_nnehmc_loss:
            nnehmc_loss_x = self.nnehmc_loss(x_data, self.px_hmc)
            ld['nnehmc_x'] = nnehmc_loss_x
            total_loss += nnehmc_loss_x

            #  aux_weight = getattr(self, 'aux_weight', 1.)
            #  if aux_weight > 0:
            #      pz_hmc = self._z_dynamics['accept_prob_hmc']
            #      nnehmc_loss_z = self.nnehmc_loss(z_data, pz_hmc)
            #      ld['nnehmc_z'] = nnehmc_loss_z
            #      total_loss += nnehmc_loss_z

        # If not using either Gaussian loss or NNEHMC loss, use standard loss
        if (not self.use_gaussian_loss) and (not self.use_nnehmc_loss):
            std_loss = self._calc_loss(x_data, z_data)
            ld['std'] = std_loss
            total_loss += std_loss

        tf.add_to_collection('losses', total_loss)

        fd = {k: v / total_loss for k, v in ld.items()}

        losses_dict = {}
        for key in ld.keys():
            losses_dict[key + '_loss'] = ld[key]
            losses_dict[key + '_frac'] = fd[key]

            tf.add_to_collection('losses', ld[key])

        return total_loss, losses_dict

    def gaussian_loss(self, x_data, z_data):
        """Calculate the Gaussian loss and return op. for calculating it."""
        #  mean = self._eps_np * self.num_steps
        #  sigma = np.max(self.sigmas)
        mean = 0.
        sigma = 1.
        return self._gaussian_loss(x_data, z_data, mean=mean, sigma=sigma)

    def nnehmc_loss(self, x_data, hmc_prob):
        """Calculate the NNEHMC loss via `self._nnehmc_loss` in `BaseModel`."""
        nnehmc_beta = getattr(self, 'nnehmc_beta', 1.)
        return self._nnehmc_loss(x_data, hmc_prob, beta=nnehmc_beta)

    def _parse_dynamics_output(self, dynamics_output):
        """Parse output dictionary from `self.dynamics.apply_transition."""
        if self.save_lf:
            op_keys = ['masks_f', 'masks_b',
                       'lf_out_f', 'lf_out_b',
                       'pxs_out_f', 'pxs_out_b',
                       'logdets_f', 'logdets_b',
                       'fns_out_f', 'fns_out_b',
                       'sumlogdet_f', 'sumlogdet_b']
            for key in op_keys:
                try:
                    op = dynamics_output[key]
                    setattr(self, key, op)
                except KeyError:
                    continue

        with tf.name_scope('l2hmc_fns'):
            self.l2hmc_fns = {
                'l2hmc_fns_f': self._extract_l2hmc_fns(self.fns_out_f),
                'l2hmc_fns_b': self._extract_l2hmc_fns(self.fns_out_b),
            }

    def _build_run_ops(self):
        """Build `run_ops` used for running inference w/ trained model."""
        keys = ['x_out', 'px', 'dynamics_eps']
        run_ops = {
            'x_out': self.x_out,
            'px': self.px,
            'dynamics_eps': self.dynamics.eps
        }

        if self.save_lf:
            keys = ['lf_out', 'pxs_out', 'masks',
                    'logdets', 'sumlogdet', 'fns_out']
            fkeys = [k + '_f' for k in keys]
            bkeys = [k + '_b' for k in keys]
            run_ops.update({k: getattr(self, k) for k in fkeys})
            run_ops.update({k: getattr(self, k) for k in bkeys})

        return run_ops

    def _build_train_ops(self):
        """Build `train_ops` used for training our model."""
        if self.hmc:
            train_ops = {}
        else:
            train_ops = {
                'loss_op': self.loss_op,
                'train_op': self.train_op,
                'x_out': self.x_out,
                'px': self.px,
                'dynamics_eps': self.dynamics.eps,
                'lr': self.lr
            }

        return train_ops

    def _extract_l2hmc_fns(self, fns):
        """Method for extracting each of the Q, S, T functions as tensors."""
        if not getattr(self, 'save_lf', True):
            return

        fnsT = tf.transpose(fns, perm=[2, 1, 0, 3, 4], name='fns_transposed')

        fn_names = ['scale', 'translation', 'transformation']
        update_names = ['v1', 'x1', 'x2', 'v2']

        l2hmc_fns = {}
        for idx, name in enumerate(fn_names):
            l2hmc_fns[name] = {}
            for subidx, subname in enumerate(update_names):
                l2hmc_fns[name][subname] = fnsT[idx][subidx]

        return l2hmc_fns
