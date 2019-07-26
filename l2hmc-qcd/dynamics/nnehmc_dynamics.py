"""
Dynamics engine for Neural Network Enhanced Hamiltonian Monte Carlo sampler on
Lattice Gauge Models.

Reference (Robust Biophysical Parameter Estimation with a Neural Network
Enhanced Hamiltonian Markov Chain Monte Carlo Sampler)
[https://infoscience.epfl.ch/record/264887/files/robust_parameter_estimation.pdf]

Author: Sam Foreman (github: @saforem2)
Date: 7/23/2019
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import numpy.random as npr
import tensorflow as tf

from variables import GLOBAL_SEED, TF_FLOAT
from network.network import FullNet


def exp(x, name=None):
    """Safe exponential using tf.check_numerics."""
    return tf.check_numerics(tf.exp(x), f'{name} is NaN')


class nnehmcDynamics(tf.keras.Model):
    """Dynamics engine of naive L2HMC sampler."""

    def __init__(self, lattice, potential_fn, **kwargs):
        """Initialization.

        Args:
            lattice: Lattice object containing multiple sample lattices.
            potential_fn: Function specifying minus log-likelihood objective
            to minimize.

        NOTE: kwargs (expected)
            num_steps: Number of leapfrog steps to use in integrator.
            eps: Initial step size to use in leapfrog integrator.
            network_arch: String specifying network architecture to use.
                Must be one of `'conv2D', 'conv3D', 'generic'`. Networks
                are defined in `../network/`
            hmc: Flag indicating whether generic HMC (no augmented
                leapfrog) should be used instead of L2HMC. Defaults to
                False.
            eps_trainable: Flag indiciating whether the step size (eps)
                should be trainable. Defaults to True.
            np_seed: Seed to use for numpy.random.
        """
        super(nnehmcDynamics, self).__init__(name='nnehmcDynamics')
        npr.seed(GLOBAL_SEED)

        self.lattice = lattice
        self.potential = potential_fn
        self.batch_size = self.lattice.samples.shape[0]
        self.x_dim = self.lattice.num_links

        # create attributes from kwargs.items()
        for key, val in kwargs.items():
            if key != 'eps':  # want to use self.eps as tf.Variable
                setattr(self, key, val)

        self.eps = tf.Variable(
            initial_value=kwargs.get('eps', 0.4),
            name='eps',
            dtype=TF_FLOAT,
            trainable=self.eps_trainable
        )

        self._construct_masks()

        if self.hmc:
            self.x_fn = lambda inp: [
                tf.zeros_like(inp[0]) for t in range(3)
            ]
            self.v_fn = lambda inp: [
                tf.zeros_like(inp[0]) for t in range(3)
            ]

        else:
            num_filters = int(self.lattice.space_size)
            net_kwargs = {
                'network_arch': self.network_arch,
                'use_bn': self.use_bn,  # whether or not to use batch norm
                'dropout_prob': self.dropout_prob,
                'x_dim': self.lattice.num_links,  # dim of target space
                'links_shape': self.lattice.links.shape,
                'num_hidden': self.num_hidden,
                'num_filters': [num_filters, int(2 * num_filters)],
                'name_scope': 'x',  # namespace in which to create network
                'factor': 2.,  # scale factor used in original paper
                '_input_shape': (self.batch_size, *self.lattice.links.shape),
                'data_format': self.data_format,
            }

            if self.network_arch == 'conv3D':
                net_kwargs.update({
                    'filter_sizes': [(3, 3, 1), (2, 2, 1)],
                })
                #  self._build_conv_nets_3D(net_kwargs)
            elif self.network_arch == 'conv2D':
                #  'num_filters': int(2 * self.lattice.space_size),
                net_kwargs.update({
                    'filter_sizes': [(3, 3), (2, 2)],
                })
                #  self._build_conv_nets_2D(net_kwargs)

            self.build_network(net_kwargs)

    def build_network(self, net_kwargs):
        """Build neural network used to train model."""
        with tf.name_scope("DynamicsNetwork"):
            self.x_fn = FullNet(model_name='XNet', **net_kwargs)

            net_kwargs['name_scope'] = 'v'  # update name scope
            net_kwargs['factor'] = 1.       # factor used in orig. paper
            self.v_fn = FullNet(model_name='VNet', **net_kwargs)

    def call(self, *args, **kwargs):
        """Call method."""
        return self.apply_transition(*args, **kwargs)

    def apply_transition(self, x_in, beta, net_weights,
                         train_phase, save_lf=False):
        """Propose a new state and perform the accept/reject step.

        Args:
            x: Batch of (x) samples (batch of links).
            beta (float): Inverse coupling constant.

        Returns:
            x_proposed: Proposed x before accept/reject step.
            v_proposed: Proposed v before accept/reject step.
            accept_prob: Probability of accepting the proposed states.
            x_out: Samples after accept/reject step.
        """
        # Simulate dynamics both forward and backward
        # Use sampled masks to compute the actual solutions
        with tf.name_scope('apply_transition'):
            outputs = self.transition_kernel(x_in, beta,
                                             net_weights,
                                             train_phase,
                                             save_lf=save_lf)
            x_proposed = outputs['x_proposed']
            v_proposed = outputs['v_proposed']
            pxs_out = outputs['accept_prob']

            if save_lf:
                lf_out = outputs['lf_out']
                logdets = outputs['logdets']
                sumlogdet = outputs['sumlogdet']

            # Accept or reject step
            with tf.name_scope('accept_mask'):
                accept_mask = tf.cast(
                    pxs_out > tf.random_uniform(tf.shape(pxs_out),
                                                seed=GLOBAL_SEED),
                    TF_FLOAT,
                    name='acccept_mask'
                )
                reject_mask = 1. - accept_mask

            # Samples after accept / reject step
            with tf.name_scope('x_out'):
                x_out = (accept_mask[:, None] * x_proposed
                         + reject_mask[:, None] * x_in)

        outputs = {
            'x_proposed': x_proposed,
            'v_proposed': v_proposed,
            'accept_prob': pxs_out,
            'x_out': x_out,
            'old_hamil': outputs['old_hamil'],
            'new_hamil': outputs['new_hamil']
        }

        if save_lf:
            outputs['lf_out_f'] = lf_out

            outputs['pxs_out_f'] = pxs_out

            outputs['logdets_f'] = logdets

            outputs['sumlogdet_f'] = sumlogdet

        return outputs

    def transition_kernel(self, x_in, beta, net_weights,
                          train_phase, save_lf=False):
        """Transition kernel of augmented leapfrog integrator."""
        #  lf_fn = self._forward_lf if forward else self._backward_lf

        with tf.name_scope('refresh_momentum'):
            v_in = tf.random_normal(tf.shape(x_in), seed=GLOBAL_SEED)

        with tf.name_scope('init'):
            x_proposed, v_proposed = x_in, v_in

            step = tf.constant(0., name='md_step', dtype=TF_FLOAT)
            batch_size = tf.shape(x_in)[0]
            logdet = tf.zeros((batch_size,))
            lf_out = tf.TensorArray(dtype=TF_FLOAT, size=self.num_steps+1,
                                    dynamic_size=True, name='lf_out',
                                    clear_after_read=False)
            logdets_out = tf.TensorArray(dtype=TF_FLOAT, size=self.num_steps+1,
                                         dynamic_size=True, name='logdets_out',
                                         clear_after_read=False)

            lf_out = lf_out.write(0, x_in)
            logdets_out = logdets_out.write(0, logdet)

        def body(step, x, v, logdet, lf_samples, logdets):
            # cast leapfrog step to integer
            i = tf.cast(step, dtype=tf.int32)
            new_x, new_v, j = self._forward_lf(x, v, beta, step,
                                               net_weights, train_phase)
            lf_samples = lf_samples.write(i+1, new_x)
            logdets = logdets.write(i+1, logdet+j)

            return (step+1, new_x, new_v, logdet+j, lf_samples, logdets)

        def cond(step, *args):
            return tf.less(step, self.num_steps)

        with tf.name_scope('MD_leapfrog'):
            outputs = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[step, x_proposed, v_proposed,
                           logdet, lf_out, logdets_out])

            step = outputs[0]
            x_proposed = outputs[1]
            v_proposed = outputs[2]
            sumlogdet = outputs[3]
            lf_out = outputs[4].stack()
            logdets_out = outputs[5].stack()

        with tf.name_scope('accept_prob'):
            accept_prob_output = self._compute_accept_prob(
                x_in,
                v_in,
                x_proposed,
                v_proposed,
                sumlogdet,
                beta
            )

        outputs = {
            'x_proposed': x_proposed,
            'v_proposed': v_proposed,
            'sumlogdet': sumlogdet,
            'accept_prob': accept_prob_output['accept_prob'],
            'old_hamil': accept_prob_output['old_hamil'],
            'new_hamil': accept_prob_output['new_hamil']
        }

        if save_lf:
            outputs['lf_out'] = lf_out
            outputs['logdets'] = logdets_out

        return outputs

    def _forward_lf(self, x, v, beta, step, net_weights, train_phase):
        """One forward augmented leapfrog step."""
        with tf.name_scope('forward_lf'):
            with tf.name_scope('get_time'):
                t = self._get_time(step, tile=tf.shape(x)[0])
            with tf.name_scope('get_mask'):
                mask, mask_inv = self._get_mask(step)

            with tf.name_scope('augmented_leapfrog'):
                sumlogdet = 0.

                v, logdet = self._update_v_forward(x, v, beta, t,
                                                   net_weights, train_phase)
                sumlogdet += logdet

                x, logdet = self._update_x_forward(x, v, t,
                                                   net_weights, train_phase,
                                                   mask, mask_inv)
                sumlogdet += logdet

                x, logdet = self._update_x_forward(x, v, t,
                                                   net_weights, train_phase,
                                                   mask_inv, mask)
                sumlogdet += logdet

                v, logdet = self._update_v_forward(x, v, beta, t,
                                                   net_weights, train_phase)
                sumlogdet += logdet

        return x, v, sumlogdet

    def _update_v_forward(self, x, v, beta, t, net_weights, train_phase):
        """Update v in the forward leapfrog step.

        Args:
            x: input position tensor
            v: input momentum tensor
            beta: inverse coupling constant
            t: current leapfrog step
            net_weights: Placeholders for the multiplicative weights by which
                to multiply the S, Q, and T functions (scale, transformation,
                translation resp.)
        Returns:
            v: Updated (output) momentum
            logdet: Jacobian factor
        """
        with tf.name_scope('update_v_forward'):
            with tf.name_scope('grad_potential'):
                grad = self.grad_potential(x, beta)

            # Sv: scale, Qv: transformation, Tv: translation
            with tf.name_scope('call_vf'):
                scale, translation, transformation = self.v_fn((x, grad, t),
                                                               train_phase)

            with tf.name_scope('net_weights_mult'):
                scale *= net_weights[0]
                translation *= net_weights[1]
                transformation *= net_weights[2]

            with tf.name_scope('scale_exp'):
                scale *= 0.5 * self.eps
                scale_exp = exp(scale, 'vf_scale')

            with tf.name_scope('transformation'):
                transformation *= self.eps
                transformation_exp = exp(transformation, 'vf_transformation')

            with tf.name_scope('v_update'):
                v = (v * scale_exp
                     - 0.5 * self.eps * (grad * transformation_exp
                                         - translation))

            with tf.name_scope('logdet_vf'):
                logdet = tf.reduce_sum(scale, axis=1, name='logdet_vf')

        return v, logdet

    def _update_x_forward(self, x, v, t, net_weights, 
                          train_phase, mask, mask_inv):
        """Update x in the forward leapfrog step."""
        with tf.name_scope('update_x_forward'):
            with tf.name_scope('call_xf'):
                scale, translation, transformation = self.x_fn(
                    [v, mask * x, t], train_phase
                )

            with tf.name_scope('net_weights_mult'):
                scale *= net_weights[0]
                translation *= net_weights[1]
                transformation *= net_weights[2]

            with tf.name_scope('scale_exp'):
                scale *= self.eps
                scale_exp = exp(scale, 'xf_scale')

            with tf.name_scope('transformation_exp'):
                transformation *= self.eps
                transformation_exp = exp(transformation,
                                         'xf_transformation')

            with tf.name_scope('x_update'):
                x = (mask * x
                     + mask_inv * (x * scale_exp + self.eps
                                   * (v * transformation_exp + translation)))

            #  return x, tf.reduce_sum(mask_inv * scale, axis=1)
            with tf.name_scope('logdet_xf'):
                logdet = tf.reduce_sum(mask_inv * scale, axis=1,
                                       name='logdet_xf')

        return x, logdet

    def _compute_accept_prob(self, xi, vi, xf, vf, sumlogdet, beta):
        """Compute the prob of accepting the proposed state given old state.
        Args:
            xi: Initial state.
            vi: Initial v.
            xf: Proposed state.
            vf: Proposed v.
            sumlogdet: Sum of the terms of the log of the determinant. 
                (Eq. 14 of original paper).
            beta: Inverse coupling constant of gauge model.
        """
        with tf.name_scope('compute_accept_prob'):
            with tf.name_scope('old_hamiltonian'):
                old_hamil = self.hamiltonian(xi, vi, beta)
            with tf.name_scope('new_hamiltonian'):
                new_hamil = self.hamiltonian(xf, vf, beta)

            with tf.name_scope('prob'):
                prob = exp(tf.minimum(
                    (old_hamil - new_hamil + sumlogdet), 0.
                ), 'accept_prob')

        # Ensure numerical stability as well as correct gradients
        #  return tf.where(tf.is_finite(prob), prob, tf.zeros_like(prob))
        accept_prob = tf.where(tf.is_finite(prob), prob, tf.zeros_like(prob))

        output = {
            'accept_prob': accept_prob,
            'old_hamil': old_hamil,
            'new_hamil': new_hamil
        }

        return output

    def _get_time(self, i, tile=1):
        """Format time as [cos(..), sin(...)]."""
        with tf.name_scope('get_time'):
            trig_t = tf.squeeze([
                tf.cos(2 * np.pi * i / self.num_steps),
                tf.sin(2 * np.pi * i / self.num_steps),
            ])

        return tf.tile(tf.expand_dims(trig_t, 0), (tile, 1))

    def _construct_masks(self):
        """Construct different binary masks for different time steps."""
        self.masks = []
        for _ in range(self.num_steps):
            # Need to use npr here because tf would generate different random
            # values across different `sess.run`
            idx = npr.permutation(np.arange(self.x_dim))[:self.x_dim // 2]
            mask = np.zeros((self.x_dim,))
            mask[idx] = 1.
            mask = tf.constant(mask, dtype=TF_FLOAT)
            self.masks.append(mask[None, :])

    def _get_mask(self, step):
        with tf.name_scope('get_mask'):
            m = tf.gather(self.masks, tf.cast(step, dtype=tf.int32))
        return m, 1. - m

    def potential_energy(self, x, beta):
        """Compute potential energy using `self.potential` and beta."""
        with tf.name_scope('potential_energy'):
            potential_energy = tf.multiply(beta, self.potential(x))

        return potential_energy

    def kinetic_energy(self, v):
        """Compute the kinetic energy."""
        with tf.name_scope('kinetic_energy'):
            kinetic_energy = 0.5 * tf.reduce_sum(v**2, axis=1)

        return kinetic_energy

    def hamiltonian(self, x, v, beta):
        """Compute the overall Hamiltonian."""
        with tf.name_scope('hamiltonian'):
            with tf.name_scope('potential'):
                potential = self.potential_energy(x, beta)
            with tf.name_scope('kinetic'):
                kinetic = self.kinetic_energy(v)
            with tf.name_scope('hamiltonian'):
                hamiltonian = potential + kinetic

        return hamiltonian

    def grad_potential(self, x, beta):
        """Get gradient of potential function at current location."""
        with tf.name_scope('grad_potential'):
            if tf.executing_eagerly():
                tfe = tf.contrib.eager
                grad_fn = tfe.gradients_function(self.potential_energy,
                                                 params=["x"])
                grad = grad_fn(x, beta)[0]
            else:
                grad = tf.gradients(self.potential_energy(x, beta), x)[0]
        return grad
