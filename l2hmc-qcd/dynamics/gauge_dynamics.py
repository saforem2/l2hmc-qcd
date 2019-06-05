"""
Dynamics engine for L2HMC sampler on Lattice Gauge Models.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Author: Sam Foreman (github: @saforem2)
Date: 1/14/2019
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import numpy.random as npr
import tensorflow as tf

from globals import GLOBAL_SEED, TF_FLOAT
from network.conv_net import ConvNet2D, ConvNet3D
from network.generic_net import GenericNet

import utils.file_io as io


def exp(x, name=None):
    """Safe exponential using tf.check_numerics."""
    return tf.check_numerics(tf.exp(x), f'{name} is NaN')


def flatten_tensor(tensor):
    """Flattens tensor along axes 1:, since axis=0 indexes sample in batch.

    Example: for a tensor of shape [b, x, y, t] -->
        returns a tensor of shape [b, x * y * t]
    """
    batch_size = tensor.shape[0]
    return tf.reshape(tensor, shape=(batch_size, -1))


class GaugeDynamics(tf.keras.Model):
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
        super(GaugeDynamics, self).__init__(name='GaugeDynamics')
        #  npr.seed(np_seed)

        self.lattice = lattice
        self.potential = potential_fn
        self.batch_size = self.lattice.samples.shape[0]
        self.x_dim = self.lattice.num_links

        # create attributes from kwargs.items()
        for key, val in kwargs.items():
            if key != 'eps':  # want to use self.eps as tf.Variable
                setattr(self, key, val)

        with tf.name_scope('eps'):
            #  self.eps = exp(self.alpha, name='eps')
            self.eps = tf.Variable(
                initial_value=kwargs.get('eps', 0.4),
                name='eps',
                dtype=TF_FLOAT,
                trainable=self.eps_trainable
            )
            io.log(f'eps.dtype: {self.eps.dtype}')

        self._construct_masks()

        if self.hmc:
            self.x_fn = lambda inp: [
                tf.zeros_like(inp[0]) for t in range(3)
            ]
            self.v_fn = lambda inp: [
                tf.zeros_like(inp[0]) for t in range(3)
            ]

        else:
            if self.network_arch == 'conv3D':
                self._build_conv_nets_3D()
            elif self.network_arch == 'conv2D':
                self._build_conv_nets_2D()
            elif self.network_arch == 'generic':
                self._build_generic_nets()
            else:
                raise AttributeError("`self._network_arch` must be one of "
                                     "`'conv3D', 'conv2D', 'generic'.`")

    def _build_conv_nets_3D(self):
        """Build ConvNet3D architecture for x and v functions."""
        kwargs = {
            '_input_shape': (self.batch_size, *self.lattice.links.shape),
            'links_shape': self.lattice.links.shape,
            'x_dim': self.lattice.num_links,  # dimensionality of target space
            'factor': 2.,  # scale factor used in original paper
            'spatial_size': self.lattice.space_size,
            'num_hidden': 2 * self.lattice.num_links,
            'num_filters': int(self.lattice.space_size),
            'filter_sizes': [(3, 3, 2), (2, 2, 2)],  # size of conv. filters
            'name_scope': 'position',
            'data_format': self.data_format,  # channels_first if using GPU
            'use_bn': self.use_bn,  # whether or not to use batch normalization
        }

        with tf.name_scope("DynamicsNetwork"):
            with tf.name_scope("XNet"):
                self.x_fn = ConvNet3D(model_name='XNet', **kwargs)

            kwargs['name_scope'] = 'momentum'
            kwargs['factor'] = 1.
            with tf.name_scope("VNet"):
                self.v_fn = ConvNet3D(model_name='VNet', **kwargs)

    def _build_conv_nets_2D(self):
        """Build ConvNet architecture for x and v functions."""
        kwargs = {
            '_input_shape': (self.batch_size, *self.lattice.links.shape),
            'links_shape': self.lattice.links.shape,
            'x_dim': self.lattice.num_links,  # dimensionality of target space
            'factor': 2.,
            'spatial_size': self.lattice.space_size,
            'num_hidden': 2 * self.lattice.num_links,
            'num_filters': int(2 * self.lattice.space_size),
            'filter_sizes': [(3, 3), (2, 2)],  # for 1st and 2nd conv. layer
            'name_scope': 'position',
            'data_format': self.data_format,
        }

        with tf.name_scope("DynamicsNetwork"):
            with tf.name_scope("XNet"):
                self.x_fn = ConvNet2D(model_name='XNet', **kwargs)

            kwargs['name_scope'] = 'momentum'
            kwargs['factor'] = 1.
            with tf.name_scope("VNet"):
                self.v_fn = ConvNet2D(model_name='VNet', **kwargs)

    def _build_generic_nets(self):
        """Build GenericNet FC-architectures for x and v fns. """

        kwargs = {
            'x_dim': self.x_dim,
            'links_shape': self.lattice.links.shape,
            'num_hidden': int(4 * self.x_dim),
            'name_scope': 'position',
            'factor': 2.
        }

        with tf.name_scope("DynamicsNetwork"):
            with tf.name_scope("XNet"):
                self.x_fn = GenericNet(model_name='XNet', **kwargs)

            kwargs['factor'] = 1.
            kwargs['name_scope'] = 'momentum'
            with tf.name_scope("VNet"):
                self.v_fn = GenericNet(model_name='VNet', **kwargs)

    def call(self, x_in, beta, net_weights, save_lf=False):
        """Call method."""
        return self.apply_transition(x_in, beta, net_weights, save_lf=save_lf)

    def apply_transition(self, x_in, beta, net_weights, save_lf=False):
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
        # Use sample masks to compute the actual solutions
        with tf.name_scope('apply_transition'):
            with tf.name_scope('transition_forward'):
                outputs_f = self.transition_kernel(x_in, beta,
                                                   net_weights,
                                                   forward=True,
                                                   save_lf=save_lf)
                x_f = outputs_f['x_proposed']
                v_f = outputs_f['v_proposed']
                accept_prob_f = outputs_f['accept_prob']

                #  x_f = outputs[0]
                #  v_f = outputs[1]
                #  accept_prob_f = outputs[2]
                if save_lf:
                    lf_out_f = outputs_f['lf_out']
                    logdets_f = outputs_f['logdets']
                    sumlogdet_f = outputs_f['sumlogdet']
                    #  lf_out_f = outputs[3]
                    #  accept_probs_f = outputs[4]
                    #  logdets_f = outputs[5]
                    #  sumlogdet_f = outputs[6]

            with tf.name_scope('transition_backward'):
                outputs_b = self.transition_kernel(x_in, beta,
                                                   net_weights,
                                                   forward=False,
                                                   save_lf=save_lf)
                x_b = outputs_b['x_proposed']
                v_b = outputs_b['v_proposed']
                accept_prob_b = outputs_b['accept_prob']

                #  x_f = outputs_b[0]
                #  v_f = outputs_b[1]
                #  accept_prob_f = outputs_b[2]
                if save_lf:
                    lf_out_b = outputs_b['lf_out']
                    logdets_b = outputs_b['logdets']
                    sumlogdet_b = outputs_b['sumlogdet']
                #  x_b = outputs[0]
                #  v_b = outputs[1]
                #  accept_prob_b = outputs[2]
                #  if save_lf:
                    #  lf_out_b = outputs[3]
                    #  accept_probs_b = outputs[4]
                    #  logdets_b = outputs[5]
                    #  sumlogdet_b = outputs[6]

            # Decide direction uniformly
            with tf.name_scope('transition_masks'):
                forward_mask = tf.cast(
                    tf.random_uniform((self.batch_size,),
                                      seed=GLOBAL_SEED) > 0.5,
                    TF_FLOAT,
                    name='forward_mask'
                )
                backward_mask = 1. - forward_mask

            # Obtain proposed states
            with tf.name_scope('x_proposed'):
                x_proposed = (forward_mask[:, None] * x_f
                              + backward_mask[:, None] * x_b)

            with tf.name_scope('v_proposed'):
                v_proposed = (forward_mask[:, None] * v_f
                              + backward_mask[:, None] * v_b)

            # Probability of accepting the proposed states
            with tf.name_scope('accept_prob'):
                accept_prob = (forward_mask * accept_prob_f
                               + backward_mask * accept_prob_b)

            # Accept or reject step
            with tf.name_scope('accept_mask'):
                accept_mask = tf.cast(
                    accept_prob > tf.random_uniform(tf.shape(accept_prob),
                                                    seed=GLOBAL_SEED),
                    TF_FLOAT,
                    name='acccept_mask'
                )
                reject_mask = 1. - accept_mask

            # Samples after accept / reject step
            with tf.name_scope('x_out'):
                x_out = (
                    accept_mask[:, None] * x_proposed
                    + reject_mask[:, None] * x_in
                )

        outputs = {
            'x_proposed': x_proposed,
            'v_proposed': v_proposed,
            'accept_prob': accept_prob,
            'x_out': x_out
        }

        if save_lf:
            outputs['lf_out_f'] = lf_out_f
            outputs['accept_probs_f'] = accept_prob_f
            outputs['lf_out_b'] = lf_out_b
            outputs['accept_probs_b'] = accept_prob_b
            outputs['forward_mask'] = forward_mask
            outputs['backward_mask'] = backward_mask
            outputs['logdets_f'] = logdets_f
            outputs['logdets_b'] = logdets_b
            outputs['sumlogdet_f'] = sumlogdet_f
            outputs['sumlogdet_b'] = sumlogdet_b

        return outputs

    def transition_kernel(self, x_in, beta, net_weights,
                          forward=True, save_lf=False):
        """Transition kernel of augmented leapfrog integrator."""
        with tf.name_scope('determine_direction'):
            lf_fn = self._forward_lf if forward else self._backward_lf

        with tf.name_scope('refresh_momentum'):
            v_in = tf.random_normal(tf.shape(x_in), seed=GLOBAL_SEED)

        with tf.name_scope('init'):
            x_proposed, v_proposed = x_in, v_in

            t = tf.constant(0., name='md_time', dtype=TF_FLOAT)
            batch_size = tf.shape(x_in)[0]
            logdet = tf.zeros((batch_size,))
            lf_out = tf.TensorArray(dtype=TF_FLOAT, size=self.num_steps,
                                    dynamic_size=True, name='lf_out',
                                    clear_after_read=False)
            logdets = tf.TensorArray(dtype=TF_FLOAT, size=self.num_steps,
                                     dynamic_size=True, name='logdets_out',
                                     clear_after_read=False)

        def body(x, v, beta, t, logdet, lf_out, logdets):
            i = tf.cast(t, dtype=tf.int32)
            with tf.name_scope('apply_lf'):
                new_x, new_v, j = lf_fn(x, v, beta, t, net_weights)
            with tf.name_scope('concat_lf_outputs'):
                lf_out = lf_out.write(i, new_x)
            with tf.name_scope('concat_logdets'):
                logdets = logdets.write(i, j)
            return new_x, new_v, beta, t + 1, logdet + j, lf_out, logdets

        def cond(x, v, beta, t, logdet, lf_out, logdets):
            with tf.name_scope('check_lf_step'):
                return tf.less(t, self.num_steps)

        with tf.name_scope('while_loop'):
            outputs = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[x_proposed, v_proposed,
                           beta, t, logdet, lf_out, logdets])

            x_proposed = outputs[0]
            v_proposed = outputs[1]
            beta = outputs[2]
            t = outputs[3]
            sumlogdet = outputs[4]
            lf_out = outputs[5].stack()
            logdets = outputs[6].stack()

        with tf.name_scope('accept_prob'):
            accept_prob = self._compute_accept_prob(
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
            'accept_prob': accept_prob,
        }

        if save_lf:
            outputs['lf_out'] = lf_out
            outputs['logdets'] = logdets
            outputs['sumlogdet'] = sumlogdet

        return outputs

    def _forward_lf(self, x, v, beta, step, net_weights):
        """One forward augmented leapfrog step."""
        with tf.name_scope('forward_lf'):
            with tf.name_scope('format_time'):
                t = self._format_time(step, tile=tf.shape(x)[0])
            with tf.name_scope('get_mask'):
                mask, mask_inv = self._get_mask_while(step)

            with tf.name_scope('augmented_leapfrog'):
                sumlogdet = 0.

                v, logdet = self._update_v_forward(x, v, beta, t, net_weights)
                sumlogdet += logdet

                x, logdet = self._update_x_forward(x, v, t, net_weights,
                                                   mask, mask_inv)
                sumlogdet += logdet

                x, logdet = self._update_x_forward(x, v, t, net_weights,
                                                   mask_inv, mask)
                sumlogdet += logdet

                v, logdet = self._update_v_forward(x, v, beta, t, net_weights)
                sumlogdet += logdet

        return x, v, sumlogdet

    def _backward_lf(self, x, v, beta, step, net_weights):
        """One backward augmented leapfrog step."""

        # Reversed index/sinusoidal time
        with tf.name_scope('backward_lf'):
            with tf.name_scope('format_time'):
                t = self._format_time(self.num_steps - step - 1,
                                      tile=tf.shape(x)[0])
            with tf.name_scope('get_mask'):
                mask, mask_inv = self._get_mask_while(
                    self.num_steps - step - 1
                )

            with tf.name_scope('augmented_leapfrog'):
                sumlogdet = 0.

                v, logdet = self._update_v_backward(x, v, beta, t, net_weights)
                sumlogdet += logdet

                x, logdet = self._update_x_backward(x, v, t, net_weights,
                                                    mask_inv, mask)
                sumlogdet += logdet

                x, logdet = self._update_x_backward(x, v, t, net_weights,
                                                    mask, mask_inv)
                sumlogdet += logdet

                v, logdet = self._update_v_backward(x, v, beta, t, net_weights)
                sumlogdet += logdet

        return x, v, sumlogdet

    def _update_v_forward(self, x, v, beta, t, net_weights):
        """Update v in the forward leapfrog step."""
        with tf.name_scope('update_v_forward'):
            with tf.name_scope('grad_potential'):
                grad = self.grad_potential(x, beta)

            with tf.name_scope('v_fn'):
                # Sv: scale, Qv: transformation, Tv: translation
                scale, translation, transformed = self.v_fn(
                    [x, grad, t, net_weights]
                )

            with tf.name_scope('scale'):
                scale *= 0.5 * self.eps

            with tf.name_scope('transformed'):
                transformed *= self.eps

            with tf.name_scope('v_update'):
                v = (v * exp(scale, 'vf_scale')
                     - (0.5 * self.eps
                        * (exp(transformed, name='vf_transformed')
                           * grad + translation)))

        return v, tf.reduce_sum(scale, axis=1)

    def _update_x_forward(self, x, v, t, net_weights, mask, mask_inv):
        """Update x in the forward leapfrog step."""
        with tf.name_scope('update_x_forward'):
            with tf.name_scope('x_fn'):
                scale, translation, transformed = self.x_fn(
                    [v, mask * x, t, net_weights]
                )

            with tf.name_scope('scale'):
                scale *= self.eps

            with tf.name_scope('transformed'):
                transformed *= self.eps

            with tf.name_scope('x_update'):
                x = (mask * x + mask_inv
                     * (x * exp(scale, 'xf_scale') + self.eps
                        * (exp(transformed, 'xf_transformed')
                           * v + translation)))

        return x, tf.reduce_sum(mask_inv * scale, axis=1)

    def _update_v_backward(self, x, v, beta, t, net_weights):
        """Update v in the backward leapfrog step. Invert the forward update"""
        with tf.name_scope('update_v_backward'):
            with tf.name_scope('grad_potential'):
                grad = self.grad_potential(x, beta)

            with tf.name_scope('v_fn'):
                scale, translation, transformed = self.v_fn(
                    [x, grad, t, net_weights]
                )

            with tf.name_scope('scale'):
                scale *= -0.5 * self.eps

            with tf.name_scope('transformed'):
                transformed *= self.eps

            with tf.name_scope('v_update'):
                v = (exp(scale, 'vb_scale')
                     * (v + 0.5 * self.eps
                        * (exp(transformed, 'vb_transformed')
                           * grad + translation)))

        return v, tf.reduce_sum(scale, axis=1)

    def _update_x_backward(self, x, v, t, net_weights, mask, mask_inv):
        """Update x in the backward lf step. Inverting the forward update."""
        with tf.name_scope('update_x_backward'):
            with tf.name_scope('x_fn'):
                scale, translation, transformed = self.x_fn(
                    [v, mask * x, t, net_weights]
                )

            with tf.name_scope('scale'):
                scale *= -self.eps

            with tf.name_scope('transformed'):
                transformed *= self.eps

            with tf.name_scope('x_update'):
                x = (mask * x + mask_inv * exp(scale, 'xb_scale')
                     * (x - self.eps
                        * (exp(transformed, 'xb_transformed')
                           * v + translation)))

        return x, tf.reduce_sum(mask_inv * scale, axis=1)

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
        return tf.where(tf.is_finite(prob), prob, tf.zeros_like(prob))

    def _format_time(self, i, tile=1):
        """Format time as [cos(..), sin(...)]."""
        with tf.name_scope('format_time'):
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

    def _get_mask_while(self, step):
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
