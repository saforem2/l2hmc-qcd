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
#  from network.conv_net2d import ConvNet2D
#  from network.generic_net import GenericNet
#  from network.old.conv_net3d import FullNet3D
from network.conv_net3d_old import ConvNet3D
from network.network import FullNet
#  from network.conv_net3d import ConvNet3D


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


def _add_to_collection(collection, ops):
    if len(ops) > 1:
        [tf.add_to_collection(collection, op) for op in ops]
    else:
        tf.add_to_collection(collection, ops)


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
        npr.seed(GLOBAL_SEED)

        self.lattice = lattice
        self.potential = potential_fn
        self.batch_size = self.lattice.samples.shape[0]
        self.x_dim = self.lattice.num_links

        # create attributes from kwargs.items()
        for key, val in kwargs.items():
            if key != 'eps':  # want to use self.eps as tf.Variable
                setattr(self, key, val)

        #  _eps_np = kwargs.get('eps', 0.4)
        #  with tf.variable_scope(reus):
        #  self.eps = tf.get_variable('eps', dtype=TF_FLOAT,
        #                             trainable=self.eps_trainable,
        #                             initializer=tf.constant(_eps_np))

        #  self.log_eps = tf.Variable(
        #      initial_value=tf.log(tf.constant(_eps_np)),
        #      name='log_eps',
        #      dtype=TF_FLOAT,
        #      trainable=self.eps_trainable
        #  )
        #
        #  self.eps = tf.exp(self.log_eps, name='eps')
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
            with tf.name_scope("XNet"):
                import utils.file_io as io
                io.log(f'Using `FullNet` architecture...')
                self.x_fn = FullNet(model_name='XNet', **net_kwargs)

            net_kwargs['name_scope'] = 'v'  # update name scope
            net_kwargs['factor'] = 1.              # factor used in orig. paper
            with tf.name_scope("VNet"):
                self.v_fn = FullNet(model_name='VNet', **net_kwargs)

    def call(self, *args, **kwargs):
        """Call method."""
        #  def call(self, x_in, beta, net_weights, train_phase, save_lf=False):
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
            with tf.name_scope('transition_forward'):
                outputs_f = self.transition_kernel(x_in, beta,
                                                   net_weights,
                                                   train_phase,
                                                   forward=True,
                                                   save_lf=save_lf)
                xf = outputs_f['x_proposed']
                vf = outputs_f['v_proposed']
                pxs_out_f = outputs_f['accept_prob']

                if save_lf:
                    lf_out_f = outputs_f['lf_out']
                    logdets_f = outputs_f['logdets']
                    sumlogdet_f = outputs_f['sumlogdet']

            with tf.name_scope('transition_backward'):
                outputs_b = self.transition_kernel(x_in, beta,
                                                   net_weights,
                                                   train_phase,
                                                   forward=False,
                                                   save_lf=save_lf)
                xb = outputs_b['x_proposed']
                vb = outputs_b['v_proposed']
                pxs_out_b = outputs_b['accept_prob']

                if save_lf:
                    lf_out_b = outputs_b['lf_out']
                    logdets_b = outputs_b['logdets']
                    sumlogdet_b = outputs_b['sumlogdet']

            # Decide direction uniformly
            with tf.name_scope('transition_masks'):
                masks_f = tf.cast(
                    tf.random_uniform((self.batch_size,),
                                      seed=GLOBAL_SEED) > 0.5,
                    TF_FLOAT,
                    name='forward_mask'
                )
                masks_b = 1. - masks_f

            # Obtain proposed states
            with tf.name_scope('x_proposed'):
                x_proposed = (masks_f[:, None] * xf
                              + masks_b[:, None] * xb)

            with tf.name_scope('v_proposed'):
                v_proposed = (masks_f[:, None] * vf
                              + masks_b[:, None] * vb)

            # Probability of accepting the proposed states
            with tf.name_scope('accept_prob'):
                accept_prob = (masks_f * pxs_out_f
                               + masks_b * pxs_out_b)

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
                x_out = (accept_mask[:, None] * x_proposed
                         + reject_mask[:, None] * x_in)

        outputs = {
            'x_proposed': x_proposed,
            'v_proposed': v_proposed,
            'accept_prob': accept_prob,
            'x_out': x_out
        }

        if save_lf:
            outputs['masks_f'] = masks_f
            outputs['masks_b'] = masks_b

            outputs['lf_out_f'] = lf_out_f
            outputs['lf_out_b'] = lf_out_b

            outputs['pxs_out_f'] = pxs_out_f
            outputs['pxs_out_b'] = pxs_out_b

            outputs['logdets_f'] = logdets_f
            outputs['logdets_b'] = logdets_b

            outputs['sumlogdet_f'] = sumlogdet_f
            outputs['sumlogdet_b'] = sumlogdet_b

        return outputs

    def transition_kernel(self, x_in, beta, net_weights,
                          train_phase, forward=True, save_lf=False):
        """Transition kernel of augmented leapfrog integrator."""
        lf_fn = self._forward_lf if forward else self._backward_lf

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
            new_x, new_v, j = lf_fn(x, v, beta, step,
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

            #  x_proposed = outputs[0]
            #  v_proposed = outputs[1]
            #  t = outputs[2]
            #  sumlogdet = outputs[3]
            #  lf_out = outputs[4].stack()
            #  logdets_out = outputs[5].stack()

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
            'sumlogdet': sumlogdet,
            'accept_prob': accept_prob,
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

    def _backward_lf(self, x, v, beta, step, net_weights, train_phase):
        """One backward augmented leapfrog step."""
        # Reversed index/sinusoidal time
        with tf.name_scope('backward_lf'):
            with tf.name_scope('get_time'):
                t = self._get_time(self.num_steps - step - 1,
                                   tile=tf.shape(x)[0])
            with tf.name_scope('get_mask'):
                mask, mask_inv = self._get_mask(
                    self.num_steps - step - 1
                )

            with tf.name_scope('augmented_leapfrog'):
                sumlogdet = 0.

                v, logdet = self._update_v_backward(x, v, beta, t,
                                                    net_weights, train_phase)
                sumlogdet += logdet

                x, logdet = self._update_x_backward(x, v, t,
                                                    net_weights, train_phase,
                                                    mask_inv, mask)
                sumlogdet += logdet

                x, logdet = self._update_x_backward(x, v, t,
                                                    net_weights, train_phase,
                                                    mask, mask_inv)
                sumlogdet += logdet

                v, logdet = self._update_v_backward(x, v, beta, t,
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

    def _update_v_backward(self, x, v, beta, t, net_weights, train_phase):
        """Update v in the backward leapfrog step. Invert the forward update"""
        with tf.name_scope('update_v_backward'):
            with tf.name_scope('grad_potential'):
                grad = self.grad_potential(x, beta)

            with tf.name_scope('call_vb'):
                scale, translation, transformation = self.v_fn(
                    [x, grad, t], train_phase
                )

            with tf.name_scope('net_weights_mult'):
                scale *= net_weights[0]
                translation *= net_weights[1]
                transformation *= net_weights[2]

            with tf.name_scope('scale_exp'):
                scale *= -0.5 * self.eps
                scale_exp = exp(scale, 'vb_scale')

            with tf.name_scope('transformation'):
                transformation *= self.eps
                transformation_exp = exp(transformation,
                                         'vb_transformation')

            with tf.name_scope('v_update'):
                v = scale_exp * (
                    v + 0.5 * self.eps * (
                        grad * transformation_exp - translation
                    )
                )

            with tf.name_scope('logdet_vb'):
                logdet = tf.reduce_sum(scale, axis=1, name='logdet_vb')

        return v, logdet

    def _update_x_backward(self, x, v, t, net_weights, 
                           train_phase, mask, mask_inv):
        """Update x in the backward lf step. Inverting the forward update."""
        with tf.name_scope('update_x_backward'):
            with tf.name_scope('call_xb'):
                scale, translation, transformation = self.x_fn(
                    [v, mask * x, t], train_phase
                )

            with tf.name_scope('net_weights_mult'):
                scale *= net_weights[0]
                translation *= net_weights[1]
                transformation *= net_weights[2]

            with tf.name_scope('scale_exp'):
                scale *= -self.eps
                scale_exp = exp(scale, 'xb_scale')

            with tf.name_scope('transformation_exp'):
                transformation *= self.eps
                transformation_exp = exp(transformation,
                                         'xb_transformation')

            with tf.name_scope('x_update'):
                x = (mask * x + mask_inv * scale_exp
                     * (x - self.eps * (v * transformation_exp + translation)))

            #  return x, tf.reduce_sum(mask_inv * scale, axis=1)
            with tf.name_scope('logdet_xb'):
                logdet = tf.reduce_sum(mask_inv * scale, axis=1,
                                       name='logdet_xb')

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
        return tf.where(tf.is_finite(prob), prob, tf.zeros_like(prob))

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
