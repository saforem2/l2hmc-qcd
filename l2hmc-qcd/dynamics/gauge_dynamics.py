"""
Dynamics engine for L2HMC sampler on Lattice Gauge Models.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Author: Sam Foreman (github: @saforem2)
Date: 1/14/2019
"""
# pylint: disable=invalid-name
from __future__ import absolute_import, print_function, division

import numpy as np
import numpy.random as npr
import tensorflow as tf

from globals import GLOBAL_SEED, TF_FLOAT
from network.conv_net import ConvNet2D, ConvNet3D
from network.generic_net import GenericNet
#  from lattice.lattice import GaugeLattice
#  import utils.file_io as io


###########################################
# Set global seed for numpy and tensorflow
###########################################
np.random.seed(GLOBAL_SEED)

if '2' not in tf.__version__:
    tf.set_random_seed(GLOBAL_SEED)


# pylint:disable=invalid-name
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

#  def _construct_masks(self):
#      """Construct masks to determine which indices to update."""
#      mask_per_step = []
#      with tf.name_scope('construct_masks'):
#          for _ in range(self.num_steps):
#              idx = npr.permutation(np.arange(self.x_dim))[:self.x_dim //
#              2]
#              mask = np.zeros((self.x_dim,))
#              mask[idx] = 1
#              mask_per_step.append(mask)
#
#          self.masks = tf.constant(np.stack(mask_per_step),
#          dtype=TF_FLOAT)
#

#  def _get_time(self, i):
#      """Get sinusoidal time for i-th augmented leapfrog step."""
#      return self.ts[i]

#  def _get_mask(self, i):
#      """Get binary masks for i-th augmented leapfrog step."""
#      m = self.masks[i]
#      return m, 1. - m

#  with tf.name_scope('alpha'):
#      if not self.hmc:
#          self.alpha = tf.get_variable(
#              'alpha',
#              initializer=tf.log(tf.constant(kwargs.get('eps', 0.1))),
#              trainable=self.eps_trainable,
#              dtype=TF_FLOAT
#          )
#      else:
#          self.alpha = tf.log(tf.constant(kwargs.get('eps', 0.1),
#                                          dtype=TF_FLOAT))


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

        self._construct_time()
        self._construct_masks()

        if self.hmc:
            self.position_fn = lambda inp: [
                tf.zeros_like(inp[0]) for t in range(3)
            ]
            self.momentum_fn = lambda inp: [
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
        """Build ConvNet3D architecture for position and momentum functions."""
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
            'scale_weight': 1.,     # multiplicative factor multiplying
            'translation_weight': 1.,
            'transformation_weight': 1.,
        }

        with tf.name_scope("DynamicsNetwork"):
            with tf.name_scope("XNet"):
                self.position_fn = ConvNet3D(model_name='XNet', **kwargs)

            kwargs['name_scope'] = 'momentum'
            kwargs['factor'] = 1.
            with tf.name_scope("VNet"):
                self.momentum_fn = ConvNet3D(model_name='VNet', **kwargs)

    def _build_conv_nets_2D(self):
        """Build ConvNet architecture for position and momentum functions."""
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
                self.position_fn = ConvNet2D(model_name='XNet', **kwargs)

            kwargs['name_scope'] = 'momentum'
            kwargs['factor'] = 1.
            with tf.name_scope("VNet"):
                self.momentum_fn = ConvNet2D(model_name='VNet', **kwargs)

    def _build_generic_nets(self):
        """Build GenericNet FC-architectures for position and momentum fns. """

        kwargs = {
            'x_dim': self.x_dim,
            'links_shape': self.lattice.links.shape,
            'num_hidden': int(4 * self.x_dim),
            'name_scope': 'position',
            'factor': 2.
        }

        with tf.name_scope("DynamicsNetwork"):
            with tf.name_scope("XNet"):
                self.position_fn = GenericNet(model_name='XNet', **kwargs)

            kwargs['factor'] = 1.
            kwargs['name_scope'] = 'momentum'
            with tf.name_scope("VNet"):
                self.momentum_fn = GenericNet(model_name='VNet', **kwargs)

    def call(self, position, beta, save_lf=False, train=True):
        """Call method."""
        #  position, beta = inputs
        return self.apply_transition(position, beta,
                                     save_lf=save_lf, train=train)

    def apply_transition(self, position, beta, save_lf=False, train=True):
        """Propose a new state and perform the accept/reject step.

        Args:
            position: Batch of (position) samples (batch of links).
            beta (float): Inverse coupling constant.

        Returns:
            position_post: Proposed position before accept/reject step.
            momentum_post: Proposed momentum before accept/reject step.
            accept_prob: Probability of accepting the proposed states.
            position_out: Samples after accept/reject step.
        """
        # Simulate dynamics both forward and backward
        # Use sample masks to compute the actual solutions
        with tf.name_scope('apply_transition'):
            with tf.name_scope('transition_forward'):
                outputs = self.transition_kernel(position, beta,
                                                 forward=True,
                                                 save_lf=save_lf,
                                                 train=train)
                position_f = outputs[0]
                momentum_f = outputs[1]
                accept_prob_f = outputs[2]
                if save_lf:
                    lf_out_f = outputs[3]
                    accept_probs_f = outputs[4]
                    logdets_f = outputs[5]
                    sumlogdet_f = outputs[6]

            with tf.name_scope('transition_backward'):
                outputs = self.transition_kernel(position, beta,
                                                 forward=False,
                                                 save_lf=save_lf,
                                                 train=train)
                position_b = outputs[0]
                momentum_b = outputs[1]
                accept_prob_b = outputs[2]
                if save_lf:
                    lf_out_b = outputs[3]
                    accept_probs_b = outputs[4]
                    logdets_b = outputs[5]
                    sumlogdet_b = outputs[6]

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
            with tf.name_scope('position_post'):
                position_post = (forward_mask[:, None] * position_f
                                 + backward_mask[:, None] * position_b)

            with tf.name_scope('momentum_post'):
                momentum_post = (forward_mask[:, None] * momentum_f
                                 + backward_mask[:, None] * momentum_b)

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
            with tf.name_scope('position_out'):
                position_out = (
                    accept_mask[:, None] * position_post
                    + reject_mask[:, None] * position
                )
        if save_lf:
            return (position_post, momentum_post, accept_prob, position_out,
                    lf_out_f, accept_probs_f, lf_out_b, accept_probs_b,
                    forward_mask, backward_mask, logdets_f, logdets_b,
                    sumlogdet_f, sumlogdet_b)
        else:
            return position_post, momentum_post, accept_prob, position_out,

    def transition_kernel(self, position, beta,
                          forward=True, save_lf=False, train=True):
        """Transition kernel of augmented leapfrog integrator."""
        #  lf_out = []
        accept_probs = []

        lf_fn = self._forward_lf if forward else self._backward_lf

        momentum = tf.random_normal(tf.shape(position), seed=GLOBAL_SEED)

        position_post, momentum_post = position, momentum

        if train:
            t = tf.constant(0., name='md_time', dtype=TF_FLOAT)
            batch_size = tf.shape(position)[0]
            logdet = tf.zeros((batch_size,))

            def body(x, v, beta, t, logdet):
                new_x, new_v, j = lf_fn(x, v, beta, t)
                return new_x, new_v, beta, t + 1, logdet + j

            def cond(x, v, beta, t, logdet):
                return tf.less(t, self.num_steps)

            with tf.name_scope('while_loop'):
                outputs = tf.while_loop(cond=cond,
                                        body=body,
                                        loop_vars=[position_post,
                                                   momentum_post,
                                                   beta, t, logdet])
                #  position_post, momentum_post, beta, t, sumlogdet = outputs
                position_post = outputs[0]
                momentum_post = outputs[1]
                beta = outputs[2]
                t = outputs[3]
                sumlogdet = outputs[4]
        else:
            lf_out = []
            logdet = []
            sumlogdet = 0.
            for t in range(self.num_steps):
                position_post, momentum_post, j = lf_fn(position_post,
                                                        momentum_post, beta, t)
                sumlogdet += j
                lf_out.append(position_post)
                logdet.append(j)

        with tf.name_scope('accept_prob'):
            accept_prob = self._compute_accept_prob(
                position,
                momentum,
                position_post,
                momentum_post,
                sumlogdet,
                beta
            )
            if save_lf:
                accept_probs.append(accept_prob)

        if save_lf:
            return (position_post, momentum_post, accept_prob,
                    lf_out, accept_probs, logdet, sumlogdet)
        else:
            return position_post, momentum_post, accept_prob

    def _forward_lf(self, position, momentum, beta, step):
        """One forward augmented leapfrog step."""
        with tf.name_scope('forward_lf'):
            # use self._get_time when using for loop in transition__kernel
            #  t = self._get_time(step)
            #  mask, mask_inv = self._get_mask(step)

            # use self._format_time when using while loop in transition_kernel
            t = self._format_time(step, tile=tf.shape(position)[0])
            mask, mask_inv = self._get_mask_while(step)

            sumlogdet = 0.

            momentum, logdet = self._update_momentum_forward(position,
                                                             momentum,
                                                             beta, t)
            sumlogdet += logdet

            position, logdet = self._update_position_forward(position,
                                                             momentum,
                                                             t, mask,
                                                             mask_inv)
            sumlogdet += logdet

            position, logdet = self._update_position_forward(position,
                                                             momentum,
                                                             t, mask_inv,
                                                             mask)
            sumlogdet += logdet

            momentum, logdet = self._update_momentum_forward(position,
                                                             momentum,
                                                             beta, t)
            sumlogdet += logdet

        return position, momentum, sumlogdet

    def _backward_lf(self, position, momentum, beta, step):
        """One backward augmented leapfrog step."""

        # Reversed index/sinusoidal time
        with tf.name_scope('backward_lf'):
            # use self._get_time when using for loop in transition__kernel
            #  t = self._get_time(self.num_steps - step - 1)
            #  mask, mask_inv = self._get_mask(self.num_steps - step - 1)

            # use self._format_time when using while loop in transition_kernel
            t = self._format_time(self.num_steps - step - 1,
                                  tile=tf.shape(position)[0])
            mask, mask_inv = self._get_mask_while(self.num_steps - step - 1)

            sumlogdet = 0.

            momentum, logdet = self._update_momentum_backward(position,
                                                              momentum,
                                                              beta, t)
            sumlogdet += logdet

            position, logdet = self._update_position_backward(position,
                                                              momentum,
                                                              t, mask_inv,
                                                              mask)
            sumlogdet += logdet

            position, logdet = self._update_position_backward(position,
                                                              momentum,
                                                              t, mask,
                                                              mask_inv)
            sumlogdet += logdet

            momentum, logdet = self._update_momentum_backward(position,
                                                              momentum,
                                                              beta, t)
            sumlogdet += logdet

        return position, momentum, sumlogdet

    def _update_momentum_forward(self, position, momentum, beta, t):
        """Update v in the forward leapfrog step."""
        with tf.name_scope('update_momentum_forward'):
            with tf.name_scope('grad_potential'):
                grad = self.grad_potential(position, beta)

            with tf.name_scope('momentum_fn'):
                scale, translation, transformed = self.momentum_fn(
                    [position, grad, t]
                )

            with tf.name_scope('scale'):
                scale *= 0.5 * self.eps

            with tf.name_scope('transformed'):
                transformed *= self.eps

            with tf.name_scope('momentum_update'):
                momentum = (momentum * exp(scale, 'vf_scale')
                            - (0.5 * self.eps
                               * (exp(transformed, name='vf_transformed')
                                  * grad + translation)))

        return momentum, tf.reduce_sum(scale, axis=1)

    def _update_position_forward(self, position, momentum, t, mask, mask_inv):
        """Update x in the forward leapfrog step."""
        with tf.name_scope('update_position_forward'):
            with tf.name_scope('position_fn'):
                scale, translation, transformed = self.position_fn(
                    [momentum, mask * position, t]
                )

            with tf.name_scope('scale'):
                scale *= self.eps

            with tf.name_scope('transformed'):
                transformed *= self.eps

            with tf.name_scope('position_update'):
                position = (mask * position + mask_inv
                            * (position * exp(scale, 'xf_scale') + self.eps
                               * (exp(transformed, 'xf_transformed')
                                  * momentum + translation)))

        return position, tf.reduce_sum(mask_inv * scale, axis=1)

    def _update_momentum_backward(self, position, momentum, beta, t):
        """Update v in the backward leapfrog step. Invert the forward update"""
        #  grad = self.grad_potential(position, beta)
        with tf.name_scope('update_momentum_backward'):
            with tf.name_scope('grad_potential'):
                grad = self.grad_potential(position, beta)

            with tf.name_scope('momentum_fn'):
                scale, translation, transformed = self.momentum_fn(
                    [position, grad, t]
                )

            with tf.name_scope('scale'):
                scale *= -0.5 * self.eps

            with tf.name_scope('transformed'):
                transformed *= self.eps

            with tf.name_scope('momentum_update'):
                momentum = (exp(scale, 'vb_scale')
                            * (momentum + 0.5 * self.eps
                               * (exp(transformed, 'vb_transformed')
                                  * grad + translation)))

        return momentum, tf.reduce_sum(scale, axis=1)

    def _update_position_backward(self, position, momentum, t, mask, mask_inv):
        """Update x in the backward lf step. Inverting the forward update."""
        with tf.name_scope('update_position_backward'):
            with tf.name_scope('position_fn'):
                scale, translation, transformed = self.position_fn(
                    [momentum, mask * position, t]
                )

            with tf.name_scope('scale'):
                scale *= -self.eps
                #  scale_exp = exp(scale, 'xb_scale')

            with tf.name_scope('transformed'):
                transformed *= self.eps
                #  transformed_exp = exp(transformed, 'xb_transformed')

            with tf.name_scope('position_update'):
                #  term1 = position * scale_exp
                #  term2 = scale_exp * (
                #      position - self.eps * (momentum * transformed_exp
                #                             + translation)
                #  )
                #  position = mask * term1 + mask_inv * term2
                position = (mask * position + mask_inv * exp(scale, 'xb_scale')
                            * (position - self.eps
                               * (exp(transformed, 'xb_transformed')
                                  * momentum + translation)))

        return position, tf.reduce_sum(mask_inv * scale, axis=1)

    def _compute_accept_prob(self, xi, vi, xf, vf, sumlogdet, beta):
        """Compute the prob of accepting the proposed state given old state.
        Args:
            xi: Initial state.
            vi: Initial momentum.
            xf: Proposed state.
            vf: Proposed momentum.
            sumlogdet: Sum of the terms of the log of the determinant. (Eq. 14
                of original paper).
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

    def _construct_time(self):
        """Convert leapfrog step index into sinusoidal time."""
        self.ts = []
        with tf.name_scope('construct_time'):
            for i in range(self.num_steps):
                t = tf.constant([np.cos(2 * np.pi * i / self.num_steps),
                                 np.sin(2 * np.pi * i / self.num_steps)],
                                dtype=TF_FLOAT)
                self.ts.append(t[None, :])

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
        m = tf.gather(self.masks, tf.cast(step, dtype=tf.int32))
        return m, 1. - m

    def potential_energy(self, position, beta):
        """Compute potential energy using `self.potential` and beta."""
        #  return beta * self.potential(position)
        with tf.name_scope('potential_energy'):
            potential_energy = tf.multiply(beta, self.potential(position))

        return potential_energy

    def kinetic_energy(self, v):
        """Compute the kinetic energy."""
        with tf.name_scope('kinetic_energy'):
            #  kinetic_energy = 0.5 * tf.reduce_sum(v**2, axis=self.axes)
            kinetic_energy = 0.5 * tf.reduce_sum(v**2, axis=1)

        return kinetic_energy

    def hamiltonian(self, position, momentum, beta):
        """Compute the overall Hamiltonian."""
        with tf.name_scope('hamiltonian'):
            hamiltonian = (self.potential_energy(position, beta)
                           + self.kinetic_energy(momentum))
        return hamiltonian

    def grad_potential(self, position, beta, check_numerics=True):
        """Get gradient of potential function at current location."""
        with tf.name_scope('grad_potential'):
            if tf.executing_eagerly():
                tfe = tf.contrib.eager
                grad_fn = tfe.gradients_function(self.potential_energy,
                                                 params=["position"])
                grad = grad_fn(position, beta)[0]
            else:
                grad = tf.gradients(self.potential_energy(position, beta),
                                    position)[0]
        return grad
