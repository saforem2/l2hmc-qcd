"""
Dynamics engine for L2HMC sampler on Lattice Gauge Models.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Author: Sam Foreman (github: @saforem2)
Date: 1/14/2019
"""
import numpy as np
import numpy.random as npr
import tensorflow as tf

import utils.file_io as io

from globals import GLOBAL_SEED, TF_FLOAT
from network.conv_net3d import ConvNet3D
from network.conv_net2d import ConvNet2D
from network.generic_net import GenericNet


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
        npr.seed(GLOBAL_SEED)

        self.lattice = lattice
        self.potential = potential_fn
        self.batch_size = self.lattice.samples.shape[0]
        self.x_dim = self.lattice.num_links

        # create attributes from kwargs.items()
        for key, val in kwargs.items():
            if key != 'eps':  # want to use self.eps as tf.Variable
                setattr(self, key, val)

        io.log(80 * '-')
        io.log(f'Args received by `GaugeDynamics`:')
        for key, val in kwargs.items():
            io.log(f'{key}: {val}')
        io.log(80 * '-')
        io.log(f'network_arch: {self.network_arch}')

        if self.num_hidden is None:
            self.num_hidden = 2 * self.lattice.num_links

        with tf.name_scope('eps'):
            #  self.eps = exp(self.alpha, name='eps')
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
            if self.network_arch.upper() == 'CONV3D':  # should be 'conv3D'
                self._build_conv_nets_3D()
            if self.network_arch.upper() == 'CONV2D':  # should be 'conv2D'
                self._build_conv_nets_2D()
            else:
                self._build_generic_nets()

    def _build_conv_nets_3D(self):
        """Build ConvNet3D architecture for x and v functions."""
        kwargs = {
            '_input_shape': (self.batch_size, *self.lattice.links.shape),
            'links_shape': self.lattice.links.shape,
            'x_dim': self.lattice.num_links,  # dimensionality of target space
            'factor': 2.,  # scale factor used in original paper
            'spatial_size': self.lattice.space_size,  # spatial size of lattice
            'num_hidden': self.num_hidden,  # num hidden nodes
            'num_filters': int(self.lattice.space_size),  # num conv. filters
            'filter_sizes': [(3, 3, 2), (2, 2, 2)],  # size of conv. filters
            'name_scope': 'position',  # namespace in which to create network
            'data_format': self.data_format,  # channels_first if using GPU
            'use_bn': self.use_bn,  # whether or not to use batch normalization
        }

        with tf.name_scope("DynamicsNetwork"):
            with tf.name_scope("XNet"):
                self.x_fn = ConvNet3D(model_name='XNet', **kwargs)

            kwargs['name_scope'] = 'momentum'  # update name scope
            kwargs['factor'] = 1.              # factor used in orig. paper
            with tf.name_scope("VNet"):
                self.v_fn = ConvNet3D(model_name='VNet', **kwargs)

    def _build_conv_nets_2D(self):
        """Build ConvNet architecture for x and v functions."""
        kwargs = {
            '_input_shape': (self.batch_size, *self.lattice.links.shape),
            'links_shape': self.lattice.links.shape,
            'x_dim': self.lattice.num_links,  # dimensionality of target space
            'factor': 2.,  # scale factor used in original paper
            'spatial_size': self.lattice.space_size,  # spatial size of lattice
            'num_hidden': self.num_hidden,  # num hidden nodes
            'num_filters': int(2 * self.lattice.space_size),  # num filters
            'filter_sizes': [(2, 2), (2, 2)],  # for 1st and 2nd conv. layer
            'name_scope': 'position',  # namespace in which to create network
            'data_format': self.data_format,  # channels_first if using GPU
            'use_bn': self.use_bn,  # whether or not to use batch normalization
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
            '_input_shape': (self.batch_size, *self.lattice.links.shape),
            'x_dim': self.lattice.num_links,  # dimensionality of target space
            'factor': 2.,  # scale factor used in original paper
            'num_hidden': self.num_hidden,  # num hidden nodes
            'name_scope': 'position',  # namespace in which to create network
            'use_bn': self.use_bn
        }

        with tf.name_scope("DynamicsNetwork"):
            with tf.name_scope("XNet"):
                self.x_fn = GenericNet(model_name='XNet', **kwargs)

            kwargs['factor'] = 1.
            kwargs['name_scope'] = 'momentum'
            with tf.name_scope("VNet"):
                self.v_fn = GenericNet(model_name='VNet', **kwargs)

    def call(self, x_in, beta, net_weights, 
             while_loop=True, save_lf=False):
        """Call method."""
        return self.apply_transition(x_in, beta, net_weights,
                                     while_loop=while_loop,
                                     #  v_in=v_in,
                                     save_lf=save_lf)

    def apply_transition(self, x_in, beta, net_weights,
                         while_loop=True, save_lf=False):
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
                                                   while_loop=while_loop,
                                                   forward=True,
                                                   #  v_in=v_in,
                                                   save_lf=save_lf)
                xf = outputs_f['x_proposed']
                vf = outputs_f['v_proposed']
                accept_prob_f = outputs_f['accept_prob']

                if save_lf:
                    lf_out_f = outputs_f['lf_out']
                    logdets_f = outputs_f['logdets']
                    sumlogdet_f = outputs_f['sumlogdet']

            with tf.name_scope('transition_backward'):
                outputs_b = self.transition_kernel(x_in, beta,
                                                   net_weights,
                                                   while_loop=while_loop,
                                                   forward=False,
                                                   #  v_in=v_in,
                                                   save_lf=save_lf)
                xb = outputs_b['x_proposed']
                vb = outputs_b['v_proposed']
                accept_prob_b = outputs_b['accept_prob']

                if save_lf:
                    lf_out_b = outputs_b['lf_out']
                    logdets_b = outputs_b['logdets']
                    sumlogdet_b = outputs_b['sumlogdet']

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
                x_proposed = (forward_mask[:, None] * xf
                              + backward_mask[:, None] * xb)

            with tf.name_scope('v_proposed'):
                v_proposed = (forward_mask[:, None] * vf
                              + backward_mask[:, None] * vb)

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
                x_out = (accept_mask[:, None] * x_proposed
                         + reject_mask[:, None] * x_in)

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

    def transition_kernel(self, x_in, beta, net_weights, while_loop=True,
                          forward=True, save_lf=False):
        """Transition kernel of augmented leapfrog integrator."""

        #  if v_in is None:
        with tf.name_scope('refresh_momentum'):
            v_in = tf.random_normal(tf.shape(x_in), seed=GLOBAL_SEED)

        if while_loop:
            outputs = self._transition_while_loop(x_in, v_in,
                                                  beta, net_weights,
                                                  forward, save_lf)
        else:
            outputs = self._transition_for_loop(x_in, v_in,
                                                beta, net_weights,
                                                forward, save_lf)

        x_proposed = outputs['x_proposed']
        v_proposed = outputs['v_proposed']
        sumlogdet = outputs['sumlogdet']

        with tf.name_scope('accept_prob'):
            accept_prob = self._compute_accept_prob(x_in, v_in,
                                                    x_proposed,
                                                    v_proposed,
                                                    sumlogdet,
                                                    beta)

        outputs['accept_prob'] = accept_prob

        return outputs

    def _transition_for_loop(self, x_in, v_in, beta, net_weights,
                             forward=True, save_lf=False):
        """Implements the transition kernel using a basic `for` loop."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        with tf.name_scope('for_loop_init'):
            x_proposed, v_proposed = x_in, v_in
            sumlogdet = 0.
            lf_out = [x_proposed]
            #  batch_size = tf.shape(x_in)[0]
            #  assert batch_size == self.batch_size
            logdets_out = [tf.zeros((self.batch_size,))]

            for i in range(self.num_steps):
                step = tf.convert_to_tensor(i, dtype=TF_FLOAT)
                x_proposed, v_proposed, logdet = lf_fn(x_proposed,
                                                       v_proposed,
                                                       beta, step,
                                                       net_weights)
                sumlogdet += logdet
                lf_out.append(x_proposed)
                logdets_out.append(logdet)

        outputs = {
            'x_proposed': x_proposed,
            'v_proposed': v_proposed,
            'sumlogdet': sumlogdet
        }

        if save_lf:
            outputs['lf_out'] = lf_out
            outputs['logdets'] = logdets_out

        return outputs

    def _transition_while_loop(self, x_in, v_in, beta, net_weights,
                               forward=True, save_lf=False):
        """Implements the transition kernel using a `tf.while_loop`."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        with tf.name_scope('while_loop_init'):
            x_proposed, v_proposed = x_in, v_in
            #  t = tf.constant(0., name='md_time', dtype=TF_FLOAT)
            step = tf.constant(0., name='md_step', dtype=TF_FLOAT)
            batch_size = tf.shape(x_in)[0]
            #  assert batch_size == self.batch_size
            logdet = tf.zeros((batch_size,))
            lf_out = tf.TensorArray(dtype=TF_FLOAT, size=self.num_steps,
                                    dynamic_size=True, name='lf_out',
                                    clear_after_read=False)
            logdets_out = tf.TensorArray(dtype=TF_FLOAT, size=self.num_steps,
                                         dynamic_size=True, name='logdets_out',
                                         clear_after_read=False)
            #  lf_out = lf_out.write(0, x_in)
            #  logdets_out.write(0, logdet)

        def body(x, v, beta, step, logdet, lf_samples, logdets):
            i = tf.cast(step, dtype=tf.int32, name='lf_step')  # cast as int
            with tf.name_scope('apply_lf'):
                new_x, new_v, j = lf_fn(x, v, beta, step, net_weights)
            with tf.name_scope('concat_lf_outputs'):
                #  if i == 0:
                #      lf_samples = lf_samples.write(i, x)
                #  else:
                lf_samples = lf_samples.write(i, new_x)
            with tf.name_scope('concat_logdets'):
                logdets = logdets.write(i+1, logdet+j)
            return (new_x, new_v, beta, step + 1,
                    logdet + j, lf_samples, logdets)

        def cond(x, v, beta, step, logdet, lf_out, logdets):
            with tf.name_scope('check_lf_step'):
                return tf.less(step, self.num_steps)

        with tf.name_scope('while_loop'):
            outputs = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[x_proposed, v_proposed,
                           beta, step, logdet,
                           lf_out, logdets_out]
            )

            x_proposed = outputs[0]
            v_proposed = outputs[1]
            beta = outputs[2]
            step = outputs[3]
            sumlogdet = outputs[4]
            lf_out = outputs[5].stack()
            logdets_out = outputs[6].stack()

        outputs = {
            'x_proposed': x_proposed,
            'v_proposed': v_proposed,
            'sumlogdet': sumlogdet
        }

        if save_lf:
            outputs['lf_out'] = lf_out
            outputs['logdets'] = logdets_out

        return outputs

    def _forward_lf(self, x, v, beta, step, net_weights):
        """One forward augmented leapfrog step."""
        with tf.name_scope('forward_lf'):
            with tf.name_scope('get_time'):
                t = self._get_time(step, tile=tf.shape(x)[0])

            if t.dtype != TF_FLOAT:
                t = tf.cast(t, dtype=TF_FLOAT)

            with tf.name_scope('get_mask'):
                mask, mask_inv = self._get_mask(step)

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
            with tf.name_scope('get_time'):
                t = self._get_time(self.num_steps - step - 1,
                                   tile=tf.shape(x)[0])

            if t.dtype != TF_FLOAT:
                t = tf.cast(t, dtype=TF_FLOAT)

            with tf.name_scope('get_mask'):
                mask, mask_inv = self._get_mask(self.num_steps - step - 1)

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

            with tf.name_scope('v_fn'):
                # Sv: scale, Qv: transformation, Tv: translation
                scale, translation, transformation = self.v_fn(
                    [x, grad, t]
                )

            with tf.name_scope('scale'):
                scale *= 0.5 * self.eps * net_weights[0]
                scale_exp = exp(scale, 'vf_scale')

            with tf.name_scope('transformation'):
                transformation *= self.eps * net_weights[1]
                transformation_exp = exp(transformation, 'vf_transformation')

            with tf.name_scope('translation'):
                translation *= net_weights[2]

            with tf.name_scope('v_update'):
                v = (v * scale_exp - 0.5 * self.eps
                     * (grad * transformation_exp + translation))

        return v, tf.reduce_sum(scale, axis=1, name='vf_logdet')

    def _update_x_forward(self, x, v, t, net_weights, mask, mask_inv):
        """Update x in the forward leapfrog step."""
        with tf.name_scope('update_x_forward'):
            with tf.name_scope('x_fn'):
                scale, translation, transformation = self.x_fn(
                    [v, mask * x, t]
                )

            with tf.name_scope('scale'):
                scale *= self.eps * net_weights[0]
                scale_exp = exp(scale, 'xf_scale')

            with tf.name_scope('transformation'):
                transformation *= self.eps * net_weights[1]
                transformation_exp = exp(transformation, 'xf_transformation')

            with tf.name_scope('translation'):
                translation *= net_weights[2]

            with tf.name_scope('x_update'):
                x = (mask * x
                     + mask_inv * (x * scale_exp + self.eps
                                   * (v * transformation_exp + translation)))

        return x, tf.reduce_sum(mask_inv * scale, axis=1, name='xf_logdet')

    def _update_v_backward(self, x, v, beta, t, net_weights):
        """Update v in the backward leapfrog step. Invert the forward update"""
        with tf.name_scope('update_v_backward'):
            with tf.name_scope('grad_potential'):
                grad = self.grad_potential(x, beta)

            with tf.name_scope('v_fn'):
                scale, translation, transformation = self.v_fn(
                    [x, grad, t]
                )

            with tf.name_scope('scale'):
                scale *= -0.5 * self.eps * net_weights[0]
                scale_exp = exp(scale, 'vb_scale')

            with tf.name_scope('transformation'):
                transformation *= self.eps * net_weights[1]
                transformation_exp = exp(transformation, 'vb_transformation')

            with tf.name_scope('translation'):
                translation *= net_weights[2]

            with tf.name_scope('v_update'):
                v = (scale_exp * (v + 0.5 * self.eps
                                  * (grad * transformation_exp + translation)))

        return v, tf.reduce_sum(scale, axis=1, name='vb_logdet')

    def _update_x_backward(self, x, v, t, net_weights, mask, mask_inv):
        """Update x in the backward lf step. Inverting the forward update."""
        with tf.name_scope('update_x_backward'):
            with tf.name_scope('x_fn'):
                scale, translation, transformation = self.x_fn(
                    [v, mask * x, t]
                )

            with tf.name_scope('scale'):
                scale *= -self.eps * net_weights[0]
                scale_exp = exp(scale, 'xb_scale')

            with tf.name_scope('transformation'):
                transformation *= self.eps * net_weights[1]
                transformation_exp = exp(transformation, 'xb_transformation')

            with tf.name_scope('translation'):
                translation *= net_weights[2]

            with tf.name_scope('x_update'):
                x = (mask * x + mask_inv * scale_exp
                     * (x - self.eps * (v * transformation_exp + translation)))

        return x, tf.reduce_sum(mask_inv * scale, axis=1, name='xb_logdet')

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

    def _get_time(self, step, tile=1):
        """Format time as [cos(..), sin(...)]."""
        with tf.name_scope('get_time'):
            trig_t = tf.squeeze([
                tf.cos(2 * np.pi * step / self.num_steps),
                tf.sin(2 * np.pi * step / self.num_steps),
            ], name='md_time', )

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
        if step.dtype != tf.int32:
            step = tf.cast(step, dtype=tf.int32)

        with tf.name_scope('get_mask'):
            if tf.executing_eagerly():
                m = self.masks[step]
            else:
                m = tf.gather(self.masks, step, name='gather_mask')
        return m, 1. - m

    def potential_energy(self, x, beta):
        """Compute potential energy using `self.potential` and beta."""
        with tf.name_scope('potential_energy'):
            potential_energy = tf.multiply(beta, self.potential(x),
                                           name='potential_energy')

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
                x_tensor = tf.constant(x)
                beta_tensor = tf.constant(beta)
                with tf.GradientTape() as tape:
                    tape.watch(x_tensor)
                    potential_energy = self.potential_energy(x_tensor,
                                                             beta_tensor)
                    grad = tape.gradient(potential_energy, x_tensor)

            else:
                grad = tf.gradients(self.potential_energy(x, beta), x)[0]

        return grad
