"""
Dynamics engine for L2HMC sampler on Lattice Gauge Models.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.


TODO:
    - Log separately the Q, S, T values for the 'x' and 'v' functions.
    - Look at raw phase values both before and after mod operation.
    - Try running generic net using 64-point precision.
    - See if 64-point precision issues with Conv3D are fixed in tf 1.13-1.14.
    - JLSE account.


Author: Sam Foreman (github: @saforem2)
Date: 1/14/2019
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import numpy.random as npr
import tensorflow as tf

from config import GLOBAL_SEED, TF_FLOAT, TF_INT
from network.network import FullNet


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


def hmc_network(inputs, train_phase):
    return [tf.zeros_like(inputs[0]) for _ in range(3)]


def _add_to_collection(collection, ops):
    if len(ops) > 1:
        _ = [tf.add_to_collection(collection, op) for op in ops]
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
        self.l2hmc_fns = {}

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
            self.x_fn = lambda inp, train_phase: [
                tf.zeros_like(inp[0]) for t in range(3)
            ]
            self.v_fn = lambda inp, train_phase: [
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
                'num_hidden1': self.num_hidden1,
                'num_hidden2': self.num_hidden2,
                'num_filters': [num_filters, int(2 * num_filters)],
                'name_scope': 'x',  # namespace in which to create network
                'factor': 2.,  # scale factor used in original paper
                '_input_shape': (self.batch_size, *self.lattice.links.shape),
                'zero_translation': self.zero_translation,
                #  'data_format': self.data_format,
            }

            if self.network_arch == 'conv3D':
                net_kwargs.update({
                    'filter_sizes': [(3, 3, 1), (2, 2, 1)],
                })
            elif self.network_arch == 'conv2D':
                net_kwargs.update({
                    'filter_sizes': [(3, 3), (2, 2)],
                })

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

    def apply_transition(self,
                         x_in,
                         beta,
                         net_weights,
                         train_phase,
                         save_lf=False):
        """Propose a new state and perform the accept/reject step.

        Args:
            x_in (placeholder): Batch of (x) samples (GaugeLattice.samples).
            beta (float): Inverse coupling constant.
            net_weights: Array of scaling weights to multiply each of the
                output functions (scale, translation, transformation).
            train_phase: Boolean tf.placeholder used to indicate if currently
                training model or running inference on trained model.
            save_lf: Flag specifying whether or not to save output leapfrog
                configs.

        Returns:
            x_proposed: Proposed x before accept/reject step.
            v_proposed: Proposed v before accept/reject step.
            accept_prob: Probability of accepting the proposed states.
            x_out: Samples after accept/reject step.
        """
        # Simulate dynamics both forward and backward
        # Use sampled masks to compute the actual solutions
        #  with tf.name_scope('apply_transition'):
        tmp_dict = {}

        with tf.name_scope('transition_forward'):
            outputs_f = self.transition_kernel(x_in, beta,
                                               net_weights,
                                               train_phase,
                                               forward=True,
                                               save_lf=save_lf)
            xf = outputs_f['x_proposed']
            vf = outputs_f['v_proposed']
            tmp_dict['pxs_out_f'] = outputs_f['accept_prob']

        with tf.name_scope('transition_backward'):
            outputs_b = self.transition_kernel(x_in, beta,
                                               net_weights,
                                               train_phase,
                                               forward=False,
                                               save_lf=save_lf)
            xb = outputs_b['x_proposed']
            vb = outputs_b['v_proposed']
            tmp_dict['pxs_out_b'] = outputs_b['accept_prob']

        def get_lf_keys(direction):
            base_keys = ['lf_out', 'logdets', 'sumlogdet', 'fns_out']
            new_keys = [k + f'_{direction}' for k in base_keys]
            return list(zip(new_keys, base_keys))

        keys_f = get_lf_keys('f')
        keys_b = get_lf_keys('b')

        if save_lf:
            tmp_dict.update({k[0]: outputs_f[k[1]] for k in keys_f})
            tmp_dict.update({k[0]: outputs_b[k[1]] for k in keys_b})

        # Decide direction uniformly
        with tf.name_scope('transition_masks'):
            with tf.name_scope('forward_mask'):
                tmp_dict['masks_f'] = tf.cast(
                    tf.random_uniform((self.batch_size,),
                                      dtype=TF_FLOAT,
                                      seed=GLOBAL_SEED) > 0.5,
                    TF_FLOAT,
                    name='forward_mask'
                )
            with tf.name_scope('backward_mask'):
                tmp_dict['masks_b'] = 1. - tmp_dict['masks_f']

        # Obtain proposed states
        with tf.name_scope('x_proposed'):
            x_proposed = (tmp_dict['masks_f'][:, None] * xf
                          + tmp_dict['masks_b'][:, None] * xb)

        with tf.name_scope('v_proposed'):
            v_proposed = (tmp_dict['masks_f'][:, None] * vf
                          + tmp_dict['masks_b'][:, None] * vb)

        # Probability of accepting the proposed states
        with tf.name_scope('accept_prob'):
            accept_prob = (tmp_dict['masks_f'] * tmp_dict['pxs_out_f']
                           + tmp_dict['masks_b'] * tmp_dict['pxs_out_b'])

        # Accept or reject step
        with tf.name_scope('accept_mask'):
            accept_mask = tf.cast(
                accept_prob > tf.random_uniform(tf.shape(accept_prob),
                                                dtype=TF_FLOAT,
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
            outputs.update(tmp_dict)

        return outputs

    def transition_kernel(self,
                          x_in,
                          beta,
                          net_weights,
                          train_phase,
                          forward=True,
                          save_lf=False):
        """Transition kernel of augmented leapfrog integrator."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        with tf.name_scope('refresh_momentum'):
            v_in = tf.random_normal(tf.shape(x_in),
                                    dtype=TF_FLOAT,
                                    seed=GLOBAL_SEED)

        with tf.name_scope('init'):
            x_proposed, v_proposed = x_in, v_in

            step = tf.constant(0., name='md_step', dtype=TF_FLOAT)
            batch_size = tf.shape(x_in)[0]
            logdet = tf.zeros((batch_size,), dtype=TF_FLOAT)
            #  fns0 = tf.zeros((4, 3, *x_in.shape),)
            lf_out = tf.TensorArray(dtype=TF_FLOAT,
                                    size=self.num_steps+1,
                                    dynamic_size=True,
                                    name='lf_out',
                                    clear_after_read=False)
            logdets_out = tf.TensorArray(dtype=TF_FLOAT,
                                         size=self.num_steps+1,
                                         dynamic_size=True,
                                         name='logdets_out',
                                         clear_after_read=False)
            fns_out = tf.TensorArray(dtype=TF_FLOAT,
                                     size=self.num_steps,
                                     dynamic_size=True,
                                     name='l2hmc_fns',
                                     clear_after_read=False)

            lf_out = lf_out.write(0, x_in)
            logdets_out = logdets_out.write(0, logdet)
            #  fns_out.write(0, fns0)

        def body(step, x, v, logdet, lf_samples, logdets, fns):
            # cast leapfrog step to integer
            i = tf.cast(step, dtype=tf.int32)
            new_x, new_v, j, _fns = lf_fn(x, v, beta, step,
                                          net_weights, train_phase)
            lf_samples = lf_samples.write(i+1, new_x)
            logdets = logdets.write(i+1, logdet+j)
            fns = fns.write(i, _fns)
            #  fns = fns.write(i, _fns)

            return (step+1, new_x, new_v, logdet+j, lf_samples, logdets, fns)

        def cond(step, *args):
            return tf.less(step, self.num_steps)

        outputs = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[step, x_proposed, v_proposed,
                       logdet, lf_out, logdets_out, fns_out])

        step = outputs[0]
        x_proposed = outputs[1]
        v_proposed = outputs[2]
        sumlogdet = outputs[3]
        lf_out = outputs[4].stack()
        logdets_out = outputs[5].stack()
        fns_out = outputs[6].stack()

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
            outputs['fns_out'] = fns_out

        return outputs

    def _forward_lf(self, x, v, beta, step, net_weights, train_phase):
        """One forward augmented leapfrog step."""
        forward_fns = []
        with tf.name_scope('forward_lf'):
            with tf.name_scope('get_time'):
                t = self._get_time(step, tile=tf.shape(x)[0])
            with tf.name_scope('get_mask'):
                mask, mask_inv = self._get_mask(step)

            sumlogdet = 0.

            v, logdet, vf_fns = self._update_v_forward(x, v, beta, t,
                                                       net_weights,
                                                       train_phase)
            sumlogdet += logdet
            forward_fns.append(vf_fns)

            x, logdet, xf_fns = self._update_x_forward(x, v, t,
                                                       net_weights,
                                                       train_phase,
                                                       mask, mask_inv)
            sumlogdet += logdet
            forward_fns.append(xf_fns)

            x, logdet, xf_fns = self._update_x_forward(x, v, t,
                                                       net_weights,
                                                       train_phase,
                                                       mask_inv, mask)
            sumlogdet += logdet
            forward_fns.append(xf_fns)

            v, logdet, vf_fns = self._update_v_forward(x, v, beta, t,
                                                       net_weights,
                                                       train_phase)
            sumlogdet += logdet
            forward_fns.append(vf_fns)

        return x, v, sumlogdet, forward_fns

    def _backward_lf(self, x, v, beta, step, net_weights, train_phase):
        """One backward augmented leapfrog step."""
        backward_fns = []
        with tf.name_scope('backward_lf'):
            with tf.name_scope('get_time'):
                # Reversed index/sinusoidal time
                t = self._get_time(self.num_steps - step - 1,
                                   tile=tf.shape(x)[0])
            with tf.name_scope('get_mask'):
                mask, mask_inv = self._get_mask(
                    self.num_steps - step - 1
                )

            sumlogdet = 0.

            v, logdet, vb_fns = self._update_v_backward(x, v, beta, t,
                                                        net_weights,
                                                        train_phase)
            sumlogdet += logdet
            backward_fns.append(vb_fns)

            x, logdet, xb_fns = self._update_x_backward(x, v, t,
                                                        net_weights,
                                                        train_phase,
                                                        mask_inv, mask)
            sumlogdet += logdet
            backward_fns.append(xb_fns)

            x, logdet, xb_fns = self._update_x_backward(x, v, t,
                                                        net_weights,
                                                        train_phase,
                                                        mask, mask_inv)
            sumlogdet += logdet
            backward_fns.append(xb_fns)

            v, logdet, vb_fns = self._update_v_backward(x, v, beta, t,
                                                        net_weights,
                                                        train_phase)
            sumlogdet += logdet
            backward_fns.append(vb_fns)

        return x, v, sumlogdet, backward_fns

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
        with tf.name_scope('update_vf'):
            grad = self.grad_potential(x, beta)

            scale, transl, transf = self.v_fn((x, grad, t), train_phase)
            fns = [scale, transl, transf]

            with tf.name_scope('vf_mul'):
                scale *= 0.5 * self.eps * net_weights[0]
                transl *= net_weights[1]
                transf *= self.eps * net_weights[2]

            with tf.name_scope('vf_exp'):
                scale_exp = tf.cast(exp(scale, 'scale_exp'), dtype=TF_FLOAT)
                transf_exp = tf.cast(exp(transf, 'transf_exp'),
                                     dtype=TF_FLOAT)

            with tf.name_scope('proposed'):
                v = (v * scale_exp
                     - 0.5 * self.eps * (grad * transf_exp - transl))

            logdet = tf.reduce_sum(scale, axis=1, name='logdet_vf')

        return v, logdet, fns

    def _update_x_forward(self, x, v, t, net_weights, 
                          train_phase, mask, mask_inv):
        """Update x in the forward leapfrog step."""
        with tf.name_scope('update_xf'):
            scale, transl, transf = self.x_fn([v, mask * x, t], train_phase)
            fns = [scale, transl, transf]

            with tf.name_scope('xf_mul'):
                scale *= self.eps * net_weights[0]
                transl *= net_weights[1]
                transf *= self.eps * net_weights[2]

            with tf.name_scope('xf_exp'):
                scale_exp = exp(scale, 'scale_exp')
                transf_exp = exp(transf, 'transformation_exp')

            with tf.name_scope('proposed'):
                x = (mask * x + mask_inv
                     * (x * scale_exp + self.eps * (v * transf_exp + transl)))

            logdet = tf.reduce_sum(mask_inv * scale, axis=1, name='logdet_xf')

        return x, logdet, fns

    def _update_v_backward(self, x, v, beta, t, net_weights, train_phase):
        """Update v in the backward leapfrog step. Invert the forward update"""
        with tf.name_scope('update_vb'):
            grad = self.grad_potential(x, beta)

            scale, transl, transf = self.v_fn([x, grad, t], train_phase)
            fns = [scale, transl, transf]

            with tf.name_scope('vb_mul'):
                scale *= -0.5 * self.eps * net_weights[0]
                transl *= net_weights[1]
                transf *= self.eps * net_weights[2]

            with tf.name_scope('vb_exp'):
                scale_exp = exp(scale, 'scale_exp')
                transf_exp = exp(transf, 'transformation_exp')

            with tf.name_scope('proposed'):
                v = scale_exp * (v + 0.5 * self.eps
                                 * (grad * transf_exp - transl))

            logdet = tf.reduce_sum(scale, axis=1, name='logdet_vb')

        return v, logdet, fns

    def _update_x_backward(self, x, v, t, net_weights, 
                           train_phase, mask, mask_inv):
        """Update x in the backward lf step. Inverting the forward update."""
        with tf.name_scope('update_xb'):
            scale, transl, transf = self.x_fn([v, mask * x, t], train_phase)
            fns = [scale, transl, transf]

            with tf.name_scope('xb_mul'):
                scale *= -self.eps * net_weights[0]
                transl *= net_weights[1]
                transf *= self.eps * net_weights[2]

            with tf.name_scope('xb_exp'):
                scale_exp = exp(scale, 'xb_scale')
                transf_exp = exp(transf, 'xb_transformation')

            with tf.name_scope('proposed'):
                x = (mask * x + mask_inv * scale_exp
                     * (x - self.eps * (v * transf_exp + transl)))

            logdet = tf.reduce_sum(mask_inv * scale, axis=1,
                                   name='logdet_xb')

        return x, logdet, fns

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
            accept_prob = tf.where(tf.is_finite(prob), prob,
                                   tf.zeros_like(prob))

        return accept_prob

    def _get_time(self, i, tile=1):
        """Format time as [cos(..), sin(...)]."""
        with tf.name_scope('get_time'):
            trig_t = tf.squeeze([
                tf.cos(2 * np.pi * i / self.num_steps),
                tf.sin(2 * np.pi * i / self.num_steps),
            ])

            t = tf.tile(tf.expand_dims(trig_t, 0), (tile, 1))

        return t

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
            m = tf.gather(self.masks, tf.cast(step, dtype=TF_INT))
            _m = 1. - m  # complementary mask
        return m, _m

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
            potential = self.potential_energy(x, beta)
            kinetic = self.kinetic_energy(v)
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
