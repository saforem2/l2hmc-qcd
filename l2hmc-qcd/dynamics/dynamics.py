"""
Dynamics engine for L2HMC sampler on Lattice Gauge Models.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Author: Sam Foreman (github: @saforem2)
Date: 1/14/2019
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import numpy.random as npr

from network.network import FullNet
from config import GLOBAL_SEED, TF_FLOAT, TF_INT


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
        _ = [tf.add_to_collection(collection, op) for op in ops]
    else:
        tf.add_to_collection(collection, ops)


class Dynamics(tf.keras.Model):
    """Dynamics engine of naive L2HMC sampler."""

    def __init__(self, potential_fn, **params):
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
        super(Dynamics, self).__init__(name='Dynamics')
        npr.seed(GLOBAL_SEED)

        self.potential = potential_fn
        self.l2hmc_fns = {}

        # create attributes from kwargs.items()
        for key, val in params.items():
            if key != 'eps':  # want to use self.eps as tf.Variable
                setattr(self, key, val)

        self._eps_np = params.get('eps', 0.4)
        self.eps = self._get_eps(use_log=False)

        self.masks = self._build_masks()

        net_params = self._network_setup()
        self.x_fn, self.v_fn = self.build_network(net_params)

    def _get_eps(self, use_log=False):
        """Create `self.eps` (i.e. the step size) as a `tf.Variable`.

        Args:
            use_log (bool): If True, initialize `log_eps` as the actual
                `tf.Variable` and set `self.eps = tf.exp(log_eps)`; otherwise,
                set `self.eps` as a `tf.Variable directly.

        Returns:
            eps: The (trainable) step size to be used in the L2HMC algorithm.
        """
        if use_log:
            log_eps = tf.Variable(
                initial_value=tf.log(tf.constant(self._eps_np)),
                name='log_eps',
                dtype=TF_FLOAT,
                trainable=self.eps_trainable
            )

            eps = tf.exp(log_eps, name='eps')

        else:
            eps = tf.Variable(
                initial_value=self._eps_np,
                name='eps',
                dtype=TF_FLOAT,
                trainable=self.eps_trainable
            )

        return eps

    def _network_setup(self):
        """Collect parameters needed to build neural network."""
        if self.hmc:
            return {}

        net_params = {
            'network_arch': self.network_arch,  # network architecture
            'use_bn': self.use_bn,              # use batch normalization
            'dropout_prob': self.dropout_prob,  # dropout only used if > 0
            'x_dim': self.x_dim,                # dim of target distribution
            'num_hidden1': self.num_hidden1,    # num. nodes in hidden layer 1
            'num_hidden2': self.num_hidden2,    # num. nodes in hidden layer 2
            'generic_activation': tf.nn.relu,   # activation fn in generic net
            'name_scope': 'x',                  # name scope in which to create
            'factor': 2.,                       # x: factor = 2.; v: factor = 1
            '_input_shape': self._input_shape,  # input shape (b4 reshaping)
        }

        network_arch = str(self.network_arch).lower()
        if network_arch in ['conv3d', 'conv2d']:
            if network_arch == 'conv2d':
                filter_sizes = [(3, 3), (2, 2)]  # size of conv. filters
            elif network_arch == 'conv3d':
                filter_sizes = [(3, 3, 1), (2, 2, 1)]

            num_filters = int(self.num_filters)  # num. filters in conv layers
            net_params.update({
                'num_filters': [num_filters, int(2 * num_filters)],
                'filter_sizes': filter_sizes
            })

        return net_params

    def build_network(self, net_params):
        """Build neural network used to train model."""
        if self.hmc:
            x_fn = lambda inputs, train_phase: [  # noqa: E731
                tf.zeros_like(inputs[0]) for _ in range(3)
            ]
            v_fn = lambda inputs, train_phase: [  # noqa: E731
                tf.zeros_like(inputs[0]) for _ in range(3)
            ]

        else:
            x_fn = FullNet(model_name='XNet', **net_params)

            net_params['name_scope'] = 'v'  # update name scope
            net_params['factor'] = 1.       # factor used in orig. paper
            v_fn = FullNet(model_name='VNet', **net_params)

        return x_fn, v_fn

    def call(self, *args, **kwargs):
        """Call method."""
        return self.apply_transition(*args, **kwargs)

    def apply_transition(self,
                         x_in,
                         beta,
                         weights,
                         train_phase,
                         save_lf=False,
                         hmc=True):
        """Propose a new state and perform the accept/reject step.

        We simulate the (molecular) dynamics update both forward and backward,
        and use sampled masks to compute the actual solutions.

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
        results_dict = {}  # holds additional data if `save_lf=True`

        args = (x_in, beta, weights, train_phase, save_lf, hmc)

        # Forward transition:
        outputs_f = self._transition_forward(*args)
        xf = outputs_f['x_proposed']
        vf = outputs_f['v_proposed']
        pxf = outputs_f['accept_prob']
        pxf_hmc = outputs_f['accept_prob_hmc']

        # Backward transition:
        outputs_b = self._transition_backward(*args)
        xb = outputs_b['x_proposed']
        vb = outputs_b['v_proposed']
        pxb = outputs_b['accept_prob']
        pxb_hmc = outputs_b['accept_prob_hmc']

        # Decide direction uniformly
        forward_mask, backward_mask = self._get_transition_masks()

        # Obtain proposed states
        with tf.name_scope('x_proposed'):
            x_proposed = (xf * forward_mask[:, None]
                          + xb * backward_mask[:, None])

        with tf.name_scope('v_proposed'):
            v_proposed = (vf * forward_mask[:, None]
                          + vb * backward_mask[:, None])

        # Probability of accepting the proposed states
        with tf.name_scope('accept_prob'):
            accept_prob = pxf * forward_mask + pxb * backward_mask
            accept_prob_hmc = pxf_hmc * forward_mask + pxb_hmc * backward_mask

        # Accept or reject step
        accept_mask, reject_mask = self._get_accept_masks(accept_prob)

        # Samples after accept / reject step
        with tf.name_scope('x_out'):
            x_out = (accept_mask[:, None] * x_proposed
                     + reject_mask[:, None] * x_in)

        outputs = {
            'x_in': x_in,
            'x_proposed': x_proposed,
            'v_proposed': v_proposed,
            'accept_prob': accept_prob,
            'x_out': x_out,
            'accept_prob_hmc': accept_prob_hmc,
        }

        if save_lf:
            results_dict['pxs_out_f'] = outputs_f['accept_prob']
            results_dict['pxs_out_b'] = outputs_b['accept_prob']
            results_dict['masks_f'] = forward_mask
            results_dict['masks_b'] = backward_mask

            def get_lf_keys(direction):
                base_keys = ['lf_out', 'logdets', 'sumlogdet', 'fns_out']
                new_keys = [k + f'_{direction}' for k in base_keys]
                return list(zip(new_keys, base_keys))

            keys_f = get_lf_keys('f')
            keys_b = get_lf_keys('b')

            results_dict.update({k[0]: outputs_f[k[1]] for k in keys_f})
            results_dict.update({k[0]: outputs_b[k[1]] for k in keys_b})

            outputs.update(results_dict)

        return outputs

    def _transition_forward(self, x, beta, weights, 
                            train_phase, save_lf, hmc=False):
        with tf.name_scope('transition_forward'):
            with tf.name_scope('refresh_momentum'):
                v_rf = tf.random_normal(tf.shape(x),
                                        dtype=TF_FLOAT,
                                        seed=GLOBAL_SEED,
                                        name='refresh_momentum_forward')

            outputs_f = self.transition_kernel(x, v_rf, beta,
                                               weights,
                                               train_phase,
                                               forward=True,
                                               save_lf=save_lf,
                                               hmc=hmc)
        return outputs_f

    def _transition_backward(self, x, beta, weights,
                             train_phase, save_lf, hmc=False):
        with tf.name_scope('transition_backward'):
            with tf.name_scope('refresh_momentum'):
                v_rb = tf.random_normal(tf.shape(x),
                                        dtype=TF_FLOAT,
                                        seed=GLOBAL_SEED,
                                        name='refresh_momentum_backward')

            outputs_b = self.transition_kernel(x, v_rb, beta,
                                               weights,
                                               train_phase,
                                               forward=False,
                                               save_lf=save_lf,
                                               hmc=hmc)
        return outputs_b

    def _get_transition_masks(self):
        with tf.name_scope('transition_masks'):
            with tf.name_scope('forward_mask'):
                forward_mask = tf.cast(
                    tf.random_uniform((self.batch_size,),
                                      dtype=TF_FLOAT,
                                      seed=GLOBAL_SEED) > 0.5,
                    TF_FLOAT,
                    name='forward_mask'
                )
            with tf.name_scope('backward_mask'):
                backward_mask = 1. - forward_mask

        return forward_mask, backward_mask

    def _get_accept_masks(self, accept_prob):
        with tf.name_scope('accept_mask'):
            accept_mask = tf.cast(
                accept_prob > tf.random_uniform(tf.shape(accept_prob),
                                                dtype=TF_FLOAT,
                                                seed=GLOBAL_SEED),
                TF_FLOAT,
                name='acccept_mask'
            )
            reject_mask = 1. - accept_mask

        return accept_mask, reject_mask

    def transition_kernel(self,
                          x_in,
                          v_in,
                          beta,
                          weights,
                          train_phase,
                          forward=True,
                          save_lf=False,
                          hmc=True):
        """Transition kernel of augmented leapfrog integrator."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        with tf.name_scope('init'):
            x_proposed, v_proposed = x_in, v_in

            step = tf.constant(0., name='md_step', dtype=TF_FLOAT)
            batch_size = tf.shape(x_in)[0]
            logdet = tf.zeros((batch_size,), dtype=TF_FLOAT)
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

        def body(step, x, v, logdet, lf_samples, logdets, fns):
            # cast leapfrog step to integer
            i = tf.cast(step, dtype=tf.int32)
            new_x, new_v, j, _fns = lf_fn(x, v, beta, step,
                                          weights, train_phase)
            lf_samples = lf_samples.write(i+1, new_x)
            logdets = logdets.write(i+1, logdet+j)
            fns = fns.write(i, _fns)

            return (step+1, new_x, new_v, logdet+j, lf_samples, logdets, fns)

        def cond(step, *args):
            return tf.less(step, self.num_steps)

        outputs = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[step, x_proposed, v_proposed,
                       logdet, lf_out, logdets_out, fns_out]
        )

        step = outputs[0]
        x_proposed = outputs[1]
        v_proposed = outputs[2]
        sumlogdet = outputs[3]
        with tf.name_scope('MD_outputs'):
            with tf.name_scope('lf_out'):
                lf_out = outputs[4].stack()
            with tf.name_scope('logdets_out'):
                logdets_out = outputs[5].stack()
            with tf.name_scope('l2hmc_fns_out'):
                fns_out = outputs[6].stack()

        accept_prob, accept_prob_hmc = self._compute_accept_prob(
            x_in, v_in, x_proposed, v_proposed, sumlogdet, beta, hmc=hmc
        )

        outputs = {
            'x_proposed': x_proposed,
            'v_proposed': v_proposed,
            'sumlogdet': sumlogdet,
            'accept_prob': accept_prob,
            'accept_prob_hmc': accept_prob_hmc,
        }

        if save_lf:
            outputs['lf_out'] = lf_out
            outputs['logdets'] = logdets_out
            outputs['fns_out'] = fns_out

        return outputs

    def _check_reversibility(self, x_in, v_in, beta, weights, training):
        outputs_f = self.transition_kernel(x_in, v_in, beta,
                                           weights,
                                           training,
                                           forward=True,
                                           save_lf=False)
        xf = outputs_f['x_proposed']
        vf = outputs_f['v_proposed']

        #  backward(forward(x, v)) --> x, v
        outputs_b = self.transition_kernel(xf, vf, beta,
                                           weights,
                                           training,
                                           forward=False,
                                           save_lf=False)
        xb = outputs_b['x_proposed']
        vb = outputs_b['v_proposed']

        outputs = {
            'xf': xf,
            'vf': vf,
            'xb': xb,
            'vb': vb
        }

        return outputs

    def _forward_lf(self, x, v, beta, step, weights, training):
        """One forward augmented leapfrog step."""
        forward_fns = []
        with tf.name_scope('forward_lf'):
            with tf.name_scope('get_time'):
                t = self._get_time(step, tile=tf.shape(x)[0])
            with tf.name_scope('get_mask'):
                mask, mask_inv = self._get_mask(step)

            sumlogdet = 0.

            v, logdet, vf_fns = self._update_v_forward(x, v, beta, t,
                                                       weights,
                                                       training)
            sumlogdet += logdet
            forward_fns.append(vf_fns)

            x, logdet, xf_fns = self._update_x_forward(x, v, t,
                                                       weights,
                                                       training,
                                                       (mask, mask_inv))
            sumlogdet += logdet
            forward_fns.append(xf_fns)

            x, logdet, xf_fns = self._update_x_forward(x, v, t,
                                                       weights,
                                                       training,
                                                       (mask_inv, mask))
            sumlogdet += logdet
            forward_fns.append(xf_fns)

            v, logdet, vf_fns = self._update_v_forward(x, v, beta, t,
                                                       weights,
                                                       training)
            sumlogdet += logdet
            forward_fns.append(vf_fns)

        return x, v, sumlogdet, forward_fns

    def _backward_lf(self, x, v, beta, step, weights, training):
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
                                                        weights,
                                                        training)
            sumlogdet += logdet
            backward_fns.append(vb_fns)

            x, logdet, xb_fns = self._update_x_backward(x, v, t,
                                                        weights,
                                                        training,
                                                        (mask_inv, mask))
            sumlogdet += logdet
            backward_fns.append(xb_fns)

            x, logdet, xb_fns = self._update_x_backward(x, v, t,
                                                        weights,
                                                        training,
                                                        (mask, mask_inv))
            sumlogdet += logdet
            backward_fns.append(xb_fns)

            v, logdet, vb_fns = self._update_v_backward(x, v, beta, t,
                                                        weights,
                                                        training)
            sumlogdet += logdet
            backward_fns.append(vb_fns)

        return x, v, sumlogdet, backward_fns

    def _update_v_forward(self, x, v, beta, t, weights, training):
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

            scale, transl, transf = self.v_fn([x, grad, t], training)

            with tf.name_scope('vf_mul'):
                scale *= 0.5 * self.eps * weights[0]
                transl *= weights[1]
                transf *= self.eps * weights[2]
                fns = [scale, transl, transf]

            with tf.name_scope('vf_exp'):
                scale_exp = tf.cast(exp(scale, 'scale_exp'), dtype=TF_FLOAT)
                transf_exp = tf.cast(exp(transf, 'transf_exp'),
                                     dtype=TF_FLOAT)

            with tf.name_scope('proposed'):
                v = (v * scale_exp
                     - 0.5 * self.eps * (grad * transf_exp - transl))

            logdet = tf.reduce_sum(scale, axis=1, name='logdet_vf')

        return v, logdet, fns

    def _update_x_forward(self, x, v, t, weights, training, masks):
        """Update x in the forward leapfrog step."""
        mask, mask_inv = masks
        with tf.name_scope('update_xf'):
            scale, transl, transf = self.x_fn([v, mask * x, t], training)

            with tf.name_scope('xf_mul'):
                scale *= self.eps * weights[0]
                transl *= weights[1]
                transf *= self.eps * weights[2]
                fns = [scale, transl, transf]

            with tf.name_scope('xf_exp'):
                scale_exp = exp(scale, 'scale_exp')
                transf_exp = exp(transf, 'transformation_exp')

            with tf.name_scope('proposed'):
                x = (mask * x + mask_inv
                     * (x * scale_exp + self.eps * (v * transf_exp + transl)))

            logdet = tf.reduce_sum(mask_inv * scale, axis=1, name='logdet_xf')

        return x, logdet, fns

    def _update_v_backward(self, x, v, beta, t, weights, training):
        """Update v in the backward leapfrog step. Invert the forward update"""
        with tf.name_scope('update_vb'):
            grad = self.grad_potential(x, beta)

            scale, transl, transf = self.v_fn([x, grad, t], training)

            with tf.name_scope('vb_mul'):
                scale *= -0.5 * self.eps * weights[0]
                transl *= weights[1]
                transf *= self.eps * weights[2]
                fns = [scale, transl, transf]

            with tf.name_scope('vb_exp'):
                scale_exp = exp(scale, 'scale_exp')
                transf_exp = exp(transf, 'transformation_exp')

            with tf.name_scope('proposed'):
                v = scale_exp * (v + 0.5 * self.eps
                                 * (grad * transf_exp - transl))

            logdet = tf.reduce_sum(scale, axis=1, name='logdet_vb')

        return v, logdet, fns

    def _update_x_backward(self, x, v, t, weights, training, masks):
        """Update x in the backward lf step. Inverting the forward update."""
        mask, mask_inv = masks
        with tf.name_scope('update_xb'):
            scale, transl, transf = self.x_fn([v, mask * x, t], training)

            with tf.name_scope('xb_mul'):
                scale *= -self.eps * weights[0]
                transl *= weights[1]
                transf *= self.eps * weights[2]
                fns = [scale, transl, transf]

            with tf.name_scope('xb_exp'):
                scale_exp = exp(scale, 'xb_scale')
                transf_exp = exp(transf, 'xb_transformation')

            with tf.name_scope('proposed'):
                x = (mask * x + mask_inv * scale_exp
                     * (x - self.eps * (v * transf_exp + transl)))

            logdet = tf.reduce_sum(mask_inv * scale, axis=1,
                                   name='logdet_xb')

        return x, logdet, fns

    def _compute_accept_prob(self, xi, vi, xf, vf, sumlogdet, beta, hmc=True):
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
        with tf.name_scope('accept_prob'):
            with tf.name_scope('old_hamiltonian'):
                old_hamil = self.hamiltonian(xi, vi, beta)
            with tf.name_scope('new_hamiltonian'):
                new_hamil = self.hamiltonian(xf, vf, beta)

            with tf.name_scope('calc_prob'):
                prob = exp(tf.minimum(
                    (old_hamil - new_hamil + sumlogdet), 0.
                ), 'accept_prob')

            # Ensure numerical stability as well as correct gradients
            accept_prob = tf.where(tf.is_finite(prob), prob,
                                   tf.zeros_like(prob))
        if hmc:
            prob_hmc = self._compute_accept_prob_hmc(old_hamil, new_hamil)
        else:
            prob_hmc = tf.zeros_like(accept_prob)

        return accept_prob, prob_hmc

    def _compute_accept_prob_hmc(self, old_hamil, new_hamil):
        """Compute the prob. of accepting the proposed state given old state.

        NOTE: This computes the accept prob. for generic HMC.
        """
        with tf.name_scope('accept_prob_hmc'):
            prob = exp(tf.minimum((old_hamil - new_hamil), 0.), 'accept_prob')
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

    def _build_masks(self):
        """Construct different binary masks for different time steps."""
        masks = []
        for _ in range(self.num_steps):
            # Need to use npr here because tf would generate different random
            # values across different `sess.run`
            idx = npr.permutation(np.arange(self.x_dim))[:self.x_dim // 2]
            mask = np.zeros((self.x_dim,))
            mask[idx] = 1.
            mask = tf.constant(mask, dtype=TF_FLOAT)
            masks.append(mask[None, :])

        return masks

    def _get_mask(self, step):
        with tf.name_scope('get_mask'):
            m = tf.gather(self.masks, tf.cast(step, dtype=TF_INT))
            _m = 1. - m  # complementary mask
        return m, _m

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
