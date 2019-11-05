"""
Dynamics engine for L2HMC sampler on Lattice Gauge Models.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

NOTE: Robust parameter estimation paper can be found at:
    https://infoscience.epfl.ch/record/264887/files/robust_parameter_estimation.pdf


Author: Sam Foreman (github: @saforem2)
Date: 1/14/2019
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import numpy.random as npr
from collections import namedtuple, OrderedDict

from network.network import FullNet
import config as cfg

__all__ = ['Dynamics']

TF_FLOAT = cfg.TF_FLOAT
TF_INT = cfg.TF_INT
GLOBAL_SEED = cfg.GLOBAL_SEED

# -----------------------
#   module-wide objects
# -----------------------
State = namedtuple('State', ['x', 'v', 'beta'])
EnergyData = namedtuple('EnergyData', ['init', 'proposed', 'out'])


def exp(x, name=None):
    """Safe exponential using tf.check_numerics."""
    return tf.check_numerics(tf.exp(x), f'{name} is NaN')


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
                         x_init,
                         beta,
                         weights,
                         train_phase,
                         model_type=None,
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

        Returns:
            x_proposed: Proposed x before accept/reject step.
            v_proposed: Proposed v before accept/reject step.
            accept_prob: Probability of accepting the proposed states.
            x_out: Samples after accept/reject step.
        """
        if model_type == 'GaugeModel':
            x_init = tf.mod(x_init, 2 * np.pi, name='x_in_mod_2_pi')

        # Call `self.transition_kernel` in the forward direction, 
        # starting from the initial `State`: `(x_init`, `v_init_f, beta)`
        # to get the proposed `State`
        with tf.name_scope('transition_forward'):
            v_init_f = tf.random_normal(tf.shape(x_init),
                                        dtype=TF_FLOAT,
                                        seed=GLOBAL_SEED,
                                        name='v_init_f')

            state_init_f = State(x_init, v_init_f, beta)
            args = (*state_init_f, weights, train_phase)
            kwargs = {'forward': True, 'hmc': hmc}
            outputs_f = self.transition_kernel(*args, **kwargs)
            xf = outputs_f['x_proposed']
            vf = outputs_f['v_proposed']
            pxf = outputs_f['accept_prob']
            pxf_hmc = outputs_f['accept_prob_hmc']
            sumlogdetf = outputs_f['sumlogdet']

        with tf.name_scope('transition_backward'):
            v_init_b = tf.random_normal(tf.shape(x_init),
                                        dtype=TF_FLOAT,
                                        seed=GLOBAL_SEED,
                                        name='v_init_b')

            state_init_b = State(x_init, v_init_b, beta)
            outputs_b = self.transition_kernel(*state_init_b,
                                               weights, train_phase,
                                               forward=False, hmc=hmc)
            xb = outputs_b['x_proposed']
            vb = outputs_b['v_proposed']
            pxb = outputs_b['accept_prob']
            pxb_hmc = outputs_b['accept_prob_hmc']
            sumlogdetb = outputs_b['sumlogdet']

        with tf.name_scope('simulate_forward_backward'):
            # Decide direction uniformly
            mask_f, mask_b = self._get_transition_masks()

            # Use forward/backward mask to reconstruct `v_init`
            v_init = (v_init_f * mask_f[:, None] + v_init_b * mask_b[:, None])

            # Obtain proposed states
            x_proposed = xf * mask_f[:, None] + xb * mask_b[:, None]
            v_proposed = vf * mask_f[:, None] + vb * mask_b[:, None]

            sumlogdet_proposed = sumlogdetf * mask_f + sumlogdetb * mask_b

            # Probability of accepting proposed states
            accept_prob = pxf * mask_f + pxb * mask_b
            # ---------------------------------------------------------------
            # NOTE: `accept_prob_hmc` can be ignored unless we are using the
            # NNEHMC loss function from the 'Robust Parameter Estimation...'
            # paper (see line 10 for a link)
            # ---------------------------------------------------------------
            # Probability of accepting proposed states (if `sumlogdet == 0`)
            accept_prob_hmc = pxf_hmc * mask_f + pxb_hmc * mask_b  # (!)

        # Accept or reject step
        with tf.name_scope('accept_reject_step'):
            # accept_mask, reject_mask
            mask_a, mask_r = self._get_accept_masks(accept_prob)

            # State (x, v)  after accept / reject step
            x_out = x_proposed * mask_a[:, None] + x_init * mask_r[:, None]
            v_out = v_proposed * mask_a[:, None] + v_init * mask_r[:, None]
            sumlogdet_out = sumlogdet_proposed * mask_a

        if model_type == 'GaugeModel':
            # Take `mod` operations here for calculating the energies
            x_proposed = tf.mod(x_proposed, 2 * np.pi,
                                name='x_proposed_mod_2pi')
            x_out = tf.mod(x_out, 2 * np.pi, name='x_out_mod_2_pi')

        with tf.name_scope('calc_energies'):
            with tf.name_scope('potential'):
                with tf.name_scope('init'):  # initial potential energy
                    pe_init = self.potential_energy(x_init, beta)
                with tf.name_scope('proposed'):  # proposed potential energy
                    pe_proposed = self.potential_energy(x_proposed, beta)
                with tf.name_scope('out'):  # output potential energy
                    pe_out = self.potential_energy(x_out, beta)

                # Create an `EnergyData` object for storing the energies and
                # add it to the `potential_energy` collection for easy-access
                # later
                pe_data = EnergyData(pe_init, pe_proposed, pe_out)
                _add_to_collection('potential_energy', pe_data)

            with tf.name_scope('kinetic'):
                with tf.name_scope('init'):  # initial kinetic energy
                    ke_init = self.kinetic_energy(v_init)
                with tf.name_scope('proposed'):  # proposed kinetic energy
                    ke_proposed = self.kinetic_energy(v_proposed)
                with tf.name_scope('out'):  # output kinetic energy
                    ke_out = self.kinetic_energy(v_out)

                ke_data = EnergyData(ke_init, ke_proposed, ke_out)
                _add_to_collection('kinetic_energy', ke_data)

            with tf.name_scope('hamiltonian'):
                with tf.name_scope('init'):  # initial hamiltonian
                    h_init = self.hamiltonian(x_init, v_init, beta)
                with tf.name_scope('proposed'):  # proposed hamiltonian
                    h_proposed = (self.hamiltonian(x_proposed,
                                                   v_proposed, beta)
                                  + sumlogdet_proposed)
                with tf.name_scope('out'):  # output hamiltonian
                    h_out = (self.hamiltonian(x_out, v_out, beta)
                             + sumlogdet_out)

                    h_data = EnergyData(h_init, h_proposed, h_out)
                    _add_to_collection('hamiltonian', h_data)

            energies = {
                'potential': pe_data,
                'kinetic': ke_data,
                'hamiltonian': h_data
            }

        outputs_fb = {
            'x_init': x_init,
            'v_init': v_init,
            'x_proposed': x_proposed,
            'v_proposed': v_proposed,
            'x_out': x_out,
            'v_out': v_out,
            'accept_prob': accept_prob,
            'accept_prob_hmc': accept_prob_hmc,
            'sumlogdet_proposed': sumlogdet_proposed,
            'sumlogdet_out': sumlogdet_out,
            'mask_f': mask_f,  # forward mask
            'mask_b': mask_b,  # backward mask
            'mask_a': mask_a,  # accept mask
            'mask_r': mask_r,  # reject mask
        }
        for val in outputs_fb.values():
            tf.add_to_collection('dynamics_out', val)

        outputs = {
            'outputs_fb': outputs_fb,
            'energies': energies,
        }

        return outputs

    def transition_kernel(self, x_in, v_in, beta,
                          weights, train_phase,
                          forward=True, hmc=True):
        """Transition kernel of augmented leapfrog integrator."""
        lf_fn = self._forward_lf if forward else self._backward_lf

        with tf.name_scope('init'):
            x_proposed, v_proposed = x_in, v_in

            step = tf.constant(0., name='md_step', dtype=TF_FLOAT)
            batch_size = tf.shape(x_in)[0]
            logdet = tf.zeros((batch_size,), dtype=TF_FLOAT)

        def body(step, x, v, logdet):
            # cast leapfrog step to integer
            #  i = tf.cast(step, dtype=tf.int32)
            new_x, new_v, j, _fns = lf_fn(x, v, beta, step,
                                          weights, train_phase)
            #  lf_samples = lf_samples.write(i+1, new_x)
            #  logdets = logdets.write(i+1, logdet+j)
            #  fns = fns.write(i, _fns)

            return (step+1, new_x, new_v, logdet+j)

        def cond(step, *args):
            return tf.less(step, self.num_steps)

        outputs = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[step, x_proposed, v_proposed, logdet]
        )

        step = outputs[0]
        x_proposed = outputs[1]
        v_proposed = outputs[2]
        sumlogdet = outputs[3]
        #  with tf.name_scope('MD_outputs'):
        #      with tf.name_scope('lf_out'):
        #          lf_out = outputs[4].stack()
        #      with tf.name_scope('logdets_out'):
        #          logdets_out = outputs[5].stack()
        #      with tf.name_scope('l2hmc_fns_out'):
        #          fns_out = outputs[6].stack()

        accept_prob, accept_prob_hmc = self._compute_accept_prob(
            x_in, v_in, x_proposed, v_proposed, sumlogdet, beta, hmc=hmc
        )

        outputs = {
            'x_init': x_in,
            'v_init': v_in,
            'x_proposed': x_proposed,
            'v_proposed': v_proposed,
            'sumlogdet': sumlogdet,
            'accept_prob': accept_prob,
            'accept_prob_hmc': accept_prob_hmc,
        }

        #  if save_lf:
        #      outputs['lf_out'] = lf_out
        #      outputs['logdets'] = logdets_out
        #      outputs['fns_out'] = fns_out

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
            step_r = self.num_steps - step - 1

            # Reversed index/sinusoidal time
            with tf.name_scope('get_time'):
                t = self._get_time(step_r, tile=tf.shape(x)[0])

            with tf.name_scope('get_mask'):
                mask, mask_inv = self._get_mask(step_r)

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
                scale_exp = tf.cast(exp(scale, 'scale_exp'), TF_FLOAT)
                transf_exp = tf.cast(exp(transf, 'transf_exp'), TF_FLOAT)

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

            logdet = tf.reduce_sum(mask_inv * scale, axis=1, name='logdet_xb')

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
                h_init = self.hamiltonian(xi, vi, beta)  # initial H
            with tf.name_scope('new_hamiltonian'):
                h_proposed = self.hamiltonian(xf, vf, beta)

            with tf.name_scope('calc_prob'):
                dh = h_init - h_proposed + sumlogdet
                prob = tf.exp(tf.minimum(dh, 0.))

            # Ensure numerical stability as well as correct gradients
            accept_prob = tf.where(tf.is_finite(prob),
                                   prob, tf.zeros_like(prob))
        if hmc:
            prob_hmc = self._compute_accept_prob_hmc(h_init, h_proposed, beta)
        else:
            prob_hmc = tf.zeros_like(accept_prob)

        return accept_prob, prob_hmc

    def _compute_accept_prob_hmc(self, h_init, h_proposed, beta):
        """Compute the prob. of accepting the proposed state given old state.

        NOTE: This computes the accept prob. for generic HMC.
        """
        with tf.name_scope('accept_prob_hmc'):
            prob = tf.exp(tf.minimum(beta * (h_init - h_proposed), 0.))
            accept_prob = tf.where(tf.is_finite(prob),
                                   prob, tf.zeros_like(prob))

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

    def _check_reversibility(self, x_in, v_in, beta, weights, training):
        outputs_f = self.transition_kernel(x_in, v_in, beta,
                                           weights,
                                           training,
                                           forward=True)
        xf = outputs_f['x_proposed']
        vf = outputs_f['v_proposed']

        #  backward(forward(x, v)) --> x, v
        outputs_b = self.transition_kernel(xf, vf, beta,
                                           weights,
                                           training,
                                           forward=False)
        xb = outputs_b['x_proposed']
        vb = outputs_b['v_proposed']

        outputs = {
            'xf': xf,
            'vf': vf,
            'xb': xb,
            'vb': vb
        }

        return outputs

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
            if tf.executing_eagerly():
                m = self.masks[step]
            else:
                m = tf.gather(self.masks, tf.cast(step, dtype=TF_INT))

        return m, 1. - m

    def grad_potential(self, x, beta):
        """Get gradient of potential function at current location."""
        with tf.name_scope('grad_potential'):
            if tf.executing_eagerly():
                tfe = tf.contrib.eager
                grad = tfe.gradients_function(self.potential_energy,
                                              params=[0])(x, beta)[0]
            else:
                grad = tf.gradients(self.potential_energy(x, beta), x)[0]

        return grad

    def potential_energy(self, x, beta):
        """Compute potential energy using `self.potential` and beta."""
        with tf.name_scope('potential_energy'):
            potential_energy = beta * self.potential(x)

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
