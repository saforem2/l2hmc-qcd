'''
base_model.py

Implements BaseModel class.

# noqa: E501


References:
-----------
[1] https://infoscience.epfl.ch/record/264887/files/robust_parameter_estimation.pdf


Author: Sam Foreman (github: @saforem2)
Date: 08/28/2019
'''
from __future__ import absolute_import, division, print_function

import config as cfg
from seed_dict import seeds
from collections import namedtuple
from dynamics.dynamics import Dynamics

import numpy as np
import tensorflow as tf

from utils.horovod_utils import warmup_lr

if cfg.HAS_HOROVOD:
    import horovod.tensorflow as hvd


TF_FLOAT = cfg.TF_FLOAT

LFdata = namedtuple('LFdata', ['init', 'proposed', 'prob'])
EnergyData = namedtuple('EnergyData', ['init', 'proposed', 'out',
                                       'proposed_diff', 'out_diff'])
SamplerData = namedtuple('SamplerData', ['data', 'dynamics_output'])
#  NetWeights = namedtuple('NetWeights', [
#      'x_scale', 'x_translation', 'x_transformation',
#      'v_scale', 'v_translation', 'v_transformation']
#  )


def _gaussian(x, mu, sigma):
    norm = tf.cast(
        1. / tf.sqrt(2 * np.pi * sigma ** 2), dtype=TF_FLOAT
    )
    exp_ = tf.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return norm * exp_

def add_to_collection(collection, tensors):
    """Helper method for adding a list of `tensors` to `collection`."""
    _ = [tf.add_to_collection(collection, tensor) for tensor in tensors]


class BaseModel(object):
    """BaseModel provides the necessary tools for training the L2HMC sampler.

    Explicitly, it serves as an abstract base class that should be
    extended through inheritance (see, for example, the `GaugeModel` and
    `GaussianMixtureModel` objects defined in the `models/` directory.

    This class is responsible for building both the loss function to be
    minimized, as well as the tensorflow operations for backpropagating the
    gradients for each training step.

    Additionally, it provides an external interface for generating new
    samples through it's `x_out` attribute. 
    """
    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict, optional): Dictionary of key, value pairs used for
                specifying model parameters.
        """
        if 'charge_weight' in params:
            self.charge_weight_np = params.pop('charge_weight', None)

        self.params = params
        self.loss_weights = {}
        for key, val in self.params.items():
            if 'weight' in key:
                self.loss_weights[key] = val
            else:
                setattr(self, key, val)

        self.eps_trainable = not self.eps_fixed
        self.global_step = self._create_global_step()

        warmup = self.params.get('warmup_lr', False)
        self.lr = self._create_lr(warmup)

        self.optimizer = self._create_optimizer()

    def build(self):
        """Build `tf.Graph` object containing operations for running model."""
        raise NotImplementedError

    def _build_eps_setter(self):
        """Create op that sets `eps` to be equal to the value in `eps-ph`."""
        return tf.assign(self.dynamics.eps, self.eps_ph, name='eps_setter')

    def _calc_energies(self, state, sumlogdet=0.):
        """Create operations for calculating the PE, KE and H of a `state`."""
        pe = self.dynamics.potential_energy(state.x, state.beta)
        ke = self.dynamics.kinetic_energy(state.v)
        h = pe + ke + sumlogdet

        # Add these operations to the `energy_ops` collection
        # for easy-access when loading the saved graph when
        # running inference w/ the trained sampler
        ops = (pe, ke, h)
        for op in ops:
            tf.add_to_collection('energy_ops', op)

        return cfg.Energy(pe, ke, h)

    def _build_sampler(self):
        """Build operations used for sampling from the dynamics engine."""
        x_dynamics, x_data = self._build_main_sampler()
        self.x_out = x_dynamics['x_out']
        self.px = x_dynamics['accept_prob']
        self.px_hmc = x_dynamics['accept_prob_hmc']

        self.dynamics_dict = x_dynamics
        self.x_diff, self.v_diff = self._check_reversibility()

        _, z_data = self._build_aux_sampler()

        return x_data, z_data

    def _build_main_sampler(self):
        """Build operations used for 'sampling' from the dynamics engine."""
        with tf.name_scope('x_dynamics'):
            args = (self.x, self.beta, self.net_weights, self.train_phase)
            kwargs = {
                'hmc': getattr(self, 'use_nnehmc_loss', None),
                'model_type': getattr(self, 'model_type', None),
            }

            x_dynamics = self.dynamics.apply_transition(*args, **kwargs)

            x_data = LFdata(x_dynamics['x_init'],
                            x_dynamics['x_proposed'],
                            x_dynamics['accept_prob'])

        return x_dynamics, x_data

    def _build_aux_sampler(self):
        """Run dynamics using initialization distribution (random normal)."""
        aux_weight = getattr(self, 'aux_weight', 1.)
        with tf.name_scope('aux_dynamics'):
            if aux_weight > 0.:
                self.z = tf.random_normal(tf.shape(self.x),
                                          dtype=TF_FLOAT,
                                          seed=seeds['z'],
                                          name='z')

                args = (self.z, self.beta, self.net_weights, self.train_phase)
                kwargs = {
                    'hmc': getattr(self, 'use_nnehmc_loss', None),
                    'model_type': getattr(self, 'model_type', None),
                }

                z_dynamics = self.dynamics.apply_transition(*args, **kwargs)

                z_data = LFdata(z_dynamics['x_init'],
                                z_dynamics['x_proposed'],
                                z_dynamics['accept_prob'])
            else:
                z_dynamics = {}
                z_data = LFdata(0., 0., 0.)

        return z_dynamics, z_data

    def _create_dynamics(self, potential_fn, **params):
        """Create Dynamics Object."""
        with tf.name_scope('create_dynamics'):
            keys = ['eps', 'hmc', 'num_steps', 'use_bn',
                    'dropout_prob', 'network_arch',
                    'num_hidden1', 'num_hidden2']

            kwargs = {
                k: getattr(self, k, None) for k in keys
            }
            kwargs.update({
                'eps_trainable': not self.eps_fixed,
                'x_dim': self.x_dim,
                'batch_size': self.batch_size,
                '_input_shape': self.x.shape
            })
            kwargs.update(params)

            dynamics = Dynamics(potential_fn=potential_fn, **kwargs)

        tf.add_to_collection('dynamics_eps', dynamics.eps)

        return dynamics

    def _create_global_step(self):
        """Create global_step tensor."""
        with tf.variable_scope('global_step'):
            global_step = tf.train.get_or_create_global_step()
        return global_step

    def _create_lr(self, warmup=False):
        """Create learning rate."""
        if self.hmc:
            return

        lr_init = getattr(self, 'lr_init', 1e-3)

        with tf.name_scope('learning_rate'):
            # HOROVOD: When performing distributed training, it can be useful
            # to "warmup" the learning rate gradually, done using the
            # `warmup_lr` method below.
            if warmup:
                kwargs = {
                    'target_lr': lr_init,
                    'warmup_steps': 1000,  # change to be ~0.1 * train_steps
                    'global_step': self.global_step,
                    'decay_steps': self.lr_decay_steps,
                    'decay_rate': self.lr_decay_rate
                }
                lr = warmup_lr(**kwargs)
            else:
                lr = tf.train.exponential_decay(lr_init, self.global_step,
                                                self.lr_decay_steps,
                                                self.lr_decay_rate,
                                                staircase=True,
                                                name='learning_rate')
        return lr

    def _create_optimizer(self):
        """Create optimizer."""
        if not hasattr(self, 'lr'):
            self._create_lr(lr_init=self.lr_init, warmup=False)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
            if self.using_hvd:
                optimizer = hvd.DistributedOptimizer(optimizer)

        return optimizer

    def _create_inputs(self):
        """Create input paceholders (if not executing eagerly).
        Returns:
            outputs: Dictionary with the following entries:
                x: Placeholder for input lattice configuration with
                    shape = (batch_size, x_dim) where x_dim is the number of
                    links on the lattice and is equal to lattice.time_size *
                    lattice.space_size * lattice.dim.
                beta: Placeholder for inverse coupling constant.
                charge_weight: Placeholder for the charge_weight (i.e. alpha_Q,
                    the multiplicative factor that scales the topological
                    charge term in the modified loss function) .
                net_weights: Array of placeholders, each of which is a
                    multiplicative constant used to scale the effects of the
                    various S, Q, and T functions from the original paper.
                    net_weights[0] = 'scale_weight', multiplies the S fn.
                    net_weights[1] = 'transformation_weight', multiplies the Q
                    fn.  net_weights[2] = 'translation_weight', multiplies the
                    T fn.
                train_phase: Boolean placeholder indicating if the model is
                    curerntly being trained. 
        """
        def make_ph(name, shape=(), dtype=TF_FLOAT):
            return tf.placeholder(dtype=dtype, shape=shape, name=name)

        with tf.name_scope('inputs'):
            if not tf.executing_eagerly():
                x_shape = (self.batch_size, self.x_dim)
                x = make_ph(dtype=TF_FLOAT, shape=x_shape, name='x')
                beta = make_ph('beta')
                x_scale_weight = make_ph('x_scale_weight')
                x_transl_weight = make_ph('x_translation_weight')
                x_transf_weight = make_ph('x_transformation_weight')
                v_scale_weight = make_ph('v_scale_weight')
                v_transl_weight = make_ph('v_translation_weight')
                v_transf_weight = make_ph('v_transformation_weight')
                net_weights = cfg.NetWeights(v_scale=v_scale_weight,
                                             v_translation=v_transl_weight,
                                             v_transformation=v_transf_weight,
                                             x_scale=x_scale_weight,
                                             x_translation=x_transl_weight,
                                             x_transformation=x_transf_weight)
                train_phase = make_ph('is_training', dtype=tf.bool)
                eps_ph = make_ph('eps_ph')

            inputs = {
                'x': x,
                'beta': beta,
                'eps_ph': eps_ph,
                'train_phase': train_phase,
                'net_weights': net_weights,
                #  'x_scale_weight': x_scale_weight,
                #  'x_transl_weight': x_transl_weight,
                #  'x_transf_weight': x_transf_weight,
                #  'v_scale_weight': v_scale_weight,
                #  'v_transl_weight': v_transl_weight,
                #  'v_transf_weight': v_transf_weight,
            }
            for key, val in inputs.items():
                print(f'{key}: {val}\n')

        for key, val in inputs.items():
            if key == 'net_weights':
                _ = [tf.add_to_collection('inputs', v) for v in val]
            else:
                tf.add_to_collection('inputs', val)
        #  _ = [tf.add_to_collection('inputs', i) for i in inputs.values()]

        return inputs

    def _create_metric_fn(self, metric):
        """Create metric function used to measure distatnce between configs."""
        raise NotImplementedError

    def _check_reversibility(self):
        x_in = tf.random_normal(self.x.shape,
                                dtype=TF_FLOAT,
                                seed=seeds['x_reverse_check'],
                                name='x_reverse_check')
        v_in = tf.random_normal(self.x.shape,
                                dtype=TF_FLOAT,
                                seed=seeds['v_reverse_check'],
                                name='v_reverse_check')

        dynamics_check = self.dynamics._check_reversibility(x_in, v_in,
                                                            self.beta,
                                                            self.net_weights,
                                                            self.train_phase)
        xb = dynamics_check['xb']
        vb = dynamics_check['vb']

        xdiff = (x_in - xb)
        vdiff = (v_in - vb)
        x_diff = tf.reduce_sum(tf.matmul(tf.transpose(xdiff), xdiff))
        v_diff = tf.reduce_sum(tf.matmul(tf.transpose(vdiff), vdiff))

        return x_diff, v_diff

    def _calc_esjd(self, x1, x2, prob):
        """Compute the expected squared jump distance (ESJD)."""
        with tf.name_scope('esjd'):
            esjd = prob * tf.reduce_sum(self.metric_fn(x1, x2), axis=1)

        return esjd

    def _loss(self, init, proposed, prob):
        """Calculate the (standard) contribution to the loss from the ESJD."""
        ls = getattr(self, 'loss_scale', 1.)
        with tf.name_scope('calc_esjd'):
            esjd = self._calc_esjd(init, proposed, prob) + 1e-4  # no div. by 0

        loss = tf.reduce_mean((ls / esjd) - (esjd / ls))

        return loss

    def _alt_loss(self, init, proposed, prob):
        """Calculate the (standard) contribution to the loss from the ESJD."""
        ls = getattr(self, 'loss_scale', 1.)
        with tf.name_scope('calc_esjd'):
            esjd = self._calc_esjd(init, proposed, prob) + 1e-4  # no div. by 0

        loss = tf.reduce_mean(-esjd / ls)

        return loss

    def _calc_loss(self, x_data, z_data):
        """Build operation responsible for calculating the total loss.

        Args:
            x_data (namedtuple): Contains `x_in`, `x_proposed`, and `px`.
            z_data (namedtuple): Contains `z_in`, `z_propsed`, and `pz`.'
            weights (namedtuple): Contains multiplicative factors that
                determine the contribution from different terms to the total
                loss function.

        Returns:
            loss_op: Operation responsible for calculating the total loss (to
                be minimized).
        """
        aux_weight = getattr(self, 'aux_weight', 1.)

        with tf.name_scope('calc_loss'):
            with tf.name_scope('x_loss'):
                x_loss = self._loss(x_data.init,
                                    x_data.proposed,
                                    x_data.prob)
            with tf.name_scope('z_loss'):
                if aux_weight > 0.:
                    z_loss = self._loss(z_data.init,
                                        z_data.proposed,
                                        z_data.prob)
                else:
                    z_loss = 0.

            loss = tf.add(x_loss, z_loss, name='loss')

        return loss

    def _gaussian_loss(self, x_data, z_data, mean, sigma):
        """Alternative Gaussian loss implemntation."""
        ls = getattr(self, 'loss_scale', 1.)
        aux_weight = getattr(self, 'aux_weight', 1.)
        with tf.name_scope('gaussian_loss'):
            with tf.name_scope('x_loss'):
                x_esjd = self._calc_esjd(x_data.init,
                                         x_data.proposed,
                                         x_data.prob)
                x_gauss = _gaussian(x_esjd, mean, sigma)
                #  x_loss = - ls * tf.reduce_mean(x_gauss, name='x_gauss_mean')
                x_loss = ls * tf.reduce_mean(x_gauss, name='x_gauss_mean')
                #  x_loss = ls * tf.log(tf.reduce_mean(x_gauss))

            with tf.name_scope('z_loss'):
                if aux_weight > 0.:
                    z_esjd = self._calc_esjd(z_data.init,
                                             z_data.proposed,
                                             z_data.prob)
                    z_gauss = _gaussian(z_esjd, mean, sigma)
                    #  aux_factor = - ls * aux_weight
                    aux_factor = ls * aux_weight
                    z_loss = aux_factor * tf.reduce_mean(z_gauss,
                                                         name='z_gauss_mean')
                    #  z_loss = aux_factor * tf.log(tf.reduce_mean(z_gauss))
                else:
                    z_loss = 0.

            gaussian_loss = tf.add(x_loss, z_loss, name='loss')

        return gaussian_loss

    def _nnehmc_loss(self, x_data, hmc_prob, beta=1., x_esjd=None):
        """Calculate the NNEHMC loss from [1] (line 10)."""
        if x_esjd is None:
            x_in, x_proposed, accept_prob = x_data
            x_esjd = self._calc_esjd(x_in, x_proposed, accept_prob)

        return tf.reduce_mean(- x_esjd - beta * hmc_prob, name='nnehmc_loss')

    def _calc_grads(self, loss):
        """Calculate the gradients to be used in backpropagation."""
        clip_value = getattr(self, 'clip_value', 0.)
        with tf.name_scope('grads'):
            #  grads = tf.gradients(loss, self.dynamics.trainable_variables)
            grads = tf.gradients(loss, tf.trainable_variables())
            if clip_value > 0.:
                grads, _ = tf.clip_by_global_norm(grads, clip_value)

        return grads

    def _apply_grads(self, loss_op, grads):
        trainable_vars = tf.trainable_variables()
        #  grads_and_vars = zip(grads, self.dynamics.trainable_variables)
        grads_and_vars = zip(grads, trainable_vars)
        ctrl_deps = [loss_op, *self.dynamics.updates]
        with tf.control_dependencies(ctrl_deps):
            train_op = self.optimizer.apply_gradients(grads_and_vars,
                                                      self.global_step,
                                                      'train_op')
        return train_op
