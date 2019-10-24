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

from config import GLOBAL_SEED, HAS_HOROVOD, TF_FLOAT
from collections import namedtuple
from dynamics.dynamics import Dynamics

import numpy as np
import tensorflow as tf

from utils.horovod_utils import warmup_lr

#  import utils.file_io as io
#  from utils.distributions import quadratic_gaussian
#  from params.gmm_params import GMM_PARAMS
#  from params.gauge_params import GAUGE_PARAMS

if HAS_HOROVOD:
    import horovod.tensorflow as hvd

LFdata = namedtuple('LFdata', ['init', 'proposed', 'prob'])
EnergyData = namedtuple('EnergyData', ['init', 'proposed', 'out',
                                       'proposed_diff', 'out_diff'])
SamplerData = namedtuple('SamplerData', ['data', 'dynamics_output'])

PARAMS = {
    'hmc': False,
    'lr_init': 1e-3,
    'lr_decay_steps': 1000,
    'lr_decay_rate': 0.96,
    'train_steps': 5000,
    'using_hvd': False,
}


def _gaussian(x, mu, sigma):
    norm = tf.cast(
        1. / tf.sqrt(2 * np.pi * sigma ** 2), dtype=TF_FLOAT
    )
    exp_ = tf.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return norm * exp_


class BaseModel:

    def __init__(self, params=None):

        if 'charge_weight' in params:
            self.charge_weight_np = params.pop('charge_weight', None)

        self._eps_np = params.get('eps', None)

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
        eps_setter = tf.assign(self.dynamics.eps,
                               self.eps_ph, name='eps_setter')
        return eps_setter

    def _build_sampler(self, aux_weight=1.):
        """Build operations used for 'sampling' from the dynamics engine."""
        args = (self.beta, self.net_weights, self.train_phase)
        kwargs = {
            'save_lf': getattr(self, 'save_lf', None),
            'hmc': getattr(self, 'use_nnehmc_loss', None),
            'model_type': getattr(self, 'model_type', None)
        }

        # NOTE if not `self.use_nnehmc_loss`:
        #   `self.px_hmc = tf.zeros_like(self.px)`
        x_dynamics_out, x_data = self._build_main_sampler(*args, **kwargs)
        self._dynamics_out = x_dynamics_out

        x_md_out = x_dynamics_out['md_outputs']
        self.x_out = x_md_out['x_out']
        self.px = x_md_out['accept_prob']
        self.px_hmc = x_md_out['accept_prob_hmc']
        self._parse_dynamics_output(x_md_out)

        # Check reversibility by testing that `backward(forward(x)) == x`
        self.x_diff, self.v_diff = self._check_reversibility()

        # Calculate kinetic energy, potential energy, 
        # and the Hamiltonian at beginning and end of 
        # trajectory (both before and after accept/reject)
        pe_data, ke_data, h_data = self._check_energy(x_md_out)
        self._energy_data = {
            'pe': pe_data,
            'ke': ke_data,
            'h': h_data
        }

        #  pe_data, ke_data, = self._check_energy(x_dynamics_out['outputs_f'],
        #                                      x_dynamics_out['outputs_b'])

        #  self._energy_outputs_dict = energy_outputs

        x_output = SamplerData(x_data, x_md_out)

        # If not running generic HMC, build (aux) sampler 
        # that draws from initialization distribution. 
        hmc = getattr(self, 'hmc', False)
        if aux_weight > 0. and not hmc:
            kwargs['save_lf'] = False
            z_dynamics, z_data = self._build_aux_sampler(*args, **kwargs)
            z_output = SamplerData(z_data, z_dynamics)

            return x_output, z_output

        return x_output

    def _build_main_sampler(self, *args, **kwargs):
        """Build operations used for 'sampling' from the dynamics engine."""
        with tf.name_scope('x_dynamics'):
            x_dynamics_out = self.dynamics.apply_transition(self.x,
                                                            *args, **kwargs)
            x_md = x_dynamics_out['md_outputs']

            x_data = LFdata(x_md['x_init'],
                            x_md['x_proposed'],
                            x_md['accept_prob'])

        return x_dynamics_out, x_data

    def _build_aux_sampler(self, *args, **kwargs):
        """Run dynamics using initialization distribution (random normal)."""
        aux_weight = getattr(self, 'aux_weight', 1.)
        with tf.name_scope('aux_dynamics'):
            if aux_weight > 0.:
                self.z = tf.random_normal(tf.shape(self.x),
                                          dtype=TF_FLOAT,
                                          seed=GLOBAL_SEED,
                                          name='z')

                z_dynamics_out = self.dynamics.apply_transition(self.z,
                                                                *args,
                                                                **kwargs)
                z_dynamics = z_dynamics_out['md_outputs']

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

        return dynamics

    def _parse_dynamics_output(self, dynamics_output):
        raise NotImplementedError

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
                scale_weight = make_ph('scale_weight')
                transl_weight = make_ph('translation_weight')
                transf_weight = make_ph('transformation_weight')
                train_phase = make_ph('is_training', dtype=tf.bool)
                eps_ph = make_ph('eps_ph')

            inputs = {
                'x': x,
                'beta': beta,
                'scale_weight': scale_weight,
                'transl_weight': transl_weight,
                'transf_weight': transf_weight,
                'train_phase': train_phase,
                'eps_ph': eps_ph,
            }

        _ = [tf.add_to_collection('inputs', i) for i in inputs.values()]

        return inputs

    def _create_metric_fn(self, metric):
        """Create metric function used to measure distatnce between configs."""
        raise NotImplementedError

    def _calc_energies(self, dynamics_output, direction):
        """Compute Hamiltonian, KE, and PE by parsing `dynamics_output` dict.

        Args:
            dynamics_output (dict): Dictionary containing either the forward or
                backward outputs from either the `dynamics._transition_forward`
                or `dynamics._transition_backward` methods.

        Returns:
            energy_dict (dict): Dictionary containing initial values, final
                values and difference (final - initial) values of the
                Hamiltonian, Kinetic energy, and Potential Energy.
        """
        x_init = dynamics_output['x_init']
        v_init = dynamics_output['v_init']
        x_proposed = dynamics_output['x_proposed']
        v_proposed = dynamics_output['v_proposed']
        x_out = dynamics_output['x_out']
        v_out = dynamics_output['v_out']

        sumlogdet = dynamics_output['sumlogdet']

        with tf.name_scope(f'check_energy_{direction}'):
            with tf.name_scope('potential_energy'):
                pe_init = self.dynamics.potential_energy(x_init, self.beta)
                pe_out = self.dynamics.potential_energy(x_out, self.beta)
                pe_prop = self.dynamics.potential_energy(x_proposed, self.beta)

                pe_diff_out = pe_out - pe_init
                pe_diff_prop = pe_prop - pe_init

            with tf.name_scope('kinetic_energy'):
                ke_init = self.dynamics.kinetic_energy(v_init)
                ke_out = self.dynamics.kinetic_energy(v_out)
                ke_prop = self.dynamics.kinetic_energy(v_proposed)

                ke_diff_out = ke_out - ke_init
                ke_diff_prop = ke_prop - ke_init

            with tf.name_scope('hamiltonian'):
                h_init = ke_init + pe_init
                h_out = ke_out + pe_out
                h_prop = ke_prop + pe_prop

                h_diff_out = (h_out - h_init - sumlogdet)
                h_diff_prop = (h_prop - h_init - sumlogdet)

        outputs = {
            'potential_init': pe_init,
            'potential_out': pe_out,
            'potential_proposed': pe_prop,
            'potential_diff_out': pe_diff_out,
            'potential_diff_proposed': pe_diff_prop,
            'kinetic_init': ke_init,
            'kinetic_out': ke_out,
            'kinetic_proposed': ke_prop,
            'kinetic_diff_out': ke_diff_out,
            'kinetic_diff_proposed': ke_diff_prop,
            'hamiltonian_init': h_init,
            'hamiltonian_proposed': h_prop,
            'hamiltonian_out': h_out,
            'hamiltonian_diff_out': h_diff_out,
            'hamiltonian_diff_proposed': h_diff_prop,
        }

        for val in list(outputs.values()):
            tf.add_to_collection(f'energies', val)
            #  tf.add_to_collection(f'energies_{direction}', val)

        return outputs

    def _calc_potential_energies(self, md_outputs):
        pe_init = self.dynamics.potential_energy(
            md_outputs['x_init'], self.beta
        )
        pe_proposed = self.dynamics.potential_energy(
            md_outputs['x_proposed'], self.beta
        )
        pe_out = self.dynamics.potential_energy(
            md_outputs['x_out'], self.beta
        )

        pe_proposed_diff = pe_proposed - pe_init
        pe_out_diff = pe_out - pe_init

        pe_data = EnergyData(pe_init, pe_proposed, pe_out,
                             pe_proposed_diff, pe_out_diff)

        _ = [tf.add_to_collection('energies', e) for e in pe_data]

        return pe_data

    def _calc_kinetic_energies(self, md_outputs):
        ke_init = self.dynamics.kinetic_energy(md_outputs['v_init'])
        ke_proposed = self.dynamics.kinetic_energy(md_outputs['v_proposed'])
        ke_out = self.dynamics.kinetic_energy(md_outputs['v_out'])

        ke_proposed_diff = ke_proposed - ke_init
        ke_out_diff = ke_out - ke_init

        ke_data = EnergyData(ke_init, ke_proposed, ke_out,
                             ke_proposed_diff, ke_out_diff)

        _ = [tf.add_to_collection('energies', e) for e in ke_data]

        return ke_data

    def _calc_hamiltonians(self, pe_data, ke_data,
                           sumlogdet_proposed, sumlogdet_out):
        h_init = pe_data.init + ke_data.init
        h_proposed = pe_data.proposed + ke_data.proposed
        h_out = pe_data.out + ke_data.out

        h_proposed_diff = h_proposed - h_init + sumlogdet_proposed
        h_out_diff = h_out - h_init + sumlogdet_out

        h_data = EnergyData(h_init, h_proposed, h_out,
                            h_proposed_diff, h_out_diff)

        _ = [tf.add_to_collection('energies', e) for e in h_data]

        return h_data

    def _check_energy(self, md_outputs):
        pe_data = self._calc_potential_energies(md_outputs)
        ke_data = self._calc_kinetic_energies(md_outputs)
        h_data = self._calc_hamiltonians(pe_data, ke_data,
                                         md_outputs['sumlogdet_proposed'],
                                         md_outputs['sumlogdet_out'])

        return pe_data, ke_data, h_data

    def _check_reversibility(self):
        x_in = tf.random_normal(self.x.shape,
                                dtype=TF_FLOAT,
                                seed=GLOBAL_SEED,
                                name='x_reverse_check')
        v_in = tf.random_normal(self.x.shape,
                                dtype=TF_FLOAT,
                                seed=GLOBAL_SEED,
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
            grads = tf.gradients(loss, self.dynamics.trainable_variables)
            if clip_value > 0.:
                grads, _ = tf.clip_by_global_norm(grads, clip_value)

        return grads

    def _apply_grads(self, loss_op, grads):
        grads_and_vars = zip(grads, self.dynamics.trainable_variables)
        ctrl_deps = [loss_op, *self.dynamics.updates]
        with tf.control_dependencies(ctrl_deps):
            train_op = self.optimizer.apply_gradients(grads_and_vars,
                                                      self.global_step,
                                                      'train_op')
        return train_op
