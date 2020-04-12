'''
base_model.py

Implements BaseModel class.

# noqa: E501


References:
-----------
# pylint:disable=line-too-long
[1] https://infoscience.epfl.ch/record/264887/files/robust_parameter_estimation.pdf


Author: Sam Foreman (github: @saforem2)
Date: 08/28/2019
'''
from __future__ import absolute_import, division, print_function

from collections import namedtuple

import numpy as np
import tensorflow as tf

import config as cfg
import utils.file_io as io

from seed_dict import seeds
from dynamics.dynamics import Dynamics
from utils.horovod_utils import warmup_lr

if cfg.HAS_HOROVOD:
    import horovod.tensorflow as hvd  # pylint: disable=import-error

TF_FLOAT = cfg.TF_FLOAT
TF_INT = cfg.TF_INT

State = cfg.State

LFdata = namedtuple('LFdata', ['init', 'proposed', 'prob'])
EnergyData = namedtuple('EnergyData', ['init', 'proposed', 'out',
                                       'proposed_diff', 'out_diff'])
SamplerData = namedtuple('SamplerData', ['data', 'dynamics_output'])


# pylint:disable=invalid-name
# pylint:disable=no-member

def _gaussian(x, mu, sigma):
    norm = tf.cast(
        1. / tf.sqrt(2 * np.pi * sigma ** 2), dtype=TF_FLOAT
    )
    exp_ = tf.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return norm * exp_


def add_to_collection(collection, tensors):
    """Helper method for adding a list of `tensors` to `collection`."""
    _ = [tf.add_to_collection(collection, tensor) for tensor in tensors]


# pylint:disable=too-many-instance-attributes, attribute-defined-outside-init
# pylint:disable=too-many-locals
class BaseModel:
    """BaseModel provides the necessary tools for training the L2HMC sampler.

    Explicitly, it serves as an abstract base class that should be
    extended through inheritance (see, for example, the `GaugeModel` and
    `GaussianMixtureModel` objects defined in the `models/` directory).

    This class is responsible for building both the loss function to be
    minimized, as well as the tensorflow operations for backpropagating the
    gradients for each training step.

    Additionally, it provides an external interface for generating new
    samples through it's `x_out` attribute.
    """
    def __init__(self, params=None, model_type=None):
        """Initialization method.

        Args:
            params (dict, optional): Dictionary of key, value pairs used for
                specifying model parameters.
        """
        self._model_type = model_type
        self._parse_params(params)
        self.global_step = self._create_global_step()
        self.lr = self._create_lr(self._warmup)
        self.optimizer = self._create_optimizer()

    def _parse_params(self, params):
        """Parse input parameters."""
        self.params = params
        # Directory to store model information
        self.log_dir = params.get('log_dir', None)
        # Number of leapfrog steps to use in MD updates
        self.num_steps = int(params.get('num_steps', 5))
        # Start from random initial samples?
        self.rand = params.get('rand', True)
        # Initial value of beta to use in annealing schedule
        self.beta_init = params.get('beta_init', None)
        # Final value of beta to use in annealing schedule
        self.beta_final = params.get('beta_final', None)
        # number of training steps
        self.train_steps = int(params.get('train_steps', None))
        # batch size to use for training
        self.batch_size = int(params.get('batch_size', None))
        # Activation function to be used in the network
        self._activation = params.get('activation', 'relu')
        # Num nodes in first and second hidden layers, respectively
        self.num_hidden1 = int(params.get('num_hidden1', None))
        self.num_hidden2 = int(params.get('num_hidden2', None))
        # Using `horovod` for distributed training?
        self.using_hvd = params.get('using_hvd', False)
        # Use alternative loss functions?
        self._use_gaussian = params.get('use_gaussian_loss', False)
        self._use_nnehmc = params.get('use_nnhehmc_loss', False)
        # Run generic HMC instead of training the L2HMC sampler?
        self.hmc = params.get('hmc', False)
        # Train the sampler with a fixed step size (eps)?
        self.eps_trainable = (not params.get('eps_fixed', False))
        # Warmup learning rate? (slowly ramp it up at the start of training)
        self._warmup = params.get('warmup_lr', False)
        # How to decay learning rate during training
        self.lr_decay_steps = int(params.get('lr_decay_steps', 10000))
        self.lr_decay_rate = params.get('lr_decay_rate', 0.96)
        # Scaling factor for generic esjd loss
        self.std_weight = float(params.get('std_weight', 1.))
        # Overall scaling factor for loss function
        self.loss_scale = float(params.get('loss_scale', 1.))
        # weight of auxiliary sampler that draws from initialization dist
        self.aux_weight = float(params.get('aux_weight', 1.))
        # whether or not to use `zero_masks` in x-updates
        self.zero_masks = params.get('zero_masks', False)
        # whether or not to use batch normalization in network
        self.use_bn = params.get('use_bn', False)
        # whether or not to use dropout in network
        self.dropout_prob = params.get('dropout_prob', 0.)
        # gradient clipping value (by global norm)
        self.clip_value = params.get('clip_value', 0.)
        # initial value of learning rate
        self.lr_init = params.get('lr_init', 1e-3)
        # number of steps after which to save model
        self.print_steps = params.get('print_steps', 1)
        # number of steps after which to print output
        self.save_steps = params.get('save_steps', 10000)
        # network architecture
        self.network_arch = params.get('network_arch', 'generic')
        # network_type: 'CartesianNet' or if None, use `FullNet`
        self._network_type = params.get('network_type', None)

        # save values taken on by leapfrog functions in summaries?
        #  self.save_lf = params.get('save_lf', False)

        # All reqd. params should have already been processed, but to be sure
        self.loss_weights = {}
        for key, val in self.params.items():
            if 'weight' in key:
                self.loss_weights[key] = val
            else:
                setattr(self, key, val)

    def build(self, params=None):
        """Build `tf.Graph` object containing operations for running model."""
        raise NotImplementedError

    def _build_inputs(self):
        """Build inputs and their respective placeholders."""
        io.log(f'INFO: Creating input placeholders...')
        inputs = self._create_inputs()
        # pylint: disable=attribute-defined-outside-init
        self._inputs = inputs
        self.x = inputs['x']
        x_shape = self.x.get_shape().as_list()
        self.batch_size = x_shape[0]
        self.x_dim = x_shape[1:]
        #  self.batch_size = self.x.shape[0]
        #  self.x_dim = self.x.shape[1:]
        self.beta = inputs['beta']
        self.eps_ph = inputs['eps_ph']
        self.net_weights = inputs['net_weights']
        self.train_phase = inputs['train_phase']
        self.global_step_ph = inputs['global_step_ph']

        # create global step setter
        self.global_step_setter = self._build_global_step_setter()

    def _build(self):
        """Helper method for building model.

        NOTE: This method builds those operations which are common to all
        models.
        """
        # ********************************************************
        # Create dynamics for running the augmented L2HMC sampler
        # --------------------------------------------------------
        io.log('INFO: Creating `Dynamics`...')
        self.dynamics = self.create_dynamics()
        self.dynamics_eps = self.dynamics.eps
        # Create operation for assigning to `dynamics.eps`
        # the value fed into the placeholder `eps_ph`.
        self.eps_setter = self._build_eps_setter()

        # **********************************************************
        # Create metric function for measuring distance b/t configs
        # ----------------------------------------------------------
        if self._model_type == 'GaugeModel':
            metric = getattr(self, 'metric', 'cos_diff')
        else:
            metric = getattr(self, 'metric', 'l2')

        with tf.name_scope('metric_fn'):
            self.metric_fn = self._create_metric_fn(metric)

        # *****************************************
        # Build sampler for obtaining new configs
        # -----------------------------------------
        xdata, zdata = self._build_sampler()

        # *******************************************************************
        # Build energy_ops to calculate energies.
        # -------------------------------------------------------------------
        with tf.name_scope('energy_ops'):
            self.v_ph = tf.placeholder(dtype=TF_FLOAT,
                                       shape=self.x.shape,
                                       name='v_placeholder')
            self.sumlogdet_ph = tf.placeholder(dtype=TF_FLOAT,
                                               shape=self.x.shape[0],
                                               name='sumlogdet_placeholder')
            self.state = State(x=self.x, v=self.v_ph, beta=self.beta)

            ph_str = 'energy_placeholders'
            _ = [tf.add_to_collection(ph_str, i) for i in self.state]
            tf.add_to_collection(ph_str, self.sumlogdet_ph)
            self.energy_ops = self._calc_energies(self.state,
                                                  self.sumlogdet_ph)

        # *******************************************************************
        # Calculate loss_op and train_op to backprop. grads through network
        # -------------------------------------------------------------------
        with tf.name_scope('calc_loss'):
            self.loss_op, self._losses_dict = self.calc_loss(xdata, zdata)

        # *******************************************************************
        # Calculate gradients and build training operation
        # -------------------------------------------------------------------
        with tf.name_scope('train'):
            io.log(f'INFO: Calculating gradients for backpropagation...')
            self.grads = self._calc_grads(self.loss_op)
            self.train_op = self._apply_grads(self.loss_op, self.grads)
            self.train_ops = self._build_train_ops()

        # *******************************************************************
        # Build `run_ops` containing ops used when running inference.
        # -------------------------------------------------------------------
        io.log(f'Collecting inference operations...')
        self.run_ops = self._build_run_ops()

    def _build_eps_setter(self):
        """Create op that sets `eps` to be equal to the value in `eps_ph`."""
        with tf.name_scope('build_eps_setter'):
            return tf.assign(self.dynamics.eps, self.eps_ph, name='eps_setter')

    def _build_global_step_setter(self):
        """Create op that sets the tensorflow `global_step`."""
        with tf.name_scope('build_global_step_setter'):
            return tf.assign(self.global_step,
                             self.global_step_ph,
                             name='global_step_setter')

    def _calc_energies(self, state, sumlogdet=0.):
        """Create operations for calculating the PE, KE and H of a `state`."""
        with tf.name_scope('calc_energies'):
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

    def _build_train_ops(self):
        """Build `train_ops` used for training the model."""
        if self.hmc:
            train_ops = {}
        else:
            train_ops = {
                'loss_op': self.loss_op,
                'train_op': self.train_op,
                'x_out': self.x_out,
                'dx_proposed': self.dx_proposed,
                'dx_out': self.dx_out,
                'sumlogdet': self.sumlogdet_out,
                'exp_energy_diff': self.exp_energy_diff,
                'px': self.px,
                'lr': self.lr,
                'dynamics_eps': self.dynamics.eps,
                #  'direction': self._direction
            }

        for val in train_ops.values():
            tf.add_to_collection('train_ops', val)

        return train_ops

    def _build_run_ops(self):
        run_ops = {
            'x_init': self.x_init,
            'v_init': self.v_init,
            'x_proposed': self.x_proposed,
            'v_proposed': self.v_proposed,
            'x_out': self.x_out,
            'v_out': self.v_out,
            'dx_out': self.dx_out,
            'dx_proposed': self.dx_proposed,
            'exp_energy_diff': self.exp_energy_diff,
            'accept_prob': self.px,
            'accept_prob_hmc': self.px_hmc,
            'sumlogdet_proposed': self.sumlogdet_proposed,
            'sumlogdet_out': self.sumlogdet_out,
        }
        for val in run_ops.values():
            tf.add_to_collection('run_ops', val)

        return run_ops

    def _build_sampler(self):
        """Build operations used for sampling from the dynamics engine."""
        with tf.name_scope('l2hmc_sampler'):
            x_dynamics, xdata = self._build_main_sampler()
            self.x_init = x_dynamics['x_init']
            self.v_init = x_dynamics['v_init']
            self.x_out = x_dynamics['x_out']
            self.v_out = x_dynamics['v_out']
            self.x_proposed = x_dynamics['x_proposed']
            self.v_proposed = x_dynamics['v_proposed']
            self.px = x_dynamics['accept_prob']
            self.px_hmc = x_dynamics['accept_prob_hmc']
            self.sumlogdet_proposed = x_dynamics['sumlogdet_proposed']
            self.sumlogdet_out = x_dynamics['sumlogdet_out']
            #  self._direction = tf.cast(x_dynamics['direction'],
            #                            dtype=TF_FLOAT)
            self.dx_proposed = self.metric_fn(self.x_proposed, self.x_init)
            self.dx_out = self.metric_fn(self.x_out, self.x_init)
            h_init = self.dynamics.hamiltonian(self.x_init,
                                               self.v_init,
                                               self.beta)
            h_out = self.dynamics.hamiltonian(self.x_out,
                                              self.v_out,
                                              self.beta)
            self.exp_energy_diff = tf.exp(h_init - h_out)

            self.x_diff, self.v_diff = self._check_reversibility()

            _, zdata = self._build_aux_sampler()

        return xdata, zdata

    def _build_main_sampler(self):
        """Build operations used for 'sampling' from the dynamics engine."""
        with tf.name_scope('main_sampler'):
            xout = self.dynamics.apply_transition(self.x, self.beta,
                                                  self.net_weights,
                                                  self.train_phase,
                                                  hmc=self._use_nnehmc)

            xdata = LFdata(xout['x_init'],
                            xout['x_proposed'],
                            xout['accept_prob'])

        return xout, xdata

    def _build_aux_sampler(self):
        """Run dynamics using initialization distribution (random normal)."""
        aux_weight = getattr(self, 'aux_weight', 1.)
        with tf.name_scope('aux_sampler'):
            if aux_weight == 0.:
                return {}, LFdata(0., 0., 0.)

            self.z = tf.random_normal(tf.shape(self.x),
                                      dtype=TF_FLOAT,
                                      seed=seeds['z'],
                                      name='z')

            zout = self.dynamics.apply_transition(self.z, self.beta,
                                                  self.net_weights,
                                                  self.train_phase,
                                                  hmc=self._use_nnehmc)

            zdata = LFdata(zout['x_init'],
                            zout['x_proposed'],
                            zout['accept_prob'])

        return zout, zdata

    def create_dynamics(self):
        """Wrapper method around `self._create_dynamics`."""
        raise NotImplementedError

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
                'eps_trainable': not getattr(self, 'eps_fixed', False),
                'x_dim': self.x_dim,
                'batch_size': self.batch_size,
                'zero_masks': self.zero_masks,
            })
            kwargs.update(params)

            dynamics = Dynamics(potential_fn=potential_fn, **kwargs)

        tf.add_to_collection('dynamics_eps', dynamics.eps)

        return dynamics

    @staticmethod
    def _create_global_step():
        """Create global_step tensor."""
        with tf.variable_scope('global_step'):
            global_step = tf.train.get_or_create_global_step()
        return global_step

    def _create_lr(self, warmup=False):
        """Create learning rate."""
        if self.hmc:
            return None

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
            self._create_lr(warmup=False)

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
                net_weights = cfg.NetWeights(x_scale=x_scale_weight,
                                             x_translation=x_transl_weight,
                                             x_transformation=x_transf_weight,
                                             v_scale=v_scale_weight,
                                             v_translation=v_transl_weight,
                                             v_transformation=v_transf_weight)
                train_phase = make_ph('is_training', dtype=tf.bool)
                eps_ph = make_ph('eps_ph')
                global_step_ph = make_ph('global_step_ph', dtype=tf.int64)

            inputs = {
                'x': x,
                'beta': beta,
                'eps_ph': eps_ph,
                'global_step_ph': global_step_ph,
                'train_phase': train_phase,
                'net_weights': net_weights,
            }
            for key, val in inputs.items():
                print(f'{key}: {val}\n')

        for key, val in inputs.items():
            if key == 'net_weights':
                _ = [tf.add_to_collection('inputs', v) for v in val]
            else:
                tf.add_to_collection('inputs', val)

        return inputs

    def _create_metric_fn(self, metric):
        """Create metric function used to measure distatnce between configs."""
        if metric == 'l1':
            def metric_fn(x1, x2):
                return tf.abs(x1 - x2)
        elif metric == 'l2':
            def metric_fn(x1, x2):
                return tf.square(x1 - x2)
        elif metric == 'cos_diff':
            def metric_fn(x1, x2):
                return 1. - tf.cos(x1 - x2)
        else:
            raise ValueError(f'metric={metric}. Expected one of:\n'
                             '`\tl1`, `l2`, or `cos_diff`.')

        return metric_fn

    def _check_reversibility(self):
        with tf.name_scope('reversibility_check'):
            x_in = tf.random_normal(self.x.shape,
                                    dtype=TF_FLOAT,
                                    seed=seeds['x_reverse_check'],
                                    name='x_reverse_check')
            v_in = tf.random_normal(self.x.shape,
                                    dtype=TF_FLOAT,
                                    seed=seeds['v_reverse_check'],
                                    name='v_reverse_check')

            with tf.name_scope('forward'):
                outputs_f = self.dynamics.transition_kernel(x_in, v_in,
                                                            self.beta,
                                                            self.net_weights,
                                                            self.train_phase,
                                                            forward=True)
                xf = outputs_f['x_proposed']
                vf = outputs_f['v_proposed']

            with tf.name_scope('backward'):
                outputs_b = self.dynamics.transition_kernel(xf, vf,
                                                            self.beta,
                                                            self.net_weights,
                                                            self.train_phase,
                                                            forward=False)
                xb = outputs_b['x_proposed']
                vb = outputs_b['v_proposed']

            with tf.name_scope('calc_diffs'):
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

    def calc_dx(self):
        """Calc. the difference traveled between input and output configs."""
        with tf.name_scope('calc_dx'):
            if hasattr(self, 'xf'):
                with tf.name_scope('dxf'):
                    dxf = self.metric_fn(self.xf, self.x_init)
            if hasattr(self, 'xb'):
                with tf.name_scope('dxb'):
                    dxb = self.metric_fn(self.xb, self.x_init)

            dx = tf.reduce_mean((dxf + dxb) / 2, axis=1)

        return dx, dxf, dxb

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

    def _calc_loss(self, xdata, zdata):
        """Build operation responsible for calculating the total loss.

        Args:
            xdata (namedtuple): Contains `x_in`, `x_proposed`, and `px`.
            zdata (namedtuple): Contains `z_in`, `z_propsed`, and `pz`.'
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
                x_loss = self._loss(xdata.init,
                                    xdata.proposed,
                                    xdata.prob)
            with tf.name_scope('z_loss'):
                if aux_weight > 0.:
                    z_loss = self._loss(zdata.init,
                                        zdata.proposed,
                                        zdata.prob)
                else:
                    z_loss = 0.

            loss = tf.add(x_loss, z_loss, name='loss')

        return loss

    def _gaussian_loss(self, xdata, zdata, mean, sigma):
        """Alternative Gaussian loss implemntation."""
        ls = getattr(self, 'loss_scale', 1.)
        aux_weight = getattr(self, 'aux_weight', 1.)
        with tf.name_scope('gaussian_loss'):
            with tf.name_scope('x_loss'):
                x_esjd = self._calc_esjd(xdata.init,
                                         xdata.proposed,
                                         xdata.prob)
                x_gauss = _gaussian(x_esjd, mean, sigma)
                #  x_loss = - ls * tf.reduce_mean(x_gauss, name='x_gauss_mean')
                x_loss = ls * tf.reduce_mean(x_gauss, name='x_gauss_mean')
                #  x_loss = ls * tf.log(tf.reduce_mean(x_gauss))

            with tf.name_scope('z_loss'):
                if aux_weight > 0.:
                    z_esjd = self._calc_esjd(zdata.init,
                                             zdata.proposed,
                                             zdata.prob)
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

    def _nnehmc_loss(self, xdata, hmc_prob, beta=1., x_esjd=None):
        """Calculate the NNEHMC loss from [1] (line 10)."""
        if x_esjd is None:
            x_in, x_proposed, accept_prob = xdata
            x_esjd = self._calc_esjd(x_in, x_proposed, accept_prob)

        return tf.reduce_mean(- x_esjd - beta * hmc_prob, name='nnehmc_loss')

    def calc_loss(self, xdata, zdata, eps=1e-4):
        """Calculate the total loss."""
        raise NotImplementedError

    def calc_loss1(self, xdata, zdata):
        """Calculate the total loss from all terms."""
        total_loss = 0.
        ld = {}

        eps = 1e-4
        with tf.name_scope('loss'):
            if self._use_gaussian:
                gaussian_loss = self._gaussian_loss(xdata, zdata,
                                                    mean=0., sigma=1.)
                ld['gaussian'] = gaussian_loss
                total_loss += gaussian_loss

            if self._use_nnehmc:
                nnehmc_beta = getattr(self, 'nnehmc_beta', 1.)
                nnehmc_loss = self._nnehmc_loss(xdata, self.px_hmc,
                                                beta=nnehmc_beta)
                ld['nnehmc'] = nnehmc_loss
                total_loss += nnehmc_loss

            if self._model_type == 'GaugeModel':
                plaq_loss, charge_loss = self.plaq_loss(xdata, zdata, eps)
                total_loss += plaq_loss
                total_loss += charge_loss
                ld['charge'] = charge_loss
                ld['plaq'] = plaq_loss
                #  if self.use_charge_loss:
                #      charge_loss = self._calc_charge_loss(xdata, zdata)
                #      ld['charge'] = charge_loss
                #      total_loss += charge_loss

            # If not using either Gaussian loss or NNEHMC loss,
            # use standard loss
            if (not self._use_gaussian) and (not self._use_nnehmc):
                std_loss = self._calc_loss(xdata, zdata)
                ld['std'] = std_loss
                total_loss += std_loss

            tf.add_to_collection('losses', total_loss)

            fd = {k: v / total_loss for k, v in ld.items()}

            losses_dict = {}
            for key in ld.keys():
                losses_dict[key + '_loss'] = ld[key]
                losses_dict[key + '_frac'] = fd[key]

                tf.add_to_collection('losses', ld[key])

        return total_loss, losses_dict

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
        with tf.name_scope('apply_grads'):
            trainable_vars = tf.trainable_variables()
            #  grads_and_vars = zip(grads, self.dynamics.trainable_variables)
            grads_and_vars = zip(grads, trainable_vars)
            ctrl_deps = [loss_op, *self.dynamics.updates]
            with tf.control_dependencies(ctrl_deps):
                train_op = self.optimizer.apply_gradients(grads_and_vars,
                                                          self.global_step,
                                                          'train_op')
        return train_op

    def _extract_l2hmc_fns(self, fns):
        """Method for extracting each of the Q, S, T functions as tensors."""
        if not getattr(self, 'save_lf', True):
            return

        fnsT = tf.transpose(fns, perm=[2, 1, 0, 3, 4], name='fns_transposed')

        fn_names = ['scale', 'translation', 'transformation']
        update_names = ['v1', 'x1', 'x2', 'v2']

        l2hmc_fns = {}
        for idx, name in enumerate(fn_names):
            l2hmc_fns[name] = {}
            for subidx, subname in enumerate(update_names):
                l2hmc_fns[name][subname] = fnsT[idx][subidx]

        return l2hmc_fns
