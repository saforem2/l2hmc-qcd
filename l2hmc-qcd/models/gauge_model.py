"""
gauge_model.py

Implements `GaugeModel` class, inheriting from `BaseModel`.

Author: Sam Foreman (github: @saforem2)
Date: 09/04/2019
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

#  import os
import time

#  import numpy as np
import tensorflow as tf

from collections import namedtuple

from base.base_model import BaseModel
from dynamics.dynamics import Dynamics
from lattice.lattice import GaugeLattice
import utils.file_io as io
#  from utils.horovod_utils import warmup_lr
from config import GLOBAL_SEED, TF_FLOAT, TF_INT, HAS_HOROVOD
#  from .params import GAUGE_PARAMS
#  from tensorflow.python.ops import control_flow_ops as control_flow_ops

if HAS_HOROVOD:
    import horovod.tensorflow as hvd


class GaugeModel(BaseModel):
    def __init__(self, params=None):
        super(GaugeModel, self).__init__(params)
        self._model_type = 'GaugeModel'

        self.loss_weights = {}
        for key, val in params.items():
            if 'weight' in key and key != 'charge_weight':
                self.loss_weights[key] = val
            elif key == 'charge_weight':
                pass
            else:
                setattr(self, key, val)

        self.eps_trainable = not self.eps_fixed
        self.charge_weight_np = params['charge_weight']
        self.build()

    def build(self):
        """Build TensorFlow graph."""
        t0 = time.time()
        io.log(80 * '-')
        io.log(f'INFO: Building graph for `GaugeModel`...')
        with tf.name_scope('init'):
            # ***********************************************
            # Create `Lattice` object
            # -----------------------------------------------
            io.log(f'INFO: Creating lattice...')
            self.lattice = self._create_lattice()
            self.batch_size = self.lattice.samples.shape[0]
            self.x_dim = self.lattice.num_links
            # ***********************************************

            # ***********************************************
            # Create inputs as `tf.placeholders`
            # -----------------------------------------------
            io.log(f'INFO: Creating input placeholders...')
            self.inputs = self._create_inputs()
            self.x = self.inputs.x
            self.beta = self.inputs.beta
            #  self.charge_weight = self.inputs.charge_weight
            self.net_weights = self.inputs.net_weights
            self.train_phase = self.inputs.train_phase
            # ***********************************************

            # ***********************************************
            # Create dynamics for running L2HMC leapfrog
            # -----------------------------------------------
            io.log(f'INFO: Creating `Dynamics`...')
            self.dynamics = self._create_dynamics()

            # ***************************************************************
            # Create metric function for measuring 'distance' between configs
            # ---------------------------------------------------------------
            metric = getattr(self, 'metric', 'cos_diff')
            self.metric_fn = self._create_metric_fn(metric)

        # *******************************************************
        # Create operations for calculating lattice observables
        # -------------------------------------------------------
        io.log(f'INFO: Creating necessary operations...')
        self.observables = self._create_observables()
        self.plaq_sums_op = self.observables.plaq_sums_op
        self.actions_op = self.observables.actions_op
        self.plaqs_op = self.observables.plaqs_op
        self.avg_plaqs_op = self.observables.avg_plaqs_op
        self.charges_op = self.observables.charges_op
        # *******************************************************

        # *******************************************************************
        # Run dynamics (i.e. augmented leapfrog) to generate new configs 
        # -------------------------------------------------------------------
        with tf.name_scope('apply_transition'):
            with tf.name_scope('main_transition'):
                x_dynamics = self.dynamics.apply_transition(
                    self.x, self.beta, self.net_weights,
                    self.train_phase, save_lf=self.save_lf
                )
            if getattr(self, 'aux_weight', 1.) > 0:
                with tf.name_scope('aux_transition'):
                    self.z = tf.random_normal(tf.shape(self.x),
                                              dtype=TF_FLOAT,
                                              seed=GLOBAL_SEED,
                                              name='z')
                    z_dynamics = self.dynamics.apply_transition(
                        self.z, self.beta, self.net_weights,
                        self.train_phase, save_lf=False
                    )

            self.x_out = x_dynamics['x_out']
            self.px = x_dynamics['accept_prob']
            self._parse_dynamics_output(x_dynamics)

        with tf.name_scope('run_ops'):
            io.log(f'INFO: Building `run_ops`...')
            run_ops = self._build_run_ops()
        # *******************************************************************

        # *******************************************************************
        # Calculate loss_op and train_op to backprop. grads through network
        # -------------------------------------------------------------------
        if not self.hmc:
            with tf.name_scope('loss'):
                io.log(f'INFO: Calculating loss function...')
                lf_data = namedtuple('lf_data', ['x_in', 'x_proposed', 'prob'])
                x_data = lf_data(x_dynamics['x_in'],
                                 x_dynamics['x_proposed'],
                                 x_dynamics['accept_prob'])
                z_data = lf_data(z_dynamics['x_in'],
                                 z_dynamics['x_proposed'],
                                 z_dynamics['accept_prob'])
                self.loss_op = self.calc_loss(x_data, z_data,
                                              self.loss_weights)

            with tf.name_scope('train'):
                io.log(f'INFO: Calculating gradients for backpropagation...')
                self.grads = self._calc_grads(self.loss_op)
                self.train_op = self._apply_grads(self.loss_op, self.grads)

        train_ops = self._build_train_ops()
        self.ops_dict = {
            'run_ops': run_ops,
            'train_ops': train_ops
        }

        # Make `run_ops` and `train_ops` collections w/ their respective ops.
        for key, val in self.ops_dict.items():
            for op in list(val.values()):
                tf.add_to_collection(key, op)

        io.log(f'INFO: Done building graph. Took: {time.time() - t0}s')
        io.log(80 * '-')
        # *******************************************************************


    def _create_lattice(self):
        """Create GaugeLattice object."""
        with tf.name_scope('lattice'):
            lattice = GaugeLattice(time_size=self.time_size,
                                   space_size=self.space_size,
                                   dim=self.dim,
                                   link_type=self.link_type,
                                   num_samples=self.num_samples,
                                   rand=self.rand)

        return lattice

    def _create_dynamics(self, **params):
        """Create `Dynamics` object."""
        with tf.name_scope('create_dynamics'):
            dynamics_keys = [
                'eps', 'hmc', 'num_steps', 'use_bn',
                'dropout_prob', 'network_arch',
                'num_hidden1', 'num_hidden2'
            ]

            dynamics_params = {
                k: getattr(self, k, None) for k in dynamics_keys
            }

            dynamics_params.update({
                'eps_trainable': not self.eps_fixed,
                'num_filters': self.lattice.space_size,
                'x_dim': self.lattice.num_links,
                'batch_size': self.num_samples,
                '_input_shape': (self.num_samples, *self.lattice.links.shape),
            })

            dynamics_params.update(params)
            samples = self.lattice.samples_tensor
            potential_fn = self.lattice.get_potential_fn(samples)
            dynamics = Dynamics(potential_fn=potential_fn, **dynamics_params)

        return dynamics

    def _create_inputs(self):
        with tf.name_scope('inputs'):
            if not tf.executing_eagerly():
                def scalar_ph(name, dtype=TF_FLOAT):
                    return tf.placeholder(dtype=dtype, shape=(), name=name)

                x = tf.placeholder(dtype=TF_FLOAT,
                                   shape=(self.batch_size, self.x_dim),
                                   name='x')

                beta = scalar_ph('beta')
                train_phase = scalar_ph('is_training', dtype=tf.bool)
                #  charge_weight = scalar_ph('charge_weight')
                scale_weight = scalar_ph('scale_weight')
                transl_weight = scalar_ph('translation_weight')
                transf_weight = scalar_ph('transformation_weight')
                net_weights = [scale_weight, transl_weight, transf_weight]

        Inputs = namedtuple('Inputs',
                            ['x', 'beta', 'net_weights', 'train_phase'])
        #  inputs = Inputs(x, beta, charge_weight, net_weights, train_phase)
        inputs = Inputs(x, beta, net_weights, train_phase)

        return inputs

    def _create_observables(self):
        """Create operations for calculating lattice observables."""
        with tf.name_scope('observables'):
            plaq_sums = self.lattice.calc_plaq_sums(self.x)
            actions = self.lattice.calc_actions(plaq_sums=plaq_sums)
            plaqs = self.lattice.calc_plaqs(plaq_sums=plaq_sums)
            avg_plaqs = tf.reduce_mean(plaqs, name='avg_plaqs')
            charges = self.lattice.calc_top_charges(plaq_sums=plaq_sums)

        Observables = namedtuple('Observables',
                                 ['plaq_sums_op', 'actions_op',
                                  'plaqs_op', 'avg_plaqs_op', 'charges_op'])
        observables = Observables(plaq_sums, actions,
                                  plaqs, avg_plaqs, charges)

        return observables

    def _create_metric_fn(self, metric):
        """Create `metric_fn` for measuring the `distance` between configs."""
        if metric == 'l1':
            def metric_fn(x1, x2):
                return tf.abs(x1 - x2)
        elif metric == 'l1':
            def metric_fn(x1, x2):
                return tf.square(x1 - x2)
        elif metric == 'cos_diff':
            def metric_fn(x1, x2):
                return 1. - tf.cos(x1 - x2)
        else:
            raise ValueError(f'metric={metric}. Expected one of:\n'
                             '`\tl1`, `l2`, or `cos_diff`.')

        return metric_fn

    def _calc_loss(self, x_data, z_data, weights):
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
        ls = getattr(self, 'loss_scale', 0.1)

        def _diff(x1, x2):
            return tf.reduce_sum(self.metric_fn(x1, x2), axis=1)

        #  aux_weight = weights.get('aux_weight', 1.)
        with tf.name_scope('calc_loss'):
            with tf.name_scope('x_loss'):
                x_loss = (x_data.prob * _diff(x_data.x_in,
                                              x_data.x_proposed)) + 1e-4
            with tf.name_scope('z_loss'):
                z_loss = (z_data.prob * _diff(z_data.x_in,
                                              z_data.x_proposed)) + 1e-4

            loss = 0.
            loss += ls * (tf.reduce_mean(1. / x_loss)
                          + tf.reduce_mean(1. / z_loss))
            loss += (- tf.reduce_mean(x_loss) - tf.reduce_mean(z_loss)) / ls

        return loss

    def _charge_loss(self, x_init, x_proposed, prob):
        dq = self.lattice.calc_top_charges_diff(x_init, x_proposed)
        charge_loss = prob * dq

        return charge_loss

    def _calc_charge_loss(self, x_data, z_data, weights):
        """Calculate the total charge loss."""
        aux_weight = weights.get('aux_weight', 1.)
        ls = self.loss_scale
        with tf.name_scope('calc_charge_loss'):
            with tf.name_scope('xq_loss'):
                xq_loss = self._charge_loss(*x_data)

            with tf.name_scope('zq_loss'):
                if aux_weight > 0.:
                    zq_loss = self._charge_loss(*z_data)
                else:
                    zq_loss = 0.

            charge_loss = 0.
            charge_loss += tf.reduce_mean((xq_loss + zq_loss) / ls,
                                          axis=0, name='charge_loss')

        return charge_loss

    def calc_loss(self, x_data, z_data, weights):
        """Calculate the total loss from all terms."""
        charge_weight = getattr(self, 'charge_weight_np', 1.)
        #  charge_weight = weights.get('charge_weight', 1.)

        loss = self._calc_loss(x_data, z_data)
        if charge_weight > 0:
            loss += self._calc_charge_loss(x_data, z_data, weights)

        return loss

    def _parse_dynamics_output(self, dynamics_output):
        """Parse output dictionary from `self.dynamics.apply_transition`."""
        with tf.name_scope('top_charge_diff'):
            x_in = dynamics_output['x_in']
            x_out = dynamics_output['x_out']
            x_dq = tf.cast(
                self.lattice.calc_top_charges_diff(x_in, x_out),
                dtype=TF_INT
            )
            self.charge_diffs_op = tf.reduce_sum(x_dq) / self.num_samples

        if self.save_lf:
            op_keys = ['masks_f', 'masks_b',
                       'lf_out_f', 'lf_out_b',
                       'pxs_out_f', 'pxs_out_b',
                       'logdets_f', 'logdets_b',
                       'fns_out_f', 'fns_out_b',
                       'sumlogdet_f', 'sumlogdet_b']
            for key in op_keys:
                try:
                    op = dynamics_output[key]
                    setattr(self, key, op)
                except KeyError:
                    continue

        with tf.name_scope('l2hmc_fns'):
            self.l2hmc_fns = {
                'l2hmc_fns_f': self._extract_l2hmc_fns(self.fns_out_f),
                'l2hmc_fns_b': self._extract_l2hmc_fns(self.fns_out_b),
            }

    def _build_run_ops(self):
        """Build run_ops dict containing grouped operations for inference."""
        run_ops = {
            'x_out': self.x_out,
            'px': self.px,
            'dynamics_eps': self.dynamics.eps,
            'actions_op': self.actions_op,
            'plaqs_op': self.plaqs_op,
            'avg_plaqs_op': self.avg_plaqs_op,
            'charges_op': self.charge_op,
            'charge_diffs_op': self.charge_diffs_op
        }

        if self.save_lf:
            keys = ['lf_out', 'pxs_out', 'masks',
                    'logdets', 'sumlogdet', 'fns_out']

            fkeys = [k + '_f' for k in keys]
            bkeys = [k + '_b' for k in keys]

            run_ops.update({k: getattr(self, k) for k in fkeys})
            run_ops.update({k: getattr(self, k) for k in bkeys})

        return run_ops

    def _build_train_ops(self):
        """Build train_ops dict containing grouped operations for training."""
        if self.hmc:
            train_ops = {}

        else:
            train_ops = {
                'train_op': self.train_op,
                'loss_op': self.loss_op,
                'x_out': self.x_out,
                'px': self.px,
                'dynamics_eps': self.dynamics.eps,
                'actions_op': self.actions_op,
                'plaqs_op': self.plaqs_op,
                'charges_op': self.charges_op,
                'charge_diffs_op': self.charge_diffs_op,
                'lr': self.lr
            }

        return train_ops

    '''
    def _build_run_ops(self):
        """Build `run_ops` used for running inference w/ trained model."""
        RunOps = namedtuple('RunOps', [
            'x_out', 'px', 'dynamics_eps', 'actions_op',
            'plaqs_op', 'avg_plaqs_op', 'charges_op', 'charge_diffs_op'
        ])

        run_ops = RunOps(self.x_out, self.px,
                         self.dynamics.eps, self.actions_op,
                         self.plaqs_op, self.avg_plaqs_op,
                         self.charges_op, self.charge_diffs_op)

        if self.save_lf:
            lfOps = namedtuple('lfOps', ['lf_out', 'pxs_out', 'masks',
                                         'logdets', 'sumlogdet', 'fns_out'])
            lf_ops_f = lfOps(self.lf_out_f, self.pxs_out_f, self.masks_f,
                             self.logdets_f, self.sumlogdet_f, self.fns_out_f)
            lf_ops_b = lfOps(self.lf_out_b,  self.pxs_out_b, self.masks_b,
                             self.logdets_b, self.sumlogdet_b, self.fns_out_b)

        tf.add_to_collection('run_ops', [*run_ops, *lf_ops_f, *lf_ops_b])

        return run_ops, lf_ops_f, lf_ops_b

    def _build_train_ops(self):
        """Build `train_ops` used for training our model."""
        if self.hmc:
            train_ops = ()
        else:
            TrainOps = namedtuple('TrainOps', [
                'train_op', 'loss_op', 'x_out', 'px',
                'dynamics_eps', 'actions_op', 'plaqs_op',
                'charges_op', 'charge_diffs_op', 'lr'
            ])

            train_ops = TrainOps(self.train_op, self.loss_op,
                                 self.x_out, self.px,
                                 self.dynamics.eps, self.actions_op,
                                 self.plaqs_op, self.charges_op,
                                 self.charge_diffs_op, self.lr)
        tf.add_to_collection('train_ops', train_ops)

        return train_ops
    '''

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
