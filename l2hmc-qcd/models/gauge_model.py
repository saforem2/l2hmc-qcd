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
from params.gauge_params import GAUGE_PARAMS
#  from tensorflow.python.ops import control_flow_ops as control_flow_ops

if HAS_HOROVOD:
    import horovod.tensorflow as hvd

LFdata = namedtuple('LFdata', ['init', 'proposed', 'prob'])


def allclose(x, y, rtol=1e-3, atol=1e-5):
    return tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol)


class GaugeModel(BaseModel):
    def __init__(self, params=None):
        super(GaugeModel, self).__init__(params)
        self._model_type = 'GaugeModel'

        if params is None:
            params = GAUGE_PARAMS

        #  self._model_type = 'GaugeModel'

        #  self.loss_weights = {}
        #  for key, val in params.items():
        #      if 'weight' in key and key != 'charge_weight':
        #          self.loss_weights[key] = val
        #      elif key == 'charge_weight':
        #          pass
        #      else:
        #          setattr(self, key, val)

        #  self.eps_trainable = not self.eps_fixed
        #  self.charge_weight_np = params['charge_weight']
        self.build()

    def build(self):
        """Build TensorFlow graph."""
        t0 = time.time()
        use_gaussian_loss = getattr(self, 'use_gaussian_loss', False)
        use_nnehmc_loss = getattr(self, 'use_nnehmc_loss', False)
        self.use_gaussian_loss = use_gaussian_loss
        self.use_nnehmc_loss = use_nnehmc_loss

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
            inputs = self._create_inputs()
            self.x = inputs['x']
            self.beta = inputs['beta']
            nw_keys = ['scale_weight', 'transl_weight', 'transf_weight']
            self.net_weights = [inputs[k] for k in nw_keys]
            self.train_phase = inputs['train_phase']
            self._inputs = inputs
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
        observables = self._create_observables()
        self.plaq_sums_op = observables['plaq_sums_op']
        self.actions_op = observables['actions_op']
        self.plaqs_op = observables['plaqs_op']
        self.avg_plaqs_op = observables['avg_plaqs_op']
        self.charges_op = observables['charges_op']
        self._observables = observables
        # *******************************************************

        # *******************************************************************
        # Run dynamics (i.e. augmented leapfrog) to generate new configs 
        # -------------------------------------------------------------------
        with tf.name_scope('l2hmc'):
            slf = self.save_lf
            hmc = True if self.use_nnehmc_loss else False
            args = (self.beta, self.net_weights, self.train_phase)

            with tf.name_scope('main_dynamics'):
                x_dynamics = self.dynamics.apply_transition(self.x, *args,
                                                            save_lf=slf,
                                                            hmc=hmc)
            if not hmc and getattr(self, 'aux_weight', 1.) > 0:
                with tf.name_scope('auxiliary_dynamics'):
                    self.z = tf.random_normal(tf.shape(self.x),
                                              dtype=TF_FLOAT,
                                              seed=GLOBAL_SEED,
                                              name='z')
                    z_dynamics = self.dynamics.apply_transition(self.z, *args,
                                                                save_lf=False,
                                                                hmc=hmc)

            self.x_out = x_dynamics['x_out']
            self.px = x_dynamics['accept_prob']
            self._parse_dynamics_output(x_dynamics)

            with tf.name_scope('check_reversibility'):
                self.x_diff, self.v_diff = self._check_reversibility()

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
                x_data = LFdata(x_dynamics['x_in'],
                                x_dynamics['x_proposed'],
                                x_dynamics['accept_prob'])

                if not hmc and getattr(self, 'aux_weight', 1.) > 0:
                    z_data = LFdata(z_dynamics['x_in'],
                                    z_dynamics['x_proposed'],
                                    z_dynamics['accept_prob'])

                else:
                    z_data = LFdata(0., 0., 0.)

                if self.use_gaussian_loss:
                    self.loss_op = self.gaussian_loss(x_data, z_data)

                elif self.use_nnehmc_loss:
                    x_hmc_prob = x_dynamics['accept_prob_hmc']
                    self.loss_op = self.nnehmc_loss(x_data, x_hmc_prob)

                else:
                    self.loss_op = self.calc_loss(x_data, z_data)

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
                                   batch_size=self.batch_size,
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
                'batch_size': self.batch_size,
                '_input_shape': (self.batch_size, *self.lattice.links.shape),
            })

            dynamics_params.update(params)
            samples = self.lattice.samples_tensor
            potential_fn = self.lattice.get_potential_fn(samples)
            dynamics = Dynamics(potential_fn=potential_fn, **dynamics_params)

        return dynamics

    def _create_observables(self):
        """Create operations for calculating lattice observables."""
        with tf.name_scope('observables'):
            plaq_sums = self.lattice.calc_plaq_sums(self.x)
            actions = self.lattice.calc_actions(plaq_sums=plaq_sums)
            plaqs = self.lattice.calc_plaqs(plaq_sums=plaq_sums)
            avg_plaqs = tf.reduce_mean(plaqs, name='avg_plaqs')
            charges = self.lattice.calc_top_charges(plaq_sums=plaq_sums)

        observables = {
            'plaq_sums_op': plaq_sums,
            'actions_op': actions,
            'plaqs_op': plaqs,
            'avg_plaqs_op': avg_plaqs,
            'charges_op': charges,
        }

        return observables

    def _create_metric_fn(self, metric):
        """Create `metric_fn` for measuring the `distance` between configs."""
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

    def _charge_loss(self, x_init, x_proposed, prob):
        dq = self.lattice.calc_top_charges_diff(x_init, x_proposed)
        charge_loss = prob * dq

        return charge_loss

    def _calc_charge_loss(self, x_data, z_data):
        """Calculate the total charge loss."""
        aux_weight = getattr(self, 'aux_weight', 1.)
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

    def calc_loss(self, x_data, z_data):
        """Calculate the total loss from all terms."""
        charge_weight = getattr(self, 'charge_weight_np', 1.)
        #  charge_weight = weights.get('charge_weight', 1.)

        loss = self._calc_loss(x_data, z_data)
        if charge_weight > 0:
            io.log('INFO: Including topological charge'
                   ' difference term in loss function.')
            loss += self._calc_charge_loss(x_data, z_data)  # , weights)

        return loss

    def gaussian_loss(self, x_data, z_data):
        loss = self._gaussian_loss(x_data, z_data, mean=0., sigma=1.)

        return loss

    def nnehmc_loss(self, x_data, hmc_prob, beta=1.):
        """Calculate the NNEHMC loss."""
        x_in, x_proposed, accept_prob = x_data
        x_esjd = self._calc_esjd(x_in, x_proposed, accept_prob)

        return tf.reduce_mean(- x_esjd - beta * hmc_prob)

    def _parse_dynamics_output(self, dynamics_output):
        """Parse output dictionary from `self.dynamics.apply_transition`."""
        with tf.name_scope('top_charge_diff'):
            x_in = dynamics_output['x_in']
            x_out = dynamics_output['x_out']
            x_dq = tf.cast(
                self.lattice.calc_top_charges_diff(x_in, x_out),
                dtype=TF_INT
            )
            self.charge_diffs_op = tf.reduce_sum(x_dq) / self.batch_size

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
            'charges_op': self.charges_op,
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
