"""
gauge_model.py

Implements `GaugeModel` class, inheriting from `BaseModel`.

Author: Sam Foreman (github: @saforem2)
Date: 09/04/2019
"""
from __future__ import absolute_import, division, print_function

import time

from collections import namedtuple
from lattice.lattice import GaugeLattice

import tensorflow as tf

import utils.file_io as io

from base.base_model import BaseModel

import config as cfg

from params.gauge_params import GAUGE_PARAMS

if cfg.HAS_HOROVOD:
    import horovod.tensorflow as hvd  # noqa: 401

TF_FLOAT = cfg.TF_FLOAT
NP_FLOAT = cfg.NP_FLOAT

LFdata = namedtuple('LFdata', ['init', 'proposed', 'prob'])
SEP_STR = 80 * '-'
SEP_STRN = 80 * '-' + '\n'


def allclose(x, y, rtol=1e-3, atol=1e-5):
    return tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol)


def split_sampler_data(sampler_data):
    return sampler_data.data, sampler_data.dynamics_output


class GaugeModel(BaseModel):
    def __init__(self, params=None):
        super(GaugeModel, self).__init__(params)
        self._model_type = 'GaugeModel'

        if params is None:
            params = GAUGE_PARAMS

        self.build(params)

    def build(self, params=None):
        """Build TensorFlow graph."""
        params = self.params if params is None else params

        charge_weight = getattr(self, 'charge_weight_np', 0.)
        self.use_charge_loss = True if charge_weight > 0. else False

        t0 = time.time()
        io.log(SEP_STRN + f'INFO: Building graph for `GaugeModel`...')
        with tf.name_scope(self._model_type):
            # ***********************************************
            # Create `Lattice` object
            # -----------------------------------------------
            io.log(f'INFO: Creating lattice...')
            self.lattice = self._create_lattice()
            self.batch_size = self.lattice.samples.shape[0]
            self.x_dim = self.lattice.num_links

            # ************************************************
            # Build inputs and their respective placeholders
            # ------------------------------------------------
            self._build_inputs()

            # ********************************************************
            # Create operations for calculating lattice observables
            # --------------------------------------------------------
            io.log(f'INFO: Creating operations for calculating observables...')
            observables = self._create_observables()
            self.plaq_sums = observables['plaq_sums']
            self.actions = observables['actions']
            self.plaqs = observables['plaqs']
            self.charges = observables['charges']
            self.avg_plaqs = observables['avg_plaqs']
            self.avg_actions = observables['avg_actions']
            self._observables = observables

            # ***************************************************************
            # Build operations common to all models (defined in `BaseModel`)
            # ---------------------------------------------------------------
            self._build()

            extra_train_ops = {
                'actions': self.actions,
                'plaqs': self.plaqs,
                'charges': self.charges,
            }
            self.train_ops.update(extra_train_ops)
            for val in extra_train_ops.values():
                tf.add_to_collection('train_ops', val)

            io.log(f'INFO: Done building graph. '
                   f'Took: {time.time() - t0}s\n' + SEP_STRN)

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

    def create_dynamics(self):
        """Create dynamics object."""
        samples = self.lattice.samples_tensor
        potential_fn = self.lattice.get_potential_fn(samples)

        kwargs = {
            'eps_trainable': not self.eps_fixed,
            'num_filters': self.lattice.space_size,
            'x_dim': self.lattice.num_links,
            'batch_size': self.batch_size,
            '_input_shape': (self.batch_size, *self.lattice.links.shape),
        }

        dynamics = self._create_dynamics(potential_fn, **kwargs)

        return dynamics

    def _create_observables(self):
        """Create operations for calculating lattice observables."""
        with tf.name_scope('observables'):
            plaq_sums = self.lattice.calc_plaq_sums(samples=self.x)
            actions = self.lattice.calc_actions(plaq_sums=plaq_sums)
            plaqs = self.lattice.calc_plaqs(plaq_sums=plaq_sums)
            charges = self.lattice.calc_top_charges(plaq_sums=plaq_sums)
            avg_plaqs = tf.reduce_mean(plaqs, name='avg_plaqs')
            avg_actions = tf.reduce_mean(actions, name='avg_actions')

        observables = {
            'plaq_sums': plaq_sums,
            'actions': actions,
            'plaqs': plaqs,
            'charges': charges,
            'avg_plaqs': avg_plaqs,
            'avg_actions': avg_actions,
        }
        for obs in observables.values():
            tf.add_to_collection('observables', obs)

        return observables

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

    def _calc_charge_diff(self, x_init, x_proposed):
        """Calculate difference in topological charge b/t x_init, x_proposed.

        Args:
            x_init: Configurations at the beginning of the trajectory.
            x_proposed: Configurations at the end of the trajectory (before MH
                accept/reject).

        Returns:
            x_dq: TensorFlow operation for calculating the difference in
                topological charge.
        """
        with tf.name_scope('top_charge_diff'):
            x_dq = tf.cast(
                self.lattice.calc_top_charges_diff(x_init, x_proposed),
                dtype=cfg.TF_INT
            )
            charge_diffs_op = tf.reduce_sum(x_dq) / self.batch_size

        return charge_diffs_op
