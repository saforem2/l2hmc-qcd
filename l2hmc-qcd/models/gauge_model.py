"""
gauge_model.py

Implements `GaugeModel` class, inheriting from `BaseModel`.

Author: Sam Foreman (github: @saforem2)
Date: 09/04/2019
"""
# pylint: disable=invalid-name, no-member
from __future__ import absolute_import, division, print_function

import time

from collections import namedtuple

import numpy as np
import tensorflow as tf

import config as cfg
import utils.file_io as io

from base.base_model import BaseModel
from lattice.lattice import GaugeLattice
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
        super(GaugeModel, self).__init__(params, model_type='GaugeModel')
        self._model_type = 'GaugeModel'

        if params is None:
            params = GAUGE_PARAMS

        self.params = params
        self.dim = int(params.get('dim', 2))
        self.link_type = params.get('link_type', 'U1')
        self.time_size = int(params.get('time_size', 8))
        self.space_size = int(params.get('space_size', 8))
        self._charge_weight = float(params.get('charge_weight', 0.))
        self._plaq_weight = float(params.get('plaq_weight', 0.))
        io.log(SEP_STR)
        io.log(f'self._charge_weight: {self._charge_weight}')
        io.log(f'self._plaq_weight: {self._plaq_weight}')
        io.log(SEP_STR)
        self.build(params)

    def build(self, params=None):
        """Build TensorFlow graph."""
        params = self.params if params is None else params

        #  charge_weight = getattr(self, 'charge_weight_np', 0.)
        #  self._charge_weight = charge_weight
        self.use_charge_loss = (self._charge_weight > 0)

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
            self._lattice_shape = (self.batch_size,
                                   self.lattice.time_size,
                                   self.lattice.space_size, 2)

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
            'hmc': self.hmc,
            'use_bn': self.use_bn,
            'num_steps': self.num_steps,
            'batch_size': self.batch_size,
            'zero_masks': self.zero_masks,
            'model_type': self._model_type,
            'activation': self._activation,
            'num_hidden1': self.num_hidden1,
            'num_hidden2': self.num_hidden2,
            'x_dim': self.lattice.num_links,
            'eps': getattr(self, 'eps', None),
            'network_arch': self.network_arch,
            'dropout_prob': self.dropout_prob,
            'network_type': self._network_type,
            'eps_trainable': self.eps_trainable,
            '_input_shape': (self.batch_size, *self.lattice.links.shape),
        }
        if self.network_arch != 'generic':
            kwargs['num_filters'] = self.lattice.space_size

        #  kwargs = {
        #      'eps_trainable': not getattr(self, 'eps_fixed', False),
        #      'num_filters': self.lattice.space_size,
        #      'x_dim': self.lattice.num_links,
        #      'batch_size': self.batch_size,
        #      'zero_masks': self.zero_masks,
        #      '_input_shape': (self.batch_size, *self.lattice.links.shape),
        #      'model_type': self._model_type,
        #  }
        #
        #  dynamics = self._create_dynamics(potential_fn, **kwargs)
        #  io.log(f'Dynamics._model_type: {dynamics._model_type}\n')
        return self._create_dynamics(potential_fn, **kwargs)

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
        dq = - self.lattice.calc_top_charges_diff(x_init, x_proposed)
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
            charge_loss *= self._charge_weight

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

    def _plaq_sums(self, x):
        """Calculate the sum around all elementary plaquettes in `x`.
        Example:
                              = - roll(x[..., 0], -1, 2)
                          ┌───<───┐
                          │       │
             -x[..., 1] = ⋁       ⋀ = roll(x[..., 1], -1, 1)
                          │       │
                          └───>───┘
                              = x[..., 0]
        """
        x = tf.reshape(x, self._lattice_shape)  # (Nb, Lt, Lx, 2)
        plaq_sums = (x[..., 0]
                     - x[..., 1]
                     - tf.roll(x[..., 0], shift=-1, axis=2)   # along `x` axis
                     + tf.roll(x[..., 1], shift=-1, axis=1))  # along `t` axis

        return plaq_sums

    @staticmethod
    def _top_charge(plaq_sums):
        """Calculate the topological charge over all samples in x."""
        charges = tf.reduce_sum(tf.sin(plaq_sums), axis=(1, 2)) / (2 * np.pi)

        return charges

    @staticmethod
    def _plaq_loss(plaqs_init, plaqs_proposed, accept_prob, eps=1e-4):
        """Calculate the expected plaquette differences b/t `x1` and `x2`."""
        #  dx = (tf.cos(plaqs_proposed) - tf.cos(plaqs_init))
        #  dy = (tf.sin(plaqs_proposed) - tf.sin(plaqs_init))
        #  tot_diff = dx ** 2 + dy ** 2
        tot_diff = 2. * (1. - tf.cos(plaqs_proposed - plaqs_init))
        plaq_loss = accept_prob * tf.reduce_sum(tot_diff, axis=(1, 2)) + eps

        return plaq_loss

    def plaq_loss(self, xdata, zdata, eps=1e-4):
        """Calculate the loss due to the plaquette and charge differences."""
        xp0 = self._plaq_sums(xdata.init)
        xp1 = self._plaq_sums(xdata.proposed)
        zp0 = self._plaq_sums(zdata.init)
        zp1 = self._plaq_sums(zdata.proposed)

        plaq_loss = tf.cast(0., TF_FLOAT)
        if self._plaq_weight > 0:
            dxp = 2. * (1. - tf.cos(xp1 - xp0))
            dzp = 2. * (1. - tf.cos(zp1 - zp0))
            xp_loss = xdata.prob * tf.reduce_sum(dxp, axis=(1, 2)) + eps
            zp_loss = zdata.prob * tf.reduce_sum(dzp, axis=(1, 2)) + eps

            term1p = self._plaq_weight * (1. / xp_loss + 1. / zp_loss)
            term2p = (xp_loss + zp_loss) / self._plaq_weight
            plaq_loss = tf.reduce_mean(term1p - term2p, axis=0,
                                       name='plaq_loss')

        charge_loss = tf.cast(0., TF_FLOAT)
        if self._charge_weight > 0:
            xq0 = self._top_charge(xp0)
            xq1 = self._top_charge(xp1)
            zq0 = self._top_charge(zp0)
            zq1 = self._top_charge(zp1)

            xq_loss = xdata.prob * (xq1 - xq0) ** 2 + eps
            zq_loss = zdata.prob * (zq1 - zq0) ** 2 + eps
            term1q = self._charge_weight * (1. / xq_loss + 1. / zq_loss)
            term2q = (xq_loss + zq_loss) / self._charge_weight
            charge_loss = tf.reduce_mean(term1q - term2q, axis=0,
                                         name='charge_loss')

        return plaq_loss, charge_loss

    @staticmethod
    def _charge_loss(q_init, q_proposed, accept_prob, eps=1e-4):
        return accept_prob * (q_proposed - q_init) ** 2 + eps

    @staticmethod
    def _gauge_esjd(x1, x2, prob, eps=1e-4):
        """Calculate the esjd."""
        esjd = prob * tf.reduce_sum(2. * (1. - tf.cos(x1 - x2)), axis=1) + eps

        return esjd

    def _calc_esjd_loss(self, xdata, zdata, eps=1e-4):
        x_esjd = self._gauge_esjd(xdata.init, xdata.proposed, xdata.prob, eps)
        z_esjd = self._gauge_esjd(zdata.init, zdata.proposed, zdata.prob, eps)
        term1 = self.loss_scale * (1. / x_esjd + 1. / z_esjd)
        term2 = (x_esjd + z_esjd) / self.loss_scale
        esjd_loss = tf.reduce_mean(term1 - term2, axis=0)

        return esjd_loss

    @staticmethod
    def _mixed_loss(x, weight):
        """Return the mixed loss."""
        return tf.reduce_mean((weight / x) - (x / weight))

        # pylint:disable=too-many-locals
    def calc_loss(self, xdata, zdata, eps=1e-4):
        """Calculate the total loss."""
        total_loss = 0.
        ld = {}

        std_loss = self.std_weight * self._calc_esjd_loss(xdata, zdata, eps)
        plaq_loss, charge_loss = self.plaq_loss(xdata, zdata, eps)

        ld['std'] = std_loss
        ld['plaq'] = plaq_loss
        ld['charge'] = charge_loss

        total_loss = std_loss + charge_loss + plaq_loss
        tf.add_to_collection('losses', total_loss)

        losses_dict = {}
        fd = {k: v / total_loss for k, v in ld.items()}

        for (lk, lv), (fk, fv) in zip(ld.items(), fd.items()):
            losses_dict[f'{lk}_loss'] = lv
            losses_dict[f'{fk}_frac'] = fv
            tf.add_to_collection('losses', lv)
        return total_loss, losses_dict
