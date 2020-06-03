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

from base.base_model import BaseModel, add_to_collection
from lattice.lattice import GaugeLattice
from dynamics.dynamics import Dynamics, DynamicsConfig, convert_to_angle
from network import NetworkConfig
from params import GAUGE_PARAMS

if cfg.HAS_HOROVOD:
    import horovod.tensorflow as hvd  # noqa: 401

TF_FLOAT = cfg.TF_FLOAT
NP_FLOAT = cfg.NP_FLOAT

PI = np.pi
TWO_PI = 2 * PI


LFdata = namedtuple('LFdata', ['init', 'proposed', 'prob'])
SEP_STR = 80 * '-'
SEP_STRN = 80 * '-' + '\n'


def allclose(x, y, rtol=1e-3, atol=1e-5):
    return tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol)


def split_sampler_data(sampler_data):
    return sampler_data.data, sampler_data.dynamics_output


def project_angle_fft_np(x, n=10):
    """Numpy version of `project_angle_fft`."""
    y = np.zeros(x.shape, dtype=x.dtype)
    for _ in range(1, n):
        y += (-2. / n) * ((-1) ** n) * np.sin(n * x)

    return y

def project_angle_fft(x, n=10):
    """Use the Fourier series representation of the linear function `x` to
    approximate the discontinuous projection of the angle to [0, 2pi].

    This gives a continuous function when calculating derivatives of the loss
    function.

    Args:
        x (array-like): array to be projected.
        n (int): Number of temrs to keep in Fourier series.

    Returns:
        angle (array-like): Projected angle (same shape as `x`).
    """
    if isinstance(x, np.ndarray):
        return project_angle_fft_np(x, n)
    y = tf.zeros_like(x)
    for _ in range(1, n):
        y += (-2. / n) * ((-1) ** n) * tf.math.sin(n * x)

    return y


def calc_plaqs(x, n=1):
    return (x[..., 0]
            - x[..., 1]
            - tf.roll(x[..., 0], shift=-n, axis=2)
            + tf.roll(x[..., 1], shift=-n, axis=1))


def project_angle(x):
    """Returns the projection of an angle `x` from [-4pi, 4pi] to [-pi, pi]."""
    return x - TWO_PI * tf.math.floor((x + PI) / TWO_PI)


class GaugeLoss:
    """Compute the loss value for the 2D U(1) lattice gauge model. """
    def __init__(self, plaq_weight, charge_weight, lattice_shape, n_arr=None):
        if n_arr is None:
            n_arr = [1]
        self.n_arr = n_arr
        self.lattice_shape = lattice_shape
        self.plaq_weight = plaq_weight
        self.charge_weight = charge_weight

    def __call__(self, x_init, x_prop, accept_prob):
        x_init = tf.reshape(x_init, self.lattice_shape)
        x_prop = tf.reshape(x_prop, self.lattice_shape)
        p_init = 0.
        p_prop = 0.
        for n in self.n_arr:
            x_init = convert_to_angle(x_init)
            x_prop = convert_to_angle(x_prop)
            p_init += calc_plaqs(x_init, n)
            p_prop += calc_plaqs(x_prop, n)

        ploss = 0.
        if self.plaq_weight > 0:
            dplaq = 1. - tf.math.cos(p_prop - p_init)
            ploss = accept_prob * tf.reduce_sum(dplaq, axis=(1, 2))
            ploss = tf.reduce_mean(-ploss / self.plaq_weight, axis=0)

        qloss = 0.
        if self.charge_weight > 0:
            #  q1 = tf.reduce_sum(tf.math.sin(p_init), axis=(1, 2)) / TWO_PI
            #  q2 = tf.reduce_sum(tf.math.sin(p_prop), axis=(1, 2)) / TWO_PI
            q1 = tf.reduce_sum(project_angle(p_init), axis=(1, 2)) / TWO_PI
            q2 = tf.reduce_sum(project_angle(p_prop), axis=(1, 2)) / TWO_PI
            qloss = accept_prob * (q2 - q1) ** 2
            qloss = tf.reduce_mean(-qloss / self.charge_weight, axis=0)

        return ploss, qloss, q1


# pylint: disable=too-many-instance-attributes
class GaugeModel(BaseModel):
    """Implements `GaugeModel` class, containing tf ops for training."""
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
        self._network_type = str(params.get('network_type', None))
        self.build(params)

    # pylint: disable=attribute-defined-outside-init
    def _build(self):
        """Helper method for building the model."""
        with tf.name_scope('dynamics'):
            dynamics, config, net_config = self.create_dynamics()
            self.dynamics = dynamics
            self.config = config
            self.net_config = net_config
            self.dynamics_eps = self.dynamics.eps  # TODO: Delete this?
            self.eps_setter = self._build_eps_setter()

        with tf.name_scope('metric_function'):
            self.metric_fn = lambda x, y: 2. * (1. - tf.math.cos(y - x))

        with tf.name_scope('sampler'):
            xdata, zdata = self._build_sampler()

        #  largest_wilson_loop = self.params.get('largest_wilson_loop', 1)
        #  n_arr = np.arange(largest_wilson_loop) + 1
        #  self.gauge_loss = GaugeLoss(self._plaq_weight, self._charge_weight,
        #                              self._lattice_shape, n_arr=n_arr)
        with tf.name_scope('calc_loss'):
            self.loss_op, self.losses_dict = self.calc_loss(xdata, zdata)

        with tf.name_scope('calc_and_apply_grads'):
            self.grads = self._calc_grads(self.loss_op)
            train_op, grads_and_vars = self._apply_grads(self.loss_op,
                                                         self.grads)
            self.train_op = train_op
            self.grads_and_vars = grads_and_vars

            train_ops = self._build_train_ops()
            run_ops = self._build_run_ops()

        return train_ops, run_ops

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
            self.train_ops, self.run_ops = self._build()

            self.q_out = self._top_charge(
                self._plaq_sums(self.run_ops['x_out'])
            )
            self.q_init = self.losses_dict.pop('q_init')
            self.q_prop = self.losses_dict.pop('q_prop')
            self.dq_prop = tf.math.abs(self.q_prop - self.q_init)
            self.dq_out = tf.math.abs(self.q_out - self.q_init)

            extra_ops = {
                'plaqs': self.plaqs,
                'charges': self.charges,
                'dq_prop': self.dq_prop,
                'dq_out': self.dq_out,
            }

            self.run_ops.update(extra_ops)
            self.train_ops.update(extra_ops)

            for val in extra_ops.values():
                tf.add_to_collection('train_ops', val)
                tf.add_to_collection('run_ops', val)

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
        dynamics_config = DynamicsConfig(
            num_steps=self.num_steps,
            eps=self.eps,
            input_shape=(self.batch_size, self.lattice.num_links),
            hmc=self.hmc,
            eps_trainable=self.eps_trainable,
            net_weights=self.net_weights,
            model_type='GaugeModel',
        )
        net_config = NetworkConfig(
            type='GaugeNetwork',
            units=self.units,
            dropout_prob=self.dropout_prob,
            activation_fn=tf.nn.relu,
        )

        dynamics = Dynamics(self.calc_action, dynamics_config, net_config)

        return dynamics, dynamics_config, net_config

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
            #  'dq': charges_out - charges
        }
        for obs in observables.values():
            tf.add_to_collection('observables', obs)

        return observables

    def _calc_charge_loss(self, x_data, z_data):
        """Calculate the total charge loss."""
        #  aux_weight = getattr(self, 'aux_weight', 1.)
        ls = self.loss_scale
        with tf.name_scope('calc_charge_loss'):
            with tf.name_scope('xq_loss'):
                xq_loss = self._charge_loss(*x_data)

            with tf.name_scope('zq_loss'):
                if self.aux_weight > 0.:
                    zq_loss = self._charge_loss(*z_data)
                else:
                    zq_loss = 0.

            charge_loss = 0.
            charge_loss += tf.reduce_mean((xq_loss + zq_loss) / ls,
                                          axis=0, name='charge_loss')
            charge_loss *= self._charge_weight

        return charge_loss

    def _plaq_sums(self, x, n=1):
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
        with tf.name_scope('calc_plaquettes'):
            x = tf.reshape(x, self._lattice_shape)  # (Nb, Lt, Lx, 2)
            plaqs = (x[..., 0]
                     - x[..., 1]
                     - tf.roll(x[..., 0], shift=-n, axis=2)   # along `x` axis
                     + tf.roll(x[..., 1], shift=-n, axis=1))  # along `t` axis
        return plaqs

    def calc_action(self, x):
        """Calculate the Wilson action of a batch of lattice configurations."""
        with tf.name_scope('calc_action'):
            plaqs = self._plaq_sums(x)
            action = tf.reduce_sum(1. - tf.cos(plaqs),
                                   axis=(1, 2), name='action')

        return action

    @staticmethod
    def _top_charge(plaqs, use_sin=True):
        """Numpy version of `_top_charge`."""
        with tf.name_scope('calc_charge'):
            charge = tf.reduce_sum((
                tf.math.sin(plaqs) if use_sin else project_angle(plaqs)
            ), axis=(1, 2)) / TWO_PI

        return charge

    def _charge_loss(self, plaqs_init, plaqs_prop, prob, use_sin=True):
        """Calculate the contribution to the loss from the charge diffs."""
        with tf.name_scope('charge_loss'):
            q_init = self._top_charge(plaqs_init, use_sin=use_sin)
            q_prop = self._top_charge(plaqs_prop, use_sin=use_sin)
            dq = prob * (q_prop - q_init) ** 2
            qloss = tf.reduce_mean(-(dq / self._charge_weight),
                                   axis=0, name='charge_loss')

        #  qloss = tf.reduce_sum(qloss, axis=0) / self.batch_size
        #  qloss = tf.reduce_mean(qloss, axis=0)

        return qloss, q_init, q_prop

    def _plaq_loss(self, plaqs_init, plaqs_prop, prob):
        """Calculate the expected plaquette differences b/t `x1` and `x2`."""
        with tf.name_scope('plaq_loss'):
            plaqs_diff = 2. * (1. - tf.cos(plaqs_prop - plaqs_init))
            dplaq = prob * tf.reduce_sum(plaqs_diff, axis=(1, 2))
            ploss = - dplaq / self._plaq_weight
            ploss = tf.reduce_mean(ploss, axis=0, name='plaq_loss')

        #  return tf.reduce_sum(ploss, axis=0) / self.batch_size
        return ploss

    def gauge_loss(self, xdata, zdata, use_sin=True):
        """Calculate the loss due to the plaquette and charge differences."""
        xp0 = self._plaq_sums(xdata.init)
        xp1 = self._plaq_sums(xdata.proposed)

        if self.aux_weight > 0:
            zp0 = self._plaq_sums(zdata.init)
            zp1 = self._plaq_sums(zdata.proposed)

        plaq_loss = tf.cast(0., TF_FLOAT)
        if self._plaq_weight > 0.:
            plaq_loss += self._plaq_loss(xp0, xp1, xdata.prob)
            if self.aux_weight > 0:
                plaq_loss += self._plaq_loss(zp0, zp1, zdata.prob)

        charge_loss = tf.cast(0., TF_FLOAT)
        if self._charge_weight > 0:
            qxl, q_init, q_prop = self._charge_loss(xp0, xp1, xdata.prob,
                                                    use_sin=use_sin)
            charge_loss += qxl
            if self.aux_weight > 0:
                qzl, _, _ = self._charge_loss(zp0, zp1, zdata.prob,
                                              use_sin=use_sin)
                charge_loss += qzl

        return plaq_loss, charge_loss, q_init, q_prop

    @staticmethod
    def _gauge_esjd(x1, x2, prob, eps=1e-4):
        """Calculate the esjd."""
        return prob * tf.reduce_sum(2. * (1. - tf.cos(x1 - x2)), axis=1) + eps

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
        losses_dict = {}
        with tf.name_scope('std_loss'):
            sloss = 0.
            if self.std_weight > 0:
                sloss = (
                    self.std_weight * self._calc_esjd_loss(xdata, zdata, eps)
                )
        with tf.name_scope('gauge_loss'):
            if self._charge_weight > 0 or self._plaq_weight > 0:
                inputs = (xdata, zdata, eps)
                ploss, qloss, q_init, q_prop = self.gauge_loss(*inputs)
                losses_dict.update({
                    'q_init': q_init,
                    'q_prop': q_prop,
                })

        total_loss = sloss + qloss + ploss

        ld = {
            'std': sloss,
            'plaq': ploss,
            'charge': qloss,
            'total': total_loss,
        }

        losses_dict.update({
            f'{k}_loss': v for k, v in ld.items()
        })

        losses_dict.update({
            f'{k}_frac': v / total_loss for k, v in ld.items()
        })

        add_to_collection('losses', list(losses_dict.values()))

        return total_loss, losses_dict
