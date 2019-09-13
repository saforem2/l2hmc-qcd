"""
base_model.py

Implements BaseModel class.

Author: Sam Foreman (github: @saforem2)
Date: 08/28/2019
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import tensorflow as tf
import numpy as np

import utils.file_io as io
from utils.horovod_utils import warmup_lr
from utils.distributions import quadratic_gaussian
from config import HAS_HOROVOD
from models.params import GAUGE_PARAMS

if HAS_HOROVOD:
    import horovod.tensorflow as hvd


PARAMS = {
    'hmc': False,
    'lr_init': 1e-3,
    'lr_decay_steps': 1000,
    'lr_decay_rate': 0.96,
    'train_steps': 5000,
    'using_hvd': False,
}


class BaseModel:

    def __init__(self, params=None):
        self.params = params

        self.loss_weights = {}
        for key, val in params.items():
            if 'weight' in key and key != 'charge_weight':
                self.loss_weights[key] = val
            elif key == 'charge_weight':
                pass
            else:
                setattr(self, key, val)

        self.eps_trainable = not self.eps_fixed
        self.charge_weight_np = getattr(params, 'charge_weight', None)
        self.global_step = self._create_global_step()

        warmup = getattr(self, 'warmup_lr', False)
        self.lr = self._create_lr(warmup)

        self.optimizer = self._create_optimizer()

    def build(self):
        """Build `tf.Graph` object containing operations for running model."""
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
                                                staircase=False,
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

    def _create_dynamics(self, **kwargs):
        """Create dynamics object used to perform augmented leapfrog update."""
        raise NotImplementedError

    def _create_inputs(self):
        """Create placeholders to hold configurations."""
        raise NotImplementedError

    def _create_metric_fn(self, metric):
        """Create metric function used to measure distatnce between configs."""
        raise NotImplementedError

    def _calc_esjd(self, x1, x2, prob):
        """Compute the expected squared jump distance (ESJD)."""
        return prob * tf.reduce_sum(self.metric_fn(x1, x2), axis=1) + 1e-4

    def _loss(self, init, proposed, prob):
        """Calculate the (standard) contribution to the loss from the ESJD."""
        ls = getattr(self, 'loss_scale', 0.1)
        with tf.name_scope('calc_esjd'):
            esjd = self._calc_esjd(init, proposed, prob)
        loss = ls * tf.reduce_mean(1. / esjd) - tf.reduce_mean(esjd) / ls

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
        def _gaussian(x, mu, sigma):
            norm = 1. / tf.sqrt(2 * np.pi * sigma**2)
            return (tf.exp(-tf.square(x - mu) / (2 * sigma)) / norm) + 1e-4

        ls = getattr(self, 'loss_scale', 0.1)
        aux_weight = getattr(self, 'aux_weight', 1.)
        with tf.name_scope('gaussian_loss'):
            with tf.name_scope('x_loss'):
                x_esjd = self._calc_esjd(x_data.init,
                                         x_data.proposed,
                                         x_data.prob)
                x_gauss = _gaussian(x_esjd, mean, sigma)
                x_loss = ls * tf.reduce_mean(x_gauss)
                #  x_loss = (ls * tf.reduce_mean(1. / x_gauss)
                #            - tf.reduce_mean(x_gauss) / ls)

            with tf.name_scope('z_loss'):
                if aux_weight > 0.:
                    z_esjd = self._calc_esjd(z_data.init,
                                             z_data.proposed,
                                             z_data.prob)
                    z_gauss = _gaussian(z_esjd, mean, sigma)
                    z_loss = ls * tf.reduce_mean(z_gauss)
                    #  z_loss = (ls * tf.reduce_mean(1. / z_gauss)
                    #            - tf.reduce_mean(z_gauss) / ls)
                else:
                    z_loss = 0.

            gaussian_loss = tf.add(x_loss, z_loss, name='gaussian_loss')

        return gaussian_loss

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
        with tf.control_dependencies([loss_op, *self.dynamics.updates]):
            train_op = self.optimizer.apply_gradients(grads_and_vars,
                                                      self.global_step,
                                                      'train_op')
        return train_op
