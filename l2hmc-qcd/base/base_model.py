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

import utils.file_io as io
from utils.horovod_utils import warmup_lr
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
        ls = getattr(self, 'loss_scale', 0.1)

        def _diff(x1, x2):
            return tf.reduce_sum(self.metric_fn(x1, x2), axis=1)
        with tf.name_scope('calc_loss'):
            with tf.name_scope('x_loss'):
                x_loss = (x_data.prob * _diff(x_data.x_in,
                                              x_data.x_proposed)) + 1e-4
            with tf.name_scope('z_loss'):
                z_loss = (z_data.prob * _diff(z_data.x_in,
                                              z_data.x_proposed)) + 1e-4
                #  z_loss = self._loss(*z_data) if aux_weight > 0. else 0.

            loss = 0.
            loss += ls * (tf.reduce_mean(1. / x_loss)
                          + tf.reduce_mean(1. / z_loss))
            loss += (- tf.reduce_mean(x_loss) - tf.reduce_mean(z_loss)) / ls

        return loss

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
