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

#  import utils.file_io as io
from utils.horovod_utils import warmup_lr
#  from utils.distributions import quadratic_gaussian
from config import HAS_HOROVOD, TF_FLOAT, GLOBAL_SEED

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


def _gaussian(x, mu, sigma):
    norm = tf.cast(
        1. / tf.sqrt(2 * np.pi * sigma ** 2), dtype=TF_FLOAT
    )
    exp_ = tf.exp(-tf.square(x - mu) / (2 * sigma))

    return norm * exp_


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

    def _create_dynamics(self, **kwargs):
        """Create dynamics object used to perform augmented leapfrog update."""
        raise NotImplementedError

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

            inputs = {
                'x': x,
                'beta': beta,
                'scale_weight': scale_weight,
                'transl_weight': transl_weight,
                'transf_weight': transf_weight,
                'train_phase': train_phase
            }

        _ = [tf.add_to_collection('inputs', i) for i in inputs.values()]

        return inputs

    def _create_metric_fn(self, metric):
        """Create metric function used to measure distatnce between configs."""
        raise NotImplementedError

    def _calc_esjd(self, x1, x2, prob):
        """Compute the expected squared jump distance (ESJD)."""
        return prob * tf.reduce_sum(self.metric_fn(x1, x2), axis=1)

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

    def _alt_gaussian_loss(self, x_data, z_data, mean, sigma):
        """Alternative Gaussian loss implemntation."""
        ls = getattr(self, 'loss_scale', 0.1)
        aux_weight = getattr(self, 'aux_weight', 1.)
        with tf.name_scope('gaussian_loss'):
            with tf.name_scope('x_loss'):
                x_esjd = self._calc_esjd(x_data.init,
                                         x_data.proposed,
                                         x_data.prob) + 1e-4
                #  xg1 = _gaussian(ls / x_esjd, mean, sigma)
                xg2 = _gaussian(x_esjd / ls, mean, sigma)
                #  x_diff = ls / x_esjd - x_esjd / ls
                #  x_gauss = _gaussian(x_diff, mean, sigma)
                x_loss = - tf.reduce_mean(xg2, name='x_loss')
                #  x_gauss = _gaussian(x_esjd, mean, sigma)
                #  x_loss_ = tf.reduce_mean(x_gauss)
                #
                #  x_gauss_inv = _gaussian(1. / x_esjd, mean, sigma)
                #  x_loss_inv_ = tf.reduce_mean(x_gauss_inv)
                #
                #  x_loss = - tf.log(ls * x_loss_inv_ - x_loss_ / ls)

            with tf.name_scope('z_loss'):
                z_esjd = self._calc_esjd(z_data.init,
                                         z_data.proposed,
                                         z_data.prob) + 1e-4
                #  zg1 = _gaussian(ls / z_esjd, mean, sigma)
                zg2 = _gaussian(z_esjd / ls, mean, sigma)
                #  z_diff = ls / z_esjd - z_esjd / ls
                #  z_gauss = _gaussian(z_diff, mean, sigma)
                z_loss = - tf.reduce_mean(zg2, name='z_loss')

            #  loss = - tf.log(x_loss + z_loss, name='gaussian_loss')
            loss = tf.add(x_loss, aux_weight * z_loss, name='loss')

        return loss

    def _gaussian_loss(self, x_data, z_data, mean, sigma):
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
        #  x_diff = np.sum((x_in - xb).T.dot(x_in - xb))
        #  v_diff = np.sum((v_in - vb).T.dot(v_in - vb))

        #  x_allclose = allclose(x_in, xb)  # xb = backward(forward(x_in))
        #  v_allclose = allclose(v_in, vb)

        return x_diff, v_diff

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
