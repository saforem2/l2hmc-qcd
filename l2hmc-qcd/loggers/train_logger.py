"""
train_logger.py

Implements TrainLogger class responsible for saving/logging data from
GaugeModel.

Author: Sam Foreman (github: @saforem2)
Date: 04/09/2019
"""
import os
import pickle

import tensorflow as tf

import utils.file_io as io

from config import TRAIN_HEADER
from utils.tf_logging import variable_summaries, grad_norm_summary
from utils.file_io import save_params
from lattice.lattice import u1_plaq_exact_tf


class TrainLogger:
    def __init__(self, model, log_dir, summaries=False):
        #  self.sess = sess
        self.model = model
        self.summaries = summaries

        self.charges_dict = {}
        self.charge_diffs_dict = {}

        self._current_state = {
            'step': 0,
            'beta': self.model.beta_init,
            'eps': self.model.eps,
            'lr': self.model.lr_init,
            'samples': self.model.samples,
        }

        self.train_data_strings = [TRAIN_HEADER]
        self.train_data = {
            'loss': {},
            'actions': {},
            'plaqs': {},
            'charges': {},
            'charge_diffs': {},
            'px': {}
        }

        if self.model.save_lf:
            self.train_data['l2hmc_fns'] = {
                'forward': [],
                'backward': [],
            }

        # log_dir will be None if using_hvd and hvd.rank() != 0
        # this prevents workers on different ranks from corrupting checkpoints
        #  if log_dir is not None and self.is_chief:
        self._create_dir_structure(log_dir)
        save_params(self.model.params, self.log_dir)

        if self.summaries:
            #  with tf.variable_scope('train_summaries'):
            self.writer = tf.summary.FileWriter(self.train_summary_dir,
                                                tf.get_default_graph())
            self.create_summaries()

    def _create_dir_structure(self, log_dir):
        """Create relevant directories for storing data.

        Args:
            log_dir: Root directory in which all other directories are created.

        Returns:
            None
        """
        dirs = {
            'log_dir': log_dir,
            'checkpoint_dir': os.path.join(log_dir, 'checkpoints'),
            'train_dir': os.path.join(log_dir, 'training'),
            'train_summary_dir': os.path.join(log_dir, 'summaries', 'train'),
        }
        files = {
            'train_log_file': os.path.join(dirs['train_dir'],
                                           'training_log.txt'),
            'current_state_file': os.path.join(dirs['train_dir'],
                                               'current_state.pkl')
        }

        for key, val in dirs.items():
            io.check_else_make_dir(val)
            setattr(self, key, val)

        for key, val in files.items():
            setattr(self, key, val)

    def create_summaries(self):
        """Create summary objects for logging in TensorBoard."""
        #  ld = self.log_dir
        tsd = self.train_summary_dir
        self.summary_writer = tf.contrib.summary.create_file_writer(tsd)

        grads_and_vars = zip(self.model.grads,
                             self.model.dynamics.trainable_variables)

        with tf.name_scope('loss'):
            tf.summary.scalar('loss', self.model.loss_op)

        with tf.name_scope('learning_rate'):
            tf.summary.scalar('learning_rate', self.model.lr)

        with tf.name_scope('step_size'):
            tf.summary.scalar('step_size', self.model.dynamics.eps)

        with tf.name_scope('avg_charge_diffs'):
            tf.summary.scalar('avg_charge_diffs',
                              tf.reduce_mean(self.model.charge_diffs_op))

        with tf.name_scope('avg_plaq'):
            tf.summary.scalar('avg_plaq', self.model.avg_plaqs_op)

        with tf.name_scope('avg_plaq_diff'):
            tf.summary.scalar('avg_plaq_diff',
                              (u1_plaq_exact_tf(self.model.beta)
                               - self.model.avg_plaqs_op))

        for k1, v1 in self.model.l2hmc_fns['out_fns_f'].items():
            for k2, v2 in v1.items():
                if 'x1' in v2.name or 'v1' in v2.name:
                    with tf.name_scope(f'{k1}_fn_{k2}_f'):
                        variable_summaries(v2)
                    #  tf.summary.scalar(f'{k2}_avg', tf.reduce_mean(v2))
                    #  tf.summary.histogram(f'{k2}', v2)

        #  for k1, v1 in self.model.l2hmc_fns['out_fns_b'].items():
        #      for k2, v2 in v1.items():
        #          with tf.name_scope(f'{k1}_fn_{k2}_b'):
        #              variable_summaries(v2)
                    #  tf.summary.scalar(f'{k2}_avg', tf.reduce_mean(v2))
                    #  tf.summary.histogram(f'{k2}', v2)

        with tf.name_scope('lf_out_f'):
            variable_summaries(self.model.lf_out_f)

        with tf.name_scope('lf_out_b'):
            variable_summaries(self.model.lf_out_b)

        for grad, var in grads_and_vars:
            try:
                _name = var.name.split('/')[-2:]
                if len(_name) > 1:
                    name = _name[0] + '/' + _name[1][:-2]
                else:
                    name = var.name[:-2]
            except (AttributeError, IndexError):
                name = var.name[:-2]
            #  with tf.name_scope(name + '/training'):
            #  with tf.name_scope(name + '/training/gradients'):
            with tf.name_scope(name):
                var_name = var.name.replace(':', '')
                tf.summary.scalar(var_name + '/mean', tf.reduce_mean(var))
                tf.summary.histogram(var_name, var)
                #  variable_summaries(var, var.name)
            grad_name = var.name.replace(':', '') + '/gradient'
            with tf.name_scope(grad_name):
                tf.summary.scalar(grad_name + '/mean', tf.reduce_mean(grad))
                tf.summary.histogram(grad_name, grad)
                #  variable_summaries(grad, var.name + '/gradients')
            if 'kernel' in var.name:
                if 'XNet' in var.name:
                    net_str = 'XNet/'
                elif 'VNet' in var.name:
                    net_str = 'VNet/'
                else:
                    net_str = ''

                with tf.name_scope(net_str):
                    if 'scale' in var.name:
                        grad_norm_summary(net_str + 'scale', grad)
                        tf.summary.histogram(net_str + 'scale', grad)
                    if 'transf' in var.name:
                        grad_norm_summary(net_str + 'transformation', grad)
                        tf.summary.histogram(net_str + 'transformation', grad)
                    if 'transl' in var.name:
                        grad_norm_summary(net_str + 'translation', grad)
                        tf.summary.histogram(net_str + 'translation', grad)

        self.summary_op = tf.summary.merge_all(name='train_summary_op')

    def save_current_state(self):
        """Save current state to pickle file.

        The current state contains the following, which makes life easier if
        we're trying to restore training from a saved checkpoint:
            * most recent samples
            * learning_rate
            * beta
            * dynamics.eps
            * training_step
        """
        with open(self.current_state_file, 'wb') as f:
            pickle.dump(self._current_state, f)

    def log_step(self, sess, step, samples_np, beta_np, net_weights):
        """Update self.logger.summaries."""
        feed_dict = {
            self.model.x: samples_np,
            self.model.beta: beta_np,
            self.model.net_weights[0]: net_weights[0],
            self.model.net_weights[1]: net_weights[1],
            self.model.net_weights[2]: net_weights[2],
            self.model.train_phase: True
        }
        summary_str = sess.run(self.summary_op, feed_dict=feed_dict)

        self.writer.add_summary(summary_str, global_step=step)
        self.writer.flush()

    def update_training(self, sess, data, net_weights, data_str):
        """Update _current state and train_data."""
        step = data['step']
        beta = data['beta']
        self._current_state['step'] = step
        self._current_state['beta'] = beta
        self._current_state['lr'] = data['lr']
        self._current_state['eps'] = data['eps']
        self._current_state['samples'] = data['samples']
        self._current_state['net_weights'] = net_weights

        key = (step, beta)

        self.charges_dict[key] = data['charges']
        self.charge_diffs_dict[key] = data['charge_diffs']

        obs_keys = ['loss', 'actions',
                    'plaqs', 'charges',
                    'charge_diffs', 'px']
        for obs_key in obs_keys:
            self.train_data[obs_key][key] = data[obs_key]

        #  self.train_data['loss'][key] = data['loss']
        #  self.train_data['actions'][key] = data['actions']
        #  self.train_data['plaqs'][key] = data['plaqs']
        #  self.train_data['charges'][key] = data['charges']
        #  self.train_data['charge_diffs'][key] = data['charge_diffs']
        #  self.train_data['px'][key] = data['px']

        self.train_data_strings.append(data_str)
        if step % self.model.print_steps == 0:
            io.log(data_str)

        if self.summaries and (step + 1) % self.model.logging_steps == 0:
            self.log_step(sess, step, data['samples'], beta, net_weights)

        if (step + 1) % self.model.save_steps == 0:
            self.save_current_state()

        if step % 100 == 0:
            io.log(TRAIN_HEADER)

    def write_train_strings(self):
        """Write training strings out to file."""
        tlf = self.train_log_file
        _ = [io.write(s, tlf, 'a') for s in self.train_data_strings]
