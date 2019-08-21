"""
train_logger.py

Implements TrainLogger class responsible for saving/logging data from
GaugeModel.

Author: Sam Foreman (github: @saforem2)
Date: 04/09/2019
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import tensorflow as tf

import utils.file_io as io

from config import TRAIN_HEADER
from .summary_utils import create_summaries
from utils.file_io import save_params


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
            'samples': self.model.lattice.samples_tensor,
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
            summaries_output = create_summaries(
                model, self.train_summary_dir, training=True
            )
            self.summary_writer = summaries_output[0]
            self.summary_op = summaries_output[1]
            self.loss_averages_op = summaries_output[2]

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
        summary_str, loss_averages = sess.run([self.summary_op,
                                               self.loss_averages_op],
                                              feed_dict=feed_dict)

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
