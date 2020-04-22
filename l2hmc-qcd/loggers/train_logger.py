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

from collections import namedtuple
from loggers.summary_utils import create_summaries
#  from .summary_utils import create_summaries
from utils.file_io import save_params

h_str = ("{:^12s}" + 9 * "{:^10s}").format(
    "STEP", "t/STEP", "LOSS", "% ACC", "EPS",
    "BETA", "ACTION", "PLAQ", "(EXACT)", "LR"
)

dash = (len(h_str) + 1) * '-'
TRAIN_HEADER = dash + '\n' + h_str + '\n' + dash


class TrainLogger(object):
    def __init__(self, model, log_dir, logging_steps=10, summaries=False):
        #  self.sess = sess
        self.model = model
        self._model_type = model._model_type
        self.summaries = summaries
        self.logging_steps = logging_steps
        self._print_steps = getattr(model, 'print_steps', 10)
        self._save_steps = getattr(model, 'save_steps', int(1e4))

        self.train_data = {}
        self.h_strf = ("{:^13s}" + 9 * "{:^12s}").format(
            "STEP", "t/STEP", "LOSS", "% ACC", "EPS", "𝞭x",
            "BETA", "LR", "exp(𝞭H)", "sumlogdet",
        )

        if model._model_type == 'GaugeModel':
            self.obs_data = {}
            self.h_strf += ("{:^12s}").format("𝞭Q (tot)")
            self.h_strf += ("{:^12s}").format("𝞭𝜙")

        self.dash = (len(self.h_strf) + 1) * '-'
        self.train_header = self.dash + '\n' + self.h_strf + '\n' + self.dash
        self.train_data_strings = [self.train_header]

        # log_dir will be None if using_hvd and hvd.rank() != 0
        # this prevents workers on different ranks from corrupting checkpoints
        #  if log_dir is not None and self.is_chief:
        dirs, files = self._create_dir_structure(log_dir)
        self.log_dir = dirs['log_dir']
        self.checkpoint_dir = dirs['checkpoint_dir']
        self.train_dir = dirs['train_dir']
        self.train_summary_dir = dirs['train_summary_dir']
        self.train_log_file = files['train_log_file']
        self.current_state_file = files['current_state_file']

        save_params(self.model.params, self.log_dir)

        if self.summaries:
            self.writer = tf.summary.FileWriter(self.train_summary_dir,
                                                tf.get_default_graph())
            self.summary_writer, self.summary_op = create_summaries(
                model, self.train_summary_dir, training=True
            )

    @staticmethod
    def _create_dir_structure(log_dir):
        """Create relevant directories for storing data.

        Args:
            log_dir: Root directory in which all other directories are created.
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

        for _, val in dirs.items():
            io.check_else_make_dir(val)

        return dirs, files

    def log_step(self, sess, data, net_weights):
        """Update self.logger.summaries."""
        feed_dict = {
            self.model.x: data['x_in'],
            self.model.beta: data['beta'],
            self.model.net_weights: net_weights,
            self.model.train_phase: True
        }
        summary_str = sess.run(self.summary_op, feed_dict=feed_dict)

        self.writer.add_summary(summary_str, global_step=data['step'])
        self.writer.flush()

    def _update(self, data, data_str):
        for key, val in data.items():
            try:
                self.train_data[key].append(val)
            except KeyError:
                self.train_data[key] = [val]

        self.train_data_strings.append(data_str)

    def update(self, sess, data, data_str, net_weights):
        """Update _current state and train_data."""
        step = data['step']
        self._update(data, data_str)

        if (step + 1) % self._print_steps == 0:
            io.log(data_str)

        if (step + 1) % 100 == 0:
            io.log(self.train_header)
            self.save_current_state(data)

        if (step + 1) % self._save_steps == 0:
            self.save_current_state(data)
            self.save_train_data()

        if self.summaries and (step + 1) % self.logging_steps == 0:
            self.log_step(sess, data, net_weights)

    def write_train_strings(self):
        """Write training strings out to file."""

        tlf = self.train_log_file
        _ = [io.write(s, tlf, 'a') for s in self.train_data_strings]

    def save_current_state(self, data):
        """Save curent state incase training needs to be restarted."""
        out_file = os.path.join(self.train_dir, 'current_state.pkl')
        io.save_pkl(data, out_file, name='current_state')

    def save_train_data(self):
        """Save train data to `.pkl` file."""
        out_file = os.path.join(self.train_dir, 'train_data.pkl')
        io.log(f'Saving train data to: {out_file}.')
        io.save_pkl(self.train_data, out_file, name='train_data')
