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

import tensorflow as tf

import utils.file_io as io

from collections import namedtuple
from .summary_utils import create_summaries
from utils.file_io import save_params

#  import pickle
#  from models.gauge_model import GaugeModel
#  from models.gmm_model import GaussianMixtureModel


h_str = ("{:^12s}" + 9 * "{:^10s}").format(
    "STEP", "t/STEP", "LOSS", "% ACC", "EPS",
    "BETA", "ACTION", "PLAQ", "(EXACT)", "LR"
)

dash = (len(h_str) + 1) * '-'
TRAIN_HEADER = dash + '\n' + h_str + '\n' + dash


TrainData = namedtuple('TrainData', ['loss', 'px', 'eps'])

ObsData = namedtuple('ObsData', [
    'actions', 'plaqs', 'charges',  # 'charge_diffs'
])

l2hmcFn = namedtuple('l2hmcFn', ['v1', 'x1', 'x2', 'v2'])
l2hmcFns = namedtuple('l2hmcFns', ['scale', 'translation', 'transformation'])


class TrainLogger:
    def __init__(self, model, log_dir, logging_steps=10, summaries=False):
        #  self.sess = sess
        self.model = model
        self.summaries = summaries
        self.logging_steps = logging_steps

        self.train_data = {}
        if model._model_type == 'GaugeModel':
            self._model_type = model._model_type
            self.obs_data = {}

            h_keys = ["t/STEP", "LOSS", "% ACC", "EPS",
                      "BETA", "LR" "ACTION", "PLAQ", "(EXACT)"]
            num_keys = len(h_keys)

            self.h_strf = ("{:^12s}".format("STEP")
                           + num_keys * "{:^10s}".format(*h_keys))

            #  self.h_strf = ("{:^12s}" + 9 * "{:^10s}").format(
            #      "STEP", "t/STEP", "LOSS", "% ACC", "EPS", "BETA", "LR"
            #      "ACTION", "PLAQ", "(EXACT)"
            #  )

        if model._model_type == 'GaussianMixtureModel':
            self._model_type = 'GaussianMixtureModel'
            self.h_strf = ("{:^12s}" + 6 * "{:^10s}").format(
                "STEP", "t/STEP", "LOSS", "% ACC", "EPS", "BETA", "LR"
            )

        self.dash = (len(self.h_strf) + 1) * '-'
        self.train_header = self.dash + '\n' + self.h_strf + '\n' + self.dash
        self.train_data_strings = [self.train_header]

        # log_dir will be None if using_hvd and hvd.rank() != 0
        # this prevents workers on different ranks from corrupting checkpoints
        #  if log_dir is not None and self.is_chief:
        self._create_dir_structure(log_dir)
        save_params(self.model.params, self.log_dir)

        if self.summaries:
            self.writer = tf.summary.FileWriter(self.train_summary_dir,
                                                tf.get_default_graph())
            self.summary_writer, self.summary_op = create_summaries(
                model, self.train_summary_dir, training=True
            )

    def _create_dir_structure(self, log_dir):
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

        for key, val in dirs.items():
            io.check_else_make_dir(val)
            setattr(self, key, val)

        for key, val in files.items():
            setattr(self, key, val)

    def log_step(self, sess, data, net_weights):
        """Update self.logger.summaries."""
        feed_dict = {
            self.model.x: data['x_in'],
            self.model.beta: data['beta'],
            self.model.net_weights[0]: net_weights[0],
            self.model.net_weights[1]: net_weights[1],
            self.model.net_weights[2]: net_weights[2],
            self.model.train_phase: True
        }
        summary_str = sess.run(self.summary_op, feed_dict=feed_dict)

        self.writer.add_summary(summary_str, global_step=data['step'])
        self.writer.flush()

    def update(self, sess, data, data_str, net_weights):
        """Update _current state and train_data."""
        print_steps = getattr(self, 'print_steps', 1)
        step = data['step']

        if (step + 1) % print_steps == 0:
            io.log(data_str)

        if (step + 1) % 100 == 0:
            io.log(self.train_header)

        for key, val in data.items():
            try:
                self.train_data[key].append(val)
            except KeyError:
                self.train_data[key] = [val]

        #  self.train_data[step] = TrainData(data['loss_op'], data['px'], data)

        '''
        if self._model_type == 'GaugeModel':
            self.obs_data[step] = ObsData(data.actions, data.plaqs,
                                          data.charges)  #, data.charge_diffs)
        '''

        self.train_data_strings.append(data_str)
        if self.summaries and (step + 1) % self.logging_steps == 0:
            self.log_step(sess, data, net_weights)

    def write_train_strings(self):
        """Write training strings out to file."""
        tlf = self.train_log_file
        _ = [io.write(s, tlf, 'a') for s in self.train_data_strings]
