"""
gauge_model_logger.py

Implements GaugeModelLogger class responsible for saving/logging data from
GaugeModel.

Author: Sam Foreman (github: @saforem2)
Date: 04/09/2019
"""
import os
import time
import pickle

from collections import Counter, OrderedDict
from scipy.stats import sem
import utils.file_io as io

from globals import FILE_PATH, NP_FLOAT, TRAIN_HEADER, RUN_HEADER
from lattice.lattice import u1_plaq_exact
from utils.tf_logging import variable_summaries

import numpy as np

import tensorflow as tf

try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


def save_params(params, log_dir):
    io.check_else_make_dir(log_dir)
    params_txt_file = os.path.join(log_dir, 'parameters.txt')
    params_pkl_file = os.path.join(log_dir, 'parameters.pkl')
    with open(params_txt_file, 'w') as f:
        for key, val in params.items():
            f.write(f"{key}: {val}\n")
    with open(params_pkl_file, 'wb') as f:
        pickle.dump(params, f)


class GaugeModelLogger:
    def __init__(self, sess, model, log_dir, summaries=False):
        self.sess = sess
        self.model = model
        self.summaries = summaries

        #  condition1 = self.model.using_hvd and hvd.rank() == 0
        #  condition2 = not self.model.using_hvd
        #  self.is_chief = condition1 or condition2

        self.charges_dict = {}
        self.charge_diffs_dict = {}

        self._current_state = {
            'step': 0,
            'beta': self.model.beta_init,
            'eps': self.model.eps,
            'lr': self.model.lr_init,
            'samples': self.model.samples,
        }

        self.train_data_strings = []
        self.train_data = {
            'loss': {},
            'actions': {},
            'plaqs': {},
            'charges': {},
            'charge_diffs': {},
            'accept_probs': {}
        }

        self.run_data_strings = []
        self.run_data = {}

        # log_dir will be None if using_hvd and hvd.rank() != 0
        # this prevents workers on different ranks from corrupting checkpoints
        #  if log_dir is not None and self.is_chief:
        self._create_dir_structure(log_dir)
        save_params(self.model.params, self.log_dir)

        if self.summaries:
            #  self.summary_placeholders = {}
            #  self.summary_ops = {}

            self.writer = tf.summary.FileWriter(self.train_summary_dir,
                                                self.sess.graph)
            self.create_summaries()

    def _create_dir_structure(self, log_dir):
        """Create relevant directories for storing data."""
        #  project_dir = os.path.abspath(os.path.dirname(FILE_PATH))
        project_dir = os.path.abspath(os.path.join('..', '..'))
        root_log_dir = os.path.join(project_dir, log_dir)
        io.check_else_make_dir(root_log_dir)

        run_num = io.get_run_num(root_log_dir)
        log_dir = os.path.abspath(os.path.join(root_log_dir,
                                               f'run_{run_num}'))
        io.check_else_make_dir(log_dir)

        self.log_dir = log_dir
        self.train_dir = os.path.join(self.log_dir, 'training')
        self.figs_dir = os.path.join(self.log_dir, 'figures')
        self.runs_dir = os.path.join(self.log_dir, 'runs')
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        self.train_runs_dir = os.path.join(self.runs_dir, 'training')
        self.train_summary_dir = os.path.join(
            self.log_dir, 'summaries', 'train'
        )

        self.train_log_file = os.path.join(self.train_dir, 'training_log.txt')
        self.current_state_file = os.path.join(self.train_dir,
                                               'current_state.pkl')

        io.make_dirs([self.train_dir, self.figs_dir, self.runs_dir,
                      self.train_runs_dir, self.train_summary_dir,
                      self.checkpoint_dir])

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in GaugeModel.

        Generates a moving average for all losses and associated summaries for
        visualizing the performance of the network.

        Args:
            total_loss: Total loss from model._calc_loss()

        Returns:
            loss_averages_op: Op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total
        # loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss;
        # do the same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of
            # the loss as the original loss name.
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op

    def create_summaries(self):
        """Create summary objects for logging in TensorBoard."""
        ld = self.log_dir
        self.summary_writer = tf.contrib.summary.create_file_writer(ld)

        grads_and_vars = zip(self.model.grads,
                             self.model.dynamics.trainable_variables)

        with tf.name_scope('loss'):
            tf.summary.scalar('loss', self.model.loss_op)

        #  self.loss_averages_op = self._add_loss_summaries(self.model.loss_op)

        with tf.name_scope('learning_rate'):
            tf.summary.scalar('learning_rate', self.model.lr)

        with tf.name_scope('step_size'):
            tf.summary.scalar('step_size', self.model.dynamics.eps)

        with tf.name_scope('tunneling_events'):
            tf.summary.scalar('tunneling_events_per_sample',
                              self.model.charge_diffs_op)

        with tf.name_scope('avg_plaq'):
            tf.summary.scalar('avg_plaq', self.model.avg_plaqs_op)

        for var in tf.trainable_variables():
            if 'batch_normalization' not in var.op.name:
                tf.summary.histogram(var.op.name, var)

        with tf.name_scope('summaries'):
            for grad, var in grads_and_vars:
                try:
                    layer, _type = var.name.split('/')[-2:]
                    name = layer + '_' + _type[:-2]
                except (AttributeError, IndexError):
                    name = var.name[:-2]

                if 'batch_norm' not in name:
                    variable_summaries(var, name)
                    variable_summaries(grad, name + '/gradients')
                    tf.summary.histogram(name + '/gradients', grad)

        self.summary_op = tf.summary.merge_all(name='summary_op')

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

    def log_step(self, step, samples_np, beta_np):
        """Update self.logger.summaries."""
        feed_dict = {
            self.model.x: samples_np,
            self.model.beta: beta_np
        }
        summary_str = self.sess.run(self.summary_op,
                                    feed_dict=feed_dict)

        self.writer.add_summary(summary_str, global_step=step)
        self.writer.flush()

    def update_training(self, data, data_str):
        """Update _current state and train_data."""
        step = data['step']
        beta = data['beta']
        self._current_state['step'] = step
        self._current_state['beta'] = beta
        self._current_state['lr'] = data['lr']
        self._current_state['eps'] = data['eps']
        self._current_state['samples'] = data['samples']

        key = (step, beta)

        self.charges_dict[key] = data['charges']
        self.charge_diffs_dict[key] = data['charge_diffs']

        self.train_data['loss'][key] = data['loss']
        self.train_data['actions'][key] = data['actions']
        self.train_data['plaqs'][key] = data['plaqs']
        self.train_data['charges'][key] = data['charges']
        self.train_data['charge_diffs'][key] = data['charge_diffs']
        self.train_data['accept_probs'][key] = data['px']

        self.train_data_strings.append(data_str)
        if step % self.model.print_steps == 0:
            io.log(data_str)

        if (step + 1) % self.model.logging_steps == 0:
            self.log_step(step, data['samples'], beta)

        if (step + 1) % self.model.save_steps == 0:
            self.model.save(self.sess, self.checkpoint_dir)
            self.save_current_state()

        if step % 100 == 0:
            io.log(TRAIN_HEADER)

        #  if step == self.model.train_steps - 1:
        #      self.model.save(self.sess, self.checkpoint_dir)
        #      self.save_current_state()
        #      self.write_train_strings()

    def write_train_strings(self):
        """Write training strings out to file."""
        tlf = self.train_log_file
        _ = [io.write(s, tlf, 'a') for s in self.train_data_strings]

    def update_running(self, step, data, data_str, run_params):
        """Update run_data generated during evaluation of L2HMC algorithm."""
        run_key = (run_params['run_steps'], run_params['beta'])

        self.run_data[run_key] = {
            'accept_probs': data['px'],
            'actions': data['actions'],
            'plaqs': data['plaqs'],
            'charges': data['charges'],
            'charge_diffs': data['charge_diffs']
        }

        self.run_data_strings.append(data_str + '\n')
        if step % self.model.print_steps == 0:
            io.log(data_str)

    def write_run_strings(self, run_strings, out_file):
        """Write evaluation (`run`) strings out to file."""
        _ = [io.write(s, out_file, 'a') for s in run_strings]

    def save_run_info(self, run_data, stats, out_dir):
        """Save observables contained in `run_data` to pickle files."""
        io.check_else_make_dir(out_dir)

        data_file = os.path.join(out_dir, 'run_data.pkl')
        io.log(f"Saving run_data to: {data_file}.")
        with open(data_file, 'wb') as f:
            pickle.dump(run_data, f)

        stats_data_file = os.path.join(out_dir, 'run_data_stats.pkl')
        io.log(f"Saving run_data_stats to: {stats_data_file}.")
        with open(stats_data_file, 'wb') as f:
            pickle.dump(stats, f)

        observables_dir = os.path.join(out_dir, 'observables')
        io.check_else_make_dir(observables_dir)
        for key, val in run_data.items():
            out_file = key + '.pkl'
            out_file = os.path.join(observables_dir, out_file)
            io.save_data(val, out_file, name=key)

        for key, val in stats.items():
            out_file = key + '_stats.pkl'
            out_file = os.path.join(observables_dir, out_file)
            io.save_data(val, out_file, name=key)

    def write_run_stats(self, run_data, stats, out_file, **kwargs):
        """Write statistics in human readable format to .txt file."""
        beta = kwargs['beta']
        training = kwargs['training']
        current_step = kwargs['current_step']
        run_steps = kwargs['run_steps']
        therm_steps = kwargs['therm_steps']

        actions_arr = np.array(
            list(run_data['actions'].values())
        )[therm_steps:, :]

        plaqs_arr = np.array(
            list(run_data['plaqs'].values())
        )[therm_steps:, :]

        charges_arr = np.array(
            list(run_data['charges'].values()),
            dtype=np.int32
        )[therm_steps:, :]

        charges_squared_arr = charges_arr ** 2

        actions_avg = np.mean(actions_arr)
        actions_err = sem(actions_arr, axis=None)

        plaqs_avg = np.mean(plaqs_arr)
        plaqs_err = sem(plaqs_arr, axis=None)

        q_avg = np.mean(charges_arr)
        q_err = sem(charges_arr, axis=None)

        q2_avg = np.mean(charges_squared_arr)
        q2_err = sem(charges_squared_arr, axis=None)

        ns = self.model.num_samples
        suscept_k1 = f'  \navg. over all {ns} samples < Q >'
        suscept_k2 = f'  \navg. over all {ns} samples < Q^2 >'
        actions_k1 = f'  \navg. over all {ns} samples < action >'
        plaqs_k1 = f'  \n avg. over all {ns} samples < plaq >'

        _est_key = '  \nestimate +/- stderr'

        suscept_stats_strings = {
            suscept_k1: f"{q_avg:.4g} +/- {q_err:.4g}",
            suscept_k2: f"{q2_avg:.4g} +/- {q2_err:.4g}",
            _est_key: {}
        }

        actions_stats_strings = {
            actions_k1: f"{actions_avg:.4g} +/- {actions_err:.4g}\n",
            _est_key: {}
        }

        plaqs_stats_strings = {
            'exact_plaq': f"{u1_plaq_exact(beta):.4g}\n",
            plaqs_k1: f"{plaqs_avg:.4g} +/- {plaqs_err:.4g}\n",
            _est_key: {}
        }

        def format_stats(stats, name=None):
            return [
                f'{name}: {i[0]:.4g} +/- {i[1]:.4g}' for i in stats
            ]

        def zip_keys_vals(stats_strings, keys, vals):
            for k, v in zip(keys, vals):
                stats_strings[_est_key][k] = v
            return stats_strings

        keys = [
            f"sample {idx}" for idx in range(len(stats['suscept'][0]))
        ]

        suscept_vals = format_stats(stats['suscept'], '< Q^2 >')
        actions_vals = format_stats(stats['actions'], '< action >')
        plaqs_vals = format_stats(stats['plaqs'], '< plaq >')

        suscept_stats_strings = zip_keys_vals(suscept_stats_strings,
                                              keys, suscept_vals)
        actions_stats_strings = zip_keys_vals(actions_stats_strings,
                                              keys, actions_vals)
        plaqs_stats_strings = zip_keys_vals(plaqs_stats_strings,
                                            keys, plaqs_vals)

        def accumulate_strings(d):
            all_strings = []
            for k1, v1 in d.items():
                if isinstance(v1, dict):
                    for k2, v2 in v1.items():
                        all_strings.append(f'{k2} {v2}')
                else:
                    all_strings.append(f'{k1}: {v1}\n')

            return all_strings

        actions_strings = accumulate_strings(actions_stats_strings)
        plaqs_strings = accumulate_strings(plaqs_stats_strings)
        suscept_strings = accumulate_strings(suscept_stats_strings)

        charge_probs_strings = []
        for k, v in stats['charge_probs'].items():
            charge_probs_strings.append(f'  probability[Q = {k}]: {v}\n')

        if training:
            str0 = (f'Topological suscept. stats after {current_step} '
                    f'training steps. Chain ran for {run_steps} steps at '
                    f'beta = {beta}.')
            str1 = (f'Total action stats after {current_step} '
                    f'training steps. Chain ran for {run_steps} steps at '
                    f'beta = {beta}.')
            str2 = (f'Average plaquette stats after {current_step} '
                    f'training steps. Chain ran for {run_steps} steps at '
                    f'beta = {beta}.')
            str3 = (f'Topological charge probabilities after '
                    f'{current_step} training steps. '
                    f'Chain ran for {run_steps} steps at beta = {beta}.')
            therm_str = ''
        else:
            str0 = (f'Topological suscept. stats for '
                    f'{run_steps} steps, at beta = {beta}.')
            str1 = (f'Total action stats for '
                    f'{run_steps} steps, at beta = {beta}.')
            str2 = (f'Average plaquette stats for '
                    f'{run_steps} steps, at beta = {beta}.')
            str3 = (f'Topological charge probabilities for '
                    f'{run_steps} steps, at beta = {beta}.')
            therm_str = (
                f'Ignoring first {therm_steps} steps for thermalization.'
            )

        sep_str0 = (1 + max(len(str0), len(therm_str))) * '-'
        sep_str1 = (1 + max(len(str1), len(therm_str))) * '-'
        sep_str2 = (1 + max(len(str2), len(therm_str))) * '-'
        sep_str3 = (1 + max(len(str3), len(therm_str))) * '-'

        io.log(f"Writing statistics to: {out_file}")

        def log_and_write(sep_str, str0, therm_str, stats_strings, file):
            io.log(sep_str)
            io.log(str0)
            io.log(therm_str)
            io.log('')
            _ = [io.log(s) for s in stats_strings]
            io.log(sep_str)
            io.log('')

            io.write(sep_str, file, 'a')
            io.write(str0, file, 'a')
            io.write(therm_str, file, 'a')
            _ = [io.write(s, file, 'a') for s in stats_strings]
            io.write('\n', file, 'a')

        log_and_write(sep_str0, str0, therm_str, suscept_strings, out_file)
        log_and_write(sep_str1, str1, therm_str, actions_strings, out_file)
        log_and_write(sep_str2, str2, therm_str, plaqs_strings, out_file)
        log_and_write(sep_str3, str3, therm_str, charge_probs_strings,
                      out_file)
