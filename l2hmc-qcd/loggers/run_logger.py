"""
run_logger.py

Implements RunLogger class responsible for saving/logging data
from `run` phase of GaugeModel.

Author: Sam Foreman (github: @saforem2)
Date: 04/24/2019
"""
import os
import pickle
import datetime

import tensorflow as tf
import numpy as np

from collections import Counter, OrderedDict
from scipy.stats import sem
import utils.file_io as io

from config import RUN_HEADER, NP_FLOAT, TF_FLOAT

from lattice.lattice import u1_plaq_exact, u1_plaq_exact_tf
from utils.tf_logging import variable_summaries

from .train_logger import save_params


FB_DICT = {
    'forward': [],
    'backward': [],
}


def arr_from_dict(d, key):
    return np.array(list(d[key]))


def autocorr(x):
    autocorr = np.correlate(x, x, mode='full')

    return autocorr[autocorr.size // 2:]


class RunLogger:
    def __init__(self, params, inputs, run_ops, save_lf_data=False):
        """
        Args:
            model: GaugeModel object.
            log_dir: Existing logdir from `TrainLogger`.
        """
        self.params = params
        self.save_lf_data = save_lf_data
        self.summaries = params['summaries']
        assert os.path.isdir(params['log_dir'])

        self.log_dir = params['log_dir']

        runs_dir = os.path.join(self.log_dir, 'runs')
        figs_dir = os.path.join(self.log_dir, 'figures')
        if os.path.isdir(runs_dir) or os.path.isdir(figs_dir):
            now = datetime.datetime.now()
            time_str = now.strftime("%H%M")
            if os.path.isdir(runs_dir):
                renamed_runs_dir = runs_dir + f'_{time_str}'
                io.log(f'Renaming existing runs_dir to: {renamed_runs_dir}')
                os.rename(runs_dir, renamed_runs_dir)
            if os.path.isdir(figs_dir):
                renamed_figs_dir = figs_dir + f'_{time_str}'
                io.log(f'Renaming existing figs_dir to: {renamed_figs_dir}')
                os.rename(figs_dir, renamed_figs_dir)

        self.runs_dir = runs_dir
        io.check_else_make_dir(self.runs_dir)

        self.figs_dir = figs_dir
        io.check_else_make_dir(self.figs_dir)

        self._reset_counter = 0
        self.run_steps = None
        self.beta = None
        self.run_data = {}
        self.run_stats = {}
        self.run_strings = [RUN_HEADER]

        if params['save_lf']:
            self.samples_arr = []
            self.lf_out = FB_DICT.copy()
            self.pxs_out = FB_DICT.copy()
            self.masks = FB_DICT.copy()
            self.logdets = FB_DICT.copy()
            self.sumlogdet = FB_DICT.copy()
            self.l2hmc_fns = FB_DICT.copy()
            #  self.lf_out = {
            #      'forward': [],
            #      'backward': [],
            #  }
            #  self.pxs_out = {
            #      'forward': [],
            #      'backward': [],
            #  }
            #  self.masks = {
            #      'forward': [],
            #      'backward': [],
            #  }
            #  self.logdets = {
            #      'forward': [],
            #      'backward': [],
            #  }
            #  self.sumlogdet = {
            #      'forward': [],
            #      'backward': [],
            #  }

        self.run_ops_dict = self.build_run_ops_dict(params, run_ops)
        self.inputs_dict = self.build_inputs_dict(inputs)

        if self.summaries:
            self.run_summaries_dir = os.path.join(self.log_dir,
                                                  'summaries', 'run')
            io.check_else_make_dir(self.run_summaries_dir)

            self.writer = tf.summary.FileWriter(self.run_summaries_dir,
                                                tf.get_default_graph())
            self.create_summaries()

    def build_run_ops_dict(self, params, run_ops):
        """Build dictionary of tensorflow operations used for inference."""
        def get_lf_keys(direction):
            base_keys = ['lf_out', 'pxs_out', 'masks',
                         'logdets', 'sumlogdet', 'fns_out']
            return [k + f'_{direction}' for k in base_keys]

        keys = ['x_out', 'px', 'actions_op',
                'plaqs_op', 'avg_plaqs_op',
                'charges_op', 'charge_diffs_op']

        run_ops_dict = {key: run_ops[idx] for idx, key in enumerate(keys)}

        if params['save_lf']:
            keys.extend(get_lf_keys('f'))
            keys.extend(get_lf_keys('b'))
            for key, val in zip(keys[7:], run_ops[7:]):
                run_ops_dict.update({key: val})
        #  for idx, key in enumerate(keys):
        #      run_ops_dict[key] = run_ops[idx]

        #  run_ops_dict = {
        #      'x_out': run_ops[0],
        #      'px': run_ops[1],
        #      'actions_op': run_ops[2],
        #      'plaqs_op': run_ops[3],
        #      'avg_plaqs_op': run_ops[4],
        #      'charges_op': run_ops[5],
        #      'charge_diffs_op': run_ops[6]
        #  }

        #  run_ops_dict.update({
        #      'lf_out_f': run_ops[7],
        #      'pxs_out_f': run_ops[8],
        #      'masks_f': run_ops[9],
        #      'logdets_f': run_ops[10],
        #      'sumlogdet_f': run_ops[11],
        #      'fns_out_f': run_ops[12],
        #      'lf_out_b': run_ops[13],
        #      'pxs_out_b': run_ops[14],
        #      'masks_b': run_ops[15],
        #      'logdets_b': run_ops[16],
        #      'sumlogdet_b': run_ops[17],
        #      'fns_out_b': run_ops[18]
        #  })

        return run_ops_dict

    def build_inputs_dict(self, inputs):
        """Build dictionary of tensorflow placeholders used as inputs."""
        inputs_dict = {
            'x': inputs[0],
            'beta': inputs[1],
            'charge_weight': inputs[2],
            'train_phase': inputs[3],
            'net_weights': [inputs[4], inputs[5], inputs[6]]
        }

        return inputs_dict

    def create_summaries(self):
        """Create summary objects for logging in TensorBoard."""
        ld = self.log_dir
        self.summary_writer = tf.contrib.summary.create_file_writer(ld)

        with tf.name_scope('avg_actions_inference'):
            tf.summary.scalar('avg_actions',
                              tf.reduce_mean(self.run_ops_dict['actions_op']))

        with tf.name_scope('avg_plaqs_inference'):
            tf.summary.scalar('avg_plaqs', self.run_ops_dict['avg_plaqs_op'])

        with tf.name_scope('avg_plaq_diff_inference'):
            tf.summary.scalar('avg_plaq_diff',
                              (u1_plaq_exact_tf(self.inputs_dict['beta'])
                               - self.run_ops_dict['avg_plaqs_op']))

        with tf.name_scope('avg_charge_diffs_inference'):
            tf.summary.scalar('avg_charge_diffs',
                              tf.reduce_mean(
                                  self.run_ops_dict['charge_diffs_op']
                              ))

        self.summary_op = tf.summary.merge_all(name='run_summary_op')

    def log_step(self, sess, step, samples_np, beta_np, net_weights):
        """Update self.logger.summaries."""
        feed_dict = {
            self.inputs_dict['x']: samples_np,
            self.inputs_dict['beta']: beta_np,
            self.inputs_dict['net_weights'][0]: net_weights[0],
            self.inputs_dict['net_weights'][1]: net_weights[1],
            self.inputs_dict['net_weights'][2]: net_weights[2],
            self.inputs_dict['train_phase']: False
        }
        #  feed_dict = {
        #      self.model.x: samples_np,
        #      self.model.beta: beta_np,
        #      self.model.net_weights[0]: net_weights[0],
        #      self.model.net_weights[1]: net_weights[1],
        #      self.model.net_weights[2]: net_weights[2],
        #      self.model.train_phase: False
        #  }
        summary_str = sess.run(self.summary_op, feed_dict=feed_dict)

        self.writer.add_summary(summary_str, global_step=step)
        self.writer.flush()

    def reset(self, run_steps, beta, weights, eps_np, dir_append=None):
        """Reset run_data and run_strings to prep for new run."""
        self.run_steps = int(run_steps)
        self.beta = beta

        self.run_data = {
            'px': {},
            'actions': {},
            'plaqs': {},
            'charges': {},
            'charge_diffs': {},
        }
        self.run_stats = {}
        self.run_strings = []
        #  if self.model.save_lf:
        if self.params['save_lf']:
            self.samples_arr = []
            self.lf_out = FB_DICT.copy()
            self.pxs_out = FB_DICT.copy()
            self.masks = FB_DICT.copy()
            self.logdets = FB_DICT.copy()
            self.sumlogdet = FB_DICT.copy()
            self.l2hmc_fns = FB_DICT.copy()

        #  eps = self.model.eps
        charge_weight = weights['charge_weight']
        net_weights = weights['net_weights']

        if charge_weight is None:
            charge_weight = 0.

        if net_weights is None:
            net_weights = [1., 1., 1.]

        eps_str = f'{eps_np:.3}'.replace('.', '')
        beta_str = f'{beta:.3}'.replace('.', '')
        qw_str = f'{charge_weight:.3}'.replace('.', '')

        run_str = (
            f'steps_{run_steps}'
            f'_beta_{beta_str}'
            f'_eps_{eps_str}'
            f'_qw_{qw_str:.2}'
            f'_{self._reset_counter}'
        )

        if dir_append:
            run_str += dir_append

        #  params = self.model.params
        self.params['net_weights'] = net_weights

        self.run_dir = os.path.join(self.runs_dir, run_str)
        io.check_else_make_dir(self.run_dir)

        if self.summaries:
            self.run_summary_dir = os.path.join(self.run_summaries_dir,
                                                run_str)
            io.check_else_make_dir(self.run_summary_dir)

            self.writer = tf.summary.FileWriter(self.run_summary_dir,
                                                tf.get_default_graph())
        save_params(self.params, self.run_dir)

        self._reset_counter += 1

        def _round_float_as_str(f):
            return f'{f:.3g}'

        weights_txt_file = os.path.join(self.run_dir, 'weights.txt')
        charge_weight_str = f'charge_weight: {charge_weight}\n'

        nw_str = [_round_float_as_str(w) for w in net_weights]
        w_str = nw_str[0] + nw_str[1] + nw_str[2]
        with open(weights_txt_file, 'w') as f:
            f.write(charge_weight_str)
            f.write(w_str)

        return self.run_dir, run_str

    def update(self, sess, data, net_weights, data_str):
        """Update run_data and append data_str to data_strings."""
        # projection of samples onto [0, 2pi) done in run_step above
        #  if self.model.save_samples:
        step = data['step']
        beta = data['beta']
        key = (step, beta)

        obs_keys = ['px', 'actions', 'plaqs', 'charges', 'charge_diffs']
        for k in obs_keys:
            self.run_data[k][key] = data[k]
        #  self.run_data['px'][key] = data['px']
        #  self.run_data['actions'][key] = data['actions']
        #  self.run_data['plaqs'][key] = data['plaqs']
        #  self.run_data['charges'][key] = data['charges']
        #  self.run_data['charge_diffs'][key] = data['charge_diffs']

        #  if self.model.save_lf:
        if self.params['save_lf']:
            samples_np = data['samples']
            self.samples_arr.append(samples_np)
            self.lf_out['forward'].extend(np.array(data['lf_out_f']))
            self.lf_out['backward'].extend(np.array(data['lf_out_b']))
            self.logdets['forward'].extend(np.array(data['logdets_f']))
            self.logdets['backward'].extend(np.array(data['logdets_b']))
            self.sumlogdet['forward'].append(np.array(data['sumlogdet_f']))
            self.sumlogdet['backward'].append(np.array(data['sumlogdet_b']))
            #  self.pxs_out['forward'].extend(np.array(data['pxs_out_f']))
            #  self.pxs_out['backward'].extend(np.array(data['pxs_out_b']))
            #  self.masks['forward'].extend(np.array(data['masks_f']))
            #  self.masks['backward'].extend(np.array(data['masks_b']))

        self.run_strings.append(data_str)

        #  if self.summaries and (step + 1) % self.model.logging_steps == 0:
        if self.summaries and (step + 1) % self.params['logging_steps'] == 0:
            self.log_step(sess, step, data['samples'], beta, net_weights)

        #  if step % (10 * self.model.print_steps) == 0:
        if step % (10 * self.params['print_steps']) == 0:
            io.log(data_str)

        if step % 100 == 0:
            io.log(RUN_HEADER)

    def calc_observables_stats(self, run_data, therm_frac=10):
        """Calculate statistics for lattice observables.

        Args:
            run_data: Dictionary of observables data. Keys denote the
                observables name.
            therm_frac: Fraction of data to throw out for thermalization.

        Returns:
            stats: Dictionary containing statistics for each observable in
            run_data. Additionally, contains `charge_probs` which is a
            dictionary of the form {charge_val: charge_val_probability}.
        """
        def get_stats(data, t_frac=10):
            if isinstance(data, dict):
                arr = np.array(list(data.values()))
            elif isinstance(data, (list, np.ndarray)):
                arr = np.array(data)

            num_steps = arr.shape[0]
            therm_steps = num_steps // t_frac
            arr = arr[therm_steps:, :]
            avg = np.mean(arr, axis=0)
            err = sem(arr, axis=0)
            stats = np.array([avg, err]).T
            return stats

        actions_stats = get_stats(run_data['actions'], therm_frac)
        plaqs_stats = get_stats(run_data['plaqs'], therm_frac)

        charges_arr = np.array(list(run_data['charges'].values()), dtype=int)
        charges_stats = get_stats(charges_arr, therm_frac)

        suscept_arr = charges_arr ** 2
        suscept_stats = get_stats(suscept_arr)

        charge_probs = {}
        counts = Counter(list(charges_arr.flatten()))
        total_counts = np.sum(list(counts.values()))
        for key, val in counts.items():
            charge_probs[key] = val / total_counts

        charge_probs = OrderedDict(sorted(charge_probs.items(),
                                          key=lambda k: k[0]))

        stats = {
            'actions': actions_stats,
            'plaqs': plaqs_stats,
            'charges': charges_stats,
            'suscept': suscept_stats,
            'charge_probs': charge_probs
        }

        return stats

    def save_attr(self, name, attr, out_dir=None, dtype=NP_FLOAT):
        if out_dir is None:
            out_dir = self.run_dir

        assert os.path.isdir(out_dir)
        out_file = os.path.join(out_dir, name + '.npz')

        if not isinstance(attr, np.ndarray) or attr.dtype != dtype:
            attr = np.array(attr, dtype=dtype)

        if os.path.isfile(out_file):
            io.log(f'File {out_file} already exists. Skipping.')
        else:
            io.log(f'Saving {name} to: {out_file}')
            np.savez_compressed(out_file, attr)

    def save_run_data(self, therm_frac=10):
        """Save run information."""
        observables_dir = os.path.join(self.run_dir, 'observables')

        io.check_else_make_dir(self.run_dir)
        io.check_else_make_dir(observables_dir)

        if self.save_lf_data:
            keys = ['lf_out', 'masks', 'logdets', 'sumlogdet', 'pxs_out']
            for key in keys:
                f = 'forward'
                b = 'backward'
                self.save_attr(key + f'_{f}', getattr(self, key)[f])
                self.save_attr(key + f'_{b}', getattr(self, key)[b])

        #  if self.save_lf_data:
        #      self.save_attr('samples_out', self.samples_arr)
        #      self.save_attr('lf_out_forward', self.lf_out['forward'])
        #      self.save_attr('lf_out_backward', self.lf_out['backward'])
        #      self.save_attr('masks_forward', self.masks['forward'])
        #      self.save_attr('masks_backward', self.masks['backward'])
        #      self.save_attr('logdets_forward', self.logdets['forward'])
        #      self.save_attr('logdets_backward', self.logdets['backward'])
        #      self.save_attr('sumlogdet_forward', self.sumlogdet['forward'])
        #      self.save_attr('sumlogdet_backward', self.sumlogdet['backward'])
        #      self.save_attr('accept_probs_forward', self.pxs_out['forward'])
        #     self.save_attr('accept_probs_backward', self.pxs_out['backward'])

        run_stats = self.calc_observables_stats(self.run_data, therm_frac)
        charges = self.run_data['charges']
        charges_arr = np.array(list(charges.values()))
        charges_autocorrs = [autocorr(x) for x in charges_arr.T]
        charges_autocorrs = [x / np.max(x) for x in charges_autocorrs]
        self.run_data['charges_autocorrs'] = charges_autocorrs

        data_file = os.path.join(self.run_dir, 'run_data.pkl')
        io.log(f"Saving run_data to: {data_file}.")
        with open(data_file, 'wb') as f:
            pickle.dump(self.run_data, f)

        stats_data_file = os.path.join(self.run_dir, 'run_stats.pkl')
        io.log(f"Saving run_stats to: {stats_data_file}.")
        with open(stats_data_file, 'wb') as f:
            pickle.dump(run_stats, f)

        for key, val in self.run_data.items():
            out_file = key + '.pkl'
            out_file = os.path.join(observables_dir, out_file)
            io.save_data(val, out_file, name=key)

        for key, val in run_stats.items():
            out_file = key + '_stats.pkl'
            out_file = os.path.join(observables_dir, out_file)
            io.save_data(val, out_file, name=key)

        history_file = os.path.join(self.run_dir, 'run_history.txt')
        io.write(RUN_HEADER, history_file, 'w')
        _ = [io.write(s, history_file, 'a') for s in self.run_strings]

        self.write_run_stats(run_stats, therm_frac)

    def write_run_stats(self, stats, therm_frac=10):
        """Write statistics in human readable format to .txt file."""
        #  run_steps = kwargs['run_steps']
        #  beta = kwargs['beta']
        #  current_step = kwargs['current_step']
        #  therm_steps = kwargs['therm_steps']
        #  training = kwargs['training']
        #  run_dir = kwargs['run_dir']
        therm_steps = self.run_steps // therm_frac

        out_file = os.path.join(self.run_dir, 'run_stats.txt')

        actions_avg, actions_err = stats['actions'].mean(axis=0)
        plaqs_avg, plaqs_err = stats['plaqs'].mean(axis=0)
        charges_avg, charges_err = stats['charges'].mean(axis=0)
        suscept_avg, suscept_err = stats['suscept'].mean(axis=0)

        #  ns = self.model.num_samples
        ns = self.params['num_samples']
        suscept_k1 = f'  \navg. over all {ns} samples < Q >'
        suscept_k2 = f'  \navg. over all {ns} samples < Q^2 >'
        actions_k1 = f'  \navg. over all {ns} samples < action >'
        plaqs_k1 = f'  \n avg. over all {ns} samples < plaq >'

        _est_key = '  \nestimate +/- stderr'

        suscept_ss = {
            suscept_k1: f"{charges_avg:.4g} +/- {charges_err:.4g}",
            suscept_k2: f"{suscept_avg:.4g} +/- {suscept_err:.4g}",
            _est_key: {}
        }

        actions_ss = {
            actions_k1: f"{actions_avg:.4g} +/- {actions_err:.4g}\n",
            _est_key: {}
        }

        plaqs_ss = {
            'exact_plaq': f"{u1_plaq_exact(self.beta):.4g}\n",
            plaqs_k1: f"{plaqs_avg:.4g} +/- {plaqs_err:.4g}\n",
            _est_key: {}
        }

        def format_stats(x, name=None):
            return [f'{name}: {i[0]:.4g} +/- {i[1]:.4g}' for i in x]

        def zip_keys_vals(stats_strings, keys, vals):
            for k, v in zip(keys, vals):
                stats_strings[_est_key][k] = v
            return stats_strings

        keys = [f"sample {idx}" for idx in range(ns)]

        suscept_vals = format_stats(stats['suscept'], '< Q^2 >')
        actions_vals = format_stats(stats['actions'], '< action >')
        plaqs_vals = format_stats(stats['plaqs'], '< plaq >')

        suscept_ss = zip_keys_vals(suscept_ss, keys, suscept_vals)
        actions_ss = zip_keys_vals(actions_ss, keys, actions_vals)
        plaqs_ss = zip_keys_vals(plaqs_ss, keys, plaqs_vals)

        def accumulate_strings(d):
            all_strings = []
            for k1, v1 in d.items():
                if isinstance(v1, dict):
                    for k2, v2 in v1.items():
                        all_strings.append(f'{k2} {v2}')
                else:
                    all_strings.append(f'{k1}: {v1}\n')

            return all_strings

        actions_strings = accumulate_strings(actions_ss)
        plaqs_strings = accumulate_strings(plaqs_ss)
        suscept_strings = accumulate_strings(suscept_ss)

        charge_probs_strings = []
        for k, v in stats['charge_probs'].items():
            charge_probs_strings.append(f'  probability[Q = {k}]: {v}\n')

        #  train_str = (f" stats after {current_step} training steps.\n"
        #               f"{ns} chains ran for {self.run_steps} steps at "
        #               f"beta = {self.beta}.")

        run_str = (f" stats for {ns} chains ran for {self.run_steps} steps "
                   f" at beta = {self.beta}.")

        #  if training:
        #      str0 = "Topological susceptibility" + train_str
        #      str1 = "Total actions" + train_str
        #      str2 = "Average plaquette" + train_str
        #      str3 = "Topological charge probabilities" + train_str[6:]
        #      therm_str = ''
        #  else:
        str0 = "Topological susceptibility" + run_str
        str1 = "Total actions" + run_str
        str2 = "Average plaquette" + run_str
        str3 = "Topological charge probabilities" + run_str[6:]
        therm_str = (
            f'Ignoring first {therm_steps} steps for thermalization.'
        )

        ss0 = (1 + max(len(str0), len(therm_str))) * '-'
        ss1 = (1 + max(len(str1), len(therm_str))) * '-'
        ss2 = (1 + max(len(str2), len(therm_str))) * '-'
        ss3 = (1 + max(len(str3), len(therm_str))) * '-'

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

        log_and_write(ss0, str0, therm_str, suscept_strings, out_file)
        log_and_write(ss1, str1, therm_str, actions_strings, out_file)
        log_and_write(ss2, str2, therm_str, plaqs_strings, out_file)
        log_and_write(ss3, str3, therm_str, charge_probs_strings, out_file)
        log_and_write(ss3, str3, therm_str, charge_probs_strings, out_file)
        log_and_write(ss3, str3, therm_str, charge_probs_strings, out_file)
