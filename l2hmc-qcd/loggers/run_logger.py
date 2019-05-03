"""
run_logger.py

Implements RunLogger class responsible for saving/logging data
from `run` phase of GaugeModel.

Author: Sam Foreman (github: @saforem2)
Date: 04/24/2019
"""
import os
import pickle

import numpy as np

from collections import Counter, OrderedDict
from scipy.stats import sem
import utils.file_io as io

from globals import RUN_HEADER

from lattice.lattice import u1_plaq_exact

from .train_logger import save_params


def arr_from_dict(d, key):
    return np.array(list(d[key]))


def autocorr(x):
    autocorr = np.correlate(x, x, mode='full')

    return autocorr[autocorr.size // 2:]


class RunLogger:
    def __init__(self, model, log_dir):
        """
        Args:
            model: GaugeModel object.
            log_dir: Existing logdir from `TrainLogger`.
        """
        #  self.sess = sess
        self.model = model
        assert os.path.isdir(log_dir)
        self.log_dir = log_dir
        self.runs_dir = os.path.join(self.log_dir, 'runs')
        self.figs_dir = os.path.join(self.log_dir, 'figures')
        io.check_else_make_dir(self.runs_dir)

        self.run_steps = None
        self.beta = None
        self.run_data = {}
        self.run_stats = {}
        self.run_strings = [RUN_HEADER]

        self.samples_arr = [] if self.model.save_samples else None

    def reset(self, run_steps, beta):
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
        self.samples_arr = [] if self.model.save_samples else None
        eps = self.model.eps
        self.run_dir = os.path.join(
            self.runs_dir, f"steps_{run_steps}_beta_{beta}_eps_{eps:.3g}"
        )
        io.check_else_make_dir(self.run_dir)
        save_params(self.model.params, self.run_dir)

    def update(self, data, data_str):
        """Update run_data and append data_str to data_strings."""
        # projection of samples onto [0, 2π) done in run_step above
        if self.model.save_samples:
            samples_np = data['samples']
            self.samples_arr.append(samples_np)

        step = data['step']
        beta = data['beta']
        key = (step, beta)
        self.run_data['px'][key] = data['px']
        self.run_data['actions'][key] = data['actions']
        self.run_data['plaqs'][key] = data['plaqs']
        self.run_data['charges'][key] = data['charges']
        self.run_data['charge_diffs'][key] = data['charge_diffs']
        self.run_strings.append(data_str)

        if step % (10 * self.model.print_steps) == 0:
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

    def save_run_data(self, therm_frac=10):
        """Save run information."""
        observables_dir = os.path.join(self.run_dir, 'observables')

        io.check_else_make_dir(self.run_dir)
        io.check_else_make_dir(observables_dir)

        if self.model.save_samples:
            samples_file = os.path.join(self.run_dir, 'run_samples.pkl')
            io.log(f"Saving samples to: {samples_file}.")
            with open(samples_file, 'wb') as f:
                pickle.dump(self.samples_arr, f)

        run_stats = self.calc_observables_stats(self.run_data, therm_frac)
        charges = self.run_data['charges']
        charges_arr = np.array(list(charges.values()))
        charges_autocorrs = [autocorr(x) for x in charges_arr.T]
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

        ns = self.model.num_samples
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
