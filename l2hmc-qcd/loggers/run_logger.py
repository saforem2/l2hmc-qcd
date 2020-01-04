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

from config import NP_FLOAT, State, NetWeights
from lattice.lattice import u1_plaq_exact

__all__ = ['RunLogger', 'arr_from_dict', 'autocorr']


#  H_STRF = ("{:^12s}" + 4 * "{:^10s}").format(
#      "STEP", "t/STEP", "% ACC", "EPS", "BETA"
#  )


def arr_from_dict(d, key):
    return np.array(list(d[key]))


def autocorr(x):
    autocorr = np.correlate(x, x, mode='full')

    return autocorr[autocorr.size // 2:]


def _rename(src, dst):
    io.log(f'INFO: Renaming {src} to {dst}')
    os.rename(src, dst)


def _get_eps():
    eps = [i for i in tf.global_variables() if 'eps' in i.name][0]
    return eps


def save_dict_npz(d, out_file):
    for key, val in d.items():
        d[key] = np.array(val)

    io.log(f'Saving  dictionary to: {out_file}.')
    np.savez_compressed(out_file, **d)


class RunLogger:
    def __init__(self, params, model_type=None, save_lf_data=True):
        """Initialization method.
        Args:
            model: GaugeModel object.
            log_dir: Existing logdir from `TrainLogger`.
        """
        self.params = params
        self.save_lf_data = save_lf_data
        self.summaries = params['summaries']
        self.model_type = model_type

        self.h_strf = ("{:^12s}" + 5 * "{:^10s}").format(
            "STEP", "t/STEP", "% ACC", "EPS", "deltaX", "BETA",
        )

        if self.model_type == 'GaugeModel':
            self.h_strf += (3 * "{:^10s}").format(
                "ACTIONS", "PLAQS", "(EXACT)"
            )

        dash = (len(self.h_strf) + 1) * '-'
        self.run_header = dash + '\n' + self.h_strf + '\n' + dash

        self.run_steps = None
        self.beta = None
        self.run_data = {}
        self.run_stats = {}
        self.run_strings = [self.run_header]
        self._build_ops_dicts()
        self._build_dir_structure(params['log_dir'])

        if self.summaries:
            self.writer = tf.summary.FileWriter(self.run_summaries_dir,
                                                tf.get_default_graph())
            self.create_summaries()

    def _build_dir_structure(self, log_dir):
        """Simplify `__init__` method by creating dir. structure here."""
        assert os.path.isdir(log_dir)
        self.log_dir = log_dir

        self.runs_dir = os.path.join(self.log_dir, 'runs')
        self.figs_dir = os.path.join(self.log_dir, 'figures')
        self.run_summaries_dir = os.path.join(self.log_dir, 'summaries', 'run')

        io.check_else_make_dir(self.runs_dir)
        io.check_else_make_dir(self.figs_dir)
        io.check_else_make_dir(self.run_summaries_dir)

    def _build_ops_dicts(self):
        """Build dicts w/  key, val pairs for running tensorflow ops."""
        self.run_ops_dict = self.build_run_ops_dict()
        self.inputs_dict = self.build_inputs_dict()

        energy_outputs = self.build_energy_ops_dict()
        self.state_ph = energy_outputs['state']
        self.sumlogdet_ph = energy_outputs['sumlogdet_ph']
        self.energy_ops_dict = energy_outputs['ops_dict']
        self.energy_dict = {}

        if self.model_type == 'GaugeModel':
            self.obs_ops_dict = self.build_obs_ops_dict()
            self.obs_dict = {k: [] for k in self.obs_ops_dict.keys()}

    @staticmethod
    def build_run_ops_dict():
        """Build dictionary of tensorflow operations used for inference."""
        # NOTE: Keys from `run_ops` dict defined in the model implementation
        keys = ['x_init', 'v_init', 'x_proposed', 'v_proposed',
                'x_out', 'v_out', 'dx', 'dxf', 'dxb', 'accept_prob',
                'accept_prob_hmc', 'sumlogdet_proposed', 'sumlogdet_out']

        ops = tf.get_collection('run_ops')

        run_ops_dict = dict(zip(keys, ops))
        eps = _get_eps()
        run_ops_dict.update({'dynamics_eps': eps})

        return run_ops_dict

    @staticmethod
    def build_obs_ops_dict():
        """Build dictionary of tensorflow ops for calculating observables."""
        keys = ['plaq_sums', 'actions', 'plaqs',
                'charges', 'avg_plaqs', 'avg_actions']
        ops = tf.get_collection('observables')

        obs_ops_dict = dict(zip(keys, ops))

        return obs_ops_dict

    @staticmethod
    def build_inputs_dict():
        """Build dictionary of tensorflow placeholders used as inputs."""
        inputs = tf.get_collection('inputs')
        x, beta, eps_ph, global_step_ph, train_phase, *nw = inputs
        net_weights = NetWeights(*nw)
        #  inputs_dict = dict(zip(keys, inputs))
        inputs_dict = {
            'x': x,
            'beta': beta,
            'eps_ph': eps_ph,
            'global_step_ph': global_step_ph,
            'train_phase': train_phase,
            'net_weights': net_weights,
        }

        return inputs_dict

    @staticmethod
    def build_energy_ops_dict():
        """Build dictionary of energy operations to calculate."""
        keys = ['potential_energy', 'kinetic_energy', 'hamiltonian']
        energy_ops = tf.get_collection('energy_ops')
        energy_ops_dict = dict(zip(keys, energy_ops))

        energy_ph = tf.get_collection('energy_placeholders')
        x_ph, v_ph, beta_ph, sumlogdet_ph = energy_ph
        state = State(x=x_ph, v=v_ph, beta=beta_ph)

        outputs = {
            'state': state,
            'sumlogdet_ph': sumlogdet_ph,
            'ops_dict': energy_ops_dict,
        }

        return outputs

    def create_summaries(self):
        """Create summary objects for logging in TensorBoard."""
        summary_list = tf.get_collection(tf.GraphKeys.SUMMARIES)
        ignore_strings = ['loss', 'learning_rate',
                          'step_size', 'train', 'eps']
        run_summaries = [
            i for i in summary_list if not any(
                s in i.name for s in ignore_strings
            )
        ]
        self.summary_op = tf.summary.merge(run_summaries)

    def log_step(self, sess, step, samples, beta, net_weights):
        """Update self.logger.summaries."""
        feed_dict = {
            self.inputs_dict['x']: samples,
            self.inputs_dict['beta']: beta,
            self.inputs_dict['net_weights']: net_weights,
            #  self.inputs_dict['scale_weight']: net_weights[0],
            #  self.inputs_dict['transl_weight']: net_weights[1],
            #  self.inputs_dict['transf_weight']: net_weights[2],
            self.inputs_dict['train_phase']: False
        }
        summary_str = sess.run(self.summary_op, feed_dict=feed_dict)

        self.writer.add_summary(summary_str, global_step=step)
        self.writer.flush()

    def clear(self):
        self.energy_dict = None
        self.run_data = None
        self.run_strings = None
        self.params['net_weights'] = None
        self.run_dir = None
        if self.summaries:
            self.run_summary_dir = None
            self.writer = None

    def existing_run(self, run_str):
        """Check if this run has been completed previously, if so skip it."""
        run_dir = os.path.join(self.runs_dir, run_str)
        run_summary_dir = os.path.join(self.run_summaries_dir, run_str)
        fig_dir = os.path.join(self.figs_dir, run_str)
        observables_dir = os.path.join(run_dir, 'observables')

        flag = False
        run_flag = False
        summary_flag = False
        fig_flag = False
        if os.path.isdir(run_dir) and os.path.isdir(observables_dir):
            io.log(f'Found existing run at: {run_dir}')
            run_flag = True
        if os.path.isdir(run_summary_dir):
            io.log(f'Found existing run at: {run_summary_dir}')
            summary_flag = True
        if os.path.isdir(fig_dir):
            io.log(f'Found existing run at: {fig_dir}')
            fig_flag = True

        if run_flag and summary_flag and fig_flag:
            flag = True

        self._existing_run = flag

        return flag

    def _get_run_str(self, **kwargs):
        """Parse parameter values and create unique string to name the dir."""
        beta = kwargs.get('beta', 5.)
        init = kwargs.get('init', None)
        eps_np = kwargs.get('eps', None)
        run_steps = kwargs.get('run_steps', 5000)
        dir_append = kwargs.get('dir_append', None)
        net_weights = kwargs.get('net_weights', NetWeights(1, 1, 1, 1, 1, 1))

        beta_str = f'{beta:.3}'.replace('.', '')
        eps_str = f'{eps_np:.3}'.replace('.', '')

        xsw = f'{net_weights.x_scale:1g}'.replace('.', '')
        xtlw = f'{net_weights.x_translation:1g}'.replace('.', '')
        xtfw = f'{net_weights.x_transformation:1g}'.replace('.', '')
        vsw = f'{net_weights.v_scale:1g}'.replace('.', '')
        vtlw = f'{net_weights.v_translation:1g}'.replace('.', '')
        vtfw = f'{net_weights.v_transformation:1g}'.replace('.', '')

        run_str = (f'steps{run_steps}'
                   f'_beta{beta_str}'
                   f'_eps{eps_str}'
                   f'_x{xsw}{xtlw}{xtfw}'
                   f'_v{vsw}{vtlw}{vtfw}')

        if init is not None:
            run_str += f'_{init}'

        if dir_append is not None:
            run_str += dir_append

        return run_str

    def reset(self, **kwargs):
        """Reset run_data and run_strings to prep for new run."""
        if self.run_data is not None or self.run_dir is not None:
            self.clear()

        beta = kwargs.get('beta', 5.)
        eps_np = kwargs.get('eps', None)
        run_steps = kwargs.get('run_steps', 5000)
        skip_existing = kwargs.get('skip_existing', False)
        net_weights = kwargs.get('net_weights',
                                 NetWeights(1., 1., 1., 1., 1., 1.))

        existing = False
        run_str = self._get_run_str(**kwargs)
        if self.existing_run(run_str):  # append current time to run_str
            existing = True
            if skip_existing:
                return existing
            now = datetime.datetime.now()
            time_str = now.strftime('%H%M')
            run_str += f'__{time_str}'

        self._set_run_dir(run_str)
        self._run_str = run_str

        self.beta = beta
        self.run_steps = int(run_steps)

        self.run_data = {}
        self.energy_dict = {}

        if self.save_lf_data:
            self.samples_arr = []

        self.run_stats = {}
        self.run_strings = []

        #  params = self.model.params
        self.params['net_weights'] = net_weights

        if self.summaries:
            self.writer = tf.summary.FileWriter(self.run_summary_dir,
                                                tf.get_default_graph())
        run_params = {
            'run_steps': self.run_steps,
            'beta': self.beta,
            'net_weights': net_weights,
            'eps': eps_np,
            'run_str': run_str,
        }

        io.save_params(self.params, self.run_dir)
        io.save_params(run_params, self.run_dir, name='run_params')

        return existing

    def _set_run_dir(self, run_str):
        """Sets dirs containing data about inference run using run_str."""
        self.run_dir = os.path.join(self.runs_dir, run_str)
        io.check_else_make_dir(self.run_dir)
        if self.summaries:
            self.run_summary_dir = os.path.join(self.run_summaries_dir,
                                                run_str)
            io.check_else_make_dir(self.run_summary_dir)

    def update(self, sess, data, data_str, net_weights):
        """Update run_data and append data_str to data_strings."""
        step = data['step']
        beta = data['beta']
        key = (step, beta)
        if step == 0:
            if self.save_lf_data:
                self.samples_arr.append(data['samples_in'])

        energies = data.pop('energies')
        for k in energies.keys():
            try:
                self.energy_dict[k].append(energies[k])
            except KeyError:
                self.energy_dict[k] = [energies[k]]

        for key, val in data.items():
            try:
                self.run_data[key].append(val)
            except KeyError:
                self.run_data[key] = [val]

        self.run_strings.append(data_str)

        log_steps = self.params.get('logging_steps', 10)
        if self.summaries and (step + 1) % log_steps == 0:
            self.log_step(sess, step, data['samples'], beta, net_weights)

        print_steps = self.params.get('print_steps', 1)
        if step % (10 * print_steps) == 0:
            io.log(data_str)

        if step % 100 == 0:
            io.log(self.run_header)

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

    def save_data(self, data, fname):
        """Save additional data to `fname` in `self.run_dir` ."""
        out_file = os.path.join(self.run_dir, fname)
        io.log(f'Saving data to {fname}...')
        with open(out_file, 'wb') as f:
            pickle.dump(data, f)

    def _calc_diffs(self, chain):
        """Calculate the difference between suscessive states in `chain`."""
        if not isinstance(chain, np.ndarray):
            chain = np.array(chain)
        return chain[1:] - chain[:1]

    def save_run_data(self, therm_frac=10, save_samples=False):
        """Save run information."""
        self._save_energy_data()
        self._write_run_history()

        if save_samples:
            self._save_samples()

        io.check_else_make_dir(self.run_dir)
        bad_keys = ['samples_in', 'samples',
                    'v_init', 'v_proposed',
                    'x_init', 'x_proposed',
                    'x_out', 'v_out', 'plaq_sums',
                    'sumlogdet_proposed', 'sumlogdet_out',
                    'dxf', 'dxb', 'eps', 'dynamics_eps', 'step', 'beta']

        observables_dir = os.path.join(self.run_dir, 'observables')
        io.check_else_make_dir(observables_dir)
        for key, val in self.run_data.items():
            if key in bad_keys and not save_samples:
                continue
            else:
                out_file = key + '.pkl'
                out_file = os.path.join(observables_dir, out_file)
                io.save_data(val, out_file, name=key)

        if self.model_type == 'GaugeModel':
            self._save_observables_data(observables_dir, therm_frac)

    def _save_samples(self):
        keys = ['x_out', 'v_out' 'x_proposed', 'v_proposed']
        samples_dict = {k: np.array(self.run_data.pop(k)) for k in keys}
        #  samples_dict = {k: np.array(self.run_data[k]) for k in keys}

        out_file = os.path.join(self.run_dir, 'run_data')
        np.savez_compressed(out_file, **samples_dict)

    def _save_energy(self, data, etype, header=None):
        """Save energy data to `.pkl` file and write stats to `.txt` file."""
        fname = etype + '.pkl'
        out_file = os.path.join(self.run_dir, fname)
        io.log(f'Saving {etype} to {out_file}...')
        with open(out_file, 'wb') as f:
            pickle.dump(data, f)

        fname_txt = etype + '.txt'
        txt_file = os.path.join(self.run_dir, fname_txt)
        with open(txt_file, 'w') as f:
            if header is not None:
                f.write(header)
                f.write('\n')
            for key, val in data.items():
                v = np.array(val)
                f.write(f'{key} (avg): {v.mean():5g} +/- {v.std():.5g}\n')
                for e in v[0][:10]:
                    f.write(f' {e:.5g} ')
                f.write('\n\n')

    def _save_energy_data(self):
        """Save energy data to `.pkl` files."""
        self._save_energy(self.energy_dict, 'energy_data_tf')
        #  self._save_energy(self.energy_dict_np, 'energy_data_np')
        #  header = (f'We compute the difference between the '
        #            'energies as alculated in tensorflow vs '
        #            'numpy as:\n   dE = E_tf - E_np\n')
        #  self._save_energy(self.energies_diffs_dict,
        #                    'energy_data_tf_np_diff', header=header)

    def _save_observables_data(self, observables_dir, therm_frac):
        """For `GaugeModel` instance, save observables data."""
        run_stats = self.calc_observables_stats(self.run_data, therm_frac)
        charges = self.run_data['charges']
        charges_arr = np.array(charges)
        #  charges_arr = np.array(list(charges.values()))
        charges_arrT = charges_arr.T
        charges_autocorrs = [autocorr(x) for x in charges_arrT]
        charges_autocorrs = [x / np.max(x) for x in charges_autocorrs]
        self.run_data['charges_autocorrs'] = charges_autocorrs

        stats_data_file = os.path.join(self.run_dir, 'run_stats.pkl')
        io.log(f"Saving run_stats to: {stats_data_file}.")
        with open(stats_data_file, 'wb') as f:
            pickle.dump(run_stats, f)

        for key, val in run_stats.items():
            out_file = key + '_stats.pkl'
            out_file = os.path.join(observables_dir, out_file)
            io.save_data(val, out_file, name=key)

        self.write_run_stats(run_stats, therm_frac)

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

        charges_arr = np.array(run_data['charges'], dtype=int)
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

    def _write_run_history(self):
        """Write the strings printed during inference to `.txt` file."""
        history_file = os.path.join(self.run_dir, 'run_history.txt')
        io.write(self.run_header, history_file, 'w')
        _ = [io.write(s, history_file, 'a') for s in self.run_strings]

    def write_run_stats(self, stats, therm_frac=10):
        """Write statistics in human readable format to .txt file."""
        therm_steps = self.run_steps // therm_frac

        out_file = os.path.join(self.run_dir, 'run_stats.txt')

        actions_avg, actions_err = stats['actions'].mean(axis=0)
        plaqs_avg, plaqs_err = stats['plaqs'].mean(axis=0)
        charges_avg, charges_err = stats['charges'].mean(axis=0)
        suscept_avg, suscept_err = stats['suscept'].mean(axis=0)

        #  ns = self.model.batch_size
        bs = self.params['batch_size']
        suscept_k1 = f'  \navg. over all {bs} samples < Q >'
        suscept_k2 = f'  \navg. over all {bs} samples < Q^2 >'
        actions_k1 = f'  \navg. over all {bs} samples < action >'
        plaqs_k1 = f'  \n avg. over all {bs} samples < plaq >'

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

        keys = [f"sample {idx}" for idx in range(bs)]

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

        run_str = (f" stats for {bs} chains ran for {self.run_steps} steps "
                   f" at beta = {self.beta}.")

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
