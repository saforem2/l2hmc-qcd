"""
inference_summarizer.py

Implements InferenceSummarizer class, which records multiple metrics that
indicate how well the trained sampler performs.

Author: Sam Foreman
Date: Mar 02, 2020
"""
import os
import numpy as np

import utils.file_io as io
from lattice.lattice import u1_plaq_exact
from plotters.data_utils import therm_arr, bootstrap


PRINT_OPTIONS = {
    'precision': 6,
    'linewidth': 120,
    'edgeitems': 10,
    'suppress': True,
    'floatmode': 'fixed',
    'sign': ' ',
}

np.set_printoptions(**PRINT_OPTIONS)


def uline(s, c='-'):
    """Returns a string of '-' with the same length as `s`."""
    return len(s) * '-'


class InferenceSummarizer:
    """"
    Implements functionality to write human-readable summary of inference run.

    Specifically, it prints/writes a human-readable summary of the metrics that
    are used for measuring the 'quality' of an inference run.

    Through the `log_summary` method, it performs bootstrap resampling to
    compute statistics of the plaquette difference, acceptance probability, and
    average distance traveled between subsequent updates of the sampler.
    """
    def __init__(self, run_dir, run_params=None):
        self._run_dir = run_dir
        if run_params is None:
            rp_file = os.path.join(run_dir, 'run_params.pkl')
            self._run_params = io.loadz(rp_file)
        self._beta = self._run_params.get('beta', None)
        self._run_steps = self._run_params.get('run_steps', None)
        self._batch_size = self._run_params.get('batch_size', None)

    def _load_observable(self, fname):
        """Load observable data data from `self._run_dir/observables/fname."""
        obs_dir = os.path.join(self._run_dir, 'observables')
        data = io.loadz(os.path.join(obs_dir, f'{fname}.pkl'))

        return np.array(data)

    def load_observables(self, fnames, run_data=None):
        """Load all observables in `fnames`."""
        if run_data is None:
            run_data = {}

        for fname in fnames:
            run_data[fname] = self._load_observable(fname)

        return run_data

    def _load_format_data(self, fnames):
        plaq_exact = u1_plaq_exact(self._beta)
        run_data = {
            'plaqs_diffs': plaq_exact - self._load_observable('plaqs'),
        }
        run_data = self.load_observables(fnames, run_data)
        therm_data = {
            k: therm_arr(v, ret_steps=False) for k, v in run_data.items()
        }

        events, rates = self.calc_tunneling_stats(therm_data['charges'])
        tunn_stats = {
            'tunneling_rates': rates,
            'tunneling_events': events,
        }

        return therm_data, tunn_stats

    @staticmethod
    def _calc_stats(arr, n_boot=10000):
        step_ax = np.argmax(arr.shape)
        chain_ax = np.argmin(arr.shape)
        arr = np.swapaxes(arr, step_ax, chain_ax)
        stds = []
        means = []
        for chain in arr:
            mean, std, _ = bootstrap(chain, n_boot=n_boot, ci=68)
            means.append(mean)
            stds.append(std)

        return np.array(means), np.array(stds)

    @staticmethod
    def calc_tunneling_stats(charges):
        """Calculate tunneling statistics from `charges`.

        Explicitly, calculate the `tunneling events` as the number of accepted
        configurations which produced a configuration with a new topological
        charge value.

        This is calculated by looking at how the topological charges changes
        between successive steps, i.e.

        ```
        charges_diff = charges[1:] - charges[:-1]
        tunneling_events = np.sum(charges_diff, axis=step_ax)
        ```

        Since we are running multiple chains in parallel, we are interested in
        the tunneling statistics for each of the individual chains.

        The `tunneling_rate` is then calculated as the total number of
        `tunneling_events` / num_steps`.
        """
        step_ax = np.argmax(charges.shape)
        num_steps = charges.shape[step_ax]
        charges_diff = np.abs(charges[1:] - charges[:-1])
        tunneling_events = np.sum(np.around(charges_diff), axis=step_ax)
        tunneling_rate = tunneling_events / num_steps

        return tunneling_events, tunneling_rate

    @staticmethod
    def _log_write_stat(out_file, key, val, std=None):
        """Log (print) and write stats about (key, val) pair to `out_file`.

        Args:
            out_file (str): Path to file where results should be written.
            key (str): Name of `val`.
            val (np.ndarray): Array containing an observable..
            std (np.ndarray): Array of (bootstrap) resampled standard
                deviations.
        """
        def log(s):
            io.log_and_write(s, out_file)

        key_str = f"< {key} > = {np.mean(val):.6g}"
        sep = uline(key_str)
        val_str = f"    {val}"
        std_str = ''
        if std is not None:
            key_std_str = f" +/- {np.mean(std):.6g}"
            key_str += key_std_str
            sep += uline(key_std_str)
            std_str = f" +/- {std}"

        log(key_str)
        log(sep)
        log(val_str)
        log(std_str)


    def _log_run_params(self, out_file):
        """Log/write `self._run_params` to `out_file`."""
        rp_str = 'run_params:'
        sep = uline(rp_str)
        io.log_and_write(rp_str, out_file)
        io.log_and_write(sep, out_file)
        keys = ['init', 'run_steps', 'batch_size',
                'beta', 'eps', 'num_steps', 'direction',
                'zero_masks', 'mix_samplers', 'net_weights', 'run_dir']
        for key in keys:
            if key in self._run_params:
                param_str = f' - {key}: {self._run_params[key]}\n'
                io.log_and_write(param_str, out_file)

    def _log_stats(self, therm_data, tunn_stats, out_file, n_boot=10000):
        """Log/write all stats in `therm_data` and `tunn_stats`."""
        for key, val in tunn_stats.items():
            self._log_write_stat(out_file, key, val)

        for key, val in therm_data.items():
            means, stds = self._calc_stats(val, n_boot=n_boot)
            self._log_write_stat(out_file, key, means, std=stds)
            io.log_and_write('\n', out_file)

    def log_summary(self, n_boot=10000, out_file=None):
        """Create human-readable summary of inference run."""
        if out_file is None:
            out_file = os.path.join(self._run_dir, 'run_summary.txt')
        io.log(f'Writing run summary statistics to {out_file}.\n')
        nw_str = f"NET_WEIGHTS: {self._run_params['net_weights']}"
        nw_uline = uline(nw_str, c='=')
        io.log_and_write(nw_str, out_file)
        io.log_and_write(nw_uline, out_file)
        fnames = ['accept_prob', 'charges', 'dx_out']
        therm_data, tunn_stats = self._load_format_data(fnames)
        self._log_stats(therm_data, tunn_stats, out_file, n_boot=n_boot)
        self._log_run_params(out_file)

        io.log_and_write(120 * '=' + '\n', out_file)

        return therm_data, tunn_stats
