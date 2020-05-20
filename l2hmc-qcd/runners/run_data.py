"""
run_data.py
Implements `RunData` object that provides a simplified interface for dealing
with the data generated during inference.
Author: Sam Foreman
Date: 03/30/2020
"""
import os

import xarray as xr
import pandas as pd
import autograd.numpy as np

import utils.file_io as io

from runners import (ENERGY_DATA, REVERSE_DATA, RUN_DATA,
                     SAMPLES, VOLUME_DIFFS, OBSERVABLES, HSTR)
from plotters.data_utils import bootstrap

#  from lattice.lattice import calc_plaqs_diffs

# pylint:disable=no-member,invalid-name


def uline(s, c='-'):
    """Returns a string of '-' with the same length as `s` (for underline)."""
    return len(s) * c


def strf(x):
    """Format the number x as a string."""
    if np.allclose(x - np.around(x), 0):
        xstr = f'{int(x)}'
    else:
        xstr = f'{x:.1}'.replace('.', '')
    return xstr


class RunData:
    """Object containing inference data."""
    def __init__(self, run_config):
        """Initialization method.
        Args:
            run_params (dict): Parameters to use for inference run.
        NOTE: The containers `RUN_DATA, SAMPLES, ENERGY_DATA, ...` are defined
        in `runners/__init__.py`
        """
        self.config = run_config
        self.data_strs = [HSTR]
        self.samples_arr = []
        self.data = {}

    def _update_from_key(self, key, outputs):
        try:
            self.data[key].append(outputs[key])
        except KeyError:
            self.data[key] = [outputs[key]]

    def _update_from_keys(self, keys, outputs):
        for key in keys:
            self._update_from_key(key, outputs)

    def _update_from_attr(self, key, outputs):
        try:
            self.data[key].append(getattr(outputs, key))
        except KeyError:
            self.data[key] = [getattr(outputs, key)]

    def _try_update(self, key, val):
        try:
            self.data[key].append(val)
        except KeyError:
            self.data[key] = [val]

    def _try_updates(self, outputs):
        self._try_update('sumlogdet_out', outputs['sld_states'].out)
        self._try_update('sumlogdet_proposed', outputs['sld_states'].proposed)
        self._try_update('xdiff_r', outputs['state_diff_r'].x)
        self._try_update('vdiff_r', outputs['state_diff_r'].v)
        self._try_update('exp_energy_diff', outputs['exp_energy_diff'])
        self._try_update('plaqs', outputs['mc_observables'].out.plaqs)
        self._try_update('charges', outputs['mc_observables'].out.charges)
        self._try_update('charges_proposed',
                         outputs['mc_observables'].proposed.charges)

    def _update_energies(self, outputs):
        self._try_update('potential_init',
                         outputs['mc_energies'].init.potential)
        self._try_update('potential_proposed',
                         outputs['mc_energies'].proposed.potential)
        self._try_update('potential_out',
                         outputs['mc_energies'].out.potential)

        self._try_update('kinetic_init',
                         outputs['mc_energies'].init.kinetic)
        self._try_update('kinetic_proposed',
                         outputs['mc_energies'].proposed.kinetic)
        self._try_update('kinetic_out',
                         outputs['mc_energies'].out.kinetic)

        self._try_update('hamiltonian_init',
                         outputs['mc_energies'].init.hamiltonian)
        self._try_update('hamiltonian_proposed',
                         outputs['mc_energies'].proposed.hamiltonian)
        self._try_update('hamiltonian_out',
                         outputs['mc_energies'].out.hamiltonian)

    def update(self, step, x, outputs):
        """Update all data object and print data summary."""
        if step % self.config.print_steps == 0:
            io.log(outputs['data_str'])
            self.data_strs.append(outputs['data_str'])

        self.samples_arr.append(x)
        for key, val in outputs.items():
            try:
                self.data[key].append(val)
            except KeyError:
                self.data[key] = [val]

    @staticmethod
    def therm_arr(arr, therm_frac=0.33):
        """Drop the first `therm_frac` percent of `arr` to account for mixing.

        Args:
            arr (array-like): Input array.
            therm_frac (float): Percent of total data to drop for
                thermalization. For example, if `therm_frac = 0.25`, the first
                25% of `arr` will be excluded from the returned array.

        Returns:
            arr_therm (array-like): Thermalized array.
            steps_arr (array-like): The accompanying `steps_arr` containing the
                updated steps index.


        Example:
            >>> arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
            >>> t_arr, steps = therm_arr(arr, therm_frac=0.25)
            >>> t_arr
            array([3, 4, 5, 6, 7, 8])
            >>> steps
            array([2, 3, 4, 5, 6, 7])
        """
        num_steps = arr.shape[0]
        therm_steps = int(therm_frac * num_steps)
        arr = arr[therm_steps:]
        steps = np.arange(therm_steps, num_steps)

        return arr, steps

    def build_dataset(self):
        """Build `xarray.Dataset` from `self.run_data`."""
        #  charges = np.array(self.data['charges'])
        #  self.data['tunneling_rate'] = self.calc_tunneling_rate(charges).T

        #  plaqs = np.array(self.observables.pop('plaqs'))
        #  beta = self.run_params['beta']
        #  self.run_data['plaqs_diffs'] = calc_plaqs_diffs(plaqs, beta)
        plot_data = {
            'plaqs_err': self.data['plaqs_diffs'],
            'charges': self.data['charges'],
            'accept_prob': self.data['accept_prob'],
            'xdiff_r': np.array(self.data['xdiff_r']).mean(axis=-1),
            'vdiff_r': np.array(self.data['vdiff_r']).mean(axis=-1),
            'sumlogdet_out': self.data['sumlogdet_out'],
            'sumlogdet_prop': self.data['sumlogdet_proposed'],
            #  'tunneling_rate': self.run_data['tunneling_rate'],
            'dplaqs': np.array(self.data['plaq_change']).mean(axis=(-1, -2)),
            'dq': self.data['charge_change'],
        }

        # ignore plot_data['forward'] data
        dataset = self._build_dataset(plot_data, filter_str='forward')

        return dataset

    def _build_dataset(self, data, filter_str=None, therm_frac=0.33):
        """Build (thermalized) `xarray.Dataset` from `data`.

        Args:
            filter_str (str): String that is used to exclude (key, value) pair
                from data e.g. if `data = {'x': 1, 'y': 2}`, and
                `filter_str='x'`, the resulting `dataset` will NOT contain
                `data['x']`.
            therm_frac (float): Percent of data to throw out to account for
                thermalization effects. For example, if `therm_frac = 0.25`,
                the first 25% of `data` will be excluded from `dataset`.

        Returns:
            dataset (xr.Dataset): Dataset composed of thermalized `data`.
        """
        _dict = {}
        therm_data = {}
        for key, val in data.items():
            cond1 = (filter_str is not None and filter_str in key)
            cond2 = (val == [])
            if cond1 or cond2:
                continue
            arr, steps = self.therm_arr(np.array(val), therm_frac=therm_frac)
            therm_data[key] = arr
            arr = arr.T
            chains = np.arange(arr.shape[0])
            _dict[key] = xr.DataArray(arr, dims=['chain', 'draw'],
                                      coords=[chains, steps])

        dataset = xr.Dataset(_dict)

        return dataset

    def build_energy_dataset(self):
        """Build `xarray.Dataset` from `self.energy_data`."""
        names = ['potential', 'kinetic', 'hamiltonian']
        types = ['init', 'proposed', 'out']
        ekeys = [f'{k1}_{k2}' for k1 in names for k2 in types]
        edata = {
            key: self.data[key] for key in ekeys
        }

        return self._build_dataset(edata)

    def build_energy_diffs_dataset(self):
        """Build `xarray.Dataset` from energy differences."""
        def ediff(k1, k2):
            etype, k1_ = k1.split('_')
            _, k2_ = k2.split('_')
            key = f'd{etype}_{k1_}_{k2_}'
            diff = (np.array(self.data[k1])
                    - np.array(self.data[k2]))
            return (key, diff)

        # Create array of pairs
        # [('potential_proposed', 'potential_init'), ...], etc.
        h = ['potential', 'kinetic', 'hamiltonian']
        t = [('proposed', 'init'), ('out', 'init'), ('out', 'proposed')]
        keys = [(f'{s[0]}_{s[1][0]}', f'{s[0]}_{s[1][1]}') for s in zip(h, t)]

        data = {}
        for pair in keys:
            key, diff = ediff(pair[0], pair[1])
            data[key] = diff

        dataset = self._build_dataset(data)

        return dataset

    def build_energy_transition_dataset(self):
        """Build `energy_transition_dataset` from `self.energy_data`."""
        _dict = {}
        for key, val in self.energy_data.items():
            arr, steps = self.therm_arr(np.array(val))
            arr -= np.mean(arr, axis=0)
            arr = arr.T
            key_ = f'{key}_minus_avg'
            _dict[key_] = xr.DataArray(arr, dims=['chain', 'draw'],
                                       coords=[np.arange(arr.shape[0]), steps])
        dataset = xr.Dataset(_dict)

        return dataset

    def save_direction_data(self):
        """Save directionality data."""
        #  forward_arr = self.run_data.get('forward', None)
        #  if forward_arr is None:
        #      io.log(f'`run_data` has no `forward` item. Returning.')
        #      return
        #  forward_arr = np.array(forward_arr)
        #  num_steps = len(forward_arr)
        #  steps_f = forward_arr.sum()
        #  steps_b = num_steps - steps_f
        #  percent_f = steps_f / num_steps
        #  percent_b = steps_b / num_steps
        #
        #  direction_file = os.path.join(self.run_params['run_dir'],
        #                                'direction_results.txt')
        #  with open(direction_file, 'w') as f:
        #      f.write(f'forward steps: {steps_f}/{num_steps}, {percent_f}\n')
        #      f.write(f'backward steps: {steps_b}/{num_steps}, {percent_b}\n')
        pass

    def save_samples_data(self, run_dir):
        """Save all samples to `run_dir/samples`."""
        #  if run_dir is None:
        #      run_dir = self.run_params['run_dir']

        #  out_dir = os.path.join(run_dir, 'samples')
        #  io.check_else_make_dir(out_dir)
        #  io.savez(self.samples_arr, os.path.join(run_dir, 'x_out.z'),
        #           name='output_samples')
        #  for key, val in self.samples_dict.items():
        #      out_file = os.path.join(out_dir, f'{key}.z')
        #      io.savez(np.array(val), out_file, name=key)

        samples_arr_file = os.path.join(run_dir, 'samples.z')
        io.savez(self.samples_arr, samples_arr_file)

    def save_reverse_data(self, run_dir):
        """Save reversibility data."""
        reverse_data = {
            'xdiff_r': self.data['xdiff_r'],
            'vdiff_r': self.data['vdiff_r'],
        }
        max_rdata = {}
        for key, val in reverse_data.items():
            max_rdata[key] = [np.max(val)]
        max_rdata_df = pd.DataFrame(max_rdata)
        out_file = os.path.join(run_dir, 'max_reversibility_results.csv')
        io.log(f'Saving `max` reversibility data to {out_file}.')
        max_rdata_df.to_csv(out_file)

    def save_run_history(self, run_dir):
        """Save run_history to `.txt` file."""
        run_history_file = os.path.join(run_dir, 'run_history.txt')
        io.log(f'Writing run history to: {run_history_file}...')
        with open(run_history_file, 'w') as f:
            for s in self.data_strs:
                f.write(f'{s}\n')

    def save_data(self, run_dir):
        """Save all `data` objects to `run_dir`."""
        #  data_file = os.path.join(run_dir, 'data.z')
        for key, val in self.data.items():
            if isinstance(val, list):
                val = np.array(val)

            fpath = os.path.join(run_dir, f'{key}.z')
            io.savez(val, fpath, name=key)

    def save(self, run_dir):
        """Save all inference data to `run_dir`."""
        self.save_samples_data(run_dir)
        self.save_reverse_data(run_dir)
        self.save_data(run_dir)
        self.save_run_history(run_dir)

    @staticmethod
    def _calc_stats(arr, n_boot=10):
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
    def calc_tunneling_rate(charges):
        """Calc. the tunneling rate as the charge difference per step."""
        # insert copy of first row at beginning of charges
        charges = np.insert(charges, 0, charges[0], axis=0)
        dq = np.abs(np.around(charges[1:]) - np.around(charges[:-1]))
        #  dq = np.floor(np.abs(charges[1:] - charges[:-1]) + 0.5)
        tunneling_rate = dq / charges.shape[0]  # divide by num steps

        return tunneling_rate

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
        if not isinstance(charges, np.ndarray):
            charges = np.array(charges)
        step_ax = 0  # data is appended for each step along axis 0
        num_steps = charges.shape[step_ax]
        charges = np.insert(charges, 0, charges[0], axis=step_ax)
        dq = np.abs(np.around(charges[1:]) - np.around(charges[:-1]))
        #  dq = np.floor(np.abs(charges[1:] - charges[:-1]) + 0.5)
        tunneling_events = np.sum(dq, axis=step_ax)

        # sum the step-wise charge differences over the step axis
        # and divide by the number of steps to get the `tunneling_rate`
        tunn_stats = {
            'tunneling_events': tunneling_events,
            'tunneling_rate': tunneling_events / num_steps,
        }
        return tunn_stats

    def thermalize_data(self, data=None, therm_frac=0.33):
        """Returns thermalized versions of entries in data."""
        if data is None:
            data = self.data

        therm_data = {}
        for key, val in data.items():
            arr, _ = self.therm_arr(np.array(val), therm_frac=therm_frac)
            therm_data[key] = arr

        return therm_data

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
        """Log/write `self.run_params` to `out_file`."""
        rp_str = 'run_params:'
        sep = uline(rp_str)
        io.log_and_write(rp_str, out_file)
        io.log_and_write(sep, out_file)
        keys = ['init', 'run_steps', 'batch_size',
                'beta', 'eps', 'num_steps', 'direction',
                'zero_masks', 'mix_samplers', 'net_weights', 'run_dir']
        for key in keys:
            if key in self.run_params:
                param_str = f' - {key}: {self.run_params[key]}\n'
                io.log_and_write(param_str, out_file)

    def _log_stats(self, therm_data, tunn_stats, out_file, n_boot=100):
        """Log/write all stats in `therm_data` and `tunn_stats`."""
        for key, val in tunn_stats.items():
            self._log_write_stat(out_file, key, val)

        for key, val in therm_data.items():
            means, stds = self._calc_stats(val, n_boot=n_boot)
            self._log_write_stat(out_file, key, means, std=stds)
            io.log_and_write('\n', out_file)

    def log_summary(self, out_file, n_boot=10):
        """Create human-readable summary of inference run."""
        #  if out_file is None:
        #      out_dir = self.run_params.get('run_dir', None)
        #      out_file = os.path.join(out_dir, 'run_summary.txt')

        io.log(f'Writing run summary statistics to {out_file}.\n')
        #  nw_str = f"NET_WEIGHTS: {self.run_params['net_weights']}"
        #  nw_uline = uline(nw_str, c='-')
        #  io.log_and_write(nw_str, out_file)
        #  io.log_and_write(nw_uline, out_file)
        data = {
            'accept_prob': self.data['accept_prob'],
            'charges': self.data['charges'],
            'plaqs_diffs': self.data['plaqs_diffs'],
            'dplaqs': self.data['plaq_change'],
            #  'dx_out': self.run_data['dx_out'],
        }
        therm_data = self.thermalize_data(data)
        tunn_stats = self.calc_tunneling_stats(therm_data['charges'])
        #  fnames = ['accept_prob', 'charges', 'dx_out']
        #  therm_data, tunn_stats = self._load_format_data(fnames)
        #  self._log_run_params(out_file)
        io.log_and_write(80*'-' + '\n\n', out_file)
        self._log_stats(therm_data, tunn_stats, out_file, n_boot=n_boot)

        io.log_and_write(120 * '=' + '\n', out_file)

        return therm_data, tunn_stats
