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
from plotters.data_utils import bootstrap, therm_arr
from plotters.inference_plots import calc_tunneling_rate

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
    def __init__(self, run_params):
        """Initialization method.
        Args:
            run_params (dict): Parameters to use for inference run.
        NOTE: The containers `RUN_DATA, SAMPLES, ENERGY_DATA, ...` are defined
        in `runners/__init__.py`
        """
        self.run_params = run_params

        self.samples_arr = []
        self.data_strs = [HSTR]
        self.observables = OBSERVABLES
        self.run_data = RUN_DATA
        self.samples_dict = SAMPLES
        self.energy_data = ENERGY_DATA
        self.reverse_data = REVERSE_DATA
        self.volume_diffs = VOLUME_DIFFS

    @staticmethod
    def _update(main_dict, new_dict, iter_main=False):
        """Update `main_dict` with new values in `new_dict`."""
        items = main_dict.items() if iter_main else new_dict.items()
        for key, val in items:
            try:
                main_dict[key].append(val)
            except KeyError:
                main_dict[key] = [val]

        return main_dict

    def _multiple_updates(self, dicts_tuple):
        for (dm, dn, iter_main) in dicts_tuple:
            dm = self._update(dm, dn, iter_main)

        return dicts_tuple

    def _update_samples(self, outputs):
        for key, val in self.samples_dict.items():
            if key in outputs:
                val.append(outputs[key])

    def update(self, step, samples, outputs):
        """"Update all data."""
        if step % self.run_params['print_steps'] == 0:
            io.log(outputs['data_str'])
            self.data_strs.append(outputs['data_str'])

        self.samples_arr.append(samples)

        for key, val in outputs['observables'].items():
            try:
                self.observables[key].append(val)
            except KeyError:
                self.observables[key] = [val]

        for key, val in outputs['energy_data'].items():
            try:
                self.energy_data[key].append(val)
            except KeyError:
                self.energy_data[key] = [val]

        for key, val in outputs['dynamics_output'].items():
            try:
                self.run_data[key].append(val)
            except KeyError:
                self.run_data[key] = [val]

        if 'volume_diffs' in outputs:
            for key, val in outputs['volume_diffs'].items():
                try:
                    self.volume_diffs[key].append(val)
                except KeyError:
                    self.volume_diffs[key] = [val]

    def update1(self, step, samples, outputs):
        """Update all data."""
        if step % self.run_params['print_steps'] == 0:
            io.log(outputs['data_str'])
            self.data_strs.append(outputs['data_str'])

        self.samples_arr.append(samples)

        tups = [(self.run_data, outputs['observables'], False),
                (self.energy_data, outputs['energy_data'], False),
                (self.reverse_data, outputs['reverse_data'], False)]
                #  (self.samples_dict, outputs['dynamics_output'], True)]

        if 'volume_diffs' in outputs:
            tups.append((self.volume_diffs, outputs['volume_diffs'], False))

        tups = self._multiple_updates(tups)

        self._update_samples(outputs['dynamics_output'])

        self.run_data['sumlogdet_out'].append(
            outputs['dynamics_output']['sumlogdet_out']
        )
        self.run_data['sumlogdet_proposed'].append(
            outputs['dynamics_output']['sumlogdet_proposed']
        )


    def build_dataset(self):
        """Build `xarray.Dataset` from `self.run_data`."""
        charges = np.array(self.observables['charges']).T
        self.run_data['tunneling_rate'] = calc_tunneling_rate(charges).T

        #  plaqs = np.array(self.observables.pop('plaqs'))
        #  beta = self.run_params['beta']
        #  self.run_data['plaqs_diffs'] = calc_plaqs_diffs(plaqs, beta)
        plot_data = {
            'plaqs_diffs': self.observables['plaqs_diffs'],
            'charges': self.observables['charges'],
            'accept_prob': self.run_data['accept_prob'],
            'xdiff_r': np.array(self.run_data['xdiff_r']).mean(axis=-1),
            'vdiff_r': np.array(self.run_data['vdiff_r']).mean(axis=-1),
            'sumlogdet_out': self.run_data['sumlogdet_out'],
            'sumlogdet_prop': self.run_data['sumlogdet_proposed'],
            'tunneling_rate': self.run_data['tunneling_rate'],
            #  'dplaqs': self.observables['dplaqs'],
            'dcharges': self.observables['dcharges'],
        }

        try:
            dataset = self._build_dataset(plot_data, filter_str='forward')
        except:
            import pudb; pudb.set_trace()
        #  dataset = self._build_dataset(self.run_data, filter_str='forward')

        return dataset

    @staticmethod
    def _build_dataset(data, filter_str=None):
        """Build `xarray.Dataset` from `data`."""
        _dict = {}
        for key, val in data.items():
            cond1 = (filter_str is not None and filter_str in key)
            cond2 = (val == [])
            if cond1 or cond2:
                continue
            arr, steps = therm_arr(np.array(val))
            arr = arr.T
            _dict[key] = xr.DataArray(arr, dims=['chain', 'draw'],
                                      coords=[np.arange(arr.shape[0]), steps])

        dataset = xr.Dataset(_dict)

        return dataset

    def build_energy_dataset(self):
        """Build `xarray.Dataset` from `self.energy_data`."""
        return self._build_dataset(self.energy_data)

    def build_energy_diffs_dataset(self):
        """Build `xarray.Dataset` from energy differences."""
        def ediff(k1, k2):
            etype, k1_ = k1.split('_')
            _, k2_ = k2.split('_')
            key = f'd{etype}_{k1_}_{k2_}'
            diff = (np.array(self.energy_data[k1])
                    - np.array(self.energy_data[k2]))
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

        #  key = [('potential_proposed', 'potential_init'),
        #          ('potential_out', 'potential_init'),
        #          ('potential_out', 'potential_proposed'),
        #          ('kinetic_proposed', 'kinetic_init'),
        #          ('kinetic_out', 'kinetic_init'),
        #          ('kinetic_out', 'kinetic_proposed'),
        #          ('hamiltonian_proposed', 'hamiltonian_init'),
        #          ('hamiltonian_out', 'hamiltonian_init'),
        #          ('hamiltonian_out', 'hamiltonian_proposed')]

        return dataset

    def build_energy_transition_dataset(self):
        """Build `energy_transition_dataset` from `self.energy_data`."""
        _dict = {}
        for key, val in self.energy_data.items():
            arr, steps = therm_arr(np.array(val))
            arr -= np.mean(arr, axis=0)
            arr = arr.T
            key_ = f'{key}_minus_avg'
            _dict[key_] = xr.DataArray(arr, dims=['chain', 'draw'],
                                       coords=[np.arange(arr.shape[0]), steps])
        dataset = xr.Dataset(_dict)

        return dataset

    def save_direction_data(self):
        """Save directionality data."""
        forward_arr = self.run_data.get('forward', None)
        if forward_arr is None:
            io.log(f'`run_data` has no `forward` item. Returning.')
            return

        forward_arr = np.array(forward_arr)
        num_steps = len(forward_arr)
        steps_f = forward_arr.sum()
        steps_b = num_steps - steps_f
        percent_f = steps_f / num_steps
        percent_b = steps_b / num_steps

        direction_file = os.path.join(self.run_params['run_dir'],
                                      'direction_results.txt')
        with open(direction_file, 'w') as f:
            f.write(f'forward steps: {steps_f}/{num_steps}, {percent_f}\n')
            f.write(f'backward steps: {steps_b}/{num_steps}, {percent_b}\n')

    def save_samples_data(self, run_dir=None):
        """Save all samples to `run_dir/samples`."""
        if run_dir is None:
            run_dir = self.run_params['run_dir']

        out_dir = os.path.join(run_dir, 'samples')
        io.check_else_make_dir(out_dir)
        for key, val in self.samples_dict.items():
            out_file = os.path.join(out_dir, f'{key}.pkl')
            io.save_pkl(np.array(val), out_file, name=key)

        samples_arr_file = os.path.join(run_dir, 'samples.pkl')
        io.save_pkl(self.samples_arr, samples_arr_file)

    def save_reverse_data(self, run_dir):
        """Save reversibility data."""
        reverse_data = {
            'xdiff_r': self.run_data['xdiff_r'],
            'vdiff_r': self.run_data['vdiff_r'],
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
        run_data_file = os.path.join(run_dir, 'run_data.pkl')
        energy_data_file = os.path.join(run_dir, 'energy_data.pkl')
        volume_diffs_file = os.path.join(run_dir, 'volume_diffs.pkl')
        observables_file = os.path.join(run_dir, 'observables.pkl')

        io.save_pkl(self.run_data, run_data_file, name='run_data')
        io.save_pkl(self.energy_data, energy_data_file, name='energy_data')
        io.save_pkl(self.volume_diffs, volume_diffs_file, name='volume_diffs')
        io.save_pkl(self.observables, observables_file, 'observables')

        observables_dir = os.path.join(run_dir, 'observables')
        io.check_else_make_dir(observables_dir)
        iters = zip(self.run_data.items(), self.observables.items())
        for (kr, vr), (ko, vo) in iters:
            fr = os.path.join(observables_dir, f'{kr}.pkl')
            fo = os.path.join(observables_dir, f'{ko}.pkl')
            io.save_pkl(np.array(vr), fr, name=kr)
            io.save_pkl(np.array(vo), fo, name=ko)

    def save(self, run_dir=None):
        """Save all inference data to `run_dir`."""
        if run_dir is None:
            run_dir = self.run_params['run_dir']

        io.save_dict(self.run_params, run_dir, name='run_params')
        volume_diffs_file = os.path.join(run_dir, 'volume_diffs.pkl')
        io.save_pkl(self.volume_diffs, volume_diffs_file)

        if 'forward' in self.run_data:
            self.save_direction_data()

        self.save_samples_data(run_dir)
        self.save_reverse_data(run_dir)
        self.save_data(run_dir)
        self.save_run_history(run_dir)

    @staticmethod
    def _calc_stats(arr, n_boot=100):
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
        tunn_stats = {
            'tunneling_events': tunneling_events,
            'tunneling_rate': tunneling_rate,
        }
        #  return tunneling_events, tunneling_rate
        return tunn_stats

    def thermalize_data(self, data=None):
        """Returns thermalized versions of entries in data."""
        if data is None:
            data = self.run_data

        therm_data = {}
        for key, val in data.items():
            therm_data[key] = therm_arr(np.array(val), ret_steps=False)

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
            out_dir = self.run_params.get('run_dir', None)
            out_file = os.path.join(out_dir, 'run_summary.txt')

        io.log(f'Writing run summary statistics to {out_file}.\n')
        nw_str = f"NET_WEIGHTS: {self.run_params['net_weights']}"
        nw_uline = uline(nw_str, c='=')
        io.log_and_write(nw_str, out_file)
        io.log_and_write(nw_uline, out_file)
        data = {
            'accept_prob': self.run_data['accept_prob'],
            'charges': self.observables['charges'],
            'plaqs_diffs': self.observables['plaqs_diffs'],
            'dplaqs': self.observables['dplaqs'],
            #  'dx_out': self.run_data['dx_out'],
        }
        therm_data = self.thermalize_data(data)
        tunn_stats = self.calc_tunneling_stats(therm_data['charges'])
        #  fnames = ['accept_prob', 'charges', 'dx_out']
        #  therm_data, tunn_stats = self._load_format_data(fnames)
        self._log_stats(therm_data, tunn_stats, out_file, n_boot=n_boot)
        self._log_run_params(out_file)

        io.log_and_write(120 * '=' + '\n', out_file)

        return therm_data, tunn_stats
