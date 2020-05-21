"""
runner.py

Includes the following:

    - Implementation of the `EnergyRunner` class responsible for running the
    tensorflow operations associated with calculating various energies at
    different points in the trajectory.

    - Implementation of the `Runner` class which is responsible for running the
    operations needed for inference.

Author: Sam Foreman (github: @saforem2)
Date: 04/09/2019
"""
import os
import sys
import time
import shlex
import pickle
import shutil
import argparse

import numpy as np
import xarray as xr
import seaborn as sns
import tensorflow as tf
import matplotlib.style as mplstyle

import utils.file_io as io

from config import (Energy, NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC, NetWeights,
                    State)
from inference.utils import set_eps
from loggers.run_logger import RunLogger
from utils.distributions import GMM
from lattice.utils import actions
from lattice.lattice import calc_plaqs_diffs, u1_plaq_exact
from plotters.data_utils import bootstrap, therm_arr
from plotters.inference_plots import traceplot_posterior

mplstyle.use('fast')
sns.set_palette('bright')


def _get_eps():
    eps = [i for i in tf.global_variables() if 'eps' in i.name][0]
    return eps


def uline(s, c='-'):
    """Returns a string of '-' with the same length as `s` (for underline)."""
    return len(s) * c


def get_run_ops():
    keys = ['x_init', 'v_init', 'x_proposed', 'v_proposed',
            'x_out', 'v_out', 'dx_out', 'dx_proposed', 'exp_energy_diff',
            'accept_prob', 'sumlogdet_proposed', 'sumlogdet_out']

    ops = tf.get_collection('run_ops')

    run_ops_dict = dict(zip(keys, ops))
    eps = _get_eps()
    run_ops_dict.update({'dynamics_eps': eps})

    return run_ops_dict


def get_obs_ops():
    """Build dictionary of tensorflow ops for calculating observables."""
    keys = ['plaq_sums', 'actions', 'plaqs',
            'charges', 'avg_plaqs', 'avg_actions']
    ops = tf.get_collection('observables')
    obs_ops_dict = dict(zip(keys, ops))

    return obs_ops_dict


def get_inputs():
    """Build dictionary of tensorflow placeholders used as inputs."""
    inputs = tf.get_collection('inputs')
    x, beta, eps_ph, global_step_ph, train_phase, *nw = inputs
    net_weights = NetWeights(*nw)
    inputs_dict = {
        'x': x,
        'beta': beta,
        'eps_ph': eps_ph,
        'global_step_ph': global_step_ph,
        'train_phase': train_phase,
        'net_weights': net_weights,
    }

    return inputs_dict


def find_params(log_dir=None):
    """Try and find `params` file."""
    if log_dir is None:
        log_dir = os.getcwd()

    fnames = ['params.z', 'parameters.z', 'params.pkl', 'parameters.pkl']
    for fname in fnames:
        params_file = os.path.join(log_dir, fname)
        if os.path.isfile(params_file):
            return io.loadz(params_file)

    raise FileNotFoundError('Unable to locate `params` file.')


# pylint:disable=invalid-name
class RunData:
    """Container object for holding / dealing with data from inference run."""
    def __init__(self, run_params, save_samples=False):
        self._save_samples = save_samples
        self.run_params = run_params
        self.print_steps = run_params['print_steps']

        self.data = {}
        self.data_strs = []
        names = ['step', 'dt', 'px', 'sumlogdet', 'exp(dH)', 'dQ', 'plaq_err']
        hstr = ''.join(["{:^12s}".format(name) for name in names])
        sep = '-' * len(hstr)
        self.header = sep + '\n' + hstr + '\n' + sep

    def update(self, step, new_data, data_str):
        """Update `self.data` with `new_data` and aggregate `data_str`."""
        for key, val in new_data.items():
            if key == 'x_out' and not self._save_samples:
                continue
            try:
                self.data[key].append(val)
            except KeyError:
                self.data[key] = [val]

        self.data_strs.append(data_str)
        if step % self.print_steps == 0:
            io.log(data_str)

        if step % 1000 == 0:
            io.log(self.header)

    @staticmethod
    def _build_dataset(data, therm_frac=0.3, filter_str=None, num_chains=None):
        """Build a(n) `xr.Dataset` from `data`."""
        d = {}
        therm_data = {}
        for key, val in data.items():
            cond1 = (filter_str is not None and filter_str in key)
            cond2 = (val == [])
            if cond1 or cond2:
                continue

            val = np.array(val)
            if num_chains is not None:
                val = val[:, :num_chains]

            arr, steps = therm_arr(val, therm_frac=therm_frac)
            therm_data[key] = arr
            arr = arr.T
            chains = np.arange(arr.shape[0])
            d[key] = xr.DataArray(arr, dims=['chain', 'draw'],
                                  coords=[chains, steps])

        dataset = xr.Dataset(d)

        return dataset

    def build_dataset(self, filter_str=None, therm_frac=0.33, num_chains=None):
        """Build `xr.Dataset` containing data."""
        plot_data = {
            'accept_prob': self.data['accept_prob'],
            'sumlogdet_out': self.data['sumlogdet_out'],
            'charges': self.data['charges'],
            'plaqs_err': calc_plaqs_diffs(np.array(self.data['plaqs']),
                                          self.run_params['beta'])
        }
        plot_dataset = self._build_dataset(plot_data,
                                           filter_str=filter_str,
                                           therm_frac=therm_frac,
                                           num_chains=num_chains)
        return plot_dataset

    def plot(self, fig_dir, title_str, **kwargs):
        """Make inference plot from `self.data`."""
        data = self.build_dataset(**kwargs)
        io.check_else_make_dir(fig_dir)
        traceplot_posterior(data, '', fname='run_data',
                            fig_dir=fig_dir, title_str=title_str)

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
    def calc_tunneling_rate(charges, therm_frac=0.33):
        """Calculate the tunneling rate as the charge difference per step."""
        charges, _ = therm_arr(charges, therm_frac=therm_frac)
        charges = np.insert(charges, 0, charges[0], axis=0)
        dq = np.abs(np.around(charges[1:]) - np.around(charges[:-1]))
        tunn_rate = np.sum(dq, axis=0) / charges.shape[0]

        return tunn_rate

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

    def save(self, run_dir, save_samples=False):
        """Save `self.data` to `run_dir`."""
        io.check_else_make_dir(run_dir)
        io.save_dict(self.run_params, run_dir, name='run_params')
        for key, val in self.data.items():
            if key == 'x_out' and not save_samples:
                continue

            out_file = os.path.join(run_dir, f'{key}.z')
            io.savez(np.array(val), out_file, name=key)

    def thermalize_data(self, data=None, therm_frac=0.33):
        """Returns thermalized versions of entries in data."""
        if data is None:
            data = self.data

        therm_data = {}
        for key, val in data.items():
            arr, _ = therm_arr(np.array(val), therm_frac=therm_frac)
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
        io.log(f'Writing run summary statistics to {out_file}.\n')
        data = {
            'accept_prob': self.data['accept_prob'],
            'charges': self.data['charges'],
            'plaqs_diffs': calc_plaqs_diffs(self.data['plaqs'],
                                            self.run_params['beta'])
        }
        therm_data = self.thermalize_data(data)
        tunn_stats = self.calc_tunneling_stats(therm_data['charges'])
        io.log_and_write(80*'-' + '\n\n', out_file)
        self._log_stats(therm_data, tunn_stats, out_file, n_boot=n_boot)

        io.log_and_write(120 * '=' + '\n', out_file)

        return therm_data, tunn_stats


class RunnerTF:
    def __init__(self, FLAGS):
        params = find_params(FLAGS.log_dir)
        self.params = params
        log_dir = params['log_dir'] if FLAGS.log_dir is None else FLAGS.log_dir
        self.log_dir = log_dir

        self.sess = self.restore()
        self.run_ops = get_run_ops()
        self.obs_ops = get_obs_ops()
        self.inputs = get_inputs()

        run_params, title_str = self.parse_flags(params, FLAGS)
        self.run_params = run_params
        self.title_str = title_str
        self.eps = run_params['eps']
        self.xdim = run_params['xdim']
        self.beta = run_params['beta']
        self.num_steps = run_params['num_steps']
        self.batch_size = run_params['batch_size']
        self.net_weights = run_params['net_weights']
        self.run_steps = run_params['run_steps']
        self.print_steps = run_params['print_steps']

        self.run_str = self.get_run_str(params)
        self.run_dir = os.path.join(self.log_dir, 'runs_tf', self.run_str)
        self.fig_dir = os.path.join(self.log_dir, 'figures_tf', self.run_str)
        #  io.check_else_make_dir(self.run_dir)
        #  io.check_else_make_dir(self.fig_dir)

        self.ops_dict = {
            'x_out': self.run_ops['x_out'],
            'accept_prob': self.run_ops['accept_prob'],
            'sumlogdet_out': self.run_ops['sumlogdet_out'],
            'exp_energy_diff': self.run_ops['exp_energy_diff'],
            'plaq_sums': self.obs_ops['plaq_sums'],
            'plaqs': self.obs_ops['plaqs'],
            'charges': self.obs_ops['charges'],
        }
        self.keys = list(self.ops_dict.keys())
        self.ops = list(self.ops_dict.values())

        self.feed_dict = {
            self.inputs['beta']: self.beta,
            self.inputs['net_weights']: self.net_weights,
            self.inputs['train_phase']: False
        }

    def restore(self, log_dir=None):
        """Restore from checkpoint."""
        if log_dir is None:
            log_dir = self.log_dir
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
        saver = tf.train.import_meta_graph(f'{ckpt_file}.meta')
        saver.restore(sess, ckpt_file)

        return sess

    def parse_flags(self, params, FLAGS):
        """Parse FLAGS."""
        net_weights = NET_WEIGHTS_HMC if FLAGS.hmc else NET_WEIGHTS_L2HMC
        beta = params['beta_final'] if FLAGS.beta is None else FLAGS.beta
        xdim = params['time_size'] * params['space_size'] * params['dim']
        batch_size = (
            params['batch_size'] if FLAGS.batch_size is None
            else FLAGS.batch_size
        )

        if FLAGS.eps is not None:
            set_eps(self.sess, FLAGS.eps)
            eps = FLAGS.eps
        else:
            eps = self.sess.run(self.run_ops['dynamics_eps'])

        run_params = {
            'eps': eps,
            'xdim': xdim,
            'beta': beta,
            'log_dir': self.log_dir,
            'num_steps': params['num_steps'],
            'batch_size': batch_size,
            'net_weights': net_weights,
            'run_steps': FLAGS.run_steps,
            'print_steps': FLAGS.print_steps,
        }

        title_str = (f"{params['time_size']}"
                     r"$\times$" + f"{params['space_size']}, "
                     r"$\beta = $" + f'{beta:.3g}, '
                     r"$N_{\mathrm{LF}} = $" + f"{params['num_steps']}, "
                     r"$N_{\mathrm{B}} = $" + f'{batch_size}, '
                     r"$\varepsilon = $" + f'{eps:.4g}')

        return run_params, title_str

    def get_run_str(self, params):
        """Get `run_str`."""
        lf = params['num_steps']
        bs = params['batch_size']
        rs = self.run_steps
        beta = self.beta
        eps = self.eps
        nw = self.net_weights
        tstr = io.get_timestr()
        tstr = tstr['timestr']
        nwstr = ''.join([str(int(i)) for i in self.net_weights])
        run_str = (f'lf{lf}_bs{bs}_steps{rs}_beta{beta}'
                   f'_eps{eps:.2g}_nw{nwstr}__{tstr}')

        return run_str

    # pylint:disable=invalid-name
    def run_step(self, step, x, run_data):
        """Perform a single run step."""
        self.feed_dict.update({
            self.inputs['x']: x,
        })

        t0 = time.time()
        outputs = self.sess.run(self.ops, feed_dict=self.feed_dict)
        dt = time.time() - t0

        out_dict = dict(zip(self.keys, outputs))

        if step > 1:
            q_old = np.array(run_data.data['charges'][-1])
            q_new = np.array(out_dict['charges'])
            dq = np.abs(q_new - q_old)
        else:
            dq = np.zeros(self.batch_size)

        out_dict['dq'] = dq
        plaq_err = calc_plaqs_diffs(out_dict['plaqs'], self.beta)

        data_str = (f"{step:>6g}/{self.run_steps:<6g} "
                    f"{dt:^11.4g} "
                    f"{np.mean(out_dict['accept_prob']):^11.4g} "
                    f"{np.mean(out_dict['sumlogdet_out']):^11.4g} "
                    f"{np.mean(out_dict['exp_energy_diff']):^11.4g} "
                    f"{np.sum(dq):^11.4g} "
                    f"{np.mean(plaq_err):^11.4g} ")

        return out_dict, data_str

    def inference(self, x=None, run_steps=None, save_samples=False):
        """Run inference."""
        if x is None:
            x = np.random.uniform(low=-np.pi, high=np.pi,
                                  size=(self.batch_size, self.xdim))
        if run_steps is None:
            run_steps = self.run_steps

        run_data = RunData(self.run_params, save_samples)
        print(run_data.header)

        for step in range(run_steps):
            outputs, data_str = self.run_step(step, x, run_data)
            x = outputs['x_out']
            run_data.update(step, outputs, data_str)

        return run_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference on trained model using tensorflow.',
        fromfile_prefix_chars='@',
    )
    parser.add_argument('--beta', dest='beta',
                        required=False, default=None,
                        help=("""Value of `beta` at which to run
                              inference."""))
    parser.add_argument('--log_dir', dest='log_dir',
                        required=False, default=None,
                        help=("""Log dir containing saved model
                              checkpoints."""))
    parser.add_argument('--eps', dest='eps',
                        required=False, default=None,
                        help=("""Step size (`eps`) to use in leapfrog
                              integrator."""))
    parser.add_argument('--batch_size', dest='batch_size',
                        required=False, default=None,
                        help=("""Batch size to use (# of chains to run in
                              parallel."""))
    parser.add_argument('--hmc', dest='hmc',
                        required=False, action='store_true',
                        help=("""Flag that when passed will run generic
                              HMC."""))
    parser.add_argument('--run_steps', dest='run_steps',
                        required=False, default=10000,
                        help=("""Number of inference steps to run."""))
    parser.add_argument('--plot_chains', dest='plot_chains',
                        required=False, default=None,
                        help=("""Number of chains to include when making
                              plots."""))
    parser.add_argument('--print_steps', dest='print_steps',
                        required=False, default=10,
                        help=("""Frequency with which to print data."""))

    if sys.argv[1].startswith('@'):
        args = parser.parse_args(shlex.split(open(sys.argv[1][1:]).read(),
                                             comments=True))
    else:
        args = parser.parse_args()

    return args


def main(FLAGS):
    """Main method."""
    runner = RunnerTF(FLAGS)
    run_data = runner.inference(FLAGS.run_steps)
    run_data.plot(runner.fig_dir, runner.title_str,
                  num_chains=FLAGS.plot_chains)
    out_file = os.path.join(runner.fig_dir, 'run_summary.txt')
    _, _ = run_data.log_summary(out_file=out_file, n_boot=10)

    if not FLAGS.dont_save:
        run_data.save(run_dir=runner.run_dir)

    _ = shutil.copy2(out_file, runner.run_dir)

    return run_data


if __name__ == '__main__':
    CLI_FLAGS = parse_args()
    for key, val in CLI_FLAGS.__dict__.items():
        io.log(f'{key}: {val}\n')

    _ = main(CLI_FLAGS)
