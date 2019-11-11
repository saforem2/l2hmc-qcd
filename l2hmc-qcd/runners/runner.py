"""
runner.py

Implements GaugeModelRunner class responsible for running the L2HMC algorithm
on a U(1) gauge model.

Author: Sam Foreman (github: @saforem2)
Date: 04/09/2019
"""
import time

import numpy as np
import utils.file_io as io

from collections import namedtuple

from config import Energy, EnergyData, State
from lattice.lattice import u1_plaq_exact
from lattice.utils import actions
from loggers.run_logger import RunLogger


class EnergyRunner:
    """EnergyRunner object for imperatively calculating energies."""
    def __init__(self, potential_fn):
        self.potential_fn = potential_fn

    def potential_energy(self, state):
        return state.beta * self.potential_fn(state.x)

    def kinetic_energy(self, state):
        return 0.5 * np.sum(state.v ** 2, axis=1)

    def hamiltonian(self, state):
        return self.potential_energy(state) + self.kinetic_energy(state)

    def calc_energies(self, state, sumlogdet=0.):
        pe = self.potential_energy(state)
        ke = self.kinetic_energy(state)
        h = pe + ke - sumlogdet

        energy = Energy(potential=pe, kinetic=ke, hamiltonian=h)

        return energy


class Runner:
    """Runner object, responsible for running inference."""
    def __init__(self, sess, params, logger=None, model_type=None):

        """Initialization method.

        Args:
            sess: A `tf.Session` object.
            params (dict): Dictionary of paramter values.
            logger: A `RunLogger` object.
        """
        self.sess = sess
        self.params = params
        self.logger = logger
        self.model_type = model_type

        if logger is not None:
            self._has_logger = True
            self._run_header = self.logger.run_header
            self.inputs_dict = self.logger.inputs_dict
            self.run_ops_dict = self.logger.run_ops_dict
            self.energy_ops_dict = self.logger.energy_ops_dict
            if model_type == 'GaugeModel':
                self.obs_ops_dict = self.logger.obs_ops_dict
        else:
            self._has_logger = False
            self._run_header = ''
            self.inputs_dict = RunLogger.build_inputs_dict()
            self.run_ops_dict = RunLogger.build_run_ops_dict()
            self.energy_ops_dict = RunLogger.build_energy_ops_dict()
            if model_type == 'GaugeModel':
                self.obs_ops_dict = RunLogger.build_obs_ops_dict()

        self._inference_keys = list(self.run_ops_dict.keys())
        self._inference_ops = list(self.run_ops_dict.values())
        self._energy_keys = list(self.energy_ops_dict.keys())
        self._energy_ops = list(self.energy_ops_dict.values())
        if model_type == 'GaugeModel':
            self._obs_keys = list(self.obs_ops_dict.keys())
            self._obs_ops = list(self.obs_ops_dict.values())
            self.energy_runner = EnergyRunner(potential_fn=actions)

        self.eps = self.sess.run(self.run_ops_dict['dynamics_eps'])

    def _calc_energies_np(self, state, sumlogdet=0.):
        """Calculate energies imperatively during inference run to compare.

        Args:
            state (State object): State is a namedtuple (defined in
                `config.py`) of the form (x, v, beta).

        Returns:
            pe: Potential energy of the state.
            ke: Kinetic energy of the state.
            h: Hamiltonian of the state.
        """
        if hasattr(self, 'energy_runner'):
            energies = self.energy_runner.calc_energies(state, sumlogdet)
        else:
            raise AttributeError('No `EnergyRunner` attribute found.')

        return energies

    def calc_energies_np(self, outputs):
        state_init = State(outputs['x_init'], outputs['v_init'], self.beta)
        state_prop = State(outputs['x_proposed'],
                           outputs['v_proposed'],
                           self.beta)
        state_out = State(outputs['x_out'], outputs['v_out'], self.beta)

        sld_prop = outputs['sumlogdet_proposed']
        sld_out = outputs['sumlogdet_out']
        einit = self._calc_energies_np(state_init)
        eprop = self._calc_energies_np(state_prop, sld_prop)
        eout = self._calc_energies_np(state_out, sld_out)

        edata = {
            'potential_init': einit.potential,
            'potential_proposed': eprop.potential,
            'potential_out': eout.potential,
            'kinetic_init': einit.kinetic,
            'kinetic_proposed': eprop.kinetic,
            'kinetic_out': eout.kinetic,
            'hamiltonian_init': einit.hamiltonian,
            'hamiltonian_proposed': eprop.hamiltonian,
            'hamiltonian_out': eout.hamiltonian
        }

        return edata

    def calc_energy_diffs(self, energies, energies_np):
        """Calculate the energy differences between energies and energies_np.

        In theory, they should both be the same. `energies` are calculated by
        running tensorflow operations, whereas `energies_np` are calculated
        imperatively in numpy.
        """
        #  np_dict = {}
        #  for key, val in energies_np._asdict().items():
        #      for k, v in val._asdict().items():
        #          name = f'{key}_{k}'
        #          np_dict[name] = v
        ediffs = {}
        for key in energies.keys():
            e = energies[key]
            e_np = energies_np[key]
            ediffs[key] = e - e_np

        return ediffs

    def run_energy_ops(self, feed_dict):
        """Run all energy ops."""
        outputs = self.sess.run(self._energy_ops, feed_dict=feed_dict)
        return dict(zip(self._energy_keys, outputs))

    def run_inference_ops(self, feed_dict):
        outputs = self.sess.run(self._inference_ops, feed_dict=feed_dict)
        return dict(zip(self._inference_keys, outputs))

    def run_obs_ops(self, feed_dict):
        """Run all observable ops."""
        outputs = self.sess.run(self._obs_ops, feed_dict=feed_dict)
        return dict(zip(self._obs_keys, outputs))

    def run_step(self, step, samples):
        """Perform a single run step."""
        feed_dict = {
            self.inputs_dict['x']: samples,
            self.inputs_dict['beta']: self.beta,
            self.inputs_dict['scale_weight']: self.scale_weight,
            self.inputs_dict['transl_weight']: self.transl_weight,
            self.inputs_dict['transf_weight']: self.transf_weight,
            self.inputs_dict['train_phase']: False
        }

        t0 = time.time()
        outputs = self.run_inference_ops(feed_dict)
        dt = time.time() - t0

        out_dict = {
            'step': step,
            'beta': self.beta,
            'eps': self.eps,
            'samples_in': samples,
            'samples': outputs['x_out'],
            'px': outputs['accept_prob']
        }

        out_dict.update(outputs)

        data_str = (f"{step:>5g}/{self.run_steps:<6g} "
                    f"{dt:^9.4g} "
                    f"{np.mean(out_dict['px']):^9.4g} "
                    f"{self.eps:^9.4g} "
                    f"{self.beta:^9.4g} ")

        if self.model_type == 'GaugeModel':
            out_dict['samples'] = np.mod(outputs['x_out'], 2 * np.pi)
            observables = self.run_obs_ops(feed_dict)
            out_dict.update(observables)
            data_str += (f"{observables['avg_actions']:^9.4g} "
                         f"{observables['avg_plaqs']:^9.4g} "
                         f"{self.plaq_exact:^9.4g} ")

        if (step % self.energy_steps) == 0:
            energies = self.run_energy_ops(feed_dict)

            # Calculate energies independently using imperative numpy functions
            energies_np = self.calc_energies_np(outputs)

            energies_diffs = {}
            for key in energies.keys():
                enp = energies_np[key]
                e = energies[key]
                energies_diffs[key] = e - enp
            #  energies_diffs = self.calc_energy_diffs(energies, energies_np)

            out_dict['energies'] = energies
            out_dict['energies_np'] = energies_np
            out_dict['energies_diffs'] = energies_diffs

            energy_str = f'{step:>5g}/{self.run_steps:<6g} \n'
            for key, val in energies_diffs.items():
                mean = np.mean(val)
                std = np.std(val)
                energy_str += f'  {key}: {mean:.5g} +/- {std:.5g}\n'
            #  energy_str += (f' {k}: {v:.5g}\n' for k, v in
            #                 energies_diffs.items())

        return out_dict, data_str, energy_str

    def _run_setup(self, **kwargs):
        """Prepare for running inference."""
        self.run_steps = int(kwargs.get('run_steps', 5000))
        # how often to run energy calculations
        self.energy_steps = int(kwargs.get('energy_steps', 1.))
        self.therm_frac = int(kwargs.get('therm_frac', 10))
        self.net_weights = kwargs.get('net_weights', [1., 1., 1.])
        self.scale_weight = self.net_weights[0]
        self.transl_weight = self.net_weights[1]
        self.transf_weight = self.net_weights[2]

        self.beta = kwargs.get('beta', self.params.get('beta_final', 5.))
        if self.model_type == 'GaugeModel':
            self.plaq_exact = u1_plaq_exact(self.beta)
        else:
            self.plaq_exact = -1.

    def run(self, **kwargs):
        """Run inference to generate samples and calculate observables."""
        self._run_setup(**kwargs)

        has_logger = self.logger is not None

        samples = kwargs.get('samples', None)
        if samples is None:
            batch_size = self.params['batch_size']
            x_dim = self.params['x_dim']
            samples = np.random.randn(*(batch_size, x_dim))

        io.log(self._run_header)
        for step in range(self.run_steps):
            out_data, data_str, energy_str = self.run_step(step, samples)
            samples = out_data['samples']

            if has_logger:
                self.logger.update(self.sess,
                                   out_data, data_str, energy_str,
                                   self.net_weights)
        if has_logger:
            self.logger.save_run_data(therm_frac=self.therm_frac)
