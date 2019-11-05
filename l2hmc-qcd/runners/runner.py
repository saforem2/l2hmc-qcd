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

from lattice.lattice import u1_plaq_exact
from loggers.run_logger import RunLogger
#  from config import RUN_HEADER


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

        self.eps = self.sess.run(self.run_ops_dict['dynamics_eps'])

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
            out_dict.update({'energies': energies})

        return out_dict, data_str

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
            out_data, data_str = self.run_step(step, samples)
            samples = out_data['samples']

            if has_logger:
                self.logger.update(self.sess,
                                   out_data, data_str,
                                   self.net_weights)
        if has_logger:
            self.logger.save_run_data(therm_frac=self.therm_frac)
