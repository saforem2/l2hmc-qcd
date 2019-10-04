import time
import numpy as np

from collections import namedtuple

import utils.file_io as io
from loggers.run_logger import RunLogger

RunStepData = namedtuple('RunStepData', [
    'step', 'beta', 'samples', 'prob',
])


class GaussianMixtureModelRunner:

    def __init__(self, sess, params, inputs, run_ops, logger=None):
        self.sess = sess
        self.params = params
        self.logger = logger

        if self.logger is not None:
            self.inputs_dict = self.logger.inputs_dict
            self.run_ops_dict = self.logger.run_ops_dict

        else:
            self.inputs_dict = RunLogger.build_inputs_dict(inputs)
            model_type = 'GaussianMixtureModel'
            self.run_ops_dict = self.logger.build_run_ops_dict(params,
                                                               run_ops,
                                                               model_type)

        self.eps = self.sess.run(self.run_ops_dict['dynamics_eps'])

    def run_step(self, step, run_steps, inputs, net_weights):
        """Perform a single run  (inference) step.

        Args:
            step (int): Current step.
            run_steps (int): Total number of run_steps to perform.
            inputs (tuple): Tuple consisting of (samples_in, beta_np, eps,
            where samples_in (np.ndarray) is the input batch of samples, beta
            (float) is the input value of beta, eps is the step size.

        Returns:
            out_data: Dictionary containing the output of running all the
            tensorflow operations in `ops` defined below.
        """
        samples_in, beta_np, eps = inputs

        keys = ['x_out', 'px']
        num_keys = len(keys)

        if self.params['save_lf']:
            keys += ['lf_out_f', 'pxs_out_f',
                     'lf_out_b', 'pxs_out_b',
                     'masks_f', 'masks_b',
                     'logdets_f', 'logdets_b',
                     'sumlogdet_f', 'sumlogdet_b']

        ops = [self.run_ops_dict[k] for k in keys]

        feed_dict = {
            self.inputs_dict['x']: samples_in,
            self.inputs_dict['beta']: beta_np,
            self.inputs_dict['net_weights'][0]: net_weights[0],
            self.inputs_dict['net_weights'][1]: net_weights[1],
            self.inputs_dict['net_weights'][2]: net_weights[2],
            self.inputs_dict['train_phase']: False
        }

        start_time = time.time()
        outputs = self.sess.run(ops, feed_dict=feed_dict)
        dt = time.time() - start_time

        out_data = {
            'step': step,
            'beta': beta_np,
            'eps': self.eps,
            'samples': outputs[0],
            'px': outputs[1]
        }

        if self.params['save_lf']:
            out_data.update({k: v for k, v in zip(keys[num_keys:],
                                                  outputs[num_keys:])})

        data_str = (f'{step:>5g}/{run_steps:<6g} '
                    f'{dt:^9.4g} '
                    f'{np.mean(outputs[1]):^9.4g} '
                    f'{self.eps:^9.4g} '
                    f'{beta_np:^9.4g} ')

        return out_data, data_str

    def run(self, **kwargs):
        """Run inference using the trained sampler."""
        run_steps = int(kwargs.get('run_steps', 5000))
        beta = kwargs.get('beta_final', self.params.get('beta_final', 1))
        net_weights = kwargs.get('net_weights', [1., 1., 1.])
        #  therm_frac = kwargs.get('therm_frac', 10)

        has_logger = self.logger is not None

        if beta is None:
            beta = 1.

        x_dim = self.params.get('x_dim', None)
        samples_np = kwargs.get('samples', None)  # initial sample configs
        if samples_np is None:
            samples_np = np.random.rand(*(self.params['batch_size'], x_dim))
        #  samples_np = np.random.randn(*(self.params['batch_size'], x_dim))

        for step in range(run_steps):
            inputs = (samples_np, beta, self.eps)
            out_data, data_str = self.run_step(step, run_steps,
                                               inputs, net_weights)
            samples_np = out_data['samples']

            if has_logger:
                self.logger.update(self.sess, out_data,
                                   net_weights, data_str)
        if has_logger:
            self.logger._write_run_history()  # XXX

        #  if has_logger:
        #      self.logger.save_run_data(therm_frac=therm_frac)
