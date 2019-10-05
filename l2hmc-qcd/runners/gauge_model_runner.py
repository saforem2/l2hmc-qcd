"""
gauge_model_runner.py

Implements GaugeModelRunner class, responsible for running inference (via the
L2HMC algorithM) on a trained model for 2-D U(1) lattice gauge theory.

Author: Sam Foreman (github: @saforem2)
Date: 09/17/2019
"""
import time

import numpy as np
import utils.file_io as io

from lattice.lattice import u1_plaq_exact
from loggers.run_logger import RunLogger


def directionalize(key):
    return key + '_f', key + '_b'


class GaugeModelRunner:
    """GaugeModelRunner object, responsible for running inference."""

    def __init__(self, sess, params, inputs, run_ops, logger=None):
        """Initialization method.

        Args:
            sess: `tf.Session()` object.
            params (dict): Dictionary of parameters.
            inputs (array-like): Array of `tf.placeholder` objects that specify
                the inputs to be used.
            run_ops (array-like): Array of operations to be run for each
                inference (run) step. Obtainable from
                `tf.get_collection('run_ops')`.
            logger: RunLogger object (defined in `loggers/`), defaults to None.
                This is to simplify communication when using Horovod since the
                RunLogger object exists only on `hvd.rank() == 0`, i.e. the
                chief node that performs all file I/O.
        """
        self.sess = sess
        self.params = params
        self.logger = logger

        if logger is not None:
            self.inputs_dict = self.logger.inputs_dict
            self.run_ops_dict = self.logger.run_ops_dict
            self._has_logger = True
            self._run_header = self.logger.run_header
        else:
            self.inputs_dict = RunLogger.build_inputs_dict(inputs)
            self.run_ops_dict = RunLogger.build_run_ops_dict(params, run_ops)
            self._has_logger = False
            self._run_header = ''

        self.eps = self.sess.run(self.run_ops_dict['dynamics_eps'])

    def run_step(self, step, run_steps, inputs, net_weights):
        """Perform a single run step.

        Args:
            step (int): Current step.
            run_steps (int): Total number of run_steps to perform.
            inputs (tuple): Tuple consisting of (samples_in, beta_np, eps,
                plaq_exact), where samples_in (np.ndarray) is the input batch
                of samples, beta (float) is the input value of beta, eps is the
                step size, and plaq_exact (float) is the expected avg. value of
                the plaquette at this value of beta.

        Returns:
            out_data: Dictionary containing the output of running all of the
            tensorflow operations in `ops` defined below.
        """
        samples_in, beta_np, eps, plaq_exact = inputs

        keys = ['x_out', 'px', 'actions_op',
                'plaqs_op', 'charges_op', 'charge_diffs_op']

        ops = [self.run_ops_dict[k] for k in keys]

        feed_dict = {
            self.inputs_dict['x']: samples_in,
            self.inputs_dict['beta']: beta_np,
            self.inputs_dict['net_weights'][0]: net_weights[0],
            self.inputs_dict['net_weights'][1]: net_weights[1],
            self.inputs_dict['net_weights'][2]: net_weights[2],
            self.inputs_dict['train_phase']: False
        }

        t0 = time.time()
        outputs = self.sess.run(ops, feed_dict=feed_dict)
        dt = time.time() - t0

        out_data = {
            'step': step,
            'beta_np': beta_np,
            'eps': self.eps,
            'samples': np.mod(outputs[0], 2 * np.pi),
            'samples_orig': outputs[0],
            'px': outputs[1],
            'actions': outputs[2],
            'plaqs': outputs[3],
            'charges': outputs[4],
            'charge_diffs': outputs[5],
        }

        if self.params['save_lf']:
            lf_outputs = self.lf_step(feed_dict)

        out_data.update(lf_outputs)

        data_str = (f'{step:>5g}/{run_steps:<6g} '
                    f'{dt:^9.4g} '                      # time / step
                    f'{np.mean(outputs[1]):^9.4g} '     # accept. prob
                    f'{self.eps:^9.4g} '                # step size
                    f'{beta_np:^9.4g} '                 # beta val
                    f'{np.mean(outputs[2]):^9.4g} '     # avg. actions
                    f'{np.mean(outputs[3]):^9.4g} '     # avg. plaquettes
                    f'{plaq_exact:^9.4g} '              # exact plaquette val
                    f'{outputs[5]:^9.4g} ')             # top. charge diff

        return out_data, data_str

    def lf_step(self, feed_dict):
        """Run auxiliary operations if `self.params['save_lf']` is True."""
        lf_keys = [*directionalize('lf_out'),
                   *directionalize('pxs_out'),
                   *directionalize('masks'),
                   *directionalize('logdets'),
                   *directionalize('sumlogdet')]
        lf_ops = [self.run_ops_dict[k] for k in lf_keys]

        lf_outputs_ = self.sess.run(lf_ops, feed_dict=feed_dict)

        lf_outputs = {
            k: v for k, v in zip(lf_keys, lf_outputs_)
        }

        return lf_outputs

    def run(self, **kwargs):
        """Run inference ot generate samples and calculate observables."""
        run_steps = int(kwargs.get('run_steps', 5000))
        net_weights = kwargs.get('net_weights', [1., 1., 1.])
        therm_frac = kwargs.get('therm_frac', 10)
        beta = kwargs.get('beta', self.params.get('beta_final', 5.))
        plaq_exact = u1_plaq_exact(beta)

        has_logger = self.logger is not None

        x_dim = self.params['x_dim']
        samples_np = np.random.randn(*(self.params['batch_size'], x_dim))

        io.log(self._run_header)
        for step in range(run_steps):
            inputs = (samples_np, beta, self.eps, plaq_exact)
            out_data, data_str = self.run_step(step, run_steps, inputs,
                                               net_weights)
            samples_np = out_data['samples']

            if has_logger:
                self.logger.update(self.sess, out_data,
                                   net_weights, data_str)

        #  if self._has_logger:
        if has_logger:
            self.logger._write_run_history()  # XXX
            self.logger.save_run_data(therm_frac=therm_frac)
