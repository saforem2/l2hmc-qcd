"""
runner.py

Implements GaugeModelRunner class responsible for running the L2HMC algorithm
on a U(1) gauge model.

Author: Sam Foreman (github: @saforem2)
Date: 04/09/2019
"""
import os
import time
import pickle
from scipy.stats import sem
from collections import Counter, OrderedDict

import numpy as np

import utils.file_io as io
from lattice.lattice import u1_plaq_exact
from loggers.run_logger import RunLogger
from config import RUN_HEADER

#  ops = [                         # list of tensorflow operations to run
#      self.model.x_out,           # new samples (MD + MH accept/reject)
#      self.model.px,              # prob. of accepting proposed samples
#      self.model.actions_op,      # tot. action of each sample
#      self.model.plaqs_op,        # avg. plaquette of each sample
#      self.model.charges_op,      # topological charge Q, of each sample
#      self.model.charge_diffs_op  # Q(x_out) - Q(samples_in)
#  ]


class GaugeModelRunner:

    def __init__(self, sess, params, inputs, run_ops, logger=None):
        """
        Args:
            sess: tf.Session() object.
            model: GaugeModel object (defined in `models/gauge_model.py`)
            logger: RunLogger object (defined in `loggers/run_logger.py`),
                defaults to None. This is to simplify communication when using
                Horovod since the RunLogger object exists only on
                hvd.rank() == 0, which is responsible for all file I/O.
        """
        self.sess = sess
        self.params = params
        self.logger = logger

        if logger is not None:
            self.inputs_dict = self.logger.inputs_dict
        else:
            self.inputs_dict = RunLogger.build_inputs_dict(inputs)

        self.run_ops_dict = self.build_run_ops_dict(params, run_ops)
        self.inputs_dict = self.build_inputs_dict(inputs)
        self.eps = self.sess.run(self.run_ops_dict['dynamics_eps'])

    def build_run_ops_dict(self, params, run_ops):
        """Build dictionary of tensorflow operations used for inference."""
        run_ops_dict = {
            'x_out': run_ops[0],
            'px': run_ops[1],
            'actions_op': run_ops[2],
            'plaqs_op': run_ops[3],
            'avg_plaqs_op': run_ops[4],
            'charges_op': run_ops[5],
            'charge_diffs_op': run_ops[6]
        }

        if params['save_lf']:
            run_ops_dict.update({
                'lf_out_f': run_ops[7],
                'pxs_out_f': run_ops[8],
                'masks_f': run_ops[9],
                'logdets_f': run_ops[10],
                'sumlogdet_f': run_ops[11],
                'fns_out_f': run_ops[12],
                'lf_out_b': run_ops[13],
                'pxs_out_b': run_ops[14],
                'masks_b': run_ops[15],
                'logdets_b': run_ops[16],
                'sumlogdet_b': run_ops[17],
                'fns_out_b': run_ops[18]
            })

        run_ops_dict['dynamics_eps'] = run_ops[-1]

        return run_ops_dict


        return inputs_dict

    def calc_charge_autocorrelation(self, charges):
        autocorr = np.correlate(charges, charges, mode='full')
        return autocorr

    def run_step(self, step, run_steps, inputs, net_weights):
        """Perform a single run step.

        Args:
            step (int): Current step.
            run_steps (int): Total number of run_steps to perform.
            inputs (tuple): Tuple consisting of (samples_in, beta_np, eps,
                plaq_exact) where samples_in (np.ndarray) is the input batch of
                samples, beta (float)is the input value of beta, eps is the
                step size, and plaq_exact (float) is the expected avg. value of
                the plaquette at this value of beta.
        Returns:
            out_data: Dictionary containing the output of running all of the
            tensorflow operations in `ops` defined below.
        """
        samples_in, beta_np, eps, plaq_exact = inputs

        keys = ['x_out', 'px', 'actions_op',
                'plaqs_op', 'charges_op', 'charge_diffs_op']

        if self.params['save_lf']:
            keys.extend(['lf_out_f', 'pxs_out_f',
                         'lf_out_b', 'pxs_out_b',
                         'masks_f', 'masks_b',
                         'logdets_f', 'logdets_b',
                         'sumlogdet_f', 'sumlogdet_b'])

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
            'samples': np.mod(outputs[0], 2 * np.pi),
            'samples_orig': outputs[0],
            'px': outputs[1],
            'actions': outputs[2],
            'plaqs': outputs[3],
            'charges': outputs[4],
            'charge_diffs': outputs[5],
        }

        if self.params['save_lf']:
            lf_outputs = {}
            for key, val in zip(keys[6:], outputs[6:]):
                lf_outputs[key] = val
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

    def run(self, run_steps, **kwargs):
        """Run the simulation to generate samples and calculate observables.

        Args:
            run_steps: Number of steps to run the sampler for.
            current_step: Integer passed when the sampler is ran intermittently
                during training, as a way to monitor the models performance
                during training. By passing the current training step as
                current_step, this data is saved to a unique directory labeled
                by the current_step.
            beta: Float value indicating the inverse coupling constant that the
                sampler is to be run at.

        Returns:
            observables: Tuple of observables dictionaries consisting of:
                (actions_dict, plaqs_dict, charges_dict, charge_diffs_dict).
        """
        run_steps = int(run_steps)

        beta = kwargs.get('beta_final', self.params.get('beta_final', 5))
        net_weights = kwargs.get('net_weights', [1., 1., 1.])
        therm_frac = kwargs.get('therm_frac', 10)

        if beta is None:
            beta = self.params['beta_final']

        if net_weights is None:
            # scale_weight, translation weight, transformation weight
            net_weights = [1., 1., 1.]

        plaq_exact = u1_plaq_exact(beta)

        x_dim = (self.params['space_size']
                 * self.params['time_size']
                 * self.params['dim'])

        # start with randomly generated samples
        samples_np = np.random.randn(*(self.params['num_samples'], x_dim))

        try:
            io.log(RUN_HEADER)
            for step in range(run_steps):
                inputs = (samples_np, beta, self.eps, plaq_exact)
                out_data, data_str = self.run_step(step, run_steps,
                                                   inputs, net_weights)
                samples_np = out_data['samples']

                if self.logger is not None:
                    self.logger.update(self.sess, out_data,
                                       net_weights, data_str)

            if self.logger is not None:
                self.logger.save_run_data(therm_frac=therm_frac)

        except (KeyboardInterrupt, SystemExit):
            io.log("\nKeyboardInterrupt detected!")
            io.log("Saving current state and exiting.")
            if self.logger is not None:
                self.logger.save_run_data(therm_frac=therm_frac)
