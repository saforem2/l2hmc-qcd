"""
trainer.py
"""
import os
import time

from collections import namedtuple

import numpy as np

import utils.file_io as io

from config import NetWeights, NP_FLOAT
from lattice.lattice import u1_plaq_exact
from dynamics.dynamics_np import convert_to_angle

TrainStepData = namedtuple('TrainStepData', [
    'step', 'beta', 'loss', 'samples', 'prob', 'lr', 'eps'
])

# pylint:disable=protected-access


def linear_add_cooling(step, temp_init, temp_final, num_steps):
    """Linear add cooling annealing schedule."""
    remaining_frac = (num_steps - step) / num_steps
    temp = temp_final + (temp_init - temp_final) * remaining_frac

    return temp


def exp_mult_cooling(step, temp_init, temp_final, num_steps, alpha=None):
    """Exponential multiplication cooling schedule."""
    if alpha is None:
        alpha = np.exp((np.log(temp_final) - np.log(temp_init)) / num_steps)

    temp = temp_init * (alpha ** step)

    return temp


class Trainer:
    """Model-independent `Trainer` object for training the L2HMC sampler."""
    def __init__(self, sess, model, logger=None, params=None):
        """Initialization method.
        Args:
            sess (`tf.Session`): Tensorflow session object.
            model: Model specifying how to train the L2HMC sampler.
            logger (`TrainLogger` object): TrainLogger object used for
                logging/keeping track of the training data.
        NOTE: If running distributed training across multiple ranks with
            `horovod`, `logger == None` for all but the `chief` (rank 0) rank
            since the chief rank is responsible for performing all file I/O.
        """
        self.sess = sess
        self.model = model
        self.logger = logger

        if params is None:
            params = model.params
        self._params = params

        self._save_train_data = params.get('save_train_data', False)
        self._train_keys = list(self.model.train_ops.keys())
        self._train_ops = list(self.model.train_ops.values())

        self._beta_init = params.get('beta_init', self.model.beta_init)
        self._beta_final = params.get('beta_final', self.model.beta_final)
        self._train_steps = params.get('train_steps', self.model.train_steps)
        self._extra_steps = params.get('extra_steps', 0)

        temp_init = 1. / self._beta_init
        temp_final = 1. / self._beta_final
        ts = self._train_steps  # pylint: disable=invalid-name

        fixed_beta = getattr(model, 'fixed_beta', False)
        if fixed_beta:
            self.beta_arr = np.array([self.model.beta_init for _ in range(ts)])
            return

        annealing_fn = params.get('annealing_fn', exp_mult_cooling)

        # pre-fetch array of all beta values used during annealing schedule
        args = (temp_init, temp_final, ts)
        temp_arr = [annealing_fn(i, *args) for i in range(ts)]
        temp_arr.extend([temp_final for _ in range(self._extra_steps)])
        temp_arr = np.array(temp_arr)
        self.beta_arr = 1. / temp_arr
        self._train_steps += self._extra_steps

    def train_step(self, step, samples, net_weights=None):
        """Perform a single training step.

        Args:
            step (int): Current training step.
            samples_np (np.ndarray): Array of input configurations.
            beta_np (float, kwarg): Input value for inverse temperature
                (coupling constant).
        Returns:
            out_data (dict): Dictionary containing outputs from the respective
                tensorflow operations.
        """
        if net_weights is None:
            net_weights = NetWeights(1., 1., 1., 1., 1., 1.)

        if self.model._model_type == 'GaugeModel':
            samples = convert_to_angle(samples)

        beta = self.beta_arr[step]

        feed_dict = {
            self.model.x: samples,
            self.model.beta: beta,
            self.model.net_weights: net_weights,
            self.model.train_phase: True,
        }

        global_step = self.sess.run(self.model.global_step)

        start_time = time.time()
        ops_out = self.sess.run(self._train_ops, feed_dict=feed_dict)
        dt = time.time() - start_time

        outputs = dict(zip(self._train_keys, ops_out))

        outputs['x_in'] = samples
        outputs['step'] = global_step
        outputs['beta'] = beta

        data_str = (
            f"{global_step:>6g}/{self._train_steps:<6g} "  # STEP / TOT_STEPS
            f"{dt:^11.4g} "                                # TIME / STEP
            f"{outputs['loss_op']:^11.4g} "                # LOSS VALUE
            f"{np.mean(outputs['px']):^11.4g} "            # ACCEPT_PROB
            f"{outputs['dynamics_eps']:^11.4g} "           # STEP_SIZE
            f"{np.mean(outputs['dx_out']):^11.4g} "        # CHANGE IN X
            f"{outputs['beta']:^11.4g} "                   # CURRENT BETA
            f"{outputs['lr']:^11.4g} "                     # CURRENT_LR
            f"{np.mean(outputs['exp_energy_diff']):^11.4g} "  # exp(H' - H)
            f"{np.mean(outputs['sumlogdet']):^11.4g} "     # SUM log(det)
        )

        if self.model._model_type == 'GaugeModel':
            outputs['x_out'] = convert_to_angle(outputs['x_out'])
            charge_diff, qstr = self._calc_charge_diff(outputs)
            outputs['dq'] = charge_diff
            data_str += qstr

            plaq_diff = u1_plaq_exact(beta) - outputs['plaqs']
            data_str += f"{np.mean(plaq_diff):>11.4g} "

        return outputs, data_str

    def _calc_charge_diff(self, outputs):
        """Calculate the difference in top. charges from prev. step."""
        try:
            q_old = self.logger.train_data['charges'][-1]
            charge_diff = np.abs((outputs['charges'] - q_old))
        except (AttributeError, IndexError, KeyError):
            charge_diff = np.zeros(outputs['charges'].shape)

        qstr = f'{np.sum(np.around(charge_diff)):^11.4g}'

        return charge_diff, qstr

    def train(self, train_steps=None, beta=None,
              samples=None, net_weights=None):
        """Train the L2HMC sampler for `train_steps` steps.
        Args:
            train_steps (int): Number of training steps to perform.
        Kwargs:
            samples (np.ndarray, optional): Initial samples to use as input
                for the first training step.
            beta (float, optional): Initial value of beta to be used in the
                annealing schedule. Overrides `self.model.beta_init`.
            trace (bool, optional): Flag specifying that the training loop
                should be wrapped in a profiler.
        """
        if train_steps is None:
            train_steps = self._train_steps

        if beta is None:
            beta = self.beta_arr[0]

        if net_weights is None:
            net_weights = NetWeights(1, 1, 1, 1, 1, 1)

        if samples is None:
            samples = np.random.randn(self.model.x.shape)

        if self.model._model_type == 'GaugeModel':
            samples = convert_to_angle(samples)

        if self.logger is not None:
            io.log(self.logger.train_header)

        try:
            initial_step = self.sess.run(self.model.global_step)
            for step in range(initial_step, train_steps):
                data, data_str = self.train_step(step,
                                                 samples,
                                                 net_weights)
                if self.logger is not None:
                    self.logger.update(self.sess, data,
                                       data_str, net_weights)

                samples = data['x_out']
                if self.model._model_type == 'GaugeModel':
                    samples = convert_to_angle(samples)

        except (KeyboardInterrupt, SystemExit):
            io.log("\nERROR: KeyboardInterrupt detected!")
            io.log("INFO: Saving current state and exiting.")
            if self.logger is not None:
                self.logger.update(data, data_str, net_weights)
                self.logger.write_train_strings()
