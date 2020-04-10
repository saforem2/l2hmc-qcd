"""
trainer.py
"""
import time

from collections import namedtuple

import numpy as np

import utils.file_io as io

from config import NetWeights, NP_FLOAT
from lattice.lattice import u1_plaq_exact

TrainStepData = namedtuple('TrainStepData', [
    'step', 'beta', 'loss', 'samples', 'prob', 'lr', 'eps'
])


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

        annealing_fn = params.get('annealing_fn', linear_add_cooling)

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
        #  dx_avg = np.mean((outputs['dxf'] + outputs['dxb']) / 2)

        data_str = (
            f"{global_step:>6g}/{self._train_steps:<6g} "
            f"{dt:^11.4g} "
            f"{outputs['loss_op']:^11.4g} "
            f"{np.mean(outputs['px']):^11.4g} "
            f"{outputs['dynamics_eps']:^11.4g} "
            f"{np.mean(outputs['dx_out']):^11.4g} "
            #  f"{dx_avg:^11.4g} "
            #  f"{outputs['dx']:^11.4g} "
            #  f"{np.mean(outputs['x_out'] - samples):^11.4g} "
            f"{outputs['beta']:^11.4g} "
            f"{outputs['lr']:^11.4g} "
            f"{np.mean(outputs['exp_energy_diff']):^11.4g} "
            f"{np.mean(outputs['sumlogdet']):^11.4g} "
            #  f"{outputs['direction']:^11.4g} "
        )

        if self.model._model_type == 'GaugeModel':
            outputs['x_out'] = np.mod(outputs['x_out'], 2 * np.pi)
            dx = 1. - np.mean(np.cos(outputs['x_out'] - samples), axis=-1)
            #  dx = np.mean(np.abs(outputs['x_out'] - samples), axis=-1)
            outputs['dx'] = dx
            plaq_diff = u1_plaq_exact(beta) - outputs['plaqs']
            data_str += (
                #  f"{np.mean(outputs['actions']):^9.4g} "
                f"{np.mean(plaq_diff):>11.4g} "
                #  f"{outputs['plaq_exact']:^9.4g}"
            )

        return outputs, data_str

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

        if net_weights is None:
            net_weights = NetWeights(1, 1, 1, 1, 1, 1)

        initial_step = self.sess.run(self.model.global_step)
        io.log(f'Global step: {initial_step}\n')

        if beta is None:
            beta = self.beta_arr[0]

        if samples is None:
            samples = np.random.randn(self.model.x.shape)

        if self.model._model_type == 'GaugeModel':
            samples = np.mod(samples, 2 * np.pi)

        assert samples.shape == self.model.x.shape

        try:
            if self.logger is not None:
                io.log(self.logger.train_header)

            for step in range(initial_step, train_steps):
                data, data_str = self.train_step(step, samples, net_weights)
                if self.logger is not None:
                    self.logger.update(self.sess, data, data_str, net_weights)

                samples = data['x_out']
                if self.model._model_type == 'GaugeModel':
                    samples = np.mod(samples, 2 * np.pi)


            if self.logger is not None:
                self.logger.write_train_strings()
            #  self.logger.save_train_data()

        except (KeyboardInterrupt, SystemExit):
            io.log("\nERROR: KeyboardInterrupt detected!")
            io.log("INFO: Saving current state and exiting.")
            if self.logger is not None:
                self.logger.update(data, data_str, net_weights)
                self.logger.write_train_strings()
