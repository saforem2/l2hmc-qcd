import time
import numpy as np

from collections import namedtuple

import utils.file_io as io
from config import NP_FLOAT, NetWeights
from lattice.lattice import u1_plaq_exact


TrainStepData = namedtuple('TrainStepData', [
    'step', 'beta', 'loss', 'samples', 'prob', 'lr', 'eps'
])


def linear_add_cooling(step, temp_init, temp_final, num_steps):
    remaining_frac = (num_steps - step) / num_steps
    temp = temp_final + (temp_init - temp_final) * remaining_frac

    return temp


def exp_mult_cooling(step, temp_init, temp_final, num_steps, alpha=None):
    if alpha is None:
        alpha = np.exp((np.log(temp_final) - np.log(temp_init)) / num_steps)

    temp = temp_init * (alpha ** step)

    return temp


class Trainer:
    """Model-independent `Trainer` object for training the L2HMC sampler."""
    def __init__(self, sess, model, logger=None, annealing_fn=None, **params):
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
        self._train_keys = list(self.model.train_ops.keys())
        self._train_ops = list(self.model.train_ops.values())

        self._beta_init = params.get('beta_init', self.model.beta_init)
        self._beta_final = params.get('beta_final', self.model.beta_final)
        self._train_steps = params.get('train_steps', self.model.train_steps)

        temp_init = 1. / self._beta_init
        temp_final = 1. / self._beta_final
        ts = self._train_steps

        fixed_beta = getattr(model, 'fixed_beta', False)
        if fixed_beta:
            self.beta_arr = np.array([self.model.beta_init for _ in range(ts)])
            return

        if annealing_fn is None:
            self.annealing_fn = linear_add_cooling

        # pre-fetch array of all beta values used during annealing schedule
        args = (temp_init, temp_final, ts)
        temp_arr = np.array([self.annealing_fn(i, *args) for i in range(ts)])
        self.beta_arr = 1. / temp_arr

    def train_step(self, step, samples, **kwargs):
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
        net_weights = kwargs.get('net_weights', NetWeights(1., 1., 1.,
                                                           1., 1., 1.))
        beta = self.beta_arr[step]

        feed_dict = {
            self.model.x: samples,
            self.model.beta: beta,
            self.model.net_weights: net_weights,
            self.model.train_phase: True,
        }

        global_step = self.sess.run(self.model.global_step)

        t0 = time.time()
        ops_out = self.sess.run(self._train_ops, feed_dict=feed_dict)
        dt = time.time() - t0

        outputs = dict(zip(self._train_keys, ops_out))
        outputs['x_in'] = samples
        outputs['step'] = global_step
        outputs['beta'] = beta

        data_str = (
            f"{global_step:>5g}/{self._train_steps:<6g} "
            f"{dt:^9.4g} "
            f"{outputs['loss_op']:^9.4g} "
            f"{np.mean(outputs['px']):^9.4g} "
            f"{outputs['dynamics_eps']:^9.4g} "
            f"{np.mean(outputs['x_out'] - samples):^9.4g} "
            f"{outputs['beta']:^9.4g} "
            f"{outputs['lr']:^9.4g} "
        )

        if self.model._model_type == 'GaugeModel':
            outputs['x_out'] = np.mod(outputs['x_out'], 2 * np.pi)
            outputs['dx'] = np.mean(outputs['x_out'] - samples, axis=-1)
            outputs['plaq_exact'] = u1_plaq_exact(beta)
            data_str += (
                f"{np.mean(outputs['actions']):^9.4g} "
                f"{np.mean(outputs['plaqs']):^9.4g} "
                f"{outputs['plaq_exact']:^9.4g}"
            )

        return outputs, data_str

    def train(self, train_steps, **kwargs):
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
        beta = kwargs.pop('beta', None)
        samples = kwargs.pop('samples', None)
        initial_step = kwargs.pop('initial_step', 0)
        io.log(f'Initial_step: {initial_step}\n')

        net_weights = kwargs.get('net_weights', NetWeights(1, 1, 1,
                                                           1, 1, 1))
        #  net_weights = kwargs.get('net_weights', [1., 1., 1.])
        initial_step = self.sess.run(self.model.global_step)
        io.log(f'Global step: {initial_step}\n')

        if beta is None:
            beta = self.beta_arr[0]

        if samples is None:
            samples = np.random.randn(self.model.x.shape, dtype=NP_FLOAT)

        if self.model._model_type == 'GaugeModel':
            samples = np.mod(samples, 2 * np.pi)

        assert samples.shape == self.model.x.shape

        try:
            if self.logger is not None:
                io.log(self.logger.train_header)
            for step in range(initial_step, train_steps):
                data, data_str = self.train_step(step, samples, **kwargs)
                samples = data['x_out']

                if self.logger is not None:
                    self.logger.update(self.sess, data, data_str, net_weights)

            if self.logger is not None:
                self.logger.write_train_strings()

        except (KeyboardInterrupt, SystemExit):
            io.log("\nERROR: KeyboardInterrupt detected!")
            io.log("INFO: Saving current state and exiting.")
            if self.logger is not None:
                self.logger.update(data, data_str, net_weights)
                self.logger.write_train_strings()
