"""
trainer.py
"""
import time

from collections import namedtuple

import numpy as np

import utils.file_io as io
from utils.attr_dict import AttrDict

from config import NET_WEIGHTS_L2HMC
from lattice.lattice import u1_plaq_exact

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
    def __init__(self, sess, model, logger=None, FLAGS=None):
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
        if FLAGS is None:
            FLAGS = AttrDict(model.params)

        self._train_keys = list(self.model.train_ops.keys())
        self._train_ops = list(self.model.train_ops.values())

        self._params = FLAGS
        self._extra_steps = FLAGS.get('extra_steps', 0)
        self._beta_init = FLAGS.get('beta_init', model.beta_init)
        self._beta_final = FLAGS.get('beta_final', model.beta_final)
        self._save_train_data = FLAGS.get('save_train_data', False)
        self._train_steps = FLAGS.get('train_steps', model.train_steps)
        self._annealing_fn = FLAGS.get('annealing_fn', exp_mult_cooling)

        fixed_beta = FLAGS.get('fixed_beta', False)
        if fixed_beta or self._beta_init == self._beta_final:
            self.beta_arr = np.array([
                self.model.beta_init for _ in range(self._train_steps)
            ])
        else:  # pre-fetch array of all beta values
            self.beta_arr = self.get_betas()
            self._train_steps += self._extra_steps

    def get_betas(self, annealing_fn=None, steps=None, extra_steps=0):
        """Pre-fetch array of beta values to be used in training."""
        t_init = 1. / self._beta_init
        t_final = 1. / self._beta_final
        steps = self._train_steps if steps is None else steps
        if annealing_fn is None:
            annealing_fn = exp_mult_cooling
        t_arr = [
            annealing_fn(i, t_init, t_final, steps) for i in range(steps)
        ]
        if extra_steps > 0:
            t_arr += [t_final for _ in range(extra_steps)]

        return 1. / np.array(t_arr)

    def train_step(self, step, x, beta, net_weights):
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
        start_time = time.time()
        fd = {  # pylint:disable=invalid-name
            self.model.x: x,
            self.model.beta: beta,
            self.model.train_phase: True,
            self.model.net_weights: net_weights,
        }
        outputs = dict(
            zip(self._train_keys, self.sess.run(self._train_ops, feed_dict=fd))
        )

        dt = time.time() - start_time

        outputs.update({
            'x_in': x,
            'beta': beta,
            'step': step,
        })

        data_str = (
            f"{step:>6g}/{self._train_steps:<6g} "         # STEP / TOT_STEPS
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

        try:
            data_str += f"{np.sum(np.around(outputs['dq_prop'])):^11.4g}"
            data_str += f"{np.sum(np.around(outputs['dq_out'])):^11.4g}"
            plaq_diff = u1_plaq_exact(outputs['beta']) - outputs['plaqs']
            data_str += f"{np.mean(plaq_diff):^11.4g} "
        except KeyError:
            pass

        return outputs, data_str

    def train(self, train_steps=None, beta=None, x=None, net_weights=None):
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

        Returns:
            data (dict): Output dictionary from final `self.train_step` call.
        """
        if self.logger is not None:
            io.log(self.logger.train_header)

        beta = self.beta_arr[0] if beta is None else beta
        x = np.random.randn(self.model.x.shape) if x is None else x
        train_steps = self._train_steps if train_steps is None else train_steps
        net_weights = NET_WEIGHTS_L2HMC if net_weights is None else net_weights

        #  try:
        initial_step = self.sess.run(self.model.global_step)
        for step in range(initial_step, train_steps):
            data, data_str = self.train_step(step, x, beta, net_weights)
            x = data['x_out']
            beta = self.beta_arr[step]
            if self.logger is not None:
                self.logger.update(self.sess, data, data_str, net_weights)

        return data
