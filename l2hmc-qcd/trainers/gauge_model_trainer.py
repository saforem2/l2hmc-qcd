"""
gauge_model_trainer.py

Implements GaugeModelTrainer class responsible for training GaugeModel.

Author: Sam Foreman (github: @saforem2)
Date: 04/09/2019
"""
#  import os
import time
import numpy as np
#  import tensorflow as tf

try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False

import utils.file_io as io
from lattice.lattice import u1_plaq_exact
from globals import NP_FLOAT


h_str = ("{:^12s}{:^10s}{:^10s}{:^10s}{:^10s}"
         "{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}")

h_strf = h_str.format("STEP", "LOSS", "t/STEP", "% ACC", "EPS",
                      "BETA", "ACTION", "PLAQ", "(EXACT)", "dQ", "LR")

dash0 = (len(h_strf) + 1) * '-'
dash1 = (len(h_strf) + 1) * '-'
TRAIN_HEADER = dash0 + '\n' + h_strf + '\n' + dash1


class GaugeModelTrainer:
    def __init__(self, sess, model, logger=None):
        self.sess = sess
        self.model = model
        self.logger = logger

    def update_beta(self, step):
        """Returns new beta to follow annealing schedule."""
        temp = ((1. / self.model.beta_init - 1. / self.model.beta_final)
                * (1. - step / float(self.model.train_steps))
                + 1. / self.model.beta_final)
        new_beta = 1. / temp

        return new_beta

    def train_step(self, step, samples_np, beta_np=None):
        """Perform a single training step.

        Args:
            step (int): Current training step.
            samples_np (np.ndarray): Array of input data.
            beta_np (float, optional): Input value for inverse coupling
                constant.

        Returns:
            out_data (dict)
        """
        start_time = time.time()

        if beta_np is None:
            beta_np = self.update_beta(step)

        fd = {
            self.model.x: samples_np,
            self.model.beta: beta_np
        }

        global_step = self.sess.run(self.model.global_step)

        ops = [
            self.model.train_op,         # apply gradients
            self.model.loss_op,          # calculate loss
            self.model.x_out,            # get new samples
            self.model.px,               # calculate accept prob.
            self.model.dynamics.eps,     # calculate current step size
            self.model.actions_op,       # calculate avg. actions
            self.model.plaqs_op,         # calculate avg. plaqs
            self.model.charges_op,       # calculate top. charges
            self.model.charge_diffs_op,  # change in top. charge / num_samples
            self.model.lr,               # evaluate learning rate
        ]

        outputs = self.sess.run(ops, feed_dict=fd)

        dt = time.time() - start_time

        out_data = {
            'step': global_step,
            'loss': outputs[1],
            'samples': np.mod(outputs[2], 2 * np.pi),
            'px': outputs[3],
            'eps': outputs[4],
            'actions': outputs[5],
            'plaqs': outputs[6],
            'charges': outputs[7],
            'charge_diffs': outputs[8],
            'lr': outputs[9],
            'beta': beta_np
        }

        #  if self.model.using_hvd:
        #      num_workers = hvd.size()
        #      step *= num_workers

        data_str = (
            f"{global_step:>5g}/{self.model.train_steps:<6g} "
            f"{outputs[1]:^9.4g} "              # loss value
            f"{dt:^9.4g} "                      # time / step
            f"{np.mean(outputs[3]):^9.4g}"      # accept prob
            f"{outputs[4]:^9.4g} "              # step size
            f"{beta_np:^9.4g} "                 # beta
            f"{np.mean(outputs[5]):^9.4g} "     # avg. actions
            f"{np.mean(outputs[6]):^9.4g} "     # avg. plaqs.
            f"{u1_plaq_exact(beta_np):^9.4g} "  # exact plaq.
            f"{outputs[8]:^9.4g} "              # charge diff
            f"{outputs[9]:^9.4g}"               # learning rate
        )

        return out_data, data_str

    def train(self, train_steps, **kwargs):
        """Train the L2HMC sampler for `train_steps`.

        Args:
            train_steps: Integer number of training steps to perform.
            **kwargs: Possible (key, value) pairs are
                'samples_np': Array of initial samples used to start
                    training.
                'beta_np': Initial value of beta used in annealing
                    schedule.
                'trace': Flag specifying that the training loop should be
                    ran through a profiler.
        """
        initial_step = kwargs.get('initial_step', 0)
        samples_np = kwargs.get('samples_np', None)
        beta_np = kwargs.get('beta_np', None)

        #  train_strings = []

        if beta_np is None:
            beta_np = self.model.beta_init

        if samples_np is None:
            samples_np = np.reshape(
                np.array(self.model.lattice.samples, dtype=NP_FLOAT),
                (self.model.num_samples, self.model.x_dim)
            )

        assert samples_np.shape == self.model.x.shape

        #  train_file = os.path.join(self.logger.train_dir, 'training_log.txt')
        try:
            #  io.log_and_write(TRAIN_HEADER, train_file)
            io.log(TRAIN_HEADER)
            for step in range(initial_step, train_steps):
                out_data, data_str = self.train_step(step, samples_np)
                samples_np = out_data['samples']

                if self.logger is not None:
                    self.logger.update_training(out_data, data_str)

            if self.logger is not None:
                self.logger.write_train_strings()

        except (KeyboardInterrupt, SystemExit):
            io.log("\nKeyboardInterrupt detected!")
            io.log("Saving current state and exiting.")
            if self.logger is not None:
                self.logger.update_training(out_data, data_str)
            #  self.model.save(self.sess,
            #                  #  self.logger._current_state,
            #                  self.logger.checkpoint_dir)
