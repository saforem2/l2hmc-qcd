import time
import numpy as np

from collections import namedtuple

import utils.file_io as io


TrainStepData = namedtuple('TrainStepData', [
    'step', 'beta', 'loss', 'samples', 'prob', 'lr', 'eps'
])


class GaussianMixtureModelTrainer:
    def __init__(self, sess, model, logger=None):
        self.sess = sess
        self.model = model
        self.logger = logger

    def update_beta(self, step, **kwargs):
        """Returns new beta to follow annealing schedule."""
        beta_init = kwargs.get('beta_init', self.model.beta_init)
        beta_final = kwargs.get('beta_final', self.model.beta_final)
        train_steps = kwargs.get('train_steps', self.model.train_steps)

        temp = ((1. / beta_init - 1. / beta_final)
                * (1. - step / float(train_steps))
                + 1. / beta_final)
        new_beta = 1. / temp

        return new_beta

    def train_step(self, step, samples_np, **kwargs):
        """Perform a single training step.

        Args:
            step (int): Current training step.
            samples_np (np.ndarray): Array of input data.
            beta_np (float, optional): Input value for inverse coupling
                constant.

        Returns:
            out_data (dict)
        """
        beta_np = kwargs.get('beta_np', None)
        net_weights = kwargs.get('net_weights', [1., 1., 1.])
        train_steps = kwargs.get('train_steps', self.model.train_steps)

        if beta_np is None:
            if self.model.fixed_beta:
                beta_np = self.model.beta_init
            else:
                beta_np = self.update_beta(step, train_steps=train_steps)

        feed_dict = {
            self.model.x: samples_np,
            self.model.beta: beta_np,
            self.model.net_weights[0]: net_weights[0],
            self.model.net_weights[1]: net_weights[1],
            self.model.net_weights[2]: net_weights[2],
            self.model.train_phase: True
        }

        global_step = self.sess.run(self.model.global_step)

        ops = [
            self.model.train_op,
            self.model.loss_op,
            self.model.x_out,
            self.model.px,
            self.model.dynamics.eps,
            self.model.lr
        ]

        start_time = time.time()
        outputs = self.sess.run(ops, feed_dict=feed_dict)
        dt = time.time() - start_time

        train_step_data = TrainStepData(
            step=global_step,
            beta=beta_np,
            loss=outputs[1],
            samples=outputs[2],
            prob=outputs[3],
            eps=outputs[4],
            lr=outputs[5]
        )

        data_str = (
            f"{global_step:>5g}/{train_steps:<6g} "
            f"{outputs[1]:^9.4g} "              # loss value
            f"{dt:^9.4g} "                      # time / step
            f"{np.mean(outputs[3]):^9.4g}"      # accept prob
            f"{outputs[4]:^9.4g} "              # step size
            f"{beta_np:^9.4g} "                 # beta
            f"{outputs[5]:^9.4g}"               # learning rate
        )

        return train_step_data, data_str

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
        net_weights = kwargs.get('net_weights', [1., 1., 1.])

        if beta_np is None:
            beta_np = self.model.beta_init

        '''
        if samples_np is None:
            samples_np = np.reshape(
                np.array(self.model.distribution.get_samples(, dtype=NP_FLOAT),
                (self.model.num_samples, self.model.x_dim)
            )
        '''

        assert samples_np.shape == self.model.x.shape

        try:
            if self.logger is not None:
                io.log(self.logger.train_header)

            for step in range(initial_step, train_steps):
                data, data_str = self.train_step(step,
                                                 samples_np,
                                                 net_weights=net_weights,
                                                 train_steps=train_steps)
                samples_np = data.samples

                if self.logger is not None:
                    self.logger.update(self.sess, data, data_str, net_weights)

            if self.logger is not None:
                self.logger.write_train_strings()

        except (KeyboardInterrupt, SystemExit):
            io.log("\nKeyboardInterrupt detected!")
            io.log("Saving current state and exiting.")
            if self.logger is not None:
                self.logger.update(data, data_str, net_weights)
