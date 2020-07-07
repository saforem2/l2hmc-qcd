"""
learning_rate.py

Implements `WarmupExponentialDecay`, a learning rate scheduler that gradually
increases the learning rate initially before returning to an exponentially
decaying schedule.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

# pylint:disable=import-error
from tensorflow.python.framework import ops
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

import utils.file_io as io

from config import lrConfig


# pylint:disable=too-many-arguments
# pylint:disable=too-few-public-methods
# pylint:disable=too-many-instance-attributes
class ReduceLROnPlateau(LearningRateSchedule):
    """A lr schedule that automatically reduces when loss stagnates."""
    def __init__(
            self,
            initial_learning_rate: float,
            decay_steps: int,
            decay_rate: float,
            staircase: bool = True,
            monitor: str = 'loss',
            factor: float = 0.1,
            patience: int = 10,
            mode: str = 'auto',
            min_delta: float = 1e-4,
            cooldown: int = 0,
            min_lr: float = 0.,
            name: str = 'ReduceLROnPlateau'
    ):
        super(ReduceLROnPlateau, self).__init__()
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0')
        self.monitor = monitor
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        if self.mode not in ['auto', 'min', 'max']:
            io.log((f'Learning rate plateau reducing mode {self.mode} '
                    f'is unknown, fallback to auto mode.'))
            self.mode = 'auto'

        if (self.mode == 'min'
                or self.mode == 'auto' and 'acc' not in self.monitor):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf

        self.cooldown_counter = 0
        self.wait = 0

    # pylint:disable=unused-argument
    def _on_train_begin(self, logs=None):
        # FIXME: Method incomplete
        self._reset()

    # pylint:disable=unused-argument,no-self-use
    def _on_epoch_end(self, epoch, logs=None):
        # FIXME: Method incomplete
        logs = logs or {}


class WarmupExponentialDecay(LearningRateSchedule):
    """A lr schedule that slowly increases before an ExponentialDecay."""

    def __init__(
            self,
            lr_config: lrConfig,
            staircase: bool = True,
            name: str = 'WarmupExponentialDecay',
    ):
        super(WarmupExponentialDecay, self).__init__()
        self.lr_config = lr_config
        self.staircase = staircase
        self.name = name

    @tf.function
    def __call__(self, step):
        with tf.name_scope(self.name or 'WarmupExponentialDecay') as name:
            initial_learning_rate = ops.convert_to_tensor_v2(
                self.lr_config.init, name='initial_learning_rate'
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.lr_config.decay_steps, dtype)
            decay_rate = tf.cast(self.lr_config.decay_rate, dtype)
            warmup_steps = tf.cast(self.lr_config.warmup_steps, dtype)

            global_step_recomp = tf.cast(step, dtype)
            if tf.less(global_step_recomp, warmup_steps):
                return tf.math.multiply(
                    initial_learning_rate,
                    tf.math.divide(global_step_recomp, warmup_steps),
                    name=name
                )

            p = global_step_recomp / decay_steps
            if self.staircase:
                p = tf.math.floor(p)

            return tf.math.multiply(
                initial_learning_rate, tf.math.pow(decay_rate, p),
                name=name
            )

    def get_config(self):
        """Return config for serialization."""
        return {
            'name': self.name,
            'staircase': self.staircase,
            'decay_rate': self.lr_config.decay_rate,
            'decay_steps': self.lr_config.decay_steps,
            'warmup_steps': self.lr_config.warmup_steps,
            'initial_learning_rate': self.lr_config.init,
        }
