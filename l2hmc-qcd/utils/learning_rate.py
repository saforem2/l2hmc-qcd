"""
learning_rate.py

Implements `WarmupExponentialDecay`, a learning rate scheduler that gradually
increases the learning rate initially before returning to an exponentially
decaying schedule.
"""
from __future__ import absolute_import, division, print_function

import time

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

import utils.file_io as io

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

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pass


class WarmupExponentialDecay(LearningRateSchedule):
    """A lr schedule that slowly increases before an ExponentialDecay."""

    def __init__(
            self,
            initial_learning_rate: float,
            decay_steps: int,
            decay_rate: float,
            warmup_steps: int,
            staircase: bool = True,
            name: str = 'WarmupExponentialDecay',
    ):
        super(WarmupExponentialDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.staircase = staircase
        self.name = name

    @tf.function
    def __call__(self, step):
        with tf.name_scope(self.name or 'WarmupExponentialDecay') as name:
            initial_learning_rate = ops.convert_to_tensor_v2(
                self.initial_learning_rate, name='initial_learning_rate'
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            decay_rate = tf.cast(self.decay_rate, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)

            global_step_recomp = tf.cast(step, dtype)
            #  if global_step_recomp < warmup_steps:
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
            'initial_learning_rate': self.initial_learning_rate,
            'decay_steps': self.decay_steps,
            'decay_rate': self.decay_rate,
            'staircase': self.staircase,
            'warmup_steps': self.warmup_steps,
            'name': self.name
        }

