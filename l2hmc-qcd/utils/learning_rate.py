"""
learning_rate.py

Implements `WarmupExponentialDecay`, a learning rate scheduler that gradually
increases the learning rate initially before returning to an exponentially
decaying schedule.
"""
from __future__ import absolute_import, division, print_function

import time

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class WarmupExponentialDecay(LearningRateSchedule):
    """A lr schedule that slowly increases before an ExponentialDecay."""

    def __init__(
            self,
            initial_learning_rate: float,
            decay_steps: int,
            decay_rate: float,
            warmup_steps: int,
            staircase: bool = True,
            name: str = None,
    ):
        super(WarmupExponentialDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.staircase = staircase
        self.name = name

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
            if global_step_recomp < warmup_steps:
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

