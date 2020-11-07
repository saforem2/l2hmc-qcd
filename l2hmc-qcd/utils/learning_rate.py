"""
learning_rate.py

Implements `WarmupExponentialDecay`, a learning rate scheduler that gradually
increases the learning rate initially before returning to an exponentially
decaying schedule.
"""
from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend as K

# pylint:disable=import-error
from tensorflow.python.framework import ops
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

import utils.file_io as io

from network.config import LearningRateConfig


def moving_average(x, n=1000):
    out = np.cumsum(x, dtype=np.float32)
    out[n:] = out[n:] - out[:-n]
    return out[n-1:] / n


class ReduceLROnPlateau(tf.keras.callbacks.Callback):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Example:

    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```
    Arguments:
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will be reduced.
            `new_lr = lr * factor`.
        patience: number of epochs with no improvement after which learning
            rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode, the learning
            rate will be reduced when the quantity monitored has stopped
            decreasing; in `'max'` mode it will be reduced when the quantity
            monitored has stopped increasing; in `'auto'` mode, the direction
            is automatically inferred from the name of the monitored quantity.
        min_delta: threshold for measuring the new optimum, to only focus on
            significant changes.
        cooldown: number of epochs to wait before resuming normal operation
            after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """
    def __init__(
            self,
            monitor='loss',
            factor=0.5,
            patience=10,
            verbose=1,
            mode='auto',
            warmup_steps=1000,
            min_delta=1e-4,
            cooldown=0,
            min_lr=1e-6,
            **kwargs
    ):
        super(ReduceLROnPlateau, self).__init__()
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            logging.warning('`epsilon` argument is deprecated and '
                            'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.warmup_steps = warmup_steps
        self.patience = patience
        self.verbose = verbose
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
            logging.warning('Learning Rate Plateau Reducing mode '
                            f'{self.mode} is unknown, fallback to auto mode.')
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            #  self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            #  self.monitor_op = lambda a, b: np.less(
            #      a, b * (1 - self.min_delta)
            #  )
            def _monitor_op(current, best):
                # if current < best - best * delta: reduce lr
                return np.less(current, best - self.min_delta)
                #  return np.less(current, best * (1 - self.min_delta))
            #  if current < best * (1 - delta):
            #  self.monitor_op = lambda a, b: np.less(a
            self.monitor_op = _monitor_op
            self.best = np.Inf
        else:
            #  self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            def _monitor_op(a, b):
                #  return np.greater(a, b * (1 + self.min_delta))
                return np.greater(a, b + self.min_delta)

            self.monitor_op = _monitor_op
            self.best = -np.Inf

        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, step, logs=None):
        if step < self.warmup_steps:
            return

        #  def _get_lr():
        #      try:
        #          lr = K.get_value(self.model.optimizer.lr)
        #      except ValueError:
        #          lr = self.model.lr(self.model.optimizer.iterations)
        #      finally:
        #          lr = None
        #
        #      return lr

        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            #  logging.warning(
            io.log(
                f'ReduceLROnPlateau conditioned on metric'
                f' {self.monitor} which is not available.'
                f' Available metrics are: {",".join(list(logs.keys()))}'
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0
            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    step = self.model.optimizer.iterations
                    old_lr = self.model._get_lr(step)
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            #  logging.warning(
                            print(
                                f'ReduceLROnPlateau (step {step}):'
                                ' Reducing learning rate from:'
                                f' {old_lr} to {new_lr}.',
                            )
                            print(f'current: {current}, best: {self.best}')
                            #  print(f'\nstep {epoch}: ReduceLROnPlateau'
                            #        ' reducing learning rate from:'
                            #        f' {old_lr} to {new_lr:g}.')
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


class WarmupExponentialDecay(LearningRateSchedule):
    """A lr schedule that slowly increases before an ExponentialDecay."""

    def __init__(
            self,
            lr_config: LearningRateConfig,
            staircase: bool = True,
            name: str = 'WarmupExponentialDecay',
    ):
        super(WarmupExponentialDecay, self).__init__()
        #  self.dtype = lr_config.lr_init.dtype
        self.lr_config = lr_config
        self.staircase = staircase
        self.name = name

    @tf.function
    def __call__(self, step):
        with tf.name_scope(self.name or 'WarmupExponentialDecay') as name:
            initial_learning_rate = ops.convert_to_tensor_v2(
                self.lr_config.lr_init, name='initial_learning_rate'
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.lr_config.decay_steps, dtype)
            decay_rate = tf.cast(self.lr_config.decay_rate, dtype)
            warmup_steps = tf.cast(self.lr_config.warmup_steps, dtype)
            global_step_recomp = tf.cast(step, dtype)
            min_lr = tf.constant(1e-5, dtype)

            # warming up?
            if tf.less(global_step_recomp, warmup_steps):
                return min_lr + tf.math.multiply(
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
            'initial_learning_rate': self.lr_config.lr_init,
        }
