"""
trainer.py

Implements methods for training L2HMC sampler
"""
from __future__ import division, absolute_import, print_function, annotations

import tensorflow as tf
import time
from typing import Callable
from src.l2hmc.dynamics.tensorflow.dynamics import Dynamics, to_u1

from src.l2hmc.loss.tensorflow.loss import LatticeLoss

from tensorflow.keras.optimizers import Optimizer

Tensor = tf.Tensor

def train_step(
        inputs: tuple[Tensor, Tensor],
        dynamics: Dynamics,
        optimizer: Optimizer,
        loss_fn: Callable = LatticeLoss,
) -> tuple[Tensor, dict]:
    start = time.time()
    x_init, beta = inputs

    with tf.GradientTape() as tape:
        x_out, metrics = dynamics((to_u1(x_init), beta))
        x_prop = metrics.pop('mc_states').proposed.x
        loss = loss_fn(x_init=x_init, x_prop=x_prop, acc=metrics['acc'])

    grads = tape.gradient(loss, dynamics.trainable_variables)
    optimizer.apply_gradients(zip(grads, dynamics.trainable_variables))
    metrics['dt'] = time.time() - start
    metrics['loss'] = loss

    return to_u1(x_out), metrics
