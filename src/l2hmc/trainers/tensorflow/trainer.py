"""
trainer.py

Implements methods for training L2HMC sampler
"""
from __future__ import absolute_import, annotations, division, print_function
import time
from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from rich.console import Console

from src.l2hmc.configs import Steps
from src.l2hmc.dynamics.tensorflow.dynamics import Dynamics, to_u1
from src.l2hmc.loss.tensorflow.loss import LatticeLoss

from src.l2hmc.utils.history import StateHistory, History

Tensor = tf.Tensor
console = Console(color_system='truecolor', log_path=False)


class Trainer:
    def __init__(
        self,
        dynamics: Dynamics,
        beta: list,
        optimizer: Optimizer,
        loss_fn: Callable = LatticeLoss,
    ):
        pass


@tf.function
def train_step(
        inputs: tuple[Tensor, Tensor | float],
        dynamics: Dynamics,
        optimizer: Optimizer,
        loss_fn: Callable = LatticeLoss,
) -> tuple[Tensor, dict]:
    x_init, beta = inputs
    with tf.GradientTape() as tape:
        x_out, tmetrics = dynamics((to_u1(x_init), tf.constant(beta)))
        x_prop = tmetrics.pop('mc_states').proposed.x
        # x_prop = metrics.pop('mc_states').proposed.x
        loss = loss_fn(x_init=x_init, x_prop=x_prop, acc=tmetrics['acc'])

    grads = tape.gradient(loss, dynamics.trainable_variables)
    optimizer.apply_gradients(zip(grads, dynamics.trainable_variables))
    metrics = {
        'loss': loss,
    }
    for key, val in tmetrics.items():
        metrics[key] = val

    return to_u1(x_out), metrics


def setup_beta(beta: list | float):  # -> list[float]
    # TODO: Deal with making sure len(beta) == len(train_steps)
    beta = beta
    pass


def train(
        steps: Steps,
        dynamics: Dynamics,
        beta: float,
        optimizer: Optimizer,
        loss_fn: Callable = LatticeLoss,
        xinit: Tensor = None,
        record_states: bool = False,
) -> dict:
    """Train the L2HMC sampler"""
    history = History(steps)
    states = StateHistory()
    mstrs = []

    if xinit is None:
        x = tf.random.uniform(dynamics.xshape, *(-np.pi, np.pi))
        x = tf.reshape(x, (x.shape[0], -1))
    else:
        x = tf.constant(xinit)

    assert isinstance(x, Tensor)
    dynamics.compile(optimizer=optimizer, loss=loss_fn)
    # train_step_ = tf.function(train_step)
    for era in range(steps.nera):
        console.rule(f'ERA: {era}')
        estart = time.time()
        for epoch in range(steps.nepoch):
            tstart = time.time()
            x, metrics = train_step((x, beta),
                                    dynamics=dynamics,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn)
            if (epoch % steps.log == 0) or (epoch % steps.print == 0):
                record = {
                    'era': era,
                    'epoch': epoch,
                    'dt': time.time() - tstart,
                }
                mc_states = metrics.pop('mc_states', None)
                if record_states and mc_states is not None:
                    states.update(mc_states)

                record.update(metrics)
                mstr = history.update(record)
                mstrs.append(mstr)
                if epoch % steps.print == 0:
                    console.log(mstr)

        console.rule()
        console.log(f'Era {era} took: {time.time() - estart:<3.2g}s')
        console.log(f'Avgs over last era:\n {history.era_summary(era)}')
        console.rule()

    return {
        'history': history,
        'dynamics': dynamics,
        'optimizer': optimizer,
        'state_history': states,
    }
