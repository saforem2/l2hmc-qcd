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

from src.l2hmc.utils.history import StateHistory, History, summarize_dict
from src.l2hmc.utils.step_timer import StepTimer

TF_FLOAT = tf.keras.backend.floatx()
Tensor = tf.Tensor
console = Console(color_system='truecolor', log_path=False)


class Trainer:
    def __init__(
            self,
            steps: Steps,
            dynamics: Dynamics,
            optimizer: Optimizer,
            loss_fn: Callable = LatticeLoss,
            keep: str | list[str] = None,
            skip: str | list[str] = None,
    ) -> None:
        self.steps = steps
        self.dynamics = dynamics
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.history = History(steps=steps)

        evals_per_step = self.dynamics.config.nleapfrog * steps.log
        self.timer = StepTimer(evals_per_step=evals_per_step)

        self.keep = [] if keep is None else keep
        self.skip = [] if skip is None else skip
        if isinstance(self.keep, str):
            self.keep = [self.keep]
        if isinstance(self.skip, str):
            self.skip = [self.skip]

    def train_step(self, inputs: tuple[Tensor, float]) -> tuple[Tensor, dict]:
        self.timer.start()
        xinit, beta = inputs
        with tf.GradientTape() as tape:
            x_out, metrics = self.dynamics((to_u1(xinit), tf.constant(beta)))
            xprop = to_u1(metrics.pop('mc_states').proposed.x)
            loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])

        grads = tape.gradient(loss, self.dynamics.trainable_variables)
        updates = zip(grads, self.dynamics.trainable_variables)
        self.optimizer.apply_gradients(updates)
        record = {
            'loss': loss,
        }
        for key, val in metrics.items():
            record[key] = val

        return to_u1(x_out), record

    @staticmethod
    def metrics_to_numpy(metrics: dict) -> dict:
        for key, val in metrics.items():
            if isinstance(val, Tensor):
                metrics[key] = val.numpy()  # type: ignore

        return metrics

    def train(
        self,
        xinit: Tensor = None,
        beta: float = 1.,
        compile: bool = True,
        jit_compile: bool = False,
    ) -> dict:
        if xinit is None:
            x = tf.random.uniform(self.dynamics.xshape,
                                  *(-np.pi, np.pi), dtype=TF_FLOAT)
            x = tf.reshape(x, (x.shape[0], -1))
        else:
            x = tf.constant(xinit, dtype=TF_FLOAT)

        assert isinstance(x, Tensor) and x.dtype == TF_FLOAT

        if compile:
            self.dynamics.compile(optimizer=self.optimizer, loss=self.loss_fn)
            train_step = tf.function(self.train_step, jit_compile=jit_compile)
        else:
            train_step = self.train_step

        summaries = []
        for era in range(self.steps.nera):
            console.rule(f'ERA: {era}')
            estart = time.time()
            for epoch in range(self.steps.nepoch):
                self.timer.start()
                x, metrics = train_step((x, beta))  # type: ignore
                should_log = (epoch % self.steps.log == 0)

                should_print = (epoch % self.steps.print == 0)
                if should_log or should_print:
                    record = {
                        'era': era,
                        'epoch': epoch,
                        'dt': self.timer.stop(),
                    }

                    record.update(self.metrics_to_numpy(metrics))
                    avgs = self.history.update(record)
                    summary = summarize_dict(avgs)
                    summaries.append(summary)
                    if should_print:
                        console.log(summary)

            console.rule()
            console.log('\n'.join([
                f'Era {era} took: {time.time() - estart:<3.2g}s',
                f'Avgs over last era:\n {self.history.era_summary(era)}',
            ]))
            console.rule()

        return {'summaries': summaries, 'history': self.history}


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
            x, metrics = train_step((x, beta),  # type: ignore
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


def setup_beta(beta: list | float):  # -> list[float]
    # TODO: Deal with making sure len(beta) == len(train_steps)
    beta = beta
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
