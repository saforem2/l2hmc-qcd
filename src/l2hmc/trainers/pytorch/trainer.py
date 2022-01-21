"""
trainer.py

Implements methods for training L2HMC sampler.
"""
from __future__ import absolute_import, division, print_function, annotations

import torch
import time
from torch import optim
from rich.console import Console
from typing import Callable
from src.l2hmc.configs import Steps
from src.l2hmc.dynamics.pytorch.dynamics import Dynamics, to_u1, random_angle
from src.l2hmc.loss.pytorch.loss import LatticeLoss
from src.l2hmc.utils.history import History, StateHistory, summarize_dict
from src.l2hmc.utils.step_timer import StepTimer

Tensor = torch.Tensor

console = Console(color_system='truecolor', log_path=False, )


class Trainer:
    def __init__(
            self,
            steps: Steps,
            dynamics: Dynamics,
            optimizer: optim.Optimizer,
            loss_fn: Callable = LatticeLoss,
    ) -> None:
        self.steps = steps
        self.dynamics = dynamics
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.history = History(steps=steps)
        evals_per_step = self.dynamics.config.nleapfrog * steps.log
        self.timer = StepTimer(evals_per_step=evals_per_step)

    def train_step(self, inputs: tuple[Tensor, float]) -> tuple[Tensor, dict]:
        self.timer.start()
        self.dynamics.train()
        x_init, beta = inputs
        if torch.cuda.is_available():
            x_init = x_init.cuda()

        x_out, metrics = self.dynamics((to_u1(x_init), beta))
        x_prop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=x_init, x_prop=x_prop, acc=metrics['acc'])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        record = {
            'dt': self.timer.stop(),
            'loss': loss.detach().cpu().numpy(),
        }
        for key, val in metrics.items():
            if isinstance(val, Tensor):
                record[key] = val.detach().cpu().numpy()
            else:
                record[key] = val

        return to_u1(x_out).detach(), record

    def train(self, xinit: Tensor = None, beta: float = 1.) -> dict:
        summaries = []
        x = xinit
        if x is None:
            x = random_angle(self.dynamics.xshape, requires_grad=True)
            x = x.reshape(x.shape[0], -1)

        for era in range(self.steps.nera):
            console.rule(f'ERA: {era}')
            estart = time.time()
            for epoch in range(self.steps.nepoch):
                x, metrics = self.train_step((x, beta))
                should_print = (epoch % self.steps.print == 0)
                should_log = (epoch % self.steps.log == 0)
                if should_print or should_log:
                    record = {
                        'era': era,
                        'epoch': epoch,
                        'dt': self.timer.stop(),
                    }

                    # Update metrics with train step metrics, tmetrics
                    record.update(metrics)
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
    optimizer: optim.Optimizer,
    loss_fn: Callable = LatticeLoss,
    xinit: Tensor = None,
    record_states: bool = False,
) -> dict:
    """Train the L2HMC sampler."""
    history = History(steps)
    states = StateHistory()
    mstrs = []

    x = xinit
    if x is None:
        x = random_angle(dynamics.xshape, requires_grad=True)
        x = x.reshape(x.shape[0], -1)

    for era in range(steps.nera):
        console.rule(f'ERA: {era}')
        estart = time.time()
        for epoch in range(steps.nepoch):
            x, metrics = train_step((x, beta),
                                    dynamics=dynamics,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn)
            if (epoch % steps.log == 0) or (epoch % steps.print == 0):
                record = {
                    'era': era,
                    'epoch': epoch,
                }
                # NOTE:
                # Pop (large) mc_states from tmetrics before updating metrics
                # preventing them from being tracked unless explicitly asked
                mc_states = metrics.pop('mc_states', None)
                if record_states and mc_states is not None:
                    states.update(mc_states)

                # Update metrics with train step metrics, tmetrics
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


def train_step(
        inputs: tuple[Tensor, float],
        dynamics: Dynamics,
        optimizer: optim.Optimizer,
        loss_fn: Callable = LatticeLoss,
) -> tuple[Tensor, dict]:
    start = time.time()
    dynamics.train()
    x_init, beta = inputs
    if torch.cuda.is_available():
        x_init = x_init.cuda()

    x_out, tmetrics = dynamics((to_u1(x_init), beta))
    x_prop = tmetrics.get('mc_states').proposed.x
    loss = loss_fn(x_init=x_init, x_prop=x_prop, acc=tmetrics['acc'])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    metrics = {
        'dt': time.time() - start,
        'loss': loss.detach().cpu().numpy(),
    }
    for key, val in tmetrics.items():
        if isinstance(val, Tensor):
            metrics[key] = val.detach().cpu().numpy()
        else:
            metrics[key] = val

    return to_u1(x_out).detach(), metrics
