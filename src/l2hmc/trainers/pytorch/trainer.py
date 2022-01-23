"""
trainer.py

Implements methods for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
import time
from typing import Callable

from accelerate import Accelerator
from rich.console import Console
from src.l2hmc.configs import Steps
from src.l2hmc.dynamics.pytorch.dynamics import Dynamics, random_angle, to_u1
from src.l2hmc.loss.pytorch.loss import LatticeLoss
from src.l2hmc.utils.history import History, StateHistory, summarize_dict
from src.l2hmc.utils.step_timer import StepTimer
import torch
from torch import optim

Tensor = torch.Tensor

console = Console(color_system='truecolor', log_path=False)


class Trainer:
    def __init__(
            self,
            steps: Steps,
            dynamics: Dynamics,
            optimizer: optim.Optimizer,
            loss_fn: Callable,
            accelerator: Accelerator,
            # device: torch.device | str = None,
    ) -> None:
        self.steps = steps
        self.dynamics = dynamics
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self._with_cuda = torch.cuda.is_available()
        self.accelerator = accelerator

        # self.device = device if device is not None else

        self.history = History(steps=steps)
        evals_per_step = self.dynamics.config.nleapfrog * steps.log
        self.timer = StepTimer(evals_per_step=evals_per_step)

    def train_step(self, inputs: tuple[Tensor, float]) -> tuple[Tensor, dict]:
        self.timer.start()
        self.dynamics.train()

        x_init, beta = inputs
        x_init = x_init.to(self.accelerator.device)

        x_out, metrics = self.dynamics((to_u1(x_init), beta))
        x_prop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=x_init, x_prop=x_prop, acc=metrics['acc'])
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        # loss.backward()
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
