"""
trainer.py

Implements methods for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
import time
from typing import Callable

from accelerate import Accelerator
import numpy as np
import torch
from torch import optim

from l2hmc.configs import Steps
from l2hmc.dynamics.pytorch.dynamics import Dynamics, random_angle, to_u1
from l2hmc.utils.history import summarize_dict, BaseHistory
# from l2hmc.utils.pytorch.history import History
from l2hmc.utils.step_timer import StepTimer
from l2hmc.utils.console import is_interactive, console

from rich.table import Table
from rich.live import Live

import logging

log = logging.getLogger(__name__)


Tensor = torch.Tensor


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

        self.history = BaseHistory(steps=steps)
        self.eval_history = BaseHistory()
        evals_per_step = self.dynamics.config.nleapfrog * steps.log
        self.timer = StepTimer(evals_per_step=evals_per_step)

    def train_step(self, inputs: tuple[Tensor, float]) -> tuple[Tensor, dict]:
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
            'loss': loss.detach().cpu().numpy(),
        }
        for key, val in metrics.items():
            record[key] = val

        return to_u1(x_out).detach(), record

    def eval_step(self, inputs: tuple[Tensor, float]) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        with torch.no_grad():
            xinit = xinit.to(self.accelerator.device)
            xout, metrics = self.dynamics((to_u1(xinit), beta))
            xprop = to_u1(metrics.pop('mc_states').proposed.x)
            loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
            metrics.update({'loss': loss.cpu().numpy()})

        return to_u1(xout).detach(), metrics

    def metrics_to_numpy(
            self,
            metrics: dict[str, Tensor | list | np.ndarray]
    ) -> dict[str, Tensor | list | np.ndarray]:
        for key, val in metrics.items():
            if isinstance(val, dict):
                metrics_ = self.metrics_to_numpy(val)
                metrics.update({f'{key}/{k}': v for k, v in metrics_.items()})

            else:
                if isinstance(val, list):
                    if isinstance(val[0], Tensor):
                        val = torch.stack(val)
                    elif isinstance(val[0], np.ndarray):
                        val = np.stack(val)
                    else:
                        raise ValueError(
                            f'Unexpected value encountered: {type(val)}'
                        )

                if not isinstance(val, Tensor):
                    val = torch.Tensor(val)

                metrics[key] = val.detach().cpu().numpy()

        return metrics

    def eval(
            self,
            x: Tensor = None,
            beta: float = 1.,
            skip: str | list[str] = None,
            width: int = 0,
    ) -> dict:
        summaries = []
        self.dynamics.eval()
        if isinstance(skip, str):
            skip = [skip]

        if x is None:
            x = random_angle(self.dynamics.xshape)
            x = x.reshape(x.shape[0], -1)

        should_print = lambda epoch: (epoch % self.steps.print == 0)  # noqa
        xarr = []
        summaries = []
        tables = {}
        table = Table(collapse_padding=True,
                      # safe_box=interactive,
                      row_styles=['dim', 'none'])
        with Live(table, console=console, screen=False) as live:
            if is_interactive() and width > 0:
                live.console.width = width

            for step in range(self.steps.test):
                self.timer.start()
                x, metrics = self.eval_step((x, beta))
                dt = self.timer.stop()
                xarr.append(x)
                loss = metrics.pop('loss').numpy()
                record = {'step': step, 'dt': dt, 'loss': loss}
                record.update(self.metrics_to_numpy(metrics))
                avgs = self.eval_history.update(record)
                summary = summarize_dict(avgs)
                summaries.append(summary)
                if step == 0:
                    for key in avgs.keys():
                        table.add_column(str(key), justify='center')
                if step % self.steps.print == 0:
                    table.add_row(*[f'{v:5}' for _, v in avgs.items()])

            tables[str(0)] = table

        return {
            'xarr': xarr,
            'history': self.eval_history,
            'summaries': summaries,
            'tables': tables,
        }

    def train(
            self,
            x: Tensor = None,
            beta: float = 1.,
            skip: str | list[str] = None,
            save_x: bool = False,
            width: int = 0,
            # keep: str | list[str] = None,
    ) -> dict:
        # x = xinit
        summaries = []
        self.dynamics.train()
        if isinstance(skip, str):
            skip = [skip]
        if x is None:
            x = random_angle(self.dynamics.xshape, requires_grad=True)
            x = x.reshape(x.shape[0], -1)

        should_log = lambda epoch: (epoch % self.steps.log == 0)      # noqa
        should_print = lambda epoch: (epoch % self.steps.print == 0)  # noqa

        xarr = []
        summaries = []
        colors = 10 * ['red', 'yellow', 'green', 'blue', 'magenta', 'cyan']
        # skip = ['FORWARD', 'BACKWARD']
        # table = Table(expand=True,
        #               highlight=True,
        #               row_styles=['dim', 'none'])
        #               show_footer=False,
        tables = {}
        era = 0
        # interactive = is_interactive()
        for era in range(self.steps.nera):
            table = Table(collapse_padding=True,
                          # safe_box=interactive,
                          row_styles=['dim', 'none'])
            with Live(table, console=console, screen=False) as live:
                if is_interactive() and width > 0:
                    live.console.width = width

                estart = time.time()
                for epoch in range(self.steps.nepoch):
                    self.timer.start()
                    x, metrics = self.train_step((x, beta))
                    dt = self.timer.stop()
                    if should_print(epoch) or should_log(epoch):
                        record = {'era': era, 'epoch': epoch, 'dt': dt}
                        if save_x:
                            xarr.append(x.detach().cpu())

                        # Update metrics with train step metrics, tmetrics
                        record.update(self.metrics_to_numpy(metrics))
                        avgs = self.history.update(record)
                        summary = summarize_dict(avgs)
                        summaries.append(summary)
                        if epoch == 0:
                            for idx, key in enumerate(avgs.keys()):
                                table.add_column(str(key).upper(),
                                                 justify='center',
                                                 style=colors[idx])
                        if should_print(epoch):
                            table.add_row(*[f'{v:5}' for _, v in avgs.items()])

                live.console.log(
                    f'Era {era} took: {time.time() - estart:<3.2g}s',
                )
                live.console.log(
                    f'Avgs over last era:\n {self.history.era_summary(era)}',
                )

            tables[str(era)] = table

        return {
            'xarr': xarr,
            'summaries': summaries,
            'history': self.history,
            'tables': tables,
        }
