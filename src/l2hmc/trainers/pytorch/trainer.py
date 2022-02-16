"""
trainer.py

Implements methods for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import asdict
import time
from typing import Callable

from accelerate import Accelerator
from accelerate.utils import extract_model_from_parallel
import numpy as np
import torch
from torch import optim
from l2hmc.loss.pytorch.loss import LatticeLoss

from l2hmc.configs import Steps, AnnealingSchedule, DynamicsConfig
from l2hmc.dynamics.pytorch.dynamics import Dynamics, random_angle, to_u1
from l2hmc.utils.history import summarize_dict, BaseHistory
from l2hmc.utils.step_timer import StepTimer
from l2hmc.utils.console import is_interactive, console

from rich.table import Table
from rich.live import Live

import logging

log = logging.getLogger(__name__)


Tensor = torch.Tensor


def add_columns(avgs: dict, table: Table) -> Table:
    for key in avgs.keys():
        if key == 'loss':
            table.add_column(str(key),
                             justify='center',
                             style='green')
        elif key == 'dt':
            table.add_column(str(key),
                             justify='center',
                             style='red')

        elif key == 'acc':
            table.add_column(str(key),
                             justify='center',
                             style='magenta')
        else:
            table.add_column(str(key),
                             justify='center')

    return table


class Trainer:
    def __init__(
            self,
            steps: Steps,
            dynamics: Dynamics,
            optimizer: optim.Optimizer,
            schedule: AnnealingSchedule,
            accelerator: Accelerator,
            loss_fn: Callable = LatticeLoss,
            aux_weight: float = 0.0,
            keep: str | list[str] = None,
            skip: str | list[str] = None,
            dynamics_config: DynamicsConfig = None,
    ) -> None:
        self.steps = steps
        self.dynamics = dynamics
        self.optimizer = optimizer
        self.schedule = schedule
        self.loss_fn = loss_fn
        self.aux_weight = aux_weight
        self._with_cuda = torch.cuda.is_available()
        self.accelerator = accelerator
        self.keep = [keep] if isinstance(keep, str) else keep
        self.skip = [skip] if isinstance(skip, str) else skip
        if dynamics_config is None:
            cfg = extract_model_from_parallel(self.dynamics).config
            dynamics_config = DynamicsConfig(**asdict(cfg))

        self.dynamics_config = dynamics_config
        self.xshape = dynamics_config.xshape
        self.nlf = dynamics_config.nleapfrog

        self.history = BaseHistory(steps=steps)
        self.eval_history = BaseHistory()
        evals_per_step = self.nlf * steps.log
        # evals_per_step = self.dynamics.config.nleapfrog * steps.log
        self.timer = StepTimer(evals_per_step=evals_per_step)

    def draw_x(self) -> Tensor:
        x = random_angle(self.xshape)
        x = x.reshape(x.shape[0], -1)
        return x

    def metric_to_numpy(
            self,
            metric: Tensor | list | np.ndarray,
    ) -> np.ndarray:
        if isinstance(metric, list):
            if isinstance(metric[0], Tensor):
                metric = torch.stack(metric)
            elif isinstance(metric[0], np.ndarray):
                metric = np.stack(metric)
            else:
                raise ValueError(
                    f'Unexpected value encountered: {type(metric)}'
                )

        if not isinstance(metric, Tensor):
            metric = torch.Tensor(metric)

        return metric.detach().cpu().numpy()

    def metrics_to_numpy(
            self,
            metrics: dict[str, Tensor | list | np.ndarray]
    ) -> dict[str, Tensor | list | np.ndarray]:
        for key, val in metrics.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    metrics[f'{key}/{k}'] = self.metric_to_numpy(v)
            else:
                try:
                    metrics[key] = self.metric_to_numpy(val)
                except ValueError:
                    log.warning(
                        f'Error converting metrics[{key}] to numpy. Skipping!'
                    )
                    continue

        return metrics

    def eval_step(self, inputs: tuple[Tensor, float]) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        xinit = xinit.to(self.accelerator.device)
        xout, metrics = self.dynamics((to_u1(xinit), beta))
        xprop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        metrics.update({'loss': loss.detach().cpu().numpy()})

        return to_u1(xout).detach(), metrics

    def eval(
            self,
            beta: float = None,
            x: Tensor = None,
            skip: str | list[str] = None,
            width: int = 150,
    ) -> dict:
        summaries = []
        self.dynamics.eval()
        if isinstance(skip, str):
            skip = [skip]

        if beta is None:
            beta = self.schedule.beta_final

        if x is None:
            x = random_angle(self.xshape)
            x = x.reshape(x.shape[0], -1)

        xarr = []
        summaries = []
        tables = {}
        table = Table(row_styles=['dim', 'none'])
        # screen = (not is_interactive())

        with Live(table, console=console, screen=False) as live:
            if width is not None and width > 0:
                live.console.width = width

            for step in range(self.steps.test):
                self.timer.start()
                x, metrics = self.eval_step((x, beta))
                dt = self.timer.stop()
                xarr.append(x)
                loss = metrics.pop('loss')
                record = {'step': step, 'dt': dt, 'loss': loss}
                record.update(self.metrics_to_numpy(metrics))
                avgs = self.eval_history.update(record)
                summary = summarize_dict(avgs)
                summaries.append(summary)
                if step == 0:
                    table = add_columns(avgs, table)
                # if step % self.steps.print == 0:
                if self.should_print(step):
                    table.add_row(*[f'{v:5}' for _, v in avgs.items()])

            tables[str(0)] = table

        return {
            'xarr': xarr,
            'history': self.eval_history,
            'summaries': summaries,
            'tables': tables,
        }

    def should_log(self, epoch):
        return (
            epoch % self.steps.log == 0
            and self.accelerator.is_local_main_process
        )

    def should_print(self, epoch):
        return (
            epoch % self.steps.print == 0
            and self.accelerator.is_local_main_process
        )

    def train_step(self, inputs: tuple[Tensor, float]) -> tuple[Tensor, dict]:
        x_init, beta = inputs
        x_init = x_init.to(self.accelerator.device)

        x_out, metrics = self.dynamics((to_u1(x_init), beta))
        x_prop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=x_init, x_prop=x_prop, acc=metrics['acc'])

        if self.aux_weight > 0:
            yinit = to_u1(self.draw_x())
            _, metrics_ = self.dynamics((yinit, beta))
            yprop = to_u1(metrics_.pop('mc_states').proposed.x)
            aux_loss = self.aux_weight * self.loss_fn(x_init=yinit,
                                                      x_prop=yprop,
                                                      acc=metrics_['acc'])
            loss = (loss + aux_loss) / (1. + self.aux_weight)

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

    def train(
            self,
            x: Tensor = None,
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
            x = random_angle(self.xshape, requires_grad=True)
            x = x.reshape(x.shape[0], -1)

        xarr = []
        summaries = []
        # colors = 10 * ['red', 'yellow', 'green', 'blue', 'magenta', 'cyan']
        # skip = ['FORWARD', 'BACKWARD']
        # table = Table(expand=True,
        #               highlight=True,
        #               row_styles=['dim', 'none'])
        #               show_footer=False,
        tables = {}
        era = 0
        # interactive = is_interactive()
        for era in range(self.steps.nera):
            beta = self.schedule.betas[str(era)]
            console.rule(f'ERA: {era}, BETA: {beta}')
            table = Table(collapse_padding=True,
                          row_styles=['dim', 'none'])
            with Live(table, console=console, screen=False) as live:
                if is_interactive() and width > 0:
                    live.console.width = width

                estart = time.time()
                for epoch in range(self.steps.nepoch):
                    self.timer.start()
                    x, metrics = self.train_step((x, beta))
                    dt = self.timer.stop()
                    if self.should_print(epoch) or self.should_log(epoch):
                        if save_x:
                            xarr.append(x.detach().cpu())

                        # Update metrics with train step metrics, tmetrics
                        record = {
                            'era': era, 'epoch': epoch,
                            'beta': beta, 'dt': dt
                        }
                        record.update(self.metrics_to_numpy(metrics))
                        avgs = self.history.update(record)
                        summary = summarize_dict(avgs)
                        summaries.append(summary)
                        if epoch == 0:
                            table = add_columns(avgs, table)
                        if self.should_print(epoch):
                            table.add_row(*[f'{v}' for _, v in avgs.items()])

                live.console.log(
                    f'Era {era} took: {time.time() - estart:<5g}s',
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
