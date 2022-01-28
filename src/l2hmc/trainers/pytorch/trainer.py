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
from l2hmc.utils.history import History, summarize_dict
from l2hmc.utils.step_timer import StepTimer
from l2hmc.utils.console import console

from rich.table import Table
from rich.live import Live


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

        self.history = History(steps=steps)
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

    def train(
            self,
            x: Tensor = None,
            beta: float = 1.,
            skip: str | list[str] = None,
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

        xdict = {}
        summaries = []
        # skip = ['FORWARD', 'BACKWARD']
        styles = {
            'ERA': 'progress.elapsed',
            'EPOCH': 'progress.percentage',
            'LOSS': 'magenta',
            'ACC': 'green',
            'DT': 'red',
            'ACC_MASK': 'white',
        }
        # table = Table(expand=True,
        #               highlight=True,
        #               show_footer=False,
        #               row_styles=['dim', 'none'])
        tables = {}
        for era in range(self.steps.nera):
            xdict[str(era)] = x
            estart = time.time()
            console.rule(f'ERA: {era}')
            table = Table(expand=True,
                          highlight=True,
                          show_footer=False,
                          row_styles=['dim', 'none'])
            with Live(table, screen=False, auto_refresh=False) as live:
                for epoch in range(self.steps.nepoch):
                    self.timer.start()
                    x, metrics = self.train_step((x, beta))
                    dt = self.timer.stop()
                    if should_print(epoch) or should_log(epoch):
                        record = {'era': era, 'epoch': epoch, 'dt': dt}

                        # Update metrics with train step metrics, tmetrics
                        record.update(self.metrics_to_numpy(metrics))
                        avgs = self.history.update(record)
                        summary = summarize_dict(avgs)
                        summaries.append(summary)
                        if epoch == 0:
                            for h in [str(i).upper() for i in avgs.keys()]:
                                cargs = {'header': h, 'justify': 'center'}
                                table.add_column(**cargs)

                        table.add_row(*[f'{v:5}' for _, v in avgs.items()])
                        live.refresh()

                live.console.rule(
                    f'Era {era} took: {time.time() - estart:<3.2g}s',
                )
                live.console.log(
                    f'Avgs over last era:\n {self.history.era_summary(era)}',
                )
                live.console.rule()

            tables[str(era)] = table

        return {
            'xdict': xdict,
            'summaries': summaries,
            'history': self.history,
            'tables': tables,
        }
