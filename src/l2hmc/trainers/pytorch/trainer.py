"""
trainer.py

Implements methods for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import asdict
import logging
import os
from pathlib import Path
import time
from typing import Callable, Any, Optional

from accelerate import Accelerator
from accelerate.utils import extract_model_from_parallel
import numpy as np
from rich import box
from rich.live import Live
from rich.table import Table
import torch
from torch import optim

from l2hmc.configs import (
    AnnealingSchedule, DynamicsConfig, LearningRateConfig, Steps
)
from l2hmc.dynamics.pytorch.dynamics import Dynamics, random_angle, to_u1
from l2hmc.loss.pytorch.loss import LatticeLoss
from l2hmc.trackers.pytorch.trackers import update_summaries
from l2hmc.utils.console import console
from l2hmc.utils.history import BaseHistory, summarize_dict
from l2hmc.utils.step_timer import StepTimer
# from torchinfo import summary as model_summary


log = logging.getLogger(__name__)


Tensor = torch.Tensor
Module = torch.nn.modules.Module


def grab(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


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
        elif key == 'dQint':
            table.add_column(str(key),
                             justify='center',
                             style='cyan')
        elif key == 'dQsin':
            table.add_column(str(key),
                             justify='center',
                             style='yellow')

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
            accelerator: Accelerator,
            optimizer: optim.Optimizer,
            schedule: AnnealingSchedule,
            lr_config: LearningRateConfig,
            loss_fn: Callable = LatticeLoss,
            keep: str | list[str] = None,
            skip: str | list[str] = None,
            aux_weight: float = 0.0,
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
        self.lr_config = lr_config
        self._dynamics = extract_model_from_parallel(  # type: Module
            self.dynamics
        )
        self.keep = [keep] if isinstance(keep, str) else keep
        self.skip = [skip] if isinstance(skip, str) else skip
        if dynamics_config is None:
            dynamics_ = extract_model_from_parallel(self.dynamics)
            cfg = dynamics_.config  # type: ignore

            dynamics_config = DynamicsConfig(**asdict(cfg))

        self.dynamics_config = dynamics_config
        self.xshape = dynamics_config.xshape
        self.nlf = dynamics_config.nleapfrog

        self.history = BaseHistory(steps=steps)
        self.timer = StepTimer(evals_per_step=self.nlf)
        self.histories = {
            'train': self.history,
            'eval': BaseHistory(),
            'hmc': BaseHistory(),
        }
        self.timers = {
            'train': self.timer,
            'eval': StepTimer(evals_per_step=self.nlf),
            'hmc': StepTimer(evals_per_step=self.nlf)
        }

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
    ) -> dict[str, list[np.ndarray] | np.ndarray | int | float]:
        m = {}
        for key, val in metrics.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    m[f'{key}/{k}'] = self.metric_to_numpy(v)
            else:
                try:
                    m[key] = self.metric_to_numpy(val)
                except ValueError:
                    log.warning(
                        f'Error converting metrics[{key}] to numpy. Skipping!'
                    )
                    continue

        return m

    def hmc_step(self, inputs: tuple[Tensor, float]) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        xinit = to_u1(xinit).to(self.accelerator.device)
        beta = torch.tensor(beta)
        # beta = torch.tensor(beta).to(self.accelerator.device)
        xout, metrics = self._dynamics.apply_transition_hmc(  # type: ignore
            (xinit, beta)
        )
        xprop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        lmetrics = self.loss_fn.lattice_metrics(xinit=to_u1(xinit),
                                                xout=to_u1(xout))
        metrics.update(lmetrics)
        metrics.update({'loss': loss.detach().cpu().numpy()})

        return to_u1(xout).detach(), metrics

    def eval_step(self, inputs: tuple[Tensor, float]) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        xinit = xinit.to(self.accelerator.device)
        xout, metrics = self.dynamics((to_u1(xinit), beta))
        xprop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        lmetrics = self.loss_fn.lattice_metrics(xinit=to_u1(xinit),
                                                xout=to_u1(xout))
        metrics.update(lmetrics)
        # metrics.update(lattice_metrics)
        metrics.update({'loss': loss.detach().cpu().numpy()})

        return to_u1(xout).detach(), metrics

    def eval(
            self,
            beta: float = None,
            x: Tensor = None,
            skip: str | list[str] = None,
            width: int = 150,
            # eval_dir: os.PathLike = None,
            run: Optional[Any] = None,
            writer: Optional[Any] = None,
            job_type: Optional[str] = 'eval',
            # hmc: bool = False,
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

        if job_type == 'hmc':
            eval_fn = self.hmc_step
        else:
            eval_fn = self.eval_step

        summaries = []
        tables = {}
        table = Table(row_styles=['dim', 'none'], box=box.SIMPLE)
        # nlog = max(10, int(self.steps.test // 500))
        # nprint = max(10, int(self.steps.test // 20))
        nprint = self.steps.test // 20
        nlog = 10 if self.steps.test < 1000 else 20
        assert job_type in ['eval', 'hmc']
        timer = self.timers[job_type]
        history = self.histories[job_type]
        assert isinstance(beta, float)

        with Live(table, console=console, screen=False) as live:
            if width is not None and width > 0:
                live.console.width = width

            for step in range(self.steps.test):
                timer.start()
                x, metrics = eval_fn((x, beta))
                dt = timer.stop()
                if step % nlog == 0 or step % nprint == 0:
                    record = {
                        'step': step, 'dt': dt, 'loss': metrics['loss'],
                        'dQint': metrics['dQint'], 'dQsin': metrics['dQsin'],
                    }
                    record.update(self.metrics_to_numpy(metrics))
                    avgs = history.update(record)
                    summary = summarize_dict(avgs)
                    summaries.append(summary)
                    if writer is not None:
                        update_summaries(step=step,
                                         prefix=job_type,
                                         metrics=record,
                                         writer=writer)
                        writer.flush()

                    if run is not None:
                        run.log({f'wandb/{job_type}': record}, commit=False)
                        run.log({f'avgs/wandb.{job_type}': avgs})

                    if step == 0:
                        table = add_columns(avgs, table)

                    if step % nprint == 0:
                        table.add_row(*[f'{v:5}' for _, v in avgs.items()])

            tables[str(0)] = table

        return {
            'timer': timer,
            'history': history,
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
        xinit, beta = inputs
        xinit = to_u1(xinit).to(self.accelerator.device)
        xout, metrics = self.dynamics((xinit, beta))
        xout = to_u1(xout)
        xprop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])

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
        self.optimizer.step()
        metrics.update(self.loss_fn.lattice_metrics(xinit=xinit, xout=xout))
        # lmetrics = self.loss_fn.lattice_metrics(xinit=xinit, xout=xout)
        # metrics.update(lmetrics)
        metrics.update({'loss': loss.detach().cpu().numpy()})

        return xout.detach(), metrics

    def save_ckpt(self, era, epoch, train_dir, **kwargs) -> None:
        dynamics = extract_model_from_parallel(self.dynamics)
        ckpt_dir = Path(train_dir).joinpath('checkpoints')
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        ckpt_file = ckpt_dir.joinpath(f'ckpt-{era}-{epoch}.tar')
        log.info(f'Saving checkpoint to: {ckpt_file.as_posix()}')
        dynamics.save(train_dir)  # type: ignore
        xeps = {
            k: grab(v) for k, v in dynamics.xeps.items()  # type:ignore
        }
        veps = {
            k: grab(v) for k, v in dynamics.veps.items()  # type:ignore
        }
        torch.save({
            'era': era,
            'epoch': epoch,
            'xeps': xeps,
            'veps': veps,
            'model_state_dict': dynamics.state_dict(),  # type: ignore
            'optimizer_state_dict': self.optimizer.state_dict(),
            **kwargs,
        }, ckpt_file)

    def train(
            self,
            x: Tensor = None,
            skip: str | list[str] = None,
            width: int = None,
            train_dir: os.PathLike = None,
            run: Any = None,
            writer: Any = None,
            # keep: str | list[str] = None,
    ) -> dict:
        skip = [skip] if isinstance(skip, str) else skip
        if width is None:
            width = max((150, int(os.environ.get('COLUMNS', 150))))

        if isinstance(skip, str):
            skip = [skip]

        if x is None:
            x = random_angle(self.xshape, requires_grad=True)
            x = x.reshape(x.shape[0], -1)

        era = 0
        gstep = 0
        epoch = 0
        tables = {}
        metrics = {}
        summaries = []
        timer = self.timers['train']
        history = self.histories['train']
        tkwargs = {
            'box': box.SIMPLE,
            'row_styles': ['dim', 'none'],
        }
        self.dynamics.train()
        for era in range(self.steps.nera):
            table = Table(**tkwargs)
            beta = self.schedule.betas[str(era)]
            if self.accelerator.is_local_main_process:
                console.width = width
                console.rule(f'ERA: {era}, BETA: {beta}')

            with Live(
                    table,
                    screen=False,
                    console=console,
                    # refresh_per_second=1,
            ) as live:
                estart = time.time()
                for epoch in range(self.steps.nepoch):
                    timer.start()
                    x, metrics = self.train_step((x, beta))
                    gstep += 1
                    dt = timer.stop()
                    if self.should_print(epoch) or self.should_log(epoch):
                        record = {
                            'era': era, 'epoch': epoch,
                            'beta': beta, 'dt': dt,
                            'loss': metrics['loss'],
                            'dQint': metrics['dQint'],
                            'dQsin': metrics['dQsin'],
                        }
                        # Update metrics with train step metrics, tmetrics
                        record.update(self.metrics_to_numpy(metrics))
                        if writer is not None:
                            update_summaries(writer=writer,
                                             # model=dynamics,  # type:ignore
                                             step=gstep,
                                             metrics=record,
                                             prefix='train')
                            writer.flush()

                        avgs = history.update(record)
                        summary = summarize_dict(avgs)
                        summaries.append(summary)
                        if run is not None:
                            run.log({'wandb/train': record}, commit=False)
                            run.log({'avgs/wandb.train': avgs})

                        if epoch == 0:
                            table = add_columns(avgs, table)
                        if self.should_print(epoch):
                            table.add_row(*[f'{v}' for _, v in avgs.items()])

            tables[str(era)] = table
            if self.accelerator.is_local_main_process:
                if writer is not None:
                    model = extract_model_from_parallel(self.dynamics)
                    model = model if isinstance(model, Module) else None
                    update_summaries(writer=writer, step=gstep, model=model)

                self.save_ckpt(era, epoch, train_dir, loss=metrics['loss'])
                live.console.log(
                    f'Era {era} took: {time.time() - estart:<5g}s\n',
                    f'Avgs over last era:\n {self.history.era_summary(era)}',
                )

        return {
            # 'xarr': xarr,
            'summaries': summaries,
            'history': self.history,
            'tables': tables,
        }
