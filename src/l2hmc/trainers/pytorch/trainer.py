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
from torch.utils.tensorboard.writer import SummaryWriter

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
        self.eval_history = BaseHistory()
        self.hmc_history = BaseHistory()
        # evals_per_step = self.nlf * steps.log
        # evals_per_step = self.dynamics.config.nleapfrog * steps.log
        self.timer = StepTimer(evals_per_step=self.nlf)
        self.eval_timer = StepTimer(evals_per_step=self.nlf)

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
        nlog = max(1, int(self.steps.test // 100))
        nprint = max(1, int(self.steps.test // 20))

        with Live(table, console=console, screen=True) as live:
            if width is not None and width > 0:
                live.console.width = width

            for step in range(self.steps.test):
                self.eval_timer.start()
                x, metrics = eval_fn((x, beta))
                dt = self.eval_timer.stop()
                if step % nlog == 0 or step % nprint == 0:
                    record = {
                        'step': step, 'dt': dt, 'loss': metrics['loss'],
                        'dQint': metrics['dQint'], 'dQsin': metrics['dQsin'],
                    }
                    record.update(self.metrics_to_numpy(metrics))
                    if writer is not None:
                        update_summaries(step=step,
                                         prefix=job_type,
                                         metrics=record,
                                         writer=writer)
                        writer.flush()
                    if run is not None:
                        run.log({f'wandb/{job_type}': record})

                    avgs = self.eval_history.update(record)
                    summary = summarize_dict(avgs)
                    summaries.append(summary)
                    if step == 0:
                        table = add_columns(avgs, table)

                    if step % nprint == 0:
                        table.add_row(*[f'{v:5}' for _, v in avgs.items()])

            tables[str(0)] = table

        return {
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

    def setup_SummaryWriter(self, outdir: os.PathLike = None):
        """Setup SummaryWriter for TensorBoard summaries."""
        if self.accelerator.is_local_main_process:
            return SummaryWriter(
                Path(outdir).as_posix() if outdir is not None else None
            )
        return None

    def train(
            self,
            x: Tensor = None,
            skip: str | list[str] = None,
            width: int = 80,
            train_dir: os.PathLike = None,
            run: Any = None,
            writer: Any = None,
            # keep: str | list[str] = None,
    ) -> dict:
        summaries = []
        self.dynamics.train()
        if isinstance(skip, str):
            skip = [skip]
        if x is None:
            x = random_angle(self.xshape, requires_grad=True)
            x = x.reshape(x.shape[0], -1)

        era = 0
        epoch = 0
        tables = {}
        metrics = {}
        summaries = []
        gstep = 0
        for era in range(self.steps.nera):
            beta = self.schedule.betas[str(era)]
            if self.accelerator.is_local_main_process:
                console.rule(f'ERA: {era}, BETA: {beta}')

            table = Table(row_styles=['dim', 'none'], box=box.SIMPLE)
            with Live(table, console=console, screen=False) as live:
                if width != 0:
                    live.console.width = width

                estart = time.time()
                for epoch in range(self.steps.nepoch):
                    self.timer.start()
                    x, metrics = self.train_step((x, beta))
                    gstep += 1
                    dt = self.timer.stop()
                    if self.should_print(epoch) or self.should_log(epoch):
                        record = {
                            'era': era, 'epoch': epoch, 'beta': beta, 'dt': dt
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

                        if run is not None:
                            run.log({'wandb/train': record})

                        avgs = self.history.update(record)
                        # wandb.log({'train/avgs': avgs})
                        summary = summarize_dict(avgs)
                        summaries.append(summary)
                        if epoch == 0:
                            table = add_columns(avgs, table)
                        if self.should_print(epoch):
                            table.add_row(*[f'{v}' for _, v in avgs.items()])

            if self.accelerator.is_local_main_process:
                if writer is not None:
                    model = extract_model_from_parallel(self.dynamics)
                    update_summaries(writer=writer, step=gstep,
                                     model=model)  # type: ignore

                self.save_ckpt(era, epoch, train_dir, loss=metrics['loss'])
                live.console.log(
                    f'Era {era} took: {time.time() - estart:<5g}s',
                )
                live.console.log(
                    f'Avgs over last era:\n {self.history.era_summary(era)}',
                )

            tables[str(era)] = table

        return {
            # 'xarr': xarr,
            'summaries': summaries,
            'history': self.history,
            'tables': tables,
        }
