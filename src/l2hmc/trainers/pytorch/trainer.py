"""
trainer.py

Implements methods for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import asdict
import logging
import os
from pathlib import Path
import wandb
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


WIDTH = int(os.environ.get('COLUMNS', 150))

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
            aux_weight: float = 0.0,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
            dynamics_config: Optional[DynamicsConfig] = None,
    ) -> None:
        self.steps = steps
        self.dynamics = dynamics
        self.optimizer = optimizer
        self.schedule = schedule
        self.loss_fn = loss_fn
        self.aux_weight = aux_weight
        self.clip_norm = lr_config.clip_norm
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

    def hmc_step(
            self,
            inputs: tuple[Tensor, float],
            eps: Tensor,
    ) -> tuple[Tensor, dict]:
        xi, beta = inputs
        xi = to_u1(xi).to(self.accelerator.device)
        beta = torch.tensor(beta).to(self.accelerator.device)
        eps = eps.to(self.accelerator.device)
        # beta = torch.tensor(beta).to(self.accelerator.device)
        xo, metrics = self._dynamics.apply_transition_hmc(  # type: ignore
            (xi, beta), eps=eps
        )
        xo = to_u1(xo)
        xp = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xi, x_prop=xp, acc=metrics['acc'])
        lmetrics = self.loss_fn.lattice_metrics(xinit=xi, xout=xo)
        metrics.update(lmetrics)
        metrics.update({'loss': loss.detach().cpu().numpy()})

        return xo.detach(), metrics

    def eval_step(self, inputs: tuple[Tensor, float]) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        xinit = xinit.to(self.accelerator.device)
        xout, metrics = self.dynamics((to_u1(xinit), beta))
        xprop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        lmetrics = self.loss_fn.lattice_metrics(xinit=to_u1(xinit),
                                                xout=to_u1(xout))
        metrics.update(lmetrics)
        metrics.update({'loss': loss.detach().cpu().numpy()})

        return to_u1(xout).detach(), metrics

    def eval(
            self,
            beta: Optional[float] = None,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            run: Optional[Any] = None,
            writer: Optional[Any] = None,
            job_type: Optional[str] = 'eval',
            nchains: Optional[int] = -1,
            eps: Optional[Tensor] = None,
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

        if eps is None and str(job_type).lower() == 'hmc':
            eps = torch.tensor(0.1)
            log.warn(
                'Step size `eps` not specified for HMC! Using default: 0.1'
            )

        assert job_type in ['eval', 'hmc']

        def eval_fn(z):
            if job_type == 'hmc':
                assert eps is not None
                return self.hmc_step(z, eps)
            return self.eval_step(z)

        summaries = []
        tables = {}
        table = Table(row_styles=['dim', 'none'], box=box.HORIZONTALS)
        nprint = max((20, self.steps.test // 20))
        nlog = max((1, min((10, self.steps.test))))
        if nlog <= self.steps.test:
            nlog = min(10, max(1, self.steps.test // 100))

        assert job_type in ['eval', 'hmc']
        timer = self.timers[job_type]
        history = self.histories[job_type]

        log.warning(f'x.shape (original): {x.shape}')
        if nchains is not None:
            if isinstance(nchains, int) and nchains > 0:
                x = x[:nchains]

        assert isinstance(x, Tensor)
        assert isinstance(beta, float)
        log.warning(f'x[:nchains].shape: {x.shape}')

        if run is not None:
            run.config.update({job_type: {'beta': beta, 'xshape': x.shape}})

        with Live(table, screen=False, auto_refresh=False) as live:
            if WIDTH is not None and WIDTH > 0:
                live.console.width = WIDTH

            for step in range(self.steps.test):
                timer.start()
                x, metrics = eval_fn((x, beta))
                dt = timer.stop()
                if step % nlog == 0 or step % nprint == 0:
                    record = {
                        'step': step, 'beta': beta, 'dt': dt,
                    }
                    avgs, summary = self.record_metrics(run=run,
                                                        step=step,
                                                        record=record,
                                                        writer=writer,
                                                        metrics=metrics,
                                                        history=history,
                                                        job_type=job_type)
                    if avgs.get('acc', 1.0) < 1e-5:
                        log.warning('Chains are stuck! Re-drawing x !')
                        x = random_angle(self.xshape)
                        x = x.reshape(x.shape[0], -1)

                    summaries.append(summary)
                    if step == 0:
                        table = add_columns(avgs, table)

                    if step % nprint == 0:
                        table.add_row(*[f'{v:5}' for _, v in avgs.items()])
                        live.refresh()

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

    def record_metrics(
            self,
            metrics: dict,
            job_type: str,
            step: Optional[int] = None,
            record: Optional[dict] = None,
            run: Optional[Any] = None,
            writer: Optional[Any] = None,
            history: Optional[BaseHistory] = None,
            model: Optional[Module] = None,
            optimizer: Optional[optim.Optimizer] = None,
    ):
        record = {} if record is None else record

        if step is not None:
            record.update({f'{job_type}_step': step})

        record.update({
            'loss': metrics.get('loss', None),
            'dQint': metrics.get('dQint', None),
            'dQsin': metrics.get('dQsin', None),
        })

        record.update(self.metrics_to_numpy(metrics))
        if history is not None:
            avgs = history.update(record)
        else:
            avgs = {k: v.mean() for k, v in record.items()}

        summary = summarize_dict(avgs)
        if writer is not None:
            assert step is not None
            update_summaries(step=step,
                             model=model,
                             writer=writer,
                             metrics=record,
                             prefix=job_type,
                             optimizer=optimizer)
            writer.flush()

        if run is not None:
            dQint = record.get('dQint', None)
            if dQint is not None:
                dQdict = {
                    f'dQint/{job_type}': {
                        'val': dQint,
                        'step': step,
                        'avg': dQint.mean(),
                    }
                }
                run.log(dQdict, commit=False)

            run.log({f'wandb/{job_type}': record}, commit=False)
            run.log({f'avgs/wandb.{job_type}': avgs})

        return avgs, summary

    def profile_step(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        xinit = to_u1(xinit).to(self.accelerator.device)
        beta = beta.to(self.accelerator.device)
        xout, metrics = self.dynamics((xinit, beta))
        xprop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        self.optimizer.zero_grad
        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(self.dynamics.parameters(),
                                         max_norm=self.clip_norm,)
        self.optimizer.step()

        return to_u1(xout).detach(), metrics

    def train_step(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        xinit = to_u1(xinit).to(self.accelerator.device)
        beta = torch.tensor(beta).to(self.accelerator.device)
        xout, metrics = self.dynamics((xinit, beta))
        xprop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        xout = to_u1(xout)

        if self.aux_weight > 0:
            yinit = to_u1(self.draw_x()).to(self.accelerator.device)
            _, metrics_ = self.dynamics((yinit, beta))
            yprop = to_u1(metrics_.pop('mc_states').proposed.x)
            aux_loss = self.aux_weight * self.loss_fn(x_init=yinit,
                                                      x_prop=yprop,
                                                      acc=metrics_['acc'])
            loss = (loss + aux_loss) / (1. + self.aux_weight)

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        # extract_model_from_parallel(self.dynamics).parameters(),
        self.accelerator.clip_grad_norm_(
            self.dynamics.parameters(),
            max_norm=self.clip_norm,
        )
        self.optimizer.step()

        metrics['loss'] = loss
        lmetrics = self.loss_fn.lattice_metrics(xinit=xinit, xout=xout)
        metrics.update(lmetrics)

        return xout.detach(), metrics

    def save_ckpt(
            self,
            era: int,
            epoch: int,
            train_dir: os.PathLike,
            metrics: Optional[dict] = None,
            run: Optional[Any] = None,
    ) -> None:
        dynamics = extract_model_from_parallel(self.dynamics)
        ckpt_dir = Path(train_dir).joinpath('checkpoints')
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        ckpt_file = ckpt_dir.joinpath(f'ckpt-{era}-{epoch}.tar')
        log.info(f'Saving checkpoint to: {ckpt_file.as_posix()}')
        # self.dynamics.save(train_dir)  # type: ignore
        xeps = {
            k: grab(v) for k, v in self.dynamics.xeps.items()  # type:ignore
        }
        veps = {
            k: grab(v) for k, v in self.dynamics.veps.items()  # type:ignore
        }
        ckpt = {
            'era': era,
            'epoch': epoch,
            'xeps': xeps,
            'veps': veps,
            'model_state_dict': self.dynamics.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if metrics is not None:
            ckpt.update(metrics)

        # torch.save(ckpt, ckpt_file)
        self.accelerator.save(ckpt, ckpt_file)
        if run is not None:
            assert run is wandb.run
            outfile = Path(train_dir).joinpath('model.pth').as_posix()
            # torch.save(self.dynamics.state_dict(), outfile.as_posix())
            self.accelerator.save(self.dynamics.state_dict(), outfile)
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(outfile)
            run.log_artifact(artifact)

    def profile(self, nsteps: int = 5) -> dict:
        self.dynamics.train()
        x = self.draw_x()
        beta = torch.tensor(1.0)
        for _ in range(nsteps):
            x, metrics = self.profile_step((x, beta))

        return metrics


    def train(
            self,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            train_dir: Optional[os.PathLike] = None,
            run: Optional[Any] = None,
            writer: Optional[Any] = None,
            # keep: str | list[str] = None,
    ) -> dict:
        skip = [skip] if isinstance(skip, str) else skip

        if train_dir is None:
            train_dir = Path(os.getcwd()).joinpath('train')
        else:
            train_dir = Path(train_dir)

        train_dir.mkdir(exist_ok=True, parents=True)

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
        rows = {}
        summaries = []
        timer = self.timers['train']
        history = self.histories['train']
        tkwargs = {
            'box': box.HORIZONTALS,
            'row_styles': ['dim', 'none'],
        }
        self.dynamics.train()
        log.warning(f'x.dtype: {x.dtype}')
        for era in range(self.steps.nera):
            table = Table(**tkwargs)
            beta = self.schedule.betas[str(era)]
            if self.accelerator.is_local_main_process:
                console.width = WIDTH
                console.print('\n')
                console.rule(
                    f'ERA: {era} / {self.steps.nera}, BETA: {beta}',
                )
                console.print('\n')

            with Live(
                    table,
                    screen=False,
                    # console=console,
                    # refresh_per_second=1,
            ) as live:
                if WIDTH is not None and WIDTH > 0:
                    live.console.width = WIDTH

                estart = time.time()
                for epoch in range(self.steps.nepoch):
                    timer.start()
                    x, metrics = self.train_step((x, beta))
                    dt = timer.stop()
                    gstep += 1
                    if self.should_print(epoch) or self.should_log(epoch):
                        record = {
                            'era': era, 'epoch': epoch, 'beta': beta, 'dt': dt,
                        }
                        avgs, summary = self.record_metrics(run=run,
                                                            step=gstep,
                                                            writer=writer,
                                                            record=record,
                                                            metrics=metrics,
                                                            job_type='train',
                                                            history=history)
                        rows[gstep] = avgs
                        summaries.append(summary)

                        if avgs.get('acc', 1.0) < 1e-5:
                            log.warning('Chains are stuck! Re-drawing x !')
                            x = random_angle(self.xshape)
                            x = x.reshape(x.shape[0], -1)

                        if epoch == 0:
                            table = add_columns(avgs, table)

                        if self.should_print(epoch):
                            table.add_row(*[f'{v}' for _, v in avgs.items()])

            tables[str(era)] = table
            if self.accelerator.is_local_main_process:
                if writer is not None:
                    model = extract_model_from_parallel(self.dynamics)
                    model = model if isinstance(model, Module) else None
                    update_summaries(step=gstep,
                                     writer=writer,
                                     model=self.dynamics,
                                     optimizer=self.optimizer)

                log.info(f'Era {era} took: {time.time() - estart:<5g}s')
                log.info('\n'.join([
                    'Avgs over last era:', f'{self.history.era_summary(era)}'
                ]))
                log.info(f'Saving checkpoint to: {train_dir}')
                ckpt_metrics = {'loss': metrics['loss']}
                st0 = time.time()
                self.save_ckpt(era, epoch, train_dir,
                               metrics=ckpt_metrics, run=run)
                log.info(f'Saving took: {time.time() - st0:<5g}s')

                log.info(
                    f'Era {era} took: {time.time() - estart:<5g}s',
                )
                log.info(
                    f'Avgs over last era:\n {self.history.era_summary(era)}',
                )

        return {
            'timer': timer,
            'rows': rows,
            'summaries': summaries,
            'history': history,
            'tables': tables,
        }
