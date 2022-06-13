"""
trainer.py

Implements methods for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from collections import defaultdict
from contextlib import nullcontext
# from dataclasses import asdict
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional

# from accelerate import Accelerator
# from accelerate.utils import extract_model_from_parallel
import numpy as np
from rich import box
from rich.live import Live
from rich.table import Table
import torch
import horovod.torch as hvd
from torch import optim
import wandb

from l2hmc.configs import (
    Steps,
    DynamicsConfig,
    AnnealingSchedule,
    LearningRateConfig,
)
from l2hmc.dynamics.pytorch.dynamics import Dynamics, random_angle
import l2hmc.group.pytorch.group as g
from l2hmc.loss.pytorch.loss import LatticeLoss
from l2hmc.trackers.pytorch.trackers import update_summaries
from l2hmc.utils.history import BaseHistory, summarize_dict
from l2hmc.utils.rich import add_columns, console
from l2hmc.utils.step_timer import StepTimer

# from mpi4py import MPI
# from torchinfo import summary as model_summary


WIDTH = int(os.environ.get('COLUMNS', 150))

log = logging.getLogger(__name__)


Tensor = torch.Tensor
Module = torch.nn.modules.Module


# LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
SIZE = hvd.size()
RANK = hvd.rank()
LOCAL_RANK = hvd.local_rank()

WITH_CUDA = torch.cuda.is_available()


def grab(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def init_weights(layer: torch.nn.Module):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)


def clamp_grad_backward_hook(model: torch.nn.Module, clip: float):
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))


class Trainer:
    # TODO: Add methods for:
    #   1. Saving / loading dynamics ? Or move to Experiment?
    #   2. Saving / loading History objects w/ dynamics?
    #   3. Resetting Timers + History
    #   4. Plotting history / running analysis directly?
    def __init__(
            self,
            steps: Steps,
            dynamics: Dynamics,
            optimizer: optim.Optimizer,
            schedule: AnnealingSchedule,
            lr_config: LearningRateConfig,
            loss_fn: Callable = LatticeLoss,
            aux_weight: Optional[float] = None,
            # accelerator: Optional[Accelerator] = None,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
            dynamics_config: Optional[DynamicsConfig] = None,
            # rank: Optional[int] = None,
    ) -> None:
        self.steps = steps
        self.dynamics = dynamics
        self.optimizer = optimizer
        self.schedule = schedule
        self.loss_fn = loss_fn
        self.aux_weight = aux_weight if aux_weight is not None else 0.0
        self.clip_norm = lr_config.clip_norm
        self._with_cuda = torch.cuda.is_available()
        if self._with_cuda:
            dynamics.cuda()

        hvd.broadcast_parameters(self.dynamics.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        self.optimizer = hvd.DistributedOptimizer(
                self.optimizer,
                named_parameters=self.dynamics.named_parameters(),
                op=hvd.Average,
                # op=hvd.Adasum if cfg.use_adasum else hvd.Average,
        )
        # if self.clip_norm:
        #     clamp_grad_backward_hook(self.dynamics, clip=self.clip_norm)
        # self.accelerator = accelerator
        # self.accelerator = accelerator if accelerator is not None else None
        # self.rank = self.accelerator.local_process_index
        # self.rank = rank if rank is not None else 0
        # self.rank = LOCAL_RANK
        # self.device = self.accelerator.device
        # self.rank = self.accelerator.local_process_index
        self.device = hvd.local_rank()
        self.rank = hvd.local_rank()
        self.lr_config = lr_config
        self.keep = [keep] if isinstance(keep, str) else keep
        self.skip = [skip] if isinstance(skip, str) else skip
        # self._dynamics = self.accelerator.unwrap_model(self.dynamics)
        if dynamics_config is None:
            dynamics_config = self.dynamics.config
        assert isinstance(dynamics_config, DynamicsConfig)
        self.nlf = dynamics_config.nleapfrog
        self.xshape = dynamics_config.xshape
        self.dynamics_config = dynamics_config
        self.verbose = self.dynamics_config.verbose
        if self.dynamics_config.group == 'U1':
            self.g = g.U1Phase()
        elif self.dynamics_config.group == 'SU3':
            self.g = g.SU3()
        else:
            raise ValueError('Unexpected value for `dynamics_config.group`')

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
        return self.g.random(list(self.xshape)).flatten(1)

    def reset_optimizer(self):
        log.warning('Resetting optimizer state!')
        self.optimizer.state = defaultdict(dict)

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
            eps: float,
            nleapfrog: Optional[int] = None,
    ) -> tuple[Tensor, dict]:
        self.dynamics.eval()
        xi, beta = inputs
        beta = torch.tensor(beta)
        if WITH_CUDA:
            xi, beta = xi.cuda(), beta.cuda()

        xo, metrics = self.dynamics.apply_transition_hmc(
            (xi, beta), eps=eps, nleapfrog=nleapfrog,
        )
        xp = metrics.pop('mc_states').proposed.x
        loss = self.loss_fn(x_init=xi, x_prop=xp, acc=metrics['acc'])
        if self.verbose:
            lmetrics = self.loss_fn.lattice_metrics(xinit=xi, xout=xo)
            metrics.update(lmetrics)

        metrics.update({'loss': loss.detach().cpu().numpy()})
        return xo.detach(), metrics

    def eval_step(
            self,
            inputs: tuple[Tensor, float]
    ) -> tuple[Tensor, dict]:
        self.dynamics.eval()
        xinit, beta = inputs
        if WITH_CUDA:
            xinit, beta = xinit.cuda(), torch.tensor(beta).cuda()
        # xinit = xinit.to(self.device)
        xout, metrics = self.dynamics((xinit, beta))
        xout = self.g.compat_proj(xout)
        xprop = metrics.pop('mc_states').proposed.x
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        if self.verbose:
            lmetrics = self.loss_fn.lattice_metrics(xinit=xinit, xout=xout)
            metrics.update(lmetrics)

        metrics.update({'loss': loss.detach().cpu().numpy()})

        return xout.detach(), metrics

    def eval(
            self,
            beta: Optional[float] = None,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            run: Optional[Any] = None,
            writer: Optional[Any] = None,
            job_type: Optional[str] = 'eval',
            nchains: Optional[int] = None,
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> dict:
        """Evaluate dynamics."""
        summaries = []
        self.dynamics.eval()
        if isinstance(skip, str):
            skip = [skip]

        if beta is None:
            beta = self.schedule.beta_final

        if x is None:
            x = self.g.random(list(self.xshape))
            x = x.reshape(x.shape[0], -1)

        if eps is None and str(job_type).lower() == 'hmc':
            eps = self.dynamics_config.eps_hmc
            assert eps is not None
            log.warn(f'Using step size eps: {eps:.4f} for generic HMC')
            # log.warn(
            #     'Step size `eps` not specified for HMC! Using default: 0.1'
            # )

        assert job_type in ['eval', 'hmc']

        def eval_fn(z):
            if job_type == 'hmc':
                assert eps is not None
                return self.hmc_step(z, eps=eps, nleapfrog=nleapfrog)
            return self.eval_step(z)

        tables = {}
        table = Table(row_styles=['dim', 'none'], box=box.HORIZONTALS)
        nprint = max(1, self.steps.test // 50)
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

        with Live(console=console) as live:
            live.update(table)
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
                                                        job_type=job_type)
                    summaries.append(summary)
                    if step == 0:
                        table = add_columns(avgs, table)
                    else:
                        table.add_row(*[f'{v}' for _, v in avgs.items()])

                    if avgs.get('acc', 1.0) < 1e-5:
                        self.reset_optimizer()
                        live.console.log('Chains are stuck! Redrawing x')
                        x = self.g.random(list(x.shape))

            # console.log(table)
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
            and LOCAL_RANK == 0
            # and self.rank == 0
            # and self.accelerator.is_local_main_process
        )

    def should_print(self, epoch):
        return (
            epoch % self.steps.print == 0
            and LOCAL_RANK == 0
            # and self.rank == 0
            # and self.accelerator.is_local_main_process
        )

    def record_metrics(
            self,
            metrics: dict,
            job_type: str,
            step: Optional[int] = None,
            record: Optional[dict] = None,
            run: Optional[Any] = None,
            writer: Optional[Any] = None,
            model: Optional[Module] = None,
            optimizer: Optional[optim.Optimizer] = None,
    ):
        record = {} if record is None else record
        assert job_type in ['train', 'eval', 'hmc']
        history = self.histories[job_type]

        if step is not None:
            record.update({f'{job_type}_step': step})

        record.update({
            'loss': metrics.get('loss', None),
            # 'dQint': metrics.get('dQint', None),
            # 'dQsin': metrics.get('dQsin', None),
        })

        record.update(self.metrics_to_numpy(metrics))
        # if history is not None:
        #     avgs = history.update(record)
        # else:
        #     avgs = {k: v.mean() for k, v in record.items() if v is not None}

        avgs = history.update(record)
        summary = summarize_dict(avgs)
        # if step is not None:
        if writer is not None and self.verbose:
            assert step is not None
            update_summaries(step=step,
                             model=model,
                             writer=writer,
                             metrics=record,
                             prefix=job_type,
                             optimizer=optimizer)
            writer.flush()

        if run is not None and self.verbose:
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
        self.optimizer.zero_grad()
        xinit = self.g.compat_proj(xinit)  # .to(self.accelerator.device)
        xout, metrics = self.dynamics((xinit, beta))
        xout = self.g.compat_proj(xout)
        xprop = self.g.compat_proj(metrics.pop('mc_states').proposed.x)

        # xinit = to_u1(xinit).to(self.accelerator.device)
        beta = beta  # .to(self.accelerator.device)
        xout, metrics = self.dynamics((xinit, beta))
        # xprop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        loss.backward()
        if self.clip_norm > 0.0:
            torch.nn.utils.clip_grad.clip_grad_value_(
                    self.dynamics.parameters(),
                    clip_value=self.clip_norm
            )

        # self.accelerator.backward(loss)
        # self.accelerator.clip_grad_norm_(self.dynamics.parameters(),
        #                                  max_norm=self.clip_norm,)
        self.optimizer.step()

        return xout.detach(), metrics

    def train_step(
            self,
            inputs: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, dict]:
        """Logic for performing a single training step"""
        xinit, beta = inputs
        xinit = self.g.compat_proj(xinit)
        beta = torch.tensor(beta)
        loss = 0.0
        if WITH_CUDA:
            xinit, beta = xinit.cuda(), beta.cuda()

        self.optimizer.zero_grad()
        # ====================================================================
        # -----------------------  Train step  -------------------------------
        # ====================================================================
        # 1. Call model on inputs to generate:
        #      a. PROPOSAL config `xprop`   (before MH acc / rej)
        #      b. OUTPUT config `xout`      (after MH acc / rej)
        # 2. Calc loss using `xinit`, `xprop` and `acc` (acceptance rate)
        # 3. Backpropagate gradients and update network weights
        # --------------------------------------------------------------------
        # [1.] Forward call
        # with self.accelerator.autocast():
        xout, metrics = self.dynamics((xinit, beta))
        xprop = self.g.compat_proj(metrics.pop('mc_states').proposed.x)

        # [2.] Calc loss
        loss += self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])

        if self.aux_weight > 0:
            yinit = self.g.random(xout.shape)
            if WITH_CUDA:
                yinit = yinit.cuda()

            _, metrics_ = self.dynamics((yinit, beta))
            yprop = self.g.compat_proj((metrics_.pop('mc_states').proposed.x))
            aux_loss = self.aux_weight * self.loss_fn(x_init=yinit,
                                                      x_prop=yprop,
                                                      acc=metrics_['acc'])
            loss += self.aux_weight * aux_loss
            # loss = loss + (aux_loss / self.aux_weight)

        # [3.] Backpropagate gradients
        # self.accelerator.backward(loss)
        # self.optimizer.zero_grad()
        loss.backward()
        # self.optimizer.synchronize()
        self.optimizer.step()
        # if self.clip_norm > 0.0:
        #     torch.nn.utils.clip_grad_value_(
        #             self.dynamics.parameters(),
        #             clip_value=self.clip_norm,
        #     )

        # if self.lr_config.clip_norm > 0.0:
        #     # clip_grad_norm(self.dynamics.parameters(),
        #     #                max_norm=self.clip_norm)
        #     # self.accelerator.clip_grad_norm_(
        #     #     self.dynamics.parameters(),
        #     #     max_norm=self.clip_norm
        #     # )

        metrics['loss'] = loss
        if self.verbose:
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
        if self.rank != 0:
            return

        # unwrapped_model = self.accelerator.unwrap_model(self.dynamics)
        ckpt_dir = Path(train_dir).joinpath('checkpoints')
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        ckpt_file = ckpt_dir.joinpath(f'ckpt-{era}-{epoch}.tar')
        if self.rank == 0:
            log.info(f'Saving checkpoint to: {ckpt_file.as_posix()}')
        # self.dynamics.save(train_dir)  # type: ignore
        xeps = {
            k: grab(v) for k, v in self.dynamics.xeps.items()
        }
        veps = {
            k: grab(v) for k, v in self.dynamics.veps.items()
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

        # self.accelerator.save(ckpt, ckpt_file)
        torch.save(ckpt, ckpt_file)
        modelfile = ckpt_dir.joinpath('model.pth')
        # self.accelerator.save(unwrapped_model.state_dict(), modelfile)
        torch.save(self.dynamics.state_dict(), modelfile)
        if run is not None:
            assert run is wandb.run
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(modelfile.as_posix())
            run.log_artifact(artifact)

    def profile(self, nsteps: int = 5) -> dict:
        self.dynamics.train()
        x = self.draw_x()
        beta = torch.tensor(1.0)
        metrics = {}
        for _ in range(nsteps):
            x, metrics = self.profile_step((x, beta))

        return metrics

    def train_epoch(
            self,
            beta: float | Tensor,
            x: Optional[Tensor] = None,
            # era: Optional[int] = None,
            run: Optional[Any] = None,
            writer: Optional[Any] = None,
            display: Optional[dict] = None,
    ) -> dict:
        """Train the sampler for a single epoch."""
        x = self.draw_x() if x is None else x
        # .to(self.accelerator.device) if x is None else x
        # if isinstance(beta, float):
        #     beta = torch.tensor(beta).to(x.device)

        avgs = {}
        summaries = {}
        table = Table(expand=True)
        timer = self.timers['train']
        for step in range(self.steps.nepoch):
            beta = torch.tensor(beta)
            # x, beta = x.to(self.device), torch.tensor(beta).to(self.device)
            if WITH_CUDA:
                x, beta = x.cuda(), beta.cuda()

            timer.start()
            x, metrics = self.train_step((x, beta))
            dt = timer.stop()

            if metrics.get('acc', 1.0) < 1e-3:
                self.reset_optimizer()
                log.warning('Chains are stuck, redrawing x!')
                x = self.draw_x()

            if self.should_log(step):
                record = {
                    'epoch': step, 'beta': beta, 'dt': dt,
                }
                avgs_, summary = self.record_metrics(
                    run=run,
                    writer=writer,
                    record=record,
                    metrics=metrics,
                    job_type='train',
                    model=self.dynamics,
                    step=timer.iterations,
                    optimizer=self.optimizer,
                    # history=self.histories['train'],
                )
                avgs[f'{step}'] = avgs_
                summaries[f'{step}'] = summary

                if step == 0:
                    table = add_columns(avgs_, table)
                else:
                    table.add_row(*[f'{v}' for _, v in avgs_.items()])
            if display is not None:
                display['job_progress'].advance(display['tasks']['step'])
                display['job_progress'].advance(display['tasks']['epoch'])

        return {'avgs': avgs, 'summaries': summaries}

    def train(
            self,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            train_dir: Optional[os.PathLike] = None,
            run: Optional[Any] = None,
            writer: Optional[Any] = None,
            # extend_last_era: Optional[bool] = True,
            # keep: str | list[str] = None,
    ) -> dict:
        skip = [skip] if isinstance(skip, str) else skip
        train_dir = (
            Path(os.getcwd()).joinpath('train')
            if train_dir is None else Path(train_dir)
        )
        train_dir.mkdir(exist_ok=True, parents=True)

        if x is None:
            x = self.g.random(list(self.xshape)).flatten(1)

        if WIDTH is not None and WIDTH > 0:
            console.width = WIDTH

        self.dynamics.train()
        # log.warning(f'x.dtype: {x.dtype}')

        era = 0
        epoch = 0
        gstep = 0
        rows = {}
        tables = {}
        metrics = {}
        summaries = []
        table = Table(expand=True)
        nepoch = self.steps.nepoch
        timer = self.timers['train']
        history = self.histories['train']
        extend = self.steps.extend_last_era
        nepoch_last_era = self.steps.nepoch
        record = {'era': 0, 'epoch': 0, 'beta': 0.0, 'dt': 0.0}
        if extend is not None and isinstance(extend, int) and extend > 1:
            nepoch_last_era *= extend

        for era in range(self.steps.nera):
            beta = self.schedule.betas[str(era)]
            table = Table(
                box=box.HORIZONTALS,
                row_styles=['dim', 'none'],
            )

            if self.rank == 0:
                ctxmgr = Live(table,
                              console=console,
                              vertical_overflow='visible')
            else:
                ctxmgr = nullcontext()

            if era == self.steps.nera - 1:
                nepoch = nepoch_last_era
            # --- Reset optimizer states when changing beta -----------
            # if beta != self.schedule.betas[str(0)]:
            #     self.reset_optimizer()
            with ctxmgr as live:
                if live is not None:
                    tstr = ' '.join([
                        f'ERA: {era}/{self.steps.nera}',
                        f'BETA: {beta:.3f}',
                    ])
                    live.console.clear_live()
                    live.console.rule(tstr)
                    live.update(table)

                epoch_start = time.time()
                for epoch in range(nepoch):
                    timer.start()
                    # if WITH_CUDA:
                    #     with torch.cuda.amp.autocast():
                    x, metrics = self.train_step((x, beta))
                    dt = timer.stop()
                    gstep += 1
                    if self.should_print(epoch) or self.should_log(epoch):
                        record = {
                            'era': era, 'epoch': epoch, 'beta': beta, 'dt': dt,
                        }
                        avgs, summary = self.record_metrics(
                            run=run,
                            step=gstep,
                            writer=writer,
                            record=record,    # template w/ step info
                            metrics=metrics,  # metrics from Dynamics
                            job_type='train'
                        )
                        rows[gstep] = avgs
                        summaries.append(summary)

                        if epoch == 0:
                            table = add_columns(avgs, table)
                        else:
                            table.add_row(*[f'{v}' for _, v in avgs.items()])

                        if avgs.get('acc', 1.0) < 1e-5:
                            self.reset_optimizer()
                            log.warning('Chains are stuck! Re-drawing x !')
                            x = random_angle(self.xshape)
                            x = x.reshape(x.shape[0], -1)

            tables[str(era)] = table
            if self.rank == 0:
                if writer is not None:
                    update_summaries(
                        step=gstep,
                        model=self.dynamics,
                        optimizer=self.optimizer,
                        writer=writer,
                    )
                #     update_summaries(writer, step=gstep, metrics=metrics)

                st0 = time.time()
                if (era + 1) == self.steps.nera or (era + 1) % 5 == 0:
                    ckpt_metrics = {'loss': metrics.get('loss', 0.0)}
                    self.save_ckpt(era, epoch, train_dir,
                                   metrics=ckpt_metrics, run=run)

                if live is not None:
                    ckptstr = '\n'.join([
                        f'Saving took: {time.time() - st0:<5g}s',
                        f'Era {era} took: {time.time() - epoch_start:<5g}s',
                    ])
                    live.console.log(ckptstr)

        return {
            'timer': timer,
            'rows': rows,
            'summaries': summaries,
            'history': history,
            'tables': tables,
        }
