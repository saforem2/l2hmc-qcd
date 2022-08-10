#!/usr/bin/env python
"""
trainer.py

Implements methods for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from collections import defaultdict
# from contextlib import nullcontext
# from dataclasses import asdict
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional
from contextlib import nullcontext
# from accelerate import Accelerator


import aim
from aim import Distribution
# from accelerate import Accelerator
# from accelerate.utils import extract_model_from_parallel
import numpy as np
from omegaconf import DictConfig
from rich import box
# from rich.console import Console
from rich.live import Live
from rich.table import Table
import torch
import horovod.torch as hvd
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import wandb
# from l2hmc.learning_rate.pytorch.learning_rate import rate

from l2hmc.configs import (
    ExperimentConfig,
)
from l2hmc.utils.rich import get_width, is_interactive
from l2hmc.dynamics.pytorch.dynamics import Dynamics
from l2hmc.group.u1.pytorch.group import U1Phase
from l2hmc.group.su3.pytorch.group import SU3
from l2hmc.lattice.u1.pytorch.lattice import LatticeU1
from l2hmc.lattice.su3.pytorch.lattice import LatticeSU3
from l2hmc.loss.pytorch.loss import LatticeLoss
from l2hmc.network.pytorch.network import NetworkFactory
from l2hmc.trackers.pytorch.trackers import update_summaries
from l2hmc.trainers.trainer import BaseTrainer
from l2hmc.utils.history import summarize_dict
from l2hmc.utils.rich import add_columns, get_console
from l2hmc.utils.step_timer import StepTimer

# from mpi4py import MPI
# from torchinfo import summary as model_summary


from rich.logging import RichHandler
# WIDTH = int(os.environ.get('COLUMNS', 150))

log = logging.getLogger(__name__)
# console = Console(theme=Theme({'logging.level.custom': 'green'}))
console = get_console()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="%X",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            console=console,
        )
    ]
)

logging.addLevelName(70, 'CUSTOM')


Tensor = torch.Tensor
Module = torch.nn.modules.Module


# LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
SIZE = hvd.size()
RANK = hvd.rank()
LOCAL_RANK = hvd.local_rank()

WITH_CUDA = torch.cuda.is_available()

GROUPS = {
    'U1': U1Phase(),
    'SU3': SU3(),
}


def grab(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class Trainer(BaseTrainer):
    def __init__(
            self,
            cfg: DictConfig | ExperimentConfig,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
    ) -> None:
        super(Trainer, self).__init__(cfg=cfg, keep=keep, skip=skip)
        assert isinstance(self.config, ExperimentConfig)
        self._gstep = 0
        self._with_cuda = torch.cuda.is_available()
        self.lattice = self.build_lattice()
        self.loss_fn = self.build_loss_fn()
        self.dynamics = self.build_dynamics()
        self._optimizer = torch.optim.Adam(
            self.dynamics.parameters(),
            lr=self.config.learning_rate.lr_init
        )
        # self.optimizer = self.build_optimizer()
        self.lr_schedule = self.build_lr_schedule()
        self.rank = hvd.local_rank()
        self.global_rank = hvd.rank()
        # self._is_chief = self.rank == 0
        self._is_chief = (self.rank == 0 and self.global_rank == 0)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert (
            isinstance(self.dynamics, Dynamics)
            and isinstance(self.dynamics, nn.Module)
            and str(self.config.dynamics.group).upper() in ['U1', 'SU3']
        )
        if self.config.dynamics.group == 'U1':
            self.g = U1Phase()
        elif self.config.dynamics.group == 'SU3':
            self.g = SU3()
        else:
            raise ValueError
        # if self.config.dynamics.group == 'U1':
        #     self.g = U1Phase()
        # elif self.config.dynamics.group == 'SU3':
        #     self.g = SU3()
        # else:
        #     raise ValueError
        # self.verbose = self.config.dynamics.verbose
        # skip_tracking = os.environ.get('SKIP_TRACKING', False)
        # self.verbose = not skip_tracking
        self.clip_norm = self.config.learning_rate.clip_norm

        compression = (
            hvd.Compression.fp16
            if self.config.compression == 'fp16'
            else hvd.Compression.none
        )
        self.optimizer = hvd.DistributedOptimizer(
            self._optimizer,
            named_parameters=self.dynamics.named_parameters(),
            compression=compression,  # type: ignore
        )
        hvd.broadcast_parameters(self.dynamics.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        # self.optimizer = hvd.DistributedOptimizer(
        #         self.optimizer,
        #         # named_parameters=self.dynamics.named_parameters(),
        #         # op=hvd.Average,
        #         # op=hvd.Adasum if cfg.use_adasum else hvd.Average,
        # )
        # hvd.broadcast_parameters(self.dynamics.networks.state_dict(),
        #                          root_rank=0)
        # hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        # self.device = hvd.local_rank()
        # self.rank = hvd.local_rank()
        # self.rank = self.accelerator.local_process_index

    def warning(self, s: str):
        if self._is_chief:
            log.warning(s)

    def draw_x(self):
        return self.g.random(
            list(self.config.dynamics.xshape)
        ).flatten(1)

    def reset_optimizer(self):
        if self._is_chief:
            self.warning('Resetting optimizer state!')
            self.optimizer.state = defaultdict(dict)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

    def build_lattice(self):
        group = str(self.config.dynamics.group).upper()
        kwargs = {
            'nchains': self.config.dynamics.nchains,
            'shape': list(self.config.dynamics.latvolume),
        }
        if group == 'U1':
            return LatticeU1(**kwargs)
        if group == 'SU3':
            c1 = (
                self.config.c1
                if self.config.c1 is not None
                else 0.0
            )

            return LatticeSU3(c1=c1, **kwargs)

        raise ValueError('Unexpected value in `config.dynamics.group`')

    def build_loss_fn(self) -> Callable:
        assert isinstance(self.lattice, (LatticeU1, LatticeSU3))
        return LatticeLoss(
            lattice=self.lattice,
            loss_config=self.config.loss,
        )

    def build_dynamics(self) -> Dynamics:
        input_spec = self.get_input_spec()
        net_factory = NetworkFactory(
            input_spec=input_spec,
            conv_config=self.config.conv,
            network_config=self.config.network,
            net_weights=self.config.net_weights,
        )
        dynamics = Dynamics(config=self.config.dynamics,
                            potential_fn=self.lattice.action,
                            network_factory=net_factory)
        if torch.cuda.is_available():
            dynamics.cuda()
        # state = dynamics.random_state(1.)
        # for step in range(self.config.dynamics.nleapfrog):
        #     xn0 = dynamics._get_xnet(step, first=True)
        #     xn1 = dynamics._get_xnet(step, first=True)
        #     vn = dynamics._get_vnet(step)
        #     if torch.cuda.is_available():
        #         xn0, xn1, vn = xn0.cuda(), xn1.cuda(), vn.cuda()
        #     xn0 = xn0.float()
        #     xn1 = xn1.float()
        #     vn = vn.float()
        #     _ = dynamics._call_xnet(step, (state.x, state.v), first=True)
        #     _ = dynamics._call_xnet(step, (state.x, state.v), first=False)
        #     _ = dynamics._call_vnet(step, (state.x, state.v))

        return dynamics

    def build_optimizer(
            self,
    ) -> torch.optim.Optimizer:
        # TODO: Expand method, re-build LR scheduler, etc
        # TODO: Replace `LearningRateConfig` with `OptimizerConfig`
        # TODO: Optionally, break up in to lrScheduler, OptimizerConfig ?
        lr = self.config.learning_rate.lr_init
        assert isinstance(self.dynamics, Dynamics)
        return torch.optim.Adam(self.dynamics.parameters(), lr=lr)

    def get_lr(self, step: int) -> float:
        if step < len(self._lr_warmup):
            return self._lr_warmup[step].item()
        return self.config.learning_rate.lr_init

    def build_lr_schedule(self):
        self._lr_warmup = torch.linspace(
            self.config.learning_rate.min_lr,
            self.config.learning_rate.lr_init,
            2 * self.steps.nepoch
        )
        return LambdaLR(
            optimizer=self._optimizer,
            lr_lambda=lambda step: self.get_lr(step)
        )

    def save_ckpt(
            self,
            era: int,
            epoch: int,
            train_dir: os.PathLike,
            metrics: Optional[dict] = None,
            run: Optional[Any] = None,
    ) -> None:
        if not self._is_chief:
            return

        # assert isinstance(self.dynamics, Dynamics)
        ckpt_dir = Path(train_dir).joinpath('checkpoints')
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        ckpt_file = ckpt_dir.joinpath(f'ckpt-{era}-{epoch}.tar')
        if self._is_chief:
            log.info(f'Saving checkpoint to: {ckpt_file.as_posix()}')
        assert isinstance(self.dynamics.xeps, nn.ParameterList)
        assert isinstance(self.dynamics.veps, nn.ParameterList)
        xeps = [e.detach().cpu().numpy() for e in self.dynamics.xeps]
        veps = [e.detach().cpu().numpy() for e in self.dynamics.veps]
        # xeps = {
        #     k: grab(v) for k, v in self.dynamics.xeps
        # }
        # veps = {
        #     k: grab(v) for k, v in self.dynamics.veps
        # }
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

        torch.save(ckpt, ckpt_file)
        modelfile = ckpt_dir.joinpath('model.pth')
        torch.save(self.dynamics.state_dict(), modelfile)
        if run is not None:
            assert run is wandb.run
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(modelfile.as_posix())
            run.log_artifact(artifact)

    def should_log(self, epoch):
        return (
            epoch % self.steps.log == 0
            # and self.rank == 0
            and self._is_chief
            # and LOCAL_RANK == 0
        )

    def should_print(self, epoch):
        return (
            epoch % self.steps.print == 0
            # and self.rank == 0
            and self._is_chief
        )

    def should_emit(self, epoch: int, nepoch: int) -> bool:
        nprint = min(
            getattr(self.steps, 'print', int(nepoch // 2)),
            int(nepoch // 2)
        )
        nlog = min(
            getattr(self.steps, 'log', int(nepoch // 4)),
            int(nepoch // 4)
        )
        emit = (
            epoch % nprint == 0
            or epoch % nlog == 0
        )

        # LOCAL_RANK == 0
        # return (
        #     self._is_chief and (
        #         (epoch % nprint == 0
        #          or epoch % nlog == 0)
        #     )
        # )
        return self._is_chief and emit

    def record_metrics(
            self,
            metrics: dict,
            job_type: str,
            step: Optional[int] = None,
            record: Optional[dict] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            model: Optional[nn.Module | Dynamics] = None,
            optimizer: Optional[Any] = None
    ):
        record = {} if record is None else record
        assert job_type in ['train', 'eval', 'hmc']
        if step is None:
            timer = self.timers.get(job_type, None)
            if isinstance(timer, StepTimer):
                step = timer.iterations

        if step is not None:
            record.update({f'{job_type}_step': step})

        record.update({
            'loss': metrics.get('loss', None),
            'dQint': metrics.get('dQint', None),
            'dQsin': metrics.get('dQsin', None),
        })
        if job_type in ['hmc', 'eval']:
            _ = record.pop('xeps', None)
            _ = record.pop('veps', None)
            if job_type == 'hmc':
                _ = record.pop('sumlogdet', None)

        if job_type == 'train' and step is not None:
            record['lr'] = self.get_lr(step)

        record.update(self.metrics_to_numpy(metrics))
        avgs = self.histories[job_type].update(record)
        summary = summarize_dict(avgs)

        if (
                step is not None
                and writer is not None
        ):
            assert step is not None
            update_summaries(step=step,
                             model=model,
                             writer=writer,
                             metrics=record,
                             prefix=job_type,
                             optimizer=optimizer)
            writer.flush()

        if self.config.init_aim or self.config.init_wandb:
            self.track_metrics(
                record=record,
                avgs=avgs,
                job_type=job_type,
                step=step,
                run=run,
                arun=arun,
            )

        return avgs, summary

    def track_metrics(
            self,
            record: dict[str, Tensor],
            avgs: dict[str, Tensor],
            job_type: str,
            step: Optional[int],
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
    ) -> None:
        if self.rank != 0:
            return
        # assert self.config.init_aim or self.config.init_wandb
        dQdict = None
        dQint = record.get('dQint', None)
        if dQint is not None:
            dQdict = {
                f'dQint/{job_type}': {
                    'val': dQint,
                    'step': step,
                    'avg': dQint.mean(),
                }
            }

        if run is not None and self.config.init_wandb:
            run.log({f'wandb/{job_type}': record}, commit=False)
            run.log({f'avgs/wandb.{job_type}': avgs})
            if dQdict is not None:
                run.log(dQdict, commit=False)
        if arun is not None and self.config.init_aim:
            kwargs = {
                'step': step,
                'job_type': job_type,
                'arun': arun
            }
            self.aim_track(avgs, prefix='avgs', **kwargs)
            self.aim_track(record, prefix='record', **kwargs)
            if dQdict is not None:
                self.aim_track({'dQint': dQint}, prefix='dQ', **kwargs)

    def profile_step(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        assert isinstance(self.dynamics, Dynamics)
        assert isinstance(self.config, ExperimentConfig)
        self.optimizer.zero_grad()
        xinit = self.g.compat_proj(xinit)  # .to(self.accelerator.device)
        xout, metrics = self.dynamics((xinit, beta))
        xout = self.g.compat_proj(xout)
        xprop = self.g.compat_proj(metrics.pop('mc_states').proposed.x)

        beta = beta  # .to(self.accelerator.device)
        xout, metrics = self.dynamics((xinit, beta))
        # xprop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        loss.backward()
        if self.config.learning_rate.clip_norm > 0.0:
            torch.nn.utils.clip_grad.clip_grad_norm(
                    self.dynamics.parameters(),
                    self.config.learning_rate.clip_norm,
            )

        # self.accelerator.backward(loss)
        # self.accelerator.clip_grad_norm_(self.dynamics.parameters(),
        #                                  max_norm=self.clip_norm,)
        self.optimizer.step()

        return xout.detach(), metrics

    def profile(self, nsteps: int = 5) -> dict:
        assert isinstance(self.dynamics, Dynamics)
        self.dynamics.train()
        x = self.draw_x()
        beta = torch.tensor(1.0)
        metrics = {}
        for _ in range(nsteps):
            x, metrics = self.profile_step((x, beta))

        return metrics

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
        if self.config.dynamics.verbose:
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
        # xout = self.g.compat_proj(xout)
        xprop = metrics.pop('mc_states').proposed.x
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        if self.config.dynamics.verbose:
            lmetrics = self.loss_fn.lattice_metrics(xinit=xinit, xout=xout)
            metrics.update(lmetrics)

        metrics.update({'loss': loss.detach().cpu().numpy()})

        return xout.detach(), metrics

    def get_context_manager(self, table: Table):
        width = get_width()
        make_live = (
            int(width) > 150          # make sure wide enough to fit table
            and hvd.size() > 1        # not worth the trouble when distributed
            and self.rank == 0        # only display from (one) main rank
            and not is_interactive()  # AND not in a jupyter / ipython kernel
        )
        if make_live:
            return Live(
                table,
                # screen=True,
                console=self.console,
                vertical_overflow='visible'
            )

        return nullcontext()

    def eval(
            self,
            beta: Optional[float] = None,
            eval_steps: Optional[int] = None,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
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
            beta = self.config.annealing_schedule.beta_final

        if x is None:
            x = self.g.random(list(self.xshape))
            x = x.reshape(x.shape[0], -1)

        if eps is None and str(job_type).lower() == 'hmc':
            eps = self.config.dynamics.eps_hmc
            assert eps is not None
            log.warn(f'Using step size eps: {eps:.4f} for generic HMC')

        assert job_type in ['eval', 'hmc']

        def eval_fn(z):
            if job_type == 'hmc':
                assert eps is not None
                return self.hmc_step(z, eps=eps, nleapfrog=nleapfrog)
            return self.eval_step(z)

        tables = {}
        table = Table(row_styles=['dim', 'none'], box=box.HORIZONTALS)

        eval_steps = self.steps.test if eval_steps is None else eval_steps
        assert isinstance(eval_steps, int)
        nprint = max(1, eval_steps // 50)
        nlog = max((1, min((10, eval_steps))))
        if nlog <= eval_steps:
            nlog = min(10, max(1, eval_steps // 100))

        assert job_type in ['eval', 'hmc']
        timer = self.timers[job_type]
        history = self.histories[job_type]

        self.warning(f'x.shape (original): {x.shape}')
        if nchains is not None:
            if isinstance(nchains, int) and nchains > 0:
                x = x[:nchains]

        assert isinstance(x, Tensor)
        assert isinstance(beta, float)
        self.warning(f'x[:nchains].shape: {x.shape}')

        if run is not None:
            run.config.update({job_type: {'beta': beta, 'xshape': x.shape}})

        ctx = self.get_context_manager(table)
        with ctx:
            for step in range(eval_steps):
                timer.start()
                x, metrics = eval_fn((x, beta))
                dt = timer.stop()
                if step % nlog == 0 or step % nprint == 0:
                    record = {
                        'step': step, 'beta': beta, 'dt': dt,
                    }
                    avgs, summary = self.record_metrics(run=run,
                                                        arun=arun,
                                                        step=step,
                                                        record=record,
                                                        writer=writer,
                                                        metrics=metrics,
                                                        job_type=job_type)
                    if not isinstance(ctx, Live) and step % nprint == 0:
                        log.info(summary)

                    summaries.append(summary)
                    if step == 0:
                        table = add_columns(avgs, table)
                    else:
                        table.add_row(*[f'{v}' for _, v in avgs.items()])

                    if avgs.get('acc', 1.0) < 1e-5:
                        self.reset_optimizer()
                        self.console.log('Chains are stuck! Redrawing x')
                        x = self.g.random(list(x.shape))

        # console.log(table)
        tables[str(0)] = table

        return {
            'timer': timer,
            'history': history,
            'summaries': summaries,
            'tables': tables,
        }

    def train_step(
            self,
            inputs: tuple[Tensor, Tensor | float],
    ) -> tuple[Tensor, dict]:
        """Logic for performing a single training step"""
        xinit, beta = inputs
        xinit = self.g.compat_proj(xinit)
        # xinit = self.to_u1(xinit)
        beta = torch.tensor(beta) if isinstance(beta, float) else beta
        if WITH_CUDA:
            xinit, beta = xinit.cuda(), beta.cuda()

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
        # xinit = self.g.compat_proj(xinit.requires_grad_(True))
        xinit.requires_grad_(True)

        self.optimizer.zero_grad()
        xout, metrics = self.dynamics((xinit, beta))
        xprop = metrics.pop('mc_states').proposed.x

        # [2.] Calc loss
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])

        aw = self.config.loss.aux_weight
        if aw > 0:
            yinit = self.g.random(xout.shape)
            yinit.requires_grad_(True)
            if WITH_CUDA:
                yinit = yinit.cuda()

            _, metrics_ = self.dynamics((yinit, beta))
            yprop = metrics_.pop('mc_states').proposed.x
            aux_loss = aw * self.loss_fn(x_init=yinit,
                                         x_prop=yprop,
                                         acc=metrics_['acc'])
            loss += aw * aux_loss

        # # [3.] Backpropagate gradients
        # self.accelerator.backward(loss)
        loss.backward()
        if self.config.learning_rate.clip_norm > 0.0:
            torch.nn.utils.clip_grad.clip_grad_norm(
                self.dynamics.parameters(),
                max_norm=self.clip_norm
            )
        self.optimizer.step()
        # self.lr_schedule.step()
        self.optimizer.synchronize()

        # ---------------------------------------
        # DEPRECATED: Removed Accelerator
        # self.optimizer.zero_grad()
        # self.accelerator.backward(loss)
        # # extract_model_from_parallel(self.dynamics).parameters(),
        # self.accelerator.clip_grad_norm_(
        #     self.dynamics.parameters(),
        #     max_norm=self.clip_norm,
        # )
        # self.optimizer.step()
        # ---------------------------------------

        metrics['loss'] = loss
        if self.config.dynamics.verbose:
            with torch.no_grad():
                lmetrics = self.loss_fn.lattice_metrics(xinit=xinit, xout=xout)
                metrics.update(lmetrics)

        return xout.detach(), metrics

    def train_epoch(
            self,
            x: Tensor,
            beta: float | Tensor,
            era: Optional[int] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            nepoch: Optional[int] = None,
            writer: Optional[Any] = None,
            extend: int = 1,
    ) -> tuple[Tensor, dict]:
        rows = {}
        summaries = []
        extend = 1 if extend is None else extend
        record = {'era': 0, 'epoch': 0, 'beta': 0.0, 'dt': 0.0}
        table = Table(
            box=box.HORIZONTALS,
            row_styles=['dim', 'none'],
        )

        nepoch = self.steps.nepoch if nepoch is None else nepoch
        assert isinstance(nepoch, int)
        nepoch *= extend
        losses = []
        ctx = self.get_context_manager(table)
        with ctx:
            if isinstance(ctx, Live):
                # tstr = ' '.join([
                #     f'ERA: {era}',
                #     f'BETA: {beta:.3f}',
                # ])
                ctx.console.clear_live()
                # ctx.console.rule(tstr)
                ctx.update(table)

            for epoch in range(nepoch):
                self.timers['train'].start()
                x, metrics = self.train_step((x, beta))  # type:ignore
                dt = self.timers['train'].stop()
                losses.append(metrics['loss'])
                self._gstep += 1
                # if self.should_emit(epoch, nepoch):
                # if (
                #         self._is_chief and (
                #             self.should_print(epoch)
                #             or self.should_log(epoch)
                #         )
                # ):
                if self.should_print(epoch) or self.should_log(epoch):
                    record = {
                        'era': era, 'epoch': epoch, 'beta': beta, 'dt': dt,
                    }
                    avgs, summary = self.record_metrics(
                        run=run,
                        arun=arun,
                        step=self._gstep,
                        writer=writer,
                        record=record,    # template w/ step info
                        metrics=metrics,  # metrics from Dynamics
                        job_type='train',
                        model=self.dynamics,
                        optimizer=self._optimizer,
                    )
                    rows[self._gstep] = avgs
                    summaries.append(summary)

                    if self.should_print(epoch) and not isinstance(ctx, Live):
                        log.info(summary)

                    if epoch == 0:
                        table = add_columns(avgs, table)
                    else:
                        table.add_row(*[f'{v}' for _, v in avgs.items()])

                    if avgs.get('acc', 1.0) < 1e-5:
                        self.reset_optimizer()
                        self.warning('Chains are stuck! Re-drawing x !')
                        x = self.draw_x()

        data = {
            'rows': rows,
            'table': table,
            'losses': losses,
            'summaries': summaries,
        }

        return x, data

    def train(
            self,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            train_dir: Optional[os.PathLike] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            nera: Optional[int] = None,
            nepoch: Optional[int] = None,
            beta: Optional[float | list[float] | dict[str, float]] = None,
    ) -> dict:
        """Perform training and return dictionary of results."""
        skip = [skip] if isinstance(skip, str) else skip
        # steps = self.steps if steps is None else steps
        train_dir = (
            Path(os.getcwd()).joinpath(
                self._created, 'train'
            )
            if train_dir is None else Path(train_dir)
        )
        train_dir.mkdir(exist_ok=True, parents=True)

        if x is None:
            x = self.g.random(list(self.xshape)).flatten(1)

        nera = self.config.steps.nera if nera is None else nera
        nepoch = self.config.steps.nepoch if nepoch is None else nepoch
        extend = self.config.steps.extend_last_era
        assert isinstance(nera, int)
        assert isinstance(nepoch, int)

        if beta is not None:
            assert isinstance(beta, (float, list))
            if isinstance(beta, list):
                assert len(beta) == nera, 'Expected len(beta) == nera'
            else:
                beta = nera * [beta]

            betas = {f'{i}': b for i, b in zip(range(nera), beta)}

        else:
            betas = self.config.annealing_schedule.setup(
                nera=nera,
                nepoch=nepoch,
            )

        beta_final = list(betas.values())[-1]
        assert beta_final is not None and isinstance(beta_final, float)
        # assert b is not None and isinstance(b, float)
        # while b < beta_final:
        self.dynamics.train()
        era = 0
        epoch = 0
        extend = 1
        for era in range(nera):
            b = torch.tensor(betas.get(str(era), beta_final))
            if era == (nera - 1) and self.steps.extend_last_era is not None:
                extend = int(self.steps.extend_last_era)

            if self._is_chief:
                if era > 1 and str(era - 1) in self.summaries['train']:
                    esummary = self.histories['train'].era_summary(f'{era-1}')
                    log.info(f'Avgs over last era:\n {esummary}\n')

                self.console.rule(f'ERA: {era} / {nera}, BETA: {b:.3f}')

            epoch_start = time.time()
            x, edata = self.train_epoch(
                x=x,
                beta=b,
                era=era,
                run=run,
                arun=arun,
                writer=writer,
                extend=extend,
                nepoch=nepoch,
            )
            st0 = time.time()

            # losses = edata['losses']
            # if losses[-1] < losses[0]:
            #     b += self.config.annealing_schedule._dbeta
            # else:
            #     b -= self.config.annealing_schedule._dbeta

            self.rows['train'][str(era)] = edata['rows']
            self.tables['train'][str(era)] = edata['table']
            self.summaries['train'][str(era)] = edata['summaries']

            if (era + 1) == nera or (era + 1) % 5 == 0:
                # ckpt_metrics = {'loss': metrics.get('loss', 0.0)}
                self.save_ckpt(era, epoch, train_dir, run=run)

            if self._is_chief:
                log.info(f'Saving took: {time.time() - st0:<5g}s')
                log.info(f'Era {era} took: {time.time() - epoch_start:<5g}s')

        return {
            'timer': self.timers['train'],
            'rows': self.rows['train'],
            'summaries': self.summaries['train'],
            'history': self.histories['train'],
            'tables': self.tables['train'],
        }

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

        # arr = metric.detach().cpu().numpy()
        # arr = arr[~np.isnan(arr)]
        # arr = arr[~np.isnan(arr)]
        # return arr
        return metric.detach().cpu().numpy()

    def aim_track(
            self,
            metrics: dict,
            step: int,
            job_type: str,
            arun: aim.Run,
            prefix: Optional[str] = None,
    ) -> None:
        context = {'subset': job_type}
        for key, val in metrics.items():
            if prefix is not None:
                name = f'{prefix}/{key}'
            else:
                name = f'{key}'

            if isinstance(val, dict):
                for k, v in val.items():
                    self.aim_track(
                        v,
                        step=step,
                        arun=arun,
                        job_type=job_type,
                        prefix=f'{name}/{k}',
                    )

            if isinstance(val, (Tensor, np.ndarray)):
                if len(val.shape) > 1:
                    dist = Distribution(val)
                    arun.track(dist,
                               step=step,
                               name=name,
                               context=context)  # type: ignore
                    arun.track(val.mean(),
                               step=step,
                               name=f'{name}/avg',
                               context=context,)  # type: ignore
            else:
                arun.track(val,
                           name=name,
                           step=step,
                           context=context)  # type: ignore
