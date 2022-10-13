"""
trainer.py

Implements methods for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from collections import defaultdict
from contextlib import nullcontext
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional

import aim
from aim import Distribution
import horovod.torch as hvd
import numpy as np
from omegaconf import DictConfig
from rich import box
from rich.live import Live
from rich.logging import RichHandler
from rich.table import Table
from rich_logger import RichTablePrinter
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange
import wandb

from l2hmc.common import ScalarLike, TensorLike, setup_torch_distributed
from l2hmc.configs import ExperimentConfig
from l2hmc.dynamics.pytorch.dynamics import Dynamics
from l2hmc.group.su3.pytorch.group import SU3
from l2hmc.group.u1.pytorch.group import U1Phase
from l2hmc.lattice.su3.pytorch.lattice import LatticeSU3
from l2hmc.lattice.u1.pytorch.lattice import LatticeU1
from l2hmc.loss.pytorch.loss import LatticeLoss
from l2hmc.network.pytorch.network import NetworkFactory
from l2hmc.trackers.pytorch.trackers import update_summaries
from l2hmc.trainers.trainer import BaseTrainer
from l2hmc.utils.history import summarize_dict
from l2hmc.utils.rich import get_width, is_interactive
from l2hmc.utils.rich import get_console
from l2hmc.utils.rich_logger import LOGGER_FIELDS
from l2hmc.utils.step_timer import StepTimer
# WIDTH = int(os.environ.get('COLUMNS', 150))

log = logging.getLogger(__name__)
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
            build_networks: bool = True,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
    ) -> None:
        super(Trainer, self).__init__(cfg=cfg, keep=keep, skip=skip)
        # skip_tracking = os.environ.get('SKIP_TRACKING', False)
        # self.verbose = not skip_tracking
        assert isinstance(self.config, ExperimentConfig)
        self._gstep = 0
        dsetup = setup_torch_distributed(self.config.backend)
        self.size = dsetup['size']
        self.rank = dsetup['rank']
        self.local_rank = dsetup['local_rank']
        self._is_chief = (self.local_rank == 0 and self.rank == 0)
        self._with_cuda = torch.cuda.is_available()
        self.lattice = self.build_lattice()
        self.loss_fn = self.build_loss_fn()
        self.dynamics = self.build_dynamics(
            build_networks=build_networks,
        )
        self.dynamics_ddp = None
        if self.config.backend == 'DDP':
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.dynamics_ddp = DDP(self.dynamics)

        if self.config.dynamics.group == 'U1':
            log.warning('Using `torch.optim.Adam` optimizer')
            self._optimizer = torch.optim.Adam(
                self.dynamics.parameters(),
                lr=self.config.learning_rate.lr_init
            )
        else:
            log.warning('Using `torch.optim.SGD` optimizer')
            self._optimizer = torch.optim.SGD(
                self.dynamics.parameters(),
                lr=self.config.learning_rate.lr_init,
            )

        self._lr_warmup = torch.linspace(
            self.config.learning_rate.min_lr,
            self.config.learning_rate.lr_init,
            2 * self.steps.nepoch
        )
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
        self.clip_norm = self.config.learning_rate.clip_norm
        if self.config.backend in ['hvd', 'horovod']:
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
        else:
            self.optimizer = self._optimizer

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

    def build_dynamics(
            self,
            build_networks: bool = True,
    ) -> Dynamics:
        input_spec = self.get_input_spec()
        net_factory = None
        if build_networks:
            net_factory = NetworkFactory(
                input_spec=input_spec,
                conv_config=self.config.conv,
                network_config=self.config.network,
                net_weights=self.config.net_weights,
            )
        dynamics = Dynamics(
            config=self.config.dynamics,
            potential_fn=self.lattice.action,
            # backend=self.config.backend,
            network_factory=net_factory,
        )
        if torch.cuda.is_available():
            dynamics.cuda()

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
            return self.config.learning_rate.lr_init
        return self.config.learning_rate.lr_init

    def build_lr_schedule(self):
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
        return (epoch % self.steps.log == 0 and self._is_chief)

    def should_print(self, epoch):
        return (epoch % self.steps.print == 0 and self._is_chief)

    def should_emit(self, epoch: int, nepoch: int) -> bool:
        nprint = min(
            getattr(self.steps, 'print', int(nepoch // 10)),
            int(nepoch // 5)
        )
        nlog = min(
            getattr(self.steps, 'log', int(nepoch // 4)),
            int(nepoch // 4)
        )
        emit = (
            epoch % nprint == 0
            or epoch % nlog == 0
        )

        return self._is_chief and emit

    def record_metrics(
            self,
            metrics: dict,
            job_type: str,
            step: Optional[int] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            model: Optional[nn.Module | Dynamics] = None,
            optimizer: Optional[Any] = None
    ) -> tuple[dict[str, ScalarLike], str]:
        assert job_type in ['train', 'eval', 'hmc']
        if step is None:
            timer = self.timers.get(job_type, None)
            if isinstance(timer, StepTimer):
                step = timer.iterations

        if step is not None:
            metrics.update({f'{job_type[0]}step': step})

        if job_type == 'train' and step is not None:
            metrics['lr'] = self.get_lr(step)

        if job_type == 'eval' and 'eps' in metrics:
            _ = metrics.pop('eps', None)

        metrics.update(self.metrics_to_numpy(metrics))
        avgs = self.histories[job_type].update(metrics)
        summary = summarize_dict(avgs)

        if (
                step is not None
                and writer is not None
        ):
            assert step is not None
            update_summaries(step=step,
                             model=model,
                             writer=writer,
                             metrics=metrics,
                             prefix=job_type,
                             optimizer=optimizer)
            writer.flush()

        if self.config.init_aim or self.config.init_wandb:
            self.track_metrics(
                record=metrics,
                avgs=avgs,
                job_type=job_type,
                step=step,
                run=run,
                arun=arun,
            )

        return avgs, summary

    def track_metrics(
            self,
            record: dict[str, TensorLike | ScalarLike],
            avgs: dict[str, ScalarLike],
            job_type: str,
            step: Optional[int],
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
    ) -> None:
        if self.local_rank != 0:
            return
        # assert self.config.init_aim or self.config.init_wandb
        dQdict = None
        dQint = record.get('dQint', None)
        if dQint is not None:
            dQdict = {
                f'dQint/{job_type}': {
                    'val': dQint,
                    'step': step,
                    'avg': dQint.mean(),  # type:ignore
                }
            }

        if run is not None:
            try:
                run.log({f'wandb/{job_type}': record}, commit=False)
                run.log({f'avgs/wandb.{job_type}': avgs})
                if dQdict is not None:
                    run.log(dQdict, commit=False)
            except ValueError:
                self.warning('Unable to track record with WandB, skipping!')
        if arun is not None:
            kwargs = {
                'step': step,
                'job_type': job_type,
                'arun': arun
            }
            try:
                self.aim_track(avgs, prefix='avgs', **kwargs)
                self.aim_track(record, prefix='record', **kwargs)
                if dQdict is not None:
                    self.aim_track({'dQint': dQint}, prefix='dQ', **kwargs)
            except ValueError:
                self.warning('Unable to track record with aim, skipping!')

    def profile_step(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        assert isinstance(self.dynamics, Dynamics)
        assert isinstance(self.config, ExperimentConfig)
        self.optimizer.zero_grad()
        xinit = self.g.compat_proj(xinit)  # .to(self.accelerator.device)
        if self.dynamics_ddp is not None:
            xout, metrics = self.dynamics_ddp((xinit, beta))
        else:
            xout, metrics = self.dynamics((xinit, beta))
        xout = self.g.compat_proj(xout)
        xprop = self.g.compat_proj(metrics.pop('mc_states').proposed.x)

        beta = beta  # .to(self.accelerator.device)
        # xprop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        loss.backward()
        if self.config.learning_rate.clip_norm > 0.0:
            torch.nn.utils.clip_grad.clip_grad_norm(
                    self.dynamics.parameters(),
                    self.config.learning_rate.clip_norm,
            )

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

        metrics.update({'loss': loss.item()})
        return xo.detach(), metrics

    def eval_step(
            self,
            inputs: tuple[Tensor, float]
    ) -> tuple[Tensor, dict]:
        self.dynamics.eval()
        xinit, beta = inputs
        if WITH_CUDA:
            xinit, beta = xinit.cuda(), torch.tensor(beta).cuda()
        if self.dynamics_ddp is not None:
            xout, metrics = self.dynamics_ddp((xinit, beta))
        else:
            xout, metrics = self.dynamics((xinit, beta))
        xprop = metrics.pop('mc_states').proposed.x
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        if self.config.dynamics.verbose:
            lmetrics = self.loss_fn.lattice_metrics(xinit=xinit, xout=xout)
            metrics.update(lmetrics)

        metrics.update({
            'loss': loss.item(),
        })

        return xout.detach(), metrics

    def get_context_manager(self, table: Table):
        width = get_width()
        make_live = (
            int(width) > 150            # make sure wide enough to fit table
            and self._is_chief
            and self.size == 1          # not worth the trouble when dist.
            and not is_interactive()    # AND not in a jupyter / ipython kernel
        )
        if make_live:
            return Live(
                table,
                # screen=True,
                transient=True,
                # auto_refresh=False,
                console=self.console,
                vertical_overflow='visible'
            )

        return nullcontext()

    def get_printer(self, job_type: str) -> RichTablePrinter | None:
        if self._is_chief and int(get_width()) > 100:
            printer = RichTablePrinter(
                key=f'{job_type[0]}step',
                fields=LOGGER_FIELDS  # type:ignore
            )

            printer.hijack_tqdm()
            # printer.expand = True

            return printer
        return None

    def _setup_eval(
            self,
            beta: Optional[float] = None,
            eval_steps: Optional[int] = None,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            run: Optional[Any] = None,
            job_type: Optional[str] = 'eval',
            nchains: Optional[int] = None,
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> dict:
        assert job_type in ['eval', 'hmc']

        if isinstance(skip, str):
            skip = [skip]

        if beta is None:
            beta = self.config.annealing_schedule.beta_final

        if nleapfrog is None and str(job_type).lower() == 'hmc':
            nleapfrog = int(self.config.dynamics.nleapfrog)
            # assert isinstance(nleapfrog, int)
            if self.config.dynamics.merge_directions:
                nleapfrog *= 2

        if eps is None and str(job_type).lower() == 'hmc':
            eps = self.config.dynamics.eps_hmc
            assert eps is not None
            log.warn(f'Using step size eps: {eps:.4f} for generic HMC')

        if x is None:
            x = self.lattice.random()

        self.warning(f'x.shape (original): {x.shape}')
        if nchains is not None:
            if isinstance(nchains, int) and nchains > 0:
                x = x[:nchains]

        assert isinstance(x, Tensor)
        self.warning(f'x[:nchains].shape: {x.shape}')

        table = Table(row_styles=['dim', 'none'], box=box.HORIZONTALS)
        eval_steps = self.steps.test if eval_steps is None else eval_steps
        assert isinstance(eval_steps, int)
        nprint = max(1, eval_steps // 50)
        nlog = max((1, min((10, eval_steps))))
        if nlog <= eval_steps:
            nlog = min(10, max(1, eval_steps // 100))

        if run is not None:
            run.config.update({
                job_type: {'beta': beta, 'xshape': x.shape}
            })

        assert isinstance(x, Tensor)
        assert isinstance(beta, float)
        assert isinstance(nlog, int)
        assert isinstance(nprint, int)
        assert isinstance(eval_steps, int)
        # assert isinstance(eps, float)
        # assert isinstance(nleapfrog, int)
        output = {
            'x': x,
            'eps': eps,
            'beta': beta,
            'nlog': nlog,
            'table': table,
            'nprint': nprint,
            'eval_steps': eval_steps,
            'nleapfrog': nleapfrog,
        }
        log.info(
            '\n'.join([
                f'{k}={v}' for k, v in output.items()
                if k != 'x'
            ])
        )
        return output

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
            dynamic_step_size: Optional[bool] = None,
    ) -> dict:
        """Evaluate dynamics."""

        assert job_type in ['eval', 'hmc']
        tables = {}
        summaries = []
        patience = 5
        stuck_counter = 0
        setup = self._setup_eval(
            x=x,
            run=run,
            beta=beta,
            eps=eps,
            nleapfrog=nleapfrog,
            skip=skip,
            nchains=nchains,
            job_type=job_type,
            eval_steps=eval_steps,
        )
        x = setup['x']
        eps = setup['eps']
        beta = setup['beta']
        # table = setup['table']
        nleapfrog = setup['nleapfrog']
        eval_steps = setup['eval_steps']
        timer = self.timers[job_type]
        history = self.histories[job_type]
        assert (
            eval_steps is not None
            and timer is not None
            and history is not None
        )

        def eval_fn(z):
            if job_type == 'hmc':
                assert eps is not None
                return self.hmc_step(z, eps=eps, nleapfrog=nleapfrog)
            return self.eval_step(z)

        self.dynamics.eval()
        # with self.get_context_manager(table) as ctx:
        printer = self.get_printer(job_type=job_type)
        for step in trange(
                eval_steps,
                dynamic_ncols=True,
                disable=(not self._is_chief),
        ):
            timer.start()
            x, metrics = eval_fn((x, beta))
            dt = timer.stop()
            if (
                    step == 0
                    or step % setup['nlog'] == 0
                    or step % setup['nprint'] == 0
            ):
                record = {
                    f'{job_type[0]}step': step,
                    'dt': dt,
                    'beta': beta,
                    'loss': metrics.pop('loss', None),
                    'dQsin': metrics.pop('dQsin', None),
                    'dQint': metrics.pop('dQint', None),
                }
                record.update(metrics)
                if job_type == 'hmc' and dynamic_step_size:
                    acc = record.get('acc_mask', None)
                    record['eps'] = eps
                    if acc is not None and eps is not None:
                        acc_avg = acc.mean()
                        if acc_avg < 0.66:
                            eps -= (eps / 10.)
                        else:
                            eps += (eps / 10.)

                avgs, summary = self.record_metrics(run=run,
                                                    arun=arun,
                                                    step=step,
                                                    writer=writer,
                                                    metrics=record,
                                                    job_type=job_type)
                summaries.append(summary)
                # table = self.update_table(
                #     table=setup['table'],
                #     step=step,
                #     avgs=avgs,
                # )
                if (
                        # not isinstance(setup['ctx'], Live)
                        step > 0 and
                        step % setup['nprint'] == 0
                ):
                    if printer is not None:
                        printer.log(avgs)
                    else:
                        log.info(summary)

                if avgs.get('acc', 1.0) < 1e-5:
                    if stuck_counter < patience:
                        stuck_counter += 1
                    else:
                        self.console.log('Chains are stuck! Redrawing x')
                        x = self.lattice.random()
                        stuck_counter = 0

            # if isinstance(ctx, Live):
            #     ctx.console.clear_live()

        # tables[str(0)] = setup['table']

        if printer is not None:
            printer.finalize()

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
        xinit = self.g.compat_proj(xinit.reshape(self.xshape))
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
        if self.dynamics_ddp is not None:
            xout, metrics = self.dynamics_ddp((xinit, beta))
        else:
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

            if self.dynamics_ddp is not None:
                _, metrics_ = self.dynamics_ddp((yinit, beta))
            else:
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
        # self.optimizer.synchronize()

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

        if isinstance(loss, Tensor):
            loss = loss.item()

        metrics['loss'] = loss
        if self.config.dynamics.verbose:
            with torch.no_grad():
                lmetrics = self.loss_fn.lattice_metrics(xinit=xinit, xout=xout)
                metrics.update(lmetrics)

        return xout.detach(), metrics

    def train_epoch_rich(
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
        table = Table(
            box=box.HORIZONTALS,
            row_styles=['dim', 'none'],
        )

        nepoch = self.steps.nepoch if nepoch is None else nepoch
        assert isinstance(nepoch, int)
        nepoch *= extend
        losses = []
        # ctx = self.get_context_manager(table)
        # with ctx:
        printer = self.get_printer(job_type='train')
        for epoch in trange(nepoch, disable=(not self._is_chief)):
            self.timers['train'].start()
            x, metrics = self.train_step((x, beta))  # type:ignore
            dt = self.timers['train'].stop()
            losses.append(metrics['loss'])
            if (
                    epoch == 0
                    # or self.should_print(epoch)
                    or self.should_log(epoch)
            ):
                record = {
                    'era': era,
                    'epoch': epoch,
                    'tstep': self._gstep,
                    'dt': dt,
                    'beta': beta,
                    'loss': metrics.pop('loss', None),
                    'dQsin': metrics.pop('dQsin', None),
                    'dQint': metrics.pop('dQint', None)
                }
                record.update(metrics)
                avgs, summary = self.record_metrics(
                    run=run,
                    arun=arun,
                    step=self._gstep,
                    writer=writer,
                    metrics=record,
                    job_type='train',
                    model=self.dynamics,
                    optimizer=self._optimizer,
                )
                rows[self._gstep] = avgs
                summaries.append(summary)

                if (
                        epoch > 0 and
                        self.should_log(epoch)
                        # self.should_print(epoch)
                        # and not isinstance(ctx, Live)
                ):
                    if printer is not None:
                        printer.log(avgs)
                    else:
                        log.info(summary)

                # table = self.update_table(
                #     table=table,
                #     avgs=avgs,
                #     step=epoch,
                # )

                if avgs.get('acc', 1.0) < 1e-5:
                    self.reset_optimizer()
                    self.warning('Chains are stuck! Re-drawing x !')
                    x = self.draw_x()

            self._gstep += 1
            # if isinstance(ctx, Live):
            #     ctx.console.clear()
            #     ctx.console.clear_live()

        if printer is not None:
            printer.finalize()

        data = {
            'rows': rows,
            'table': table,
            'losses': losses,
            'summaries': summaries,
        }

        return x, data

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
        # record = {'era': 0, 'epoch': 0, 'beta': 0.0, 'dt': 0.0}
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
                # ctx._redirect_stdout = T
                ctx.console.clear_live()
                # ctx.console.rule(tstr)
                ctx.update(table)

            for epoch in range(nepoch):
                self.timers['train'].start()
                x, metrics = self.train_step((x, beta))  # type:ignore
                dt = self.timers['train'].stop()
                losses.append(metrics['loss'])
                # if self.should_emit(epoch, nepoch):
                # if (
                #         self._is_chief and (
                #             self.should_print(epoch)
                #             or self.should_log(epoch)
                #         )
                # ):
                if (
                        epoch == 0
                        # or self.should_print(epoch)
                        or self.should_log(epoch)
                ):
                    record = {
                        'era': era,
                        'epoch': epoch,
                        'tstep': self._gstep,
                        'dt': dt,
                        'beta': beta,
                        'loss': metrics.pop('loss', None),
                        'dQsin': metrics.pop('dQsin', None),
                        'dQint': metrics.pop('dQint', None)
                    }
                    record.update(metrics)
                    avgs, summary = self.record_metrics(
                        run=run,
                        arun=arun,
                        step=self._gstep,
                        writer=writer,
                        metrics=record,
                        job_type='train',
                        model=self.dynamics,
                        optimizer=self._optimizer,
                    )
                    rows[self._gstep] = avgs
                    summaries.append(summary)

                    if (
                            self.should_print(epoch)
                            # and not isinstance(ctx, Live)
                    ):
                        log.info(summary)

                    table = self.update_table(
                        table=table,
                        avgs=avgs,
                        step=epoch,
                    )
                    # if epoch == 0:
                    #     table = add_columns(avgs, table)
                    # else:
                    #     table.add_row(*[f'{v}' for _, v in avgs.items()])

                    if avgs.get('acc', 1.0) < 1e-5:
                        self.reset_optimizer()
                        self.warning('Chains are stuck! Re-drawing x !')
                        x = self.draw_x()

                self._gstep += 1
                if isinstance(ctx, Live):
                    ctx.console.clear()
                    ctx.console.clear_live()

        data = {
            'rows': rows,
            'table': table,
            'losses': losses,
            'summaries': summaries,
        }

        return x, data

    def _setup_training(
            self,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            train_dir: Optional[os.PathLike] = None,
            nera: Optional[int] = None,
            nepoch: Optional[int] = None,
            beta: Optional[float | list[float] | dict[str, float]] = None,
    ) -> dict:
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
        # nprint = min(
        #     getattr(self.steps, 'print', int(nepoch // 10)),
        #     int(nepoch // 5)
        # )
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
        return {
            'x': x,
            'nera': nera,
            'nepoch': nepoch,
            'extend': extend,
            'betas': betas,
            'train_dir': train_dir,
            'beta_final': beta_final,
        }

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
        self.dynamics.train()
        setup = self._setup_training(
            x=x,
            skip=skip,
            train_dir=train_dir,
            nera=nera,
            nepoch=nepoch,
            beta=beta,
        )
        era = 0
        epoch = 0
        extend = 1
        x = setup['x']
        nera = setup['nera']
        betas = setup['betas']
        nepoch = setup['nepoch']
        extend = setup['extend']
        train_dir = setup['train_dir']
        beta_final = setup['beta_final']
        assert x is not None
        assert nera is not None
        assert train_dir is not None
        # for era in trange(nera, disable=(not self._is_chief)):
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
            x, edata = self.train_epoch_rich(
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

            self.rows['train'][str(era)] = edata['rows']
            self.tables['train'][str(era)] = edata['table']
            self.summaries['train'][str(era)] = edata['summaries']
            losses = torch.Tensor([i for i in edata['losses'][1:]])
            if self.config.annealing_schedule.dynamic:
                dy_avg = (losses[1:] - losses[:-1]).mean().item()
                if dy_avg > 0:
                    b -= (b / 10.)  # self.config.annealing_schedule._dbeta
                else:
                    b += (b / 10.)  # self.config.annealing_schedule._dbeta

            if (era + 1) == nera or (era + 1) % 5 == 0:
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

    def train_dynamic(
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
        self.dynamics.train()
        setup = self._setup_training(
            x=x,
            skip=skip,
            train_dir=train_dir,
            nera=nera,
            nepoch=nepoch,
            beta=beta,
        )
        era = 0
        epoch = 0
        extend = 1
        x = setup['x']
        nera = setup['nera']
        betas = setup['betas']
        nepoch = setup['nepoch']
        extend = setup['extend']
        train_dir = setup['train_dir']
        beta_final = setup['beta_final']
        b = torch.tensor(betas.get(str(era), beta_final))
        assert x is not None
        assert nera is not None
        assert train_dir is not None
        while b < beta_final:
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

            losses = torch.stack(edata['losses'][1:])
            if self.config.annealing_schedule.dynamic:
                dy_avg = (losses[1:] - losses[:-1]).mean().item()
                if dy_avg > 0:
                    b -= (b / 10.)  # self.config.annealing_schedule._dbeta
                else:
                    b += (b / 10.)  # self.config.annealing_schedule._dbeta

            self.rows['train'][str(era)] = edata['rows']
            self.tables['train'][str(era)] = edata['table']
            self.summaries['train'][str(era)] = edata['summaries']

            if (era + 1) == nera or (era + 1) % 5 == 0:
                self.save_ckpt(era, epoch, train_dir, run=run)

            if self._is_chief:
                log.info(f'Saving took: {time.time() - st0:<5g}s')
                log.info(f'Era {era} took: {time.time() - epoch_start:<5g}s')

            era += 1

        return {
            'timer': self.timers['train'],
            'rows': self.rows['train'],
            'summaries': self.summaries['train'],
            'history': self.histories['train'],
            'tables': self.tables['train'],
        }

    def metric_to_numpy(
            self,
            metric: Tensor | list | np.ndarray | float | None,
    ) -> np.ndarray:
        if isinstance(metric, float):
            return np.array(metric)
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
            try:
                metric = torch.Tensor(metric)
            except TypeError:
                metric = torch.tensor(0.0)

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
