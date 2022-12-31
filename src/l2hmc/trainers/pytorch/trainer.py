"""
trainer.py

Implements methods for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from collections import defaultdict
from contextlib import nullcontext
import logging
import os
import json
from pathlib import Path
import time
from typing import Any, Callable, Optional

import aim
from aim import Distribution
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
from rich import box, print_json
from rich.live import Live
# from rich.logging import RichHandler
from rich.table import Table
# from rich_logger import RichTablePrinter
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import wandb

# from l2hmc.utils.logger import get_pylogger

import l2hmc.configs as configs
from l2hmc.common import (
    ScalarLike,
    TensorLike,
    get_timestamp,
)
from l2hmc.utils.dist import setup_torch_distributed
from l2hmc.configs import CHECKPOINTS_DIR, ExperimentConfig, CONF_DIR
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
# from l2hmc.utils.rich import get_console
# from l2hmc.utils.rich_logger import LOGGER_FIELDS
from l2hmc.utils.step_timer import StepTimer
# WIDTH = int(os.environ.get('COLUMNS', 150))

# from tqdm.rich import trange
from tqdm.auto import trange


# console = get_console()
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(message)s",
#     datefmt="%X",
#     handlers=[
#         RichHandler(
#             rich_tracebacks=True,
#             tracebacks_show_locals=True,
#             console=console,
#             enable_link_path=False,
#             show_path=False,
#         )
#     ]
# )
# log = logging.getLogger(__name__)

# handler = RichHandler(
#     rich_tracebacks=True,
#     show_path=False,
#     console=console,
# )
# log.handlers = [handler]

# logging.addLevelName(70, 'CUSTOM')

# log = logging.getLogger(__name__)
# log = get_pylogger(__name__)
log = logging.getLogger(__name__)
# log.setLevel('INFO')

Tensor = torch.Tensor
Module = torch.nn.modules.Module

WITH_CUDA = torch.cuda.is_available()

GROUPS = {
    'U1': U1Phase(),
    'SU3': SU3(),
}


def grab(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def load_ds_config(fpath: os.PathLike) -> dict:
    ds_config_path = Path(fpath)
    # assert ds_config_path.is_file()
    with ds_config_path.open('r') as f:
        ds_config = json.load(f)

    return ds_config
    

class Trainer(BaseTrainer):
    def __init__(
            self,
            cfg: DictConfig | ExperimentConfig,
            build_networks: bool = True,
            ckpt_dir: Optional[os.PathLike] = None,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
    ) -> None:
        super().__init__(cfg=cfg, keep=keep, skip=skip)
        assert self.config.dynamics.group.upper() in ['U1', 'SU3']
        if self.config.dynamics.group == 'U1':
            self.g = U1Phase()
        elif self.config.dynamics.group == 'SU3':
            self.g = SU3()
        else:
            raise ValueError
        # if not isinstance(self.config, ExperimentConfig):
        self.config: ExperimentConfig = instantiate(cfg)
        # skip_tracking = os.environ.get('SKIP_TRACKING', False)
        # self.verbose = not skip_tracking
        # assert isinstance(
        #     self.config,
        #     (configs.ExperimentConfig, ExperimentConfig)
        # )
        self.clip_norm = self.config.learning_rate.clip_norm
        self._lr_warmup = torch.linspace(
            self.config.learning_rate.min_lr,
            self.config.learning_rate.lr_init,
            2 * self.steps.nepoch
        )
        # assert isinstance(self.dynamics, Dynamics)
        dsetup = setup_torch_distributed(self.config.backend)
        self._dtype = torch.get_default_dtype()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.size = dsetup['size']
        self.rank = dsetup['rank']
        self.local_rank = dsetup['local_rank']
        self._is_chief = (self.local_rank == 0 and self.rank == 0)
        # self._is_chief: bool = self.check_if_chief()
        self._with_cuda = torch.cuda.is_available()
        self.lattice = self.build_lattice()
        self.loss_fn = self.build_loss_fn()
        self.dynamics: Dynamics = self.build_dynamics(
            build_networks=build_networks,
        )
        self.ckpt_dir = (
            Path(CHECKPOINTS_DIR).joinpath('checkpoints')
            if ckpt_dir is None
            else Path(ckpt_dir).resolve()
        )
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)

        # --------------------------------------------------------
        # NOTE: 
        #   - If we want to try and resume training,
        #     `self.load_ckpt()` will attempt to find a 
        #     the most recently saved checkpoint that is
        #     compatible with the current specified architecture.
        # --------------------------------------------------------
        if self.config.restore:
            output = self.load_ckpt()
            self.dynamics: Dynamics = output['dynamics']
            # self._optimizer: torch.optim.Optimizer = output['optimizer']
            ckpt: dict = output['ckpt']
            self._gstep = ckpt.get('gstep', ckpt.get('step', 0))
            if self._is_chief:
                self.warning(
                    f'Restoring global step from ckpt! '
                    f'self._gstep: {self._gstep}'
                )
        else:
            self._gstep = 0

        if self.config.dynamics.group == 'U1':
            self.warning('Using `torch.optim.Adam` optimizer')
            self._optimizer = torch.optim.Adam(
                self.dynamics.parameters(),
                lr=self.config.learning_rate.lr_init
            )
        else:
            self.warning('Using `torch.optim.SGD` optimizer')
            self._optimizer = torch.optim.SGD(
                self.dynamics.parameters(),
                lr=self.config.learning_rate.lr_init,
            )

        self.use_fp16 = False
        self.ds_config = None
        self.dynamics_engine = None
        if self.config.backend == 'DDP':
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.optimizer = self._optimizer
            self.dynamics_engine = DDP(self.dynamics)

        elif self.config.backend.lower() in ['ds', 'deepspeed']:
            import deepspeed
            # if self.config.ds_config_path is None:
            ds_config_path = Path(CONF_DIR).joinpath('ds_config.json')
            # else:
            #     ds_config_path = Path(self.config.ds_config_path)
            # assert ds_config_path.is_file()
            self.ds_config = load_ds_config(ds_config_path)
            self.info(f'Loaded DeepSpeed config from: {ds_config_path}')
            if self._is_chief:
                print_json(json.dumps(self.ds_config, indent=4))
            if self._with_cuda:
                self.dynamics.cuda()
            if self.ds_config['fp16']['enabled']:
                self.dynamics = self.dynamics.to(torch.half)
            params = filter(
                lambda p: p.requires_grad,
                self.dynamics.parameters()
            )
            engine, optimizer, _, _ = deepspeed.initialize(
                model=self.dynamics,
                model_parameters=params,  # type:ignore
                # optimizer=self._optimizer,
                config=self.ds_config
            )
            assert engine is not None
            assert optimizer is not None
            self.dynamics_engine = engine
            self.optimizer = optimizer
            self.use_fp16 = self.dynamics_engine.fp16_enabled()
            self._device = self.dynamics_engine.local_rank
            if self.use_fp16:
                self._dtype = torch.half

        elif self.config.backend in ['hvd', 'horovod']:
            import horovod.torch as hvd
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

        assert (
            isinstance(self.dynamics, Dynamics)
            and isinstance(self.dynamics, nn.Module)
            and str(self.config.dynamics.group).upper() in ['U1', 'SU3']
        )


    def warning(self, s: str) -> None:
        if self._is_chief:
            log.warning(s)

    def info(self, s: str) -> None:
        if self._is_chief:
            log.info(s)

    def distribute_dynamics(
            self,
            dynamics: Optional[Dynamics] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            backend: Optional[str] = None,
            build_networks: bool = True,
            ds_config: Optional[dict] = None,
    ):  # tuple[Dynamics, torch.optim.Optimizer]:
        # model = self.dynamics if model is None else model
        # optimizer = self.optimizer if optimizer is None else optimizer
        # backend = self.config.backend if backend is None else backend

        # if dynamics is None:
        #     dynamics = (
        #         self.build_dynamics(build_networks)
        #         if self.dynamics is None else self.dynamics
        #     )

        # if self._with_cuda:
        #     dynamics.cuda()

        # if optimizer is None:
        #     optimizer = (
        #         self.build_optimizer()
        #         if self.optimizer is None else self.optimizer
        #     )

        # self.load_ckpt(
        #     dynamics,
        #     optimizer,
        #     build_networks=build_networks
        # )

        # assert dynamics is not None and optimizer is not None
        # if backend.lower() in ['deepspeed', 'ds']:
        #     import deepspeed
        #     model, optimizer, _, _ = deepspeed.initialize(
        #         model=dynamics,
        #         model_parameters=dynamics.parameters(),  # type:ignore
        #         optimizer=optimizer,
        #         config=ds_config
        #     )
        #     assert model is not None
        #     assert optimizer is not None
        #     return (model, optimizer)

        # elif backend.lower() in ['hvd', 'horovod']:
        #     import horovod.torch as hvd
        #     hvd.broadcast_parameters(dynamics.state_dict(), root_rank=0)
        #     hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # assert dynamics is not None
        # assert optimizer is not None
        # return (dynamics, optimizer)
        pass

    def draw_x(self):
        return self.g.random(
            list(self.config.dynamics.xshape)
        ).flatten(1)

    def reset_optimizer(self):
        if self._is_chief:
            import horovod.torch as hvd
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

    def build_optimizer(
            self,
            dynamics: Optional[Dynamics] = None,
            build_networks: bool = True,
    ) -> torch.optim.Optimizer:
        # # TODO: Expand method, re-build LR scheduler, etc
        # # TODO: Replace `LearningmodelRateConfig` with `OptimizerConfig`
        # # TODO: Optionally, break up in to lrScheduler, OptimizerConfig ?
        # lr = self.config.learning_rate.lr_init
        # assert isinstance(self.dynamics, Dynamics)
        # return torch.optim.Adam(self.dynamics.parameters(), lr=lr)
        if dynamics is None:
            dynamics = self.build_dynamics(build_networks=build_networks)

        assert dynamics is not None
        if self.config.dynamics.group == 'U1':
            optimizer = torch.optim.Adam(
                dynamics.parameters(),
                lr=self.config.learning_rate.lr_init
            )
        else:
            optimizer = torch.optim.SGD(
                dynamics.parameters(),
                lr=self.config.learning_rate.lr_init
            )

        return optimizer

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

    def get_lr(self, step: int) -> float:
        if step < len(self._lr_warmup):
            return self.config.learning_rate.lr_init
        return self.config.learning_rate.lr_init

    def build_lr_schedule(self):
        return LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda step: self.get_lr(step)
        )

    def save_ckpt(
            self,
            era: int,
            epoch: int,
            metrics: Optional[dict] = None,
            run: Optional[Any] = None,
    ) -> None:
        if not self._is_chief:
            return

        tstamp = get_timestamp('%Y-%m-%d-%H%M%S')
        step = self._gstep
        ckpt_file = self.ckpt_dir.joinpath(f'ckpt-{era}-{epoch}-{step}-{tstamp}.tar')
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
            'gstep': self._gstep,
            'model_state_dict': self.dynamics.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if metrics is not None:
            ckpt.update(metrics)

        torch.save(ckpt, ckpt_file)
        modelfile = self.ckpt_dir.joinpath('model.pth')
        torch.save(self.dynamics.state_dict(), modelfile)
        if run is not None:
            assert run is wandb.run
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(modelfile.as_posix())
            run.log_artifact(artifact)

    def load_ckpt(
            self,
            dynamics: Optional[Dynamics] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            build_networks: bool = True,
            era: Optional[int] = None,
            epoch: Optional[int] = None,
    ) -> dict:
        if dynamics is None:
            dynamics = (
                self.dynamics if self.dynamics is not None
                else self.build_dynamics(build_networks=build_networks)
                # dynamics = self.build_dynamics(build_networks=build_networks)
            )

        if optimizer is None:
            optimizer = (
                self.optimizer if self.optimizer is not None
                else self.build_optimizer()
            )

        output = {
            'dynamics': dynamics,
            'optimizer': optimizer,
            'ckpt': {}
        }
        # if not self._is_chief:
        #     return output

        ckpt_file = None
        ckpts = [
            Path(self.ckpt_dir).joinpath(i)
            for i in os.listdir(self.ckpt_dir)
            if i.endswith('.tar')
        ]

        log.info(f'Looking for checkpoints in:\n {self.ckpt_dir}')
        if len(ckpts) == 0:
            log.warning('No checkpoints found to load from')
            # return (dynamics, optimizer)
            return output

        if era is not None:
            match = f'ckpt-{era}'
            if epoch is not None:
                match += f'-{epoch}'
            for ckpt in ckpts:
                if match in ckpt.as_posix():
                    ckpt_file = ckpt
        else:
            ckpts = sorted(
                ckpts,
                key=lambda t: os.stat(t).st_mtime
            )
            ckpt_file = ckpts[-1]

        modelfile = self.ckpt_dir.joinpath('model.pth')
        if modelfile is not None:
            log.info(f'Loading model from: {modelfile}')
            dynamics.load_state_dict(torch.load(modelfile))
            output['modelfile'] = modelfile
        if ckpt_file is not None:
            ckpt_file = Path(self.ckpt_dir).joinpath(ckpt_file)
            log.info(f'Loading checkpoint from: {ckpt_file}')
            ckpt = torch.load(ckpt_file)
            output['ckpt'] = ckpt
            output['ckpt_file'] = ckpt_file
            # if (gstep := ckpt.get('gstep', None)) is not None:
            #     self._gstep = gstep
                    
            # if isinstance(optimizer, torch.optim.Optimizer):
            #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # return dynamics, optimizer
        return output

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
        if self.dynamics_engine is not None:
            xout, metrics = self.dynamics_engine((xinit, beta))
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
            eps: Optional[float] = None,
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
        beta = torch.tensor(beta).to(self._device)
        if WITH_CUDA:
            xinit = xinit.cuda()
            beta = beta.cuda()
            # xinit, beta = xinit.cuda(), torch.tensor(beta).cuda()
        if self.use_fp16:
            xinit = xinit.half()
            beta = beta.half()
            # self.dynamics_engine = self.dynamics_engine.to(xinit.dtype)

        if self.dynamics_engine is not None:
            xout, metrics = self.dynamics_engine((xinit, beta))
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

    def get_context_manager(self, table: Table) -> Live | nullcontext:
        make_live = (
            self._is_chief
            and self.size == 1          # not worth the trouble when dist.
            and not is_interactive()    # AND not in a jupyter / ipython kernel
            and int(get_width()) > 100  # make sure wide enough to fit table
        )
        if make_live:
            return Live(
                table,
                # screen=True,
                transient=True,
                # redirect_stdout=True,
                # auto_refresh=False,
                console=self.console,
                vertical_overflow='visible'
            )

        return nullcontext()

    def get_printer(
            self,
            job_type: str,  # pyright:ignore
    ) -> None:
        # ) -> RichTablePrinter | None:
        # if self._is_chief and int(get_width()) > 100:
        #     printer = RichTablePrinter(
        #         key=f'{job_type[0]}step',
        #         fields=LOGGER_FIELDS  # type:ignore
        #     )

        #     printer.hijack_tqdm()
        #     # printer.expand = True

        #     return printer
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
            nprint: Optional[int] = None,
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
        nprint = (
            max(1, min(50, eval_steps // 50))
            if nprint is None else nprint
        )
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
            'nprint': nprint,
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
            nprint: Optional[int] = None,
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
            nprint=nprint,
        )
        x = setup['x']
        eps = setup['eps']
        beta = setup['beta']
        table = setup['table']
        nleapfrog = setup['nleapfrog']
        eval_steps = setup['eval_steps']
        assert x is not None and beta is not None
        nlog = setup.get('nlog', self.config.steps.log)
        nprint = setup.get('nprint', self.config.steps.print)
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

        def _should_emit(step):
            return (step % nlog == 0 or step % nprint == 0)

        self.dynamics.eval()
        # printer = self.get_printer(job_type=job_type)
        x = self.warmup(beta)
        with self.get_context_manager(table) as ctx:
            for step in trange(
                    eval_steps,
                    dynamic_ncols=True,
                    disable=(not self._is_chief),
            ):
                # if isinstance(ctx, Live):
                #     ctx.console.clear()
                #     ctx.refresh()

                timer.start()
                x, metrics = eval_fn((x, beta))
                dt = timer.stop()
                # if (
                #         step >= 0 and
                #         (step % setup['nlog'] == 0
                #          or step % setup['nprint'] == 0)
                # ):
                if _should_emit(step):
                    if isinstance(ctx, Live):
                        ctx.console.clear_live()
                        ctx.console.clear()
                        ctx.refresh()
                    record = {
                        f'{job_type[0]}step': step,
                        'dt': dt,
                        'beta': beta,
                        'loss': metrics.pop('loss', None),
                        'dQsin': metrics.pop('dQsin', None),
                        'dQint': metrics.pop('dQint', None),
                    }
                    record.update(metrics)
                    avgs, summary = self.record_metrics(run=run,
                                                        arun=arun,
                                                        step=step,
                                                        writer=writer,
                                                        metrics=record,
                                                        job_type=job_type)
                    summaries.append(summary)
                    table = self.update_table(
                        table=table,
                        step=step,
                        avgs=avgs,
                    )

                    if (
                            step % nprint == 0
                            # step % setup['nprint'] == 0
                            # and (not isinstance(ctx, Live))
                    ):
                        log.info(summary)
                        if isinstance(ctx, Live):
                            ctx.refresh()

                    if avgs.get('acc', 1.0) < 1e-5:
                        if stuck_counter < patience:
                            stuck_counter += 1
                        else:
                            log.warning('Chains are stuck! Redrawing x')
                            x = self.lattice.random()
                            stuck_counter = 0

                    if job_type == 'hmc' and dynamic_step_size:
                        acc = record.get('acc_mask', None)
                        record['eps'] = eps
                        if acc is not None and eps is not None:
                            acc_avg = acc.mean()
                            if acc_avg < 0.66:
                                eps -= (eps / 10.)
                            else:
                                eps += (eps / 10.)

                    # if isinstance(ctx, Live):
                    #     ctx.console.clear_live()
                    #     # ctx.refresh()

        if isinstance(ctx, Live):
            ctx.console.clear_live()

        tables[str(0)] = setup['table']
        self.dynamics.train()

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
        # xinit = xinit.to(self._device).to(self._dtype)
        # # beta = beta.to(self._device).to(self._dtype)
        xinit = self.g.compat_proj(xinit.reshape(self.xshape))
        beta = torch.tensor(beta) if isinstance(beta, float) else beta
        if WITH_CUDA:
            if self.config.dynamics.group == 'U1':
                # Send to GPU with specified (real) precision
                xinit = xinit.to(self._device).to(self._dtype)
                beta = beta.to(self._device).to(self._dtype)
                # xinit, beta = xinit.cuda(), beta.cuda()

        if self.use_fp16:
            xinit = xinit.half()
            beta = beta.half()

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
        # xinit = xinit.to(self._dtype).to(self._device)
        # beta = torch.tensor(beta).to(self._dtype).to(self._device)
        if self.use_fp16:
            xinit = xinit.half()
            beta = torch.tensor(beta).half()

        if self.dynamics_engine is not None:
            xout, metrics = self.dynamics_engine((xinit, beta))
        else:
            xout, metrics = self.dynamics((xinit, beta))

        xprop = metrics.pop('mc_states').proposed.x

        # [2.] Calc loss
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])

        if (aw := self.config.loss.aux_weight) > 0:
            yinit = self.g.random(xout.shape)
            yinit.requires_grad_(True)
            if WITH_CUDA:
                yinit = yinit.cuda()

            if self.use_fp16:
                yinit = yinit.half()

            if self.dynamics_engine is not None:
                _, metrics_ = self.dynamics_engine((yinit, beta))
            else:
                _, metrics_ = self.dynamics((yinit, beta))

            yprop = metrics_.pop('mc_states').proposed.x
            aux_loss = aw * self.loss_fn(x_init=yinit,
                                         x_prop=yprop,
                                         acc=metrics_['acc'])
            loss += aw * aux_loss

        # # [3.] Backpropagate gradients
        # self.accelerator.backward(loss)
        if self.config.backend.lower() in ['ds', 'deepspeed']:
            if self.dynamics_engine is not None:
                self.dynamics_engine.backward(loss)  # type:ignore
                self.dynamics_engine.step()  # type:ignore
        else:
            loss.backward()
            if self.config.learning_rate.clip_norm > 0.0:
                torch.nn.utils.clip_grad.clip_grad_norm(
                    self.dynamics.parameters(),
                    max_norm=self.clip_norm
                )
            self.optimizer.step()


        if isinstance(loss, Tensor):
            loss = loss.item()

        metrics['loss'] = loss
        if self.config.dynamics.verbose:
            with torch.no_grad():
                lmetrics = self.loss_fn.lattice_metrics(xinit=xinit, xout=xout)
                metrics.update(lmetrics)

        return xout.detach(), metrics

    def train_step_detailed(
            self,
            x: Optional[Tensor] = None,
            beta: Optional[Tensor | float] = None,
            era: int = 0,
            epoch: int = 0,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            rows: Optional[dict] = None,
            summaries: Optional[list] = None,
            verbose: bool = True,
    ) -> tuple[Tensor, dict]:
        """Logic for performing a single training step"""
        if x is None:
            x = self.dynamics.lattice.random()
        if beta is None:
            beta = self.config.annealing_schedule.beta_init

        if isinstance(beta, float):
            beta = torch.tensor(beta).to(self._dtype).to(self._device)

        self.timers['train'].start()
        xout, metrics = self.train_step((x, beta))
        dt = self.timers['train'].stop()
        record = {
            'era': era,
            'epoch': epoch,
            'tstep': self._gstep,
            'dt': dt,
            'beta': beta,
            'loss': metrics.pop('loss', None),
            'dQsin': metrics.pop('dQsin', None),
            'dQint': metrics.pop('dQint', None),
            **metrics,
        }
        # record.update(metrics)
        avgs, summary = self.record_metrics(
            run=run,
            arun=arun,
            step=self._gstep,
            writer=writer,
            metrics=record,
            job_type='train',
            model=self.dynamics,
            optimizer=self.optimizer,
        )
        if rows is not None:
            rows[self._gstep] = avgs
        if summaries is not None:
            summaries.append(summary)

        if verbose:
            log.info(summary)

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
            nprint: Optional[int] = None,
            nlog: Optional[int] = None,
    ) -> tuple[Tensor, dict]:
        self.dynamics.train()
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

        log_freq = self.steps.log if nlog is None else nlog
        print_freq = self.steps.print if nprint is None else nprint
        # log.info(f'log_freq: {log_freq}')
        # log.info(f'print_freq: {print_freq}')

        def should_print(epoch):
            return (self._is_chief and (epoch % print_freq == 0))

        def should_log(epoch):
            return (self._is_chief and (epoch % log_freq == 0))

        with ctx:
            if isinstance(ctx, Live):
                # tstr = ' '.join([
                #     f'ERA: {era} / {self.steps.nera - 1}',
                #     f'BETA: {beta:.3f}',
                # ])
                # ctx.console.rule(tstr)
                ctx.console.clear_live()
                ctx.update(table)

            for epoch in trange(
                    nepoch,
                    dynamic_ncols=True,
                    disable=(not self._is_chief),
            ):
                self.timers['train'].start()
                x, metrics = self.train_step((x, beta))  # type:ignore
                self._gstep += 1
                dt = self.timers['train'].stop()
                losses.append(metrics['loss'])
                if (should_log(epoch) or should_print(epoch)):
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
                        optimizer=self.optimizer,
                    )
                    rows[self._gstep] = avgs
                    summaries.append(summary)

                    if (
                            should_print(epoch)
                            # and not isinstance(ctx, Live)
                    ):
                        log.info(summary)
                    table = self.update_table(
                        table=table,
                        avgs=avgs,
                        step=epoch,
                    )
                    if isinstance(ctx, Live):
                        ctx.console.clear()
                        ctx.console.clear_live()
                        ctx.refresh()

                    if avgs.get('acc', 1.0) < 1e-5:
                        self.reset_optimizer()
                        self.warning('Chains are stuck! Re-drawing x !')
                        x = self.draw_x()

                    # if isinstance(ctx, Live):
                    #     ctx.console.clear()
                    #     ctx.console.clear_live()

                if isinstance(ctx, Live):
                    ctx.console.clear()
                    ctx.console.clear_live()
                    ctx.refresh()

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
        # nprint = min(
        #     getattr(self.steps, 'print', int(nepoch // 10)),
        #     int(nepoch // 5)
        # )

        if x is None:
            x = self.g.random(list(self.xshape)).flatten(1)

        nera = self.config.steps.nera if nera is None else nera
        nepoch = self.config.steps.nepoch if nepoch is None else nepoch
        assert nera is not None and isinstance(nera, int)
        assert nepoch is not None and isinstance(nepoch, int)
        if beta is None:
            betas = self.config.annealing_schedule.setup(
                nera=nera,
                nepoch=nepoch
            )
        elif isinstance(beta, (list, np.ndarray)):
            nera = len(beta)
            betas = {f'{i}': b for i, b in zip(range(nera), beta)}
        elif isinstance(beta, (int, float)):
            betas = {f'{i}': b for i, b in zip(range(nera), nera * [beta])}
        elif isinstance(beta, dict):
            nera = len(list(beta.keys()))
            betas = {f'{i}': b for i, b in beta.items()}
        else:
            raise TypeError(
                'Expected `beta` to be one of: `float, list, dict`,'
                f' received: {type(beta)}'
            )

        beta_final = list(betas.values())[-1]
        assert beta_final is not None and isinstance(beta_final, float)
        return {
            'x': x,
            'nera': nera,
            'nepoch': nepoch,
            'betas': betas,
            'train_dir': train_dir,
            'beta_final': beta_final,
        }

    def warmup(
            self,
            beta: float,
            nsteps: int = 100,
            tol: float = 1e-3,
    ) -> Tensor:
        self.dynamics.eval()
        x = self.dynamics.lattice.random().to(self._dtype).to(self._device)
        from l2hmc.lattice.u1.pytorch.lattice import plaq_exact
        # if not isinstance(beta, Tensor):
        #     beta = torch.tensor(beta)

        btensor = torch.tensor(beta, dtype=torch.float)
        # beta = beta.to(self._dtype).to(self._device)
        # btensor = torch.tensor(beta).to(self._dtype).to(self._device)
        pexact = plaq_exact(btensor).to(self._device).to(self._dtype)
        for _ in range(nsteps):
            # x, metrics = self.dynamics((x, beta))
            x, metrics = self.hmc_step((x, beta))
            plaqs = metrics.get('plaqs', None)
            if plaqs is not None:
                pdiff = (plaqs - pexact).abs().sum()
                if pdiff < tol:
                    log.warning(f'Chains thermalized! plaq_diff: {pdiff:.4f}')
                    return x

        self.dynamics.train()
        return x

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
            nprint: Optional[int] = None,
            nlog: Optional[int] = None,
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
        # extend = setup['extend']
        train_dir = setup['train_dir']
        beta_final = setup['beta_final']
        assert x is not None
        assert nera is not None
        assert train_dir is not None
        b0 = betas.get('0', beta_final)
        x = self.warmup(b0)
        # for era in trange(nera, disable=(not self._is_chief)):
        for era in range(nera):
            b = torch.tensor(betas.get(str(era), beta_final))
            if era == (nera - 1) and self.steps.extend_last_era is not None:
                extend = int(self.steps.extend_last_era)

            if self._is_chief:
                if era > 1 and str(era - 1) in self.summaries['train']:
                    esummary = self.histories['train'].era_summary(f'{era-1}')
                    log.info(f'Avgs over last era:\n {esummary}\n')

                self.console.rule(f'ERA: {era} / {nera - 1}, BETA: {b:.3f}')

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
                nprint=nprint,
                nlog=nlog,
            )

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

            if self._is_chief:
                st0 = time.time()
                self.save_ckpt(era, epoch, run=run)
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
        # extend = setup['extend']
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
                self.save_ckpt(era, epoch, run=run)

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
