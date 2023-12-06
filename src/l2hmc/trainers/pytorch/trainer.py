"""
trainer.py

Implements methods for training L2HMC sampler.
"""
from __future__ import absolute_import, annotations, division, print_function
from collections import defaultdict
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable, Optional, Sequence

import aim
from aim import Distribution
import deepspeed
from enrich.logging import RichHandler as EnRichHandler
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
from rich import box, print_json
from rich.console import ConsoleRenderable
from rich.live import Live
from rich.logging import RichHandler as RichHandler
from rich.panel import Panel
from rich.table import Table
import torch
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LambdaLR
import wandb

from l2hmc.common import ScalarLike, get_timestamp, print_dict
from l2hmc.configs import BF16_SYNONYMS, CHECKPOINTS_DIR, FP16_SYNONYMS, ExperimentConfig
from l2hmc.configs import PT_DTYPES
from l2hmc.dynamics.pytorch.dynamics import Dynamics
from l2hmc.group.su3.pytorch.group import SU3
from l2hmc.group.u1.pytorch.group import U1Phase
from l2hmc.lattice.su3.pytorch.lattice import LatticeSU3
from l2hmc.lattice.u1.pytorch.lattice import LatticeU1, plaq_exact
from l2hmc.loss.pytorch.loss import LatticeLoss
from l2hmc.network.pytorch.network import NetworkFactory
from l2hmc.trackers.pytorch.trackers import update_summaries
from l2hmc.trainers.trainer import BaseTrainer
from l2hmc.utils.dist import setup_torch_distributed
from l2hmc.utils.history import summarize_dict
import l2hmc.utils.live_plots as plotter
from l2hmc.utils.rich import is_interactive
from l2hmc.utils.rich import get_console
from l2hmc.utils.step_timer import StepTimer
# WIDTH = int(os.environ.get('COLUMNS', 150))

# from tqdm.rich import trange
# if is_interactive():
#     from tqdm.rich import trange
# else:
# from tqdm.auto import trange

log = logging.getLogger(__name__)
# log = get_logger(__name__)
lh = log.handlers if len(log.handlers) > 0 else []

console = get_console()
for h in lh:
    if isinstance(h, (RichHandler, EnRichHandler)):
        console = h.console



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
    log.info(
        f'Loading DeepSpeed Config from: {ds_config_path.as_posix()}'
    )
    if ds_config_path.suffix == '.json':
        with ds_config_path.open('r') as f:
            ds_config = json.load(f)
        return ds_config
    if ds_config_path.suffix == '.yaml':
        import yaml
        with ds_config_path.open('r') as stream:
            ds_config = dict(yaml.safe_load(stream))
        return ds_config
    raise TypeError('Unexpected FileType')


def box_header(header: str):
    # header = f'ERA: {era} / {nera}, BETA: {b:.3f}'
    headerlen = len(header) + 2
    log.info('┏' + headerlen * '━' + '┓')
    log.info(f'┃ {header} ┃')
    log.info('┗' + headerlen * '━' + '┛')


class Trainer(BaseTrainer):
    def __init__(
            self,
            cfg: DictConfig | ExperimentConfig,
            build_networks: bool = True,
            ckpt_dir: Optional[os.PathLike] = None,
            keep: Optional[str | Sequence[str]] = None,
            skip: Optional[str | Sequence[str]] = None,
    ) -> None:
        super().__init__(cfg=cfg, keep=keep, skip=skip)
        assert self.config.dynamics.group.upper() in ['U1', 'SU3']
        if self.config.dynamics.group == 'U1':
            self.g = U1Phase()
        elif self.config.dynamics.group == 'SU3':
            self.g = SU3()
        else:
            raise ValueError
        self.config: ExperimentConfig = instantiate(cfg)
        # skip_tracking = os.environ.get('SKIP_TRACKING', False)
        # self.verbose = not skip_tracking
        self.clip_norm = self.config.learning_rate.clip_norm
        self._lr_warmup = torch.linspace(
            self.config.learning_rate.min_lr,
            self.config.learning_rate.lr_init,
            2 * self.steps.nepoch
        )
        # self.use_fp16: bool = (
        #     self.config.precision.lower() in ['fp16', '16', 'half']
        # )
        self.dtype = PT_DTYPES.get(self.config.precision, None)
        assert self.dtype is not None
        dsetup: dict = setup_torch_distributed(self.config.backend)
        # self._device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.size: int = dsetup['size']
        self.rank: int = dsetup['rank']
        self.local_rank: int = dsetup['local_rank']
        self._is_orchestrator: bool = (self.local_rank == 0 and self.rank == 0)
        self._with_cuda: bool = torch.cuda.is_available()
        # self._dtype = torch.get_default_dtype()
        self._dtype = self.dtype
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        # if torch.cuda.is_available() and self.use_fp16:
        # torch.set_autocast_gpu_dtype(self.dtype)
        # self._dtype = torch.get_autocast_gpu_dtype()
        self.warning(f'Using {self.dtype} on {self.device}!')
        self.lattice = self.build_lattice()
        self.loss_fn = self.build_loss_fn()
        self.dynamics: Dynamics = self.build_dynamics(
            build_networks=build_networks,
        )
        # minlogfreq = int(self.config.steps.nepoch // 20)
        # logfreq = (
        #     minlogfreq if self.config.steps.log is None
        #     else self.config.steps.log
        # )
        self.ckpt_dir: Path = (
            Path(CHECKPOINTS_DIR).joinpath('checkpoints')
            if ckpt_dir is None
            else Path(ckpt_dir).resolve()
        )
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        # self.dynamics = self.dynamics.to(self.dtype)
        # log.warning(f'{self.dynamics.dtype=}')
        # if self.use_fp16:
        #     self.dynamics = self.dynamics.to(torch.half)
        #     # self.warning(f'Dynamics.dtype: {self.dynamics.dtype}')
        self._fstep = 0
        self._bstep = 0
        self._gstep = 0
        self._estep = 0
        self._hstep = 0
        # --------------------------------------------------------
        # NOTE:
        #   - If we want to try and resume training,
        #     `self.load_ckpt()` will attempt to find a
        #     the most recently saved checkpoint that is
        #     compatible with the current specified architecture.
        # --------------------------------------------------------
        if self.config.restore:
            output: dict = self.load_ckpt()
            self.dynamics: Dynamics = output['dynamics']
            # self._optimizer: torch.optim.Optimizer = output['optimizer']
            ckpt: dict = output['ckpt']
            self._gstep: int = ckpt.get('gstep', ckpt.get('step', 0))
            if self._is_orchestrator:
                self.warning(
                    f'Restoring global step from ckpt! '
                    f'self._gstep: {self._gstep}'
                )
        else:
            self._gstep: int = 0
        # if self.config.dynamics.group == 'U1':
        self.warning('Using `torch.optim.Adam` optimizer')
        self._optimizer = torch.optim.Adam(
            self.dynamics.parameters(),
            lr=self.config.learning_rate.lr_init
        )
        self.num_params = self.count_parameters(self.dynamics)
        self.autocast_context_train = torch.autocast(  # type:ignore
                dtype=self._dtype,
                device_type=self.device,
                enabled=(
                    self._dtype != torch.float64
                    and self.device != 'cpu'
                ),
                # 'cuda' if torch.cuda.is_available() else 'cpu'
        )
        # self.enable_autocast = (
        #     self._dtype != torch.float64
        #     and self.device != 'cpu'
        # )
        # if self.enable_autocast:
        #     self.autocast_context_train = torch.autocast(  # type:ignore
        #             dtype=self._dtype,
        #             enabled=self.enable_autocast,
        #             device_type=self.device,
        #             # 'cuda' if torch.cuda.is_available() else 'cpu'
        #     )
        # else:
        #     self.autocast_context_train = nullcontext()
        # else:
        #     self.warning('Using `torch.optim.SGD` optimizer')
        #     self._optimizer = torch.optim.SGD(
        #         self.dynamics.parameters(),
        #         lr=self.config.learning_rate.lr_init,
        #     )

        # --------------------------------------------------------------------
        # BACKEND SPECIFIC SETUP
        # ----------------------
        self.ds_config = {}
        self.grad_scaler = None  # Needed for Horovod, DDP
        self.dynamics_engine = None
        if self.config.backend == 'DDP':
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.optimizer = self._optimizer
            find_unused_parameters = (
                str(self.config.dynamics.group).lower() == 'su3'
            )
            self.dynamics_engine = DDP(
                self.dynamics,
                find_unused_parameters=find_unused_parameters,
            )
            if self._dtype != torch.float64:
                self.grad_scaler = GradScaler()
            # self.grad_scaler = GradScaler(
            #     enabled=(self._dtype != torch.float64)
            # )
        # NOTE: DeepSpeed automatically handles gradient scaling when training
        # with mixed precision.
        elif self.config.backend.lower() in ['ds', 'deepspeed']:
            self._setup_deepspeed()
        elif self.config.backend.lower() in ['hvd', 'horovod']:
            # self.grad_scaler = GradScaler()
            self._setup_horovod()
        else:
            self.optimizer = self._optimizer
        logfreq = self.config.steps.log
        log.warning(f'logging with freq {logfreq} for wandb.watch')
        # if self.rank == 0 and wandb.run is not None:
        # if self._is_chief and self.config.use_wandb and wandb.run is not None:
        if self.config.use_wandb and wandb.run is not None:
            wandb.run.watch(
                # (
                #     self.dynamics.xnet,
                #     self.dynamics.vnet,
                #     self.dynamics.xeps,
                #     self.dynamics.veps
                # ),
                # (
                #     self.dynamics,
                #     self.dynamics.xeps,
                #     self.dynamics.veps,
                # ),
                self.dynamics,
                # self.dynamics,
                log='all',
                log_freq=logfreq,
                # log_graph=True,
            )
        assert (
            isinstance(self.dynamics, Dynamics)
            and isinstance(self.dynamics, nn.Module)
            and str(self.config.dynamics.group).upper() in {'U1', 'SU3'}
        )

    def count_parameters(self, model: Optional[nn.Module] = None) -> int:
        """Count the total number of parameters in `model`."""
        model = self.dynamics if model is None else model
        num_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        log.info(f'num_params in model: {num_params}')
        if self.config.init_wandb and wandb.run is not None:
            wandb.run.config['NUM_PARAMS'] = num_params
        return num_params

    def _setup_deepspeed(self) -> None:
        # TODO: Move ds_config_path to `conf/config.yaml` to be overridable
        self.ds_config = self.prepare_ds_config()
        # if (
        #         (fp16 := self.ds_config.get('fp16', None)) is not None
        #         and fp16.get('enabled', False)
        # ):
        # if self.use_fp16:
        if self.dtype == torch.bfloat16:
            log.warning('Using `bf16` in DeepSpeed config...')
            self.ds_config |= {
                'bf16': {
                    'enabled': True,
                }
            }
            self.dynamics = self.dynamics.to(torch.bfloat16)
        if self.dtype == torch.float16:
            log.warning('Using `fp16` in DeepSpeed config...')
            self.ds_config |= {
                'fp16': {
                    'enabled': True,
                }
            }
            self.dynamics = self.dynamics.to(torch.float16)
        # if self._is_chief:
        if self.rank == 0:
            print_json(json.dumps(self.ds_config, indent=4))
        if 'optimizer' in self.ds_config.items():
            engine, optimizer, *_ = deepspeed.initialize(
                model=self.dynamics,
                config=self.ds_config,
                model_parameters=self.dynamics.parameters()  # type:ignore
                # model_parameters=filter(
                #     lambda p: p.requires_grad,
                #     self.dynamics.parameters(),
                # ),
            )
        else:
            # optimizer = self._optimizer
            engine, optimizer, *_ = deepspeed.initialize(
                model=self.dynamics,
                config=self.ds_config,
                optimizer=self._optimizer,
                model_parameters=self.dynamics.parameters()  # type:ignore
            )
        assert engine is not None
        assert optimizer is not None
        self.dynamics_engine = engine
        self.optimizer = optimizer
        # self.use_fp16 = self.dynamics_engine.fp16_enabled()
        self.device = self.dynamics_engine.local_rank
        # if self.use_fp16:
        #     self._dtype = torch.half

    def _setup_horovod(self) -> None:
        import horovod.torch as hvd
        compression = (
            hvd.Compression.fp16 if (
                self.dtype in {*BF16_SYNONYMS, *FP16_SYNONYMS}
            )
            else hvd.Compression.none
        )
        self.optimizer = hvd.DistributedOptimizer(
            self._optimizer,
            named_parameters=self.dynamics.named_parameters(),
            compression=compression,  # type: ignore
        )
        hvd.broadcast_parameters(self.dynamics.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

    def prepare_ds_config(self) -> dict:
        if self.config.backend.lower() not in ['ds', 'deepspeed']:
            return {}
        ds_config = {}
        # ds_config_path = Path(CONF_DIR).joinpath('ds_config.json')
        assert self.config.ds_config_path is not None
        ds_config = load_ds_config(self.config.ds_config_path)
        self.info(
            f'Loaded DeepSpeed config from: {self.config.ds_config_path}'
        )
        pname = 'l2hmc-qcd'
        if self.config.debug_mode:
            pname += '-debug'
        if self.config.init_wandb:
            ds_config['wandb'].update({
                'enabled': True,
                'project': pname,
                'group': f'{self.config.framework}/{self.config.backend}',
            })
        else:
            ds_config['wandb'] = {}
        opath = Path(os.getcwd()).joinpath('ds_outputs').resolve()
        # ds_config['tensorboard'] = {
        #     'enabled': True,
        #     'output_path': opath.joinpath('ds_tensorboard').as_posix(),
        # }
        ds_config['csv_monitor'] = {
            'enabled': True,
            'output_path': opath.joinpath('ds_csv_monitor').as_posix(),
        }
        ds_config.update({
            'gradient_accumulation_steps': 1,
            'train_micro_batch_size_per_gpu': 1,
        })
        ds_config['train_batch_size'] = (
            self.size
            * ds_config['gradient_accumulation_steps']
            * ds_config['train_micro_batch_size_per_gpu']
        )
        scheduler = ds_config.get('scheduler', None)
        if scheduler is not None:
            sparams = scheduler.get('params', None)
            if sparams is not None:
                ds_config['scheduler']['params'].update({
                    'warmup_num_steps': self.config.steps.nepoch,
                    'total_num_steps': (
                        self.config.steps.nera * self.config.steps.nepoch
                    )
                })
        # if not self.use_fp16:
        # if self.dtype not in {*BF16_SYNONYMS, *FP16_SYNONYMS}:
        #     fp16 = ds_config.get('fp16', None)
        #     if fp16 is not None:
        #         self.warning('Turning of `fp16` in ds_config!')
        #         ds_config.update({
        #             'fp16': {
        #                 'enabled': False,
        #             }
        #         })
        zero_opt_config = ds_config.get('zero_optimization', None)
        if zero_opt_config is not None:
            hostname = str(
                socket.gethostbyaddr(socket.gethostname())[0]
            ).lower()
            if hostname.startswith('thetagpu'):
                nvme_path = Path('/raid/scratch/').resolve()
            else:
                nvme_path = Path('/local/scratch').resolve()
            if nvme_path.exists():
                nvme_path = nvme_path.as_posix()
                self.info(f'[{hostname}] Setting NVMe path to: {nvme_path}')
                zero_opt_config['offload_param']['nvme_path'] = nvme_path
                zero_opt_config['offload_optimizer']['nvme_path'] = nvme_path
                ds_config['zero_optimization'] = zero_opt_config
        # if ds_config['scheduler'].get('params', None) is not None:
        #     ds_config['scheduler']['params'].update({
        #         'warmup_num_steps': self.config.steps.nepoch,
        #         'total_num_steps': (
        #             self.config.steps.nera * self.config.steps.nepoch
        #         )
        #     })
        self.config.set_ds_config(ds_config)
        self.ds_config = ds_config
        return ds_config

    def warning(self, s: str) -> None:
        if self._is_orchestrator:
            log.warning(s)

    def info(self, s: str) -> None:
        if self._is_orchestrator:
            log.info(s)

    def draw_x(self):
        return self.g.random(
            list(self.config.dynamics.xshape)
        ).flatten(1)

    def draw_v(self):
        return self.g.random_momentum(
            list(self.config.dynamics.xshape)
        )

    def reset_optimizer(self):
        if self._is_orchestrator:
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
        return (
            torch.optim.Adam(
                dynamics.parameters(), lr=self.config.learning_rate.lr_init
            )
            if self.config.dynamics.group == 'U1'
            else torch.optim.SGD(
                dynamics.parameters(), lr=self.config.learning_rate.lr_init
            )
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

    def get_lr(self, step: int) -> float:
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
        if self.rank != 0 or not self.config.save:
            return
        tstamp = get_timestamp('%Y-%m-%d-%H%M%S')
        step = self._gstep
        ckpt_file = self.ckpt_dir.joinpath(
            f'ckpt-{era}-{epoch}-{step}-{tstamp}.tar'
        )
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
        modelfile = self.ckpt_dir.joinpath(
            f'model-{era}-{epoch}-{step}-{tstamp}.pth'
        )
        torch.save(self.dynamics.state_dict(), modelfile)
        self.info(f'Saving checkpoint to: {ckpt_file.as_posix()}')
        self.info(f'Saving modelfile to: {modelfile.as_posix()}')
        if wandb.run is not None and self.config.init_wandb:
            # assert run is wandb.run
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(modelfile.as_posix())
            wandb.run.log_artifact(artifact)

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
        ckpts = [
            Path(self.ckpt_dir).joinpath(i)
            for i in os.listdir(self.ckpt_dir)
            if i.endswith('.tar')
        ]
        modelfiles = [
            Path(self.ckpt_dir).joinpath(i)
            for i in os.listdir(self.ckpt_dir)
            if i.endswith('.pth')
        ]

        self.info(f'Looking for checkpoints in:\n {self.ckpt_dir}')
        if not ckpts:
            self.warning('No checkpoints found to load from')
            return output

        ckpt_file = None
        modelfile = None
        if era is not None:
            cmatch = f'ckpt-{era}'
            mmatch = f'model-{era}'
            if epoch is not None:
                cmatch += f'-{epoch}'
                mmatch += f'-{epoch}'
            for ckpt in ckpts:
                if cmatch in ckpt.as_posix():
                    ckpt_file = ckpt
            for mfile in modelfiles:
                if mmatch in mfile.as_posix():
                    modelfile = mfile
        else:
            ckpts = sorted(
                ckpts,
                key=lambda t: os.stat(t).st_mtime
            )
            mfiles = sorted(
                modelfiles,
                key=lambda t: os.stat(t).st_mtime
            )
            ckpt_file = ckpts[-1]
            modelfile = mfiles[-1]

        # modelfile = self.ckpt_dir.joinpath('model.pth')
        if modelfile is not None:
            self.info(f'Loading model from: {modelfile}')
            dynamics.load_state_dict(torch.load(modelfile))
            output['modelfile'] = modelfile
        if ckpt_file is not None:
            ckpt_file = Path(self.ckpt_dir).joinpath(ckpt_file)
            self.info(f'Loading checkpoint from: {ckpt_file}')
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
        return (epoch % self.steps.log == 0 and self._is_orchestrator)

    def should_print(self, epoch):
        return (epoch % self.steps.print == 0 and self._is_orchestrator)

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

        return self._is_orchestrator and emit

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
        assert job_type in {'train', 'eval', 'hmc'}
        if step is None:
            timer = self.timers.get(job_type, None)
            if isinstance(timer, StepTimer):
                step = timer.iterations

        if step is not None:
            metrics[f'{job_type[0]}step'] = step

        if job_type == 'train' and step is not None:
            metrics['lr'] = self.get_lr(step)

        if job_type == 'eval' and 'eps' in metrics:
            _ = metrics.pop('eps', None)
        metrics.update(self.metrics_to_numpy(metrics))
        avgs = self.histories[job_type].update(metrics)
        summary = summarize_dict(avgs)
        metrics |= {
            'xeps': torch.tensor(self.dynamics.xeps),
            'veps': torch.tensor(self.dynamics.veps)
        }
        metrics |= {
            f'{k}/avg': v for k, v in avgs.items()
        }
        if (
                step is not None
                and writer is not None
        ):
            assert step is not None
            update_summaries(
                step=step,
                # model=model,
                writer=writer,
                # with_grads=True,
                metrics=metrics,
                prefix=job_type,
                optimizer=optimizer,
                use_tb=self.config.use_tb,
                use_wandb=(
                    self.config.use_wandb
                    and self.config.init_wandb
                )
            )
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
            record: dict[str, torch.Tensor | np.ndarray | list | ScalarLike],
            avgs: dict[str, ScalarLike],
            job_type: str,
            step: Optional[int],
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
    ) -> None:
        if self.local_rank != 0:
            return
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

        # if run is not None:
        if wandb.run is not None and self.config.init_wandb:
            wandb.run.log(dQdict, commit=False)
            # wandb.run.log({f'{job_type}/{k}/avg': v for k, v in avgs.items()})
            # with StopWatch(
            #         msg=f"`wandb.log({job_type}.metrics)`",
            #         wbtag=f'wblog/{job_type}',
            #         iter=step,
            #         prefix='TrackingTimers/',
            #         log_output=False,
            # ):
            #     record = {
            #         f'{job_type}/metrics/{k}': v for k, v in record.items()
            #     }
            #     record |= {
            #         f'{job_type}/metrics/{k}/avg': v for k, v in avgs.items()
            #     }
            #     try:
            #         wandb.run.log(record, commit=False)
            #         if dQdict is not None:
            #             wandb.run.log(dQdict)
            #     except ValueError:
            #         self.warning(
            #             'Unable to track record with WandB, skipping!'
            #         )
            # try:
            #     wandb.run.log({f'{job_type}.metrics': record}, commit=False)
            #     wandb.run.log({f'{job_type}.avgs': avgs})
            #     # wandb.run.log({f'wandb/{job_type}': record}, commit=False)
            #     # wandb.run.log({f'avgs/wandb.{job_type}': avgs})
            #     if dQdict is not None:
            #         wandb.run.log(dQdict, commit=False)
        if arun is not None:
            from aim import Distribution
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
        try:
            self.optimizer.zero_grad()
        except Exception:
            pass
        xinit = self.g.compat_proj(xinit)  # .to(self.accelerator.device)
        if self.dynamics_engine is not None:
            xout, metrics = self.dynamics_engine((xinit, beta))
        else:
            xout, metrics = self.dynamics((xinit, beta))
        xout = self.g.compat_proj(xout)
        xprop = self.g.compat_proj(metrics.pop('mc_states').proposed.x)
        beta = beta
        # xprop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(xinit, x_prop=xprop, acc=metrics['acc'])
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
            inputs: tuple[Tensor, float | Tensor],
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> tuple[Tensor, dict]:
        self.dynamics.eval()
        xi, beta = inputs
        beta = torch.tensor(beta) if isinstance(beta, float) else beta
        assert isinstance(beta, Tensor)
        beta = beta.to(self.device)
        xi = self.g.compat_proj(
            self.dynamics.unflatten(xi.to(self.device))
        )
        xo, metrics = self.dynamics.apply_transition_hmc(
            (xi, beta), eps=eps, nleapfrog=nleapfrog,
        )
        xp = metrics.pop('mc_states').proposed.x
        loss = self.loss_fn(x_init=xi, x_prop=xp, acc=metrics['acc'])
        if self.config.dynamics.verbose:
            lmetrics = self.loss_fn.lattice_metrics(xinit=xi, xout=xo)
            metrics.update(lmetrics)
        metrics.update({'loss': loss.item()})
        self.dynamics.train()
        self._hstep += 1
        return xo.detach(), metrics

    def eval_step(
            self,
            inputs: tuple[Tensor, float]
    ) -> tuple[Tensor, dict]:
        self.dynamics.eval()
        xinit, beta = inputs
        beta = torch.tensor(beta).to(self.device)
        xinit = self.g.compat_proj(
            self.dynamics.unflatten(xinit.to(self.device))
        )
        # with torch.autocast(  # type:ignore
        #         device_type='cuda' if WITH_CUDA else 'cpu',
        #         dtype=torch.float32
        # ):
        xout, metrics = self.dynamics((xinit, beta))
        xprop = metrics.pop('mc_states').proposed.x
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        if self.config.dynamics.verbose:
            lmetrics = self.loss_fn.lattice_metrics(xinit=xinit, xout=xout)
            metrics.update(lmetrics)
        metrics.update({
            'loss': loss.item(),
        })
        self.dynamics.train()
        self._estep += 1
        return xout.detach(), metrics

    def get_context_manager(
            self,
            renderable: ConsoleRenderable,
    ) -> Live | nullcontext:
        # if make_live := (
        #     self._is_orchestrator
        #     and self.size == 1  # not worth the trouble when dist.
        #     and not is_interactive()  # AND not in a jupyter / ipython kernel
        #     and int(get_width()) > 100  # make sure wide enough to fit table
        # ):
        #     return Live(
        #         renderable,
        #         console=console,
        #         # screen=True,
        #         transient=True,
        #         # redirect_stdout=True,
        #         auto_refresh=False,
        #         vertical_overflow='visible'
        #     )
        #
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
            skip: Optional[str | Sequence[str]] = None,
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
            self.warning(
                'Step size `eps` not specified for HMC! '
                f'Using default: {eps:.4f} for generic HMC'
            )

        if x is None:
            x = self.lattice.random()

        self.warning(f'x.shape (original): {x.shape}')
        if nchains is not None:
            if isinstance(nchains, int) and nchains > 0:
                x = x[:nchains]  # type: ignore

        assert isinstance(x, Tensor)
        if nchains is not None:
            self.warning(f'x[:nchains].shape: {x.shape}')

        table = Table(row_styles=['dim', 'none'], expand=True)
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
            # 'nprint': nprint,
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
            skip: Optional[str | Sequence[str]] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            job_type: Optional[str] = 'eval',
            nchains: Optional[int] = None,
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
            dynamic_step_size: Optional[bool] = None,
            nprint: Optional[int] = None,
            make_plots: bool = True,
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
        panel = Panel(table)
        ctx = self.get_context_manager(panel)
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
            and nlog is not None
            and nprint is not None
        )
        device_type = 'cuda' if WITH_CUDA else 'cpu'
        if device_type == 'cuda':
            fpctx = torch.autocast(  # type:ignore
                # dtype=dtype,
                device_type=device_type,
            )
        else:
            fpctx = nullcontext()

        def eval_fn(z):
            with fpctx:
                if job_type == 'hmc':
                    assert eps is not None
                    return self.hmc_step(z, eps=eps, nleapfrog=nleapfrog)
                return self.eval_step(z)

        def refresh_view():
            if isinstance(ctx, Live):
                ctx.refresh()
                # ctx.console.clear_live()

        def _should_emit(step):
            return (step % nlog == 0 or step % nprint == 0)

        plots = None
        if is_interactive() and make_plots:
            plots = plotter.init_plots()
        self.dynamics.eval()
        with ctx:
            x = self.warmup(beta=beta, x=x)
            # for step in trange(
            #         eval_steps,
            #         dynamic_ncols=True,
            #         disable=(not self._is_chief),
            #         leave=True,
            #         desc=job_type
            # ):
            for step in range(eval_steps):
                timer.start()
                x, metrics = eval_fn((x, beta))
                dt = timer.stop()

                if _should_emit(step):
                    record = {
                        f'{job_type[0]}step': step,
                        'dt': dt,
                        'beta': beta,
                        'loss': metrics.pop('loss', None),
                        'dQsin': metrics.pop('dQsin', None),
                        'dQint': metrics.pop('dQint', None),
                    }
                    record.update(metrics)
                    avgs, summary = self.record_metrics(
                        run=run,
                        arun=arun,
                        step=step,
                        writer=writer,
                        metrics=record,
                        job_type=job_type
                    )
                    summaries.append(summary)
                    table = self.update_table(
                        table=table,
                        step=step,
                        avgs=avgs,
                    )
                    if step % nprint == 0:
                        log.info(summary)
                    refresh_view()
                    if avgs.get('acc', 1.0) < 1e-5:
                        if stuck_counter < patience:
                            stuck_counter += 1
                        else:
                            self.warning('Chains are stuck! Redrawing x')
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
                    if (
                            is_interactive()
                            and self._is_orchestrator
                            and plots is not None
                    ):
                        if len(self.histories[job_type].history.keys()) == 0:
                            plotter.update_plots(
                                history=metrics,
                                plots=plots,
                                logging_steps=nlog,
                            )
                        else:
                            plotter.update_plots(
                                history=self.histories[job_type].history,
                                plots=plots,
                                logging_steps=nlog,
                            )
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

    def calc_loss(
            self,
            xinit: torch.Tensor,
            xprop: torch.Tensor,
            acc: torch.Tensor,
    ) -> torch.Tensor:
        # loss = self.loss_fn(xinit, xprop, acc)
        # if loss.isnan():
        #     log.critical(f'loss.isnan()!: {loss}')
        #     loss = torch.ones_like(loss, requires_grad=True) * 1e-6
        return self.loss_fn(xinit, xprop, acc)

    def forward_step(
            self,
            x: torch.Tensor,
            beta: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        x.requires_grad_(True)
        try:
            self.optimizer.zero_grad()
        except Exception:
            pass
        with self.autocast_context_train:
            if self.dynamics_engine is not None:
                xout, metrics = self.dynamics_engine((x, beta))
            else:
                xout, metrics = self.dynamics((x, beta))
        self._fstep += 1
        return xout, metrics

    def backward_step(
            self,
            loss: torch.Tensor
    ) -> torch.Tensor:
        """Backpropagate gradients."""
        if (
                self.config.backend.lower() in ['ds', 'deepspeed']
                and self.dynamics_engine is not None
        ):
            self.dynamics_engine.backward(loss)  # type:ignore
            self.dynamics_engine.step()  # type:ignore
        elif self.grad_scaler is None:
            loss.backward()
            if self.config.learning_rate.clip_norm > 0.0:
                torch.nn.utils.clip_grad.clip_grad_norm(
                    parameters=self.dynamics.parameters(),
                    max_norm=self.clip_norm
                )
            self.optimizer.step()
        else:
            self.grad_scaler.scale(loss).backward()  # type:ignore
            self.grad_scaler.unscale_(self.optimizer)
            if self.config.learning_rate.clip_norm > 0:
                torch.nn.utils.clip_grad.clip_grad_norm(
                    parameters=self.dynamics.parameters(),
                    max_norm=self.config.learning_rate.clip_norm
                )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        self._bstep += 1
        return loss

    def train_step(
            self,
            inputs: tuple[Tensor, Tensor | float],
    ) -> tuple[Tensor, dict]:
        """Logic for performing a single training step"""
        xinit, beta = inputs
        xinit = self.g.compat_proj(xinit.reshape(self.xshape))
        beta = torch.tensor(beta) if isinstance(beta, float) else beta
        assert isinstance(beta, Tensor)
        # ====================================================================
        # -----------------------  Train step  -------------------------------
        # ====================================================================
        # 1. Call model on inputs to generate:
        #      a. PROPOSAL config `xprop`   (before MH acc / rej)
        #      b. OUTPUT config `xout`      (after MH acc / rej)
        # 2. Calc loss using `xinit`, `xprop` and `acc` (acceptance rate)
        # 3. Backpropagate gradients and update network weights
        # --------------------------------------------------------------------
        # t0 = time.perf_counter()
        xout, metrics = self.forward_step(x=xinit, beta=beta)
        # t1 = time.perf_counter()
        xprop = metrics.pop('mc_states').proposed.x
        loss = self.calc_loss(xinit=xinit, xprop=xprop, acc=metrics['acc'])
        # xout.reshape_as()
        # t2 = time.perf_counter()
        aux_loss = 0.0
        if (aw := self.config.loss.aux_weight) > 0:
            yinit = self.dynamics.unflatten(
                self.g.random(xinit.shape).to(self.device)
            )
            _, metrics_ = self.forward_step(x=yinit, beta=beta)
            yprop = metrics_.pop('mc_states').proposed.x
            aux_loss = self.calc_loss(
                xinit=yinit,
                xprop=yprop,
                acc=metrics_['acc']
            )
            aux_loss += aw * aux_loss
        loss_tot = loss + aux_loss
        # t3 = time.perf_counter()
        loss = self.backward_step(loss_tot)
        if isinstance(loss_tot, Tensor):
            loss_tot = loss_tot.item()
        metrics['loss'] = loss_tot
        # t4 = time.perf_counter()
        if self.config.dynamics.verbose:
            with torch.no_grad():
                lmetrics = self.loss_fn.lattice_metrics(xinit=xinit, xout=xout)
                metrics.update(lmetrics)
        # t5 = timer.perf_counter()
        self._gstep += 1
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
            beta = torch.tensor(beta).to(self.device)

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

        self._gstep += 1
        return xout.detach(), record

    def eval_step_detailed(
            self,
            job_type: str,
            x: Optional[Tensor] = None,
            beta: Optional[float] = None,
            verbose: bool = True,
    ) -> tuple[Tensor, dict]:
        if x is None:
            x = self.dynamics.lattice.random()
        if beta is None:
            beta = self.config.annealing_schedule.beta_init

        # if isinstance(beta, float):
        #     beta = torch.tensor(beta).to(self.device)

        self.timers[job_type].start()
        if job_type == 'eval':
            xout, metrics = self.eval_step((x, beta))
        elif job_type == 'hmc':
            xout, metrics = self.hmc_step((x, beta))
        else:
            raise ValueError(
                f'Job type should be eval or hmc, got: {job_type}'
            )

        dt = self.timers[job_type].stop()
        record = {
            'dt': dt,
            'beta': beta,
            'loss': metrics.pop('loss', None),
            'dQsin': metrics.pop('dQsin', None),
            'dQint': metrics.pop('dQint', None),
            **metrics,
        }
        # record.update(metrics)
        _, summary = self.record_metrics(
            step=self._gstep,
            metrics=record,
            job_type=job_type,
            # run=run,
            # arun=arun,
            # writer=writer,
            # model=self.dynamics,
            # optimizer=self.optimizer,
        )
        if verbose:
            log.info(summary)

        self._estep += 1
        return xout, record

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
            warmup: bool = True,
            plots: Optional[Any] = None,
    ) -> tuple[Tensor, dict]:
        self.dynamics.train()
        rows = {}
        summaries = []
        extend = 1 if extend is None else extend
        # record = {'era': 0, 'epoch': 0, 'beta': 0.0, 'dt': 0.0}
        # from rich.layout import Layout
        table = Table(
            expand=True,
            box=box.HORIZONTALS,
            row_styles=['dim', 'none'],
        )
        panel = Panel(table)
        # layout = Layout(table)

        nepoch = self.steps.nepoch if nepoch is None else nepoch
        assert isinstance(nepoch, int)
        nepoch *= extend
        losses = []
        ctx = self.get_context_manager(panel)

        log_freq = self.steps.log if nlog is None else nlog
        print_freq = self.steps.print if nprint is None else nprint
        assert log_freq is not None and isinstance(log_freq, int)
        assert print_freq is not None and isinstance(print_freq, int)
        # log.info(f'log_freq: {log_freq}')
        # log.info(f'print_freq: {print_freq}')

        def should_print(epoch):
            return (self._is_orchestrator and (epoch % print_freq == 0))

        def should_log(epoch):
            return (self._is_orchestrator and (epoch % log_freq == 0))

        def refresh_view():
            if isinstance(ctx, Live):
                # ctx.console.clear_live()
                # ctx.update(table)
                ctx.refresh()

        patience = 10
        stuck_iters = 0
        with ctx:
            if warmup:
                wt0 = time.perf_counter()
                x = self.warmup(beta=beta, x=x)
                self.info(
                    f'Thermalizing configs @ {beta:.2f} took '
                    f'{time.perf_counter() - wt0:.4f} s'
                )

            summary = ""
            # for epoch in trange(
            #         nepoch,
            #         dynamic_ncols=True,
            #         disable=(not self._is_chief),
            #         desc='Training',
            #         leave=True
            # ):
            for epoch in range(nepoch):
                self.timers['train'].start()
                x, metrics = self.train_step((x, beta))  # type:ignore
                dt = self.timers['train'].stop()
                losses.append(metrics['loss'])
                # if (acc := metrics.get('acc', None)) is not None:
                #     if isinstance(acc, Tensor):
                #         if acc.mean() < 1e-5:
                #             # self.reset_optimizer()
                #             self.warning('Chains are stuck! Re-drawing x !')
                #             x = self.warmup(beta)

                # if (should_log(epoch) or should_print(epoch)):
                if should_log(epoch):
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

                    table = self.update_table(
                        table=table,
                        avgs=avgs,
                        step=epoch,
                    )
                    if avgs.get('acc', 1.0) < 1e-5:
                        if stuck_iters < patience:
                            stuck_iters += 1
                        else:
                            self.warning('Chains are stuck! Redrawing x')
                            x = self.lattice.random()
                            stuck_iters = 0

                    refresh_view()
                    if (
                            is_interactive()
                            and self._is_orchestrator
                            and plots is not None
                    ):
                        if len(self.histories['train'].history.keys()) == 0:
                            plotter.update_plots(
                                metrics,
                                plots,
                                logging_steps=log_freq
                            )
                        else:
                            plotter.update_plots(
                                self.histories['train'].history,
                                plots=plots,
                                logging_steps=log_freq
                            )

                if should_print(epoch):
                    refresh_view()
                    log.info(summary)

                if isinstance(ctx, Live):
                    # ctx.console.clear()
                    ctx.console.clear_live()
                    # ctx.refresh()

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
            skip: Optional[str | Sequence[str]] = None,
            train_dir: Optional[os.PathLike] = None,
            nera: Optional[int] = None,
            nepoch: Optional[int] = None,
            beta: Optional[float | Sequence[float] | dict[str, float]] = None,
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
            beta: float | torch.Tensor,
            nsteps: int = 100,
            tol: float = 1e-5,
            x: Optional[Tensor] = None,
            nchains: Optional[int] = None,
    ) -> Tensor:
        """Thermalize configs"""
        self.dynamics.eval()
        if x is None:
            x = (
                self.dynamics.lattice.random()
            ).to(self.device)

        if nchains is not None:
            x = x[:nchains]

        if isinstance(beta, float):
            beta = torch.tensor(beta).to(self.device)

        pexact = (
            plaq_exact(beta).to(self.device).to(self._dtype)
            if self.config.dynamics.group == 'U1'
            else None
        )
        for step in range(nsteps):
            x, metrics = self.hmc_step((x, beta))
            plaqs = metrics.get('plaqs', None)
            assert x is not None and isinstance(x, Tensor)
            if plaqs is not None and pexact is not None:
                pdiff = (plaqs - pexact).abs().sum()
                if pdiff < tol:
                    log.warning(
                        f'Chains thermalized!'
                        f' step: {step},'
                        f' plaq_diff: {pdiff:.4f}'
                    )
                    return x
                if nsteps > 100 and step % 10 == 0 and self._is_orchestrator:
                    log.info(
                        f'(warm-up) step: {step}, plaqs: {plaqs.mean():.4f}'
                    )

        self.dynamics.train()
        return x

    def train(
            self,
            x: Optional[Tensor] = None,
            skip: Optional[str | Sequence[str]] = None,
            train_dir: Optional[os.PathLike] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            nera: Optional[int] = None,
            nepoch: Optional[int] = None,
            nprint: Optional[int] = None,
            nlog: Optional[int] = None,
            beta: Optional[float | Sequence[float] | dict[str, float]] = None,
            warmup: bool = True,
            make_plots: bool = True
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
        beta_final: float = setup['beta_final']
        assert x is not None and isinstance(x, Tensor)
        assert nera is not None
        assert train_dir is not None
        plots = plotter.init_plots() if is_interactive() and make_plots else None
        # self.info(f'[TRAINING] x.dtype: {x.dtype}')
        # self.info(f'[TRAINING] self._dtype: {self._dtype}')
        for era in range(nera):
            b = torch.tensor(betas.get(str(era), beta_final))
            if era == (nera - 1) and self.steps.extend_last_era is not None:
                extend = int(self.steps.extend_last_era)

            if self._is_orchestrator:
                if era > 1 and str(era - 1) in self.summaries['train']:
                    esummary = self.histories['train'].era_summary(f'{era-1}')
                    log.info(f'Avgs over last era:\n {esummary}\n')
                box_header(f'ERA: {era} / {nera}, BETA: {b:.3f}')

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
                warmup=warmup,
                plots=plots,
            )

            self.rows['train'][str(era)] = edata['rows']
            self.tables['train'][str(era)] = edata['table']
            self.summaries['train'][str(era)] = edata['summaries']
            losses = torch.Tensor(list(edata['losses'][1:]))
            if self.config.annealing_schedule.dynamic:
                dy_avg = (losses[1:] - losses[:-1]).mean().item()
                if dy_avg > 0:
                    b -= (b / 10.)  # self.config.annealing_schedule._dbeta
                else:
                    b += (b / 10.)  # self.config.annealing_schedule._dbeta

            if self._is_orchestrator and self.config.save:
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
            skip: Optional[str | Sequence[str]] = None,
            train_dir: Optional[os.PathLike] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            nera: Optional[int] = None,
            nepoch: Optional[int] = None,
            beta: Optional[float | Sequence[float] | dict[str, float]] = None,
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

            if self._is_orchestrator:
                if era > 1 and str(era - 1) in self.summaries['train']:
                    esummary = self.histories['train'].era_summary(f'{era-1}')
                    log.info(f'Avgs over last era:\n {esummary}\n')

                box_header(f'ERA: {era} / {nera}, BETA: {b:.3f}')

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

            if (era + 1) == nera or (era + 1) % 5 == 0 and self.config.save:
                self.save_ckpt(era, epoch, run=run)

            if self._is_orchestrator:
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

        return metric.to(torch.float32).detach().cpu().numpy()

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
            name = f'{prefix}/{key}' if prefix is not None else f'{key}'
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

    def print_weights(self, grab: bool = True):
        _ = print_dict(dict(self.dynamics.named_parameters()), grab=grab)

    def print_grads(self, grab: bool = True):
        _ = print_dict({
            k: v.grad for k, v in self.dynamics.named_parameters()
        }, grab=grab)

    def print_grads_and_weights(self, grab: bool = True):
        log.info(80 * '-')
        log.info('GRADS:')
        self.print_grads(grab=grab)
        log.info(80 * '-')
        log.info('WEIGHTS:')
        self.print_weights(grab=grab)
        log.info(80 * '-')
