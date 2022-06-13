"""
pytorch/experiment.py

Implements ptExperiment, a pytorch-specific subclass of the
Experiment base class.
"""
from __future__ import absolute_import, annotations, division, print_function
# import os
import logging
from typing import Any, Callable, Optional
# from accelerate.accelerator import Accelerator

from omegaconf import DictConfig
import torch
import horovod.torch as hvd
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter
# from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
# from build.lib.l2hmc.configs import ExperimentConfig
from l2hmc.common import save_and_analyze_data
# from mpi4py import MPI

from l2hmc.configs import ExperimentConfig

from l2hmc.dynamics.pytorch.dynamics import Dynamics
from l2hmc.experiment.experiment import BaseExperiment
from l2hmc.lattice.su3.pytorch.lattice import LatticeSU3
from l2hmc.lattice.u1.pytorch.lattice import LatticeU1
from l2hmc.loss.pytorch.loss import LatticeLoss
from l2hmc.network.pytorch.network import NetworkFactory
from l2hmc.trainers.pytorch.trainer import Trainer

log = logging.getLogger(__name__)

# LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')

SIZE = hvd.size()
RANK = hvd.rank()
LOCAL_RANK = hvd.local_rank()


class Experiment(BaseExperiment):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)
        assert isinstance(self.config, ExperimentConfig)
        # self.accelerator = self.build_accelerator()

    def build_lattice(self):
        group = str(self.config.dynamics.group).upper()
        lat_args = (
            self.config.dynamics.nchains,
            tuple(self.config.dynamics.latvolume)
        )
        if group == 'U1':
            return LatticeU1(*lat_args)
        if group == 'SU3':
            c1 = self.config.c1 if self.config.c1 is not None else 0.0
            return LatticeSU3(*lat_args, c1=c1)
        raise ValueError(
            'Unexpected value for `dynamics.group`: '
            f'{self.config.dynamics.group}'
        )

    def update_wandb_config(
            self,
            run_id: Optional[str] = None,
    ):
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        # size = 'DDP' if torch.cuda.device_count() > 1 else 'local'
        self._update_wandb_config(device=device, run_id=run_id)

    def build_accelerator(self, **kwargs):
        assert self.config.framework == 'pytorch'
        from accelerate import Accelerator
        # return Accelerator(**asdict(self.config.accelerator))
        # return Accelerator(log_with=['all'])
        return Accelerator(**kwargs)

    def build_dynamics(self, potential_fn: Callable):
        # assert self.lattice is not None
        input_spec = self.get_input_spec()
        net_factory = NetworkFactory(
            input_spec=input_spec,
            conv_config=self.config.conv,
            network_config=self.config.network,
            net_weights=self.config.net_weights,
        )
        return Dynamics(config=self.config.dynamics,
                        potential_fn=potential_fn,
                        network_factory=net_factory)

    def build_loss(self, lattice: LatticeU1 | LatticeSU3):
        # assert (
        #     self.lattice is not None
        #     and isinstance(self.lattice, (LatticeU1, LatticeSU3))
        # )
        return LatticeLoss(
            lattice=lattice,
            loss_config=self.config.loss,
        )

    def build_optimizer(
            self,
            dynamics: Dynamics
    ) -> torch.optim.Optimizer:
        # TODO: Expand method, re-build LR scheduler, etc
        # TODO: Replace `LearningRateConfig` with `OptimizerConfig`
        # TODO: Optionally, break up in to lrScheduler, OptimizerConfig ?
        lr = self.config.learning_rate.lr_init
        return torch.optim.Adam(dynamics.parameters(), lr=lr)

    def build_trainer(
            self,
            dynamics: Dynamics,
            optimizer: optim.Optimizer,
            loss_fn: Callable,
    ) -> Trainer:
        return Trainer(
            loss_fn=loss_fn,
            dynamics=dynamics,
            optimizer=optimizer,
            # accelerator=self.accelerator,
            steps=self.config.steps,
            schedule=self.config.annealing_schedule,
            lr_config=self.config.learning_rate,
            dynamics_config=self.config.dynamics,
            aux_weight=self.config.loss.aux_weight,
        )

    def init_wandb(
            self,
            dynamics: Optional[Any] = None,
            loss_fn: Optional[Callable] = None
    ):
        run = super()._init_wandb()
        if self.config.framework == 'pytorch':
            run.watch(
                dynamics,
                log='all',
                log_graph=True,
                criterion=loss_fn,
                log_freq=self.config.steps.log
            )

        return run

    def get_summary_writer(
            self,
            job_type: str,
    ):
        sdir = super()._get_summary_dir(job_type=job_type)
        return SummaryWriter(sdir)

    def build(self, init_wandb: bool = True):
        return self._build(init_wandb=init_wandb)

    def _build(self, init_wandb: bool = True):
        if self._is_built:
            # assert self.accelerator is not None
            assert self.lattice is not None
            assert self.trainer is not None
            assert self.dynamics is not None
            assert self.optimizer is not None
            assert self.loss_fn is not None
            return {
                'run': self.run,
                'trainer': self.trainer,
                'dynamics': self.dynamics,
                'optimizer': self.optimizer,
                'loss_fn': self.loss_fn,
            }

        lattice = self.build_lattice()
        loss_fn = self.build_loss(lattice)
        dynamics = self.build_dynamics(potential_fn=lattice.action)
        # if SIZE > 1:
        #     dynamics = DDP(dynamics, device_ids=[RANK], output_device=RANK)

        optimizer = self.build_optimizer(dynamics=dynamics)
        # dynamics, optimizer = self.accelerator.prepare(
        #     dynamics, optimizer
        # )
        # accelerator = self.build_accelerator()
        trainer = self.build_trainer(
            dynamics=dynamics,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
        self.lattice = lattice
        self.loss_fn = loss_fn
        self.trainer = trainer
        self.dynamics = dynamics
        self.optimizer = optimizer
        # self.accelerator = accelerator
        self.run = None
        # self.rank = self.accelerator.local_process_index
        # if self.trainer.rank == 0 and init_wandb:
        # if self.accelerator.is_local_main_process and init_wandb:
        if RANK == 0 and init_wandb:
            if self.run is None:
                self.run = self.init_wandb(dynamics=dynamics, loss_fn=loss_fn)

        self._is_built = True

        return {
            'run': self.run,
            'trainer': self.trainer,
            'dynamics': self.dynamics,
            'optimizer': self.optimizer,
            'loss_fn': self.loss_fn,
        }

    def train(
        self,
        run: Optional[Any] = None,
        nchains: Optional[int] = None,
    ):
        # nchains = 16 if nchains is None else nchains
        jobdir = self.get_jobdir(job_type='train')
        if run is not None:
            assert run.isinstance(wandb.run)

        writer = None
        if RANK == 0:
            writer = self.get_summary_writer(job_type='train')

        output = self.trainer.train(run=run,
                                    writer=writer,
                                    train_dir=jobdir)
        # if self.trainer.rank == 0:
        if RANK == 0:
            dset = output['history'].get_dataset()
            nchains = int(
                min(self.cfg.dynamics.nchains,
                    max(64, self.cfg.dynamics.nchains // 8))
            )
            _ = save_and_analyze_data(dset,
                                      run=run,
                                      outdir=jobdir,
                                      output=output,
                                      nchains=nchains,
                                      job_type='train',
                                      framework='pytorch')

        if writer is not None:
            writer.close()

        return output

    def evaluate(
            self,
            job_type: str,
            run: Optional[Any] = None,
            therm_frac: float = 0.1,
            nchains: Optional[int] = None,
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ):
        """Evaluate model."""
        if RANK != 0:
            return

        assert job_type in ['eval', 'hmc']
        jobdir = self.get_jobdir(job_type)
        writer = self.get_summary_writer(job_type)
        # if RANK == 0:
        #     writer = self.get_summary_writer(job_type)
        # else:
        #     writer = None
        run = self.run if run is None else run
        assert run is wandb.run
        output = self.trainer.eval(
            run=run,
            writer=writer,
            nchains=nchains,
            job_type=job_type,
            eps=eps,
            nleapfrog=nleapfrog,
        )
        dataset = output['history'].get_dataset(therm_frac=therm_frac)
        if run is not None:
            assert run is wandb.run
            dQint = dataset.data_vars.get('dQint').values
            drop = int(0.1 * len(dQint))
            dQint = dQint[drop:]
            run.summary[f'dQint_{job_type}'] = dQint
            run.summary[f'dQint_{job_type}.mean'] = dQint.mean()

        _ = save_and_analyze_data(
            dataset,
            run=run,
            outdir=jobdir,
            output=output,
            nchains=nchains,
            job_type=job_type,
            framework='pytorch',
        )

        return output
