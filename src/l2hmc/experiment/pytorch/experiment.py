"""
pytorch/experiment.py

Implements ptExperiment, a pytorch-specific subclass of the
Experiment base class.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

import aim
from aim import Distribution
import horovod.torch as hvd
from omegaconf import DictConfig
import torch
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter
import wandb

from l2hmc.common import save_and_analyze_data
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

Tensor = torch.Tensor
SIZE = hvd.size()
RANK = hvd.rank()
LOCAL_RANK = hvd.local_rank()


class Experiment(BaseExperiment):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)
        assert isinstance(self.config, ExperimentConfig)
        # self.accelerator = self.build_accelerator()

    def visualize_model(self, x: Optional[Tensor] = None):
        from torchviz import make_dot  # type: ignore
        if x is None:
            state = self.trainer.dynamics.random_state(1.)
            x = state.x
            v = state.v
        else:
            v = torch.rand_like(x)

        s, t, q = self.trainer.dynamics._call_xnet(
            0, inputs=(x, v), first=True
        )
        xparams = dict(
            self.trainer.dynamics.networks['xnet'].named_parameters()
        )
        make_dot(s, params=xparams).render('scale-xnet-0', format='png')
        make_dot(t, params=xparams).render('transl-xnet-0', format='png')
        make_dot(q, params=xparams).render('transf-xnet-0', format='png')

    def build_lattice(self):
        group = str(self.config.dynamics.group).upper()
        lat_args = {
            'nchains': self.config.dynamics.nchains,
            'shape': list(self.config.dynamics.latvolume),
        }
        if group == 'U1':
            return LatticeU1(**lat_args)
        if group == 'SU3':
            c1 = self.config.c1 if self.config.c1 is not None else 0.0
            return LatticeSU3(c1=c1, **lat_args)
        raise ValueError(
            'Unexpected value for `dynamics.group`: '
            f'{self.config.dynamics.group}'
        )

    def update_wandb_config(
            self,
            run_id: Optional[str] = None,
    ) -> None:
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        # size = 'DDP' if torch.cuda.device_count() > 1 else 'local'
        self._update_wandb_config(device=device, run_id=run_id)

    def build_accelerator(self, **kwargs):
        assert self.config.framework == 'pytorch'
        from accelerate import Accelerator
        # return Accelerator(**asdict(self.config.accelerator))
        # return Accelerator(log_with=['all'])
        return Accelerator(**kwargs)

    def build_dynamics(self, potential_fn: Callable) -> Dynamics:
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

    def build_loss(
            self,
            lattice: LatticeU1 | LatticeSU3
    ) -> LatticeLoss:
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

        return run

    def get_summary_writer(
            self,
            job_type: str,
    ):
        # sdir = super()._get_summary_dir(job_type=job_type)
        sdir = os.getcwd()
        return SummaryWriter(sdir)

    def build(
            self,
            init_wandb: bool = True,
            init_aim: bool = True,
    ):
        return self._build(
            init_wandb=init_wandb,
            init_aim=init_aim,
        )

    def _build(
            self,
            init_wandb: bool = True,
            init_aim: bool = True,
    ):
        if self._is_built:
            # assert self.accelerator is not None
            assert self.lattice is not None
            assert self.trainer is not None
            assert self.dynamics is not None
            assert self.optimizer is not None
            assert self.loss_fn is not None
            return {
                'lattice': self.lattice,
                'loss_fn': self.loss_fn,
                'dynamics': self.dynamics,
                'optimizer': self.optimizer,
                'trainer': self.trainer,
                'run': getattr(self, 'run', None),
                'arun': getattr(self, 'arun', None),
            }

        run = None
        arun = None
        # self.rank = self.accelerator.local_process_index
        # if self.trainer.rank == 0 and init_wandb:
        # if self.accelerator.is_local_main_process and init_wandb:
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        if RANK == 0:
            if init_wandb:
                log.warning(f'Initialize WandB from {rank}:{local_rank}')
                run = self.init_wandb(dynamics=self.dynamics,
                                      loss_fn=self.loss_fn)
            if init_aim:
                log.warning(f'Initializing Aim from {rank}:{local_rank}')
                arun = self.init_aim()
                if arun is not None:
                    # assert arun is aim.Run
                    # assert arun is not None and arun is aim.Run
                    if torch.cuda.is_available():
                        arun['ngpus'] = SIZE
                    else:
                        arun['ncpus'] = SIZE

        self.run = run
        self.arun = arun
        self.lattice = self.build_lattice()
        self.loss_fn = self.build_loss(self.lattice)
        self.dynamics = self.build_dynamics(potential_fn=self.lattice.action)
        # if SIZE > 1:
        #     dynamics = DDP(dynamics, device_ids=[RANK], output_device=RANK)

        self.optimizer = self.build_optimizer(dynamics=self.dynamics)
        # dynamics, optimizer = self.accelerator.prepare(
        #     dynamics, optimizer
        # )
        # accelerator = self.build_accelerator()
        self.trainer = self.build_trainer(
            loss_fn=self.loss_fn,
            dynamics=self.dynamics,
            optimizer=self.optimizer,
        )
        if self.run is not None and run is wandb.run:
            self.run.watch(
                self.trainer.dynamics,
                log='all',
                log_graph=True,
                criterion=self.trainer.loss_fn,
                log_freq=self.config.steps.log,
            )
        # self.lattice = lattice
        # self.loss_fn = loss_fn
        # self.trainer = trainer
        # self.dynamics = dynamics
        # self.optimizer = optimizer
        # self.accelerator = accelerator
        self._is_built = True
        assert callable(self.loss_fn)
        assert isinstance(self.trainer, Trainer)
        assert isinstance(self.dynamics, Dynamics)
        assert isinstance(self.lattice, (LatticeU1, LatticeSU3))
        return {
            'lattice': self.lattice,
            'loss_fn': self.loss_fn,
            'dynamics': self.dynamics,
            'optimizer': self.optimizer,
            'trainer': self.trainer,
            'run': self.run,
            'arun': self.arun,
        }

    def _assert_is_built(self):
        # assert self.accelerator is not None
        assert self._is_built
        assert self.lattice is not None
        assert self.trainer is not None
        assert self.dynamics is not None
        assert self.optimizer is not None
        assert self.loss_fn is not None

    def train(
        self,
        nchains: Optional[int] = None,
    ):
        # nchains = 16 if nchains is None else nchains
        jobdir = self.get_jobdir(job_type='train')
        writer = None
        if RANK == 0:
            writer = self.get_summary_writer(job_type='train')

        output = self.trainer.train(run=self.run,
                                    arun=self.arun,
                                    writer=writer,
                                    train_dir=jobdir)

        # dataset = output['history'].get_dataset(therm_frac=0.0)
        # dataset = self.trainer.histories['train'].get_dataset(therm_frac=0.0)
        dset = output['history'].get_dataset()
        dQint = dset.data_vars.get('dQint', None)
        if dQint is not None:
            dQint = dQint.values
            if self.run is not None:
                import wandb
                assert self.run is wandb.run
                self.run.summary['dQint_train'] = dQint
                self.run.summary['dQint_train'] = dQint.mean()

            if self.arun is not None:
                from aim import Distribution
                assert isinstance(self.arun, aim.Run)
                dQdist = Distribution(dQint)
                self.arun.track(dQdist,
                                name='dQint',
                                context={'subset': 'train'})
                self.arun.track(dQint.mean(),
                                name='dQint.avg',
                                context={'subset': 'train'})

        nchains = int(
            min(self.cfg.dynamics.nchains,
                max(64, self.cfg.dynamics.nchains // 8))
        )
        if RANK == 0:
            timing = self.trainer.timers['train'].save_and_write(
                outdir=Path(os.getcwd()),
            )

            _ = save_and_analyze_data(dset,
                                      run=self.run,
                                      arun=self.arun,
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
            # run: Optional[Any] = None,
            # arun: Optional[Any] = None,
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
        # run = self.run if run is None else run
        # assert run is wandb.run
        # arun = self.arun if run is None else arun
        output = self.trainer.eval(
            run=self.run,
            arun=self.arun,
            writer=writer,
            nchains=nchains,
            job_type=job_type,
            eps=eps,
            nleapfrog=nleapfrog,
        )
        dataset = output['history'].get_dataset(therm_frac=therm_frac)
        dQint = dataset.data_vars.get('dQint', None)
        if dQint is not None:
            dQint = dQint.values
            drop = int(0.1 * len(dQint))
            dQint = dQint[drop:]
            if self.run is not None:
                assert self.run is wandb.run
                self.run.summary[f'dQint_{job_type}'] = dQint
                self.run.summary[f'dQint_{job_type}.mean'] = dQint.mean()

            if self.arun is not None:
                import aim
                assert isinstance(self.arun, aim.Run)
                self.arun.track(
                    dQint.mean(),
                    name=f'dQint_{job_type}.avg'
                )
                dQdist = Distribution(dQint)
                self.arun.track(dQdist,
                                name='dQint',
                                context={'subset': job_type})
                self.arun.track(dQint.mean(),
                                name='dQint.avg',
                                context={'subset': job_type})

        _ = self.trainer.timers[job_type].save_and_write(
            outdir=Path(os.getcwd()),
        )

        _ = save_and_analyze_data(
            dataset,
            run=self.run,
            arun=self.arun,
            outdir=jobdir,
            output=output,
            nchains=nchains,
            job_type=job_type,
            framework='pytorch',
        )

        return output
