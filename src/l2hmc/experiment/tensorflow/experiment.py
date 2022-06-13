"""
tensorflow/experiment.py

Contains implementation of tensorflow-specific Experiment object,
a subclass of the base `l2hmc/Experiment` object.
"""
from __future__ import absolute_import, division, print_function, annotations

import logging
from omegaconf import DictConfig

from typing import Any, Optional, Callable
from l2hmc.configs import ExperimentConfig
from l2hmc.dynamics.tensorflow.dynamics import Dynamics
from l2hmc.lattice.su3.tensorflow.lattice import LatticeSU3
from l2hmc.lattice.u1.tensorflow.lattice import LatticeU1

import tensorflow as tf
import horovod.tensorflow as hvd
from l2hmc.loss.tensorflow.loss import LatticeLoss

from l2hmc.network.tensorflow.network import NetworkFactory
from l2hmc.trainers.tensorflow.trainer import Trainer
from l2hmc.common import save_and_analyze_data

import wandb


from l2hmc.experiment.experiment import BaseExperiment

log = logging.getLogger(__name__)

# GLOBAL_RANK = hvd.rank()
RANK = hvd.rank()
LOCAL_RANK = hvd.local_rank()


class Experiment(BaseExperiment):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)
        assert isinstance(self.config, ExperimentConfig)

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
        device = (
            'gpu' if len(tf.config.list_physical_devices('GPU')) > 0
            else 'cpu'
        )
        # size = 'horovod' if hvd.size() > 1 else 'local'
        self._update_wandb_config(device=device, run_id=run_id)

    def build_dynamics(self):
        assert self.lattice is not None
        input_spec = self.get_input_spec()
        net_factory = NetworkFactory(
            input_spec=input_spec,
            conv_config=self.config.conv,
            network_config=self.config.network,
            net_weights=self.config.net_weights,
        )
        return Dynamics(config=self.config.dynamics,
                        potential_fn=self.lattice.action,
                        network_factory=net_factory)

    def build_loss(self):
        assert (
            self.lattice is not None
            and isinstance(self.lattice, (LatticeU1, LatticeSU3))
        )
        return LatticeLoss(
            lattice=self.lattice,
            loss_config=self.config.loss,
        )

    def build_optimizer(
            self,
            dynamics: Optional[Dynamics] = None  # pyright: ignore type:ignore
    ) -> None:
        # TODO: Expand method, re-build LR scheduler, etc
        # TODO: Replace `LearningRateConfig` with `OptimizerConfig`
        # TODO: Optionally, break up in to lrScheduler, OptimizerConfig ?
        return tf.keras.optimizers.Adam(self.config.learning_rate.lr_init)

    def build_trainer(
            self,
            dynamics: Dynamics,
            optimizer: tf.keras.optimizers.Optimizer,
            loss_fn: Callable,
    ) -> Trainer:
        return Trainer(
            rank=RANK,
            loss_fn=loss_fn,
            dynamics=dynamics,
            optimizer=optimizer,
            steps=self.config.steps,
            lr_config=self.config.learning_rate,
            # compile=(not self.config.debug_mode),
            dynamics_config=self.config.dynamics,
            aux_weight=self.config.loss.aux_weight,
            schedule=self.config.annealing_schedule,
        )

    def init_wandb(self):
        return super()._init_wandb()

    def get_summary_writer(
            self,
            job_type: str,
    ):
        sdir = super()._get_summary_dir(job_type=job_type)
        return tf.summary.create_file_writer(sdir)  # type:ignore

    def build(self, init_wandb: bool = True):
        return self._build(init_wandb=init_wandb)

    def _build(self, init_wandb: bool = True):
        if self._is_built:
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
        self.lattice = self.build_lattice()
        loss_fn = self.build_loss()
        dynamics = self.build_dynamics()
        optimizer = self.build_optimizer(dynamics)
        trainer = self.build_trainer(
            dynamics=dynamics,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        self.run = None
        if RANK == 0 and init_wandb:
            if (
                    not hasattr(self, 'run')
                    or getattr(self, 'run', None) is None
            ):
                local_rank = hvd.local_rank()
                rank = hvd.rank()
                log.warning(f'Initializing WandB from {rank}:{local_rank}')
                self.run = self.init_wandb()

        self._is_built = True
        self.trainer = trainer
        self.dynamics = dynamics
        self.optimizer = optimizer
        self.loss_fn = loss_fn
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
        nchains = int(
            min(self.cfg.dynamics.nchains,
                max(64, self.cfg.dynamics.nchains // 8))
        )
        jobdir = self.get_jobdir(job_type='train')
        if run is not None:
            assert run.isinstance(wandb.run)

        writer = None
        # if self.trainer.rank == 0:
        if RANK == 0:
            writer = self.get_summary_writer(job_type='train')

        output = self.trainer.train(run=run, writer=writer, train_dir=jobdir)
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
    ) -> dict:
        """Evaluate model."""
        # if self.trainer.rank != 0:
        if RANK != 0:
            return {}

        assert job_type in ['eval', 'hmc']
        jobdir = self.get_jobdir(job_type)
        writer = self.get_summary_writer(job_type)

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
            framework='tensorflow',
        )

        return output
