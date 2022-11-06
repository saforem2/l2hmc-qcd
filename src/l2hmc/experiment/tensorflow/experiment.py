"""
tensorflow/experiment.py

Contains implementation of tensorflow-specific Experiment object,
a subclass of the base `l2hmc/Experiment` object.
"""
from __future__ import absolute_import, division, print_function, annotations
import os

import logging
from omegaconf import DictConfig

from typing import Any, Optional
from pathlib import Path
from l2hmc.dynamics.tensorflow.dynamics import Dynamics
from l2hmc.lattice.su3.tensorflow.lattice import LatticeSU3
from l2hmc.lattice.u1.tensorflow.lattice import LatticeU1

import tensorflow as tf
import horovod.tensorflow as hvd

import l2hmc.configs as configs
from l2hmc.trainers.tensorflow.trainer import Trainer
from l2hmc.configs import ExperimentConfig


from l2hmc.experiment.experiment import BaseExperiment

log = logging.getLogger(__name__)

# GLOBAL_RANK = hvd.rank()
RANK = hvd.rank()
LOCAL_RANK = hvd.local_rank()


class Experiment(BaseExperiment):
    def __init__(
            self,
            cfg: DictConfig,
            build_networks: bool = True,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
    ) -> None:
        super().__init__(cfg=cfg)
        assert isinstance(
            self.config,
            (ExperimentConfig,
             configs.ExperimentConfig)
        )
        self._rank = hvd.rank()
        self._local_rank = hvd.local_rank()
        self.ckpt_dir = self.config.get_checkpoint_dir()
        self.trainer: Trainer = self.build_trainer(
            keep=keep,
            skip=skip,
            build_networks=build_networks,
            ckpt_dir=self.ckpt_dir,
        )

        run = None
        arun = None
        if self._rank == 0 and self.config.init_wandb:
            # import wandb
            log.warning(
                f'Initialize WandB from {self._rank}:{self._local_rank}'
            )
            run = super()._init_wandb()
            run.config['SIZE'] = hvd.size()

        if self._rank == 0 and self.config.init_aim:
            log.warning(
                f'Initializing Aim from {self._rank}:{self._local_rank}'
            )
            arun = self.init_aim()
            arun['SIZE'] = hvd.size()

        self.run = run
        self.arun = arun
        self._is_built = True
        assert callable(self.trainer.loss_fn)
        assert isinstance(self.trainer, Trainer)
        assert isinstance(self.trainer.dynamics, Dynamics)
        assert isinstance(self.trainer.lattice, (LatticeU1, LatticeSU3))
        # if not isinstance(self.cfg, ExperimentConfig):
        #     self.cfg = hydra.utils.instantiate(cfg)
        #     assert isinstance(self.config, ExperimentConfig)

    def visualize_model(self) -> None:
        assert self.trainer is not None and isinstance(self.trainer, Trainer)
        state = self.trainer.dynamics.random_state(1.)
        x = self.trainer.dynamics.flatten(state.x)
        v = self.trainer.dynamics.flatten(state.v)
        _ = self.trainer.dynamics._call_vnet(0, (x, v), training=True)
        _ = self.trainer.dynamics._call_xnet(
            0,
            (x, v),
            first=True,
            training=True
        )
        vnet = self.trainer.dynamics._get_vnet(0)
        xnet = self.trainer.dynamics._get_xnet(0, first=True)

        vdot = tf.keras.utils.model_to_dot(vnet,
                                           show_shapes=True,
                                           expand_nested=True,
                                           show_layer_activations=True)
        xdot = tf.keras.utils.model_to_dot(xnet,
                                           show_shapes=True,
                                           expand_nested=True,
                                           show_layer_activations=True)
        log.info('Saving model visualizations to: [xnet,vnet].png')
        outdir = Path(self._outdir).joinpath('network_diagrams')
        outdir.mkdir(exist_ok=True, parents=True)
        xdot.write_png(outdir.joinpath('xnet.png').as_posix())
        vdot.write_png(outdir.joinpath('vnet.png').as_posix())

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

    def build_trainer(
            self,
            build_networks: bool = True,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
            ckpt_dir: Optional[os.PathLike] = None,
    ) -> Trainer:
        return Trainer(
            self.cfg,
            skip=skip,
            keep=keep,
            ckpt_dir=ckpt_dir,
            build_networks=build_networks,
        )

    def init_wandb(self):
        return super()._init_wandb()

    def init_aim(self):
        run = super()._init_aim()
        return run

    def get_summary_writer(self, outdir: Optional[os.PathLike] = None):
        outdir = self._outdir if outdir is None else outdir
        return tf.summary.create_file_writer(  # type:ignore
            Path(outdir).as_posix()
        )

    def build(
            self,
            init_wandb: Optional[bool] = None,
            init_aim: Optional[bool] = None
    ):
        return self._build(
            init_wandb=init_wandb,
            init_aim=init_aim
        )

    def _build(
            self,
            init_wandb: Optional[bool] = None,
            init_aim: Optional[bool] = None,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
    ):
        if self._is_built:
            assert self.trainer is not None
            return {
                'trainer': self.trainer,
                'run': getattr(self, 'run', None),
                'arun': getattr(self, 'arun', None),
            }

        self.trainer = self.build_trainer(
            keep=keep,
            skip=skip,
        )

        run = None
        arun = None
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        if RANK == 0 and init_wandb:
            log.warning(f'Initializing WandB from {rank}:{local_rank}')
            run = self.init_wandb()
        if RANK == 0 and init_aim:
            log.warning(f'Initializing Aim from {rank}:{local_rank}')
            arun = self.init_aim()
            # assert arun is not None and arun is aim.Run
            ndevices = hvd.size()
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                arun['ngpus'] = ndevices
            else:
                arun['ncpus'] = ndevices

        self.run = run
        self.arun = arun
        assert isinstance(self.trainer, Trainer)
        assert callable(self.trainer.loss_fn)
        assert isinstance(self.trainer, Trainer)
        assert isinstance(self.trainer.dynamics, Dynamics)
        assert isinstance(self.trainer.lattice, (LatticeU1, LatticeSU3))
        self._is_built = True
        return {
            'lattice': self.trainer.lattice,
            'loss_fn': self.trainer.loss_fn,
            'dynamics': self.trainer.dynamics,
            'optimizer': self.trainer.optimizer,
            'trainer': self.trainer,
            'run': self.run,
            'arun': self.arun,
        }

    def train(
            self,
            nchains: Optional[int] = None,
            x: Optional[tf.Tensor] = None,
            skip: Optional[str | list[str]] = None,
            writer: Optional[Any] = None,
            nera: Optional[int] = None,
            nepoch: Optional[int] = None,
            beta: Optional[float | list[float] | dict[str, float]] = None,
    ):
        jobdir = self.get_jobdir(job_type='train')
        writer = None
        if RANK == 0:
            writer = self.get_summary_writer()
            writer.set_as_default()  # type:ignore

        if self.config.annealing_schedule.dynamic:
            output = self.trainer.train_dynamic(
                x=x,
                nera=nera,
                nepoch=nepoch,
                run=self.run,
                arun=self.arun,
                writer=writer,
                train_dir=jobdir,
                skip=skip,
                beta=beta,
            )
        else:
            output = self.trainer.train(
                x=x,
                nera=nera,
                nepoch=nepoch,
                run=self.run,
                arun=self.arun,
                writer=writer,
                train_dir=jobdir,
                skip=skip,
                beta=beta,
            )
        if self.trainer._is_chief:
            output['dataset'] = self.save_dataset(
                # output=output,
                nchains=nchains,
                job_type='train',
                outdir=jobdir,
                tables=output.get('tables', None)
            )

        if writer is not None:
            writer.close()

        return output

    def evaluate(
            self,
            job_type: str,
            therm_frac: float = 0.1,
            beta: Optional[float] = None,
            nchains: Optional[int] = None,
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
            eval_steps: Optional[int] = None,
            nprint: Optional[int] = None,
    ) -> dict:
        """Evaluate model."""
        assert job_type in ['eval', 'hmc']
        if not self.trainer._is_chief:
            return {}

        jobdir = self.get_jobdir(job_type=job_type)
        writer = self.get_summary_writer()
        writer.set_as_default()  # type:ignore

        output = self.trainer.eval(
            run=self.run,
            arun=self.arun,
            writer=writer,
            nchains=nchains,
            beta=beta,
            job_type=job_type,
            eps=eps,
            nleapfrog=nleapfrog,
            eval_steps=eval_steps,
            nprint=nprint,
        )
        output['dataset'] = self.save_dataset(
            # output=output,
            job_type=job_type,
            outdir=jobdir,
            therm_frac=therm_frac,
            tables=output.get('tables', None),
        )

        if writer is not None:
            writer.close()

        return output
