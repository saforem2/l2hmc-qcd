"""
tensorflow/experiment.py

Contains implementation of tensorflow-specific Experiment object,
a subclass of the base `l2hmc/Experiment` object.
"""
from __future__ import absolute_import, division, print_function, annotations

import logging
from omegaconf import DictConfig
import os

from typing import Optional, Callable
from l2hmc.dynamics.tensorflow.dynamics import Dynamics
from l2hmc.lattice.su3.tensorflow.lattice import LatticeSU3
from l2hmc.lattice.u1.tensorflow.lattice import LatticeU1

import tensorflow as tf
import horovod.tensorflow as hvd
from l2hmc.loss.tensorflow.loss import LatticeLoss
import aim

from l2hmc.network.tensorflow.network import NetworkFactory
from l2hmc.trainers.tensorflow.trainer import Trainer
from l2hmc.common import save_and_analyze_data

# import wandb


from l2hmc.experiment.experiment import BaseExperiment

log = logging.getLogger(__name__)

# GLOBAL_RANK = hvd.rank()
RANK = hvd.rank()
LOCAL_RANK = hvd.local_rank()


class Experiment(BaseExperiment):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)
        # if not isinstance(self.cfg, ExperimentConfig):
        #     self.cfg = hydra.utils.instantiate(cfg)
        #     assert isinstance(self.config, ExperimentConfig)

    def visualize_model(self) -> None:
        state = self.dynamics.random_state(1.)
        x = self.dynamics.flatten(state.x)
        v = self.dynamics.flatten(state.v)
        _ = self.dynamics._call_vnet(0, (x, v), training=True)
        _ = self.dynamics._call_xnet(0, (x, v), first=True, training=True)
        vnet = self.dynamics._get_vnet(0)
        xnet = self.dynamics._get_xnet(0, first=True)

        vdot = tf.keras.utils.model_to_dot(vnet,
                                           show_shapes=True,
                                           expand_nested=True,
                                           show_layer_activations=True)
        xdot = tf.keras.utils.model_to_dot(xnet,
                                           show_shapes=True,
                                           expand_nested=True,
                                           show_layer_activations=True)
        log.info('Saving model visualizations to: [xnet,vnet].png')
        xdot.write_png('xnet.png')
        vdot.write_png('vnet.png')

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
    ):
        device = (
            'gpu' if len(tf.config.list_physical_devices('GPU')) > 0
            else 'cpu'
        )
        # size = 'horovod' if hvd.size() > 1 else 'local'
        self._update_wandb_config(device=device, run_id=run_id)

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

    def init_aim(self):
        run = super()._init_aim()
        return run

    def get_summary_writer(
            self,
            job_type: str,
    ):
        # sdir = super()._get_summary_dir(job_type=job_type)
        sdir = os.getcwd()
        return tf.summary.create_file_writer(sdir)  # type:ignore

    def build(
            self,
            init_wandb: bool = True,
            init_aim: bool = True
    ):
        return self._build(
            init_wandb=init_wandb,
            init_aim=init_aim
        )

    def _build(
            self,
            init_wandb: bool = True,
            init_aim: bool = True,
    ):
        if self._is_built:
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

        self.lattice = self.build_lattice()
        self.loss_fn = self.build_loss(self.lattice)
        self.dynamics = self.build_dynamics(potential_fn=self.lattice.action)
        self.optimizer = self.build_optimizer(dynamics=self.dynamics)
        self.trainer = self.build_trainer(
            loss_fn=self.loss_fn,
            dynamics=self.dynamics,
            optimizer=self.optimizer,
        )

        run = None
        arun = None
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        if RANK == 0:
            if init_wandb:
                log.warning(f'Initializing WandB from {rank}:{local_rank}')
                run = self.init_wandb()
            if init_aim:
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
        assert callable(self.loss_fn)
        assert isinstance(self.trainer, Trainer)
        assert isinstance(self.dynamics, Dynamics)
        assert isinstance(self.lattice, (LatticeU1, LatticeSU3))
        self._is_built = True
        return {
            'lattice': self.lattice,
            'loss_fn': self.loss_fn,
            'dynamics': self.dynamics,
            'optimizer': self.optimizer,
            'trainer': self.trainer,
            'run': self.run,
            'arun': self.arun,
        }

    def train(
            self,
            nchains: Optional[int] = None,
    ):
        # nchains = int(
        #     min(self.cfg.dynamics.nchains,
        #         max(64, self.cfg.dynamics.nchains // 8))
        # )
        jobdir = self.get_jobdir(job_type='train')
        writer = None
        if RANK == 0:
            writer = self.get_summary_writer(job_type='train')

        output = self.trainer.train(
            run=self.run,
            arun=self.arun,
            writer=writer,
            train_dir=jobdir
        )

        dataset = output['history'].get_dataset(therm_frac=0.0)
        dQint = dataset.data_vars.get('dQint', None)
        if dQint is not None:
            dQint = dQint.values
            if self.run is not None:
                import wandb
                assert self.run is wandb.run
                self.run.summary['dQint_train'] = dQint
                self.run.summary['dQint_train'] = dQint.mean()

            if self.arun is not None:
                import aim
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
            _ = save_and_analyze_data(dataset,
                                      run=self.run,
                                      arun=self.arun,
                                      outdir=jobdir,
                                      output=output,
                                      nchains=nchains,
                                      job_type='train',
                                      framework='tensorflow')

        _ = self.trainer.timers['train'].save_and_write(
            outdir=jobdir,
            fname=f'step_timer-train-{RANK}:{LOCAL_RANK}'
        )

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
    ) -> dict:
        """Evaluate model."""
        # if self.trainer.rank != 0:
        if RANK != 0:
            return {}

        assert job_type in ['eval', 'hmc']
        jobdir = self.get_jobdir(job_type)
        writer = self.get_summary_writer(job_type)

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
                import wandb
                assert self.run is wandb.run
                self.run.summary[f'dQint_{job_type}'] = dQint
                self.run.summary[f'dQint_{job_type}.mean'] = dQint.mean()

            if self.arun is not None:
                from aim import Distribution
                assert isinstance(self.arun, aim.Run)
                dQdist = Distribution(dQint)
                self.arun.track(dQdist,
                                name=f'dQint_{job_type}',
                                context={'subset': job_type})
                self.arun.track(dQint.mean(),
                                name=f'dQint_{job_type}.avg',
                                context={'subset': job_type})

        _ = save_and_analyze_data(
            dataset,
            run=self.run,
            arun=self.arun,
            outdir=jobdir,
            output=output,
            nchains=nchains,
            job_type=job_type,
            framework='tensorflow',
        )
        _ = self.trainer.timers[job_type].save_and_write(
            outdir=jobdir,
            fname=f'step_timer-{job_type}-{RANK}:{LOCAL_RANK}'
        )

        if writer is not None:
            writer.close()

        return output
