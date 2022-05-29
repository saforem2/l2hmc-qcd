"""
experiment.py

Contains implementation of Experiment object, defined by a static config.
"""
import logging
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import os
import wandb

from pathlib import Path
from typing import Optional, Any, Callable
from l2hmc.configs import InputSpec, HERE, OUTDIRS_FILE, ExperimentConfig


log = logging.getLogger(__name__)


class Experiment:
    """Convenience class for running framework independent experiments."""
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.config = instantiate(cfg)
        assert isinstance(self.config, ExperimentConfig)
        assert self.config.framework in ['pytorch', 'tensorflow']
        self.lattice = self.build_lattice()
        self._is_built = False

    def train(self):
        # TODO: Finish implementation??
        pass

    def build_lattice(self):
        framework = self.config.framework
        latvolume = self.config.dynamics.latvolume
        nchains = self.config.dynamics.nchains
        if self.config.dynamics.group == 'U1':
            if framework == 'pytorch':
                from l2hmc.lattice.u1.pytorch.lattice import LatticeU1
                return LatticeU1(nchains, tuple(latvolume))
            if framework == 'tensorflow':
                from l2hmc.lattice.u1.tensorflow.lattice import LatticeU1
                return LatticeU1(nchains, tuple(latvolume))

            raise ValueError(f'Unexpected value for `framework`: {framework}')

        if self.config.dynamics.group == 'SU3':
            if framework == 'pytorch':
                from l2hmc.lattice.su3.pytorch.lattice import LatticeSU3
                c1 = self.config.c1 if self.config.c1 is not None else 0.0
                return LatticeSU3(nchains, tuple(latvolume), c1=c1)
            if framework == 'tensorflow':
                from l2hmc.lattice.su3.tensorflow.lattice import LatticeSU3
                c1 = self.config.c1 if self.config.c1 is not None else 0.0
                return LatticeSU3(nchains, tuple(latvolume), c1=c1)

            raise ValueError(f'Unexpected value for `framework`: {framework}')

        raise ValueError(
            'Unexpected value for `dynamics.group`: '
            f'{self.config.dynamics.group}'
        )

    def get_input_spec(self) -> InputSpec:
        assert self.lattice is not None
        xdim = self.config.dynamics.xdim
        xshape = self.config.dynamics.xshape
        if self.config.dynamics.group == 'U1':
            input_dims = {
                'xnet': {'x': [xdim, 2], 'v': [xdim, ]},
                'vnet': {'x': [xdim, 2], 'v': [xdim, ]},
            }
        else:
            input_dims = {
                'xnet': {'x': [xdim, ], 'v': [xdim, ]},
                'vnet': {'x': [xdim, ], 'v': [xdim, ]},
            }

        input_spec = InputSpec(xshape=tuple(xshape), **input_dims)
        # vnet={'v': [xdim, ], 'x': [xdim, ]},
        # xnet={'v': [xdim, ], 'x': [xdim, 2]})
        return input_spec

    def update_wandb_config(
            self,
            run_id: Optional[str] = None,
    ):

        if self.config.framework == 'pytorch':
            import torch
            device = 'gpu' if torch.cuda.is_available() else 'cpu'
            # size = 'DDP' if torch.cuda.device_count() > 1 else 'local'

        elif self.config.framework == 'tensorflow':
            import tensorflow as tf
            # import horovod.tensorflow as hvd
            device = (
                'gpu' if len(tf.config.list_physical_devices('GPU')) > 0
                else 'cpu'
            )
            # size = 'horovod' if hvd.size() > 1 else 'local'
        else:
            raise ValueError('Unable to update `wandbConfig`')

        group = [self.config.framework, device]

        self.config.wandb.setup.update({'group': '/'.join(group)})
        if run_id is not None:
            self.config.wandb.setup.update({'id': run_id})

        latstr = 'x'.join([str(i) for i in self.config.dynamics.latvolume])
        self.config.wandb.setup.update({
            'tags': [
                f'{self.config.framework}',
                f'nlf-{self.config.dynamics.nleapfrog}',
                f'beta_final-{self.config.annealing_schedule.beta_final}',
                f'{latstr}',
                f'{self.config.dynamics.group}',
            ]
        })

    def build_accelerator(self):
        assert self.config.framework == 'pytorch'
        from accelerate.accelerator import Accelerator
        # return Accelerator(**asdict(self.config.accelerator))
        return Accelerator()

    def build_dynamics(self):
        assert self.lattice is not None
        input_spec = self.get_input_spec()
        if self.config.framework == 'pytorch':
            from l2hmc.dynamics.pytorch.dynamics import Dynamics
            from l2hmc.network.pytorch.network import NetworkFactory
            net_factory = NetworkFactory(input_spec=input_spec,
                                         conv_config=self.config.conv,
                                         network_config=self.config.network,
                                         net_weights=self.config.net_weights)
            return Dynamics(config=self.config.dynamics,
                            potential_fn=self.lattice.action,
                            network_factory=net_factory)
        elif self.config.framework == 'tensorflow':
            from l2hmc.dynamics.tensorflow.dynamics import Dynamics
            from l2hmc.network.tensorflow.network import NetworkFactory
            net_factory = NetworkFactory(input_spec=input_spec,
                                         conv_config=self.config.conv,
                                         network_config=self.config.network,
                                         net_weights=self.config.net_weights)
            return Dynamics(config=self.config.dynamics,
                            potential_fn=self.lattice.action,
                            network_factory=net_factory)
        raise ValueError('Unexpected value encountered in cfg.framework.')

    def build_loss(self):
        assert self.lattice is not None
        if self.config.framework == 'pytorch':
            from l2hmc.loss.pytorch.loss import LatticeLoss
        elif self.config.framework == 'tensorflow':
            from l2hmc.loss.tensorflow.loss import LatticeLoss
        else:
            raise ValueError(
                'Unexpected value encountered for `self.config.framework`'
            )

        return LatticeLoss(
            lattice=self.lattice,  # type:ignore
            loss_config=self.config.loss
        )

    def build_optimizer(self, dynamics: Optional[Any] = None):
        """Build framework-dependent optimizer. Adam by default."""
        # TODO: Expand method, re-build LR scheduler, etc
        # TODO: Replace `LearningRateConfig` with `OptimizerConfig`
        # TODO: Optionally, break up in to lrScheduler, OptimizerConfig ?
        lr = self.config.learning_rate.lr_init
        assert self.config.framework in ['torch', 'pytorch', 'tensorflow']
        assert dynamics is not None
        if self.config.framework in ['torch', 'pytorch']:
            from torch.optim import Adam
            return Adam(dynamics.parameters(), lr=lr)
        if self.config.framework == 'tensorflow':
            import tensorflow as tf
            return tf.keras.optimizers.Adam(lr)

        raise ValueError('Unable to build optimizer.')

    def build_trainer(
            self,
            dynamics,
            optimizer,
            loss_fn,
            accelerator: Optional[Any] = None
    ):
        if self.config.framework == 'pytorch':
            from l2hmc.trainers.pytorch.trainer import Trainer
            if accelerator is None:
                accelerator = self.build_accelerator()

            dynamics = dynamics.to(accelerator.device)
            optimizer = self.build_optimizer(dynamics=dynamics)
            dynamics, optimizer = accelerator.prepare(dynamics, optimizer)

            return Trainer(loss_fn=loss_fn,
                           dynamics=dynamics,
                           optimizer=optimizer,
                           accelerator=accelerator,
                           steps=self.config.steps,
                           schedule=self.config.annealing_schedule,
                           lr_config=self.config.learning_rate,
                           dynamics_config=self.config.dynamics,
                           aux_weight=self.config.loss.aux_weight)

        if self.config.framework == 'tensorflow':
            import horovod.tensorflow as hvd
            from l2hmc.trainers.tensorflow.trainer import Trainer

            return Trainer(loss_fn=loss_fn,
                           dynamics=dynamics,
                           optimizer=optimizer,
                           rank=hvd.rank(),
                           steps=self.config.steps,
                           lr_config=self.config.learning_rate,
                           compile=(not self.config.debug_mode),
                           dynamics_config=self.config.dynamics,
                           aux_weight=self.config.loss.aux_weight,
                           schedule=self.config.annealing_schedule)

        raise ValueError('Unable to build Trainer.')

    def init_wandb(
            self,
            dynamics: Optional[Any] = None,
            loss_fn: Optional[Callable] = None
    ):
        from wandb.util import generate_id
        from l2hmc.utils.rich import print_config

        run_id = generate_id()
        self.update_wandb_config(run_id=run_id)
        log.warning(f'os.getcwd(): {os.getcwd()}')
        # wandb.tensorboard.patch(root_logdir=os.getcwd())
        run = wandb.init(dir=os.getcwd(), **self.config.wandb.setup)
        assert run is wandb.run and run is not None
        wandb.define_metric('dQint_eval', summary='mean')
        run.log_code(HERE.as_posix())
        cfg_dict = OmegaConf.to_container(self.cfg,
                                          resolve=True,
                                          throw_on_missing=False)
        run.config.update(cfg_dict)
        print_config(DictConfig(self.config), resolve=True)
        if self.config.framework == 'pytorch':
            run.watch(
                dynamics,
                log='all',
                log_graph=True,
                criterion=loss_fn,
                log_freq=self.config.steps.log
            )

        return run

    def get_jobdir(self, job_type: str) -> Path:
        here = Path(self.config.get('outdir', os.getcwd()))
        jobdir = here.joinpath(job_type)
        jobdir.mkdir(exist_ok=True, parents=True)
        assert jobdir is not None
        with open(OUTDIRS_FILE, 'a') as f:
            f.write(Path(jobdir).resolve().as_posix())

        return jobdir

    def get_summary_writer(
            self,
            job_type: str,
    ):
        jobdir = self.get_jobdir(job_type=job_type)
        sdir = jobdir.joinpath('summaries')
        sdir.mkdir(exist_ok=True, parents=True)
        sdir = sdir.as_posix()
        if self.config.framework == 'tensorflow':
            import tensorflow as tf
            return tf.summary.create_file_writer(sdir)  # type:ignore
        if self.config.framework == 'pytorch':
            from torch.utils.tensorboard.writer import SummaryWriter
            return SummaryWriter(sdir)

        raise ValueError('Unable to get summary writer')

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

        loss_fn = self.build_loss()
        dynamics = self.build_dynamics()
        optimizer = self.build_optimizer(dynamics)
        assert self.config.framework in ['torch', 'pytorch', 'tensorflow']
        if self.config.framework in ['torch', 'pytorch']:
            accelerator = self.build_accelerator()
            IS_CHIEF = accelerator.is_local_main_process
            trainer = self.build_trainer(dynamics=dynamics,
                                         loss_fn=loss_fn,
                                         optimizer=optimizer,
                                         accelerator=accelerator)
        elif self.config.framework == 'tensorflow':
            trainer = self.build_trainer(dynamics=dynamics,
                                         loss_fn=loss_fn,
                                         optimizer=optimizer)
            rank = getattr(trainer, 'rank', 0)
            IS_CHIEF = (rank == 0)
        else:
            raise ValueError('Unable to build.')

        run = None
        if IS_CHIEF and init_wandb:
            run = self.init_wandb(dynamics=dynamics, loss_fn=loss_fn)

        self._is_built = True
        self.run = run
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
