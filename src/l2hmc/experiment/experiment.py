"""
experiment.py

Contains implementation of Experiment object, defined by a static config.
"""
import logging
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import os
import wandb

from abc import ABC, abstractmethod

from pathlib import Path
from typing import Optional, Any

from l2hmc.configs import InputSpec, HERE, OUTDIRS_FILE, ExperimentConfig
from l2hmc.common import get_timestamp, is_interactive


log = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """Convenience class for running framework independent experiments."""
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.config = instantiate(cfg)
        assert isinstance(self.config, ExperimentConfig)
        assert self.config.framework in ['pytorch', 'tensorflow']
        self.lattice = self.build_lattice()
        self._is_built = False
        self.run = None
        self.loss_fn = None
        self.trainer = None
        self.dynamics = None
        self.optimizer = None
        super().__init__()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def build_lattice(self):
        pass

    @abstractmethod
    def build_dynamics(self):
        pass

    @abstractmethod
    def build_loss(self):
        pass

    @abstractmethod
    def build_optimizer(self, dynamics: Optional[Any] = None):
        """Build framework-dependent optimizer. Adam by default."""
        pass

    @abstractmethod
    def build_trainer(
            self,
            dynamics,
            optimizer,
            loss_fn,
            accelerator: Optional[Any] = None
    ):
        pass

    @abstractmethod
    def get_summary_writer(
            self,
            job_type: str,
    ):
        pass

    @abstractmethod
    def update_wandb_config(self, run_id: Optional[str] = None) -> None:
        pass


    @abstractmethod
    def init_wandb(self):
        pass

    @abstractmethod
    def _build(self, init_wandb: bool = True):
        pass


    def build(self, init_wandb: bool = True):
        return self._build(init_wandb=init_wandb)

    def get_input_spec(self) -> InputSpec:
        assert self.lattice is not None
        xdim = self.config.dynamics.xdim
        xshape = self.config.dynamics.xshape
        if self.config.dynamics.group == 'U1':
            input_dims = {
                'xnet': {'x': [xdim, 2], 'v': [xdim, ]},
                'vnet': {'x': [xdim, 2], 'v': [xdim, ]},
            }
        elif self.config.dynamics.group == 'SU3':
            input_dims = {
                'xnet': {'x': [xdim, ], 'v': [xdim, ]},
                'vnet': {'x': [xdim, ], 'v': [xdim, ]},
            }
        else:
            raise ValueError('Unexpected value for `config.dynamics.group`')

        input_spec = InputSpec(xshape=tuple(xshape), **input_dims)
        return input_spec

    def _update_wandb_config(
            self,
            device: str,
            run_id: Optional[str] = None,
    ) -> None:
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

    def _init_wandb(self):
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

        return run

    def get_jobdir(self, job_type: str) -> Path:
        outdir = self.cfg.get('outdir', None)
        if outdir is None:
            outdir = Path(os.getcwd())
            if is_interactive():
                framework = self.cfg.get('framework', None)
                outdir = outdir.joinpath(
                    'outputs',
                    get_timestamp('%Y-%m-%d-%H%M%S'),
                    framework,
                )
        jobdir = outdir.joinpath(job_type)
        jobdir.mkdir(exist_ok=True, parents=True)
        assert jobdir is not None
        setattr(self, f'{job_type}_dir', jobdir)
        if hasattr(self, 'run') and getattr(self, 'run', None) is not None:
            assert self.run is not None and self.run is wandb.run
            self.run.config[f'{job_type}_dir'] = jobdir

        with open(OUTDIRS_FILE, 'a') as f:
            f.write(Path(jobdir).resolve().as_posix())

        return jobdir

    def _get_summary_dir(
            self,
            job_type: str,
    ) -> str:
        jobdir = self.get_jobdir(job_type=job_type)
        sdir = jobdir.joinpath('summaries')
        sdir.mkdir(exist_ok=True, parents=True)
        return sdir.as_posix()
