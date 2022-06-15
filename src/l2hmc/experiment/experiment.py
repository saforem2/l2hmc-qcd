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
from typing import Optional

from l2hmc.configs import (
    AIM_DIR,
    InputSpec,
    HERE,
    OUTDIRS_FILE,
    ExperimentConfig
)
from l2hmc.common import get_timestamp, is_interactive
# import l2hmc.utils.plot_helpers as hplt


log = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """Convenience class for running framework independent experiments."""
    def __init__(self, cfg: DictConfig) -> None:
        self._created = get_timestamp('%Y-%m-%d-%H%M%S')
        self.cfg = cfg
        self.config = instantiate(cfg)
        assert isinstance(self.config, ExperimentConfig)
        assert self.config.framework in ['pytorch', 'tensorflow']
        # self.lattice = self.build_lattice()
        self._is_built = False
        self.run = None
        self.lattice = None
        self.loss_fn = None
        self.trainer = None
        self.dynamics = None
        self.optimizer = None
        self._outdir = self.get_outdir()
        super().__init__()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def build_lattice(self):
        group = str(self.config.dynamics.group).upper()
        lat_args = {
            'nchains': self.config.dynamics.nchains,
            'shape': list(self.config.dynamics.latvolume),
        }
        if group == 'U1':
            if self.config.framework in ['tf', 'tensorflow']:
                from l2hmc.lattice.u1.tensorflow.lattice import LatticeU1
                return LatticeU1(**lat_args)
            elif self.config.framework in ['pt', 'torch', 'pytorch']:
                from l2hmc.lattice.u1.pytorch.lattice import LatticeU1
                return LatticeU1(**lat_args)
        if group == 'SU3':
            c1 = self.config.c1 if self.config.c1 is not None else 0.0
            if self.config.framework in ['tf', 'tensorflow']:
                from l2hmc.lattice.su3.tensorflow.lattice import LatticeSU3
                return LatticeSU3(c1=c1, **lat_args)
            elif self.config.framework in ['pt', 'torch', 'pytorch']:
                from l2hmc.lattice.su3.pytorch.lattice import LatticeSU3
                return LatticeSU3(c1=c1, **lat_args)
        raise ValueError(
            'Unexpected value for `dynamics.group`: '
            f'{self.config.dynamics.group}'
        )

    @abstractmethod
    def build_dynamics(self):
        pass

    @abstractmethod
    def build_loss(self):
        pass

    @abstractmethod
    def build_optimizer(self):
        """Build framework-dependent optimizer. Adam by default."""
        # assert self.dynamics is not None
        pass

    @abstractmethod
    def build_trainer(
            self,
            dynamics,
            loss_fn,
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

    def init_aim(self):
        return self._init_aim()

    @abstractmethod
    def _build(
            self,
            init_wandb: bool = True,
            init_aim: bool = True
    ):
        pass

    def build(
            self,
            init_wandb: bool = True,
            init_aim: bool = True
    ):
        return self._build(init_wandb=init_wandb,
                           init_aim=init_aim)

    def get_input_spec(self) -> InputSpec:
        # assert self.lattice is not None
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

    def _init_aim(self):
        try:
            from aim import Run  # type:ignore
            # tstamp = get_timestamp()
            run = Run(repo=AIM_DIR.as_posix())
            run['hparams'] = OmegaConf.to_container(self.cfg,
                                                    resolve=True,
                                                    throw_on_missing=True)
            # run['train_dir'] = self.get_jobdir('train')
            run['outdir'] = self._outdir.as_posix()
            return run

        except (ImportError, ModuleNotFoundError) as e:
            log.warning(e)
            return None

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

    def get_outdir(self) -> Path:
        outdir = self.cfg.get('outdir', None)
        if outdir is None:
            outdir = Path(os.getcwd())
            if is_interactive():
                framework = self.cfg.get('framework', None)
                outdir = outdir.joinpath(
                    'outputs',
                    self._created,
                    framework,
                )

        return outdir

    def get_jobdir(self, job_type: str) -> Path:
        jobdir = self._outdir.joinpath(job_type)
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
