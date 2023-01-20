"""
experiment.py

Contains implementation of Experiment object, defined by a static config.
"""
from __future__ import absolute_import, annotations, division, print_function
from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
from typing import Optional

import aim
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import wandb
import xarray as xr
import socket

from l2hmc.common import get_timestamp, is_interactive, save_and_analyze_data
from l2hmc.configs import AIM_DIR, ExperimentConfig, HERE, OUTDIRS_FILE
from l2hmc.trainers.trainer import BaseTrainer
# from l2hmc.utils.logger import get_pylogger
from l2hmc.utils.step_timer import StepTimer
# import l2hmc.utils.plot_helpers as hplt

# log = get_pylogger(__name__)
log = logging.getLogger(__name__)


# def get_logger(rank: int) -> logging.Logger:
#     return (
#         logging.getLogger(__name__)
#         if rank == 0
#         else logging.getLogger(None)
#     )


class BaseExperiment(ABC):
    """Convenience class for running framework independent experiments."""

    def __init__(
            self,
            cfg: DictConfig,
    ) -> None:
        super().__init__()
        self._created = get_timestamp('%Y-%m-%d-%H%M%S')
        self.cfg = cfg
        self.config: ExperimentConfig = instantiate(cfg)
        # import l2hmc.configs as configs
        # assert isinstance(self.config, (
        #     ExperimentConfig,
        #     configs.ExperimentConfig,
        # ))
        assert self.config.framework.lower() in [
            'pt', 'tf', 'pytorch', 'torch', 'tensorflow'
        ]
        self._is_built = False
        self.run = None
        self.arun = None
        self.trainer: BaseTrainer
        self._outdir, self._jobdirs = self.get_outdirs()
        # self.lattice = self.build_lattice()
        # self._build_networks = build_networks
        # self.keep = keep
        # self.skip = skip

    @abstractmethod
    def visualize_model(self, x: Optional[Any] = None) -> None:
        pass

    @abstractmethod
    def train(self) -> dict:
        pass

    @abstractmethod
    def evaluate(self, job_type: str) -> dict:
        pass

    @abstractmethod
    def build_trainer(
            self,
            dynamics,
            loss_fn,
    ):
        pass

    @abstractmethod
    def get_summary_writer(self):
        pass

    @abstractmethod
    def update_wandb_config(self, run_id: Optional[str] = None) -> None:
        """Must Be overridden to specify uniquie run_id for W&B run"""
        pass

    @abstractmethod
    def init_wandb(self):
        pass

    def init_aim(self) -> aim.Run:
        return self._init_aim()

    @abstractmethod
    def _build(
            self,
            init_wandb: bool = True,
            init_aim: bool = True
    ) -> dict:
        pass

    def build(
            self,
            init_wandb: bool = True,
            init_aim: bool = True
    ) -> dict:
        return self._build(init_wandb=init_wandb,
                           init_aim=init_aim)

    def _init_aim(self) -> aim.Run:
        from aim import Run  # type:ignore
        # tstamp = get_timestamp()
        run = Run(
            repo=AIM_DIR.as_posix(),
            experiment='l2hmc-qcd',
            log_system_params=True,
        )
        # run['config'] = self.config
        run['config'] = OmegaConf.to_container(self.cfg,
                                               resolve=True,
                                               throw_on_missing=True)
        run['outdir'] = str(self._outdir.as_posix())
        run['hostname'] = str(os.environ.get('HOST', 'localhost'))
        # assert run is Run
        return run

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
                f'{self.config.backend}',
                f'nlf-{self.config.dynamics.nleapfrog}',
                f'beta_final-{self.config.annealing_schedule.beta_final}',
                f'{latstr}',
                f'{self.config.dynamics.group}',
            ]
        })

    def _init_wandb(
            self,
    ):
        if self.run is not None and self.run is wandb.run:
            raise ValueError('WandB already initialized!')

        from wandb.util import generate_id

        run_id = generate_id()
        self.update_wandb_config(run_id=run_id)
        # log.warning(f'os.getcwd(): {os.getcwd()}')
        wandb.tensorboard.patch(root_logdir=os.getcwd())
        # nlf = self.config.dynamics.nleapfrog
        # vol = 'x'.join([str(i) for i in self.config.dynamics.latvolume])
        # be = self.config.backend
        # ABBREVIATIONS = {
        #     'ds': 'ds',
        #     'deepspeed': 'ds',
        #     'hvd': 'hvd',
        #     'horovod': 'hvd',
        #     'ddp': 'ddp',
        #     'pt': 'pt',
        #     'pytorch': 'pt',
        #     'torch': 'pt',
        #     'tf': 'tf',
        #     'tensorflow': 'tf',
        # }
        # be = ABBREVIATIONS.get(self.config.backend.lower())
        # fw = ABBREVIATIONS.get(self.config.framework)
        # wbname = f'{fw}-{be}-{vol}-nlf{nlf}-{run_id[:4]}'
        # wbname = f'lf{nlf}-{be}-{fw}-{vol}-{run_id[:4]}'
        run = wandb.init(
            dir=os.getcwd(),
            # name=wbname,
            **self.config.wandb.setup,
        )
        # if self.config.framework in ['pt', 'torch', 'pytorch']:
        #     assert run is not None and run is wandb.run
        #     if dynamics is not None:
        #         run.watch(
        #             dynamics,
        #             log='all',
        #             log_graph=True,
        #             criterion=criterion,
        #         )
        assert run is wandb.run and run is not None
        wandb.define_metric('dQint_eval', summary='mean')
        run.log_code(HERE.as_posix())
        cfg_dict = OmegaConf.to_container(self.cfg,
                                          resolve=True,
                                          throw_on_missing=True)
        run.config.update(cfg_dict)
        hostname = socket.gethostbyaddr(socket.gethostname())[0].lower()
        if 'thetagpu' in hostname:
            run.config['hostname'] = 'ThetaGPU'
        elif 'polaris' in hostname:
            run.config['hostname'] = 'Polaris'
        else:
            run.config['hostname'] = hostname
        # run.config['hvd_size'] = SIZE
        # print_config(DictConfig(self.config), resolve=True)

        return run

    def get_outdirs(self) -> tuple[Path, dict[str, Path]]:
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
        jobdirs = {
            'train': Path(outdir).joinpath('train'),
            'eval': Path(outdir).joinpath('eval'),
            'hmc': Path(outdir).joinpath('hmc')
        }
        for _, val in jobdirs.items():
            val.mkdir(exist_ok=True, parents=True)

        return outdir, jobdirs

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

    def save_timers(
            self,
            job_type: str,
            outdir: Optional[os.PathLike] = None,
    ) -> None:
        # outdir = self._outdir if outdir is None else outdir
        outdir = (
            self._jobdirs.get(job_type, None)
            if outdir is None else outdir
        )
        assert outdir is not None
        timerdir = Path(outdir).joinpath('timers')
        timerdir.mkdir(exist_ok=True, parents=True)
        timers = getattr(self.trainer, 'timers', None)
        if timers is not None:
            timer = timers.get(job_type, None)
            if timer is not None:
                global_rank = getattr(self.trainer, 'global_rank', 0)
                rank = getattr(self.trainer, 'rank', 0)
                assert isinstance(timer, StepTimer)
                fname = (
                    f'step-timer-{job_type}-{global_rank}-{rank}'
                )
                timer.save_and_write(outdir=timerdir, fname=fname)

    def save_summaries(
            self,
            summaries: list[str],
            job_type: str,
    ) -> None:
        # TODO: Deal with `self.trainer.summaries` being dict
        outdir = self.get_jobdir(job_type)
        outfile = outdir.joinpath('summaries.txt')
        with open(outfile.as_posix(), 'a') as f:
            f.write('\n'.join(summaries))

    def save_dataset(
            self,
            job_type: str,
            dset: Optional[xr.Dataset] = None,
            tables: Optional[dict] = None,
            nchains: Optional[int] = None,
            outdir: Optional[os.PathLike] = None,
            therm_frac: Optional[float] = None,
    ) -> xr.Dataset:
        # assert isinstance(self.trainer, BaseTrainer)
        # if output is None:
        #     summaries = self.trainer.summaries.get(job_type, None)
        #     history = self.trainer.histories.get(job_type, None)
        # else:
        #     summaries = output.get('summaries', None)
        #     history = output.get('history', None)
        summary = self.trainer.summaries.get(job_type, None)
        history = self.trainer.histories.get(job_type, None)
        summaries = []
        if summary is not None:
            summaries = [f'{k} {v}' for k, v in summary.items()]
            self.save_summaries(summaries, job_type=job_type)
        if history is None:
            raise ValueError(f'Unable to recover history for {job_type}')

        assert history is not None  # and isinstance(history, BaseHistory)
        # therm_frac = 0.1 if therm_frac is None else therm_frac
        dset = history.get_dataset(therm_frac=therm_frac)
        assert isinstance(dset, xr.Dataset)

        chains_to_plot = int(
            min(self.cfg.dynamics.nchains,
                max(64, self.cfg.dynamics.nchains // 8))
        )
        chains_to_plot = nchains if nchains is not None else chains_to_plot
        outdir = self._outdir if outdir is None else outdir
        assert outdir is not None
        self.save_timers(job_type=job_type, outdir=outdir)
        import l2hmc.configs as configs
        assert isinstance(
            self.config, (configs.ExperimentConfig, ExperimentConfig)
        )
        logfreq = self.config.steps.log
        _ = save_and_analyze_data(dset,
                                  run=self.run,
                                  arun=self.arun,
                                  logfreq=logfreq,
                                  outdir=outdir,
                                  tables=tables,
                                  summaries=summaries,
                                  nchains=nchains,
                                  job_type=job_type,
                                  framework=self.config.framework)
        log.info('Done saving and analyzing data.')
        log.info('Creating summaries for WandB, Aim')
        dQint = dset.data_vars.get('dQint', None)
        if dQint is not None:
            dQint = dQint.values
            dQint = np.where(
                np.isnan(dQint),
                np.zeros_like(dQint),
                dQint,
            )
            # dQint = dQint[~np.isnan()]
            if self.run is not None:
                import wandb
                assert self.run is wandb.run
                self.run.summary[f'dQint_{job_type}'] = dQint
                self.run.summary[f'dQint_{job_type}'] = dQint.mean()

            if self.arun is not None:
                from aim import Distribution
                assert isinstance(self.arun, aim.Run)
                dQdist = Distribution(dQint)
                self.arun.track(dQdist,
                                name='dQint',
                                context={'subset': job_type})
                self.arun.track(dQdist,
                                name='dQint',
                                context={'subset': job_type})
        return dset
