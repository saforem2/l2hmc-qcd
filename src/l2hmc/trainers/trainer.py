"""
trainer.py

Contains BaseTrainer (ABC) object for training L2HMC dynamics
"""
from __future__ import absolute_import, annotations, division, print_function
from abc import ABC, abstractmethod
import logging
from typing import Any, Optional, Sequence

import aim
from hydra.utils import instantiate
import numpy as np
from omegaconf.dictconfig import DictConfig
from rich.console import Console
from rich.table import Table

from l2hmc.common import get_timestamp
from l2hmc.configs import ExperimentConfig, InputSpec
# import l2hmc.configs as configs
from l2hmc.utils.history import BaseHistory
from l2hmc.utils.rich import add_columns, get_console
from l2hmc.utils.step_timer import StepTimer


log = logging.getLogger(__name__)
# from l2hmc import get_logger
# log = get_logger(__name__)


class BaseTrainer(ABC):
    def __init__(
            self,
            cfg: DictConfig | ExperimentConfig,
            keep: Optional[str | Sequence[str]] = None,
            skip: Optional[str | Sequence[str]] = None,
    ):
        self._created = get_timestamp()
        if isinstance(cfg, DictConfig):
            self.config: ExperimentConfig = instantiate(cfg)
        else:
            self.config: ExperimentConfig = cfg

        # self._is_chief: bool = self.check_if_chief()

        # assert isinstance(self.config,
        #                   (configs.ExperimentConfig,
        #                    ExperimentConfig))
        assert self.config.framework in [
            'pt',
            'tf',
            'torch',
            'pytorch',
            'tensorflow',
        ]
        self._is_built = False
        self.loss_fn = None
        self.lattice = None
        self.dynamics = None
        self.schedule = None
        self.optimizer = None
        self.lr_schedule = None
        self.steps = self.config.steps
        self.xshape = self.config.dynamics.xshape
        self.keep = [keep] if isinstance(keep, str) else keep
        self.skip = [skip] if isinstance(skip, str) else skip
        self.rows = {
            'train': {},
            'eval': {},
            'hmc': {},
        }
        self.tables = {
            'train': {},
            'eval': {},
            'hmc': {},
        }
        self.summaries = {
            'train': {},
            'eval': {},
            'hmc': {},
        }
        self.histories = {
            'train': BaseHistory(),
            'eval': BaseHistory(),
            'hmc': BaseHistory()
        }
        self._nlf = self.config.dynamics.nleapfrog
        if self.config.dynamics.merge_directions:
            self._nlf *= 2

        self.timers = {
            'train': StepTimer(evals_per_step=self._nlf),
            'eval': StepTimer(evals_per_step=self._nlf),
            'hmc': StepTimer(evals_per_step=self._nlf),
        }

        # self.steps = self.config.steps
        # self.lattice = self.build_lattice()
        # self.loss_fn = self.build_loss_fn()
        # self.dynamics = self.build_dynamics()
        # self.optimizer = self.build_optimizer()
        # self.lr_schedule = self.build_lr_schedule()
        # self.schedule = self.build_annealing_schedule()
        # self.xshape = self.config.dynamics.xshape
        self.keep = [keep] if isinstance(keep, str) else keep
        self.skip = [skip] if isinstance(skip, str) else skip
        self.histories = {
            'train': BaseHistory(),
            'eval': BaseHistory(),
            'hmc': BaseHistory()
        }
        self._nlf = self.config.dynamics.nleapfrog
        if self.config.dynamics.merge_directions:
            self._nlf *= 2

        self.timers = {
            'train': StepTimer(evals_per_step=self._nlf),
            'eval': StepTimer(evals_per_step=self._nlf),
            'hmc': StepTimer(evals_per_step=self._nlf),
        }

    @abstractmethod
    def warning(self, s: str) -> None:
        pass

    @abstractmethod
    def info(self, s: str) -> None:
        pass

    @abstractmethod
    def draw_x(self):
        pass

    @abstractmethod
    def reset_optimizer(self):
        pass

    @abstractmethod
    def build_lattice(self):
        pass

    @abstractmethod
    def build_loss_fn(self):
        pass

    @abstractmethod
    def build_dynamics(self, build_networks: bool = True):
        pass

    @abstractmethod
    def build_optimizer(self):
        pass

    # @abstractmethod
    # def build_lr_schedule(self):
    #     pass

    @abstractmethod
    def save_ckpt(self) -> None:
        pass

    @abstractmethod
    def should_log(self, epoch):
        pass

    @abstractmethod
    def should_print(self, epoch):
        pass

    @abstractmethod
    def record_metrics(
            self,
            metrics: dict,
            job_type: str,
            step: Optional[int] = None,
            record: Optional[dict] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            model: Optional[Any] = None,
            optimizer: Optional[Any] = None,
    ):
        pass

    @abstractmethod
    def hmc_step(
            self,
            inputs: tuple[Any, float],
            eps: float,
            nleapfrog: Optional[int] = None,
    ):
        pass

    @abstractmethod
    def eval_step(
            self,
            inputs: tuple[Any, float],
    ):
        pass

    @abstractmethod
    def eval(
            self,
            beta: Optional[float] = None,
            x: Optional[Any] = None,
            skip: Optional[str | list[str]] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            job_type: Optional[str] = 'eval',
            nchains: Optional[int] = None,
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> dict:
        pass

    @abstractmethod
    def train_step(
            self,
            inputs: tuple[Any, float],
    ):
        pass

    @abstractmethod
    def train_epoch(
            self,
            inputs: tuple[Any, float],
    ):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def metric_to_numpy(self, metric: Any):
        pass

    def update_table(
            self,
            table: Table,
            step: int,
            avgs: dict,
    ) -> Table:
        if step == 0:
            table = add_columns(avgs, table)
        else:
            table.add_row(
                *[f'{v:5}' for _, v in avgs.items()]
            )
        return table

    def metrics_to_numpy(
            self,
            metrics: dict[str, Any]
    ) -> dict[str, list[np.ndarray]]:
        m = {}
        for key, val in metrics.items():
            if val is None:
                m[key] = np.nan

            if isinstance(val, dict):
                for k, v in val.items():
                    m[f'{key}/{k}'] = self.metric_to_numpy(v)

            elif isinstance(val, (float, int, bool, np.floating)):
                m[key] = val

            else:
                try:
                    m[key] = self.metric_to_numpy(val)
                except ValueError as e:
                    log.exception(e)
                    log.error(
                        f'Error converting metrics[{key}] to numpy. Skipping!'
                    )
                    continue

        return m

    @abstractmethod
    def aim_track(
            self,
            metrics: dict,
            step: int,
            job_type: str,
            arun: aim.Run,
            prefix: Optional[str] = None,
    ) -> None:
        pass

    def get_input_spec(self) -> InputSpec:
        xshape = self.config.dynamics.xshape
        if self.config.dynamics.group == 'U1':
            xdim = self.config.dynamics.xdim
            input_dims = {
                'xnet': {'x': [xdim, 2], 'v': [xdim, ]},
                'vnet': {'x': [xdim, ], 'v': [xdim, ]},
            }
        elif self.config.dynamics.group == 'SU3':
            xdim = np.cumprod(xshape[1:-2])[-1] * 8
            input_dims = {
                'xnet': {'x': [xdim, ], 'v': [xdim, ]},
                'vnet': {'x': [xdim, ], 'v': [xdim, ]},
            }
        else:
            raise ValueError('Unexpected value for `config.dynamics.group`')

        return InputSpec(xshape=tuple(xshape), **input_dims)
