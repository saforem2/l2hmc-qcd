"""
trainer.py

Contains BaseTrainer (ABC) object for training L2HMC dynamics
"""
from __future__ import absolute_import, division, print_function, annotations
# import time
# from typing import Callable
# import numpy as np
# from src.l2hmc.configs import Steps
# from src.l2hmc.dynamics.pytorch.dynamics import Dynamics, to_u1, random_angle
# from src.l2hmc.loss.pytorch.loss import LatticeLoss
# from src.l2hmc.utils.history import StateHistory

from abc import ABC, abstractmethod
from typing import Callable, Optional

from omegaconf.dictconfig import DictConfig
from l2hmc.configs import ExperimentConfig, Steps
from hydra.utils import instantiate

from l2hmc.utils.history import BaseHistory
from l2hmc.utils.step_timer import StepTimer

# steps
# dynamics
# optimizer
# schedule
# lr_config
# loss_fn
# aux_weight
# keep
# skip

class BaseTrainer(ABC):
    def __init__(
        self,
        cfg: DictConfig,
        keep: Optional[str | list[str]] = None,
        skip: Optional[str | list[str]] = None,
    ):
        self.cfg = cfg
        self.config = instantiate(cfg)
        assert isinstance(self.config, ExperimentConfig)
        self.steps = self.build_steps()
        self.loss_fn = self.build_loss_fn()
        self.dynamics = self.build_dynamics()
        self.optimizer = self.build_optimizer()
        self.lr_schedule = self.build_lr_schedule()
        self.schedule = self.build_annealing_schedule()
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
    def build_steps(self) -> Steps:
        pass

    @abstractmethod
    def build_loss_fn(self) -> Callable:
        pass

    @abstractmethod
    def build_dynamics(self):
        pass

    @abstractmethod
    def build_optimizer(self):
        pass

    @abstractmethod
    def build_lr_schedule(self):
        pass

    @abstractmethod
    def build_annealing_schedule(self):
        pass
