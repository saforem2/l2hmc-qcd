"""
trainer.py

Contains BaseTrainer (ABC) object for training L2HMC dynamics
"""
from __future__ import absolute_import, division, print_function, annotations
import time
from typing import Callable
import numpy as np
from src.l2hmc.configs import Steps
from src.l2hmc.dynamics.pytorch.dynamics import Dynamics, to_u1, random_angle
from src.l2hmc.loss.pytorch.loss import LatticeLoss
from src.l2hmc.utils.history import History, StateHistory

from abc import ABC


class BaseTrainer(ABC):
    def __init__(self):
        pass
