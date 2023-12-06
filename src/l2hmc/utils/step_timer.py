"""
step_timer.py

Contains implementation of StepTimer to automatically track timing data.
"""
from __future__ import absolute_import, annotations, division, print_function
import time
from typing import Optional
import logging

import numpy as np
import os
from pathlib import Path
import json
import pandas as pd

import functools
import datetime

# from l2hmc import get_logger
from l2hmc.common import get_timestamp

# log = get_logger(__name__)
log = logging.getLogger(__name__)


def log_execution_and_time(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        NOW = get_timestamp()
        start = time.time()
        log.info(
            f"{NOW} - Start execution of: {function.__name__}"
        )
        result = function(*args, **kwargs)
        end = time.time()
        log.info(f"{function.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


class BaseTimer:
    def __init__(
                self,
                name: str = 'BaseTimer',
                desc: Optional[str] = None
    ):
        self.name = name
        self.desc = desc
        self.data: list = []
        self.iterations: int = 0
        self.started: float = time.time()
        self._created: float = time.time()

    def start(self) -> None:
        self.started = time.time()

    def stop(self) -> float:
        dt = time.time() - self.started
        self.data.append(dt)
        self.iterations += 1
        return dt


class TrainTimer:
    def __init__(self) -> None:
        self.step_timer = StepTimer
        self.epoch_timer = StepTimer


class StepTimer:
    def __init__(self, evals_per_step: int = 1) -> None:
        self.data = []
        self.t = time.time()
        self.iterations = 0
        self.evals_per_step = evals_per_step

    def start(self) -> None:
        self.t = time.time()

    def stop(self) -> float:
        dt = time.time() - self.t
        self.data.append(dt)
        self.iterations += 1
        return dt

    def get_eval_rate(self, evals_per_step: Optional[int] = None) -> dict:
        if evals_per_step is None:
            evals_per_step = self.evals_per_step

        elapsed = np.sum(self.data)
        num_evals = evals_per_step * len(self.data)
        eval_rate = num_evals / elapsed
        return {
            'eval_rate': eval_rate,
            'elapsed': elapsed,
            'num_evals': num_evals,
            'num_steps': len(self.data),
            'evals_per_step': evals_per_step,
        }

    def write_eval_rate(
            self,
            outdir: os.PathLike,
            mode: str = 'a',
            evals_per_step: Optional[int] = None,
    ) -> dict:
        eval_rate = self.get_eval_rate(evals_per_step)
        outfile = Path(outdir).joinpath('step_timer_output.json')
        with open(outfile, mode) as f:
            json.dump(eval_rate, f)

        return eval_rate

    def save_data(self, outfile: os.PathLike, mode: str = 'a') -> pd.DataFrame:
        df = pd.DataFrame(self.data)
        fpath = Path(outfile).resolve()
        fpath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(fpath.as_posix(), mode=mode)

        return df

    def save_and_write(
            self,
            outdir: os.PathLike,
            mode: str = 'a',
            fname: Optional[str] = None,
            evals_per_step: Optional[int] = None,
    ) -> dict:
        fname = 'step_timer' if fname is None else fname
        outfile = Path(outdir).joinpath(f'{fname}.csv')
        df = self.save_data(outfile=outfile, mode=mode)
        data = self.write_eval_rate(outdir=outdir,
                                    evals_per_step=evals_per_step)
        data.update({'df': df})

        return data
