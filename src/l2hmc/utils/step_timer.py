"""
step_timer.py

Contains implementation of StepTimer to automatically track timing data.
"""
from __future__ import absolute_import, annotations, division, print_function
import time

import numpy as np
import os
from pathlib import Path
import json
import pandas as pd


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

    def get_eval_rate(self, evals_per_step: int = None) -> dict:
        if evals_per_step is None:
            evals_per_step = self.evals_per_step

        elapsed = np.sum(self.data)
        num_evals = evals_per_step * len(self.data)
        eval_rate = num_evals / elapsed
        output = {
            'eval_rate': eval_rate,
            'elapsed': elapsed,
            'num_evals': num_evals,
            'num_steps': len(self.data),
            'evals_per_step': evals_per_step,
        }

        return output

    def write_eval_rate(
            self,
            outdir: os.PathLike,
            mode: str = 'w',
            evals_per_step: int = None,
    ) -> dict:
        eval_rate = self.get_eval_rate(evals_per_step)
        outfile = Path(outdir).joinpath('step_timer_output.json')
        with open(outfile, mode) as f:
            json.dump(eval_rate, f)

        return eval_rate

    def save_data(self, outfile: os.PathLike, mode: str = 'w') -> pd.DataFrame:
        df = pd.DataFrame(self.data)
        fpath = Path(outfile).resolve()
        fpath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(fpath.as_posix(), mode=mode)

        return df

    def save_and_write(
            self,
            outdir: os.PathLike,
            mode: str = 'w',
            evals_per_step: int = None,
    ) -> dict:
        outfile = Path(outdir).joinpath('step_timer.csv')
        df = self.save_data(outfile=outfile, mode=mode)
        data = self.write_eval_rate(outdir=outdir,
                                    evals_per_step=evals_per_step)
        data.update({'df': df})

        return data
