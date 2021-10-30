"""
step_timer.py
"""
from __future__ import absolute_import, print_function, division, annotations

from pathlib import Path
import time
from utils.hvd_init import SIZE
import pandas as pd
import numpy as np
from typing import Union

from utils.logger import Logger

logger = Logger()

class StepTimer:
    def __init__(self, evals_per_step: int = 1):
        self.data = []
        self.t = time.time()
        self.evals_per_step = evals_per_step

    def start(self):
        self.t = time.time()

    def stop(self):
        dt = time.time() - self.t
        self.data.append(dt)

        return dt

    def reset(self):
        self.data = []

    def get_eval_rate(self, evals_per_step: int = None):
        if evals_per_step is None:
            evals_per_step = self.evals_per_step

        elapsed = np.sum(self.data)
        num_evals = SIZE * evals_per_step * len(self.data)
        eval_rate = num_evals / elapsed
        output = {
            'eval_rate': eval_rate,
            'total_time': elapsed,
            'num_evals': num_evals,
            'size': SIZE,
            'num_steps': len(self.data),
            'evals_per_step': evals_per_step,
        }

        return output

    def write_eval_rate(
            self,
            outdir: Union[str, Path],
            mode: str = 'w',
            **kwargs,
    ):
        eval_rate = self.get_eval_rate(**kwargs)
        outfile = Path(outdir).joinpath('eval_rate.txt')
        logger.debug(f'Writing eval rate to: {outfile}')
        with open(outfile, mode) as f:
            f.write('\n'.join([f'{k}: {v}' for k, v  in eval_rate.items()]))

        return eval_rate

    def save_and_write(
            self,
            outdir: Union[str, Path],
            mode: str = 'w',
            **kwargs,
    ):
        eval_rate = self.write_eval_rate(outdir, mode=mode, **kwargs)
        outfile = str(Path(outdir).joinpath('step_times.csv'))
        data = self.save_data(outfile, mode=mode)
        return {'eval_rate': eval_rate, 'step_times': data}

    def save_data(self, outfile: str, mode='w'):
        df = pd.DataFrame(self.data)

        fpath = Path(outfile).resolve()
        fpath.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f'Saving step times to: {outfile}')
        df.to_csv(str(fpath), mode=mode)

        return df
