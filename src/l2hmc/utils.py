"""
utils.py

Contains various utilities for training L2HMC sampler.
"""
from __future__ import absolute_import, division, print_function, annotations

import os
import time

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr


import logging
logger = logging.getLogger('l2hmc')


class Metrics:
    def __init__(self, keys: Optional[list[str]] = None) -> None:
        self._keys = [] if keys is None else keys
        self.data = {
            key: [] for key in self._keys
        }

    def update(self, metrics: dict) -> None:
        for key, val in metrics.items():
            try:
                self.data[key].append(val)
            except KeyError:
                self.data[key] = [val]


class StepTimer:
    def __init__(self, evals_per_update: int = 1):
        self.data = []
        self.t = time.time()
        self.evals_per_update = evals_per_update

    def start(self):
        self.t = time.time()

    def stop(self):
        dt = time.time() - self.t
        self.data.append(dt)

        return dt

    def reset(self):
        self.data = []

    def get_eval_rate(
            self,
            evals_per_update: int = 1,
    ) -> dict:
        eper = (
            self.evals_per_update if evals_per_update is None
            else evals_per_update
        )
        elapsed = np.sum(self.data)
        nevals = eper * len(self.data)
        eval_rate = nevals / elapsed
        return {
            'eval_rate': eval_rate,
            'total_time': elapsed,
            'num_evals': nevals,
            'num_steps': len(self.data),
            'evals_per_update': evals_per_update,
        }

    def write_eval_rate(
            self,
            outdir: os.PathLike,
            mode: str = 'w',
            evals_per_update: int = 1,
    ):
        eval_rate = self.get_eval_rate(evals_per_update)
        outfile = Path(outdir).joinpath('eval_rate.txt')
        logger.debug(f'Writing eval rate to: {outfile}')
        with open(outfile, mode) as f:
            f.write('\n'.join([f'{k}: {v}' for k, v  in eval_rate.items()]))

        return eval_rate

    def save_and_write(
            self,
            outdir: os.PathLike,
            mode: str = 'w',
            evals_per_update: int = 1,
    ):
        eval_rate = self.write_eval_rate(outdir, mode, evals_per_update)
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


def _summarize_metric(key: str, x: np.ndarray, window: int = 10) -> str:
    assert isinstance(x, np.ndarray)
    x = x.squeeze()
    if window > 0 and len(x.shape) > 0:
        window = min((x.shape[0], window))
        return f'{str(key)}={np.mean(x[-window:]):<3.2g}'
    return f'{str(key)}={np.mean(x):<3.2g}'


def _summarize_float(key: str, x: float) -> str:
    return f'{str(key)}={x:<3.2f}'


def summarize_items(
        key: str,
        x: dict | list | np.ndarray | float,
        window: int = 10,
) -> str:
    if isinstance(x, dict):
        dstr = ', '.join([
            _summarize_metric(k, v, window) for k, v in x.items()
        ])
        return '\n'.join([str(key), f' {dstr}'])
    elif isinstance(x, float):
        return _summarize_float(key, x)
    elif isinstance(x, list):
        return _summarize_metric(key, np.array(x), window)
    elif isinstance(x, np.ndarray):
        return _summarize_metric(key, x, window)
    raise TypeError(
        'Expected `x` to be one of: [dict, float, list, np.ndarray]'
        f'Instead, got: {type(x)}'
    )


class History:
    def __init__(
            self,
            keys: Optional[list[str]] = None,
            timer: Optional[StepTimer] = None,
            data: Optional[dict] = None,
            evals_per_update: int = 1,
    ) -> None:
        self.steps = []
        self._evals_per_update = evals_per_update
        self._keys = keys if keys is not None else []
        self.data_avgs = {key: [] for key in self._keys}
        self.data = data if data is not None else {k: [] for k in self._keys}
        self.timer = timer if timer is None else StepTimer(evals_per_update)

    def _reset(self):
        self.steps = []
        self.running_avgs = {key: [] for key in self._keys}
        self.data = {key: [] for key in self._keys}

    @staticmethod
    def _running_avg(x: list, window: int = 10) -> np.ndarray:
        arr = np.array(x)
        win = min((window, arr.shape[0]))
        if len(arr.shape) == 1:
            return arr[-win:].mean()

        return arr[-win:].mean(0)

    def _running_avgs(self, window: int) -> dict[str, np.ndarray]:
        return {k: self._running_avg(v, window) for k, v in self.data.items()}

    def update_running_avgs(self, window: int) -> dict:
        ravgs = self._running_avgs(window)
        self.data_avgs.update(**ravgs)
        return ravgs

    def summarize_dict(
            self,
            data: dict,
            window: int = 0,
            pre: Optional[Union[str, list[str]]] = None,
            skip: Optional[Union[str, list[str]]] = None,
            keep: Optional[Union[str, list[str]]] = None,
    ) -> str:
        skip = [] if skip is None else skip
        keep = list(data.keys()) if keep is None else keep
        fstrs = [
            summarize_items(k, v, window) for k, v in data.items()
            if k not in skip and k in keep
        ]
        if pre is not None:
            fstrs = [pre, *fstrs] if isinstance(pre, str) else [*pre] + fstrs

        return ' '.join(fstrs)

    def summary(
            self,
            data: Optional[dict] = None,
            window: int = 1,
            pre: Optional[Union[str, list[str]]] = None,
            skip: Optional[Union[str, list[str]]] = None,
            keep: Optional[Union[str, list[str]]] = None,
    ) -> str:
        data = self.data if data is None else data
        return self.summarize_dict(data, window, pre=pre, skip=skip, keep=keep)

    def update(
            self,
            metrics: dict,
            step: int = None,
    ) -> None:
        if step is not None:
            self.steps.append(step)

        for k, v in metrics.items():
            try:
                self.data[k].append(v)
            except KeyError:
                self.data[k] = [v]

    def to_DataArray(self, x: Union[list, np.ndarray]) -> xr.DataArray:
        arr = np.array(x)
        # steps = np.arange(len(arr))
        if len(arr.shape) == 1:                     # [ndraws]
            ndraws = arr.shape[0]
            dims = ['draw']
            coords = [np.arange(len(arr))]
            return xr.DataArray(arr, dims=dims, coords=coords)

        if len(arr.shape) == 2:                   # [nchains, ndraws]
            arr = arr.T
            nchains, ndraws = arr.shape
            dims = ('chain', 'draw')
            coords = [np.arange(nchains), np.arange(ndraws)]
            return xr.DataArray(arr, dims=dims, coords=coords)

        if len(arr.shape) == 3:                   # [nchains, nlf, ndraws]
            arr = arr.T
            nchains, nlf, ndraws = arr.shape
            dims = ('chain', 'leapfrog', 'draw')
            coords = [np.arange(nchains), np.arange(nlf), np.arange(ndraws)]
            return xr.DataArray(arr, dims=dims, coords=coords)

        else:
            raise ValueError('Invalid shape encountered')

    def get_dataset(self, data: Optional[dict[str, list]] = None):
        data = self.data if data is None else data
        data_vars = {}
        for key, val in data.items():
            # TODO: FIX ME
            # if isinstance(val, list):
            #     if isinstance(val[0], (dict, AttrDict)):
            #         tmp = invert
            #      data_vars[key] = dataset = self.get_dataset(val)

            data_vars[key] = self.to_DataArray(val)

        return xr.Dataset(data_vars)
