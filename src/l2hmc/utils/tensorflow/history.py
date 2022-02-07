"""
tensorflow/history.py

Implements tfHistory, containing minor modifications from base History class.
"""
from __future__ import absolute_import, print_function, division, annotations

from typing import Any

import tensorflow as tf
import numpy as np
from l2hmc.utils.history import BaseHistory


class History(BaseHistory):
    def update(self, metrics: dict) -> dict:
        avgs = {}
        era = metrics.get('era', 0)
        for key, val in metrics.items():
            avg = None
            if isinstance(val, (float, int)):
                avg = val
            else:
                if isinstance(val, dict):
                    for k, v in val.items():
                        key = f'{key}/{k}'
                        try:
                            avg = self._update(key=key, val=v)
                        # TODO: Figure out how to deal with exception
                        except tf.errors.InvalidArgumentError:
                            continue
                else:
                    avg = self._update(key=key, val=val)

            if avg is not None:
                avgs[key] = avg
                try:
                    self.era_metrics[str(era)][key].append(avg)
                except KeyError:
                    self.era_metrics[str(era)][key] = [avg]

        return avgs

    def _update(self, key: str, val: Any) -> float:
        if val is None:
            raise ValueError(f'None encountered: {key}: {val}')

        if isinstance(val, list):
            val = np.array(val)

        try:
            self.history[key].append(val)
        except KeyError:
            self.history[key] = [val]

        if isinstance(val, (float, int)):
            return val

        try:
            return tf.reduce_mean(val)
        except Exception:
            return val
