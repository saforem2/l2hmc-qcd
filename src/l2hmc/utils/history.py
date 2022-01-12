"""
history.py

Contains implementation of History object for tracking / aggregating metrics.
"""
from __future__ import absolute_import, division, print_function, annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from src.l2hmc.configs import Steps, MonteCarloStates


@dataclass
class StateHistory:
    def __post_init__(self):
        self.init = {'x': [], 'v': []}
        self.proposed = {'x': [], 'v': []}
        self.out = {'x': [], 'v': []}

    def update(self, mc_states: MonteCarloStates) -> None:
        self.init['x'].append(mc_states.init.x.numpy())
        self.init['v'].append(mc_states.init.v.numpy())
        self.proposed['x'].append(mc_states.proposed.x.numpy())
        self.proposed['v'].append(mc_states.proposed.v.numpy())
        self.out['x'].append(mc_states.out.x.numpy())
        self.out['v'].append(mc_states.out.v.numpy())


class History:
    def __init__(self, steps: Steps):
        self.steps = steps
        self.history = {}
        self.era_metrics = {str(era): {} for era in range(steps.nera)}

    def _update(self, key: str, val: Any) -> float:
        if isinstance(val, list):
            val = np.array(val)

        if hasattr(val, 'numpy'):
            val = val.numpy()

        try:
            self.history[key].append(val)
        except KeyError:
            self.history[key] = [val]

        if isinstance(val, (float, int)):
            return val

        return val.mean()

    def era_summary(self, era) -> str:
        emetrics = self.era_metrics[str(era)]
        return '\n'.join([
            f'- {k}={np.mean(v):<3.2g}' for k, v in emetrics.items()
        ])

    def update(self, metrics: dict) -> str:
        avgs = {}
        era = metrics.get('era', None)
        assert era is not None
        for key, val in metrics.items():
            name = key
            avg = None
            if isinstance(val, dict):
                for k, v in val.items():
                    name = f'{key}/{k}'
                    try:
                        avg = self._update(key=name, val=v)
                    except Exception:
                        continue
            else:
                avg = self._update(key=key, val=val)

            avgs[name] = avg
            try:
                self.era_metrics[str(era)][name].append(avg)
            except KeyError:
                self.era_metrics[str(era)][name] = [avg]

        return ', '.join([f'{key}={val:<3.2f}' for key, val in avgs.items()])
