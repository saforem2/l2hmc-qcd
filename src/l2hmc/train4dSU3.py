import os
from pathlib import Path
from typing import Optional

import lovely_tensors as lt
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from l2hmc import get_logger

from l2hmc.common import grab_tensor, print_dict
from l2hmc.configs import dict_to_list_of_overrides, get_experiment
from l2hmc.experiment.pytorch.experiment import evaluate  # noqa  # noqa
from l2hmc.utils.dist import setup_torch
from l2hmc.utils.plot_helpers import set_plot_style

lt.monkey_patch()

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['COLORTERM'] = 'truecolor;'
os.environ['MASTER_PORT'] = '5433'
os.environ['MPLBACKEND'] = 'module://matplotlib-backend-kitty'

log = get_logger(__name__)

_ = setup_torch(precision='float64', backend='DDP', seed=4351)

set_plot_style()

from l2hmc.utils.plot_helpers import (  # noqa
    set_plot_style,
    plot_scalar,
    plot_chains,
    plot_leapfrogs
)


def plot_metrics(metrics: dict, title: Optional[str] = None, **kwargs):
    for key, val in metrics.items():
        fig, ax = plot_metric(val, name=key, **kwargs)
        if title is not None:
            ax.set_title(title)


def plot_metric(
        metric: torch.Tensor,
        name: Optional[str] = None,
        **kwargs,
):
    assert len(metric) > 0
    if isinstance(metric[0], (int, float, bool, np.floating)):
        y = np.stack(metric)
        return plot_scalar(y, ylabel=name, **kwargs)
    element_shape = metric[0].shape
    if len(element_shape) == 2:
        y = grab_tensor(torch.stack(metric))
        return plot_leapfrogs(y, ylabel=name)
    if len(element_shape) == 1:
        y = grab_tensor(torch.stack(metric))
        return plot_chains(y, ylabel=name, **kwargs)
    if len(element_shape) == 0:
        y = grab_tensor(torch.stack(metric))
        return plot_scalar(y, ylabel=name, **kwargs)
    raise ValueError


def main():
    from l2hmc.experiment.pytorch.experiment import train_step
    set_plot_style()

    su3conf = Path('./conf/su3-min.yaml')
    su3conf
    su3conf.is_file()
    with su3conf.open('r') as stream:
        conf = dict(yaml.safe_load(stream))

    overrides = dict_to_list_of_overrides(conf)
    ptExpSU3 = get_experiment(overrides=[*overrides], build_networks=True)
    state = ptExpSU3.trainer.dynamics.random_state(6.0)
    xhmc, history_hmc = evaluate(
        nsteps=10,
        exp=ptExpSU3,
        beta=state.beta,
        x=state.x,
        eps=0.1,
        nleapfrog=1,
        job_type='hmc',
        nlog=1,
        nprint=2,
        grab=True
    )
    plot_metrics(history_hmc, title='HMC', marker='.')
    xeval, history_eval = evaluate(
        nsteps=10,
        exp=ptExpSU3,
        beta=6.0,
        x=state.x,
        job_type='eval',
        nlog=1,
        nprint=2,
        grab=True,
    )
    plot_metrics(history_eval, title='Evaluate', marker='.')
    x = state.x
    for step in range(20):
        log.info(f"train step: {step}")
        x, metrics = ptExpSU3.trainer.train_step((x, state.beta))
        if step % 5 == 0:
            print_dict(metrics, grab=True)


if __name__ == '__main__':
    main()
