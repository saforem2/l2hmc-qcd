"""
trainer.py

Implements methods for training L2HMC sampler.
"""
from __future__ import absolute_import, division, print_function, annotations

import torch
import time
from torch import optim
import numpy as np
from src.l2hmc.configs import Steps
from src.l2hmc.dynamics.pytorch.dynamics import Dynamics, to_u1, random_angle
from src.l2hmc.loss.pytorch.loss import LatticeLoss

Tensor = torch.Tensor


def train_step(
        inputs: tuple[Tensor, float],
        dynamics: Dynamics,
        optimizer: optim.Optimizer,
        loss_fn: LatticeLoss,
) -> tuple[Tensor, dict]:
    start = time.time()
    dynamics.train()
    x_init, beta = inputs

    if torch.cuda.is_available():
        x_init = x_init.cuda()

    x_out, metrics = dynamics((to_u1(x_init), beta))
    x_prop = metrics.pop('mc_states').proposed.x
    loss = loss_fn(x_init=x_init, x_prop=x_prop, acc=metrics['acc'])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    metrics['dt'] = time.time() - start
    metrics['loss'] = loss.detach().cpu().numpy()

    return to_u1(x_out).detach(), metrics


def train(
        dynamics: Dynamics,
        optimizer: optim.Optimizer,
        loss_fn: LatticeLoss,
        steps: Steps,
        beta: list[float] | float,
        x: Tensor = None,
        window: int = 10,
        skip: list[str] | str = None,
        keep: list[str] | str = None,
        history: dict = None,
) -> dict:
    dynamics.train()

    if x is None:
        x = random_angle(dynamics.xshape, requires_grad=True)
        x = x.reshape(x.shape[0], -1)

    train_logs = []
    if history is None:
        history = {}

    if isinstance(beta, list):
        assert len(beta) == steps.train

    elif isinstance(beta, float):
        beta = np.array(steps.train * [beta], dtype=np.float32).tolist()
    else:
        raise TypeError(f'beta expected to be `float | list[float]`,\n got: {type(beta)}')


    assert (isinstance(beta, list) and isinstance(beta[0], float))
    for step, b in zip(range(steps.train), beta):
        x, metrics = train_step((to_u1(x), b),
                                dynamics=dynamics,
                                optimizer=optimizer,
                                loss_fn=loss_fn)
        if (step + 1) % steps.log == 0:
            for key, val in metrics.items():
                try:
                    history[key].append(val)
                except KeyError:
                    history[key] = [val]
