"""
l2hmc/trackers/pytorch/trackers.py

Contains various utilities for tracking / logging metrics in TensorBoard
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
from typing import Optional, Union, Any

import numpy as np
import wandb
import torch
# from torch.utils.tensorboard.writer import SummaryWriter

Tensor = torch.Tensor
Array = np.ndarray
Scalar = Union[float, int, bool]
ArrayLike = Union[Tensor, Array, Scalar]

log = logging.getLogger(__name__)


def log_item(
        tag: str,
        val: ArrayLike,
        run: Optional[Any] = None,
        step: Optional[Any] = None,
        commit: Optional[bool] = True,
):
    if isinstance(val, Tensor):
        val = val.detach().cpu().numpy()

    iter_tag = '/'.join([tag.split('/')[0]] + ['iter'])
    data = {tag: val} if step is None else {tag: val, iter_tag: step}

    if isinstance(val, np.ndarray):
        data.update({f'{tag}/avg': val.mean()})
        if len(val.shape) > 1:
            data.update({
                tag: val.reshape(val.shape[0], -1),
            })

    if run is not None:
        run.log(data, commit=commit)


def log_dict(
        d: dict,
        run: Optional[Any] = None,
        step: Optional[int] = None,
        prefix: Optional[str] = None,
        commit: Optional[bool] = True,
):
    """Create summaries for all items in `d`."""
    for idx, (key, val) in enumerate(d.items()):
        c = (commit and idx == len(list(d.keys())) - 1)
        pre = key if prefix is None else f'{prefix}/{key}'
        if isinstance(val, dict):
            log_dict(run=run, step=step,
                     d=val, prefix=pre, commit=(c and commit))
        else:
            log_item(run=run, step=step, val=val, tag=pre, commit=c)


def log_list(
        x: list,
        prefix: str,
        run: Optional[Any] = None,
        commit: Optional[bool] = True,
        step: Optional[int] = None,
):
    """Create summaries for all entries in `x`."""
    for idx, t in enumerate(x):
        if isinstance(t, Tensor):
            t = t.detach().numpy()

        name = getattr(t, 'name', None)
        tag = f'{prefix}/{name}' if name is not None else prefix
        c = (commit and idx == len(x) - 1)
        log_item(run=run, val=t, step=step, tag=tag, commit=c)


def update_summaries(
        run: Optional[Any] = None,
        step: Optional[int] = None,
        metrics: Optional[dict[str, ArrayLike]] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        prefix: Optional[str] = None,
) -> None:
    commit = (model is None and optimizer is None)
    if run is not None and metrics is not None:
        if prefix is not None:
            run.log({prefix: metrics}, commit=commit)
        else:
            run.log(metrics, commit=commit)

    if metrics is not None:
        log_dict(d=metrics, step=step, prefix=prefix, run=run)
        if run is not None and run is wandb.run:
            if prefix is not None:
                run.log({prefix: metrics}, commit=commit)
            else:
                run.log(metrics, commit=commit)
    assert isinstance(step, int) if step is not None else None
    if model is not None:
        commit = (optimizer is None)
        for name, param in model.named_parameters():
            if param.requires_grad:
                tag = f'model/{name}'
                val = param.detach()
                log_item(run=run, val=val, step=step, tag=tag, commit=commit)

    if optimizer is not None:
        assert isinstance(optimizer, torch.optim.Optimizer)
        state_dict = optimizer.state_dict()
        param_groups = state_dict.get('param_groups', None)
        if param_groups is not None:
            for idx, (key, val) in enumerate(param_groups.items()):
                c = (idx == (len(list(param_groups.keys())) - 1))
                if isinstance(val, dict):
                    log_dict(d=val,
                             step=step,
                             run=run,
                             commit=c,
                             prefix=f'optimizer/{key}')
                elif isinstance(val, Tensor) and val.grad is not None:
                    t = f'optimizer/{key}'
                    log_item(run=run, val=val.detach(), step=step, tag=t,
                             commit=c)
