"""
l2hmc/trackers/pytorch/trackers.py
Contains various utilities for tracking / logging metrics in TensorBoard
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

Tensor = torch.Tensor
Array = np.ndarray
Scalar = Union[float, int, bool]
ArrayLike = Union[Tensor, Array, Scalar]

log = logging.getLogger(__name__)


def log_item(
        tag: str,
        val: ArrayLike,
        writer: SummaryWriter,
        step: Optional[int] = None,
):
    if step is not None:
        iter_tag = '/'.join([tag.split('/')[0]] + ['iter'])
        writer.add_scalar(tag=iter_tag, scalar_value=step, global_step=step)

    if isinstance(val, (Tensor, Array)):
        if isinstance(val, Tensor):
            val = val.detach()

        if len(val.shape) > 0:
            writer.add_histogram(tag=tag, values=val, global_step=step)
            writer.add_scalar(f'{tag}/avg', val.mean())

        elif isinstance(val, (int, float, bool)) or len(val.shape) == 0:
            writer.add_scalar(tag=tag, scalar_value=val, global_step=step)
        else:
            log.warning(f'Unexpected type encountered for: {tag}')
            log.warning(f'{tag}.type: {type(val)}')


def log_dict(
        writer: SummaryWriter,
        d: dict,
        step: Optional[int] = None,
        prefix: Optional[str] = None
):
    """Create TensorBoard summaries for all items in `d`."""
    for key, val in d.items():
        pre = key if prefix is None else f'{prefix}/{key}'
        if isinstance(val, dict):
            log_dict(writer=writer, d=val, step=step, prefix=pre)
        else:
            log_item(writer=writer, val=val, step=step, tag=pre)


def log_list(
        writer: SummaryWriter,
        x: list,
        prefix: str,
        step: Optional[int] = None,
):
    """Create TensorBoard summaries for all entries in `x`."""
    for t in x:
        if isinstance(t, Tensor):
            t = t.detach().numpy()

        name = getattr(t, 'name', getattr(t, '__name__', None))
        tag = name if prefix is None else f'{prefix}/{name}'
        assert tag is not None
        log_item(writer=writer, val=t, step=step, tag=tag)


def update_summaries(
        writer: SummaryWriter,
        step: Optional[int] = None,
        metrics: Optional[dict[str, ArrayLike]] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        prefix: Optional[str] = None,
) -> None:
    if metrics is not None:
        log_dict(writer=writer, d=metrics, step=step, prefix=prefix)
    assert isinstance(step, int) if step is not None else None
    if model is not None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                tag = f'model/{name}'
                log_item(writer=writer, val=param.detach(), step=step, tag=tag)

    if optimizer is not None:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    tag = f'optimizer/{getattr(p, "name", str(p))}'
                    log_item(writer=writer, val=p.detach(), step=step, tag=tag)
