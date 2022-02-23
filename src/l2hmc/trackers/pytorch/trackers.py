"""
l2hmc/trackers/pytorch/trackers.py

Contains various utilities for tracking / logging metrics in TensorBoard
"""
from __future__ import absolute_import, print_function, division, annotations
from typing import Union

import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

Tensor = torch.Tensor
Array = np.ndarray
Scalar = Union[float, int, bool]
ArrayLike = Union[Tensor, Array, Scalar]


def log_item(
        tag: str,
        val: ArrayLike,
        writer: SummaryWriter,
        step: int = None,
):
    if (
            'dt' in tag
            or 'beta' in tag
            or 'era' in tag
            or 'epoch' in tag
            or isinstance(val, (float, int, bool))
    ):
        writer.add_scalar(tag=tag, scalar_value=val, global_step=step)
    else:
        writer.add_histogram(tag=tag, values=val, global_step=step)
        writer.add_scalar(tag=f'{tag}/avg',
                          global_step=step,
                          scalar_value=val.mean())


def log_dict(
        writer: SummaryWriter,
        d: dict,
        step: int = None,
        prefix: str = None
):
    """Create TensorBoard summaries for all items in `d`."""
    for key, val in d.items():
        pre = key if prefix is None else f'{prefix}/{key}'
        if isinstance(val, dict):
            for k, v in val.items():
                log_item(writer=writer, tag=f'{pre}/{k}', val=v, step=step)
        else:
            log_item(writer=writer, val=val, step=step, tag=pre)


def log_list(
        writer: SummaryWriter,
        x: list,
        prefix: str,
        step: int = None,
):
    """Create TensorBoard summaries for all entries in `x`."""
    for t in x:
        name = getattr(t, 'name', None)
        tag = f'{prefix}/{name}' if name is not None else prefix
        log_item(writer=writer, val=t, step=step, tag=tag)


def update_summaries(
        writer: SummaryWriter,
        step: int = None,
        metrics: dict[str, ArrayLike] = None,
        model: torch.nn.Module = None,
        # optimizer: torch.optim.Optimizer = None,
        prefix: str = None,
):
    if metrics is not None:
        pre = 'metrics' if prefix is None else '/'.join([prefix, 'metrics'])
        log_dict(writer=writer, d=metrics, step=step, prefix=pre)
    if model is not None:
        pre = 'model' if prefix is None else '/'.join(['model', prefix])
        for name, param in model.named_parameters():
            if param.requires_grad:
                tag = f'{pre}/{name}'
                log_item(writer=writer, val=param, step=step, tag=tag)
