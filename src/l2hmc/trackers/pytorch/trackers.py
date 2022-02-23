"""
l2hmc/trackers/pytorch/trackers.py

Contains various utilities for tracking / logging metrics in TensorBoard
"""
from __future__ import absolute_import, print_function, division, annotations
from typing import Union

import torch
from torch.utils.tensorboard.writer import FileWriter
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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
        return
    else:
        writer.add_histogram(tag=tag, values=val, global_step=step)
        writer.add_scalar(tag=f'{tag}_avg',
                          global_step=step,
                          scalar_value=val.mean())
        # elif isinstance(val, (Tensor, Array)):
        #     if len(val.shape) > 1:
        #         writer.add_histogram(tag=tag,
        #                              values=val,
        #                              global_step=step)
        #         writer.add_scalar(tag=f'{tag}_avg',
        #                           global_step=step,
        #                           scalar_value=val.mean())
        #     elif isinstance(val, (float, int, bool)):
        #         writer.add_scalar(tag=tag,
        #                           global_step=step,
        #                           scalar_value=val)


def log_dict(
        writer: SummaryWriter,
        d: dict,
        step: int = None,
        prefix: str = None
):
    """Create TensorBoard summaries for all items in `d`."""
    prefix = '' if prefix is None else prefix
    for key, val in d.items():
        pre = key if prefix is None else f'{prefix}/{key}'
        if isinstance(val, dict):
            log_dict(writer=writer, d=val, step=step, prefix=pre)
        elif isinstance(val, list):
            log_list(writer=writer, x=val, step=step, prefix=pre)
        else:
            log_item(writer=writer, val=val, step=step, tag=pre)


def log_list(
        writer: SummaryWriter,
        x: list,
        step: int = None,
        prefix: str = None
):
    """Create TensorBoard summaries for all entries in `x`."""
    for t in x:
        name = ''
        if hasattr(t, 'name'):
            name = getattr(t, 'name', '')

        tag = f'{prefix}/{name}' if prefix is not None else name
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
        pre = 'model' if prefix is None else '/'.join([prefix, 'metrics'])
        for name, param in model.named_parameters():
            if param.requires_grad:
                tag = f'{pre}/{name}'
                log_item(writer=writer, val=param, step=step, tag=tag)
