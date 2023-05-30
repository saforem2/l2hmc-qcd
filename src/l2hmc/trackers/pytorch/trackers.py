"""
l2hmc/trackers/pytorch/trackers.py
Contains various utilities for tracking / logging metrics in TensorBoard / W&B
"""
from __future__ import absolute_import, annotations, division, print_function
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
import wandb
from l2hmc import get_logger

from l2hmc.common import grab_tensor

Tensor = torch.Tensor
Array = np.ndarray
Scalar = Union[float, int, bool, np.floating]
ArrayLike = Union[Tensor, Array, Scalar]

log = get_logger(__name__)


def log_dict(
        writer: SummaryWriter,
        d: dict,
        step: Optional[int] = None,
        prefix: Optional[str] = None
) -> None:
    """Create TensorBoard summaries for all items in `d`."""
    for key, val in d.items():

        pre = key if prefix is None else f'{prefix}/{key}'
        if isinstance(val, dict):
            log_dict(writer=writer, d=val, step=step, prefix=pre)
        else:
            log_item(writer=writer, val=val, step=step, tag=pre)


def log_dict_wandb(
        d: dict,
        step: Optional[int] = None,
        prefix: Optional[str] = None
) -> None:
    """Create WandB summaries for all items in `d`."""
    if prefix is not None and step is not None:
        d |= {'iter': step}
    wandb.log({prefix: d}) if prefix is not None else wandb.log(d)


def log_list(
        writer: SummaryWriter,
        x: list,
        prefix: str,
        step: Optional[int] = None,
) -> None:
    """Create TensorBoard summaries for all entries in `x`."""
    for t in x:
        name = getattr(t, 'name', getattr(t, '__name__', None))
        tag = name if prefix is None else f'{prefix}/{name}'
        assert tag is not None
        log_item(writer=writer, val=t, step=step, tag=tag)


def log_step(
        tag: str,
        step: int,
        writer: SummaryWriter
) -> None:
    iter_tag = '/'.join([tag.split('/')[0]] + ['iter'])
    writer.add_scalar(tag=iter_tag, scalar_value=step, global_step=step)


def check_tag(tag: str) -> str:
    tags = tag.split('/')
    if len(tags) > 2 and (tags[0] == tags[1]):
        return '/'.join(tags[1:])
    return tag


def log_item(
        tag: str,
        val: float | int | bool | list | np.ndarray | torch.Tensor,
        writer: SummaryWriter,
        step: Optional[int] = None,
) -> None:
    if step is not None:
        log_step(tag, step, writer)

    tag = check_tag(tag)
    if isinstance(val, (Tensor, Array)):
        if (
                (isinstance(val, Tensor) and torch.is_complex(val))
                or (isinstance(val, Array) and np.iscomplexobj(val))
        ):
            log_item(tag=f'{tag}.real', val=val.real, writer=writer, step=step)
            log_item(tag=f'{tag}.imag', val=val.imag, writer=writer, step=step)
        elif len(val.shape) > 0:
            writer.add_scalar(f'{tag}/avg', val.mean(), global_step=step)
            if len(val.shape) > 0:
                try:
                    writer.add_histogram(
                        tag=tag,
                        values=val,
                        global_step=step
                    )
                except ValueError:
                    log.error(f'Error adding histogram for: {tag}')
        else:
            writer.add_scalar(tag, val, global_step=step)

    elif isinstance(val, list):
        log_list(writer=writer, x=val, step=step, prefix=tag)

    elif (
            isinstance(val, (float, int, bool, np.floating))
            or len(val.shape) == 0
    ):
        writer.add_scalar(tag=tag, scalar_value=val, global_step=step)

    else:
        log.warning(f'Unexpected type encountered for: {tag}')
        log.warning(f'{tag}.type: {type(val)}')


def log_params_and_grads(
        model: nn.Module,
        step: Optional[int] = None,
        with_grads: bool = True,
) -> None:
    params = {
        f'params/{k}': (
            torch.nan_to_num(v)
            if v is not None
            else None
        )
        for k, v in model.named_parameters()
    }
    grads = {}
    if with_grads:
        grads = {
            f'grads/{k}.grad': (
                torch.nan_to_num(v.grad)
                if v.grad is not None
                else None
            )
            for k, v in model.named_parameters()
        }
    if step is not None:
        params |= {'iter': step}
        grads |= {'iter': step}
    wandb.log(params, commit=False)
    wandb.log(grads)


def update_summaries(
        writer: SummaryWriter,
        step: Optional[int] = None,
        metrics: Optional[dict[str, ArrayLike]] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        prefix: Optional[str] = None,
        with_grads: bool = True,
) -> None:
    if metrics is not None:
        log_dict(writer=writer, d=metrics, step=step, prefix=prefix)
    assert isinstance(step, int) if step is not None else None
    if model is not None:
        log_params_and_grads(model=model, step=step, with_grads=with_grads)
        params = {
            f'model/{k}': (
                grab_tensor(v) if v.requires_grad else None
            )
            for k, v in model.named_parameters()
        }
        log_dict(writer=writer, d=params, step=step)
    # if model is not None:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             tag = f'model/{name}'
    #             log_item(writer=writer, val=param, step=step, tag=tag)
    # if optimizer is not None:
    #     for group in optimizer.param_groups:
    #         for p in group['params']:
    #             if p.grad is not None:
    #                 tag = f'optimizer/{getattr(p, "name", str(p))}'
    #                 log_item(writer=writer, val=p.detach(), step=step, tag=tag)
