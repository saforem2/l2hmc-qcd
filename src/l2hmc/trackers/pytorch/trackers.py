"""
l2hmc/trackers/pytorch/trackers.py
Contains various utilities for tracking / logging metrics in TensorBoard / W&B
"""
from __future__ import absolute_import, annotations, division, print_function
from typing import Optional, Union
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
import wandb
# from l2hmc import get_logger

from l2hmc.common import grab_tensor
from l2hmc.utils.history import StopWatch

Tensor = torch.Tensor
Array = np.ndarray
Scalar = Union[float, int, bool, np.floating]
ArrayLike = Union[Tensor, Array, Scalar]

# log = get_logger(__name__)
log = logging.getLogger(__name__)


def log_dict(
        writer: SummaryWriter,
        d: dict,
        step: Optional[int] = None,
        prefix: Optional[str] = None,
        nchains: Optional[int] = None
) -> None:
    """Create TensorBoard summaries for all items in `d`"""
    for key, val in d.items():
        pre = key if prefix is None else f'{prefix}/{key}'
        if isinstance(val, dict):
            log_dict(
                writer=writer,
                d=val,
                step=step,
                prefix=pre,
                nchains=nchains
            )
        else:
            log_item(
                writer=writer,
                val=val,
                step=step,
                tag=pre,
                nchains=nchains
            )


def log_dict_wandb(
        d: dict,
        step: Optional[int] = None,
        prefix: Optional[str] = None,
        commit: bool = True,
) -> None:
    """Create WandB summaries for all items in `d`"""
    if prefix is not None and step is not None:
        d |= {f'{prefix}/iter': step}
    wandb.log(
        d if prefix is None else {
            f'{prefix}/{k}': v for k, v in d.items()
        },
        commit=commit
    )


def log_list(
        writer: SummaryWriter,
        x: list,
        prefix: str,
        step: Optional[int] = None,
        nchains: Optional[int] = None
) -> None:
    """Create TensorBoard summaries for all entries in `x`"""
    for t in x:
        name = getattr(t, 'name', getattr(t, '__name__', None))
        tag = name if prefix is None else f'{prefix}/{name}'
        assert tag is not None
        log_item(writer=writer, val=t, step=step, tag=tag, nchains=nchains)


def log_step(
        tag: str,
        step: int,
        writer: SummaryWriter
) -> None:
    iter_tag = '/'.join([tag.split('/')[0]] + ['iter'])
    writer.add_scalar(tag=iter_tag, scalar_value=step, global_step=step)


def check_tag(tag: str) -> str:
    tags = tag.split('/')
    return '/'.join(tags[1:]) if len(tags) > 2 and (tags[0] == tags[1]) else tag


def log_item(
        tag: str,
        val: float | int | bool | list | np.ndarray | torch.Tensor,
        writer: SummaryWriter,
        step: Optional[int] = None,
        nchains: Optional[int] = None
) -> None:
    if step is not None:
        log_step(tag, step, writer)
    tag = check_tag(tag)
    if isinstance(val, (Tensor, Array)):
        if nchains is not None and len(val.shape) > 0:
            val = val[:nchains]
        if (
                (isinstance(val, Tensor) and torch.is_complex(val))
                or (isinstance(val, Array) and np.iscomplexobj(val))
        ):
            log_item(tag=f'{tag}/real', val=val.real, writer=writer, step=step)
            log_item(tag=f'{tag}/imag', val=val.imag, writer=writer, step=step)
        elif len(val.shape) > 0:
            writer.add_scalar(f'{tag}/avg', val.mean(), global_step=step)
            val = (
                val[:nchains] if len(val.shape) > 0
                and nchains is not None else val
            )
            if len(val.shape) > 0:
                if nchains is not None:
                    val = val[:nchains]
            try:
                writer.add_histogram(tag=tag, values=val, global_step=step)
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


def as_tensor(
        x: torch.Tensor | list | None,
        grab: bool = False,
        nchains: Optional[int] = None,
) -> torch.Tensor | None | np.ndarray | Scalar:
    if x is None:
        return x
    if nchains is not None:
        try:
            x = x[:nchains]
        except Exception:
            pass
    if isinstance(x, torch.Tensor):
        x = torch.nan_to_num(x)
    if isinstance(x, list):
        x = torch.stack(x)
    return grab_tensor(x) if grab else x


def log_params_and_grads(
        model: nn.Module,
        step: Optional[int] = None,
        with_grads: bool = True,
        nchains: Optional[int] = None,
) -> None:
    if wandb.run is None:
        return
    params = {
        f'params/{k}': as_tensor(v, nchains=nchains)
        for k, v in model.named_parameters()
    }
    grads = {}
    if with_grads:
        grads = {
            f'grads/{k}/grad': as_tensor(v.grad, nchains=nchains)
            for k, v in model.named_parameters()
        }
    if step is not None:
        step_ = torch.tensor(step)
        params |= {'params/iter': step_}
        grads |= {'grads/iter': step_}
    wandb.log(params, commit=False)
    try:
        wandb.log(grads)
    except Exception:
        log.critical(
            'Failed to `wandb.log(grads)` '
        )


def update_summaries(
        writer: SummaryWriter,
        step: Optional[int] = None,
        metrics: Optional[dict[str, ArrayLike]] = None,
        model: Optional[torch.nn.Module] = None,
        prefix: str = '',
        with_grads: bool = True,
        use_tb: bool = True,
        use_wandb: bool = True,
        nchains: int = 8,
        optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    if metrics is not None:
        if use_tb:
            with StopWatch(
                    iter=step,
                    wbtag=f'tblogdict/{prefix}',
                    msg=f"`log_dict(prefix={prefix}, nchains={nchains})`",
                    prefix='TrackingTimers/',
                    log_output=False,
            ):
                log_dict(
                    writer=writer,
                    d=metrics,
                    step=step,
                    prefix=prefix,
                    nchains=nchains
                )
        if use_wandb:
            # pfix = f'{prefix}-wb' if prefix is not None else 'metrics-wb'
            with StopWatch(
                    iter=step,
                    wbtag=f'wblogdict/{prefix}',
                    msg=f"`log_dict_wandb(prefix={prefix})`",
                    prefix='TrackingTimers/',
                    log_output=False,
            ):
                metrics = (
                    {f'{prefix}/wb/{k}': v for k, v in metrics.items()}
                    if use_tb else
                    {f'{prefix}/{k}': v for k, v in metrics.items()}
                )
                log_dict_wandb(metrics, step)
    assert isinstance(step, int) if step is not None else None
    if model is not None:
        if use_wandb:
            with StopWatch(
                    iter=step,
                    wbtag=f"wblogwng/{prefix}",
                    msg="`log_params_and_grads()`",
                    prefix='TrackingTimers/',
                    log_output=False,
            ):
                log_params_and_grads(
                    model=model,
                    step=step,
                    with_grads=with_grads,
                    nchains=nchains
                )
        if use_tb:
            with StopWatch(
                    iter=step,
                    msg='`log_dict(grads)`',
                    prefix='TrackingTimers/',
                    wbtag=f"tblogwng/{prefix}",
                    log_output=False,
            ):
                params = {
                    f'model/{k}': (
                        as_tensor(v, grab=True, nchains=nchains)
                        if v.requires_grad else None
                    )
                    for k, v in model.named_parameters()
                }
                log_dict(writer=writer, d=params, step=step, nchains=nchains)
                if with_grads:
                    grads = {
                        f'grads-wb/{k}': (
                            as_tensor(v.grad, grab=True, nchains=nchains)
                            if v.requires_grad else None
                        )
                        for k, v in model.named_parameters()
                    }
                    log_dict(writer=writer, d=grads, step=step, nchains=nchains)
