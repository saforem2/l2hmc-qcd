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

# from l2hmc.network.pytorch.network import nested_children
# from l2hmc.utils.logger import get_pylogger
# from l2hmc.common import grab_tensor

# from l2hmc.common import grab_tensor

Tensor = torch.Tensor
Array = np.ndarray
Scalar = Union[float, int, bool, np.floating]
ArrayLike = Union[Tensor, Array, Scalar]

# log = get_pylogger(__name__)
log = logging.getLogger(__name__)


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


def log_list(
        writer: SummaryWriter,
        x: list,
        prefix: str,
        step: Optional[int] = None,
) -> None:
    """Create TensorBoard summaries for all entries in `x`."""
    for t in x:
        # if isinstance(t, Tensor):
        #     # t = grab_tensor(t)
        #     t = t.detach().numpy()

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
    return '/'.join(tags[1:]) if len(tags) > 2 and (tags[0] == tags[1]) else tag


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
                log_item(writer=writer, val=param, step=step, tag=tag)
        # children = nested_children(model)
        # for name, child in children.items():
        #     log_item(
        #         writer=writer,
        #         val=child.parameters(),
        #         step=step,
        #         tag=tag
        #     )
        # for m in model.register_buffer
        #     if isinstance(m, nn.Linear):

    if optimizer is not None:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    tag = f'optimizer/{getattr(p, "name", str(p))}'
                    log_item(writer=writer, val=p.detach(), step=step, tag=tag)
