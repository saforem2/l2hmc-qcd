"""
l2hmc/trackers/tensorflow/trackers.py

Contains various utilities for tracking / logging metrics in TensorBoard
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
from typing import Optional, Union, Any

import numpy as np
import tensorflow as tf
import wandb
from tensorflow.python.keras import backend as K


log = logging.getLogger(__name__)

Array = np.ndarray
Tensor = tf.Tensor
Model = tf.keras.models.Model
Scalar = Union[float, int, bool]
ArrayLike = Union[Tensor, Array, Scalar]
Optimizer = tf.keras.optimizers.Optimizer


def log_item(
        tag: str,
        val: ArrayLike,
        run: Optional[Any] = None,
        step: Optional[int] = None,
        commit: Optional[bool] = True,
):
    iter_tag = '/'.join([tag.split('/')[0]] + ['iter'])
    tag = tag.rstrip(':0')
    data = {tag: val} if step is None else {tag: val, iter_tag: step}
    if isinstance(val, Tensor):
        assert tf.is_tensor(val)
        val = val.numpy()  # type:ignore

    if isinstance(val, np.ndarray):
        data.update({f'{tag}/avg': val.mean()})
        if len(val.shape) > 1:
            data.update({tag: val.reshape(val.shape[0], -1)})

    if run is not None:
        run.log(data, commit=commit)


def log_item_old(
        tag: str,
        val: ArrayLike,
        step: Optional[int] = None,
):
    if (
            'dt' in tag
            or 'era' in tag
            or 'epoch' in tag
            or isinstance(val, (int, float, bool))
    ):
        tf.summary.scalar(tag, val, step=step)
    else:
        tf.summary.histogram(tag, val, step=step)
        tf.summary.scalar(f'{tag}/avg', tf.reduce_mean(val), step=step)


def log_dict(
        d: dict,
        step: int,
        prefix: Optional[str] = None,
        run: Optional[Any] = None,
        commit: Optional[bool] = True,
):
    """Create tensorboard summaries for all items in `d`"""
    for idx, (key, val) in enumerate(d.items()):
        c = (commit and (idx == len(list(d.keys())) - 1))
        pre = key if prefix is None else f'{prefix}/{key}'
        if isinstance(val, dict):
            log_dict(val, step=step, prefix=pre, commit=(c and commit))
        else:
            log_item(pre, val, step=step, run=run, commit=commit)


def log_list(
        x,
        step,
        prefix=None,
        run: Optional[Any] = None,
        commit: Optional[bool] = True,
):
    for idx, t in enumerate(x):
        name = getattr(t, 'name', None)
        c = (commit and (idx == len(x) - 1))
        tag = f'{prefix}/{name}' if name is not None else prefix
        log_item(tag, t, step=step, run=run, commit=c)


def update_summaries(
        step: int,
        run: Optional[Any] = None,
        metrics: Optional[dict] = None,
        model: Optional[Model] = None,
        optimizer: Optional[Optimizer] = None,
        prefix: Optional[str] = None,
):
    """"Create summary objects."""
    commit = (model is None and optimizer is None)
    if run is not None and metrics is not None:
        if prefix is not None:
            run.log({prefix: metrics}, commit=commit)
        else:
            run.log(metrics, commit=commit)

    if metrics is not None:
        log_dict(metrics, step, prefix=prefix, run=run)
        if run is not None and run is wandb.run:
            if prefix is not None:
                run.log({prefix: metrics}, commit=commit)
            else:
                run.log(metrics, commit=commit)

    if model is not None:
        commit = (optimizer is not None)
        log_list(
            model.variables,
            run=run,
            step=step,
            prefix='model',
            commit=commit,
        )

    if optimizer is not None:
        ostr = 'optimizer'
        opre = ostr if prefix is None else '/'.join([ostr, prefix])
        lpre = 'lr' if prefix is None else '/'.join(['lr', prefix])
        if run is not None:
            run.log(
                {lpre: K.get_value(optimizer.lr), f'{lpre}/iter': step},
                commit=False
            )
        log_list(
            optimizer.variables(),
            run=run,
            step=step,
            prefix=opre,
            commit=True
        )
