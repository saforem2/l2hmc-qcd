"""
l2hmc/trackers/tensorflow/trackers.py

Contains various utilities for tracking / logging metrics in TensorBoard
"""
from __future__ import absolute_import, print_function, division, annotations
from typing import Union

import tensorflow as tf
import numpy as np
import logging
from tensorflow.python.keras import backend as K


log = logging.getLogger(__name__)

Model = tf.keras.models.Model
Optimizer = tf.keras.optimizers.Optimizer
Tensor = tf.Tensor
Array = np.ndarray
Scalar = Union[float, int, bool]
ArrayLike = Union[Tensor, Array, Scalar]


def log_item(
        tag: str,
        val: ArrayLike,
        step: int = None,
):
    if (
            'dt' in tag
            or 'beta' in tag
            or 'era' in tag
            or 'epoch' in tag
            or isinstance(val, (float, int, bool))
    ):
        tf.summary.scalar(tag, val, step=step)

    else:
        tf.summary.histogram(tag, val, step=step)
        tf.summary.scalar(f'{tag}/avg', tf.reduce_mean(val), step=step)

    # if tag in ['era', 'epoch', 'dt', 'beta']:
    #     tf.summary.scalar(tag, val, step=step)
    # else:
    #     tf.summary.histogram(tag, val, step=step)
    #     tf.summary.histogram(f'{tag}_avg', tf.reduce_mean(val), step=step)
    #     # elif isinstance(val, (Tensor, Array)):
    #     #     if len(val.shape) > 1:
    #     #         tf.summary.histogram(tag, val, step=step)
    #     #         tf.summary.scalar(
    #     #             f'{tag}_avg', tf.reduce_mean(val), step=step
    #     #         )
    #     #     else:
    #     #         tf.summary.scalar(tag, val, step=step)


def log_dict(d: dict, step: int, prefix: str = None):
    """Create tensorboard summaries for all items in `d`"""
    for key, val in d.items():
        pre = key if prefix is None else f'{prefix}/{key}'
        if isinstance(val, dict):
            for k, v in val.items():
                log_item(f'{pre}/{k}', v, step=step)
        else:
            log_item(pre, val, step=step)


def log_list(x, step, prefix=None):
    for t in x:
        name = getattr(t, 'name', None)
        tag = f'{prefix}/{name}' if name is not None else prefix
        log_item(tag, t, step=step)


def update_summaries(
        step: int,
        metrics: dict = None,
        model: Model = None,
        optimizer: Optimizer = None,
        prefix: str = None,
):
    """"Create summary objects."""
    if metrics is not None:
        pre = 'metrics' if prefix is None else '/'.join([prefix, 'metrics'])
        log_dict(metrics, step, prefix=pre)

    if model is not None:
        mpre = 'model' if prefix is None else '/'.join(['model', prefix])
        log_list(model.variables, step=step, prefix=mpre)

    if optimizer is not None:
        ostr = 'optimizer'
        opre = ostr if prefix is None else '/'.join([ostr, prefix])
        lpre = 'lr' if prefix is None else '/'.join(['lr', prefix])
        tf.summary.scalar(lpre, K.get_value(optimizer.lr), step=step)
        log_list(optimizer.variables(), step=step, prefix=opre)
