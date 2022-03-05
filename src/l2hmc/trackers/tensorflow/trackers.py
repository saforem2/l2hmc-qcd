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
            or 'era' in tag
            or 'epoch' in tag
            or isinstance(val, (int, float, bool))
    ):
        tf.summary.scalar(tag, val, step=step)
    else:
        tf.summary.histogram(tag, val, step=step)
        tf.summary.scalar(f'{tag}/avg', tf.reduce_mean(val), step=step)


def log_dict(d: dict, step: int, prefix: str = None):
    """Create tensorboard summaries for all items in `d`"""
    for key, val in d.items():
        pre = key if prefix is None else f'{prefix}/{key}'
        if isinstance(val, dict):
            log_dict(val, step=step, prefix=pre)
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
        log_dict(metrics, step, prefix=prefix)

    if model is not None:
        log_list(model.variables, step=step, prefix='model')

    if optimizer is not None:
        ostr = 'optimizer'
        opre = ostr if prefix is None else '/'.join([ostr, prefix])
        lpre = 'lr' if prefix is None else '/'.join(['lr', prefix])
        tf.summary.scalar(lpre, K.get_value(optimizer.lr), step=step)
        log_list(optimizer.variables(), step=step, prefix=opre)
