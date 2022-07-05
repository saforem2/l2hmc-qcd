"""
l2hmc/trackers/tensorflow/trackers.py
Contains various utilities for tracking / logging metrics in TensorBoard
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
from typing import Optional, Union

import numpy as np
import tensorflow as tf
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
        step: Optional[int] = None,
):
    if step is not None:
        iter_tag = '/'.join([tag.split('/')[0]] + ['iter'])
        tf.summary.scalar(iter_tag, step, step=step)

    if isinstance(val, list):
        log_list(val, step=step, prefix=tag)

    if isinstance(val, (Tensor, Array)):
        if len(val.shape) > 0:
            tf.summary.histogram(tag, val, step=step)
            tf.summary.scalar(f'{tag}/avg', tf.reduce_mean(val), step=step)
    elif isinstance(val, (int, float, bool)) or len(val.shape) == 0:
        tf.summary.scalar(tag, val, step=step)

    # if (
    #         'dt' in tag
    #         # or 'beta' in tag
    #         or 'era' in tag
    #         or 'epoch' in tag
    #         or 'loss' in tag
    #         or isinstance(val, (int, float, bool))
    # ):
    #     tf.summary.scalar(tag, val, step=step)
    # else:
    #     tf.summary.histogram(tag, val, step=step)
    #     tf.summary.scalar(f'{tag}/avg', tf.reduce_mean(val), step=step)


def log_dict(d: dict, step: int, prefix: Optional[str] = None):
    """Create tensorboard summaries for all items in `d`"""
    for key, val in d.items():
        pre = key if prefix is None else f'{prefix}/{key}'
        if isinstance(val, dict):
            log_dict(val, step=step, prefix=pre)
        elif isinstance(val, list):
            log_list(val, step, prefix=prefix)
        else:
            log_item(pre, val, step=step)


def log_list(x, step, prefix: Optional[str] = None):
    for idx, t in enumerate(x):
        name = getattr(t, 'name', getattr(t, '__name__', None))
        if name is None:
            name = f'{idx}'
        tag = name if prefix is None else f'{prefix}/{name}'
        assert tag is not None
        log_item(tag, t, step=step)


def update_summaries(
        step: int,
        metrics: Optional[dict] = None,
        model: Optional[Model] = None,
        optimizer: Optional[Optimizer] = None,
        prefix: Optional[str] = None,
):
    """"Create summary objects."""
    if metrics is not None:
        log_dict(metrics, step, prefix=prefix)

    if model is not None:
        log_list(model.variables, step=step, prefix='model')
        # weights = {
        #     'model/iter': step,
        #     'model/variables': model.variables,
        # }
        for layer in model.layers:
            # w = layer.get_weights()
            weights = layer.get_weights()
            log_list(
                weights,
                step=step,
                prefix=f'model/{layer.name}.weights'
            )
            # log_list(
            #     w,
            #     step=step,
            #     prefix=f'model/{layer.name}.weights',
            # )
            # log_item(
            #     tag=f'model/{layer.name}.bias',
            #     val=b,
            #     step=step,
            # )
            # log_item(w, step=step, prefix=f'model/{layer.name}.weight')
            # log_list(b, step=step, prefix=f'model/{layer.name}.bias')
            # weights[layer] = wb
        # log_dict(weights, step=step, prefix='model')

        # log_dict(weights, step=step, prefix='weights')
        # log_list(weights, step=step, prefix=f'model/{layer}')

    if optimizer is not None:
        ostr = 'optimizer'
        opre = ostr if prefix is None else '/'.join([ostr, prefix])
        lpre = 'lr' if prefix is None else '/'.join(['lr', prefix])
        tf.summary.scalar(lpre, K.get_value(optimizer.lr), step=step)
        log_list(optimizer.variables(), step=step, prefix=opre)
