"""
l2hmc/trackers/tensorflow/trackers.py

Contains various utilities for tracking / logging metrics in TensorBoard
"""
from __future__ import absolute_import, print_function, division, annotations

import tensorflow as tf
from tensorflow.python.keras import backend as K


Model = tf.keras.models.Model
Optimizer = tf.keras.optimizers.Optimizer


def log_dict(d: dict, step: int, prefix: str = None):
    """Create tensorboard summaries for all items in `d`"""
    prefix = '' if prefix is None else prefix
    for key, val in d.items():
        if isinstance(val, dict):
            summarize_dict(val, step, prefix=f'{prefix}_{key}')
        else:
            name = '/'.join([prefix, str(key)])
            tf.summary.histogram(name, val, step=step)
            tf.summary.scalar(f'{name}_avg', tf.reduce_mean(val), step=step)


def summarize_list(x, step, prefix=None):
    for t in x:
        name = f'{prefix}/{t.name}'
        sname = f'{prefix}/{t.name}_avg'
        tf.summary.histogram(name=name, data=t, step=step)
        tf.summary.scalar(name=sname, data=tf.reduce_mean(t), step=step)


def update_summaries(
        step: int,
        metrics: dict,
        model: Model = None,
        optimizer: Optimizer = None,
        prefix: str = None,
):
    """"Create summary objects."""
    summarize_dict(metrics, step, prefix='/'.join(*[prefix, 'metrics']))
    if model is not None:
        model_prefix = '/'.join(*[prefix, 'model'])
        summarize_list(model.variables, step=step, prefix=model_prefix)

    if optimizer is not None:
        opt_prefix = '/'.join(*[prefix, 'optimizer'])
        lr_name = '/'.join(*[opt_prefix, 'learning_rate'])
        tf.summary.scalar(lr_name, K.get_value(optimizer.lr), step=step)
        summarize_list(optimizer.variables(), step=step, prefix=opt_prefix)
