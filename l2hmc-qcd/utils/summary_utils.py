"""
summary_utils.py

Implements helper methods for creating TensorBoard summaries.

Author: Sam Foreman (github: @saforem2)
Date: 07/29/2020
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf


def summarize_dict(d, step, prefix=None):
    """Create summaries for all items in `d`."""
    if prefix is None:
        prefix = ''
    for key, val in d.items():
        name = f'{prefix}/{key}'
        tf.summary.histogram(name, val, step=step)
        tf.summary.scalar(f'{name}_avg', tf.reduce_mean(val), step=step)


def summarize_list(x, step, prefix=None):
    """Create summary objects for all items in `x`."""
    if prefix is None:
        prefix = ''
    for t in x:
        name = f'{prefix}/{t.name}'
        tf.summary.histogram(name, t, step)
        tf.summary.scalar(f'{name}_avg', tf.reduce_mean(t), step=step)


def update_summaries(step, metrics, dynamics):
    """Create summary objects.

    NOTE: Explicitly, we create summary objects for all entries in
      - metrics
      - dynamics.variables
      - dynamics.optimizer.variables()

    Returns:
        None
    """
    #  learning_rate = dynamics.lr(tf.constant(step))
    opt_cfg = dynamics.optimizer.get_config()
    learning_rate = opt_cfg['learning_rate']
    opt_vars = dynamics.optimizer.variables()
    summarize_dict(metrics, step, prefix='training')
    summarize_list(dynamics.variables, step, prefix='dynamics')
    summarize_list(opt_vars, step, prefix='dynamics.optimizer')
    tf.summary.scalar('training/learning_rate', learning_rate, step)
