"""
summary_utils.py

Collection of helper methods for creating various summary objects for logging
data in TensorBoard.

Author: Sam Foreman (github: @saforem2)
Date: 08/16/2019
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lattice.lattice import u1_plaq_exact_tf
from utils.tf_logging import variable_summaries, grad_norm_summary


def _create_training_summaries(model):
    """Create summary objects for training operations in TensorBoard."""
    with tf.name_scope('loss'):
        tf.summary.scalar('loss', model.loss_op)

    with tf.name_scope('learning_rate'):
        tf.summary.scalar('learning_rate', model.lr)

    with tf.name_scope('step_size'):
        tf.summary.scalar('step_size', model.dynamics.eps)


def _create_grad_norm_summaries(grad, var):
    """Create grad_norm summaries."""
    if 'XNet' in var.name:
        net_str = 'XNet/'
    elif 'VNet' in var.name:
        net_str = 'VNet/'
    else:
        net_str = ''
    with tf.name_scope(net_str):
        if 'scale' in var.name:
            grad_norm_summary(net_str + 'scale', grad)
            tf.summary.histogram(net_str + 'scale', grad)
        if 'transf' in var.name:
            grad_norm_summary(net_str + 'transformation', grad)
            tf.summary.histogram(net_str + 'transformation', grad)
        if 'transl' in var.name:
            grad_norm_summary(net_str + 'translation', grad)
            tf.summary.histogram(net_str + 'translation', grad)


def _create_pair_summaries(grad, var):
    """Create summary objects for a gradient, variable pair."""
    try:
        _name = var.name.split('/')[-2:]
        if len(_name) > 1:
            name = _name[0] + '/' + _name[1][:-2]
        else:
            name = var.name[:-2]
    except (AttributeError, IndexError):
        name = var.name[:-2]

    with tf.name_scope(name):
        var_name = var.name.replace(':', '')
        variable_summaries(var, name=var_name)

    grad_name = name + '/gradient'
    with tf.name_scope(grad_name):
        variable_summaries(grad, name=grad_name)


def _create_obs_summaries(model):
    """Create summary objects for physical observables."""
    with tf.name_scope('avg_charge_diffs'):
        tf.summary.scalar('avg_charge_diffs',
                          tf.reduce_mean(model.charge_diffs_op))

    with tf.name_scope('avg_plaq'):
        tf.summary.scalar('avg_plaq', model.avg_plaqs_op)

    with tf.name_scope('avg_plaq_diff'):
        tf.summary.scalar('avg_plaq_diff',
                          (u1_plaq_exact_tf(model.beta)
                           - model.avg_plaqs_op))


def _create_md_summaries(model):
    """Create summary objects for each of the MD functions and outputs."""
    for k1, v1 in model.l2hmc_fns['out_fns_f'].items():
        for k2, v2 in v1.items():
            with tf.name_scope(f'{k1}_fn_{k2}_f'):
                variable_summaries(v2)

    for k1, v1 in model.l2hmc_fns['out_fns_b'].items():
        for k2, v2 in v1.items():
            with tf.name_scope(f'{k1}_fn_{k2}_b'):
                variable_summaries(v2)

    with tf.name_scope('lf_out_f'):
        variable_summaries(model.lf_out_f)

    with tf.name_scope('lf_out_b'):
        variable_summaries(model.lf_out_b)


def create_summaries(model, summary_dir, training=True):
    """Create summary objects for logging in TensorBoard."""
    summary_writer = tf.contrib.summary.create_file_writer(summary_dir)

    grads_and_vars = zip(model.grads,
                         model.dynamics.trainable_variables)

    if training:
        _create_training_summaries(model)

    _create_obs_summaries(model)

    for grad, var in grads_and_vars:
        _create_pair_summaries(grad, var)

        if 'kernel' in var.name:
            _create_grad_norm_summaries(grad, var)

    summary_op = tf.summary.merge_all(name='train_summary_op')

    return summary_writer, summary_op
