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


def grad_norm_summary(name_scope, grad):
    with tf.name_scope(name_scope + '_gradients'):
        grad_norm = tf.sqrt(tf.reduce_mean(grad ** 2))
        summary_name = name_scope + '_grad_norm'
        tf.summary.scalar(summary_name, grad_norm)


def check_var_and_op(name, var):
    return (name in var.name or name in var.op.name)


def variable_summaries(var, name=''):
    """Attach summaries to a Tensor (for TensorBoard visualization)."""
    mean_name = 'mean'
    stddev_name = 'stddev'
    max_name = 'max'
    min_name = 'min'
    hist_name = 'histogram'
    if name != '':
        mean_name = name + '/' + mean_name
        stddev_name = name + '/' + stddev_name
        max_name = name + '/' + max_name
        min_name = name + '/' + min_name
        hist_name = name + '/' + hist_name

    mean = tf.reduce_mean(var)
    tf.summary.scalar(mean_name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(stddev_name, stddev)
    tf.summary.scalar(max_name, tf.reduce_max(var))
    tf.summary.scalar(min_name, tf.reduce_min(var))
    tf.summary.histogram(hist_name, var)
    # activation summaries
    #  tf.summary.histogram(tensor_name + '/activations', x)
    #  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def add_loss_summaries(total_loss):
    """Add summaries for losses in GaugeModel.

    Generates a moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from model._calc_loss()

    Returns:
        loss_averages_op: Op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total
    # loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of
        # the loss as the original loss name.
        tf.summary.scalar(l.op.name, l)
        tf.summary.scalar(l.op.name + 'moving_avg', loss_averages.average(l))

    return loss_averages_op


def make_summaries_from_collection(collection, names):
    try:
        for op, name in zip(tf.get_collection(collection), names):
            variable_summaries(op, name)
    except AttributeError:
        pass


def _create_loss_summaries(total_loss):
    """Add summaries for losses in GaugeModel.

    Generates a moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss operation.

    Returns:
        loss_averages_op: Operation for generating moving averages of losses.
    """
    # Compute the moving avg. of all individ. losses and total loss
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    std_names = ['std_loss', 'x_std_loss', 'z_std_loss']
    std_losses = losses[:3]
    for name, loss in zip(std_names, std_losses):
        with tf.name_scope(name):
            tf.summary.scalar(name + '/raw', loss)
            tf.summary.scalar(name + '/moving_avg',
                              loss_averages.average(loss))

    charge_names = ['charge_loss', 'xq_loss', 'zq_loss']
    charge_losses = losses[3:]
    for name, loss in zip(charge_names, charge_losses):
        with tf.name_scope(name):
            tf.summary.scalar(name + '/raw', loss)
            tf.summary.scalar(name + '/moving_avg',
                              loss_averages.average(loss))

    with tf.name_scope('total_loss'):
        tf.summary.scalar('total_loss/raw', total_loss)
        tf.summary.scalar('total_loss/moving_avg',
                          loss_averages.average(total_loss))

    return loss_averages_op


def _create_training_summaries(model):
    """Create summary objects for training operations in TensorBoard."""
    with tf.name_scope('loss'):
        tf.summary.scalar('loss', model.loss_op)

    with tf.name_scope('learning_rate'):
        tf.summary.scalar('learning_rate', model.lr)

    with tf.name_scope('step_size'):
        tf.summary.scalar('step_size', model.dynamics.eps)

    with tf.name_scope('px'):
        tf.summary.scalar('px', tf.reduce_mean(model.px))

    #  with tf.name_scope('kinetic_energy'):
    #      tf.summary.scalar('kinetic_energy', model.ke_proposed)

    with tf.name_scope('x_out'):
        variable_summaries(model.x_out, 'x_out')


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


def _create_l2hmc_summaries(model):
    """Create summary objects for each of the MD functions and outputs."""
    for k1, v1 in model.l2hmc_fns['l2hmc_fns_f'].items():
        for k2, v2 in v1.items():
            with tf.name_scope(f'{k1}_fn_{k2}_f'):
                variable_summaries(v2)

    for k1, v1 in model.l2hmc_fns['l2hmc_fns_b'].items():
        for k2, v2 in v1.items():
            with tf.name_scope(f'{k1}_fn_{k2}_b'):
                variable_summaries(v2)

    with tf.name_scope('lf_out_f'):
        variable_summaries(model.lf_out_f)

    with tf.name_scope('lf_out_b'):
        variable_summaries(model.lf_out_b)

    with tf.name_scope('sumlogdet_f'):
        variable_summaries(model.sumlogdet_f)

    with tf.name_scope('sumlogdet_b'):
        variable_summaries(model.sumlogdet_b)

    with tf.name_scope('logdets_f'):
        variable_summaries(model.logdets_f)

    with tf.name_scope('logdets_b'):
        variable_summaries(model.logdets_b)


def create_summaries(model, summary_dir, training=True):
    """Create summary objects for logging in TensorBoard."""
    summary_writer = tf.contrib.summary.create_file_writer(summary_dir)

    grads_and_vars = zip(model.grads,
                         model.dynamics.trainable_variables)

    if training:
        _create_training_summaries(model)  # loss, lr, eps, accept prob 

    if model._model_type == 'GaugeModel':
        _create_obs_summaries(model)    # lattice observables

    # log S, T, Q functions (forward/backward)
    _create_l2hmc_summaries(model)

    #  _ = _create_loss_summaries(model.loss_op)

    for grad, var in grads_and_vars:
        _create_pair_summaries(grad, var)

        if 'kernel' in var.name:
            _create_grad_norm_summaries(grad, var)

    summary_op = tf.summary.merge_all(name='train_summary_op')

    return summary_writer, summary_op
