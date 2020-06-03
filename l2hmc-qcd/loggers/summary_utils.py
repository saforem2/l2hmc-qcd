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
import utils.file_io as io

# pylint: disable=no-member
# pylint: disable=invalid-name


def grad_norm_summary(name_scope, grad):
    """Create scalar summaries of RMS values of gradients."""
    with tf.name_scope(name_scope + '_gradients'):
        grad_norm = tf.sqrt(tf.reduce_mean(grad ** 2))
        summary_name = name_scope + '_grad_norm'
        tf.summary.scalar(summary_name, grad_norm)


def check_var_and_op(name, var):
    """Check if `name` in `var.name` or `var.op.name`."""
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
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(mean_name, mean)
    tf.summary.scalar(stddev_name, stddev)
    tf.summary.scalar(max_name, tf.reduce_max(var))
    tf.summary.scalar(min_name, tf.reduce_min(var))
    tf.summary.histogram(hist_name, var)


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
    for loss in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of
        # the loss as the original loss name.
        tf.summary.scalar(loss.op.name, loss)
        tf.summary.scalar(loss.op.name + 'moving_avg',
                          loss_averages.average(loss))

    return loss_averages_op


def make_summaries_from_collection(collection, names):
    """Make summaries from `tf.collection` of variables."""
    try:
        for op, name in zip(tf.get_collection(collection), names):
            variable_summaries(op, name)
    except AttributeError:
        pass


def _create_training_summaries(model):
    """Create summary objects for training operations in TensorBoard."""
    skip_keys = ['train_op', 'loss_op']
    for key, val in model.train_ops.items():
        if key in skip_keys:
            continue
        with tf.name_scope(key):
            variable_summaries(val, key)


def _network_summary(net):
    for key, val in net.layers_dict.items():
        if key == 'hidden_layers':
            for idx, layer in enumerate(val):
                w, b = layer.weights
                variable_summaries(w, f'{key}{idx}/weights')
                variable_summaries(b, f'{key}{idx}/biases')
        elif key in ['scale_layer', 'transformation_layer']:
            w, b = val.layer.weights
            variable_summaries(w, f'{key}/weights')
            variable_summaries(b, f'{key}/biases')
            variable_summaries(val.coeff, f'{key}/coeff')
        else:
            w, b = val.weights
            variable_summaries(w, f'{key}/weights')
            variable_summaries(b, f'{key}/biases')


def network_summaries(model):
    """Create summary objects of all network weights/biases."""
    if not model.hmc:
        names = ('XNet', 'VNet')
        nets = (model.dynamics.xnet, model.dynamics.vnet)

        for name, net in zip(names, nets):
            with tf.name_scope(name):
                _network_summary(net)


def _create_energy_summaries(model):
    """Create summary objects for initial and proposed KE and PE."""
    for key, val in model.energy_ops.items():
        with tf.name_scope(key):
            for k, v in val.items():
                variable_summaries(v, k)


def _create_grad_norm_summaries(grad, var):
    """Create grad_norm summaries."""
    with tf.name_scope('grad_norm'):
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
    with tf.name_scope('avg_plaq_diff'):
        tf.summary.scalar('avg_plaq_diff',
                          (u1_plaq_exact_tf(model.beta) - model.avg_plaqs))

    with tf.name_scope('top_charge'):
        tf.summary.histogram('top_charge', model.charges)

    with tf.name_scope('charge_traces'):
        for idx, charge in enumerate(tf.transpose(model.charges[:5])):
            tf.summary.scalar(f'q_chain{idx}', charge)


def _create_l2hmc_summaries(model):
    """Create summary objects for each of the MD functions and outputs."""
    with tf.name_scope('l2hmc_forward'):
        for k1, v1 in model.l2hmc_fns['l2hmc_fns_f'].items():
            with tf.name_scope(f'{k1}_fn'):
                for k2, v2 in v1.items():
                    with tf.name_scope(f'{k2}'):
                        variable_summaries(v2)

    with tf.name_scope('l2hmc_backward'):
        for k1, v1 in model.l2hmc_fns['l2hmc_fns_b'].items():
            with tf.name_scope(f'{k1}_fn'):
                for k2, v2 in v1.items():
                    with tf.name_scope(f'{k2}'):
                        variable_summaries(v2)

    with tf.name_scope('lf_out'):
        with tf.name_scope('forward'):
            variable_summaries(model.lf_out_f)
        with tf.name_scope('backward'):
            variable_summaries(model.lf_out_b)

    with tf.name_scope('sumlogdet'):
        with tf.name_scope('forward'):
            variable_summaries(model.sumlogdet_f)
        with tf.name_scope('backward'):
            variable_summaries(model.sumlogdet_b)

    with tf.name_scope('logdets'):
        with tf.name_scope('forward'):
            variable_summaries(model.logdets_f)
        with tf.name_scope('backward'):
            variable_summaries(model.logdets_b)


def _loss_summaries(model):
    with tf.name_scope('losses'):
        for key, val in model.losses_dict.items():
            tf.summary.scalar(key, val)


def create_summaries(model, summary_dir, training=True):
    """Create summary objects for logging in TensorBoard."""
    summary_writer = tf.contrib.summary.create_file_writer(summary_dir)

    if training:
        name = 'train_summary_op'
        _create_training_summaries(model)  # loss, lr, eps, accept prob
    else:
        name = 'inference_summary_op'

    try:
        _create_obs_summaries(model)    # lattice observables
    except AttributeError:
        pass

    network_summaries(model)
    try:
        _loss_summaries(model)
        add_loss_summaries(model.loss_op)
    except (KeyError, AttributeError):
        io.log('Unable to create loss summaries.')

    for grad, var in model.grads_and_vars:
        _create_pair_summaries(grad, var)

        if 'kernel' in var.name:
            _create_grad_norm_summaries(grad, var)

    summary_op = tf.summary.merge_all(name=name)

    return summary_writer, summary_op
