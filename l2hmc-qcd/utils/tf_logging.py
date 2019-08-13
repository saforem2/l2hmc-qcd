from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf

TOWER_NAME = 'tower'


def get_run_num(log_dir):
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    contents = os.listdir(log_dir)
    if contents == []:
        return 1
    else:
        run_nums = []
        for item in contents:
            try:
                run_nums.append(int(item.split('_')[-1]))
                #  run_nums.append(int(''.join(x for x in item if x.isdigit())))
            except ValueError:
                continue
        return sorted(run_nums)[-1] + 1
    #  if contents == ['.DS_Store']:
    #      return 1
    #  else:
    #      for item in contents:
    #          if os.path.isdir(log_dir + item):
    #              run_dirs.append(item)
    #      run_nums = [int(str(i)[3:]) for i in run_dirs]
    #      prev_run_num = max(run_nums)
    #      return prev_run_num + 1


def make_run_dir(log_dir):
    if log_dir.endswith('/'):
        _dir = log_dir
    else:
        _dir = log_dir + '/'
    run_num = get_run_num(_dir)
    run_dir = _dir + f'run_{run_num}/'
    if os.path.isdir(run_dir):
        raise f'Directory: {run_dir} already exists, exiting!'
    else:
        print(f'Creating directory for new run: {run_dir}')
        os.makedirs(run_dir)
    return run_dir


def check_log_dir(log_dir):
    if not os.path.isdir(log_dir):
        raise ValueError(f'Unable to locate {log_dir}, exiting.')
    else:
        if not log_dir.endswith('/'):
            log_dir += '/'
        info_dir = log_dir + 'run_info/'
        figs_dir = log_dir + 'figures/'
        if not os.path.isdir(info_dir):
            os.makedirs(info_dir)
        if not os.path.isdir(figs_dir):
            os.makedirs(figs_dir)
    return log_dir, info_dir, figs_dir


#  def create_log_dir():
#      """Create directory for storing information about experiment."""
#      #  root_log_dir = '../../log_mog_tf/'
#      #  root_log_dir = os.path.join(ROOT_DIR, log_mog_tf)
#      root_log_dir = os.path.join(os.path.split(ROOT_DIR)[0], 'log_mog_tf')
#      log_dir = make_run_dir(root_log_dir)
#      info_dir = log_dir + 'run_info/'
#      figs_dir = log_dir + 'figures/'
#      if not os.path.isdir(info_dir):
#          os.makedirs(info_dir)
#      if not os.path.isdir(figs_dir):
#          os.makedirs(figs_dir)
#      return log_dir, info_dir, figs_dir

def grad_norm_summary(name_scope, grad):
    with tf.name_scope(name_scope + '_gradients'):
        grad_norm = tf.sqrt(tf.reduce_mean(grad ** 2))
        summary_name = name_scope + '_grad_norm'
        tf.summary.scalar(summary_name, grad_norm)


def check_var_and_op(name, var):
    return (name in var.name or name in var.op.name)


def variable_summaries(var, name=''):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)"""
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


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


def activation_summary(x):
    """Helper to create summaries for activations.
    
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparisty of activations.

    Args:
        x: Tensor
    Returns:
        None
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation in tensorboard.
    #  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    try:
        tensor_name = x.op.name
    except AttributeError:
        tensor_name = x.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
