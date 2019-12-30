from __future__ import absolute_import, division, print_function

import os
import numpy as np
import tensorflow as tf
import config as cfg

from tensorflow.python import debug as tf_debug  # noqa: F401
from tensorflow.python.client import timeline  # noqa: F401
from tensorflow.core.protobuf import rewriter_config_pb2

import utils.file_io as io

if cfg.HAS_HOROVOD:
    import horovod.tensorflow as hvd

Weights = cfg.Weights


def set_global_step(sess, global_step):
    """Explicitly sets the global step when restoring a training session.

    Args:
        sess (tf.Session): Session in which to set the global step.
        global_step (int): Desired global step.
    """
    graph = tf.get_default_graph()
    global_step_setter = graph.get_operation_by_name('global_step_setter')
    inputs = tf.get_collection('inputs')
    global_step_tensor = [
        i for i in tf.global_variables() if 'global_step' in i.name
    ][0]
    global_step_ph = [i for i in inputs if 'global_step' in i.name][0]
    global_step_np = sess.run(global_step_tensor)
    io.log(f'INFO: Original value of `global_step`: {global_step_np}')
    io.log(f'INFO: Setting `global_step` to: {global_step}')
    sess.run(global_step_setter, feed_dict={global_step_ph: global_step})
    global_step_np = sess.run(global_step_tensor)
    io.log(f'INFO: New value of `global_step`: {global_step_np}')

def _get_net_weights(net, weights):
    for layer in net.layers:
        if hasattr(layer, 'layers'):
            weights = _get_net_weights(layer, weights)
        else:
            try:
                weights[net.name].update({
                    layer.name: Weights(*layer.get_weights())
                })
            except KeyError:
                weights.update({
                    net.name: {
                        layer.name: Weights(*layer.get_weights())
                    }
                })

    return weights


def get_coeffs(generic_net):
    return (generic_net.coeff_scale, generic_net.coeff_transformation)


def get_net_weights(model):
    weights = {
        'xnet': _get_net_weights(model.dynamics.xnet, {}),
        'vnet': _get_net_weights(model.dynamics.vnet, {}),
    }

    return weights


def create_config(params):
    """Helper method for creating a tf.ConfigProto object."""
    config = tf.ConfigProto(allow_soft_placement=True)
    time_size = params.get('time_size', None)
    if time_size is not None and time_size > 8:
        off = rewriter_config_pb2.RewriterConfig.OFF
        config_attrs = config.graph_options.rewrite_options
        config_attrs.arithmetic_optimization = off

    gpu = params.get('gpu', False)
    if gpu:
        # Horovod: pin GPU to be used to process local rank 
        # (one GPU per process)
        config.gpu_options.allow_growth = True
        #  config.allow_soft_placement = True
        if cfg.HAS_HOROVOD and params['horovod']:
            config.gpu_options.visible_device_list = str(hvd.local_rank())

    if cfg.HAS_MATPLOTLIB:
        params['_plot'] = True

    theta = params.get('theta', False)
    if theta:
        params['_plot'] = False
        io.log("Training on Theta @ ALCF...")
        params['data_format'] = 'channels_last'
        os.environ["KMP_BLOCKTIME"] = str(0)
        os.environ["KMP_AFFINITY"] = (
            "granularity=fine,verbose,compact,1,0"
        )
        # NOTE: KMP affinity taken care of by passing -cc depth to aprun call
        OMP_NUM_THREADS = 62
        config.allow_soft_placement = True
        config.intra_op_parallelism_threads = OMP_NUM_THREADS
        config.inter_op_parallelism_threads = 0

    return config, params


def create_session(config, checkpoint_dir, monitored=False):
    if monitored:
        sess_kwargs = {
            'checkpoint_dir': checkpoint_dir,
            'hooks': [],
            'config': config,
            'save_summaries_secs': None,
            'save_summaries_steps': None,
        }

        return tf.train.MonitoredTrainingSession(**sess_kwargs)

    return tf.Session(config=config)


def latest_meta_file(checkpoint_dir=None):
    """Returns the most recent meta-graph (`.meta`) file in checkpoint_dir."""
    if not os.path.isdir(checkpoint_dir) or checkpoint_dir is None:
        return

    meta_files = [i for i in os.listdir(checkpoint_dir) if i.endswith('.meta')]
    step_nums = [int(i.split('-')[-1].rstrip('.meta')) for i in meta_files]
    step_num = sorted(step_nums)[-1]
    meta_file = os.path.join(checkpoint_dir, f'model.ckpt-{step_num}.meta')

    return meta_file


def count_trainable_params(out_file, log=False):
    """Count the total number of trainable parameters in a tf.Graph object.

    Args:
        out_file (str): Path to file where all trainable parameters will be
            written.
        log (bool): Whether or not to print trainable parameters to console
            (std-out).
    Returns:
        None
    """
    if log:
        writer = io.log_and_write
    else:
        writer = io.write

    io.log(f'Writing parameter counts to: {out_file}.')
    writer(80 * '-', out_file)
    total_params = 0
    for var in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = var.get_shape()
        writer(f'var: {var}', out_file)
        writer(f'  var.shape: {shape}', out_file)
        writer(f'  len(var.shape): {len(shape)}', out_file)
        var_params = 1  # variable parameters
        for dim in shape:
            writer(f'    dim: {dim}', out_file)
            var_params *= dim.value
        writer(f'variable_parameters: {var_params}', out_file)
        writer(80 * '-', out_file)
        total_params += var_params

    writer(80 * '-', out_file)
    writer(f'Total parameters: {total_params}', out_file)


def train_setup(FLAGS, log_file=None, root_dir=None,
                run_str=True, model_type='GaugeModel'):
    io.log(80 * '-')
    io.log("Starting training using L2HMC algorithm...")
    tf.keras.backend.clear_session()
    tf.reset_default_graph()

    # ---------------------------------
    # Parse command line arguments;
    # copy key, val pairs from FLAGS
    # to params.
    # ---------------------------------
    try:
        FLAGS_DICT = FLAGS.__dict__
    except AttributeError:
        FLAGS_DICT = FLAGS

    params = {k: v for k, v in FLAGS_DICT.items()}

    params['log_dir'] = io.create_log_dir(FLAGS,
                                          log_file=log_file,
                                          root_dir=root_dir,
                                          run_str=run_str,
                                          model_type=model_type)
    params['summaries'] = not getattr(FLAGS, 'no_summaries', False)
    save_steps = getattr(FLAGS, 'save_steps', None)
    train_steps = getattr(FLAGS, 'train_steps', None)

    if 'no_summaries' in params:
        del params['no_summaries']

    if save_steps is None and train_steps is not None:
        params['save_steps'] = params['train_steps'] // 4

    else:
        params['save_steps'] = 1000

    if params.get('horovod', False):
        params['using_hvd'] = True
        num_workers = hvd.size()
        params['num_workers'] = num_workers

        # ---------------------------------------------------------
        # Horovod: Scale initial lr by of num GPUs.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # NOTE: Even with a linear `warmup` of the learning rate,
        #       the training remains unstable as evidenced by
        #       exploding gradients and NaN tensors.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #  params['lr_init'] *= num_workers
        # ---------------------------------------------------------

        # Horovod: adjust number of training steps based on number of GPUs.
        #  params['train_steps'] //= num_workers

        # Horovod: adjust save_steps and lr_decay_steps accordingly.
        #  params['save_steps'] //= num_workers
        #  params['lr_decay_steps'] //= num_workers

        #  if params['summaries']:
        #      params['logging_steps'] //= num_workers

        # ---------------------------------------------------------
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial
        # variable states from rank 0 to all other processes. This
        # is necessary to ensure consistent initialization of all
        # workers when training is started with random weights or
        # restored from a checkpoint.
        # ---------------------------------------------------------
        hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    else:
        params['using_hvd'] = False
        hooks = []

    return params, hooks


def check_reversibility(model, sess, net_weights=None, out_file=None):
    rand_samples = np.random.randn(*model.x.shape)
    if net_weights is None:
        net_weights = cfg.NetWeights(1., 1., 1., 1., 1., 1.)

    io.log(f'Net weights used in reversibility test:\n\t {net_weights}.\n')

    feed_dict = {
        model.x: rand_samples,
        model.beta: 1.,
        model.net_weights: net_weights,
        model.train_phase: False
    }

    # Check reversibility
    x_diff, v_diff = sess.run([model.x_diff,
                               model.v_diff], feed_dict=feed_dict)
    reverse_str = (f'Reversibility results:\n '
                   f'\t x_diff: {x_diff:.10g}, v_diff: {v_diff:.10g}')

    if out_file is not None:
        io.log_and_write(reverse_str, out_file)

    return reverse_str, x_diff, v_diff
