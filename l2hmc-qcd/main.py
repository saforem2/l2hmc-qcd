"""
gauge_model_main.py

Main method implementing the L2HMC algorithm for a 2D U(1) lattice gauge theory
with periodic boundary conditions.

Following an object oriented approach, there are separate classes responsible
for each major part of the algorithm:

    (1.) Creating the loss function to be minimized during training and
    building the corresponding TensorFlow graph.

        - This is done using the `GaugeModel` class, found in
        `models/gauge_model.py`.

        - The `GaugeModel` class depends on the `Dynamics` class
        (found in `dynamics/gauge_dynamics.py`) that performs the augmented
        leapfrog steps outlined in the original paper.

    (2.) Training the model by minimizing the loss function over both the
    target and initialization distributions.
        - This is done using the `GaugeModelTrainer` class, found in
        `trainers/gauge_model_trainer.py`.

    (3.) Running the trained sampler to generate statistics for lattice
    observables.
        - This is done using the `GaugeModelRunner` class, found in
        `runners/gauge_model_runner.py`.

Author: Sam Foreman (github: @saforem2)
Date: 04/10/2019
"""
from __future__ import absolute_import, division, print_function

import os
import time
import pickle

from collections import namedtuple

import config as cfg
from seed_dict import seeds, xnet_seeds, vnet_seeds
#  from config import (GLOBAL_SEED, HAS_COMET, HAS_HOROVOD, HAS_MATPLOTLIB,
#                      NP_FLOAT)
from models.gauge_model import GaugeModel
from loggers.train_logger import TrainLogger
from trainers.trainer import Trainer
#  from trainers.gauge_model_trainer import GaugeModelTrainer

import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug  # noqa: F401
from tensorflow.python.client import timeline  # noqa: F401
from tensorflow.core.protobuf import rewriter_config_pb2

import inference
import utils.file_io as io

from utils.parse_args import parse_args

if cfg.HAS_COMET:
    from comet_ml import Experiment

if cfg.HAS_HOROVOD:
    import horovod.tensorflow as hvd

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)

SEP_STR = 80 * '-'  # + '\n'

NP_FLOAT = cfg.NP_FLOAT


Weights = namedtuple('Weights', ['w', 'b'])
#  NetWeights = cfg.NetWeights

#  NetWeights = namedtuple('NetWeights', [
#      'x_scale', 'x_translation', 'x_transformation',
#      'v_scale', 'v_translation', 'v_transformation']
#  )


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
        'xnet': _get_net_weights(model.dynamics.x_fn, {}),
        'vnet': _get_net_weights(model.dynamics.v_fn, {}),
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
        #  var_shape_str = f'  var.shape: {shape}'
        writer(f'  var.shape: {shape}', out_file)
        writer(f'  len(var.shape): {len(shape)}', out_file)
        var_params = 1  # variable parameters
        for dim in shape:
            writer(f'    dim: {dim}', out_file)
            #  dim_strs += f'    dim: {dim}\'
            var_params *= dim.value
        writer(f'variable_parameters: {var_params}', out_file)
        writer(80 * '-', out_file)
        total_params += var_params

    writer(80 * '-', out_file)
    writer(f'Total parameters: {total_params}', out_file)


def train_setup(FLAGS, log_file=None, root_dir=None,
                run_str=True, model_type='GaugeModel'):
    io.log(SEP_STR)
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

    #  if FLAGS.gpu:
    #      params['data_format'] = 'channels_last'
    #      #  params['data_format'] = 'channels_first'
    #  else:
    #      io.log("Using CPU for training.")
    #      params['data_format'] = 'channels_last'

    #  if getattr(FLAGS, 'float64', False):
    #  if params.get('float64', False):
    #      io.log(f'INFO: Setting floating point precision to `float64`.')
    #      set_precision('float64')

    #  if getattr(FLAGS, 'horovod', False):
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


def check_reversibility(model, sess):
    rand_samples = np.random.randn(*model.x.shape)
    net_weights = cfg.NetWeights(1., 1., 1., 1., 1., 1.)
    feed_dict = {
        model.x: rand_samples,
        model.beta: 1.,
        model.net_weights: net_weights,
        #  model.net_weights.x_scale: 1.,
        #  model.net_weights.x_translation: 1.,
        #  model.net_weights.x_transformation: 1.,
        #  model.net_weights.v_scale: 1.,
        #  model.net_weights.v_translation: 1.,
        #  model.net_weights.v_transformation: 1.,
        #  model.net_weights[0]: 1.,
        #  model.net_weights[1]: 1.,
        #  model.net_weights[2]: 1.,
        model.train_phase: False
    }

    # Check reversibility
    x_diff, v_diff = sess.run([model.x_diff,
                               model.v_diff], feed_dict=feed_dict)
    reverse_str = (f'Reversibility results:\n '
                   f'\t x_diff: {x_diff:.10g}, v_diff: {v_diff:.10g}')

    return reverse_str, x_diff, v_diff


def train_l2hmc(FLAGS, log_file=None):
    """Create, train, and run L2HMC sampler on 2D U(1) gauge model."""
    tf.keras.backend.set_learning_phase(True)
    params, hooks = train_setup(FLAGS, log_file)

    # ---------------------------------------------------------------
    # NOTE: Conditionals required for file I/O if we're not using
    #       Horovod, `is_chief` should always be True otherwise,
    #       if using Horovod, we only want to perform file I/O
    #       on hvd.rank() == 0, so check that first.
    # ---------------------------------------------------------------
    condition1 = not params['using_hvd']
    condition2 = params['using_hvd'] and hvd.rank() == 0
    is_chief = condition1 or condition2

    if is_chief:
        log_dir = params['log_dir']
        checkpoint_dir = os.path.join(log_dir, 'checkpoints/')
        io.check_else_make_dir(checkpoint_dir)

    else:
        log_dir = None
        checkpoint_dir = None

    io.log(SEP_STR)
    io.log('L2HMC PARAMETERS:')
    for key, val in params.items():
        io.log(f'  {key}: {val}')
    io.log(SEP_STR)

    # --------------------------------------------------------
    # Create model and train_logger
    # --------------------------------------------------------
    model = GaugeModel(params)
    weights = get_net_weights(model)
    xnet = model.dynamics.x_fn.generic_net
    vnet = model.dynamics.v_fn.generic_net
    coeffs = {
        'xnet': {
            'coeff_scale': xnet.coeff_scale,
            'coeff_transformation': xnet.coeff_transformation,
        },
        'vnet': {
            'coeff_scale': vnet.coeff_scale,
            'coeff_transformation': vnet.coeff_transformation,
        },
    }
    if is_chief:
        logging_steps = params.get('logging_steps', 10)
        train_logger = TrainLogger(model, log_dir,
                                   logging_steps=logging_steps,
                                   summaries=params['summaries'])
    else:
        train_logger = None

    # -------------------------------------------------------
    # Setup config and init_feed_dict for tf.train.Scaffold
    # -------------------------------------------------------
    config, params = create_config(params)

    # set initial value of charge weight using value from FLAGS
    #  charge_weight_init = params['charge_weight']
    #  net_weights_init = [1., 1., 1.]
    net_weights_init = cfg.NetWeights(x_scale=1.,
                                      x_translation=1.,
                                      x_transformation=1.,
                                      v_scale=1.,
                                      v_translation=1.,
                                      v_transformation=1.)
    samples_init = np.reshape(np.array(model.lattice.samples,
                                       dtype=NP_FLOAT),
                              (model.batch_size, model.x_dim))
    beta_init = model.beta_init

    # ----------------------------------------------------------------
    #  Create MonitoredTrainingSession
    #
    #  NOTE: The MonitoredTrainingSession takes care of session
    #        initialization, restoring from a checkpoint, saving to a
    #        checkpoint, and closing when done or an error occurs.
    # ----------------------------------------------------------------
    sess_kwargs = {
        'checkpoint_dir': checkpoint_dir,
        #  'scaffold': scaffold,
        'hooks': hooks,
        'config': config,
        'save_summaries_secs': None,
        'save_summaries_steps': None
    }

    global_var_init = tf.global_variables_initializer()
    local_var_init = tf.local_variables_initializer()
    uninited = tf.report_uninitialized_variables()
    #  global_vars = tf.global_variables()
    #  is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    sess = tf.train.MonitoredTrainingSession(**sess_kwargs)
    tf.keras.backend.set_session(sess)
    sess.run([global_var_init, local_var_init])
    uninited_out = sess.run(uninited)
    io.log(f'tf.report_uninitialized_variables() len = {uninited_out}')

    # Check reversibility and write results out to `.txt` file.
    reverse_str, x_diff, v_diff = check_reversibility(model, sess)
    reverse_file = os.path.join(model.log_dir, 'reversibility_test.txt')
    io.log_and_write(reverse_str, reverse_file)

    io.save_dict(seeds, out_dir=model.log_dir, name='seeds')
    io.save_dict(xnet_seeds, out_dir=model.log_dir, name='xnet_seeds')
    io.save_dict(vnet_seeds, out_dir=model.log_dir, name='vnet_seeds')

    # ----------------------------------------------------------
    #                       TRAINING
    # ----------------------------------------------------------
    trainer = Trainer(sess, model, train_logger, **params)

    train_kwargs = {
        'samples': samples_init,
        'beta': beta_init,
        'net_weights': net_weights_init
    }

    t0 = time.time()
    trainer.train(model.train_steps, **train_kwargs)

    reverse_str, x_diff, v_diff = check_reversibility(model, sess)
    io.log_and_write(reverse_str, reverse_file)

    io.log(SEP_STR)
    io.log(f'Training completed in: {time.time() - t0:.3g}s')
    io.log(SEP_STR)

    weights = get_net_weights(model)
    xcoeffs = sess.run(list(coeffs['xnet'].values()))
    vcoeffs = sess.run(list(coeffs['vnet'].values()))
    weights['xnet']['GenericNet'].update({
        'coeff_scale': xcoeffs[0],
        'coeff_transformation': xcoeffs[1]
    })
    weights['vnet']['GenericNet'].update({
        'coeff_scale': vcoeffs[0],
        'coeff_transformation': vcoeffs[1]
    })

    weights_file = os.path.join(model.log_dir, 'weights.pkl')
    with open(weights_file, 'wb') as f:
        pickle.dump(weights, f)

    params_file = os.path.join(os.getcwd(), 'params.pkl')
    with open(params_file, 'wb') as f:
        pickle.dump(model.params, f)

    # Count all trainable paramters and write them out (w/ shapes) to txt file
    count_trainable_params(os.path.join(params['log_dir'],
                                        'trainable_params.txt'))

    # close MonitoredTrainingSession and reset the default graph
    sess.close()
    tf.reset_default_graph()

    return model, train_logger


def main(FLAGS):
    """Main method for creating/training/running L2HMC for U(1) gauge model."""
    log_file = 'output_dirs.txt'

    #  if getattr(FLAGS, 'float64', False):
    USING_HVD = getattr(FLAGS, 'horovod', False)
    if cfg.HAS_HOROVOD and USING_HVD:
        io.log("INFO: USING HOROVOD")
        hvd.init()
        rank = hvd.rank()
        print(f'Setting seed from rank: {rank}')
        tf.set_random_seed(rank * seeds['global_tf'])

    if FLAGS.hmc:   # run generic HMC sampler
        inference.run_hmc(FLAGS, log_file=log_file)
    else:           # train l2hmc sampler
        model, train_logger = train_l2hmc(FLAGS, log_file)


if __name__ == '__main__':
    FLAGS = parse_args()
    using_hvd = getattr(FLAGS, 'horovod', False)
    if not using_hvd:
        tf.set_random_seed(seeds['global_tf'])

    t0 = time.time()
    main(FLAGS)
    io.log('\n\n' + SEP_STR)
    io.log(f'Time to complete: {time.time() - t0:.4g}')
    io.log(SEP_STR)
