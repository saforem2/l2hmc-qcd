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

        - The `GaugeModel` class depends on the `GaugeDynamics` class
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
import os
import random
import time
import tensorflow as tf
import numpy as np

from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorflow.core.protobuf import rewriter_config_pb2

import utils.file_io as io

from globals import GLOBAL_SEED, NP_FLOAT
from utils.parse_args import parse_args
from utils.model_loader import load_model
from models.gauge_model import GaugeModel
from loggers.train_logger import TrainLogger
from loggers.run_logger import RunLogger
from trainers.gauge_model_trainer import GaugeModelTrainer
from plotters.gauge_model_plotter import GaugeModelPlotter
from plotters.leapfrog_plotters import LeapfrogPlotter
from runners.gauge_model_runner import GaugeModelRunner

try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)

# -------------------------------------------------------------------------
# Set random seeds for tensorflow and numpy
# -------------------------------------------------------------------------
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)        # `python` build-in pseudo-random generator
np.random.seed(GLOBAL_SEED)     # numpy pseudo-random generator
tf.set_random_seed(GLOBAL_SEED)


def create_config(FLAGS, params):
    """Create tensorflow config."""
    config = tf.ConfigProto()
    if FLAGS.time_size > 8:
        off = rewriter_config_pb2.RewriterConfig.OFF
        config_attrs = config.graph_options.rewrite_options
        config_attrs.arithmetic_optimization = off

    if FLAGS.gpu:
        # Horovod: pin GPU to be used to process local rank (one GPU per
        # process)
        config.gpu_options.allow_growth = True
        #  config.allow_soft_placement = True
        if HAS_HOROVOD and FLAGS.horovod:
            num_gpus = hvd.size()
            io.log(f"Number of GPUs: {num_gpus}")
            config.gpu_options.visible_device_list = str(hvd.local_rank())

    if HAS_MATPLOTLIB:
        params['_plot'] = True

    if FLAGS.theta:
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


def count_trainable_params(out_file):
    t0 = time.time()
    io.log(f'Writing parameter counts to: {out_file}.')
    io.log_and_write(80 * '-', out_file)
    total_params = 0
    for var in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = var.get_shape()
        io.log_and_write(f'var: {var}', out_file)
        #  var_shape_str = f'  var.shape: {shape}'
        io.log_and_write(f'  var.shape: {shape}', out_file)
        io.log_and_write(f'  len(var.shape): {len(shape)}', out_file)
        var_params = 1  # variable parameters
        for dim in shape:
            io.log_and_write(f'    dim: {dim}', out_file)
            #  dim_strs += f'    dim: {dim}\'
            var_params *= dim.value
        io.log_and_write(f'variable_parameters: {var_params}', out_file)
        io.log_and_write(80 * '-', out_file)
        total_params += var_params

    io.log_and_write(80 * '-', out_file)
    io.log_and_write(f'Total parameters: {total_params}', out_file)
    t1 = time.time() - t0
    io.log_and_write(f'Took: {t1} s to complete.', out_file)


def hmc(FLAGS, params=None, log_file=None):
    """Create and run generic HMC sampler using trained params from L2HMC."""
    condition1 = not FLAGS.horovod
    condition2 = FLAGS.horovod and hvd.rank() == 0
    is_chief = condition1 or condition2
    if not is_chief:
        return -1

    FLAGS.hmc = True

    FLAGS.log_dir = io.create_log_dir(FLAGS, log_file=log_file)

    if params is None:
        params = {}
        for key, val in FLAGS.__dict__.items():
            params[key] = val

    params['hmc'] = True
    params['use_bn'] = False
    params['log_dir'] = FLAGS.log_dir
    params['data_format'] = None
    params['eps_trainable'] = False

    figs_dir = os.path.join(params['log_dir'], 'figures')
    io.check_else_make_dir(figs_dir)

    io.log(80 * '-' + '\n')
    io.log('HMC PARAMETERS:')
    for key, val in params.items():
        io.log(f'  {key}: {val}')
    io.log(80 * '-' + '\n')

    # create tensorflow config (`config_proto`) to configure session
    config, params = create_config(FLAGS, params)
    tf.reset_default_graph()
    sess = tf.Session(config=config)

    model = GaugeModel(params=params)
    run_logger = RunLogger(model, params['log_dir'], save_lf_data=False)
    plotter = GaugeModelPlotter(run_logger.figs_dir)

    sess.run(tf.global_variables_initializer())

    runner = GaugeModelRunner(sess, model, run_logger)

    if FLAGS.hmc_beta is None:
        betas = [FLAGS.beta_final]
    else:
        betas = [float(FLAGS.hmc_beta)]

    for beta in betas:
        # to ensure hvd.rank() == 0
        if run_logger is not None:
            run_dir, run_str = run_logger.reset(model.run_steps, beta)

        t0 = time.time()

        runner.run(model.run_steps, beta)

        run_time = time.time() - t0
        io.log(f'Took: {run_time} s to complete run.')

        if plotter is not None and run_logger is not None:
            plotter.plot_observables(run_logger.run_data, beta, run_str)
            if FLAGS.save_lf:
                lf_plotter = LeapfrogPlotter(plotter.out_dir, run_logger)
                lf_plotter.make_plots(run_dir, num_samples=20)

    return sess, model, runner, run_logger


def l2hmc(FLAGS, log_file=None):
    """Create, train, and run L2HMC sampler on 2D U(1) gauge model."""
    tf.keras.backend.clear_session()
    tf.reset_default_graph()

    # ---------------------------------------------------------------------
    # Parse command line arguments and set parameters for correct values.
    # ---------------------------------------------------------------------
    FLAGS.log_dir = io.create_log_dir(FLAGS, log_file=log_file)
    if FLAGS.save_steps is None and FLAGS.train_steps is not None:
        FLAGS.save_steps = FLAGS.train_steps // 4

    params = {}
    for key, val in FLAGS.__dict__.items():
        params[key] = val

    if FLAGS.gpu:
        io.log("Using GPU for training.")
        params['data_format'] = 'channels_first'
    else:
        io.log("Using CPU for training.")
        params['data_format'] = 'channels_last'

    if FLAGS.horovod:
        params['using_hvd'] = True
        num_workers = hvd.size()
        params['num_workers'] = num_workers
        params['train_steps'] //= num_workers
        params['save_steps'] //= num_workers
        params['lr_decay_steps'] //= num_workers
        if params['summaries']:
            params['logging_steps'] // num_workers
        hooks = [
            # Horovod: BroadcastGlobalVariablesHook broadcasts initial
            # variable states from rank 0 to all other processes. This
            # is necessary to ensure consistent initialization of all
            # workers when training is started with random weights or
            # restored from a checkpoint.
            hvd.BroadcastGlobalVariablesHook(0),
        ]
        #  params['run_steps'] //= num_workers
        #  params['lr_init'] *= hvd.size()
    else:
        params['using_hvd'] = False
        hooks = []

    # Conditionals required for file I/O
    # if we're not using Horovod, `is_chief` should always be True
    # otheerwise, if using Horovod, we only want to perform file I/O
    # on hvd.rank() == 0, so check that first
    condition1 = not FLAGS.horovod
    condition2 = FLAGS.horovod and hvd.rank() == 0
    is_chief = condition1 or condition2

    if is_chief:
        assert FLAGS.log_dir == params['log_dir']
        log_dir = FLAGS.log_dir
        checkpoint_dir = os.path.join(log_dir, 'checkpoints/')
        io.check_else_make_dir(checkpoint_dir)

    else:
        log_dir = None
        checkpoint_dir = None

    io.log(80 * '-' + '\n')
    io.log('L2HMC PARAMETERS:')
    for key, val in params.items():
        io.log(f'  {key}: {val}')
    io.log(80 * '-' + '\n')

    # --------------------------------------------------------
    # Create model and train_logger
    # --------------------------------------------------------
    model = GaugeModel(params=params)
    if is_chief:
        train_logger = TrainLogger(model, log_dir, FLAGS.summaries)
    else:
        train_logger = None

    # --------------------------------------------------
    # Setup config and MonitoredTrainingSession
    # --------------------------------------------------
    config, params = create_config(FLAGS, params)
    tf.keras.backend.set_learning_phase(True)

    # set initial value of charge weight using value from FLAGS
    charge_weight_init = FLAGS.charge_weight
    net_weights_init = [1., 1., 1.]
    samples_init = np.reshape(np.array(model.lattice.samples, dtype=NP_FLOAT),
                              (model.num_samples, model.x_dim))
    beta_init = model.beta_init

    init_feed_dict = {
        model.x: samples_init,
        model.beta: beta_init,
        model.charge_weight: charge_weight_init,
        model.net_weights[0]: net_weights_init[0],  # scale_weight
        model.net_weights[1]: net_weights_init[1],  # transformation_weight
        model.net_weights[2]: net_weights_init[2],  # translation_weight
    }

    # ensure all variables are initialized
    target_collection = []
    if is_chief:
        collection = tf.local_variables() + target_collection
    else:
        collection = tf.local_variables()

    local_init_op = tf.variables_initializer(collection)
    ready_for_local_init_op = tf.report_uninitialized_variables(collection)

    scaffold = tf.train.Scaffold(
        init_feed_dict=init_feed_dict,
        local_init_op=local_init_op,
        ready_for_local_init_op=ready_for_local_init_op
    )
    # The MonitoredTrainingSession takes care of session
    # initialization, restoring from a checkpoint, saving to a
    # checkpoint, and closing when done or an error occurs.
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=checkpoint_dir,
        scaffold=scaffold,
        hooks=hooks,
        config=config,
        save_summaries_secs=None,
        save_summaries_steps=None
    )

    # ----------------------------------------------------------
    # TRAINING
    # ----------------------------------------------------------
    trainer = GaugeModelTrainer(sess, model, train_logger)
    kwargs = {
        'samples_np': samples_init,
        'beta_np': beta_init,
        'net_weights': net_weights_init
    }

    try:
        trainer.train(model.train_steps, **kwargs)
    except TypeError:
        import pdb
        pdb.set_trace()

    trainable_params_file = os.path.join(FLAGS.log_dir, 'trainable_params.txt')
    count_trainable_params(trainable_params_file)

    # close MonitoredTrainingSession and prepare for inference
    sess.close()

    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------
    tf.keras.backend.set_learning_phase(False)

    sess = tf.Session(config=config)
    if is_chief:
        saver = tf.train.Saver()
        saver.restore(sess,
                      tf.train.latest_checkpoint(train_logger.checkpoint_dir))
        model.sess = sess
        run_logger = RunLogger(model, train_logger.log_dir, save_lf_data=False)
        plotter = GaugeModelPlotter(model, run_logger.figs_dir)
    else:
        run_logger = None
        plotter = None

    # Create GaugeModelRunner for inference
    runner = GaugeModelRunner(sess, model, run_logger)
    betas = [model.beta_final]  # model.beta_final + 1]
    if FLAGS.loop_net_weights:
        net_weights_arr = np.array([[1, 1, 1],  # [Q, S, T]
                                    [0, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 0],
                                    [1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1],
                                    [0, 0, 0]], dtype=NP_FLOAT)
    else:
        net_weights_arr = np.array([[1, 1, 1]], dtype=NP_FLOAT)

    therm_frac = 10
    for net_weights in net_weights_arr:
        weights = {
            'charge_weight': charge_weight_init,
            'net_weights': net_weights
        }
        for beta in betas:
            if run_logger is not None:
                run_dir, run_str = run_logger.reset(model.run_steps,
                                                    beta, **weights)
            t0 = time.time()
            runner.run(model.run_steps, beta,
                       weights['net_weights'], therm_frac)
            run_time = time.time() - t0
            io.log(80 * '-')
            io.log(f'Took: {run_time} s to complete run.')
            io.log(80 * '-')

            if plotter is not None and run_logger is not None:
                plotter.plot_observables(run_logger.run_data, beta, run_str,
                                         **weights)
                if FLAGS.save_lf:
                    lf_plotter = LeapfrogPlotter(plotter.out_dir, run_logger)
                    lf_plotter.make_plots(run_dir, num_samples=20)

    return sess, model, train_logger


def run_hmc(FLAGS, params=None, log_file=None):
    """Run generic HMC."""
    condition1 = not FLAGS.horovod
    condition2 = FLAGS.horovod and hvd.rank() == 0
    is_chief = condition1 or condition2
    io.log('\n' + 80 * '-')
    io.log(("Running generic HMC algorithm "
            "with learned parameters from L2HMC..."))
    if is_chief:
        hmc_sess, _, _, _ = hmc(FLAGS, params, log_file)
        hmc_sess.close()
        tf.reset_default_graph()


def run_l2hmc(FLAGS, log_file=None):
    """Train and run L2HMC algorithm."""
    io.log('\n' + 80 * '-')
    io.log("Running L2HMC algorithm...")
    l2hmc_sess, l2hmc_model, l2hmc_train_logger = l2hmc(FLAGS, log_file)
    l2hmc_sess.close()
    tf.reset_default_graph()

    return l2hmc_model, l2hmc_train_logger


def main(FLAGS):
    """Main method for creating/training/running L2HMC for U(1) gauge model."""
    t0 = time.time()
    if HAS_HOROVOD and FLAGS.horovod:
        io.log("INFO: USING HOROVOD")
        log_file = 'output_dirs.txt'
        hvd.init()
    else:
        log_file = None

    if FLAGS.hmc_eps is None:
        eps_arr = [0.1, 0.15, 0.2, 0.25]
    else:
        eps_arr = [float(FLAGS.hmc_eps)]

    if FLAGS.hmc:
        run_hmc(FLAGS, log_file)
        for eps in eps_arr:
            FLAGS.eps = eps
            run_hmc(FLAGS, log_file)

    else:
        model, logger = run_l2hmc(FLAGS, log_file)

        if FLAGS.run_hmc:
            # Run HMC with the trained step size from L2HMC (not ideal)
            params = model.params
            params['hmc'] = True
            params['log_dir'] = FLAGS.log_dir = None
            if logger is not None:
                params['eps'] = FLAGS.eps = logger._current_state['eps']
            else:
                params['eps'] = FLAGS.eps

            run_hmc(FLAGS, params, log_file)

            for eps in eps_arr:
                params['log_dir'] = FLAGS.log_dir = None
                params['eps'] = FLAGS.eps = eps
                run_hmc(FLAGS, params, log_file)

    io.log('\n\n')
    io.log(80 * '-')
    io.log(f'Time to complete: {time.time() - t0:.4g}')
    io.log(80 * '-')


if __name__ == '__main__':
    args = parse_args()
    main(args)
