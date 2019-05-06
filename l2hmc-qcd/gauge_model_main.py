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
import datetime
import tensorflow as tf

from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorflow.core.protobuf import rewriter_config_pb2

import utils.file_io as io

from globals import FILE_PATH
from utils.parse_args import parse_args
from utils.model_loader import load_model
from models.gauge_model import GaugeModel
from loggers.train_logger import TrainLogger
from loggers.run_logger import RunLogger
from trainers.gauge_model_trainer import GaugeModelTrainer
from plotters.gauge_model_plotter import GaugeModelPlotter
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
    sess = tf.Session(config=config)

    model = GaugeModel(params=params)
    run_logger = RunLogger(model, params['log_dir'])
    plotter = GaugeModelPlotter(run_logger.figs_dir)

    sess.run(tf.global_variables_initializer())

    runner = GaugeModelRunner(sess, model, run_logger)

    betas = [FLAGS.beta_final, FLAGS.beta_final + 1]
    for beta in betas:
        if run_logger is not None:
            run_logger.reset(model.run_steps, beta)

        runner.run(model.run_steps, beta)

        if plotter is not None and run_logger is not None:
            plotter.plot_observables(run_logger.run_data, beta)

    return sess, model, runner, run_logger


def l2hmc(FLAGS, log_file=None):
    """Create, train, and run L2HMC sampler on 2D U(1) gauge model."""
    FLAGS.log_dir = io.create_log_dir(FLAGS, log_file=log_file)

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
        params['train_steps'] //= num_workers
        params['save_steps'] //= num_workers
        params['lr_decay_steps'] //= num_workers
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

    model = GaugeModel(params=params)
    if is_chief:
        train_logger = TrainLogger(model, log_dir, FLAGS.summaries)
        run_logger = RunLogger(model, train_logger.log_dir)
        plotter = GaugeModelPlotter(run_logger.figs_dir)
    else:
        train_logger = None
        run_logger = None
        plotter = None

    config, params = create_config(FLAGS, params)
    #  sess = tf.Session(config=config)
    tf.keras.backend.set_learning_phase(True)

    # The MonitoredTrainingSession takes care of session
    # initialization, restoring from a checkpoint, saving to a
    # checkpoint, and closing when done or an error occurs.
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=checkpoint_dir,
        hooks=hooks,
        config=config,
        save_summaries_secs=None,
        save_summaries_steps=None
    )

    trainer = GaugeModelTrainer(sess, model, train_logger)

    try:
        trainer.train(model.train_steps)
    except:
        # i.e. Tensor had Inf / NaN values caused by high learning rate
        io.log(80 * '-')
        io.log('Training crashed! Decreasing lr_init by 10% and retrying...')
        io.log(f'Previous lr_init: {FLAGS.lr_init}')
        FLAGS.lr_init *= 0.9
        io.log(f'New lr_init: {FLAGS.lr_init}')
        sess.close()
        tf.reset_default_graph()
        l2hmc(FLAGS)

    tf.keras.backend.set_learning_phase(False)

    runner = GaugeModelRunner(sess, model, run_logger)

    #  betas = np.arange(model.beta_init, model.beta_final, 1)
    betas = [model.beta_final, model.beta_final + 1]
    for beta in betas:
        if run_logger is not None:
            run_logger.reset(model.run_steps, beta)

        runner.run(model.run_steps, beta)

        if plotter is not None and run_logger is not None:
            plotter.plot_observables(run_logger.run_data, beta)

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
    if HAS_HOROVOD and FLAGS.horovod:
        io.log("INFO: USING HOROVOD")
        log_file = 'output_dirs.txt'
        hvd.init()
    else:
        log_file = None

    eps_arr = [0.1, 0.2, 0.3]

    if FLAGS.hmc:
        run_hmc(FLAGS, log_file)
        for eps in eps_arr:
            FLAGS.eps = eps
            run_hmc(FLAGS, log_file)

    else:
        # Run L2HMC
        model, logger = run_l2hmc(FLAGS, log_file)

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


if __name__ == '__main__':
    args = parse_args()
    main(args)
