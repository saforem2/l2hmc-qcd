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
import pickle
import datetime
import tensorflow as tf

from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorflow.core.protobuf import rewriter_config_pb2


import utils.file_io as io
from utils.parse_args import parse_args
from models.gauge_model import GaugeModel
from loggers.train_logger import TrainLogger
from loggers.run_logger import RunLogger
from trainers.gauge_model_trainer import GaugeModelTrainer
from plotters.gauge_model_plotter import GaugeModelPlotter
from runners.gauge_model_runner import GaugeModelRunner

from globals import TRAIN_HEADER, RUN_HEADER, FILE_PATH

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
        io.log("Using gpu for training.")
        params['data_format'] = 'channels_first'
        #  os.environ["KMP_BLOCKTIME"] = str(0)
        #  os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
        # Horovod: pin GPU to be used to process local rank (one GPU per
        # process)
        #  config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        if HAS_HOROVOD and FLAGS.horovod:
            num_gpus = hvd.size()
            io.log(f"Number of GPUs: {num_gpus}")
            config.gpu_options.visible_device_list = str(hvd.local_rank())
    else:
        params['data_format'] = 'channels_last'

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


def create_log_dir(FLAGS):
    """Automatically create and name `log_dir` to save model data to.

    The created directory will be located in `logs/YYYY_M_D/`, and will have
    the format (without `_qw{QW}` if running generic HMC):

        `lattice{LX}_batch{NS}_lf{LF}_eps{SS}_qw{QW}`

    Returns:
        FLAGS, with FLAGS.log_dir being equal to the newly created log_dir.

    NOTE: If log_dir does not already exist, it is created.
    """
    LX = FLAGS.space_size
    NS = FLAGS.num_samples
    LF = FLAGS.num_steps
    SS = str(FLAGS.eps).lstrip('0.')
    if FLAGS.charge_weight == 0:
        QW = str('0')
    else:
        QW = str(FLAGS.charge_weight).rstrip('.0')
    if FLAGS.hmc:
        run_str = f'HMC_lattice{LX}_batch{NS}_lf{LF}_eps{SS}'
    else:
        run_str = f'lattice{LX}_batch{NS}_lf{LF}_eps{SS}_qw{QW}'

    now = datetime.datetime.now()
    date_str = f'{now.year}_{now.month}_{now.day}'
    project_dir = os.path.abspath(os.path.dirname(FILE_PATH))
    if FLAGS.log_dir is None:
        root_log_dir = os.path.join(project_dir, 'logs', date_str, run_str)
    else:
        root_log_dir = os.path.join(project_dir, FLAGS.log_dir, date_str,
                                    run_str)
    io.check_else_make_dir(root_log_dir)
    run_num = io.get_run_num(root_log_dir)
    log_dir = os.path.abspath(os.path.join(root_log_dir,
                                           f'run_{run_num}'))

    return log_dir


def hmc(FLAGS, l2hmc_model=None, l2hmc_train_logger=None):
    """Create and run generic HMC sampler using trained params from L2HMC."""
    condition1 = not FLAGS.horovod
    condition2 = FLAGS.horovod and hvd.rank() == 0
    is_chief = condition1 or condition2
    if not is_chief:
        return -1

    if l2hmc_model is None:
        if FLAGS.log_dir is None:
            FLAGS.log_dir = create_log_dir(FLAGS)

        params = {
            'using_hvd': FLAGS.horovod,
        }

        for key, val in FLAGS.__dict__.items():
            params[key] = val

    else:
        params = l2hmc_model.params

    if l2hmc_train_logger is not None:
        params['eps'] = l2hmc_train_logger._current_state['eps']
        params['hmc'] = True
        params['beta_init'] = l2hmc_model.beta_init
        params['beta_final'] = l2hmc_model.beta_final
        params['log_dir'] = os.path.join(l2hmc_train_logger.log_dir, 'HMC')
        io.check_else_make_dir(params['log_dir'])

    figs_dir = os.path.join(FLAGS.log_dir, 'figures')

    io.check_else_make_dir(FLAGS.log_dir)
    io.check_else_make_dir(figs_dir)

    eps = params['eps']  # step size for (generic) HMC leapfrog integrator

    # create tensorflow config (`config_proto`) to configure session
    config, params = create_config(FLAGS, params)
    sess = tf.Session(config=config)

    model = GaugeModel(params=params)
    run_logger = RunLogger(sess, model, FLAGS.log_dir)
    runner = GaugeModelRunner(sess, model, run_logger)
    plotter = GaugeModelPlotter(figs_dir)

    sess.run(tf.global_variables_initializer())

    betas = [FLAGS.beta_final - 1,
             FLAGS.beta_final,
             FLAGS.beta_final + 1]
    for beta in betas:
        if run_logger is not None:
            run_logger.reset(model.run_steps, beta)

        runner.run(model.run_steps, beta)

        if plotter is not None and run_logger is not None:
            plotter.plot_observables(run_logger.run_data, beta)

    return sess, model, runner, run_logger


def l2hmc(FLAGS):
    """Create, train, and run L2HMC sampler on 2D U(1) gauge model."""
    if FLAGS.log_dir is None:
        FLAGS.log_dir = create_log_dir(FLAGS)

    params = {}
    for key, val in FLAGS.__dict__.items():
        params[key] = val

    if FLAGS.horovod:
        params['using_hvd'] = True
        num_workers = hvd.size()
        params['train_steps'] //= num_workers
        params['save_steps'] //= num_workers
        params['lr_decay_steps'] //= num_workers
        #  params['run_steps'] //= num_workers
        #  params['lr_init'] *= hvd.size()
    else:
        params['using_hvd'] = False

    config, params = create_config(FLAGS, params)
    sess = tf.Session(config=config)

    model = GaugeModel(params=params)

    condition1 = not FLAGS.horovod
    condition2 = FLAGS.horovod and hvd.rank() == 0
    is_chief = condition1 or condition2

    if is_chief:
        log_dir = params.get('log_dir', 'logs')
        train_logger = TrainLogger(sess, model, log_dir, FLAGS.summaries)
        run_logger = RunLogger(sess, model, train_logger.log_dir)
        plotter = GaugeModelPlotter(run_logger.figs_dir)
    else:
        train_logger = None
        run_logger = None
        plotter = None

    trainer = GaugeModelTrainer(sess, model, train_logger)

    sess.run(tf.global_variables_initializer())

    if FLAGS.horovod:
        sess.run(hvd.broadcast_global_variables(0))

    trainer.train(model.train_steps)

    runner = GaugeModelRunner(sess, model, run_logger)

    #  betas = np.arange(model.beta_init, model.beta_final, 1)
    betas = [model.beta_final - 1, model.beta_final, model.beta_final + 1]
    for beta in betas:
        if run_logger is not None:
            run_logger.reset(model.run_steps, beta)

        runner.run(model.run_steps, beta)

        if plotter is not None and run_logger is not None:
            plotter.plot_observables(run_logger.run_data, beta)

    return sess, model, train_logger


def main(FLAGS):
    """Main method for creating/training/running L2HMC for U(1) gauge model."""
    if HAS_HOROVOD and FLAGS.horovod:
        io.log("INFO: USING HOROVOD")
        hvd.init()

    condition1 = not FLAGS.horovod
    condition2 = FLAGS.horovod and hvd.rank() == 0
    is_chief = condition1 or condition2

    if not FLAGS.hmc:
        io.log('\n' + 80 * '-')
        io.log("Running L2HMC algorithm...")
        l2hmc_sess, l2hmc_model, l2hmc_train_logger = l2hmc(FLAGS)
        l2hmc_sess.close()
        tf.reset_default_graph()

        io.log('\n' + 80 * '-')
        io.log(("Running generic HMC algorithm "
                "with learned parameters from L2HMC..."))

        if is_chief:
            hmc_sess, _, _, _ = hmc(FLAGS, l2hmc_model, l2hmc_train_logger)
            hmc_sess.close()
    else:
        if is_chief:
            hmc_sess, _, _, _ = hmc(FLAGS)
            hmc_sess.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
