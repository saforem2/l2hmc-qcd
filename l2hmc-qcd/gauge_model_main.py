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
import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorflow.core.protobuf import rewriter_config_pb2


import utils.file_io as io
from utils.parse_args import parse_args
from lattice.lattice import u1_plaq_exact
from models.gauge_model import GaugeModel
from loggers.gauge_model_logger import GaugeModelLogger
from trainers.gauge_model_trainer import GaugeModelTrainer
from plotters.gauge_model_plotter import GaugeModelPlotter
from runners.gauge_model_runner import GaugeModelRunner

from globals import TRAIN_HEADER, RUN_HEADER

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


def create_config_proto(FLAGS, params):
    """Create tensorflow config."""
    config_proto = tf.ConfigProto()
    if FLAGS.time_size > 8:
        off = rewriter_config_pb2.RewriterConfig.OFF
        config_proto_attrs = config_proto.graph_options.rewrite_options
        config_proto_attrs.arithmetic_optimization = off

    if FLAGS.gpu:
        io.log("Using gpu for training.")
        params['data_format'] = 'channels_first'
        os.environ["KMP_BLOCKTIME"] = str(0)
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
        # Horovod: pin GPU to be used to process local rank (one GPU per
        # process)
        config_proto.allow_soft_placement = True
        config_proto.gpu_options.allow_growth = True
        if HAS_HOROVOD and FLAGS.horovod:
            config_proto.gpu_options.visible_device_list = str(
                hvd.local_rank()
            )
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
        config_proto.allow_soft_placement = True
        config_proto.intra_op_parallelism_threads = OMP_NUM_THREADS
        config_proto.inter_op_parallelism_threads = 0

    return config_proto, params


def hmc(FLAGS, l2hmc_model=None, l2hmc_logger=None):
    """Create and run generic HMC sampler using trained params from L2HMC."""
    condition1 = not FLAGS.horovod
    condition2 = FLAGS.horovod and hvd.rank() == 0
    is_chief = condition1 or condition2
    if not is_chief:
        return -1

    if l2hmc_model is not None:
        hmc_params = l2hmc_model.params

    else:
        for key, val in FLAGS.__dict__.items():
            hmc_params[key] = val

    if l2hmc_logger is not None:
        hmc_params['eps'] = l2hmc_logger._current_state['eps']
        hmc_params['hmc'] = True
        hmc_params['beta_init'] = l2hmc_model.beta_init
        hmc_params['beta_final'] = l2hmc_model.beta_final
        hmc_log_dir = os.path.join(l2hmc_logger.log_dir, 'HMC')

    else:
        hmc_log_dir = hmc_params.get('log_dir', 'logs')
        hmc_log_dir = os.path.join(hmc_log_dir, 'HMC')

    hmc_config_proto, hmc_params = create_config_proto(FLAGS,
                                                       hmc_params)
    hmc_sess = tf.Session(config=hmc_config_proto)

    hmc_model = GaugeModel(params=hmc_params)

    hmc_logger = GaugeModelLogger(hmc_sess, hmc_model, hmc_log_dir)
    runs_dir = os.path.join(hmc_logger.log_dir, 'runs')
    io.check_else_make_dir(runs_dir)

    hmc_runner = GaugeModelRunner(hmc_sess, hmc_model, runs_dir)
    hmc_plotter = GaugeModelPlotter(hmc_logger.figs_dir)

    hmc_sess.run(tf.global_variables_initializer())

    run_kwargs = {
        'beta': hmc_model.beta_final
    }
    run_data = hmc_runner.run(int(hmc_model.run_steps),
                              beta_np=run_kwargs['beta'],
                              ret=True)

    hmc_plotter.plot_observables(run_data, **run_kwargs)

    betas = np.arange(hmc_model.beta_init, hmc_model.beta_final, 1)
    for beta in betas:
        run_kwargs['beta'] = beta
        run_data = hmc_runner.run(int(hmc_model.run_steps),
                                  beta_np=beta, ret=True)
        hmc_plotter.plot_observables(run_data, **run_kwargs)

    return hmc_sess, hmc_model, hmc_runner, hmc_logger


def l2hmc(FLAGS):
    """Create, train, and run L2HMC sampler on 2D U(1) gauge model."""
    params = {}
    for key, val in FLAGS.__dict__.items():
        params[key] = val

    params['using_hvd'] = True if FLAGS.horovod else False

    if FLAGS.hmc:
        params['eps_trainable'] = False

    config_proto, params = create_config_proto(FLAGS, params)
    sess = tf.Session(config=config_proto)

    model = GaugeModel(params=params)

    condition1 = not FLAGS.horovod
    condition2 = FLAGS.horovod and hvd.rank() == 0
    is_chief = condition1 or condition2

    if is_chief:
        log_dir = params.get('log_dir', 'logs')
        logger = GaugeModelLogger(sess, model, log_dir)
    else:
        logger = None

    trainer = GaugeModelTrainer(sess, model, logger)

    sess.run(tf.global_variables_initializer())

    if FLAGS.horovod:
        io.log(f"Number of GPUs: {hvd.size())}")
        sess.run(hvd.broacast_global_variables(0))

    trainer.train(model.train_steps)

    if is_chief:
        run_kwargs = {
            'beta': model.beta_final
        }
        runs_dir = os.path.join(logger.log_dir, 'runs')
        io.check_else_make_dir(runs_dir)
        plotter = GaugeModelPlotter(logger.figs_dir)
        runner = GaugeModelRunner(sess, model, runs_dir)

        run_data = runner.run(int(model.run_steps),
                              beta_np=run_kwargs['beta'],
                              ret=True)
        plotter.plot_observables(run_data, **run_kwargs)

        betas = np.arange(model.beta_init, model.beta_final, 1)
        for beta in betas:
            run_kwargs['beta'] = beta
            run_data = runner.run(int(model.run_steps), beta_np=beta, ret=True)
            plotter.plot_observables(run_data, **run_kwargs)

    return sess, model, logger


def main(FLAGS):
    """Main method for creating/training/running L2HMC for U(1) gauge model."""
    if HAS_HOROVOD and FLAGS.horovod:
        io.log("INFO: USING HOROVOD")
        hvd.init()

    io.log('\n' + 80 * '-')
    io.log("Running L2HMC algorithm...")

    l2hmc_sess, l2hmc_model, l2hmc_logger = l2hmc(FLAGS)

    l2hmc_sess.close()
    tf.reset_default_graph()

    io.log('\n' + 80 * '-')
    io.log(("Running generic HMC algorithm "
            "with learned parameters from L2HMC..."))

    condition1 = not FLAGS.horovod
    condition2 = FLAGS.horovod and hvd.rank() == 0
    is_chief = condition1 or condition2

    if is_chief:
        hmc_sess, _, _, _ = hmc(FLAGS, l2hmc_model, l2hmc_logger)
        hmc_sess.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
