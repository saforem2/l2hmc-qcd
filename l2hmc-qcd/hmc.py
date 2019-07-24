"""
gauge_model_hmc.py


Runs inference using generic HMC for the 2D U(1) lattice gauge theory model.

Author: Sam Foreman (github: @saforem2)
Date: 07/16/2019
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
import gauge_model_main as gauge_model_main


from globals import GLOBAL_SEED, NP_FLOAT
from utils.parse_args import parse_args
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

# -------------------------------------------
# Set random seeds for tensorflow and numpy
# -------------------------------------------
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)        # `python` build-in pseudo-random generator
np.random.seed(GLOBAL_SEED)     # numpy pseudo-random generator
tf.set_random_seed(GLOBAL_SEED)


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
    params['eps_fixed'] = True

    figs_dir = os.path.join(params['log_dir'], 'figures')
    io.check_else_make_dir(figs_dir)

    io.log(80 * '-' + '\n')
    io.log('HMC PARAMETERS:')
    for key, val in params.items():
        io.log(f'  {key}: {val}')
    io.log(80 * '-' + '\n')

    # create tensorflow config (`config_proto`) to configure session
    config, params = gauge_model_main.create_config(FLAGS, params)
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
        # --------------------
        #   run generic HMC
        # --------------------
        run_hmc(FLAGS, log_file)
        for eps in eps_arr:
            FLAGS.eps = eps
            run_hmc(FLAGS, log_file)

    else:
        io.log('Nothing to do!')

    io.log('\n\n')
    io.log(80 * '-')
    io.log(f'Time to complete: {time.time() - t0:.4g}')
    io.log(80 * '-')


if __name__ == '__main__':
    args = parse_args()
    main(args)
