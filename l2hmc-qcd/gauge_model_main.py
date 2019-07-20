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
try:
    from comet_ml import Experiment
    HAS_COMET = True
except ImportError:
    HAS_COMET = False

import os
import random
import time
import pickle
import tensorflow as tf
import numpy as np

from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorflow.core.protobuf import rewriter_config_pb2

import utils.file_io as io

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

SEP_STR = 80 * '-' + '\n'

# -------------------------------------------
# Set random seeds for tensorflow and numpy
# -------------------------------------------
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)        # `python` build-in pseudo-random generator
np.random.seed(GLOBAL_SEED)     # numpy pseudo-random generator
tf.set_random_seed(GLOBAL_SEED)


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


def create_config(FLAGS, params, train_phase=True):
    """Helper method for creating a tf.ConfigProto object."""
    if train_phase:
        config = tf.ConfigProto()
    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True)
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


def setup_train(FLAGS, log_file=None):
    io.log('\n' + 80 * '-')
    io.log("Running L2HMC algorithm...")
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
        io.log(f"Number of GPUs: {num_workers}")
        params['num_workers'] = num_workers

        # Horovod: Scale initial lr by of num GPUs.
        #  params['lr_init'] *= num_workers
        # Horovod: adjust number of training steps based on number of GPUs.
        params['train_steps'] //= num_workers + 1
        # Horovod: adjust save_steps and lr_decay_steps accordingly.
        params['save_steps'] //= num_workers
        params['lr_decay_steps'] //= num_workers + 1

        if params['summaries']:
            params['logging_steps'] // num_workers + 1

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

    return FLAGS, params, hooks


def train_l2hmc(FLAGS, log_file=None, experiment=None):
    """Create, train, and run L2HMC sampler on 2D U(1) gauge model."""
    tf.keras.backend.set_learning_phase(True)
    FLAGS, params, hooks = setup_train(FLAGS, log_file)

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

    io.log(SEP_STR)
    io.log('L2HMC PARAMETERS:')
    for key, val in params.items():
        io.log(f'  {key}: {val}')
    io.log(SEP_STR)

    # --------------------------------------------------------
    # Create model and train_logger
    # --------------------------------------------------------
    model = GaugeModel(params=params)
    if is_chief:
        train_logger = TrainLogger(model, log_dir, FLAGS.summaries)
        #  run_logger = RunLogger(model,
        #                         log_dir,
        #                         save_lf_data=False,
        #                         summaries=False)
        #  plotter = GaugeModelPlotter(model, run_logger.figs_dir)
    else:
        train_logger = None
        #  train_logger = run_logger = plotter = None

    # --------------------------------------------------
    # Setup config and MonitoredTrainingSession
    # --------------------------------------------------
    config, params = create_config(FLAGS, params, train_phase=True)


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
        model.train_phase: True
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

    #  sess = tf.train.MonitoredTrainingSession(
    #      checkpoint_dir=checkpoint_dir,
    #      scaffold=scaffold,
    #      hooks=hooks,
    #      config=config,
    #      save_summaries_secs=None,
    #      save_summaries_steps=None
    #  )
    sess_kwargs = {
        'checkpoint_dir': checkpoint_dir,
        'scaffold': scaffold,
        'hooks': hooks,
        'config': config,
        'save_summaries_secs': None,
        'save_summaries_steps': None
    }

    sess = tf.train.MonitoredTrainingSession(**sess_kwargs)
    #  tf.keras.backend.set_session(sess)

    #  with tf.train.MonitoredTrainingSession(**kwargs) as sess:

    # ----------------------------------------------------------
    #   TRAINING
    # ----------------------------------------------------------
    trainer = GaugeModelTrainer(sess, model, train_logger)
    train_kwargs = {
        'samples_np': samples_init,
        'beta_np': beta_init,
        'net_weights': net_weights_init
    }

    trainer.train(model.train_steps, **train_kwargs)

    if HAS_COMET and experiment is not None:
        experiment.log_parameters(params)
        g = sess.graph
        experiment.set_model_graph(g)

    params_file = os.path.join(os.getcwd(), 'params.pkl')
    with open(params_file, 'wb') as f:
        pickle.dump(model.params, f)

    # Count all trainable paramters and write them out (w/ shapes) to txt file
    count_trainable_params(os.path.join(FLAGS.log_dir, 'trainable_params.txt'))

    # close MonitoredTrainingSession and reset the default graph
    sess.close()
    tf.reset_default_graph()

    return FLAGS, params, model, train_logger


def run_setup(FLAGS, params):
    """Set up relevant (initial) values to use when running inference."""
    # -------------------------------------------------  
    if params['loop_net_weights']:  # loop over different values of [Q, S, T]
        net_weights_arr = np.zeros((9, 3), dtype=NP_FLOAT)
        mask_arr = np.array([[1, 1, 1],                     # [Q, S, T]
                             [0, 1, 1],                     # [ , S, T]
                             [1, 0, 1],                     # [Q,  , T]
                             [1, 1, 0],                     # [Q, S,  ]
                             [1, 0, 0],                     # [Q,  ,  ]
                             [0, 1, 0],                     # [ , S,  ]
                             [0, 0, 1],                     # [ ,  , T]
                             [0, 0, 0]], dtype=NP_FLOAT)    # [ ,  ,  ]
        net_weights_arr[:mask_arr.shape[0], :] = mask_arr   # [?, ?, ?]
        net_weights_arr[-1, :] = np.random.randn(3)

    else:  # set [Q, S, T] = [1, 1, 1]
        net_weights_arr = np.array([[1, 1, 1]], dtype=NP_FLOAT)

    # if a value has been passed in `kwargs['beta_inference']` use it
    # otherwise, use `model.beta_final`
    betas = [FLAGS.beta_final
             if FLAGS.beta_inference is None
             else FLAGS.beta_inference]

    # if a value has been passed in `kwargs['charge_weight_inference']` use it
    #  qw_train = params['charge_weight']
    #  qw_run = FLAGS.charge_weight_inference
    #  charge_weight = qw_train if qw_run is None else qw_run

    init_dict = {
        'net_weights_arr': net_weights_arr,
        'betas': betas,
        'charge_weight': params['charge_weight'],
    }

    return init_dict


def run_l2hmc(FLAGS, params, checkpoint_dir, experiment=None):
    """Run inference using trained L2HMC sampler."""
    #  assert os.path.isdir(checkpoint_dir)
    tf.keras.backend.set_learning_phase(False)

    # Conditionals required for file I/O
    # if we're not using Horovod, `is_chief` should always be True
    # otheerwise, if using Horovod, we only want to perform file I/O
    # on hvd.rank() == 0, so check that first
    condition1 = not FLAGS.horovod
    condition2 = FLAGS.horovod and hvd.rank() == 0
    is_chief = condition1 or condition2

    if is_chief:
        if checkpoint_dir is not None:
            assert os.path.isdir(checkpoint_dir)
        else:
            raise ValueError(f'Must pass a `checkpoint_dir` to `run_l2hmc`.')
        #  params['run_steps'] //= num_workers
        #  params['lr_init'] *= hvd.size()
    else:
        checkpoint_dir = None

    # --------------------------------------------------------
    # Create model and train_logger
    # --------------------------------------------------------
    model = GaugeModel(params=params)

    init_dict = run_setup(FLAGS, params)

    net_weights_arr = init_dict['net_weights_arr']
    betas = init_dict['betas']
    charge_weight = init_dict['charge_weight']

    if is_chief:
        #  if run_logger is None:
        #  TODO: Fix RunLogger summaries
        run_logger = RunLogger(model,
                               model.log_dir,
                               save_lf_data=False,
                               summaries=False)

        plotter = GaugeModelPlotter(model, run_logger.figs_dir,
                                    experiment=experiment)

        net_weights_file = os.path.join(model.log_dir, 'net_weights.txt')
        np.savetxt(net_weights_file, net_weights_arr,
                   delimiter=', ', newline='\n', fmt="%-.4g")

    else:
        run_logger = plotter = None

    # --------------------------------------------------
    # Setup config and MonitoredTrainingSession
    # --------------------------------------------------
    config, params = create_config(FLAGS, params)
    sess = tf.Session(config=config)
    #  tf.keras.backend.set_session(sess)
    if is_chief:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

    #  init_feed_dict = {
    #      model.x: samples_init,
    #      model.beta: betas[0],
    #      model.charge_weight: charge_weight,
    #      model.net_weights[0]: net_weights_arr[0, 0],  # scale_weight
    #      model.net_weights[1]: net_weights_arr[0, 1],  # transformation_weight
    #      model.net_weights[2]: net_weights_arr[0, 2],  # translation_weight
    #      model.train_phase: False
    #  }

    # ensure all variables are initialized
    #  target_collection = []
    #  if is_chief:
    #      collection = tf.local_variables() + target_collection
    #  else:
    #      collection = tf.local_variables()
    #
    #  local_init_op = tf.variables_initializer(collection)
    #  ready_for_local_init_op = tf.report_uninitialized_variables(collection)

    #  scaffold = tf.train.Scaffold(
    #      init_feed_dict=init_feed_dict,
    #      local_init_op=local_init_op,
    #      ready_for_local_init_op=ready_for_local_init_op
    #  )
    # The MonitoredTrainingSession takes care of session
    # initialization, restoring from a checkpoint, saving to a
    # checkpoint, and closing when done or an error occurs.
    #  sess = tf.train.MonitoredTrainingSession(
    #      checkpoint_dir=checkpoint_dir,
    #      scaffold=scaffold,
    #      hooks=hooks,
    #      config=config,
    #      save_summaries_secs=None,
    #      save_summaries_steps=None
    #  )

    # ----------------------------------------------------------
    # INFERENCE
    # ----------------------------------------------------------
    runner = GaugeModelRunner(sess, model, run_logger)

    for net_weights in net_weights_arr:
        weights = {
            'charge_weight': charge_weight,
            'net_weights': net_weights
        }
        for beta in betas:
            if run_logger is not None:
                # There are two subtle points worth pointing out:
                #   1. The value of `charge_weight` is specified when resetting
                #      the run logger, which will  then be used for the
                #      remainder of inference.
                #   2. The value of `net_weight` is spcified by passing it
                #      directly to the GaugeModelRunner.run(...) method.
                #
                run_dir, run_str = run_logger.reset(model.run_steps, beta,
                                                    **weights)
            t0 = time.time()

            runner.run(model.run_steps,
                       beta,
                       weights['net_weights'])

            # log the total time spent running inference
            run_time = time.time() - t0
            io.log(SEP_STR)
            io.log(f'Took: {run_time} s to complete run.\n')
            io.log(SEP_STR)

            if plotter is not None and run_logger is not None:
                plotter.plot_observables(run_logger.run_data,
                                         beta, run_str, **weights)
                if params['save_lf']:
                    lf_plotter = LeapfrogPlotter(plotter.out_dir, run_logger)
                    num_samples = min((model.num_samples, 20))
                    lf_plotter.make_plots(run_dir, num_samples=num_samples)


def main(FLAGS):
    """Main method for creating/training/running L2HMC for U(1) gauge model."""
    t0 = time.time()
    if HAS_HOROVOD and FLAGS.horovod:
        io.log("INFO: USING HOROVOD")
        log_file = 'output_dirs.txt'
        hvd.init()
    else:
        log_file = None

    condition1 = not FLAGS.horovod
    condition2 = FLAGS.horovod and hvd.rank() == 0
    is_chief = condition1 or condition2

    if FLAGS.comet and is_chief:
        experiment = Experiment(api_key="r7rKFO35BJuaY3KT1Tpj4adco",
                                project_name="l2hmc-qcd",
                                workspace="saforem2")
        name = (f'{FLAGS.network_arch}_'
                f'lf{FLAGS.num_steps}_'
                f'batch{FLAGS.num_samples}_'
                f'qw{FLAGS.charge_weight}_'
                f'aux{FLAGS.aux_weight}')
        experiment.set_name(name)

    else:
        experiment = None

    if FLAGS.hmc_eps is None:
        eps_arr = [0.1, 0.15, 0.2, 0.25]
    else:
        eps_arr = [float(FLAGS.hmc_eps)]

    if FLAGS.hmc:
        # --------------------
        #   run generic HMC
        # --------------------
        import gauge_model_hmc as hmc
        hmc.run_hmc(FLAGS, log_file)
        for eps in eps_arr:
            FLAGS.eps = eps
            hmc.run_hmc(FLAGS, log_file)
    else:
        # ------------------------
        #   train l2hmc sampler
        # ------------------------
        FLAGS, params, model, train_logger = train_l2hmc(FLAGS,
                                                         log_file,
                                                         experiment)
        if experiment is not None:
            experiment.log_parameters(model.params)

        if FLAGS.inference:
            if train_logger is not None:
                checkpoint_dir = train_logger.checkpoint_dir
            else:
                checkpoint_dir = None

            # ---------------------------------------------
            #   run inference using trained l2hmc sampler
            # ---------------------------------------------
            run_l2hmc(FLAGS, params, checkpoint_dir)

        # -----------------------------------------------------------
        #  run HMC following inference if --run_hmc flag was passed
        # -----------------------------------------------------------
        if FLAGS.run_hmc:
            # Run HMC with the trained step size from L2HMC (not ideal)
            params = model.params
            params['hmc'] = True
            params['log_dir'] = FLAGS.log_dir = None
            if train_logger is not None:
                params['eps'] = FLAGS.eps = train_logger._current_state['eps']
            else:
                params['eps'] = FLAGS.eps

            hmc.run_hmc(FLAGS, params, log_file)

            for eps in eps_arr:
                params['log_dir'] = FLAGS.log_dir = None
                params['eps'] = FLAGS.eps = eps
                hmc.run_hmc(FLAGS, params, log_file)

    io.log('\n\n')
    io.log(80 * '-')
    io.log(f'Time to complete: {time.time() - t0:.4g}')
    io.log(80 * '-')


if __name__ == '__main__':
    args = parse_args()
    main(args)
