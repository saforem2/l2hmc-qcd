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

import numpy as np
import tensorflow as tf

# pylint:disable=import-error
# pylint:disable=unused-import
# pylint:disable=too-many-statements
# pylint:disable=no-name-in-module, invalid-name
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline

import config as cfg
import utils.file_io as io

from seed_dict import seeds, vnet_seeds, xnet_seeds
from models.gauge_model import GaugeModel
from plotters.plot_utils import plot_singular_values, weights_hist
from loggers.train_logger import TrainLogger
from utils.file_io import timeit
from utils.parse_args import parse_args
from trainers.trainer import Trainer
from trainers.train_setup import (check_reversibility, count_trainable_params,
                                  create_config, get_net_weights, train_setup)

if cfg.HAS_HOROVOD:
    import horovod.tensorflow as hvd

try:
    tf.logging.set_verbosity(tf.logging.INFO)
except AttributeError:
    pass

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

SEP_STR = 80 * '-'  # + '\n'

NP_FLOAT = cfg.NP_FLOAT

Weights = namedtuple('Weights', ['w', 'b'])

# pylint:disable=too-many-statements

def log_params(params):
    io.log(SEP_STR + '\nL2HMC PARAMETERS:\n')
    for key, val in params.items():
        io.log(f' - {key} : {val}\n')
    io.log(SEP_STR)


def create_monitored_training_session(**sess_kwargs):
    global_var_init = tf.global_variables_initializer()
    local_var_init = tf.local_variables_initializer()
    uninited = tf.report_uninitialized_variables()
    sess = tf.train.MonitoredTrainingSession(**sess_kwargs)
    tf.keras.backend.set_session(sess)
    sess.run([global_var_init, local_var_init])
    uninited_out = sess.run(uninited)
    io.log(f'tf.report_uninitialized_variables() len = {uninited_out}')

    return sess


def pkl_dump(d, pkl_file):
    """Dump `d` to `pkl_file`."""
    with open(pkl_file, 'wb') as f:
        pickle.dump(d, f)


def save_params(model):
    """Save model parameters to `.pkl` files.

    Additionally, write out all trainable parameters (w/ sizes) to `.txt` file.
    """
    out_file = os.path.join(model.log_dir, 'trainable_params.txt')
    count_trainable_params(out_file)

    io.save_pkl(model.params, os.path.join(os.getcwd(), 'params.pkl'))

    dynamics_dir = os.path.join(model.log_dir, 'dynamics')
    io.check_else_make_dir(dynamics_dir)
    out_file = os.path.join(dynamics_dir, 'dynamics_params.pkl')
    io.save_pkl(model.dynamics._params, out_file)


def save_masks(model, sess):
    """Save `model.dynamics.masks` for inference."""
    try:
        masks_file = os.path.join(model.log_dir, 'dynamics_mask.pkl')
        masks_file_ = os.path.join(model.log_dir, 'dynamics_mask.np')
        masks = sess.run(model.dynamics.masks)
        np.array(masks).tofile(masks_file_)
        io.log(f'dynamics.masks:\n\t {masks}')
        pkl_dump(masks, masks_file)
    except:
        import pudb; pudb.set_trace()


def save_seeds(model):
    """Save network seeds for reproducibility."""
    io.save_dict(seeds, out_dir=model.log_dir, name='seeds')
    io.save_dict(xnet_seeds, out_dir=model.log_dir, name='xnet_seeds')
    io.save_dict(vnet_seeds, out_dir=model.log_dir, name='vnet_seeds')


def save_weights(model, sess):
    """Save network weights to `.pkl` file."""
    xw_file = os.path.join(model.log_dir, 'xnet_weights.pkl')
    xnet_weights = model.dynamics.xnet.save_weights(sess, xw_file)

    vw_file = os.path.join(model.log_dir, 'vnet_weights.pkl')
    vnet_weights = model.dynamics.vnet.save_weights(sess, vw_file)
    model_weights = {
        'xnet': xnet_weights,
        'vnet': vnet_weights,
    }
    io.save_pkl(model_weights, os.path.join(model.log_dir,
                                            'weights.pkl'))


def save_eps(model, sess):
    """Save final value of `eps` (step size) at the end of training."""
    eps_np = sess.run(model.dynamics.eps)
    eps_dict = {'eps': eps_np}
    io.save_pkl(eps_dict, os.path.join(model.log_dir, 'eps_np.pkl'))



@timeit
def train_l2hmc(FLAGS, log_file=None):
    """Create, train, and run L2HMC sampler on 2D U(1) gauge model."""
    start_time = time.time()
    tf.keras.backend.set_learning_phase(True)

    if FLAGS.restore and FLAGS.log_dir is not None:
        params = io.load_params(FLAGS.log_dir)
        if FLAGS.horovod and params['using_hvd']:  # should be the same
            num_workers = hvd.size()
            assert num_workers == params['num_workers']
            hooks = [hvd.BroadcastGlobalVariablesHook(0)]
            params['logging_steps'] *= num_workers
        else:
            hooks = []

    else:
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

    params['zero_masks'] = FLAGS.zero_masks

    if is_chief:
        log_dir = params['log_dir']
        checkpoint_dir = os.path.join(log_dir, 'checkpoints/')
        io.check_else_make_dir(checkpoint_dir)

    else:
        log_dir = None
        checkpoint_dir = None

    log_params(params)
    # --------------------------------------------------------
    # Create model and train_logger
    # --------------------------------------------------------
    model = GaugeModel(params)

    # Only create `TrainLogger` if `hvd.rank == 0`
    if is_chief:
        logging_steps = params.get('logging_steps', 10)
        train_logger = TrainLogger(model, log_dir,
                                   logging_steps=logging_steps,
                                   summaries=params['summaries'])
    else:
        train_logger = None

    # -------------------------------------------------------
    # Setup `tf.ConfigProto` object for `tf.Session`
    # -------------------------------------------------------
    config, params = create_config(params)

    net_weights_init = cfg.NetWeights(
        x_scale=FLAGS.x_scale_weight,
        x_translation=FLAGS.x_translation_weight,
        x_transformation=FLAGS.x_transformation_weight,
        v_scale=FLAGS.v_scale_weight,
        v_translation=FLAGS.v_translation_weight,
        v_transformation=FLAGS.v_transformation_weight,
    )
    samples_init = np.array(model.lattice.samples_array, dtype=NP_FLOAT)
    beta_init = model.beta_init

    # ----------------------------------------------------------------
    #  Create MonitoredTrainingSession
    #
    #  NOTE: The MonitoredTrainingSession takes care of session
    #        initialization, restoring from a checkpoint, saving to a
    #        checkpoint, and closing when done or an error occurs.
    # ----------------------------------------------------------------
    save_steps = FLAGS.save_steps
    sess = create_monitored_training_session(hooks=hooks,
                                             config=config,
                                             #  scaffold=scaffold,
                                             save_summaries_secs=None,
                                             save_summaries_steps=None,
                                             save_checkpoint_steps=save_steps,
                                             checkpoint_dir=checkpoint_dir)
    #  sess.run([
    #      model.dynamics.xnet.generic_net.coeff_scale.initializer,
    #      model.dynamics.vnet.generic_net.coeff_scale.initializer,
    #      model.dynamics.xnet.generic_net.coeff_transformation.initializer,
    #      model.dynamics.vnet.generic_net.coeff_transformation.initializer,
    #  ])

    # ----------------------------------------------------------
    #                       TRAINING
    # ----------------------------------------------------------
    trainer = Trainer(sess, model, train_logger, params)

    trainer.train(model.train_steps,
                  beta=beta_init,
                  samples=samples_init,
                  net_weights=net_weights_init)

    if is_chief:
        save_masks(model, sess)
        save_params(model)
        save_seeds(model)
        save_weights(model, sess)
        save_eps(model, sess)
        # wfile = os.path.join(model.log_dir, 'dynamics_weights.h5')
        # model.dynamics.save_weights(wfile)
        #  io.save_dict(model.params, os.path.join(os.getcwd()), 'params.pkl')

    # close MonitoredTrainingSession and reset the default graph
    sess.close()
    tf.reset_default_graph()
    io.log(f'{SEP_STR}\n training took:'
           f'{time.time()-start_time:.3g}s \n{SEP_STR}')

    return model, train_logger


@timeit
def main(FLAGS):
    """Main method for creating/training/running L2HMC for U(1) gauge model."""
    log_file = 'output_dirs.txt'

    using_hvd = getattr(FLAGS, 'horovod', False)
    if cfg.HAS_HOROVOD and using_hvd:
        io.log("INFO: USING HOROVOD")
        hvd.init()
        rank = hvd.rank()
        print(f'Setting seed from rank: {rank}')
        # multiply the global seed by the rank so each rank gets diff seed
        tf.set_random_seed(rank * seeds['global_tf'])

    #  if FLAGS.hmc:   # run generic HMC sampler
    #      inference.run_hmc(FLAGS, log_file=log_file)
    #  else:           # train l2hmc sampler
    #  plot_singular_values(model.log_dir)
    model, _ = train_l2hmc(FLAGS, log_file)


if __name__ == '__main__':
    FLAGS = parse_args()
    USING_HVD = getattr(FLAGS, 'horovod', False)
    if not USING_HVD:
        tf.set_random_seed(seeds['global_tf'])

    t0 = time.time()
    main(FLAGS)
    io.log('\n\n' + SEP_STR)
    io.log(f'Time to complete: {time.time() - t0:.4g}')
    io.log(SEP_STR)
