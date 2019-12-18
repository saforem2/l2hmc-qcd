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
import config as cfg

from seed_dict import seeds, vnet_seeds, xnet_seeds
from collections import namedtuple

import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug  # noqa: F401
from tensorflow.python.client import timeline  # noqa: F401

import inference
import utils.file_io as io

from utils.parse_args import parse_args
from models.gauge_model import GaugeModel
from plotters.plot_utils import weights_hist
from loggers.train_logger import TrainLogger
from trainers.trainer import Trainer
from trainers.train_setup import (check_reversibility, count_trainable_params,
                                  create_config, get_net_weights, train_setup)

if cfg.HAS_HOROVOD:
    import horovod.tensorflow as hvd

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)

SEP_STR = 80 * '-'  # + '\n'

NP_FLOAT = cfg.NP_FLOAT


Weights = namedtuple('Weights', ['w', 'b'])


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


def train_l2hmc(FLAGS, log_file=None):
    """Create, train, and run L2HMC sampler on 2D U(1) gauge model."""
    t0 = time.time()
    tf.keras.backend.set_learning_phase(True)

    if FLAGS.restore and FLAGS.log_dir is not None:
        params = io.load_params(FLAGS.log_dir)
        if FLAGS.horovod and params['using_hvd']:  # should be the same
            num_workers = hvd.size()
            assert num_workers == params['num_workers']
            hooks = [hvd.BroadcastGlobalVariablesHook(0)]
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

    if is_chief:
        weights = get_net_weights(model)
        xnet = model.dynamics.xnet.generic_net
        vnet = model.dynamics.vnet.generic_net
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
    global_step = tf.train.get_or_create_global_step()

    # ----------------------------------------------------------------
    #  Create MonitoredTrainingSession
    #
    #  NOTE: The MonitoredTrainingSession takes care of session
    #        initialization, restoring from a checkpoint, saving to a
    #        checkpoint, and closing when done or an error occurs.
    # ----------------------------------------------------------------
    sess = create_monitored_training_session(hooks=hooks,
                                             config=config,
                                             #  scaffold=scaffold,
                                             save_summaries_secs=None,
                                             save_summaries_steps=None,
                                             checkpoint_dir=checkpoint_dir)
    sess.run([
        model.dynamics.xnet.generic_net.coeff_scale.initializer,
        model.dynamics.vnet.generic_net.coeff_scale.initializer,
        model.dynamics.xnet.generic_net.coeff_transformation.initializer,
        model.dynamics.vnet.generic_net.coeff_transformation.initializer,
    ])

    # Check reversibility and write results out to `.txt` file.
    reverse_file = os.path.join(model.log_dir, 'reversibility_test.txt')
    check_reversibility(model, sess, net_weights_init, out_file=reverse_file)

    if is_chief:  # save copy of seeds dictionaries for reproducibility
        io.save_dict(seeds, out_dir=model.log_dir, name='seeds')
        io.save_dict(xnet_seeds, out_dir=model.log_dir, name='xnet_seeds')
        io.save_dict(vnet_seeds, out_dir=model.log_dir, name='vnet_seeds')

    # ----------------------------------------------------------
    #                       TRAINING
    # ----------------------------------------------------------
    trainer = Trainer(sess, model, train_logger, **params)

    initial_step = sess.run(global_step)
    trainer.train(model.train_steps,
                  beta=beta_init,
                  samples=samples_init,
                  initial_step=initial_step,
                  net_weights=net_weights_init)

    check_reversibility(model, sess, out_file=reverse_file)

    if is_chief:
        wfile = os.path.join(model.log_dir, 'dynamics_weights.h5')
        model.dynamics.save_weights(wfile)

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

        _ = weights_hist(model.log_dir, weights=weights)

        def pkl_dump(d, pkl_file):
            with open(pkl_file, 'wb') as f:
                pickle.dump(d, f)

        pkl_dump(weights, os.path.join(model.log_dir, 'weights.pkl'))
        pkl_dump(model.params, os.path.join(os.getcwd(), 'params.pkl'))

        # Count all trainable paramters and write them (w/ shapes) to txt file
        count_trainable_params(os.path.join(params['log_dir'],
                                            'trainable_params.txt'))

    # close MonitoredTrainingSession and reset the default graph
    sess.close()
    tf.reset_default_graph()
    io.log(f'{SEP_STR}\nTraining took: {time.time()-t0:.3g}s\n{SEP_STR}')

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
