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
from trainers.train_setup import (check_reversibility, count_trainable_params,
                                  create_config, get_net_weights, train_setup)

import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug  # noqa: F401
from tensorflow.python.client import timeline  # noqa: F401

import inference
import utils.file_io as io

from trainers.trainer import Trainer
from utils.parse_args import parse_args
from models.gauge_model import GaugeModel
from plotters.plot_utils import weights_hist
from loggers.train_logger import TrainLogger

if cfg.HAS_COMET:
    from comet_ml import Experiment

if cfg.HAS_HOROVOD:
    import horovod.tensorflow as hvd

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)

SEP_STR = 80 * '-'  # + '\n'

NP_FLOAT = cfg.NP_FLOAT


Weights = namedtuple('Weights', ['w', 'b'])


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
    sess = tf.train.MonitoredTrainingSession(**sess_kwargs)
    tf.keras.backend.set_session(sess)
    sess.run([global_var_init, local_var_init])
    uninited_out = sess.run(uninited)
    io.log(f'tf.report_uninitialized_variables() len = {uninited_out}')
    sess.run(model.dynamics.xnet.generic_net.coeff_scale.initializer)
    sess.run(model.dynamics.xnet.generic_net.coeff_transformation.initializer)
    sess.run(model.dynamics.vnet.generic_net.coeff_scale.initializer)
    sess.run(model.dynamics.vnet.generic_net.coeff_transformation.initializer)

    # Check reversibility and write results out to `.txt` file.
    reverse_str, x_diff, v_diff = check_reversibility(model, sess,
                                                      net_weights_init)
    reverse_file = os.path.join(model.log_dir, 'reversibility_test.txt')
    io.log_and_write(reverse_str, reverse_file)

    if is_chief:
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
        #  _ = weights_hist(weights, model.log_dir)

        weights_file = os.path.join(model.log_dir, 'weights.pkl')
        with open(weights_file, 'wb') as f:
            pickle.dump(weights, f)

        params_file = os.path.join(os.getcwd(), 'params.pkl')
        with open(params_file, 'wb') as f:
            pickle.dump(model.params, f)

        # Count all trainable paramters and write them (w/ shapes) to txt file
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
