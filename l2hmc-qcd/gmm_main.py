"""
gmm_main.py

Main method implementing the training phase of the L2HMC algorithm for a 2D
Gaussian Mixture Model.

Author: Sam Foreman (github: @saforem2)
Date: 09/18/2019
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import pickle

import numpy as np
import tensorflow as tf
from collections import namedtuple

import utils.file_io as io

from update import set_precision
from dynamics.dynamics import Dynamics
from params.gmm_params import GMM_PARAMS
from plotters.plot_utils import _gmm_plot
from loggers.train_logger import TrainLogger
from utils.distributions import GMM, gen_ring
from models.gmm_model import GaussianMixtureModel
from trainers.gmm_trainer import GaussianMixtureModelTrainer
from main import train_setup, create_config, count_trainable_params
from config import (
    GLOBAL_SEED, NP_FLOAT, HAS_HOROVOD, HAS_COMET, HAS_MATPLOTLIB
)

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt

if HAS_HOROVOD:
    import horovod.tensorflow as hvd

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)

SEP_STR = 80 * '-'

tf.set_random_seed(GLOBAL_SEED)


def create_session(config, checkpoint_dir, monitored=False):
    if monitored:
        sess_kwargs = {
            'checkpoint_dir': checkpoint_dir,
            'hooks': [],
            'config': config,
            'save_summaries_secs': None,
            'save_summaries_steps': None,
        }

        return tf.train.MonitoredTrainingSession(**sess_kwargs)

    return tf.Session(config=config)


def plot_target_distribution(distribution, target_samples=None, **kwargs):
    fig, ax = plt.subplots()
    ax = _gmm_plot(distribution, target_samples, **kwargs)

    return fig, ax


def train_l2hmc(FLAGS, log_file=None, experiment=None):
    """Create, and train `GaussianMixtureModel` via the L2HMC algorithm."""
    tf.keras.backend.set_learning_phase(True)
    params, hooks = train_setup(FLAGS, log_file)

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
    io.log('L2HMC_PARAMETERS:')
    for key, val in params.items():
        io.log(f'  {key}: {val}')
    io.log(SEP_STR)

    model = GaussianMixtureModel(params)
    target_samples = model.distribution.get_samples(int(1e6))

    if is_chief:
        figs_dir = os.path.join(model.log_dir, 'figures')
        io.check_else_make_dir(figs_dir)

        out_file = os.path.join(figs_dir, 'target_distribution.pdf')
        kwargs = {
            'out_file': out_file,
            'fill': False,
            'ellipse': False,
            'title': 'Target distribution of GMM',
            'ls': '',
            'axis_scale': 'scaled'
        }
        _, _ = plot_target_distribution(model.distribution,
                                        target_samples, **kwargs)

        train_logger = TrainLogger(model, params['log_dir'],
                                   logging_steps=10,
                                   summaries=params['summaries'])

        config, params = create_config(params)
        checkpoint_dir = os.path.join(model.log-dir, 'checkpoints')
        io.check_else_make_dir(checkpoint_dir)
        sess = create_session(config, checkpoint_dir, monitored=True)
        tf.keras.backend.set_session(sess)

    samples_init = np.random.randn(*model.x.shape)
    feed_dict = {
        model.x: samples_init,
        model.beta: 1.,
        model.net_weights[0]: 1.,
        model.net_weights[1]: 1.,
        model.net_weights[2]: 1.,
        model.train_phase: False
    }

    # Check reversibility
    x_diff, v_diff = sess.run([model.x_diff,
                               model.v_diff], feed_dict=feed_dict)
    io.log(f'Reversibility results:\n \t{x_diff:.5g}, {v_diff:.5g}')

    # TRAINING
    trainer = GaussianMixtureModelTrainer(sess, model, logger=train_logger)

    train_kwargs = {
        'samples_np': np.random.randn(*model.x.shape),
        'beta_np': model.beta_init,
        'net_weights': [1., 1., 1.],
        'print_steps': 1.,
    }

    train_steps = getattr(FLAGS, 'train_steps', 5000)
    t0 = time.time()
    trainer.train(train_steps, **train_kwargs)

    io.log(SEP_STR)
    io.log(f'Training completed in: {time.time() - t0:.4g}s')
    io.log(SEP_STR)

    params_file = os.path.join(os.getcwd(), 'params.pkl')
    with open(params_file, 'wb') as f:
        pickle.dump(model.params, f)

    count_trainable_params(os.path.join(params['log_dir'],
                                        'trainable_params.txt'))

    sess.close()
    tf.reset_default_graph()

    return model, train_logger



def main(FLAGS):
    log_file = 'output_dirs.txt'

    USING_HVD = getattr(FLAGS, 'horovod', False)
    if HAS_HOROVOD and USING_HVD:
        io.log('INFO: USING HOROVOD FOR DISTRIBUTED TRAINING')
        hvd.init()

    if FLAGS.hmc:
        inference.run_hmc(FLAGS, log_file=log_file)
    else:
        model, train_logger = train_l2hmc(FLAGS, log_file)


if __name__ == '__main__':
    FLAGS = GMM_PARAMS

    args = parse_gmm_args()
    try:
        FLAGS.update(args.__dict__)
    except AttributeError:
        import pudb
        pudb.set_trace()

    t0 = time.time()
    main(FLAGS)
    io.log('\n\n' + SEP_STR)
    io.log(f'Time to complete: {time.time() - t0:.4g}s')
    io.log(SEP_STR)
