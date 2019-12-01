"""
gmm_main.py

Main method implementing the training phase of the L2HMC algorithm for a 2D
Gaussian Mixture Model.

Author: Sam Foreman (github: @saforem2)
Date: 09/18/2019
"""
from __future__ import absolute_import, division, print_function

import os
import time
import pickle

import config as cfg
from seed_dict import seeds, xnet_seeds, vnet_seeds
from main import (count_trainable_params, create_config, train_setup,
                  check_reversibility, get_net_weights)
from models.gmm_model import GaussianMixtureModel
from plotters.plot_utils import _gmm_plot
from loggers.train_logger import TrainLogger
#  from trainers.gmm_trainer import GaussianMixtureModelTrainer
from trainers.trainer import Trainer

import numpy as np
import tensorflow as tf

import utils.file_io as io

from params.gmm_params import GMM_PARAMS
from utils.parse_gmm_args import parse_args as parse_gmm_args

#  from utils.distributions import gen_ring, GMM

if cfg.HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt

if cfg.HAS_HOROVOD:
    import horovod.tensorflow as hvd

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)

SEP_STR = 80 * '-'

#  tf.set_random_seed(cfg.GLOBAL_SEED)


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


def save_distribution_params(distribution, out_dir):
    mus = distribution.mus
    sigmas = distribution.sigmas
    pis = distribution.pis

    mus_file = os.path.join(out_dir, 'mus.pkl')
    sigmas_file = os.path.join(out_dir, 'sigmas.pkl')
    pis_file = os.path.join(out_dir, 'pis.pkl')

    def _save(data, name, out_file):
        with open(out_file, 'wb') as f:
            pickle.dump(data, f)
        io.log(f'INFO: `{name}` saved to {out_file}.')

    _save(mus, 'means', mus_file)
    _save(sigmas, 'sigmas', sigmas_file)
    _save(pis, 'pis', pis_file)


def train_l2hmc(FLAGS, log_file=None):
    """Create, and train `GaussianMixtureModel` via the L2HMC algorithm."""
    tf.keras.backend.set_learning_phase(True)
    params, hooks = train_setup(FLAGS, log_file,
                                root_dir='gmm_logs',
                                run_str=True,
                                model_type='GaussianMixtureModel')

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

        weights = get_net_weights(model)
        xnet = model.dynamics.x_fn.generic_net
        vnet = model.dynamics.v_fn.generic_net
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

        train_logger = TrainLogger(model, params['log_dir'],
                                   logging_steps=10,
                                   summaries=params['summaries'])

        config, params = create_config(params)
        checkpoint_dir = os.path.join(model.log_dir, 'checkpoints')
        io.check_else_make_dir(checkpoint_dir)
        sess = create_session(config, checkpoint_dir, monitored=True)
        tf.keras.backend.set_session(sess)

    net_weights_init = cfg.NetWeights(x_scale=1.,
                                      x_translation=1.,
                                      x_transformation=1.,
                                      v_scale=1.,
                                      v_translation=1.,
                                      v_transformation=1.)
    beta_init = model.beta_init
    samples_init = np.random.randn(*model.x.shape)

    # Check reversibility
    reverse_str, x_diff, v_diff = check_reversibility(model, sess)
    reverse_file = os.path.join(model.log_dir, 'reversibility_test.txt')
    io.log_and_write(reverse_str, reverse_file)

    if is_chief:
        io.save_dict(seeds, out_dir=model.log_dir, name='seeds')
        io.save_dict(xnet_seeds, out_dir=model.log_dir, name='xnet_seeds')
        io.save_dict(vnet_seeds, out_dir=model.log_dir, name='xnet_seeds')

    # TRAINING
    trainer = Trainer(sess, model, train_logger, **params)

    train_kwargs = {
        'samples': samples_init,
        'beta': beta_init,
        'net_weights':  net_weights_init,
    }

    train_steps = FLAGS.get('train_steps', 5000)
    t0 = time.time()
    trainer.train(train_steps, **train_kwargs)

    reverse_str, x_diff, v_diff = check_reversibility(model, sess)
    io.log_and_write(reverse_str, reverse_file)

    io.log(SEP_STR)
    io.log(f'Training completed in: {time.time() - t0:.4g}s')
    io.log(SEP_STR)

    if is_chief:
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

        weights_file = os.path.join(model.log_dir, 'weights.pkl')
        with open(weights_file, 'wb') as f:
            pickle.dump(weights, f)

        if train_logger is not None:
            save_distribution_params(model.distribution, train_logger.log_dir)

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

    USING_HVD = FLAGS.get('horovod', False)
    if cfg.HAS_HOROVOD and USING_HVD:
        io.log('INFO: USING HOROVOD FOR DISTRIBUTED TRAINING')
        hvd.init()

    model, train_logger = train_l2hmc(FLAGS, log_file)


if __name__ == '__main__':
    FLAGS = GMM_PARAMS
    args = parse_gmm_args()
    #  set_seed(args.global_seed)

    #  io.log('\n' + 80 * '=' + '\n')
    #  io.log(f'config.TF_FLOAT: {cfg.TF_FLOAT}')
    #  if args.float64:
    #      set_precision('float64')

    #  io.log(f'config.TF_FLOAT: {cfg.TF_FLOAT}')
    #  io.log('\n' + 80 * '=' + '\n')

    FLAGS.update(args.__dict__)
    t0 = time.time()
    main(FLAGS)
    io.log('\n\n' + SEP_STR)
    io.log(f'Time to complete: {time.time() - t0:.4g}s')
    io.log(SEP_STR)
