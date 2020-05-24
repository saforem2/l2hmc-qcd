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

from collections import namedtuple

import numpy as np
import tensorflow as tf

# pylint:disable=import-error, unused-import
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline

import config as cfg
import utils.file_io as io

from config import NetWeights, Weights, PI, TWO_PI
from seed_dict import seeds, vnet_seeds, xnet_seeds
from models.gauge_model import GaugeModel
from loggers.train_logger import TrainLogger
from utils.file_io import timeit
from utils.attr_dict import AttrDict
from utils.parse_args import parse_args
from plotters.plot_utils import plot_singular_values, weights_hist
from plotters.train_plots import plot_train_data
from trainers.trainer import Trainer
from trainers.train_setup import count_trainable_params, create_config

if cfg.HAS_HOROVOD:
    import horovod.tensorflow as hvd  # pylint: disable=import-error

try:
    tf.logging.set_verbosity(tf.logging.DEBUG)
except AttributeError:
    pass

#  os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

SEP_STR = 80 * '-'  # + '\n'

# pylint:disable=too-many-statements
# pylint:disable=import-error
# pylint:disable=unused-import
# pylint:disable=too-many-statements
# pylint:disable=no-name-in-module, invalid-name
# pylint:disable=redefined-outer-name

def log_params(params):
    io.log(SEP_STR + '\nL2HMC PARAMETERS:\n')
    for key, val in params.items():
        io.log(f' - {key} : {val}\n')
    io.log(SEP_STR)


def create_sess(FLAGS, **sess_kwargs):
    """Create `tf.train.MonitoredTrainingSession`."""
    sess_kwargs.update({
        'config': create_config(FLAGS),
        'save_checkpoint_steps': FLAGS.save_steps,
    })
    global_var_init = tf.compat.v1.global_variables_initializer()
    local_var_init = tf.compat.v1.local_variables_initializer()
    uninited = tf.compat.v1.report_uninitialized_variables()
    sess = tf.compat.v1.train.MonitoredTrainingSession(**sess_kwargs)
    tf.compat.v1.keras.backend.set_session(sess)
    sess.run([global_var_init, local_var_init])
    uninited_out = sess.run(uninited)
    io.log(f'tf.report_uninitialized_variables() len = {uninited_out}')

    return sess


def _get_global_var(name):
    try:
        var = [i for i in tf.global_variables() if name in i.name][0]
    except IndexError:
        var = None
    return var


def get_global_vars(names):
    """Get dictionary of all global variables that are not None."""
    global_vars = {name: _get_global_var(name) for name in names}
    for k, v in global_vars:
        if v is None:
            _ = global_vars.pop(k)
    return global_vars


def save_params(model):
    """Save model parameters to `.z` files.

    Additionally, write out all trainable parameters (w/ sizes) to `.txt` file.
    """
    out_file = os.path.join(model.log_dir, 'trainable_params.txt')
    count_trainable_params(out_file)

    model.params['network_type'] = model.network_type
    io.savez(dict(model.params), os.path.join(os.getcwd(), 'params.z'))
    io.savez(dict(model.params), os.path.join(model.log_dir, 'params.z'))


def save_masks(model, sess):
    """Save `model.dynamics.masks` for inference."""
    masks_file = os.path.join(model.log_dir, 'dynamics_mask.z')
    masks_file_ = os.path.join(model.log_dir, 'dynamics_mask.np')
    masks = sess.run(model.dynamics.masks)
    np.array(masks).tofile(masks_file_)
    io.log(f'dynamics.masks:\n\t {masks}')
    io.savez(masks, masks_file)


def save_seeds(model):
    """Save network seeds for reproducibility."""
    io.save_dict(seeds, out_dir=model.log_dir, name='seeds')
    io.save_dict(xnet_seeds, out_dir=model.log_dir, name='xnet_seeds')
    io.save_dict(vnet_seeds, out_dir=model.log_dir, name='vnet_seeds')


def save_weights(model, sess):
    """Save network weights to `.z` file."""
    xw_file = os.path.join(model.log_dir, 'xnet_weights.z')
    vw_file = os.path.join(model.log_dir, 'vnet_weights.z')
    w_file = os.path.join(model.log_dir, 'dynamics_weights.z')

    xnet_weights = model.dynamics.xnet.save_weights(sess, xw_file)
    vnet_weights = model.dynamics.vnet.save_weights(sess, vw_file)

    weights = {
        'xnet': xnet_weights,
        'vnet': vnet_weights,
    }
    io.savez(weights, w_file, name='dynamics_weights')

    return weights


def save_eps(model, sess):
    """Save final value of `eps` (step size) at the end of training."""
    eps_np = sess.run(model.dynamics.eps)
    eps_dict = {'eps': eps_np}
    io.savez(eps_dict, os.path.join(model.log_dir, 'eps_np.z'))

    return eps_np


def restore_state_and_params(log_dir):
    """Returns previous training state and updated params from `log_dir`."""
    params = io.loadz(os.path.join(log_dir, 'params.z'))
    state = io.loadz(os.path.join(log_dir, 'training', 'current_state.z'))
    params['lr_init'] = state['lr']
    params['beta_init'] = state['beta']

    return state, AttrDict(params)


@timeit
def build_model(FLAGS, log_file=None):
    """Build `GaugeModel` object by parsing params from FLAGS."""
    tf.keras.backend.set_learning_phase(True)
    log_dir = FLAGS.log_dir
    if log_dir is None:
        log_dir = io.create_log_dir(FLAGS, 'GaugeModel', log_file)

    FLAGS.update({
        'log_dir': log_dir,
        'summaries': not FLAGS.no_summaries,
        'save_steps': FLAGS.train_steps // 4,
        'keep_data': FLAGS.save_train_data,
    })

    if FLAGS.restore:
        #  params = io.loadz(os.path.join(params['log_dir'], 'params.z'))
        FLAGS = AttrDict(io.loadz(os.path.join(log_dir, 'params.z')))
        state = io.loadz(os.path.join(log_dir, 'training', 'current_state.z'))
        FLAGS.update({
            'lr_init': state['lr'],
            'beta_init': state['beta'],
        })

    hooks = []
    if FLAGS.horovod:
        FLAGS.update({
            'using_hvd': True,
            'num_workers': hvd.size()
        })
        hooks += [hvd.BroadcastGlobalVariablesHook(0)]

    return GaugeModel(FLAGS), hooks, FLAGS


@timeit
def build_logger_and_ckpt_dir(model):
    """Build TrainLogger object."""
    ckpt_dir = os.path.join(model.log_dir, 'checkpoints')
    io.check_else_make_dir(ckpt_dir)
    train_logger = TrainLogger(model)

    return train_logger, ckpt_dir


def save(FLAGS, model, sess, logger, state=None):
    """Saves data from training run."""
    _ = save_eps(model, sess)
    if not FLAGS.hmc:
        _ = save_weights(model, sess)
    save_masks(model, sess)
    save_params(model)
    save_seeds(model)
    logger.write_train_strings()
    logger.save_train_data()
    if state is not None:
        fpath = os.path.join(model.log_dir, 'training', 'final_state.z')
        io.savez(state, fpath, name='final_state')


@timeit
def train(FLAGS, log_file=None):
    """Train L2HMC sampler and log/plot results."""
    IS_CHIEF = (
        not FLAGS.horovod
        or FLAGS.horovod and hvd.rank() == 0
    )

    state_final = None
    if FLAGS.restore:
        state_final, FLAGS = restore_state_and_params(FLAGS.log_dir)

    model, hooks, FLAGS = build_model(FLAGS, log_file)

    logger, ckpt_dir = None, None
    if IS_CHIEF:  # Only create logger on chief rank
        logger, ckpt_dir = build_logger_and_ckpt_dir(model)

    sess = create_sess(FLAGS,
                       hooks=hooks,
                       checkpoint_dir=ckpt_dir,
                       save_summaries_secs=None,
                       save_summaries_steps=None)

    trainer = Trainer(sess, model, logger, FLAGS)

    shape = (FLAGS.batch_size, model.lattice.num_links)
    x_init = np.random.uniform(-PI, PI, size=shape)

    nw_init = NetWeights(FLAGS.x_scale_weight,
                         FLAGS.x_translation_weight,
                         FLAGS.x_transformation_weight,
                         FLAGS.v_scale_weight,
                         FLAGS.v_translation_weight,
                         FLAGS.v_transformation_weight)

    if FLAGS.restore:
        x_init = state_final['x_in']
        restore_ops = [model.global_step_setter, model.eps_setter]
        sess.run(restore_ops, feed_dict={
            model.global_step_ph: state_final['step'],
            model.eps_ph: state_final['dynamics_eps'],
        })

    state_final = trainer.train(x=x_init,
                                net_weights=nw_init,
                                beta=FLAGS.beta_init)

    if IS_CHIEF and FLAGS.save_train_data:
        save(FLAGS, model, sess, logger, state_final)
        _ = plot_train_data(logger.train_data, FLAGS, num_chains=10)

    sess.close()
    tf.compat.v1.reset_default_graph()

    return model


if __name__ == '__main__':
    args = parse_args()
    FLAGS = AttrDict(args.__dict__)
    if not tf.executing_eagerly():
        if FLAGS.horovod:
            hvd.init()
            rank = hvd.rank()
            tf.compat.v1.set_random_seed(rank * seeds['global_tf'])
        else:
            tf.compat.v1.set_random_seed(seeds['global_tf'])

    log_file = os.path.join(os.getcwd(), 'output_dirs.txt')
    _ = train(FLAGS, log_file)
