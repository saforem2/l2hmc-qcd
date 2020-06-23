"""
gauge_model.py

Implements `GaugeModel` class.
"""
from __future__ import absolute_import, division, print_function

import time

from collections import namedtuple

import numpy as np
import tensorflow as tf

from utils.learning_rate import WarmupExponentialDecay


import utils.file_io as io

from config import PI, TF_FLOAT, TF_INT

from lattice.lattice import GaugeLattice
from lattice.utils import u1_plaq_exact_tf
from utils.attr_dict import AttrDict
from dynamics.dynamics import Dynamics

if tf.__version__.startswith('1.'):
    TF_VERSION = '1.x'
elif tf.__version__.startswith('2.'):
    TF_VERSION = '2.x'


#  from utils.horovod_utils import warmup_lr
#  if HAS_HOROVOD:
try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


NAMES = [
    'STEP', 'dt', 'LOSS', 'px', 'eps', 'BETA', 'sumlogdet', 'dQ', 'plaq_err',
]
HSTR = ''.join(["{:^12s}".format(name) for name in NAMES])
SEP = '-' * len(HSTR)
HEADER = '\n'.join([SEP, HSTR, SEP])

RUN_NAMES = [
    'STEP', 'dt', 'px', 'sumlogdet', 'dQ', 'plaq_err',
]
RUN_HSTR = ''.join(["{:^12s}".format(name) for name in RUN_NAMES])
RUN_SEP = '-' * len(RUN_HSTR)
RUN_HEADER = '\n'.join([RUN_SEP, RUN_HSTR, RUN_SEP])

#  HEADER = SEP + '\n' + HSTR + '\n' + SEP

lfData = namedtuple('lfData', ['init', 'proposed', 'prob'])


def exp_mult_cooling(step, temp_init, temp_final, num_steps, alpha=None):
    """Annealing function."""
    if alpha is None:
        alpha = tf.exp(
            (tf.math.log(temp_final) - tf.math.log(temp_init)) / num_steps
        )
        #  alpha = tf.exp(tf.math.log(temp_final) - tf.math.log(temp_init))
        #  alpha = np.exp((np.log(temp_final) - np.log(temp_init)) / num_steps)

    temp = temp_init * (alpha ** step)

    return tf.cast(temp, TF_FLOAT)


def get_betas(steps, beta_init, beta_final):
    """Get array of betas to use in annealing schedule."""
    t_init = 1. / beta_init
    t_final = 1. / beta_final
    t_arr = [
        exp_mult_cooling(i, t_init, t_final, steps) for i in range(steps)
    ]

    return 1. / tf.convert_to_tensor(np.array(t_arr))


# pylint:disable=invalid-name
class GaugeModel:
    """Implements `GaugeModel`, a convenience wrapper around `Dynamics`."""

    def __init__(self, params, dynamics_config, net_config):
        self._model_type = 'GaugeModel'
        self.params = params
        self.net_config = net_config
        self.dynamics_config = dynamics_config

        ######################################################################
        # NOTE:
        # -----
        # If either eps is trainable or we're not running HMC,
        # there exist parameters to be trained, so we should create the
        # optimizer and associated training step function.
        #
        # This is equivalent to saying that if eps is not trainable, and
        # dynamics_config.hmc is True, there are no trainable parameters, so
        # don't bother creating optimizer.
        ######################################################################
        self._has_trainable_params = True
        if self.dynamics_config.hmc and not self.dynamics_config.eps_trainable:
            self._has_trainable_params = False

        self.parse_params(params)

        self.lattice = GaugeLattice(self.time_size,
                                    self.space_size,
                                    self.lattice_dim,
                                    batch_size=self.batch_size)
        self.potential_fn = self.lattice.calc_actions

        self.dynamics = Dynamics(self.potential_fn,
                                 dynamics_config, net_config,
                                 separate_nets=self.separate_networks)

        if self._has_trainable_params:
            self.lr = self.create_lr(warmup=self.warmup_lr)
            self.optimizer = self.create_optimizer()
            self.dynamics.compile(
                optimizer=self.optimizer,
                loss=self.calc_loss,
            )

        if not tf.executing_eagerly or TF_VERSION == '1.x':
            self._build()

    def parse_params(self, params):
        """Set instance attributes from `params`."""
        self.params = AttrDict(params)

        lattice_shape = params.get('lattice_shape', None)
        if lattice_shape is None:
            batch_size = params.get('batch_size', None)
            time_size = params.get('time_size', None)
            space_size = params.get('space_size', None)
            dim = params.get('dim', 2)
            lattice_shape = (batch_size, time_size, space_size, dim)
        else:
            batch_size, time_size, space_size, dim = lattice_shape
            batch_size = lattice_shape[0]
            time_size = lattice_shape[1]
            space_size = lattice_shape[2]
            dim = lattice_shape[3]

        self.lattice_shape = lattice_shape
        self.lattice_dim = dim
        self.time_size = time_size
        self.batch_size = batch_size
        self.space_size = space_size
        self.xdim = time_size * space_size * dim
        self.input_shape = (batch_size, self.xdim)

        self.run_steps = params.get('run_steps', int(1e3))
        self.print_steps = params.get('print_steps', 10)
        self.logging_steps = params.get('logging_steps', 50)
        self.save_run_data = params.get('save_run_data', True)
        self.separate_networks = params.get('separate_networks', False)

        eager_execution = params.get('eager_execution', False)
        self.compile = not eager_execution

        if self._has_trainable_params:
            self.lr_init = params.get('lr_init', None)
            self.warmup_lr = params.get('warmup_lr', False)
            self.warmup_steps = params.get('warmup_steps', None)
            self.using_hvd = params.get('horovod', False)
            self.lr_decay_steps = params.get('lr_decay_steps', None)
            self.lr_decay_rate = params.get('lr_decay_rate', None)
            self.plaq_weight = params.get('plaq_weight', 0.)
            self.charge_weight = params.get('charge_weight', 0.)
            self.train_steps = params.get('train_steps', int(1e4))
            self.save_train_data = params.get('save_train_data', True)
            self.save_steps = params.get('save_steps', self.train_steps // 10)

            self.beta_init = params.get('beta_init', None)
            self.beta_final = params.get('beta_final', None)
            beta = params.get('beta', None)
            if self.beta_init == self.beta_final or beta is not None:
                self.beta = self.beta_init
                self.betas = tf.convert_to_tensor(
                    tf.cast(self.beta * np.ones(self.train_steps),
                            dtype=TF_FLOAT)
                )
            else:
                if self.train_steps is not None:
                    self.betas = get_betas(self.train_steps,
                                           self.beta_init,
                                           self.beta_final)

    # pylint:disable=attribute-defined-outside-init
    def _build(self):
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        inputs = self._build_inputs()
        self.x = inputs['x']
        self.beta = inputs['beta']
        self.eps_ph = inputs['eps_ph']
        self.global_step_ph = inputs['global_step_ph']

        plaqs_err, charges = self.calc_observables(self.x, self.beta)
        self.plaqs_err = plaqs_err
        self.charges = charges

        loss, x_out, px, sumlogdet = self.train_step(self.x, self.beta,
                                                     self.global_step == 0)
        self.px = px
        self.loss = loss
        self.x_out = x_out
        self.sumlogdet = sumlogdet

    def calc_loss(self, x_init, x_prop, accept_prob):
        """Calculate the total loss."""
        ps_init = self.lattice.calc_plaq_sums(samples=x_init)
        ps_prop = self.lattice.calc_plaq_sums(samples=x_prop)

        plaq_loss = 0.
        if self.plaq_weight > 0:
            dplaq = 1. - tf.math.cos(ps_prop - ps_init)
            ploss = accept_prob * tf.reduce_sum(dplaq, axis=(1, 2))
            plaq_loss = tf.reduce_mean(-ploss / self.plaq_weight, axis=0)

        charge_loss = 0.
        if self.charge_weight > 0:
            q_init = self.lattice.calc_top_charges(plaq_sums=ps_init)
            q_prop = self.lattice.calc_top_charges(plaq_sums=ps_prop)
            qloss = accept_prob * (q_prop - q_init) ** 2
            charge_loss = tf.reduce_mean(-qloss / self.charge_weight, axis=0)

        total_loss = plaq_loss + charge_loss

        return total_loss

    def calc_observables(self, x, beta, use_sin=True):
        """Calculate observables."""
        ps = self.lattice.calc_plaq_sums(x)
        plaqs = self.lattice.calc_plaqs(plaq_sums=ps)
        charges = self.lattice.calc_top_charges(plaq_sums=ps, use_sin=use_sin)
        plaqs_err = u1_plaq_exact_tf(beta) - plaqs

        return plaqs_err, charges

    def _build_inputs(self):
        """Create input placeholders."""
        def make_ph(name, shape=(), dtype=TF_FLOAT):
            return tf.compat.v1.placeholder(
                dtype=dtype, shape=shape, name=name
            )

        with tf.name_scope('inputs'):
            if not tf.executing_eagerly():
                x = make_ph(dtype=TF_FLOAT, shape=self.input_shape, name='x')
                beta = make_ph('beta')
                eps_ph = make_ph('eps_ph')
                global_step_ph = make_ph('global_step_ph', dtype=tf.int64)

            inputs = {
                'x': x,
                'beta': beta,
                'eps_ph': eps_ph,
                'global_step_ph': global_step_ph,
            }

        return inputs

    def create_lr(self, warmup=False):
        """Create the learning rate schedule to be used during training."""
        if warmup:
            name = 'WarmupExponentialDecay'
            warmup_steps = self.warmup_steps
            if warmup_steps is None:
                warmup_steps = self.train_steps // 20

            return WarmupExponentialDecay(self.lr_init, self.lr_decay_steps,
                                          self.lr_decay_rate, warmup_steps,
                                          staircase=True, name=name)

        return tf.keras.optimizers.schedules.ExponentialDecay(
            self.lr_init,
            decay_steps=self.lr_decay_steps,
            decay_rate=self.lr_decay_rate,
            staircase=True
        )

    def create_optimizer(self):
        """Create the optimizer to use for backpropagation."""
        if tf.executing_eagerly():
            return tf.keras.optimizers.Adam(self.lr)

        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        if self.using_hvd:
            optimizer = hvd.DistributedOptimizer(optimizer)

        return optimizer

    def train_step(self, x, beta, first_step):
        """Perform a single training step."""
        with tf.GradientTape() as tape:
            states, px, sld_states = self.dynamics((x, beta), training=True)
            loss = self.calc_loss(states.init.x, states.proposed.x, px)

        if self.using_hvd:
            # Horovod: add Horovod Distributed GradientTape
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss, self.dynamics.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,
                                           self.dynamics.trainable_variables))
        # Horovod:
        #   Broadcast initial variable states from rank 0 to all other
        #   processes. This is necessary to ensure consistent initialization of
        #   all workers when training is started with random weights or
        #   restored from a checkpoint.
        # NOTE:
        #   Broadcast should be done after the first gradient step to ensure
        #   optimizer initialization.
        if first_step and self.using_hvd:
            hvd.broadcast_variables(self.dynamics.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        return loss, states.out.x, px, sld_states.out

    def restore_from_checkpoint(self, ckpt_dir=None):
        """Helper method for restoring from checkpoint."""
        step_init = tf.Variable(0, dtype=TF_INT)
        if ckpt_dir is not None:
            checkpoint = tf.train.Checkpoint(step=step_init,
                                             dynamics=self.dynamics,
                                             optimizer=self.optimizer)
            manager = tf.train.CheckpointManager(
                checkpoint, directory=ckpt_dir, max_to_keep=3
            )
            if manager.latest_checkpoint:
                io.log(f'Restored from: {manager.latest_checkpoint}')
                checkpoint.restore(manager.latest_checkpoint)
                step_init = checkpoint.step

        return checkpoint, manager, step_init

    def run_step(self, x, beta):
        """Perform a single inference step."""
        return self.dynamics((x, beta), training=False)
