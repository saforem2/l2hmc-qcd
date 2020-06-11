"""
gauge_model.py

Implements `GaugeModel` class.
"""
from __future__ import absolute_import, division, print_function

import os
import time
from collections import namedtuple

import numpy as np
import tensorflow as tf
from config import (
    HAS_HOROVOD,  TF_FLOAT, TF_INT, NP_FLOAT, PI, TWO_PI, NetWeights,
)
import utils.file_io as io
from utils.attr_dict import AttrDict
from base.base_model import add_to_collection
from lattice.lattice import GaugeLattice, u1_plaq_exact_tf
from dynamics.dynamics import Dynamics, DynamicsConfig
from network import NetworkConfig
from utils.horovod_utils import warmup_lr

if HAS_HOROVOD:
    import horovod.tensorflow as hvd


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
    def __init__(self, params, lattice_shape, dynamics_config, net_config):
        self._model_type = 'GaugeModel'
        self.params = params
        self.parse_params(params, lattice_shape)
        self.dynamics_config = dynamics_config
        self.net_config = net_config

        self.lattice = GaugeLattice(self.time_size,
                                    self.space_size,
                                    self.lattice_dim,
                                    batch_size=self.batch_size)
        self.potential_fn = self.lattice.calc_actions

        self.dynamics = Dynamics(self.potential_fn,
                                 dynamics_config, net_config,
                                 separate_nets=self.separate_networks)

        self.lr = self.create_lr(warmup=self.warmup_lr)
        self.optimizer = self.create_optimizer()
        self.dynamics.compile(
            optimizer=self.optimizer,
            loss_fn=self.calc_loss,
        )
        self.betas = get_betas(self.train_steps,
                               self.beta_init,
                               self.beta_final)

        if not tf.executing_eagerly():
            self._build()

    def parse_params(self, params, lattice_shape):
        """Set instance attributes from `params`."""
        self.params = AttrDict(params)
        self.separate_networks = params.separate_networks
        self.lr_init = params.lr_init
        self.warmup_lr = params.warmup_lr
        self.using_hvd = params.horovod
        self.lr_decay_steps = params.lr_decay_steps
        self.lr_decay_rate = params.lr_decay_rate
        self.plaq_weight = params.plaq_weight
        self.charge_weight = params.charge_weight
        self.train_steps = params.train_steps
        self.logging_steps = params.logging_steps
        self.print_steps = params.print_steps
        self.beta_init = params.beta_init
        self.beta_final = params.beta_final
        batch_size, time_size, space_size, dim = lattice_shape
        self.batch_size = batch_size
        self.time_size = time_size
        self.space_size = space_size
        self.lattice_dim = dim
        self.xdim = time_size * space_size * dim
        self.lattice_shape = lattice_shape
        self.input_shape = (batch_size, self.xdim)
        self.save_steps = params.get('save_steps', self.train_steps // 4)

    # pylint:disable=attribute-defined-outside-init
    def _build(self):
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        inputs = self._build_inputs()
        self.x = inputs['x']
        self.beta = inputs['beta']
        self.eps_ph = inputs['eps_ph']
        #  self.train_phase = inputs['train_phase']
        #  self.net_weights = inputs['net_weights']
        self.global_step_ph = inputs['global_step_ph']

        plaqs_err, charges = self.calc_observables(self.x, self.beta)
        self.plaqs_err = plaqs_err
        self.charges = charges

        loss, x_out, px, sumlogdet = self.train_step(self.x, self.beta,
                                                     self.global_step == 0)
        self.loss = loss
        self.x_out = x_out
        self.px = px
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

    def calc_observables(self, x, beta):
        """Calculate observables."""
        ps = self.lattice.calc_plaq_sums(x)
        plaqs = self.lattice.calc_plaqs(plaq_sums=ps)
        charges = self.lattice.calc_top_charges(plaq_sums=ps)
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
                #  xsw = make_ph('x_scale_weight')
                #  xtw = make_ph('x_translation_weight')
                #  xqw = make_ph('x_transformation_weight')
                #  vsw = make_ph('v_scale_weight')
                #  vtw = make_ph('v_translation_weight')
                #  vqw = make_ph('v_transformation_weight')
                #  net_weights = NetWeights(xsw, xtw, xqw, vsw, vtw, vqw)
                #  train_phase = make_ph('is_training', dtype=tf.bool)

            inputs = {
                'x': x,
                'beta': beta,
                'eps_ph': eps_ph,
                'global_step_ph': global_step_ph,
                #  'train_phase': train_phase,
                #  'net_weights': net_weights,
            }

        return inputs

    def create_lr(self, warmup=False):
        """Create the learning rate schedule to be used during training."""
        if tf.executing_eagerly():
            return tf.keras.optimizers.schedules.ExponentialDecay(
                self.lr_init,
                decay_steps=self.lr_decay_steps,
                decay_rate=self.lr_decay_rate,
                staircase=True
            )

        if warmup:
            return warmup_lr(target_lr=self.lr_init,
                             warmup_steps=int(0.1 * self.train_steps),
                             global_step=self.global_step,
                             decay_steps=self.lr_decay_steps,
                             decay_rate=self.lr_decay_rate)

        return tf.compat.v1.train.exponential_decay(
            self.lr_init,
            self.global_step,
            self.lr_decay_steps,
            self.lr_decay_rate,
            staircase=True,
            name='learning_rate'
        )

    def create_optimizer(self):
        """Create the optimizer to use for backpropagation."""
        if tf.executing_eagerly():
            return tf.keras.optimizers.Adam(self.lr)

        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        if self.using_hvd:
            optimizer = hvd.DistributedOptimizer(optimizer)

        return optimizer

    def run_step(self, x, beta):
        """Perform a single inference step."""
        states, px, sld_states = self.dynamics((x, beta), training=False)


    def train_step(self, x, beta, first_step):
        """Perform a single training step."""
        with tf.GradientTape() as tape:
            states, px, sld_states = self.dynamics((x, beta), training=True)
            loss = self.calc_loss(states.init.x, states.proposed.x, px)

        if self.using_hvd:
            # Horovod: add Horovod Distributed GradientTape
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss, self.dynamics.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,
                                           self.dynamics.trainable_weights))
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
        step_init = tf.Variable(0, dtype=TF_INT)
        if ckpt_dir is not None:
            #  checkpoint = tf.train.Checkpoint(model=self.dynamics,
            kwargs = {}
            iterator = enumerate(zip(self.dynamics.xnets, self.dynamics.vnets))
            for idx, (xnet, vnet) in iterator:
                kwargs[f'xnet{idx}'] = xnet
                kwargs[f'vnet{idx}'] = vnet

            checkpoint = tf.train.Checkpoint(step=step_init,
                                             dynamics=self.dynamics,
                                             optimizer=self.optimizer,
                                             **kwargs)
            manager = tf.train.CheckpointManager(
                checkpoint, directory=ckpt_dir, max_to_keep=3
            )
            if manager.latest_checkpoint:
                io.log(f'Restored from: {manager.latest_checkpoint}')
                checkpoint.restore(manager.latest_checkpoint)
                step_init = checkpoint.step

        return checkpoint, manager, step_init

    def run_eager(self, run_steps, beta, x=None,
                  save_run_data=False, ckpt_dir=None):
        """Run inference using eager execution."""
        if x is None:
            x = tf.random.uniform(shape=self.input_shape,
                                  minval=-PI, maxval=PI)
            x = tf.cast(x, dtype=TF_FLOAT)

        _, q_new = self.calc_observables(x, self.beta_init)

        px_arr = []
        dq_arr = []
        data_strs = [RUN_HEADER]
        charges_arr = [q_new.numpy()]
        if ckpt_dir is not None:
            _, _, _ = self.restore_from_checkpoint(ckpt_dir)

        if not self.separate_networks:
            dynamics_call = tf.function(self.dynamics)
        else:
            dynamics_call = self.dynamics

        io.log(RUN_SEP)
        io.log(f'Running inference on trained model at with:')
        io.log(f'  beta: {beta}')
        io.log(f'  dynamics.eps: {self.dynamics.eps.numpy():.4g}')
        io.log(RUN_SEP)
        io.log(RUN_HEADER)
        for step in np.arange(run_steps):
            t0 = time.time()
            x = tf.reshape(x, self.input_shape)
            states, px, sld_states = dynamics_call((x, beta), training=False)
            #  states, px, sld_states = self.dynamics((x, beta),
            #                                         training=False)
            x = states.out.x
            sld = sld_states.out
            x = tf.reshape(x, self.lattice_shape)
            dt = time.time() - t0

            q_old = q_new
            plaqs_err, q_new = self.calc_observables(x, beta)
            dq = tf.math.abs(q_new - q_old)

            data_str = (
                f"{step:>6g}/{run_steps:<6g} "
                f"{dt:^11.4g} "
                f"{np.mean(px.numpy()):^11.4g} "
                f"{np.mean(sld.numpy()):^11.4g} "
                f"{np.mean(dq.numpy()):^11.4g} "
                f"{np.mean(plaqs_err.numpy()):^11.4g} "
            )

            if step % self.print_steps == 0:
                io.log(data_str)
                data_strs.append(data_str)

            if save_run_data:
                px_arr.append(px.numpy())
                dq_arr.append(dq.numpy())
                charges_arr.append(q_new.numpy())

            if step % 100 == 0:
                io.log(RUN_HEADER)

        outputs = {
            'px': px_arr,
            'dq': dq_arr,
            'charges_arr': charges_arr,
            'x': tf.reshape(x, self.input_shape),
        }

        data_strs.append(RUN_HEADER)

        return outputs, data_strs

    def train_eager(self, x=None, save_train_data=False, ckpt_dir=None):
        if x is None:
            x = tf.random.uniform(shape=self.input_shape,
                                  minval=-PI, maxval=PI)
            x = tf.cast(x, dtype=TF_FLOAT)

        _, q_new = self.calc_observables(x, self.beta_init)

        px_arr = []
        dq_arr = []
        loss_arr = []
        data_strs = [HEADER]
        charges_arr = [q_new.numpy()]
        step_init = tf.Variable(0, dtype=TF_INT)
        if ckpt_dir is not None:
            ckpt, manager, step_init = self.restore_from_checkpoint(ckpt_dir)
        #  self.observables['charges'].append(charges.numpy())
        train_steps = np.arange(self.train_steps)
        step = int(step_init.numpy())
        betas = self.betas[step:]
        steps = train_steps[step:]

        if self.separate_networks:
            train_step_fn = tf.function(self.train_step)
        else:
            train_step_fn = self.train_step

        io.log(HEADER)
        for step, beta in zip(steps, betas):
            t0 = time.time()
            x = tf.reshape(x, self.input_shape)
            #  loss, x, px, sld = self.train_step(x, beta, step == 0)
            loss, x, px, sld = train_step_fn(x, beta, step == 0)
            x = tf.reshape(x, self.lattice_shape)
            dt = time.time() - t0

            q_old = q_new
            plaqs_err, q_new = self.calc_observables(x, beta)
            dq = tf.math.abs(q_new - q_old)

            data_str = (
                f"{step:>6g}/{self.train_steps:<6g} "
                f"{dt:^11.4g} "
                f"{loss.numpy():^11.4g} "
                f"{np.mean(px.numpy()):^11.4g} "
                f"{self.dynamics.eps.numpy():^11.4g} "
                f"{beta:^11.4g} "
                f"{np.mean(sld.numpy()):^11.4g} "
                f"{np.mean(dq.numpy()):^11.4g} "
                f"{np.mean(plaqs_err.numpy()):^11.4g} "
            )

            if step % self.print_steps == 0:
                io.log(data_str)
                data_strs.append(data_str)

            if save_train_data and step % self.logging_steps == 0:
                px_arr.append(px.numpy())
                dq_arr.append(dq.numpy())
                loss_arr.append(loss.numpy())
                charges_arr.append(q_new.numpy())

            if step % self.save_steps == 0 and ckpt_dir is not None:
                ckpt.step.assign(step)
                manager.save()
                #  checkpoint.save(file_prefix=ckpt_prefix)

            if step % 100 == 0:
                io.log(HEADER)

        if ckpt_dir is not None:
            manager.save()
            #  checkpoint.save(file_prefix=ckpt_prefix)

        outputs = {
            'px': px_arr,
            'dq': dq_arr,
            'loss_arr': loss_arr,
            'charges_arr': charges_arr,
            'x': tf.reshape(x, self.input_shape),
            #  'data_strs': data_strs,
        }

        data_strs.append(HEADER)

        return outputs, data_strs
