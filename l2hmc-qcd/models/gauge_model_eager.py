"""
gauge_model_eager.py

Implements `GaugeModel` object, compatible with eager execution in tensorflow.
"""
from __future__ import absolute_import, division, print_function

import time
from collections import namedtuple
from utils.attr_dict import AttrDict

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from utils.horovod_utils import warmup_lr
import config as cfg
from config import (TF_FLOAT, NP_FLOAT, PI, TWO_PI, NET_WEIGHTS_HMC,
                    NET_WEIGHTS_L2HMC, MonteCarloStates)
import utils.file_io as io
from lattice.lattice import GaugeLattice, u1_plaq_exact_tf
from dynamics.dynamics import Dynamics, DynamicsConfig
from network import NetworkConfig

if cfg.HAS_HOROVOD:
    import horovod.tensorflow as hvd

NAMES = [
    'STEP', 'dt', 'LOSS', 'px', 'eps', 'BETA', 'sumlogdet', 'dQ', 'plaq_err',
]
HSTR = ''.join(["{:^12s}".format(name) for name in NAMES])
SEP = '-' * len(HSTR)
HEADER = SEP + '\n' + HSTR + '\n' + SEP


# pylint:disable=invalid-name

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


class GaugeModel:
    def __init__(self, params, lattice_shape, dynamics_config, net_config):
        self.params = AttrDict(params)
        self.plaq_weight = params.plaq_weight
        self.charge_weight = params.charge_weight
        self.train_steps = params.train_steps
        self.beta_init = params.beta_init
        self.beta_final = params.beta_final
        self.dynamics_config = dynamics_config
        self.net_config = net_config
        batch_size, time_size, space_size, dim = lattice_shape
        self.batch_size = batch_size
        self.time_size = time_size
        self.space_size = space_size
        self.dim = dim
        self.xdim = time_size * space_size * dim
        self.lattice_shape = lattice_shape
        self.input_shape = (batch_size, self.xdim)

        self.lattice = GaugeLattice(time_size, space_size,
                                    dim, batch_size=batch_size)
        self.potential_fn = self.lattice.calc_actions
        self.dynamics = Dynamics(self.potential_fn,
                                 dynamics_config,
                                 net_config)
        self.optimizer = self.create_optimizer()
        self.dynamics.compile(
            optimizer=self.optimizer,
            loss_fn=self.calc_loss,
        )

        self.loss_arr = []
        self.px_arr = []
        self.data_strs = []
        self.observables = {
            'plaqs_err': [],
            'charges': [],
            'dq': [],
        }
        self.betas = get_betas(self.train_steps,
                               self.beta_init,
                               self.beta_final)

    def create_learning_rate(self, warmup=False):
        """Create learning rate."""
        pass

    @staticmethod
    def create_optimizer():
        """Create optimizer."""
        optimizer = tf.keras.optimizers.Adam(5e-4)
        return optimizer

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

    def calc_observables(self, step, x, beta):
        """Calculate and update observables."""
        ps = self.lattice.calc_plaq_sums(x)
        plaqs = self.lattice.calc_plaqs(plaq_sums=ps)
        charges = self.lattice.calc_top_charges(plaq_sums=ps)
        plaqs_err = u1_plaq_exact_tf(beta) - plaqs
        #  self.observables['charges'].append(charges.numpy())
        #  self.observables['plaqs_err'].append(plaqs_err.numpy())
        #  if step > 1:
        #      #  dq = np.abs(self.observables['charges'][-1] - charges)
        #      dq = tf.math.abs(self.observables['charges'][-1] - charges)
        #      self.observables['dq'].append(dq.numpy())
        #  else:
        #      dq = tf.convert_to_tensor(np.zeros(self.batch_size))

        return plaqs_err, charges

    def train_step(self, x, beta):
        """Perform a single training step."""
        with tf.GradientTape() as tape:
            mc_states, px, sld_states = self.dynamics((x, beta), training=True)
            loss = self.calc_loss(mc_states.init.x, mc_states.proposed.x, px)
        grads = tape.gradient(loss, self.dynamics.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,
                                           self.dynamics.trainable_weights))
        return loss, mc_states.out.x, px, sld_states.out

    def train(self):
        """Main training loop."""
        x = tf.random.uniform(shape=self.input_shape, minval=-PI, maxval=PI)
        x = tf.cast(x, dtype=TF_FLOAT)
        plaqs_err, charges = self.calc_observables(0, x, self.beta_init)
        self.observables['plaqs_err'].append(plaqs_err.numpy())
        self.observables['charges'].append(charges.numpy())

        io.log(HEADER)
        for step, beta in zip(np.arange(self.train_steps), self.betas):
            t0 = time.time()
            x = tf.reshape(x, self.input_shape)
            loss, x, px, sld = self.train_step(x, beta)
            x = tf.reshape(x, self.lattice_shape)
            dt = time.time() - t0

            plaqs_err, charges = self.calc_observables(step, x, beta)
            dq = tf.math.abs(
                charges - self.observables['charges'][-1]
            )
            self.px_arr.append(px.numpy())
            self.loss_arr.append(loss.numpy())
            self.observables['dq'].append(dq.numpy())
            self.observables['charges'].append(charges.numpy())
            self.observables['plaqs_err'].append(plaqs_err.numpy())
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
            io.log(data_str)
            self.data_strs.append(data_str)

            if step % 100 == 0:
                io.log(HEADER)
