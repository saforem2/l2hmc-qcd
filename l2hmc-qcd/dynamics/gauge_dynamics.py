"""
gauge_dynamics.py

Implements the GaugeDynamics class by subclassing the `BaseDynamics` class.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Reference [Robust Parameter Estimation with a Neural Network Enhanced
Hamiltonian Markov Chain Monte Carlo Sampler]
https://infoscience.epfl.ch/record/264887/files/robust_parameter_estimation.pdf

Author: Sam Foreman (github: @saforem2)
Date: 7/3/2020
"""
# noqa:401
from __future__ import absolute_import, division, print_function

import os
import time

from typing import NoReturn

import numpy as np
import tensorflow as tf

from config import (BIN_DIR, GaugeDynamicsConfig, lrConfig, NetWeights,
                    NetworkConfig, PI, State, TWO_PI, MonteCarloStates)
from dynamics.base_dynamics import BaseDynamics
from network.gauge_network import GaugeNetwork
from utils.attr_dict import AttrDict
import utils.file_io as io
from utils.file_io import timeit  # pylint:disable=unused-import
from utils.seed_dict import vnet_seeds, xnet_seeds
from lattice.utils import u1_plaq_exact_tf
#  from lattice.lattice import GaugeLattice
from lattice.gauge_lattice import GaugeLattice

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False


TIMING_FILE = os.path.join(BIN_DIR, 'timing_file.log')


def convert_to_angle(x):
    """Returns x in -pi <= x < pi."""
    x = tf.math.floormod(x + PI, TWO_PI) - PI
    return x


def build_dynamics(flags):
    """Build dynamics using parameters from FLAGS."""
    '''
    if flags.log_dir is not None:
        flags_file = os.path.join(flags.log_dir, 'training', 'FLAGS.z')
        flags_ = AttrDict(dict(io.loadz(flags_file)))
        cfgdir = os.path.join(flags.log_dir, 'training', 'dynamics_configs')
        if flags.get('eps', None) is None:
            eps_file = os.path.join(flags.log_dir, 'training',
                                    'train_data', 'eps.z')
            io.log(f'Restoring `eps` from: {eps_file}...')
            flags.eps = io.loadz(eps_file)[-1]
        #  flags.update(flags_)
        # FIXME: Ensure that files exist...
        if os.path.isdir(cfgdir):
            cfg_file = os.path.join(cfgdir, 'dynamics_config.z')
            prms_file = os.path.join(cfgdir, 'dynamics_params.z')
            ncfg_file = os.path.join(cfgdir, 'network_config.z')
            lcfg_file = os.path.join(cfgdir, 'lr_config.z')
            try:
                config_ = io.loadz(cfg_file)
                params_ = io.loadz(prms_file)
                net_config_ = io.loadz(ncfg_file)
                lr_config_ = io.loadz(lcfg_file)
                flags_.update(params_)
                flags_.eps = flags.eps
                config = GaugeDynamicsConfig(**config_)
                net_config = NetworkConfig(**net_config_)
                lr_config = lrConfig(**lr_config_)
                return GaugeDynamics(flags_, config, net_config, lr_config)
            except FileNotFoundError:
                pass
    '''

    net_config = NetworkConfig(
        name='GaugeNetwork',
        units=flags.units,
        activation_fn=flags.get('activation_fn', tf.nn.relu),
        dropout_prob=flags.dropout_prob,
    )

    config = GaugeDynamicsConfig(
        model_type='GaugeModel',
        eps=flags.eps,
        hmc=flags.hmc,
        use_ncp=flags.use_ncp,
        num_steps=flags.num_steps,
        eps_trainable=not flags.eps_fixed,
        separate_networks=flags.separate_networks,
    )

    lr_config = lrConfig(
        init=flags.lr_init,
        decay_rate=flags.lr_decay_rate,
        decay_steps=flags.lr_decay_steps,
        warmup_steps=flags.warmup_steps,
    )

    flags = AttrDict({
        'horovod': flags.horovod,
        'plaq_weight': flags.plaq_weight,
        'charge_weight': flags.charge_weight,
        'lattice_shape': flags.lattice_shape,
    })

    dynamics = GaugeDynamics(flags, config, net_config, lr_config)

    return dynamics


# pylint:disable=attribute-defined-outside-init
# pylint:disable=too-many-instance-attributes,unused-argument
# pylint:disable=invalid-name,too-many-locals,too-many-arguments,
# pylint:disable=too-many-ancestors
class GaugeDynamics(BaseDynamics):
    """Implements the dynamics engine for the L2HMC sampler."""

    def __init__(
            self,
            params: AttrDict,
            config: GaugeDynamicsConfig,
            network_config: NetworkConfig,
            lr_config: lrConfig,
    ) -> NoReturn:

        self.aux_weight = params.get('aux_weight', 0.)
        self.plaq_weight = params.get('plaq_weight', 10.)
        self.charge_weight = params.get('charge_weight', 0.1)

        self.lattice_shape = params.get('lattice_shape', None)
        self.lattice = GaugeLattice(self.lattice_shape)

        params.update({
            'batch_size': self.lattice_shape[0],
            'xdim': np.cumprod(self.lattice_shape[1:])[-1],
            'mask_type': 'checkerboard',
        })

        super(GaugeDynamics, self).__init__(
            params=params,
            config=config,
            name='GaugeDynamics',
            normalizer=convert_to_angle,
            network_config=network_config,
            lr_config=lr_config,
            potential_fn=self.lattice.calc_actions,
        )

        if self.config.hmc:
            net_weights = NetWeights(0., 0., 0., 0., 0., 0.)
        elif self.config.use_ncp:
            net_weights = NetWeights(1., 1., 1., 1., 1., 1.)
        else:
            net_weights = NetWeights(0., 1., 1., 1., 1., 1.)

        self._parse_params(params, net_weights=net_weights)

    def _get_network(self, step):
        if self.config.separate_networks:
            xnet = getattr(self, f'xnets{int(step)}', None)
            vnet = getattr(self, f'vnets{int(step)}', None)
            return xnet, vnet

        return self.xnets, self.vnets

    def _build_networks(self):
        if self.config.separate_networks:
            def _new_net(name, seeds_=None):
                self.xnets = []
                self.vnets = []
                factor = 2. if name == 'XNet' else 1.
                for idx in range(self.config.num_steps):
                    #  new_seeds = {
                    #      key: int(idx * val) for key, val in seeds_.items()
                    #  }
                    net = GaugeNetwork(self.net_config,
                                       self.xdim, factor=factor,
                                       #  net_seeds=new_seeds,
                                       name=f'{name}_step{idx}')
                    if name == 'XNet':
                        setattr(self, f'xnets{int(idx)}', net)
                        self.xnets.append(net)
                    elif name == 'VNet':
                        setattr(self, f'vnets{int(idx)}', net)
                        self.vnets.append(net)

            _new_net('XNet')  # , xnet_seeds)
            _new_net('VNet')  # , vnet_seeds)

        else:
            self.xnets = GaugeNetwork(self.net_config, factor=2.,
                                      xdim=self.xdim, name='XNet')
            self.vnets = GaugeNetwork(self.net_config, factor=1.,
                                      xdim=self.xdim, name='VNet')

    @staticmethod
    def mixed_loss(loss, weight):
        """Returns: tf.reduce_mean(weight / loss - loss / weight)."""
        return tf.reduce_mean((weight / loss) - (loss / weight))

    def calc_losses(self, states: MonteCarloStates, accept_prob: tf.Tensor):
        """Calculate the total loss."""
        dtype = states.init.x.dtype

        # FIXME:
        #   - Should we stack `states = [states.init.x, states.proposed.x]`
        #     and call `self.lattice.calc_plaq_sums(states)`?
        wl_init = self.lattice.calc_wilson_loops(states.init.x)
        wl_prop = self.lattice.calc_wilson_loops(states.proposed.x)
        #  ps_init = self.lattice.calc_plaq_sums(samples=states.init.x)
        #  ps_prop = self.lattice.calc_plaq_sums(samples=states.proposed.x)

        # Calculate the plaquette loss
        ploss = tf.constant(0., dtype=dtype)
        if self.plaq_weight > 0:
            dwloops = 2 * (1. - tf.math.cos(wl_prop - wl_init))
            ploss = accept_prob * tf.reduce_sum(dwloops, axis=(1, 2))

            ploss = tf.reduce_mean(-ploss / self.plaq_weight, axis=0)
            #  ploss = self.mixed_loss(ploss, self.plaq_weight)

        # Calculate the charge loss
        qloss = tf.constant(0., dtype=dtype)
        if self.charge_weight > 0:
            q_init = self.lattice.calc_charges(wloops=wl_init, use_sin=True)
            q_prop = self.lattice.calc_charges(wloops=wl_prop, use_sin=True)
            qloss = accept_prob * (q_prop - q_init) ** 2

            qloss = tf.reduce_mean(-qloss / self.charge_weight, axis=0)
            #  qloss = self.mixed_loss(qloss, self.charge_weight)

        return ploss, qloss

    def train_step(self,
                   x: tf.Tensor,
                   beta: tf.Tensor,
                   first_step: bool = False):
        """Perform a single training step."""
        start = time.time()
        with tf.GradientTape() as tape:
            states, accept_prob, sumlogdet = self(x, beta, training=True)
            ploss, qloss = self.calc_losses(states, accept_prob)
            loss = ploss + qloss
            if self.aux_weight > 0:
                if first_step:
                    print(f'aux_weight: {self.aux_weight}; made it in here!')
                z = tf.random.normal(x.shape, dtype=x.dtype)
                states_, accept_prob_, _ = self(z, beta, training=True)
                ploss_, qloss_ = self.calc_losses(states_, accept_prob_)
                loss += ploss_ + qloss_

        if self.using_hvd:
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss, self.trainable_variables)
        if self.clip_val > 0:
            grads = [tf.clip_by_norm(g, self.clip_val) for g in grads]

        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables),
        )

        metrics = AttrDict({
            'dt': time.time() - start,
            'loss': loss,
            'ploss': ploss,
            'qloss': qloss,
            'accept_prob': accept_prob,
            'eps': self.eps,
            'beta': states.init.beta,
            'sumlogdet': sumlogdet.out,
        })

        observables = self.calc_observables(states)
        metrics.update(**observables)

        # Horovod:
        #    Broadcast initial variable states from rank 0 to all other
        #    processes. This is necessary to ensure consistent initialization
        #    of all workers when training is started with random weights or
        #    restored from a checkpoint.
        # NOTE:
        #    Broadcast should be done after the first gradient step to ensure
        #    optimizer intialization.
        #  if self.optimizer.iterations.numpy() == 0 and self.using_hvd:
        if first_step and HAS_HOROVOD:
            hvd.broadcast_variables(self.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        return states.out.x, metrics

    def test_step(self, x, beta):
        """Perform a single inference step."""
        start = time.time()
        states, px, sld = self(x, beta, training=False)
        ploss, qloss = self.calc_losses(states, px)
        loss = ploss + qloss

        metrics = AttrDict({
            'dt': time.time() - start,
            'loss': loss,
            'ploss': ploss,
            'qloss': qloss,
            'accept_prob': px,
            'eps': self.eps,
            'beta': states.init.beta,
            'sumlogdet': sld.out,
        })

        observables = self.calc_observables(states)
        metrics.update(**observables)

        return states.out.x, metrics

    def _calc_observables(self, state):
        """Calculate the observables for a particular state.

        NOTE: We track the error in the plaquette instead of the actual value.
        """
        wloops = self.lattice.calc_wilson_loops(state.x)
        charges = self.lattice.calc_charges(wloops=wloops, use_sin=False)
        charges_ = self.lattice.calc_charges(wloops=wloops, use_sin=True)
        plaqs = self.lattice.calc_plaqs(wloops=wloops, beta=state.beta)

        return plaqs, charges, charges_

    def calc_observables(self, states):
        """Calculate observables."""
        #  _, q_init = self._calc_observables(states.init)
        #  q_init = self.lattice.calc_charges(x=states.init.x)
        #  q_init_ = self.lattice.calc_charges(x=states.init.x)
        #  plaqs, q_out, q_out_approx = self._calc_observables(states.out)
        _, q_init, q_init_ = self._calc_observables(states.init)
        plaqs, q_out, q_out_ = self._calc_observables(states.out)

        observables = AttrDict({
            'dq': tf.math.abs(q_out - q_init),  # FIXME: Change to dQ ** 2
            'dq_sin': tf.math.abs(q_out_ - q_init_),
            'charges': q_out,
            'plaqs': plaqs,
        })

        return observables

    def _update_v_forward(
            self,
            network: tf.keras.layers.Layer,
            state: State,
            t: tf.Tensor,
            training: bool
    ):
        """Update the momentum `v` in the forward leapfrog step.

        NOTE: We pass a modified State object, where `state.x` is ensured to be
        in [-pi, pi) to satisfy the group constraint (x in U(1)).
        """
        state = State(x=self.normalizer(state.x),
                      v=state.v, beta=state.beta)
        return super()._update_v_forward(network, state, t, training)

    def _update_x_foward(self, network, state, t, masks, training):
        """Update the position `x` in the forward leapfrog step."""
        if not self.config.use_ncp:
            return super()._update_x_forward(network, state,
                                             t, masks, training)
        m, mc = masks

        # map x from [-pi, pi] to [-inf, inf]
        #  x = t.math.tan(state.x / 2.)
        S, T, Q = network((state.v, m * state.x, t), training)

        transl = self._xtw * T
        transf = self._xqw * (self.eps * Q)
        scale = self._xsw * (self.eps * S)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        x_transf = tf.math.tan(state.x/2)
        y = (2 * tf.math.atan(x_transf * expS)
             + self.eps * (state.v * expQ + transl))
        xf = (m * state.x) + (mc * y)
        state_out = State(x=xf, v=state.v, beta=state.beta)

        denom = (2 * tf.math.cosh(scale)
                 - 2 * tf.math.cos(state.x) * tf.math.sinh(scale))
        log_jac = tf.math.log(1. / denom)
        logdet = tf.reduce_sum(mc * log_jac, axis=1)

        #  logdet = tf.reduce_sum(mc * scale, axis=-1)

        return state_out, logdet

    def _update_x_backward(self, network, state, t, masks, training):
        """Update the position `x` in the backward leapfrog step."""
        if not self.config.use_ncp:
            return super()._update_x_backward(network, state,
                                              t, masks, training)
        m, mc = masks
        # map x from [-pi, pi] to [-inf, inf]
        #  x = tf.math.tan(state.x / 2.)
        S, T, Q = network((state.v, m * state.x, t), training)

        transl = self._xtw * T
        transf = self._xqw * (self.eps * Q)
        scale = self._xsw * (-self.eps * S)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        x_transf = tf.math.tan(state.x / 2)
        term1 = 2 * tf.math.atan(expS * x_transf)
        term2 = expS * self.eps * (state.v * expQ + transl)
        y = term1 - term2
        #  y = expS * (state.x - self.eps * (state.v * expQ + transl))
        xb = (m * state.x) + (mc * y)
        state_out = State(x=xb, v=state.v, beta=state.beta)

        denom = 2 * (tf.math.cosh(scale)
                     - tf.math.cos(state.x) * tf.math.sinh(scale))
        log_jac = tf.math.log(1. / denom)
        logdet = tf.reduce_sum(mc * log_jac, axis=1)

        #  logdet = tf.reduce_sum(mc * scale, axis=-1)

        return state_out, logdet
