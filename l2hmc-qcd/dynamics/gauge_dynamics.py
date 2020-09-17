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
# pylint:disable=too-many-instance-attributes,too-many-locals
# pylint:disable=invalid-name,too-many-arguments,too-many-ancestors
# pylint:disable=unused-import,unused-argument,attribute-defined-outside-init
from __future__ import absolute_import, division, print_function

import os
import json
import time

from math import pi
from typing import NoReturn, Optional, Tuple

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

import utils.file_io as io

from config import (BIN_DIR, GaugeDynamicsConfig, LearningRateConfig,
                    MonteCarloStates, NetWeights, NetworkConfig, State)
from lattice.gauge_lattice import GaugeLattice
from dynamics.base_dynamics import BaseDynamics
from utils.attr_dict import AttrDict
from utils.seed_dict import vnet_seeds  # noqa:F401
from utils.seed_dict import xnet_seeds  # noqa:F401
from network.gauge_network import GaugeNetwork
from network.gauge_conv_network import ConvolutionConfig, GaugeNetworkConv
from network.functional_net import get_gauge_network

NUM_RANKS = hvd.size()

TIMING_FILE = os.path.join(BIN_DIR, 'timing_file.log')
TF_FLOAT = tf.keras.backend.floatx()

INPUTS = Tuple[tf.Tensor, tf.Tensor]

ACTIVATIONS = {
    'relu': tf.nn.relu,
    'tanh': tf.nn.tanh,
    'leaky_relu': tf.nn.leaky_relu
}


def project_angle(x):
    """Returns the projection of an angle `x` from [-4pi, 4pi] to [-pi, pi]."""
    return x - 2 * np.pi * tf.math.floor((x + np.pi) / (2 * np.pi))


def convert_to_angle(x):
    """Returns x in -pi <= x < pi."""
    x = tf.math.floormod(x + pi, 2 * pi) - pi
    return x


def build_test_dynamics():
    """Build quick test dynamics for debugging."""
    jfile = os.path.abspath(os.path.join(BIN_DIR, 'test_dynamics_flags.json'))
    with open(jfile, 'rt') as f:
        flags = json.load(f)
    flags = AttrDict(flags)
    return build_dynamics(flags)


def build_dynamics(flags):
    """Build dynamics using configs from FLAGS."""
    lr_config = LearningRateConfig(**dict(flags.get('lr_config', None)))
    config = GaugeDynamicsConfig(**dict(flags.get('dynamics_config', None)))
    net_config = NetworkConfig(**dict(flags.get('network_config', None)))
    conv_config = None
    if flags.get('use_conv_net', False):
        conv_config = flags.get('conv_config', None)
        input_shape = flags.get('lattice_shape', None)[1:]
        conv_config.update({
            'input_shape': input_shape,
        })
        conv_config = ConvolutionConfig(**conv_config)

    return GaugeDynamics(flags, config, net_config, lr_config, conv_config)


class GaugeDynamics(BaseDynamics):
    """Implements the dynamics engine for the L2HMC sampler."""

    def __init__(
            self,
            params: AttrDict,
            config: GaugeDynamicsConfig,
            network_config: Optional[NetworkConfig] = None,
            lr_config: Optional[LearningRateConfig] = None,
            conv_config: Optional[ConvolutionConfig] = None
    ) -> NoReturn:

        self.aux_weight = config.get('aux_weight', 0.)
        self.plaq_weight = config.get('plaq_weight', 10.)
        self.charge_weight = config.get('charge_weight', 0.1)
        self.zero_init = config.get('zero_init', False)
        self._gauge_eq_masks = config.get('gauge_eq_masks', True)
        self.use_conv_net = config.get('use_conv_net', False)
        self.lattice_shape = config.get('lattice_shape', None)
        self.use_mixed_loss = config.get('use_mixed_loss', False)

        self.lattice = GaugeLattice(self.lattice_shape)
        self.batch_size = self.lattice_shape[0]
        self.xdim = np.cumprod(self.lattice_shape[1:])[-1]

        #  names = ['aux_weight', 'plaq_weight', 'charge_weight',
        #           'zero_init', 'gauge_eq_masks', 'use_conv_net',
        #           'lattice_shape', 'batch_size', 'xdim',
        #           'use_tempered_trajectories', 'use_mixed_loss']

        self.config = config
        self.lr_config = lr_config
        self.conv_config = conv_config
        self.net_config = network_config
        if not self.use_conv_net:
            self.conv_config = None

        params.update({
            'batch_size': self.lattice_shape[0],
            'xdim': np.cumprod(self.lattice_shape[1:])[-1],
        })

        super(GaugeDynamics, self).__init__(
            params=params,
            config=config,
            name='GaugeDynamics',
            normalizer=convert_to_angle,
            network_config=network_config,
            lr_config=lr_config,
            potential_fn=self.lattice.calc_actions,
            should_build=False,
        )
        self._has_trainable_params = True
        if self.config.hmc:
            net_weights = NetWeights(0., 0., 0., 0., 0., 0.)
            self.config.use_ncp = False
            self.config.separate_networks = False
            self.use_conv_net = False
            self.conv_config = None
            self.xnet, self.vnet = self._build_hmc_networks()
            if self.config.eps_fixed:
                self._has_trainable_params = False
        else:
            if self.config.use_ncp:
                net_weights = NetWeights(1., 1., 1., 1., 1., 1.)
            else:
                net_weights = NetWeights(0., 1., 1., 1., 1., 1.)

            kinit = 'zeros' if self.zero_init else None
            xnet_input_shapes = {
                'x': (self.xdim, 2), 'v': (self.xdim,), 't': (2,)
            }
            vnet_input_shapes = {
                'x': (self.xdim,), 'v': (self.xdim,), 't': (2,),
            }

            args = (self.lattice_shape, self.net_config)
            self.vnet = get_gauge_network(*args, name='VNet',
                                          kernel_initializer=kinit,
                                          factor=1., conv_config=None,
                                          input_shapes=vnet_input_shapes)

            xnet_kwargs = {
                'factor': 2.,
                'kernel_initializer': kinit,
                'conv_config': self.conv_config,
                'input_shapes': xnet_input_shapes,
            }

            if self.config.separate_networks:
                self.xnets = [
                    get_gauge_network(*args, **xnet_kwargs, name=f'XNet{i}')
                    for i in range(self.config.num_steps)
                ]
            else:
                self.xnet = get_gauge_network(*args, **xnet_kwargs)

        self.net_weights = self._parse_net_weights(net_weights)
        if self._has_trainable_params:
            self.lr_config = lr_config
            self.lr = self._create_lr(lr_config)
            self.optimizer = self._create_optimizer()

    def transition_kernel_tempered(
                self, state: State, forward: bool, training: bool = None
    ):
        """Transition kernel of the augmented leapfrog integrator.

        NOTE: This method attempts to perform simulated annealing WITHIN
        a trajectory, by linearly decreasing beta by 10% over the first
        half of the trajectory, followed by a linear increase back to its
        original value over the second half of the trajectory.
        """
        lf_fn = self._forward_lf if forward else self._backward_lf
        state_ = State(state.x, state.v, state.beta)
        sumlogdet = tf.zeros((self.batch_size,), dtype=TF_FLOAT)

        step_mid = self.config.num_steps // 2
        beta_mid = state.beta - 0.5 * state.beta

        betas = tf.concat([
            tf.linspace(state.beta, beta_mid, step_mid),  # beta0 --> beta_mid
            tf.linspace(beta_mid, state.beta, step_mid),  # beta_mid --> beta0
        ], axis=-1)
        #  betas = tf.concat([betas1, betas2], axis=-1)

        for step in tf.range(self.config.num_steps):
            beta = tf.gather(betas, step)
            state_ = State(state_.x, state_.v, beta)
            state_, logdet = lf_fn(step, state_, training)
            sumlogdet += logdet

        px = self.compute_accept_prob(state, state_, sumlogdet)

        return state_, px, sumlogdet

    def save_config(self, config_dir: str):
        """Helper method for saving configuration objects."""
        io.save_dict(self.config, config_dir, name='dynamics_config')
        io.save_dict(self.net_config, config_dir, name='network_config')
        io.save_dict(self.lr_config, config_dir, name='lr_config')
        io.save_dict(self.params, config_dir, name='dynamics_params')
        if self.conv_config is not None and self.use_conv_net:
            io.save_dict(self.conv_config, config_dir, name='conv_config')

    def get_config(self):
        return {
            'config': self.config,
            'network_config': self.net_config,
            'conv_config': self.conv_config,
            'lr_config': self.lr_config,
            'params': self.params
        }

    def _get_network(self, step: int):
        if self.config.separate_networks:
            xnet = getattr(self, f'xnets{int(step)}', None)
            vnet = getattr(self, f'vnets{int(step)}', None)
            return xnet, vnet

        return self.xnet, self.vnet

    def _build_masks(self):
        """Construct different binary masks for different time steps."""
        def rolled_reshape(m, ax, shape=None):
            if shape is None:
                shape = (self.batch_size, -1)

            return sum([np.roll(m, i, ax).reshape(shape) for i in range(4)])

        masks = []
        zeros = np.zeros(self.lattice_shape, dtype=np.float32)

        if self._gauge_eq_masks:
            mh_ = zeros.copy()
            mv_ = zeros.copy()
            mh_[:, ::4, :, 1] = 1.  # Horizontal masks
            mv_[:, :, ::4, 0] = 1.  # Vertical masks

            mh = rolled_reshape(mh_, ax=1)
            mv = rolled_reshape(mv_, ax=2)
            for i in range(self.config.num_steps):
                mask = mh if i % 2 == 0 else mv
                masks.append(tf.constant(mask))
        else:
            p = zeros.copy()
            for idx, _ in np.ndenumerate(zeros):
                p[idx] = (sum(idx) % 2 == 0)

            for i in range(self.config.num_steps):
                m = p if i % 2 == 0 else (1. - p)
                mask = tf.reshape(m, (self.batch_size, -1))
                masks.append(tf.constant(mask))

        return masks

    @staticmethod
    def mixed_loss(loss: tf.Tensor, weight: float):
        """Returns: tf.reduce_mean(weight / loss - loss / weight)."""
        return tf.reduce_mean((weight / loss) - (loss / weight))

    def calc_losses(self, states: MonteCarloStates, accept_prob: tf.Tensor):
        """Calculate the total loss."""
        #  dtype = states.init.x.dtype
        # ==== FIXME: Should we stack
        # ==== `states = [states.init.x, states.proposed.x]`
        # ==== and call `self.lattice.calc_plaq_sums(states)`?
        wl_init = self.lattice.calc_wilson_loops(states.init.x)
        wl_prop = self.lattice.calc_wilson_loops(states.proposed.x)

        # Calculate the plaquette loss
        #  ploss = tf.cast(0., dtype=dtype)
        ploss = 0.
        if self.plaq_weight > 0:
            dwloops = 2 * (1. - tf.math.cos(wl_prop - wl_init))
            ploss = accept_prob * tf.reduce_sum(dwloops, axis=(1, 2))

            # ==== FIXME: Try using mixed loss??
            if self.use_mixed_loss:
                ploss = self.mixed_loss(ploss, self.plaq_weight)
            else:
                ploss = tf.reduce_mean(-ploss / self.plaq_weight, axis=0)

        # Calculate the charge loss
        #  qloss = tf.cast(0., dtype=dtype)
        qloss = 0.
        if self.charge_weight > 0:
            q_init = tf.reduce_sum(tf.sin(wl_init), axis=(1, 2)) / (2 * np.pi)
            q_prop = tf.reduce_sum(tf.sin(wl_prop), axis=(1, 2)) / (2 * np.pi)
            #  q_prop = tf.sin(wl_prop)
            #  dq2 = (q_prop - q_init)
            qloss = accept_prob * (q_prop - q_init) ** 2 + 1e-4
            if self.use_mixed_loss:
                qloss = self.mixed_loss(qloss, self.charge_weight)
            else:
                qloss = tf.reduce_mean(-qloss / self.charge_weight, axis=0)
            #  dq_sum = tf.reduce_sum((q_prop - q_init) ** 2, axis=(1, 2))
            #  qloss = accept_prob * dq_sum + 1e-4
            #  qloss = (tf.reduce_mean(self.charge_weight / dq_)
            #           - tf.reduce_mean(dq_ / self.charge_weight))

            #  q_init = self.lattice.calc_charges(wloops=wl_init, use_sin=True)
            #  q_prop = self.lattice.calc_charges(wloops=wl_prop, use_sin=True)
            #  qloss = accept_prob * (q_prop - q_init) ** 2

            # ==== FIXME: Try using mixed loss??
            #  qloss = self.mixed_loss(qloss, self.charge_weight)
            #  qloss = tf.reduce_mean(-qloss / self.charge_weight, axis=0)

        return ploss, qloss

    def calc_wilson_loops(self, x):
        """Calculate the plaqs by summing the links in the CCW direction."""
        x = tf.reshape(x, shape=self.lattice_shape)
        plaqs = (x[..., 0]
                 - x[..., 1]
                 - tf.roll(x[..., 0], shift=-1, axis=2)
                 + tf.roll(x[..., 1], shift=-1, axis=1))

        return plaqs

    def calc_charges(self, x=None, wloops=None, use_sin=False):
        """Calculate the topological charges for a batch of lattices."""
        if wloops is None:
            try:
                wloops = self.calc_wilson_loops(x)
            except ValueError as err:
                raise err

        q = tf.sin(wloops) if use_sin else project_angle(wloops)

        return tf.reduce_sum(q, axis=(1, 2), name='charges') / (2 * np.pi)

    def train_step(self, data):
        """Perform a single training step."""
        start = time.time()
        with tf.GradientTape() as tape:
            x, beta = data
            #  v = tf.random.normal((self.batch_size, self.xdim // 2))
            tape.watch(x)
            #  tape.watch(v)
            states, accept_prob, sumlogdet = self((x, beta), training=True)
            ploss, qloss = self.calc_losses(states, accept_prob)
            loss = ploss + qloss
            if self.aux_weight > 0:
                z = tf.random.normal(x.shape, dtype=x.dtype)
                states_, accept_prob_, _ = self((z, beta), training=True)
                ploss_, qloss_ = self.calc_losses(states_, accept_prob_)
                loss += ploss_ + qloss_

        #  if self.using_hvd:
        if NUM_RANKS > 1:
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss, self.trainable_variables)
        #  if self.clip_val > 0:
        #      grads = [tf.clip_by_norm(g, self.clip_val) for g in grads]

        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables),
        )

        metrics = AttrDict({
            'dt': time.time() - start,
            'loss': loss,
            'ploss': ploss,
            'qloss': qloss,
        })

        if self.aux_weight > 0:
            metrics.update({
                'ploss_aux': ploss_,
                'qloss_aux': qloss_
            })

        #  metrics = AttrDict({
        metrics.update({
            #  'dt': time.time() - start,
            #  'loss': loss,
            #  'ploss': ploss,
            #  'qloss': qloss,
            'accept_prob': accept_prob,
            'eps': self.eps,
            'beta': states.init.beta,
            'sumlogdet': sumlogdet.out,
        })

        observables = self.calc_observables(states)
        metrics.update(**observables)

        #  if loss > 50:
        #      x_out = tf.random.normal(states.out.x.shape,
        #                               dtype=states.out.x.dtype)
        #  else:
        #      x_out = states.out.x

        # Horovod:
        #    Broadcast initial variable states from rank 0 to all other
        #    processes. This is necessary to ensure consistent initialization
        #    of all workers when training is started with random weights or
        #    restored from a checkpoint.
        # NOTE:
        #    Broadcast should be done after the first gradient step to ensure
        #    optimizer intialization.
        #  if self.optimizer.iterations.numpy() == 0 and self.using_hvd:
        #  if first_step and HAS_HOROVOD:
        if self.optimizer.iterations == 0 and NUM_RANKS > 1:
            hvd.broadcast_variables(self.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        return states.out.x, metrics

    def test_step(self, data):
        """Perform a single inference step."""
        start = time.time()
        states, px, sld = self(data, training=False)
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
        q_sin = self.lattice.calc_charges(wloops=wloops, use_sin=True)
        q_proj = self.lattice.calc_charges(wloops=wloops, use_sin=False)
        plaqs = self.lattice.calc_plaqs(wloops=wloops, beta=state.beta)

        return plaqs, q_sin, q_proj

    def calc_observables(self, states):
        """Calculate observables."""
        _, q_init_sin, q_init_proj = self._calc_observables(states.init)
        plaqs, q_out_sin, q_out_proj = self._calc_observables(states.out)
        dq_sin = tf.math.abs(q_out_sin - q_init_sin)
        dq_proj = tf.math.abs(q_out_proj - q_init_proj)

        observables = AttrDict({
            'dq': dq_proj,
            'dq_sin': dq_sin,
            'charges': q_out_proj,
            'plaqs': plaqs,
        })

        return observables

    def _scattered_xnet(self, inputs, mask, step, training=None):
        """Call `self.xnet` on non-zero entries of `x` via `tf.gather_nd`."""
        #  m, _ = masks
        if len(mask) == 2:
            mask, _ = mask

        x, v, t = inputs
        shape = (self.batch_size, -1)
        m = tf.reshape(mask, shape)
        idxs = tf.where(m)
        _x = tf.reshape(tf.gather_nd(x, idxs), shape)
        _x = tf.concat([tf.math.cos(_x), tf.math.sin(_x)], axis=-1)
        if not self.config.separate_networks:
            S, T, Q = self.xnet((_x, v, t), training)
        else:
            xnet = self.xnets[step]
            S, T, Q = xnet((x, v, t), training)

        return S, T, Q

    def _call_xnet(self, inputs, mask, step, training=None):
        """Call `self.xnet` to get Sx, Tx, Qx for updating `x`."""
        if len(mask) == 2:
            mask, _ = mask

        x, v, t = inputs

        x_cos = mask * tf.math.cos(x)
        x_sin = mask * tf.math.sin(x)
        if self.config.use_conv_net:
            x_cos = tf.reshape(x_cos, self.lattice_shape)
            x_sin = tf.reshape(x_sin, self.lattice_shape)

        x = tf.concat([x_cos, x_sin], axis=-1)
        if not self.config.separate_networks:
            S, T, Q = self.xnet((x, v, t), training)
        else:
            xnet = self.xnets[step]
            S, T, Q = xnet((x, v, t), training)
            #  if step % 2 == 0:
            #      S, T, Q = self.xnet_even((x, v, t), training)
            #  else:
            #      S, T, Q = self.xnet_odd((x, v, t), training)

        return S, T, Q

    def _update_v_forward(
                self,
                state: State,
                step: int,
                training: bool = None
    ):
        """Update the momentum `v` in the forward leapfrog step.

        Args:
            network (tf.keras.Layers): Network to use
            state (State): Input state
            t (float): Current leapfrog step, represented as periodic time.
            training (bool): Currently training?

        Returns:
            new_state (State): New state, with updated momentum.
            logdet (float): Jacobian factor
        """
        x = self.normalizer(state.x)
        grad = self.grad_potential(x, state.beta)
        t = self._get_time(step, tile=tf.shape(x)[0])

        S, T, Q = self.vnet((x, grad, t), training)

        scale = self._vsw * (0.5 * self.eps * S)
        transl = self._vtw * T
        transf = self._vqw * (self.eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        vf = state.v * expS - 0.5 * self.eps * (grad * expQ - transl)

        state_out = State(x=x, v=vf, beta=state.beta)
        logdet = tf.reduce_sum(scale, axis=1)

        return state_out, logdet

    def _update_x_forward(
                self,
                state: State,
                step: int,
                masks: Tuple[tf.Tensor, tf.Tensor],  # [m, 1. - m]
                training: bool = None
    ):
        """Update the position `x` in the forward leapfrog step.

        Args:
            state (State): Input state
            t (float): Current leapfrog step, represented as periodic time.
            training (bool): Currently training?

        Returns:
            new_state (State): New state, with updated momentum.
            logdet (float): logdet of Jacobian factor.
        """
        if self.config.hmc:
            return super()._update_x_forward(state, step, masks, training)
        #  if self.config.use_ncp:
        #      return self._update_xf_ncp(state, step, masks, training)

        m, mc = masks
        x = self.normalizer(state.x)
        t = self._get_time(step, tile=tf.shape(x)[0])

        S, T, Q = self._call_xnet((x, state.v, t), m, step, training)

        scale = self._xsw * (self.eps * S)
        transl = self._xtw * T
        transf = self._xqw * (self.eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        if self.config.use_ncp:
            _x = 2 * tf.math.atan(tf.math.tan(state.x/2.) * expS)
            _y = _x + self.eps * (state.v * expQ + transl)
            xf = self.normalizer((m * state.x) + (mc * _y))

            cterm = tf.math.cos(state.x / 2) ** 2
            sterm = (expS * tf.math.sin(state.x / 2)) ** 2
            logdet_ = tf.math.log(expS / (cterm + sterm))
            logdet = tf.reduce_sum(mc * logdet_, axis=1)

        else:
            y = x * expS + self.eps * (state.v * expQ + transl)
            xf = self.normalizer(m * x + mc * y)
            logdet = tf.reduce_sum(mc * scale, axis=1)

        state_out = State(x=xf, v=state.v, beta=state.beta)

        return state_out, logdet

    def _update_v_backward(
                self,
                state: State,
                step: int,
                training: bool = None
    ):
        """Update the momentum `v` in the backward leapfrog step.

        Args:
            state (State): Input state.
            t (float): Current leapfrog step, represented as periodic time.
            training (bool): Currently training?

        Returns:
            new_state (State): New state, with updated momentum.
            logdet (float): Jacobian factor.
        """
        x = self.normalizer(state.x)
        grad = self.grad_potential(x, state.beta)
        t = self._get_time(step, tile=tf.shape(x)[0])
        S, T, Q = self.vnet((x, grad, t), training)

        scale = self._vsw * (-0.5 * self.eps * S)
        transf = self._vqw * (self.eps * Q)
        transl = self._vtw * T

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        vb = expS * (state.v + 0.5 * self.eps * (grad * expQ - transl))

        state_out = State(x=x, v=vb, beta=state.beta)
        logdet = tf.reduce_sum(scale, axis=1)

        return state_out, logdet

    def _update_x_backward(
                self,
                state: State,
                step: int,
                masks: Tuple[tf.Tensor, tf.Tensor],   # [m, 1. - m]
                training: bool = None
    ):
        """Update the position `x` in the backward leapfrog step.

        Args:
            state (State): Input state
            t (float): Current leapfrog step, represented as periodic time.
            training (bool): Currently training?


        Returns:
            new_state (State): New state, with updated momentum.
            logdet (float): logdet of Jacobian factor.
        """
        if self.config.hmc:
            return super()._update_x_backward(state, step, masks, training)
        #  if self.config.use_ncp:
        #      return self._update_xb_ncp(state, step, masks, training)

        # Call `XNet` using `self._scattered_xnet`
        m, mc = masks
        x = self.normalizer(state.x)
        t = self._get_time(step, tile=tf.shape(x)[0])
        S, T, Q = self._call_xnet((x, state.v, t), m, step, training)

        scale = self._xsw * (-self.eps * S)
        transl = self._xtw * T
        transf = self._xqw * (self.eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        if self.config.use_ncp:
            term1 = 2 * tf.math.atan(expS * tf.math.tan(state.x / 2))
            term2 = expS * self.eps * (state.v * expQ + transl)
            y = term1 - term2
            xb = self.normalizer((m * state.x) + (mc * y))

            cterm = tf.math.cos(state.x / 2) ** 2
            sterm = (expS * tf.math.sin(state.x / 2)) ** 2
            logdet_ = tf.math.log(expS / (cterm + sterm))
            logdet = tf.reduce_sum(mc * logdet_, axis=1)

        else:
            y = expS * (x - self.eps * (state.v * expQ + transl))
            xb = self.normalizer(m * x + mc * y)
            logdet = tf.reduce_sum(mc * scale, axis=1)

        state_out = State(xb, v=state.v, beta=state.beta)
        return state_out, logdet
