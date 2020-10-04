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
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

import utils.file_io as io

from config import BIN_DIR
from lattice.gauge_lattice import GaugeLattice
from utils.attr_dict import AttrDict
from utils.seed_dict import vnet_seeds  # noqa:F401
from utils.seed_dict import xnet_seeds  # noqa:F401
from network.config import (ConvolutionConfig, LearningRateConfig,
                            NetworkConfig)
from network.functional_net import get_gauge_network
from dynamics.config import GaugeDynamicsConfig
from dynamics.base_dynamics import (BaseDynamics, MonteCarloStates, NetWeights,
                                    State)

NUM_RANKS = hvd.size()

TIMING_FILE = os.path.join(BIN_DIR, 'timing_file.log')
TF_FLOAT = tf.keras.backend.floatx()

#  INPUTS = Tuple[tf.Tensor, tf.Tensor]


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

    if config.get('use_conv_net', False):
        conv_config = flags.get('conv_config', None)
        input_shape = config.get('lattice_shape', None)[1:]
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
    ):
        # ====
        # Set attributes from `config`
        self.aux_weight = config.get('aux_weight', 0.)
        self.plaq_weight = config.get('plaq_weight', 0.)
        self.charge_weight = config.get('charge_weight', 0.01)
        self._gauge_eq_masks = config.get('gauge_eq_masks', False)
        self.lattice_shape = config.get('lattice_shape', None)
        self._alpha = tf.constant(1.)
        #  self._alpha = tf.Variable(initial_value=1., trainable=False)
        if config.use_tempered_traj:
            self._alpha = tf.Variable(initial_value=1., trainable=True)

        self.lattice = GaugeLattice(self.lattice_shape)
        self.batch_size = self.lattice_shape[0]
        self.xdim = np.cumprod(self.lattice_shape[1:])[-1]

        self.config = config
        self.lr_config = lr_config
        self.conv_config = conv_config
        self.net_config = network_config
        if not self.config.use_conv_net:
            self.conv_config = None

        params.update({
            'batch_size': self.lattice_shape[0],
            'xdim': np.cumprod(self.lattice_shape[1:])[-1],
        })

        super().__init__(
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
            self.config.use_conv_net = False
            self.conv_config = None
            self.xnet, self.vnet = self._build_hmc_networks()
            if self.config.eps_fixed:
                self._has_trainable_params = False
        else:
            if self.config.use_ncp:
                net_weights = NetWeights(1., 1., 1., 1., 1., 1.)
            else:
                net_weights = NetWeights(0., 1., 1., 1., 1., 1.)

            self.xnet, self.vnet = self._build_networks(self.net_config,
                                                        self.conv_config)
            # ============

        self.net_weights = self._parse_net_weights(net_weights)
        if self._has_trainable_params:
            self.lr_config = lr_config
            self.lr = self._create_lr(lr_config)
            self.optimizer = self._create_optimizer()

    def _build_networks(
            self,
            net_config: NetworkConfig = None,
            conv_config: ConvolutionConfig = None,
    ):
        """Build position and momentum networks.

        Returns:
            xnet: tf.keras.models.Model
            vnet: tf.keras.models.Model
        """
        if net_config is None:
            net_config = self.net_config

        if conv_config is None and self.config.use_conv_net:
            conv_config = self.conv_config

        xshape = (self.xdim, 2)
        if self.config.use_conv_net:
            xshape = (*self.lattice_shape[1:], 2)

        kinit = None
        if self.config.zero_init:
            kinit = 'zeros'

        # ====
        # xNet configuration
        xnet_cfg = {
            'factor': 2.0,
            'net_config': net_config,
            'conv_config': conv_config,
            'kernel_initializer': kinit,
            'lattice_shape': self.lattice_shape,
            'input_shapes': {
                'x': xshape, 'v': (self.xdim,), 't': (2,)
            }
        }

        # ====
        # xNet configuration
        vnet_cfg = {
            'factor': 1.0,
            'net_config': net_config,
            'conv_config': None,
            'kernel_initializer': kinit,
            'lattice_shape': self.lattice_shape,
            'input_shapes': {
                'x': (self.xdim,), 'v': (self.xdim,), 't': (2,)
            }
        }

        if self.config.separate_networks:
            # ====
            # Build separate networks
            vnet = [
                get_gauge_network(**vnet_cfg, name=f'VNet{i}')
                for i in range(self.config.num_steps)
            ]

            xnet = [
                get_gauge_network(**xnet_cfg, name=f'XNet{i}')
                for i in range(self.config.num_steps)
            ]

        else:
            # ====
            # Build single network
            vnet = get_gauge_network(**vnet_cfg, name='VNet')
            xnet = get_gauge_network(**xnet_cfg, name='XNet')

        return xnet, vnet

    def transition_kernel_tempered(
            self,
            state: State,
            forward: bool,
            training: bool = None,
    ):
        """Transition kernel of the augmented leapfrog integrator."""
        if forward:
            lf_fn = self._forward_lf_tempered
        else:
            lf_fn = self._backward_lf_tempered

        state_prop = State(state.x, state.v, state.beta)
        sumlogdet = tf.zeros((self.batch_size,), dtype=TF_FLOAT)
        if self._verbose:
            logdets = tf.TensorArray(TF_FLOAT,
                                     dynamic_size=True,
                                     size=self.batch_size,
                                     clear_after_read=False)
            energies = tf.TensorArray(TF_FLOAT,
                                      dynamic_size=True,
                                      size=self.batch_size,
                                      clear_after_read=False)
            energies = energies.write(0, self.hamiltonian(state_prop))
            logdets = logdets.write(0, sumlogdet)
        for step in tf.range(self.config.num_steps):
            state_prop, logdet = lf_fn(step, state_prop, training)
            sumlogdet += logdet

            if self._verbose:
                logdets = logdets.write(step, logdet)
                energies = energies.write(step, self.hamiltonian(state_prop))

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)

        metrics = AttrDict({
            'sumlogdet': sumlogdet,
            'accept_prob': accept_prob,
        })

        if self._verbose:
            metrics.update({
                'energies': [
                    energies.read(i) for i in range(self.config.num_steps)
                ],
                'logdets': [
                    logdets.read(i) for i in range(self.config.num_steps)
                ],
            })

        return state_prop, metrics

    def transition_kernel_directional(
            self,
            state: State,
            forward: bool,
            training: bool = None,
    ):
        """Implements a series of directional updates."""
        state_prop = State(state.x, state.v, state.beta)
        sumlogdet = tf.zeros((self.batch_size,), dtype=TF_FLOAT)
        if self._verbose:
            logdets = tf.TensorArray(TF_FLOAT,
                                     dynamic_size=True,
                                     size=self.batch_size,
                                     clear_after_read=False)
            energies = tf.TensorArray(TF_FLOAT,
                                      dynamic_size=True,
                                      size=self.batch_size,
                                      clear_after_read=False)
            energies = energies.write(0, self.hamiltonian(state_prop))
            logdets = logdets.write(0, sumlogdet)

        # ====
        # Forward for first half of trajectory
        for step in range(self.config.num_steps // 2):
            state_prop, logdet = self._forward_lf(step, state_prop, training)
            sumlogdet += logdet

            if self._verbose:
                logdets = logdets.write(step, logdet)
                energies = energies.write(step, self.hamiltonian(state_prop))

        # ====
        # Flip momentum
        state_prop = State(state_prop.x, -1. * state_prop.v, state_prop.beta)

        # ====
        # Backward for second half of trajectory
        for step in range(self.config.num_steps // 2, self.config.num_steps):
            state_prop, logdet = self._backward_lf(step, state_prop, training)
            sumlogdet += logdet

            logdets = logdets.write(step, logdet)
            energies = energies.write(step, self.hamiltonian(state_prop))

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)
        metrics = AttrDict({
            'sumlogdet': sumlogdet,
            'accept_prob': accept_prob,
        })
        if self._verbose:
            metrics.update({
                'energies': [
                    energies.read(i) for i in range(self.config.num_steps)
                ],
                'logdets': [
                    logdets.read(i) for i in range(self.config.num_steps)
                ],
            })

        return state_prop, metrics

    def transition_kernel_sep_nets(
            self,
            state: State,
            forward: bool,
            training: bool = None,
    ):
        """Implements a transition kernel when using separate networks."""
        lf_fn = self._forward_lf if forward else self._backward_lf
        state_prop = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = tf.zeros((self.batch_size,))
        if self._verbose:
            logdets = tf.TensorArray(TF_FLOAT,
                                     dynamic_size=True,
                                     size=self.batch_size,
                                     clear_after_read=False)
            energies = tf.TensorArray(TF_FLOAT,
                                      dynamic_size=True,
                                      size=self.batch_size,
                                      clear_after_read=False)
            energies = energies.write(0, self.hamiltonian(state_prop))
            logdets = logdets.write(0, sumlogdet)
        #  logdets = [sumlogdet]
        #  energies = [self.hamiltonian(state_prop)]

        for step in range(self.config.num_steps):
            state_prop, logdet = lf_fn(step, state_prop, training)
            sumlogdet += logdet

            if self._verbose:
                logdets = logdets.write(step, logdet)
                energies = energies.write(step, self.hamiltonian(state_prop))

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)

        metrics = AttrDict({
            'sumlogdet': sumlogdet,
            'accept_prob': accept_prob,
        })
        if self._verbose:
            metrics.update({
                'energies': [
                    energies.read(i) for i in range(self.config.num_steps)
                ],
                'logdets': [
                    logdets.read(i) for i in range(self.config.num_steps)
                ],
            })

        return state_prop, metrics

    def transition_kernel(
            self, state: State, forward: bool, training: bool = None
    ):
        """Transition kernel of the augmented leapfrog integrator."""
        if self.config.use_tempered_traj:
            return self.transition_kernel_tempered(state, forward, training)

        if self.config.separate_networks:
            return self.transition_kernel_sep_nets(state, forward, training)

        if self.config.directional_updates:
            return self.transition_kernel_directional(state, training)

        return super().transition_kernel(state, forward, training)

    def _forward_lf_tempered(
            self, step: int, state: State, training: bool = None
    ):

        def _tempered_v(step, v):
            if step < self.config.num_steps // 2:
                return v * tf.math.sqrt(self._alpha)
            return v / tf.math.sqrt(self._alpha)

        sumlogdet = tf.zeros((self.batch_size,), dtype=TF_FLOAT)
        m, mc = self._get_mask(step)
        v = _tempered_v(step, state.v)
        state = State(state.x, v, state.beta)
        state, logdet = self._update_v_forward(state, step, training)
        sumlogdet += logdet
        state, logdet = self._update_x_forward(state, step, (m, mc), training)
        sumlogdet += logdet
        state, logdet = self._update_x_forward(state, step, (mc, m), training)
        sumlogdet += logdet
        state, logdet = self._update_v_forward(state, step, training)
        sumlogdet += logdet
        v = _tempered_v(step, state.v)
        state = State(state.x, v, state.beta)

        return state, sumlogdet

    def _backward_lf_tempered(
            self, step: int, state: State, training: bool = None
    ):
        """Run the augmented leapfrog integrator in the backward direction."""
        def _tempered_v(step, v):
            if step < self.config.num_steps // 2:
                return v / tf.math.sqrt(self._alpha)
            return v * tf.math.sqrt(self._alpha)

        sumlogdet = 0.
        step = self.config.num_steps - step - 1  # reversed step
        m, mc = self._get_mask(step)
        v = _tempered_v(step, state.v)
        state = State(state.x, v, state.beta)
        state, logdet = self._update_v_backward(state, step, training)
        sumlogdet += logdet
        state, logdet = self._update_x_backward(state, step, (mc, m), training)
        sumlogdet += logdet
        state, logdet = self._update_x_backward(state, step, (m, mc), training)
        sumlogdet += logdet
        state, logdet = self._update_v_backward(state, step, training)
        sumlogdet += logdet
        v = _tempered_v(step, state.v)
        state = State(state.x, v, state.beta)

        return state, logdet

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
            xnet = self.xnet[step]
            S, T, Q = xnet((x, v, t), training)

        return S, T, Q

    def _call_vnet(self, inputs, step, training=None):
        """Call `self.xnet` to get Sx, Tx, Qx for updating `x`."""
        x, grad, t = inputs
        if not self.config.separate_networks:
            S, T, Q = self.vnet((x, grad, t), training)
        else:
            vnet = self.vnet[step]
            S, T, Q = vnet((x, grad, t), training)

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

        #  x = tf.stack([x_cos, x_sin], axis=-1)
        x = tf.concat([x_cos, x_sin], -1)
        if not self.config.separate_networks:
            S, T, Q = self.xnet((x, v, t), training)
        else:
            xnet = self.xnet[step]
            S, T, Q = xnet((x, v, t), training)

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

        S, T, Q = self._call_vnet((x, grad, t), step, training)

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
            _x = 2 * tf.math.atan(tf.math.tan(x/2.) * expS)
            _y = _x + self.eps * (state.v * expQ + transl)
            xf = (m * x) + (mc * _y)

            cterm = tf.math.cos(x / 2) ** 2
            sterm = (expS * tf.math.sin(x / 2)) ** 2
            logdet_ = tf.math.log(expS / (cterm + sterm))
            logdet = tf.reduce_sum(mc * logdet_, axis=1)

        else:
            y = x * expS + self.eps * (state.v * expQ + transl)
            xf = (m * x) + (mc * y)
            logdet = tf.reduce_sum(mc * scale, axis=1)

        xf = self.normalizer(xf)
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
        S, T, Q = self._call_vnet((x, grad, t), step, training)

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
            xb = (m * x) + (mc * y)

            cterm = tf.math.cos(x / 2) ** 2
            sterm = (expS * tf.math.sin(x / 2)) ** 2
            logdet_ = tf.math.log(expS / (cterm + sterm))
            logdet = tf.reduce_sum(mc * logdet_, axis=1)

        else:
            y = expS * (x - self.eps * (state.v * expQ + transl))
            xb = m * x + mc * y
            logdet = tf.reduce_sum(mc * scale, axis=1)

        xb = self.normalizer(xb)
        state_out = State(xb, v=state.v, beta=state.beta)
        return state_out, logdet

    @staticmethod
    def mixed_loss(loss: tf.Tensor, weight: float):
        """Returns: tf.reduce_mean(weight / loss - loss / weight)."""
        return tf.reduce_mean((weight / loss) - (loss / weight))

    def calc_losses(self, states: MonteCarloStates, accept_prob: tf.Tensor):
        """Calculate the total loss."""
        wl_init = self.lattice.calc_wilson_loops(states.init.x)
        wl_prop = self.lattice.calc_wilson_loops(states.proposed.x)

        # Calculate the plaquette loss
        ploss = 0.
        if self.plaq_weight > 0:
            dwloops = 2 * (1. - tf.math.cos(wl_prop - wl_init))
            ploss = accept_prob * tf.reduce_sum(dwloops, axis=(1, 2))

            # ==== FIXME: Try using mixed loss??
            if self.config.use_mixed_loss:
                ploss = self.mixed_loss(ploss, self.plaq_weight)
            else:
                ploss = tf.reduce_mean(-ploss / self.plaq_weight, axis=0)

        # Calculate the charge loss
        qloss = 0.
        if self.charge_weight > 0:
            q_init = tf.reduce_sum(tf.sin(wl_init), axis=(1, 2)) / (2 * np.pi)
            q_prop = tf.reduce_sum(tf.sin(wl_prop), axis=(1, 2)) / (2 * np.pi)
            qloss = (accept_prob * (q_prop - q_init) ** 2) + 1e-4
            if self.config.use_mixed_loss:
                qloss = self.mixed_loss(qloss, self.charge_weight)
            else:
                qloss = tf.reduce_mean(-qloss / self.charge_weight, axis=0)

        return ploss, qloss

    def train_step(self, inputs):
        """Perform a single training step."""
        start = time.time()
        with tf.GradientTape() as tape:
            x, beta = inputs
            tape.watch(x)
            states, data = self((x, beta), training=True)
            accept_prob = data.get('accept_prob', None)
            ploss, qloss = self.calc_losses(states, accept_prob)
            loss = ploss + qloss
            if self.aux_weight > 0:
                z = tf.random.normal(x.shape, dtype=x.dtype)
                states_, data_ = self((z, beta), training=True)
                accept_prob_ = data_.get('accept_prob', None)
                ploss_, qloss_ = self.calc_losses(states_, accept_prob_)
                loss += ploss_ + qloss_

        if NUM_RANKS > 1:
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables),
        )

        metrics = AttrDict({
            'dt': time.time() - start,
            'loss': loss,
        })
        if self.plaq_weight > 0 and self.charge_weight > 0:
            metrics.update({
                'ploss': ploss,
                'qloss': qloss
            })
        if self.aux_weight > 0:
            metrics.update({
                'ploss_aux': ploss_,
                'qloss_aux': qloss_
            })

        metrics.update({
            'accept_prob': accept_prob,
            'eps': self.eps,
            'beta': states.init.beta,
        })

        if self._verbose:
            metrics.update({
                'Hf_start': data.forward.energies[0],
                'Hf_mid': data.forward.energies[self.config.num_steps//2],
                'Hf_end': data.forward.energies[-1],
                'Hb_start': data.backward.energies[0],
                'Hb_mid': data.backward.energies[self.config.num_steps//2],
                'Hb_end': data.backward.energies[-1],
                'ld_f_start': data.forward.logdets[0],
                'ld_f_mid': data.forward.logdets[self.config.num_steps//2],
                'ld_f_end': data.forward.logdets[-1],
                'ld_b_start': data.backward.logdets[0],
                'ld_b_mid': data.backward.logdets[self.config.num_steps//2],
                'ld_b_end': data.backward.logdets[-1],
                #  'sumlogdet': sumlogdet.out,
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
        if self.optimizer.iterations == 0 and NUM_RANKS > 1:
            hvd.broadcast_variables(self.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        return states.out.x, metrics

    def test_step(self, data):
        """Perform a single inference step."""
        start = time.time()
        states, data = self(data, training=False)
        accept_prob = data.get('accept_prob', None)
        ploss, qloss = self.calc_losses(states, accept_prob)
        loss = ploss + qloss

        metrics = AttrDict({
            'dt': time.time() - start,
            'loss': loss,
        })
        if self.plaq_weight > 0 and self.charge_weight > 0:
            metrics.update({
                'ploss': ploss,
                'qloss': qloss
            })

        metrics.update({
            'accept_prob': accept_prob,
            'eps': self.eps,
            'beta': states.init.beta,
        })

        if self._verbose:
            metrics.update({
                'Hf_start': data.forward.energies[0],
                'Hf_mid': data.forward.energies[self.config.num_steps//2],
                'Hf_end': data.forward.energies[-1],
                'Hb_start': data.backward.energies[0],
                'Hb_mid': data.backward.energies[self.config.num_steps//2],
                'Hb_end': data.backward.energies[-1],
                'ld_f_start': data.forward.logdets[0],
                'ld_f_mid': data.forward.logdets[self.config.num_steps//2],
                'ld_f_end': data.forward.logdets[-1],
                'ld_b_start': data.backward.logdets[0],
                'ld_b_mid': data.backward.logdets[self.config.num_steps//2],
                'ld_b_end': data.backward.logdets[-1],
                #  'sumlogdet': sumlogdet.out,
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

    def save_config(self, config_dir: str):
        """Helper method for saving configuration objects."""
        io.save_dict(self.config, config_dir, name='dynamics_config')
        io.save_dict(self.net_config, config_dir, name='network_config')
        io.save_dict(self.lr_config, config_dir, name='lr_config')
        io.save_dict(self.params, config_dir, name='dynamics_params')
        if self.conv_config is not None and self.config.use_conv_net:
            io.save_dict(self.conv_config, config_dir, name='conv_config')

    def get_config(self):
        """Get configuration as dict."""
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
