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
# pylint:disable=no-name-in-module
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

from tensorflow.python.keras import backend as K

try:
    import horovod.tensorflow as hvd

    NUM_RANKS = hvd.size()
    NUM_WORKERS = NUM_RANKS * hvd.local_size()
    HAS_HOROVOD = True
    print(f'hvd.size : {hvd.size()}')
    print(f'hvd.local_size: {hvd.local_size()}')

except (ImportError, ModuleNotFoundError):
    NUM_RANKS = 1
    NUM_WORKERS = NUM_RANKS
    HAS_HOROVOD = False

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
    log_dir = flags['dynamics_config'].pop('log_dir', None)
    config = GaugeDynamicsConfig(**dict(flags.get('dynamics_config', None)))
    #  config = GaugeDynamicsConfig(**dict(flags.get('dynamics_config', None)))
    net_config = NetworkConfig(**dict(flags.get('network_config', None)))
    conv_config = None

    if config.get('use_conv_net', False):
        conv_config = flags.get('conv_config', None)
        input_shape = config.get('lattice_shape', None)[1:]
        conv_config.update({
            'input_shape': input_shape,
        })
        conv_config = ConvolutionConfig(**dict(conv_config))

    dynamics = GaugeDynamics(
        params=flags,
        config=config,
        network_config=net_config,
        lr_config=lr_config,
        conv_config=conv_config,
        log_dir=log_dir
    )

    return dynamics


class GaugeDynamics(BaseDynamics):
    """Implements the dynamics engine for the L2HMC sampler."""

    def __init__(
            self,
            params: AttrDict,
            config: GaugeDynamicsConfig,
            network_config: Optional[NetworkConfig] = None,
            lr_config: Optional[LearningRateConfig] = None,
            conv_config: Optional[ConvolutionConfig] = None,
            log_dir: Optional[str] = None,
    ):
        # ====
        # Set attributes from `config`
        self.aux_weight = config.get('aux_weight', 0.)
        self.plaq_weight = config.get('plaq_weight', 0.)
        self.charge_weight = config.get('charge_weight', 0.01)
        self._gauge_eq_masks = config.get('gauge_eq_masks', False)
        self.lattice_shape = config.get('lattice_shape', None)
        self._combined_updates = config.get('combined_updates', False)
        self._alpha = tf.constant(1.)
        #  self._alpha = tf.Variable(initial_value=1., trainable=False)

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

            log_dir = self.config.get('log_dir', None)
            if log_dir is None:
                self.xnet, self.vnet = self._build_networks(self.net_config,
                                                            self.conv_config)
            else:
                io.log(f'Loading `xnet`, `vnet`, from {log_dir} !!')
                self.xnet, self.vnet = self._load_networks(log_dir)
            # ============

        self.net_weights = self._parse_net_weights(net_weights)
        if self._has_trainable_params:
            self.lr_config = lr_config
            self.lr = self._create_lr(lr_config, auto=True)
            self.optimizer = self._create_optimizer()

    def _load_networks(self, log_dir):
        models_dir = os.path.join(log_dir, 'training', 'models')
        #  xnet_paths = [
        #      os.path.join(models_dir, f'dynamics_xnet{i}')
        #      for i in range(self.config.num_steps)
        #  ]
        #  vnet_paths = [
        #      os.path.join(models_dir, f'dynamics_vnet{i}')
        #      for i in range(self.config.num_steps)
        #  ]
        xnet = []
        vnet = []
        for i in range(self.config.num_steps):
            xnet_path = os.path.join(models_dir, f'dynamics_xnet{i}')
            vnet_path = os.path.join(models_dir, f'dynamics_vnet{i}')
            if os.path.isdir(xnet_path) and os.path.isdir(vnet_path):
                print(f'Loading xNet{i} from: {xnet_path}...')
                xnet.append(tf.keras.models.load_model(xnet_path))
                print(f'Loading vNet{i} from: {vnet_path}...')
                vnet.append(tf.keras.models.load_model(vnet_path))
            else:
                print(f'Unable to load model from: {xnet_path}...')
                print(f'Creating new network for xNet{i}...')
                xnet_, vnet_ = self._build_network(step=i)
                xnet.append(xnet)
                vnet.append(vnet)

        return xnet, vnet

    def save_networks(self, log_dir):
        """Save networks to disk."""
        models_dir = os.path.join(log_dir, 'training', 'models')
        io.check_else_make_dir(models_dir)
        eps_file = os.path.join(models_dir, 'eps.z')
        io.savez(self.eps.numpy(), eps_file, name='eps')
        if self.config.separate_networks:
            xnet_paths = [
                os.path.join(models_dir, f'dynamics_xnet{i}')
                for i in range(self.config.num_steps)
            ]
            vnet_paths = [
                os.path.join(models_dir, f'dynamics_vnet{i}')
                for i in range(self.config.num_steps)
            ]
            for idx, (xf, vf) in enumerate(zip(xnet_paths, vnet_paths)):
                xnet = self.xnet[idx]  # type: tf.keras.models.Model
                vnet = self.vnet[idx]  # type: tf.keras.models.Model
                io.log(f'Saving `xnet{idx}` to {xf}.')
                io.log(f'Saving `vnet{idx}` to {vf}.')
                xnet.save(xf)
                vnet.save(vf)
        else:
            xnet_paths = os.path.join(models_dir, 'dynamics_xnet')
            vnet_paths = os.path.join(models_dir, 'dynamics_vnet')
            io.log(f'Saving `xnet` to {xnet_paths}.')
            io.log(f'Saving `vnet` to {vnet_paths}.')
            self.xnet.save(xnet_paths)
            self.vnet.save(vnet_paths)

    def _build_network(
            self,
            step: int = None,
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
            'conv_config': conv_config,
            'kernel_initializer': kinit,
            'lattice_shape': self.lattice_shape,
            'input_shapes': {
                'x': (self.xdim,), 'v': (self.xdim,), 't': (2,)
            }
        }

        vname = f'VNet{step}' if step is not None else 'VNet'
        xname = f'XNet{step}' if step is not None else 'XNet'
        vnet = get_gauge_network(**vnet_cfg, name=vname)
        xnet = get_gauge_network(**xnet_cfg, name=xname)

        return xnet, vnet

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
        if self.config.separate_networks:
            xnet = []
            vnet = []
            for step in range(self.config.num_steps):
                xnet_, vnet_ = self._build_network(step=step,
                                                   net_config=net_config,
                                                   conv_config=conv_config)
                xnet.append(xnet_)
                vnet.append(vnet_)

        else:
            xnet, vnet = self._build_network(net_config, conv_config)

        return xnet, vnet

    def transition_kernel_directional(
            self,
            state: State,
            forward: bool,
            training: bool = None,
    ):
        """Implements a series of directional updates."""
        state_prop = State(state.x, state.v, state.beta)
        sumlogdet = tf.zeros((self.batch_size,), dtype=TF_FLOAT)
        logdets = tf.TensorArray(TF_FLOAT,
                                 dynamic_size=True,
                                 size=self.batch_size,
                                 clear_after_read=False)
        energies = tf.TensorArray(TF_FLOAT,
                                  dynamic_size=True,
                                  size=self.batch_size,
                                  clear_after_read=False)
        # ====
        # Forward for first half of trajectory
        for step in range(self.config.num_steps // 2):
            if self._verbose:
                logdets = logdets.write(step, sumlogdet)
                energies = energies.write(step, self.hamiltonian(state_prop))

            state_prop, logdet = self._forward_lf(step, state_prop, training)
            sumlogdet += logdet

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
        logdets = tf.TensorArray(TF_FLOAT,
                                 dynamic_size=True,
                                 size=self.batch_size,
                                 clear_after_read=False)
        energies = tf.TensorArray(TF_FLOAT,
                                  dynamic_size=True,
                                  size=self.batch_size,
                                  clear_after_read=False)

        for step in range(self.config.num_steps):
            if self._verbose:
                energy = self.hamiltonian(state_prop)
                logdets = logdets.write(step, sumlogdet)
                energies = energies.write(step, energy)

            state_prop, logdet = lf_fn(step, state_prop, training)
            sumlogdet += logdet

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)

        metrics = AttrDict({
            'sumlogdet': sumlogdet,
            'accept_prob': accept_prob,
        })
        if self._verbose:
            logdets = logdets.write(step, sumlogdet)
            energies = energies.write(step, self.hamiltonian(state_prop))
            metrics.update({
                'H': [], 'Hw': [], 'logdets': [],
            })
            for i in range(self.config.num_steps):
                energy_ = energies.read(i)
                logdets_ = logdets.read(i)
                metrics['H'].append(energy_)
                metrics['logdets'].append(logdets_)
                metrics['Hw'].append(energy_ - logdets_)

            #  metrics.update({
            #      'energies_transformed': [
            #          (energies.read(i) - logdets.read(i))
            #          for i in range(self.config.num_steps)
            #      ]
            #  })
        #  metrics.update({
        #      'energies': {
        #          f'energy_lf{i}': energies.read(i)
        #          energies.read(i) for i in range(self.config.num_steps)
        #      },
        #      'logdets': [
        #          logdets.read(i) for i in range(self.config.num_steps)
        #      ],
        #  })

        return state_prop, metrics

    def _transition_kernel_forward(
            self,
            state: State,
            training: bool = None
    ):
        """Run the augmented leapfrog sampler in the forward direction."""
        sumlogdet = tf.zeros((self.batch_size,))
        state_prop = State(state.x, state.v, state.beta)
        if self._verbose:
            kwargs = {
                'dynamic_size': True,
                'size': self.batch_size,
                'clear_after_read': False
            }
            logdets = tf.TensorArray(TF_FLOAT, **kwargs)
            energies = tf.TensorArray(TF_FLOAT, **kwargs)

            logdets = logdets.write(0, sumlogdet)
            energies = energies.write(0, self.hamiltonian(state_prop))

        state_prop, logdet = self._half_v_update_forward(state_prop,
                                                         0, training)
        sumlogdet += logdet

        for step in range(self.config.num_steps):
            state_prop, logdet = self._full_x_update_forward(state_prop,
                                                             step, training)
            sumlogdet += logdet

            if step < self.config.num_steps - 1:
                state_prop, logdet = self._full_v_update_forward(
                    state_prop, step, training
                )
                sumlogdet += logdet
                if self._verbose:
                    logdets = logdets.write(step + 1, sumlogdet)
                    energies = energies.write(step + 1,
                                              self.hamiltonian(state_prop))

        state_prop, logdet = self._half_v_update_forward(state_prop,
                                                         step, training)
        sumlogdet += logdet

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)

        metrics = AttrDict({
            'sumlogdet': sumlogdet,
            'accept_prob': accept_prob,
        })

        if self._verbose:
            logdets = logdets.write(step, sumlogdet)
            energies = energies.write(step, self.hamiltonian(state_prop))
            metrics.update({
                'H': [], 'Hw': [], 'logdets': [],
            })
            for i in range(self.config.num_steps):
                energy_ = energies.read(i)
                logdets_ = logdets.read(i)
                metrics['H'].append(energy_)
                metrics['logdets'].append(logdets_)
                metrics['Hw'].append(energy_ - logdets_)

        return state_prop, metrics

    def _transition_kernel_backward(
            self,
            state: State,
            training: bool = None
    ):
        """Run the augmented leapfrog sampler in the forward direction."""
        sumlogdet = tf.zeros((self.batch_size,))
        state_prop = State(state.x, state.v, state.beta)
        if self._verbose:
            kwargs = {
                'dynamic_size': True,
                'size': self.batch_size,
                'clear_after_read': False
            }
            logdets = tf.TensorArray(TF_FLOAT, **kwargs)
            energies = tf.TensorArray(TF_FLOAT, **kwargs)

            logdets = logdets.write(0, sumlogdet)
            energies = energies.write(0, self.hamiltonian(state_prop))

        state_prop, logdet = self._half_v_update_backward(state_prop,
                                                          0, training)
        sumlogdet += logdet
        for step in range(self.config.num_steps):
            state_prop, logdet = self._full_x_update_backward(state_prop,
                                                              step, training)
            sumlogdet += logdet

            if step < self.config.num_steps - 1:
                state_prop, logdet = self._full_v_update_backward(
                    state_prop, step, training
                )
                sumlogdet += logdet

                if self._verbose:
                    logdets = logdets.write(step+1, sumlogdet)
                    energy = self.hamiltonian(state_prop)
                    energies = energies.write(step+1, energy)

        state_prop, logdet = self._half_v_update_backward(state_prop,
                                                          step, training)
        sumlogdet += logdet

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)

        metrics = AttrDict({
            'sumlogdet': sumlogdet,
            'accept_prob': accept_prob,
        })

        if self._verbose:
            logdets = logdets.write(step, sumlogdet)
            energies = energies.write(step, self.hamiltonian(state_prop))
            metrics.update({
                'H': [], 'Hw': [], 'logdets': [],
            })
            for i in range(self.config.num_steps):
                energy_ = energies.read(i)
                logdets_ = logdets.read(i)
                metrics['H'].append(energy_)
                metrics['logdets'].append(logdets_)
                metrics['Hw'].append(energy_ - logdets_)

            #  logdets = logdets.write(self.config.num_steps, sumlogdet)
            #  energies = energies.write(self.config.num_steps,
            #                            self.hamiltonian(state_prop))
            #  metrics.update({
            #      'energies': [
            #          energies.read(i) for i in range(self.config.num_steps)
            #      ],
            #      'logdets': [
            #          logdets.read(i) for i in range(self.config.num_steps)
            #      ],
            #  })

        return state_prop, metrics

    def transition_kernel(
            self,
            state: State,
            forward: bool,
            training: bool = None,
            verbose: bool = False,
    ):
        """Transition kernel of the augmented leapfrog integrator."""
        step = self.optimizer.iterations
        if self.config.separate_networks:
            if self._combined_updates:
                if forward:
                    if step == 0 and verbose:
                        print('Using `self._transition_kernel_forward`  !!')
                    return self._transition_kernel_forward(state, training)

                if step == 0 and verbose:
                    print('Using `self._transition_kernel_backward`  !!')
                return self._transition_kernel_backward(state, training)

            if step == 0 and verbose:
                print('Using `self._transition_kernel_sep_nets` !!')
            return self.transition_kernel_sep_nets(state, forward, training)

        if self.config.directional_updates:
            if step == 0 and verbose:
                print('Using `self._transition_kernel_directional` !!')
            return self.transition_kernel_directional(state, training)

        if step == 0 and verbose:
            print('Using `super().transition_kernel` !!')

        return super().transition_kernel(state, forward, training)

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
        if not self.config.separate_networks:
            return self.vnet(inputs, training)

        vnet = self.vnet[step]
        return vnet(inputs, training)

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

        x = tf.stack([x_cos, x_sin], axis=-1)
        #  x = tf.concat([x_cos, x_sin], -1)
        if not self.config.separate_networks:
            return self.xnet((x, v, t), training)

        xnet = self.xnet[step]
        return xnet((x, v, t), training)

    def _full_v_update_forward(
            self,
            state: State,
            step: int,
            training: bool = None,
    ):
        """Perform a full-step momentum update in the forward direction."""
        x = self.normalizer(state.x)
        grad = self.grad_potential(x, state.beta)
        t = self._get_time(step, tile=tf.shape(x)[0])

        S, T, Q = self._call_vnet((x, grad, t), step, training)

        scale = self._vsw * (self.eps * S)
        transl = self._vtw * T
        transf = self._vqw * (self.eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        vf = state.v * expS - self.eps * (grad * expQ - transl)

        state_out = State(x=x, v=vf, beta=state.beta)
        logdet = tf.reduce_sum(scale, axis=1)

        return state_out, logdet

    def _half_v_update_forward(
            self,
            state: State,
            step: int,
            training: bool = None,
    ):
        """Perform a half-step momentum update in the forward direction."""
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
        if self.config.hmc:
            return super()._update_v_forward(state, step, training)

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

    def _full_x_update_forward(
            self,
            state: State,
            step: int,
            training: bool = None
    ):
        """Perform a full-step position update in the forward direction."""
        m, mc = self._get_mask(step)
        sumlogdet = tf.zeros((self.batch_size,))
        state, logdet = self._update_x_forward(state, step,
                                               (m, mc), training)
        sumlogdet += logdet
        state, logdet = self._update_x_forward(state, step,
                                               (mc, m), training)
        sumlogdet += logdet

        return state, sumlogdet

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

    def _full_v_update_backward(
            self,
            state: State,
            step: int,
            training: bool = None
    ):
        """Perform a full update of the momentum in the backward direction."""
        step_r = self.config.num_steps - step - 1
        x = self.normalizer(state.x)
        grad = self.grad_potential(x, state.beta)
        t = self._get_time(step_r, tile=tf.shape(x)[0])
        S, T, Q = self._call_vnet((x, grad, t), step_r, training)

        scale = self._vsw * (-self.eps * S)
        transf = self._vqw * (self.eps * Q)
        transl = self._vtw * T

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        vb = expS * (state.v + self.eps * (grad * expQ - transl))

        state_out = State(x=x, v=vb, beta=state.beta)
        logdet = tf.reduce_sum(scale, axis=1)

        return state_out, logdet

    def _half_v_update_backward(
            self,
            state: State,
            step: int,
            training: bool = None
    ):
        """Perform a half update of the momentum in the backward direction."""
        step_r = self.config.num_steps - step - 1
        x = self.normalizer(state.x)
        grad = self.grad_potential(x, state.beta)
        t = self._get_time(step_r, tile=tf.shape(x)[0])
        S, T, Q = self._call_vnet((x, grad, t), step_r, training)

        scale = self._vsw * (-0.5 * self.eps * S)
        transf = self._vqw * (self.eps * Q)
        transl = self._vtw * T

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        vb = expS * (state.v + 0.5 * self.eps * (grad * expQ - transl))

        state_out = State(x=x, v=vb, beta=state.beta)
        logdet = tf.reduce_sum(scale, axis=1)

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

    def _full_x_update_backward(
            self,
            state: State,
            step: int,
            training: bool = None
    ):
        """Perform a full-step position update in the backward direction."""
        step_r = self.config.num_steps - step - 1
        m, mc = self._get_mask(step_r)
        sumlogdet = tf.zeros((self.batch_size,))

        state, logdet = self._update_x_backward(state, step_r,
                                                (mc, m), training)
        sumlogdet += logdet

        state, logdet = self._update_x_backward(state, step_r,
                                                (m, mc), training)
        sumlogdet += logdet

        return state, sumlogdet

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

    def _get_lr(self, step=None):
        if step is None:
            step = self.optimizer.iterations

        #  if isinstance(self.lr, callable):
        if callable(self.lr):
            return self.lr(step)

        return K.get_value(self.optimizer.lr)

    def train_step(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor]
    ) -> (tf.Tensor, AttrDict):
        """Perform a single training step.

        Returns:
            states.out.x (tf.Tensor): Next `x` state in the Markov Chain.
            metrics (AttrDict): Dictionary of various metrics for logging.
        """
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

        if HAS_HOROVOD:
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables),
        )

        # Horovod:
        #    Broadcast initial variable states from rank 0 to all other
        #    processes. This is necessary to ensure consistent initialization
        #    of all workers when training is started with random weights or
        #    restored from a checkpoint.
        # NOTE:
        #    Broadcast should be done after the first gradient step to ensure
        #    optimizer intialization.
        if self.optimizer.iterations == 0 and HAS_HOROVOD:
            hvd.broadcast_variables(self.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        metrics = AttrDict({
            #  'lr': self._get_lr(),
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
                'qloss_aux': qloss_,
                'accept_prob_aux': accept_prob_,
            })

        # Separated from [1038] for ordering when printing
        metrics.update({
            'accept_prob': accept_prob,
            'eps': self.eps,
            'beta': states.init.beta,
            'sumlogdet': data.get('sumlogdet', None),
            'accept_mask': data.get('accept_mask', None),
        })

        if self._verbose:
            metrics.update({
                #  'Hf': data.forward.energies,
                'Hf': data.forward.H,
                'Hwf': data.forward.Hw,
                # ----
                'Hf_start': data.forward.H[0],
                'Hwf_start': data.forward.Hw[0],
                # ----
                'Hf_mid': data.forward.H[self.config.num_steps//2],
                'Hwf_mid': data.forward.Hw[self.config.num_steps//2],
                # ----
                'Hf_end': data.forward.H[-1],
                'Hwf_end': data.forward.Hw[-1],
                # ----
                'ldf_start': data.forward.logdets[0],
                'ldf_mid': data.forward.logdets[self.config.num_steps//2],
                'ldf_end': data.forward.logdets[-1],
                'sldf': data.forward.logdets,
                # ----
                'Hb': data.backward.H,
                'Hwb': data.backward.Hw,
                # ----
                'Hb_start': data.backward.H[0],
                'Hwb_start': data.backward.Hw[0],
                # ----
                'Hb_mid': data.backward.H[self.config.num_steps//2],
                'Hwb_mid': data.backward.Hw[self.config.num_steps//2],
                # ----
                'Hb_end': data.backward.H[-1],
                'Hwb_end': data.backward.Hw[-1],
                # ----
                'ldb_start': data.backward.logdets[0],
                'ldb_mid': data.backward.logdets[self.config.num_steps//2],
                'ldb_end': data.backward.logdets[-1],
                'sldb': data.backward.logdets,
            })

        observables = self.calc_observables(states)
        metrics.update(**observables)

        #  metrics.update({
        #      'lr': self._get_lr(),
        #  })

        return states.out.x, metrics

    def test_step(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor]
    ) -> (tf.Tensor, AttrDict):
        """Perform a single inference step.

        Returns:
            states.out.x (tf.Tensor): Next `x` state in the Markov Chain.
            metrics (AttrDict): Dictionary of various metrics for logging.
        """
        start = time.time()
        states, data = self(inputs, training=False)
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
            'sumlogdet': data.get('sumlogdet', None),
            'accept_mask': data.get('accept_mask', None),
        })

        if self._verbose:
            metrics.update({
                #  'Hf': data.forward.energies,
                'Hf': data.forward.H,
                'Hwf': data.forward.Hw,
                # ----
                'Hf_start': data.forward.H[0],
                'Hwf_start': data.forward.Hw[0],
                # ----
                'Hf_mid': data.forward.H[self.config.num_steps//2],
                'Hwf_mid': data.forward.Hw[self.config.num_steps//2],
                # ----
                'Hf_end': data.forward.H[-1],
                'Hwf_end': data.forward.Hw[-1],
                # ----
                'ldf_start': data.forward.logdets[0],
                'ldf_mid': data.forward.logdets[self.config.num_steps//2],
                'ldf_end': data.forward.logdets[-1],
                'sldf': data.forward.logdets,
                # ----
                'Hb': data.backward.H,
                'Hwb': data.backward.Hw,
                # ----
                'Hb_start': data.backward.H[0],
                'Hwb_start': data.backward.Hw[0],
                # ----
                'Hb_mid': data.backward.H[self.config.num_steps//2],
                'Hwb_mid': data.backward.Hw[self.config.num_steps//2],
                # ----
                'Hb_end': data.backward.H[-1],
                'Hwb_end': data.backward.Hw[-1],
                # ----
                'ldb_start': data.backward.logdets[0],
                'ldb_mid': data.backward.logdets[self.config.num_steps//2],
                'ldb_end': data.backward.logdets[-1],
                'sldb': data.backward.logdets,
            })

        observables = self.calc_observables(states)
        metrics.update(**observables)

        return states.out.x, metrics

    def _calc_observables(self, state: State):
        """Calculate the observables for a particular state.

        NOTE: We track the error in the plaquette instead of the actual value.
        """
        wloops = self.lattice.calc_wilson_loops(state.x)
        q_sin = self.lattice.calc_charges(wloops=wloops, use_sin=True)
        q_int = self.lattice.calc_charges(wloops=wloops, use_sin=False)
        plaqs = self.lattice.calc_plaqs(wloops=wloops, beta=state.beta)

        return plaqs, q_sin, q_int

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

    def _get_time(self, i, tile=1):
        """Format the MCMC step as:
           ```
           [cos(2pi*step/num_steps), sin(2pi*step/num_steps)]
           ```
        """
        #  if self.config.separate_networks:
        #      trig_t = tf.squeeze([0, 0])
        #  else:
        i = tf.cast(i, dtype=TF_FLOAT)
        trig_t = tf.squeeze([
            tf.cos(2 * np.pi * i / self.config.num_steps),
            tf.sin(2 * np.pi * i / self.config.num_steps),
        ])

        t = tf.tile(tf.expand_dims(trig_t, 0), (tile, 1))

        return t

    def _build_masks(self):
        """Construct different binary masks for different time steps."""
        def rolled_reshape(m, ax, shape=None):
            if shape is None:
                shape = (self.batch_size, -1)

            return sum([tf.roll(m, i, ax).reshape(shape) for i in range(4)])

        masks = []
        zeros = np.zeros(self.lattice_shape, dtype=np.float32)
        #  zeros = tf.zeros(self.lattice_shape, dtype=TF_FLOAT)

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
                mask = tf.reshape(tf.constant(m), (self.batch_size, -1))
                mask = tf.convert_to_tensor(mask)
                masks.append(mask)

        return masks
