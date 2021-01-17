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
    HAS_HOROVOD = True
except ImportError:
    from utils import Horovod as hvd
    HAS_HOROVOD = False

NUM_RANKS = hvd.size()
NUM_WORKERS = hvd.size()

import utils.file_io as io

from config import BIN_DIR
from lattice.gauge_lattice import GaugeLattice, Charges
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
    config = GaugeDynamicsConfig(**dict(flags.get('dynamics_config', None)))

    lr_config = None
    if flags.get('lr_config', None) is not None:
        lr_config = LearningRateConfig(**dict(flags.get('lr_config', None)))

    net_config = None
    if flags.get('network_config', None) is not None:
        net_config = NetworkConfig(**dict(flags.get('network_config', None)))

    conv_config = None
    if flags.get('conv_config', None) is not None and config.use_conv_net:
        conv_config = flags.get('conv_config', None)
        input_shape = config.x_shape[1:]
        conv_config.input_shape = input_shape

    dynamics = GaugeDynamics(
        params=flags,
        config=config,
        network_config=net_config,
        lr_config=lr_config,
        conv_config=conv_config,
    )

    log_dir = flags['dynamics_config'].get('log_dir', None)
    if log_dir is not None and log_dir != '':
        io.log(
            '\n'.join([120*'#', f'LOADING NETWORKS FROM: {log_dir}', 120*'#'])
        )
        io.log(f'LOADING NETWORKS FROM: {log_dir}  !!!')
        io.log(120 * '#')
        xnet, vnet = dynamics._load_networks(log_dir)
        dynamics.xnet = xnet
        dynamics.vnet = vnet

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
    ):
        # ====
        # Set attributes from `config`
        self.aux_weight = config.aux_weight
        self.plaq_weight = config.plaq_weight
        self.charge_weight = config.charge_weight
        self._gauge_eq_masks = config.gauge_eq_masks
        self.lattice_shape = config.x_shape
        #  self.lattice_shape = config.lattice_shape
        self._combined_updates = config.combined_updates
        self._alpha = tf.constant(1.)

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
            if self.config.eps_fixed:
                self._has_trainable_params = False

            self.xnet, self.vnet = self._build_hmc_networks()

        else:
            if self.config.use_ncp:
                net_weights = NetWeights(1., 1., 1., 1., 1., 1.)
            else:
                net_weights = NetWeights(0., 1., 1., 1., 1., 1.)

            self.xnet, self.vnet = self._build_networks(
                net_config=self.net_config,
                conv_config=self.conv_config,
                #  log_dir=self.config.get('log_dir', None)
            )

        self.net_weights = self._parse_net_weights(net_weights)
        if self._has_trainable_params:
            self.lr_config = lr_config
            self.lr = self._create_lr(lr_config, auto=True)
            self.optimizer = self._create_optimizer()

    def _load_eps(
            self,
            log_dir: str = None,
    ) -> (list, list):
        """Load `xeps` and `veps` from saved model."""
        models_dir = os.path.join(log_dir, 'training', 'models')
        if not os.path.isdir(models_dir):
            raise ValueError('Unable to locate `models_dir`: {models_dir}')

        veps_file = os.path.join(models_dir, 'veps.z')
        xeps_file = os.path.join(models_dir, 'xeps.z')
        xeps = io.loadz(xeps_file)
        veps = io.loadz(veps_file)

        xeps = list(xeps)
        veps = list(veps)

        return xeps, veps

    def _load_networks(
            self,
            log_dir: str = None
    ) -> ([tf.keras.Model], [tf.keras.Model]):
        """Load networks from `log_dir`.

        Builds new networks if unable to load or
        self.config.num_steps > # networks available to load.
        """
        models_dir = os.path.join(log_dir, 'training', 'models')
        if not os.path.isdir(models_dir):
            raise ValueError('Unable to locate `models_dir: {models_dir}`')

        try:
            _lcfgs = dict(io.loadz(os.path.join(log_dir, 'configs.z')))
            _lnum_steps = _lcfgs['dynamics_config']['num_steps']
        except (FileNotFoundError, KeyError):
            _lnum_steps = self.config.num_steps

        if self.config.num_steps != _lnum_steps:
            io.log('Mismatch between '
                   f'self.config.num_steps = {self.config.num_steps} and '
                   f'loaded_config.num_steps = {_lnum_steps}', level='WARNING')

        if self.config.separate_networks:
            vnets = []
            xnets_first = []
            xnets_second = []
            xnet = []
            vnet = []
            for i in range(self.config.num_steps):
                vp = os.path.join(models_dir, f'dynamics_vnet{i}')
                if os.path.isdir(vp):  # Load vnet from vp...
                    vnets.append(vp)
                    print(f'Loading vnet{i}_second from: {vp}...')
                    vnet.append(tf.keras.models.load_model(vp))

                    # Check if xnet_first and xnet_second networks exist...
                    xp0 = os.path.join(models_dir, f'dynamics_xnet_first{i}')
                    xp1 = os.path.join(models_dir, f'dynamics_xnet_second{i}')
                    if os.path.isdir(xp0) and os.path.isdir(xp1):
                        # If so, load them into `xnets_first`, `xnets_second`
                        xnets_first.append(xp0)
                        xnets_second.append(xp1)
                        print(f'Loading xnet{i}_first from: {xp0}...')
                        print(f'Loading xnet{i}_second from: {xp1}...')
                        xnet.append(
                            (tf.keras.models.load_model(xp0),
                             tf.keras.models.load_model(xp1))
                        )
                else:
                    print(f'Unable to load model from {vp}...')
                    print(f'Initializing new network for step {i}')
                    xnet_, vnet_ = self._build_network(step=i)
                    xnet.append(xnet_)
                    vnet.append(vnet_)

        return xnet, vnet

    def save_networks(self, log_dir):
        """Save networks to disk."""
        models_dir = os.path.join(log_dir, 'training', 'models')
        io.check_else_make_dir(models_dir)
        veps_file = os.path.join(models_dir, 'veps.z')
        xeps_file = os.path.join(models_dir, 'xeps.z')

        io.savez([e.numpy() for e in self.veps], veps_file, name='veps')
        io.savez([e.numpy() for e in self.xeps], xeps_file, name='xeps')
        if self.config.separate_networks:
            xnet_first_paths = [
                os.path.join(models_dir, f'dynamics_xnet_first{i}')
                for i in range(self.config.num_steps)
            ]
            xnet_second_paths = [
                os.path.join(models_dir, f'dynamics_xnet_second{i}')
                for i in range(self.config.num_steps)
            ]

            vnet_paths = [
                os.path.join(models_dir, f'dynamics_vnet{i}')
                for i in range(self.config.num_steps)
            ]

            paths = zip(xnet_first_paths, xnet_second_paths, vnet_paths)
            #  for idx, (xf0, xf1, vf) in enumerate(zip(xnet_paths, vnet_paths)):
            for idx, (xf0, xf1, vf) in enumerate(paths):
                xnets = self.xnet[idx]  # type: tf.keras.models.Model
                vnet = self.vnet[idx]  # type: tf.keras.models.Model
                #  io.log(f'Saving `xnet_first{idx}` to {xf0}.')
                #  io.log(f'Saving `xnet_second{idx}` to {xf1}.')
                #  io.log(f'Saving `vnet{idx}` to {vf}.')
                xnets[0].save(xf0)
                xnets[1].save(xf1)
                vnet.save(vf)
        else:
            xnet_paths = os.path.join(models_dir, 'dynamics_xnet')
            vnet_paths = os.path.join(models_dir, 'dynamics_vnet')
            #  io.log(f'Saving `xnet` to {xnet_paths}.')
            #  io.log(f'Saving `vnet` to {vnet_paths}.')
            self.xnet.save(xnet_paths)
            self.vnet.save(vnet_paths)

    def _get_network_configs(
            self,
            net_config: NetworkConfig,
            conv_config: ConvolutionConfig
    ):
        """Returns `cfgs` for passing to `get_gauge_network`."""
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

        # ==== xNet
        xnet_cfg = {
            'factor': 2.0,
            'net_config': net_config,
            'conv_config': conv_config,
            'kernel_initializer': kinit,
            'x_shape': self.lattice_shape,
            'input_shapes': {
                'x': xshape, 'v': (self.xdim,), 't': (2,)
            }
        }

        # ==== vNet
        vnet_cfg = {
            'factor': 1.0,
            'net_config': net_config,
            'conv_config': conv_config,
            'kernel_initializer': kinit,
            'x_shape': self.lattice_shape,
            'input_shapes': {
                'x': (self.xdim,), 'v': (self.xdim,), 't': (2,)
            }
        }

        return AttrDict({'xnet': xnet_cfg, 'vnet': vnet_cfg})

    def _build_network(
            self,
            step: int = None,
            net_config: NetworkConfig = None,
            conv_config: ConvolutionConfig = None,
    ):
        """Build single instances of the position and momentum networks.

        Returns:
            xnet: tf.keras.models.Model
            vnet: tf.keras.models.Model
        """
        cfgs = self._get_network_configs(net_config, conv_config)

        if self.config.separate_networks:
            io.log('Using separate (x, v)-networks for each LF step!!')
            vnet = get_gauge_network(**cfgs['vnet'], name=f'vvet{step}')
            xnet = (
                get_gauge_network(**cfgs['xnet'], name=f'xnet_first{step}'),
                get_gauge_network(**cfgs['xnet'], name=f'xnet_second{step}')
            )

        else:
            io.log('Using a single (x, v)-network for all LF steps!!')
            vnet = get_gauge_network(**cfgs['vnet'], name='vnet')
            xnet = get_gauge_network(**cfgs['xnet'], name='xnet')

        return xnet, vnet

    def _build_networks(
            self,
            net_config: NetworkConfig = None,
            conv_config: ConvolutionConfig = None,
            log_dir: str = None,
    ):
        """Build position and momentum networks.

        If `self.config.separate_networks`, build an array of identical copies
        of `xnet`, `vnet` for each leapfrog step (generally makes the model
        more expressive).

        Otherwise, build a single instance of `xnet` and `vnet` to use for
        different leapfrog steps.

        Returns:
            xnet: tf.keras.models.Model
            vnet: tf.keras.models.Model
        """
        if log_dir is not None:
            return self._load_networks(log_dir)

        cfgs = self._get_network_configs(net_config, conv_config)

        if self.config.separate_networks:
            io.log('Using separate (x, v)-networks for each LF step!!')
            vnet = [
                get_gauge_network(**cfgs['vnet'], name=f'vnet{i}')
                for i in range(self.config.num_steps)
            ]
            xnet = [
                (get_gauge_network(**cfgs['xnet'], name=f'xnet_first{i}'),
                 get_gauge_network(**cfgs['xnet'], name=f'xnet_second{i}'))
                for i in range(self.config.num_steps)
            ]

        else:
            io.log('Using a single (x, v)-network for all LF steps!!')
            vnet = get_gauge_network(**cfgs['vnet'], name='vnet')
            xnet = get_gauge_network(**cfgs['xnet'], name='xnet')

        return xnet, vnet

    def _init_metrics(self, state: State) -> (tf.TensorArray, tf.TensorArray):
        """Create logdet/energy metrics for verbose logging."""
        metrics = super()._init_metrics(state)

        if not self._verbose:
            return metrics

        kwargs = {
            'size': self.config.num_steps+1,
            'element_shape': (self.batch_size,),
            'dynamic_size': False,
            'clear_after_read': False
        }
        sinq = tf.TensorArray(TF_FLOAT, **kwargs)
        intq = tf.TensorArray(TF_FLOAT, **kwargs)
        plaqs = tf.TensorArray(TF_FLOAT, **kwargs)
        p4x4 = tf.TensorArray(TF_FLOAT, **kwargs)

        #  sinq = self.lattice.calc_charges(x=state.x, use_sin=True)
        plaqs_arr, charges, plaqs4x4_arr = self._calc_observables(state)
        plaqs = plaqs.write(0, plaqs_arr)
        p4x4 = p4x4.write(0, plaqs4x4_arr)
        sinq = sinq.write(0, charges.sinQ)
        intq = intq.write(0, charges.intQ)
        metrics.update({
            'sinQ': sinq,
            'intQ': intq,
            'plaqs': plaqs,
            'p4x4': p4x4,
            #  'sinQ': sinq.write(0, charges.sinQ),
            #  'intQ': intq.write(0, charges.intQ),
        })

        return metrics

    def _update_metrics(self, metrics, step, state, sumlogdet, **kwargs):
        """Write to metrics."""
        sinq = self.lattice.calc_charges(x=state.x, use_sin=True)
        metrics['sinQ'] = metrics['sinQ'].write(step, sinq)

        return super()._update_metrics(metrics, step, state,
                                       sumlogdet, **kwargs)

    def transition_kernel_directional(
            self,
            state: State,
            forward: bool,
            training: bool = None,
    ):
        """Implements a series of directional updates."""
        state_prop = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = tf.zeros((self.batch_size,))
        metrics = self._init_metrics(state_prop)

        def _update_metrics(data, step):
            for key, val in data.items():
                try:
                    metrics[key] = metrics[key].write(step, val)
                except AttributeError:
                    metrics[key][step] = val

            return metrics

        def _get_metrics(state, logdet):
            energy = self.hamiltonian(state)
            #  charges = self.lattice.calc_both_charges(x=state_prop.x)
            plaqs, charges, p4x4 = self._calc_observables(state)
            escaled = energy - logdet
            return {
                'H': energy, 'Hw': escaled, 'logdets': logdet,
                'sinQ': charges.sinQ, 'intQ': charges.intQ,
                'plaqs': plaqs, 'p4x4': p4x4,
                #  'sinQ': charges.sinQ, 'intQ': charges.intQ,
            }

        def _stack_metrics():
            for key, val in metrics.items():
                if isinstance(val, tf.TensorArray):
                    metrics[key] = val.stack()
            return metrics

        # ====
        # Forward for first half of trajectory
        for step in range(self.config.num_steps // 2):
            state_prop, logdet = self._forward_lf(step, state_prop, training)
            sumlogdet += logdet
            if self._verbose:
                data = _get_metrics(state_prop, sumlogdet)
                metrics = _update_metrics(data, step+1)

        # ====
        # Flip momentum
        state_prop = State(state_prop.x, -1. * state_prop.v, state_prop.beta)

        # ====
        # Backward for second half of trajectory
        for step in range(self.config.num_steps // 2, self.config.num_steps):
            state_prop, logdet = self._backward_lf(step, state_prop, training)
            sumlogdet += logdet
            if self._verbose:
                data = _get_metrics(state_prop, sumlogdet)
                metrics = _update_metrics(data, step+1)

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)
        metrics['sumlogdet'] = sumlogdet
        metrics['accept_prob'] = accept_prob
        if self._verbose:
            data = _get_metrics(state_prop, sumlogdet)
            metrics = _update_metrics(data, step+1)
            metrics = _stack_metrics()

        return state_prop, metrics

    def _transition_kernel_forward(
            self,
            state: State,
            training: bool = None
    ):
        """Run the augmented leapfrog sampler in the forward direction."""
        state_prop = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = tf.zeros((self.batch_size,))
        metrics = self._init_metrics(state_prop)

        def _update_metrics(data, step):
            for key, val in data.items():
                metrics[key] = metrics[key].write(step, val)

            return metrics

        def _get_metrics(state, logdet):
            energy = self.hamiltonian(state)
            charges = self.lattice.calc_both_charges(x=state_prop.x)
            escaled = energy - logdet
            return {
                'H': energy, 'Hw': escaled, 'logdets': logdet,
                'sinQ': charges.sinQ, 'intQ': charges.intQ,
            }

        def _stack_metrics():
            for key, val in metrics.items():
                if isinstance(val, tf.TensorArray):
                    metrics[key] = val.stack()
            return metrics

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
                    data = _get_metrics(state_prop, sumlogdet)
                    metrics = _update_metrics(data, step+1)

        state_prop, logdet = self._half_v_update_forward(state_prop,
                                                         step, training)
        sumlogdet += logdet

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)
        metrics['sumlogdet'] = sumlogdet
        metrics['accept_prob'] = accept_prob
        if self._verbose:
            data = _get_metrics(state_prop, sumlogdet)
            metrics = _update_metrics(data, step+1)
            metrics = _stack_metrics()

        return state_prop, metrics

    def _transition_kernel_backward(
            self,
            state: State,
            training: bool = None
    ):
        """Run the augmented leapfrog sampler in the forward direction."""
        state_prop = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = tf.zeros((self.batch_size,))
        metrics = self._init_metrics(state_prop)

        def _update_metrics(data, step):
            for key, val in data.items():
                metrics[key] = metrics[key].write(step, val)

            return metrics

        def _get_metrics(state, logdet):
            energy = self.hamiltonian(state)
            charges = self.lattice.calc_both_charges(x=state_prop.x)
            escaled = energy - logdet
            return {
                'H': energy, 'Hw': escaled, 'logdets': logdet,
                'sinQ': charges.sinQ, 'intQ': charges.intQ,
            }

        def _stack_metrics():
            for key, val in metrics.items():
                if isinstance(val, tf.TensorArray):
                    metrics[key] = val.stack()
            return metrics

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
                    data = _get_metrics(state_prop, sumlogdet)
                    metrics = _update_metrics(data, step+1)

        state_prop, logdet = self._half_v_update_backward(state_prop,
                                                          step, training)
        sumlogdet += logdet

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)
        metrics['sumlogdet'] = sumlogdet
        metrics['accept_prob'] = accept_prob
        if self._verbose:
            data = _get_metrics(state_prop, sumlogdet)
            metrics = _update_metrics(data, step+1)
            metrics = _stack_metrics()

        return state_prop, metrics

    @staticmethod
    def split_metrics_by_accept_reject(metrics, mask_a, mask_r=None):
        if mask_r is None:
            mask_r = 1. - mask_a

        metrics_a = {}
        metrics_r = {}
        for key, val in metrics.items():
            if len(val.shape) == 1:
                val_a = mask_a * val
                val_r = mask_r * val
            if len(val.shape) == 2:
                val_a = tf.convert_to_tensor([mask_a * i for i in val])
                val_r = tf.convert_to_tensor([mask_r * i for i in val])

            metrics_a[key] = val_a
            metrics_r[key] = val_r

        return {
            'accept': metrics_a,
            'reject': metrics_r
        }

    def _transition_kernel(
            self,
            state: State,
            forward: bool,
            training: bool = None,
    ):
        """Implements a transition kernel when using separate networks."""
        def _update_metrics(data, step):
            for key, val in data.items():
                metrics[key] = metrics[key].write(step, val)

            return metrics

        def _get_metrics(state, logdet):
            energy = self.hamiltonian(state)
            plaqs, charges, p4x4 = self._calc_observables(state)
            return {
                'H': energy, 'Hw': energy - logdet, 'logdets': logdet,
                'sinQ': charges.sinQ, 'intQ': charges.intQ,
                'plaqs': plaqs, 'p4x4': p4x4,
            }

        def _stack_metrics():
            for key, val in metrics.items():
                if isinstance(val, tf.TensorArray):
                    metrics[key] = val.stack()
            return metrics

        # -- Setup -------------------------------------------------
        lf_fn = self._forward_lf if forward else self._backward_lf
        state_prop = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = tf.zeros((self.batch_size,))
        metrics = self._init_metrics(state_prop)

        # -- Loop over leapfrog steps ----------------------------
        for step in range(self.config.num_steps):
            state_prop, logdet = lf_fn(step, state_prop, training)
            sumlogdet += logdet
            if self._verbose:
                data = _get_metrics(state_prop, sumlogdet)
                metrics = _update_metrics(data, step+1)

        # -- Compute accept prob and update metrics ------------------------
        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)
        metrics.update({'sumlogdet': sumlogdet, 'accept_prob': accept_prob})
        if self._verbose:
            data = _get_metrics(state_prop, sumlogdet)
            metrics = _update_metrics(data, step+1)
            metrics = _stack_metrics()

        return state_prop, metrics

    def transition_kernel(
            self,
            state: State,
            forward: bool,
            training: bool = None,
            verbose: bool = False,
    ):
        """Transition kernel of the augmented leapfrog integrator."""
        # -- NOTE --------------------------------------------------------
        # If using `self._combined_updates`, we combine the half-step
        # momentum-updates into a single full-step momentum updates in the
        # inner leapfrog steps.
        if self._combined_updates:
            return (
                self._transition_kernel_forward(state, training)
                if forward else
                self._transition_kernel_backward(state, training)
            )

        # ====
        # Using directional updates? (Experimental, not well tested!!)
        if self.config.directional_updates:
            return self.transition_kernel_directional(state, training)

        return self._transition_kernel(state, forward, training)

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

    def _convert_to_cartesian(self, x: tf.Tensor, mask: tf.Tensor):
        """Convert `x` from an angle to [cos(x), sin(x)]."""
        if len(mask) == 2:
            mask, _ = mask

        xcos = mask * tf.math.cos(x)
        xsin = mask * tf.math.sin(x)
        if self.config.use_conv_net:
            xcos = tf.reshape(xcos, self.lattice_shape)
            xsin = tf.reshape(xsin, self.lattice_shape)

        x = tf.stack([xcos, xsin], axis=-1)

        return x

    def _call_xnet(self, inputs, mask, step,
                   training=None, first: bool = False):
        """Call `self.xnet` to get Sx, Tx, Qx for updating `x`."""
        x, v, t = inputs
        x = self._convert_to_cartesian(x, mask)

        if not self.config.separate_networks:
            return self.xnet((x, v, t), training)

        xnet0, xnet1 = self.xnet[step]
        if first:
            return xnet0((x, v, t), training)

        return xnet1((x, v, t), training)

    def _full_v_update_forward(
            self,
            state: State,
            step: int,
            training: bool = None,
    ):
        """Perform a full-step momentum update in the forward direction."""
        eps = self.veps[step]
        x = self.normalizer(state.x)
        grad = self.grad_potential(x, state.beta)
        t = self._get_time(step, tile=tf.shape(x)[0])

        S, T, Q = self._call_vnet((x, grad, t), step, training)

        scale = self._vsw * (eps * S)
        transl = self._vtw * T
        transf = self._vqw * (eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        vf = state.v * expS - eps * (grad * expQ - transl)

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
        eps = self.veps[step]
        x = self.normalizer(state.x)
        grad = self.grad_potential(x, state.beta)
        t = self._get_time(step, tile=tf.shape(x)[0])

        S, T, Q = self._call_vnet((x, grad, t), step, training)

        scale = self._vsw * (0.5 * eps * S)
        transl = self._vtw * T
        transf = self._vqw * (eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        vf = state.v * expS - 0.5 * eps * (grad * expQ - transl)

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

        eps = self.veps[step]
        x = self.normalizer(state.x)
        grad = self.grad_potential(x, state.beta)
        t = self._get_time(step, tile=tf.shape(x)[0])

        S, T, Q = self._call_vnet((x, grad, t), step, training)

        scale = self._vsw * (0.5 * eps * S)
        transl = self._vtw * T
        transf = self._vqw * (eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        vf = state.v * expS - 0.5 * eps * (grad * expQ - transl)

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
                                               (m, mc), training, first=True)
        sumlogdet += logdet
        state, logdet = self._update_x_forward(state, step,
                                               (mc, m), training, first=False)
        sumlogdet += logdet

        return state, sumlogdet

    def _update_x_forward(
                self,
                state: State,
                step: int,
                masks: Tuple[tf.Tensor, tf.Tensor],  # [m, 1. - m]
                training: bool = None,
                first: bool = True,
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
        eps = self.xeps[step]
        x = self.normalizer(state.x)
        t = self._get_time(step, tile=tf.shape(x)[0])

        S, T, Q = self._call_xnet((x, state.v, t), m, step, training, first)

        scale = self._xsw * (eps * S)
        transl = self._xtw * T
        transf = self._xqw * (eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        if self.config.use_ncp:
            _x = 2 * tf.math.atan(tf.math.tan(x/2.) * expS)
            _y = _x + eps * (state.v * expQ + transl)
            xf = (m * x) + (mc * _y)

            cterm = tf.math.cos(x / 2) ** 2
            sterm = (expS * tf.math.sin(x / 2)) ** 2
            logdet_ = tf.math.log(expS / (cterm + sterm))
            logdet = tf.reduce_sum(mc * logdet_, axis=1)

        else:
            y = x * expS + eps * (state.v * expQ + transl)
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
        eps = self.veps[step_r]
        x = self.normalizer(state.x)
        grad = self.grad_potential(x, state.beta)
        t = self._get_time(step_r, tile=tf.shape(x)[0])
        S, T, Q = self._call_vnet((x, grad, t), step_r, training)

        scale = self._vsw * (-eps * S)
        transf = self._vqw * (eps * Q)
        transl = self._vtw * T

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        vb = expS * (state.v + eps * (grad * expQ - transl))

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
        eps = self.veps[step_r]
        x = self.normalizer(state.x)
        grad = self.grad_potential(x, state.beta)
        t = self._get_time(step_r, tile=tf.shape(x)[0])
        S, T, Q = self._call_vnet((x, grad, t), step_r, training)

        scale = self._vsw * (-0.5 * eps * S)
        transf = self._vqw * (eps * Q)
        transl = self._vtw * T

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        vb = expS * (state.v + 0.5 * eps * (grad * expQ - transl))

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
        eps = self.veps[step]
        x = self.normalizer(state.x)
        grad = self.grad_potential(x, state.beta)
        t = self._get_time(step, tile=tf.shape(x)[0])
        S, T, Q = self._call_vnet((x, grad, t), step, training)

        scale = self._vsw * (-0.5 * eps * S)
        transf = self._vqw * (eps * Q)
        transl = self._vtw * T

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        vb = expS * (state.v + 0.5 * eps * (grad * expQ - transl))

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
                training: bool = None,
                first: bool = True,
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
        eps = self.xeps[step]
        x = self.normalizer(state.x)
        t = self._get_time(step, tile=tf.shape(x)[0])
        S, T, Q = self._call_xnet((x, state.v, t), m, step, training, first)

        scale = self._xsw * (-eps * S)
        transl = self._xtw * T
        transf = self._xqw * (eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        if self.config.use_ncp:
            term1 = 2 * tf.math.atan(expS * tf.math.tan(state.x / 2))
            term2 = expS * eps * (state.v * expQ + transl)
            y = term1 - term2
            xb = (m * x) + (mc * y)

            cterm = tf.math.cos(x / 2) ** 2
            sterm = (expS * tf.math.sin(x / 2)) ** 2
            logdet_ = tf.math.log(expS / (cterm + sterm))
            logdet = tf.reduce_sum(mc * logdet_, axis=1)

        else:
            y = expS * (x - eps * (state.v * expQ + transl))
            xb = m * x + mc * y
            logdet = tf.reduce_sum(mc * scale, axis=1)

        xb = self.normalizer(xb)
        state_out = State(xb, v=state.v, beta=state.beta)
        return state_out, logdet

    @staticmethod
    def mixed_loss(loss: tf.Tensor, weight: float) -> (tf.Tensor):
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

    def _get_lr(self, step=None) -> (tf.Tensor):
        if step is None:
            step = self.optimizer.iterations

        #  if isinstance(self.lr, callable):
        if callable(self.lr):
            return self.lr(step)

        return K.get_value(self.optimizer.lr)

    @tf.function
    def train_step(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor],
    ) -> (tf.Tensor, AttrDict):
        """Perform a single training step.

        Returns:
            states.out.x (tf.Tensor): Next `x` state in the Markov Chain.
            metrics (AttrDict): Dictionary of various metrics for logging.
        """
        def _traj_summ(x, key=None):
            if key is not None:
                return {
                    f'{key}': tf.squeeze(x),
                    f'{key}_start': x[0],
                    f'{key}_mid': x[midpt],
                    f'{key}_end': x[-1],
                }

            return (x[0], x[midpt], x[1])

        start = time.time()
        with tf.GradientTape() as tape:
            x, beta = inputs
            tape.watch(x)
            states, metrics = self((x, beta), training=True)
            accept_prob = metrics.get('accept_prob', None)
            ploss, qloss = self.calc_losses(states, accept_prob)
            loss = ploss + qloss

            if self.aux_weight > 0:
                z = tf.random.normal(x.shape, dtype=x.dtype)
                states_, metrics_ = self((z, beta), training=True)
                accept_prob_ = metrics_.get('accept_prob', None)
                ploss_, qloss_ = self.calc_losses(states_, accept_prob_)
                loss += ploss_ + qloss_

        if HAS_HOROVOD:
            tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables),
        )

        # -- NOTE (Horovod) --------------------------------------------------
        # * Broadcast initial variable states from rank 0 to all other
        #   processes. This is necessary to ensure consistent initialization
        #   of all workers when training is started with random weights or
        #   restored from a checkpoint.
        # * Broadcast should be done after the first gradient step to ensure
        #   optimizer intialization.
        if self.optimizer.iterations == 0 and HAS_HOROVOD:
            hvd.broadcast_variables(self.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        data = AttrDict({
            #  'lr': self._get_lr(),
            'dt': time.time() - start,
            'loss': loss,
        })
        if self.plaq_weight > 0 and self.charge_weight > 0:
            data.update({
                'ploss': ploss,
                'qloss': qloss
            })
        if self.aux_weight > 0:
            data.update({
                'ploss_aux': ploss_,
                'qloss_aux': qloss_,
                'accept_prob_aux': accept_prob_,
            })

        midpt = self.config.num_steps // 2

        # Separated from [1038] for ordering when printing
        #  mask_a = metrics.get('accept_mask', None)
        data.update({
            'accept_prob': accept_prob,
            'accept_mask': metrics.get('accept_mask', None),
            'beta': states.init.beta,
            'sumlogdet': metrics.get('sumlogdet', None),
        })
        data.update(**_traj_summ(self.xeps, 'xeps'))
        data.update(**_traj_summ(self.veps, 'veps'))

        if self._verbose and not self.config.hmc:
            metricsf = metrics.get('forward', None)
            metricsb = metrics.get('backward', None)
            for (kf, vf), (kb, vb) in zip(metricsf.items(), metricsb.items()):
                data.update(**_traj_summ(vf, f'{kf}f'))
                data.update(**_traj_summ(vb, f'{kb}b'))

        data.update(metrics)
        data.update(self.calc_observables(states))
        #  data.update({k: v for k, v in metrics.items()})
        #  data.update({
        #      k: v for k, v in self.calc_observables(states).items()
        #  })
        #  data.update(**metrics)

        return states.out.x, data

    @tf.function
    def test_step(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor]
    ) -> (tf.Tensor, AttrDict):
        """Perform a single training step.

        Returns:
            states.out.x (tf.Tensor): Next `x` state in the Markov Chain.
            metrics (AttrDict): Dictionary of various metrics for logging.
        """
        def _traj_summ(x, key=None):
            """Helper fn for summarizing `x` along the trajectory"""
            #  return {f'{key}': tf.squeeze(x)} if key is not None
            #  if key is not None:
            #      return {
            #          f'{key}': tf.squeeze(x),
            #      }
            #
            #  return (x[0], x[midpt], x[1])
            return (x[0], x[midpt], x[1]) if key is None else {
                f'{key}': tf.squeeze(x)
            }

        start = time.time()
        x, beta = inputs
        states, metrics = self((x, beta), training=True)
        accept_prob = metrics.get('accept_prob', None)
        ploss, qloss = self.calc_losses(states, accept_prob)
        loss = ploss + qloss
        dt = time.time() - start
        data = AttrDict({'dt': dt, 'loss': loss})
        if self.plaq_weight > 0 and self.charge_weight > 0:
            data.update({'ploss': ploss, 'qloss': qloss})

        midpt = self.config.num_steps // 2
        data.update({
            'accept_prob': accept_prob,
            'beta': states.init.beta,
        })
        data.update(**_traj_summ(self.xeps, 'xeps'))
        data.update(**_traj_summ(self.veps, 'veps'))

        if self._verbose and not self.config.hmc:
            for (kf, vf), (kb, vb) in zip(metrics.forward.items(),
                                          metrics.backward.items()):
                data.update(**_traj_summ(vf, f'{kf}f'))
                data.update(**_traj_summ(vb, f'{kb}b'))
        data.update(metrics)
        data.update(self.calc_observables(states))
        #  data.update({k: v for k, v in metrics.items()})
        #  data.update({
        #      k: v for k, v in self.calc_observables(states).items()
        #  })

        return states.out.x, data

    def _calc_observables(
            self, state: State
    ) -> (tf.Tensor, Charges):
        """Calculate the observables for a particular state.

        NOTE: We track the error in the plaquette instead of the actual value.
        """
        wloops = self.lattice.calc_wilson_loops(state.x)
        wloops4x4 = self.lattice.calc_wilson_loops4x4(state.x)
        charges = self.lattice.calc_both_charges(x=state.x)
        plaqs = self.lattice.calc_plaqs(wloops=wloops, beta=state.beta)
        plaqs4x4 = self.lattice.calc_plaqs4x4(wloops=wloops4x4,
                                              beta=state.beta)

        return plaqs, charges, plaqs4x4

    def calc_observables(
            self,
            states: MonteCarloStates
    ) -> (AttrDict):
        """Calculate observables."""
        _, q_init, _ = self._calc_observables(states.init)
        plaqs, q_out, p4x4 = self._calc_observables(states.out)
        dqsin = tf.math.abs(q_out.sinQ - q_init.sinQ)
        dqint = tf.math.abs(q_out.intQ - q_init.intQ)

        observables = AttrDict({
            'dq_int': dqint,
            'dq_sin': dqsin,
            'charges': q_out.intQ,
            'plaqs': plaqs,
            'p4x4': p4x4,
        })

        return observables

    def save_config(self, config_dir: str):
        """Helper method for saving configuration objects."""
        io.save_dict(self.config.__dict__,
                     config_dir, name='dynamics_config')
        io.save_dict(self.net_config.__dict__,
                     config_dir, name='network_config')
        io.save_dict(self.lr_config.__dict__,
                     config_dir, name='lr_config')
        io.save_dict(self.params,
                     config_dir, name='dynamics_params')
        io.save_dict(self.get_config(), config_dir, name='config')
        if self.conv_config is not None and self.config.use_conv_net:
            io.save_dict(self.conv_config.__dict__,
                         config_dir, name='conv_config')

    def get_config(self):
        """Get configuration as dict."""
        return {
            'config': self.config,
            'network_config': self.net_config,
            'conv_config': self.conv_config,
            'lr_config': self.lr_config,
            'params': self.params
        }

    def _get_time(self, i, tile=1):
        """Format the MCMC step as:
           ```
           [cos(2 * step/num_steps), sin(2pi*step/num_steps)]
           ```
        """
        #  if self.config.separate_networks:
        #      trig_t = tf.squeeze([0, 0])
        #  else:
        #  i = tf.cast(i, dtype=TF_FLOAT)
        trig_t = tf.squeeze([
            tf.cos(2 * np.pi * i / self.config.num_steps),
            tf.sin(2 * np.pi * i / self.config.num_steps),
        ])

        t = tf.tile(tf.expand_dims(trig_t, 0), (tile, 1))

        return t

    def _build_conv_mask(self):
        """Construct checkerboard mask with size L * L and 2 channels."""
        _, tsize, xsize, channels = self.lattice_shape
        arr = np.linspace(0, xsize * (xsize + 1) - 1, xsize * (xsize + 1))
        mask = (arr % 2 == 1).reshape(xsize, xsize+1)[:, :-1]
        mask_conj = ~mask
        x_mask = np.stack([mask, mask_conj], axis=0)
        x_mask = x_mask.reshape(1, channels, xsize, xsize)
        x_mask_conj = ~x_mask

        return tf.convert_to_tensor(x_mask), tf.convert_to_tensor(x_mask_conj)

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
