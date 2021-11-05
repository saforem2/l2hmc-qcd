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
from __future__ import absolute_import, division, print_function, annotations
from dataclasses import asdict
from pathlib import Path

import sys
import os
import json
import time

from math import pi
from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend as K

from utils.hvd_init import SIZE, RANK, LOCAL_RANK, HAS_HOROVOD


try:
    import horovod.tensorflow as hvd
    COMPRESS = True
    HAS_HOROVOD = True
except ImportError:
    from utils import Horovod as hvd
    HAS_HOROVOD = False
    COMPRESS = False


here = os.path.dirname(__file__)
parent = os.path.abspath(os.path.dirname(here))
if parent not in sys.path:
    sys.path.append(parent)


import utils.file_io as io
from utils.logger import Logger

from config import BIN_DIR
from lattice.gauge_lattice import GaugeLattice, Charges, area_law
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

logger = Logger()

PI = pi
TWO_PI = 2. * PI

def project_angle(x: tf.Tensor) -> tf.Tensor:
    """Returns the projection of an angle `x` from [-4pi, 4pi] to [-pi, pi]."""
    return x - TWO_PI * tf.math.floor((x + PI) / TWO_PI)


def convert_to_angle(x: tf.Tensor) -> tf.Tensor:
    """Returns x in -pi <= x < pi."""
    x = tf.math.floormod(x + PI, TWO_PI) - PI
    return x


def build_test_dynamics() -> GaugeDynamics:
    """Build quick test dynamics for debugging."""
    jfile = os.path.abspath(os.path.join(BIN_DIR, 'test_dynamics_flags.json'))
    with open(jfile, 'rt') as f:
        flags = json.load(f)
    flags = AttrDict(flags)
    return build_dynamics(flags)


def build_dynamics(configs: dict[str, Any]) -> GaugeDynamics:
    """Build dynamics using configs from FLAGS."""
    config = GaugeDynamicsConfig(**dict(configs.get('dynamics_config', None)))

    lr_config = None
    if configs.get('lr_config', None) is not None:
        lr_config = LearningRateConfig(**dict(configs.get('lr_config', None)))

    net_config = None
    if configs.get('network_config', None) is not None:
        net_config = NetworkConfig(**dict(configs.get('network_config', None)))

    conv_config = None
    if configs.get('conv_config', None) is not None and config.use_conv_net:
        conv_config = configs.get('conv_config', {})

        if isinstance(conv_config, ConvolutionConfig):
            conv_config = asdict(conv_config)

        conv_config['input_shape'] = config.x_shape[1:]
        if isinstance(conv_config, dict):
            conv_config = ConvolutionConfig(**conv_config)

    dynamics = GaugeDynamics(
        params=configs,
        config=config,
        network_config=net_config,
        lr_config=lr_config,
        conv_config=conv_config,
    )

    return dynamics


class GaugeDynamics(BaseDynamics):
    """Implements the dynamics engine for the L2HMC sampler."""

    def __init__(
            self,
            params: AttrDict,
            config: GaugeDynamicsConfig,
            network_config: NetworkConfig = None,
            lr_config: LearningRateConfig = None,
            conv_config: ConvolutionConfig = None,
    ):
        # -- Set attributes from `config` -----------
        self.aux_weight = config.aux_weight
        self.plaq_weight = config.plaq_weight
        self.charge_weight = config.charge_weight
        self._gauge_eq_masks = config.gauge_eq_masks
        self.lattice_shape = config.x_shape
        self._xshape = config.x_shape
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

        if HAS_HOROVOD:
            if COMPRESS:
                self._fp16 = hvd.Compression.fp16
            else:
                self._fp16 = hvd.Compression.none
        else:
            self._fp16 = None

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
            )

        if self.config.net_weights is None:
            self.net_weights = self._parse_net_weights(net_weights)
        else:
            net_weights = NetWeights(*self.config.net_weights)
            self.net_weights = self._parse_net_weights(net_weights)

        if not self.config.hmc and not self.config.eps_fixed:
            self.lr_config = lr_config
            self.lr = self._create_lr(lr_config, auto=True)
            self.optimizer = self._create_optimizer()

    def _set_net_weights(self, net_weights: NetWeights):
        self.net_weights = net_weights
        self._xsw = net_weights.x_scale
        self._xtw = net_weights.x_translation
        self._xqw = net_weights.x_transformation
        self._vsw = net_weights.v_scale
        self._vtw = net_weights.v_translation
        self._vqw = net_weights.v_transformation

    def _load_eps(self, log_dir: str):
        """Load xeps and veps from saved files."""

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

    def _load_networks(self, log_dir: str) -> dict:
        """Load networks from `log_dir`.

        Builds new networks if unable to load or
        self.config.num_steps > # networks available to load.
        """
        models_dir = os.path.join(log_dir, 'training', 'models')
        if not self.config.separate_networks:
            vp = os.path.join(models_dir, 'dynamics_vnet')
            vnet = [tf.keras.models.load_model(vp)]

            xp = os.path.join(models_dir, 'dynamics_xnet')
            xnet = [tf.keras.models.load_model(xp)]
        else:
            xnet, vnet = [], []
            for i in range(self.config.num_steps):
                vp = os.path.join(models_dir, rf'dynamics_vnet{i}')
                logger.debug(f'Loading vnet{i}_second from: {vp}...')
                vnet.append(tf.keras.models.load_model(vp))

                # Check if xnet_first and xnet_second networks exist...
                xp0 = os.path.join(models_dir, f'dynamics_xnet_first{i}')
                xp1 = os.path.join(models_dir, f'dynamics_xnet_second{i}')
                if os.path.isdir(xp0) and os.path.isdir(xp1):
                    # If so, load them into `xnets_first`, `xnets_second`
                    #  xnets_first.append(xp0)
                    #  xnets_second.append(xp1)
                    logger.debug(f'Loading xnet{i}_first from: {xp0}...')
                    logger.debug(f'Loading xnet{i}_second from: {xp1}...')
                    xnet.append(
                        (tf.keras.models.load_model(xp0),
                         tf.keras.models.load_model(xp1))
                    )

                else:
                    xp = os.path.join(models_dir, f'dynamics_xnet{i}')
                    xnet.append(tf.keras.models.load_model(xp))

        return {'xnet': xnet, 'vnet': vnet}

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
            for idx, (xf0, xf1, vf) in enumerate(paths):
                xnets = self.xnet[idx]  # type: tf.keras.models.Model
                vnet = self.vnet[idx]  # type: tf.keras.models.Model
                xnets[0].save(xf0)
                xnets[1].save(xf1)
                vnet.save(vf)
        else:
            xnet = self.xnet[0]  # type: tf.keras.models.Model
            vnet = self.vnet[0]  # type: tf.keras.models.Model
            xnet.save(os.path.join(models_dir, 'dynamics_xnet'))
            vnet.save(os.path.join(models_dir, 'dynamics_vnet'))

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

        # -- xNetwork ---------------------------------------------
        xshape = (self.xdim, 2)  # NOTE: x = [cos(x), sin(x)]
        if self.config.use_conv_net:
            xshape = (*self.lattice_shape[1:], 2)

        xnet_cfg = {
            'factor': 2.0,              # xFactor
            'net_config': net_config,
            'conv_config': conv_config,
            'x_shape': self.lattice_shape,
            # NOTE: input_shapes differ for xNet, vNet
            'input_shapes': {'x': xshape, 'v': (self.xdim,)}
        }

        # -- vNetwork ---------------------------------------------
        vnet_cfg = {
            'factor': 1.0,              # vFactor
            'net_config': net_config,
            'conv_config': None,        # use dense layers for vNet
            'x_shape': self.lattice_shape,
            # NOTE: input_shapes differ for xNet, vNet
            'input_shapes': {'x': (self.xdim,), 'v': (self.xdim,)}
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
        vn = f'vnet{step}' if self.config.separate_networks else 'vnet'
        vnet = get_gauge_network(**cfgs['vnet'], name=vn)

        if self.config.separate_networks:
            xnet = (
                get_gauge_network(**cfgs['xnet'], name=f'xnet_first{step}'),
                get_gauge_network(**cfgs['xnet'], name=f'xnet_second{step}')
            )
        else:
            xnet = get_gauge_network(**cfgs['xnet'], name=f'xnet')

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
            logger.debug('Using separate (x, v)-networks for each LF step!!')
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
            logger.debug('Using a single (x, v)-network for all LF steps!!')
            vnet = [get_gauge_network(**cfgs['vnet'], name='vnet')]
            xnet = [get_gauge_network(**cfgs['xnet'], name='xnet')]

        return xnet, vnet

    def _init_metrics(
            self,
            state: State,
    ) -> dict[str, tf.TensorArray]:
        """Create logdet/energy metrics for verbose logging."""
        metrics = super()._init_metrics(state)

        if not self._verbose:
            return metrics

        kwargs = {
            'size': self.config.num_steps+1,
            'element_shape': (state.x.shape[0],),
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
        sumlogdet = tf.zeros((state.x.shape[0],))
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
            plaqs, charges, p4x4 = self._calc_observables(state)
            escaled = energy - logdet
            return {
                'H': energy, 'Hw': escaled, 'logdets': logdet,
                'sinQ': charges.sinQ, 'intQ': charges.intQ,
                'plaqs': plaqs, 'p4x4': p4x4,
            }

        def _stack_metrics():
            for key, val in metrics.items():
                if isinstance(val, tf.TensorArray):
                    metrics[key] = val.stack()
            return metrics

        # -- Forward for first half of trajectory ---------------------------
        for step in range(self.config.num_steps // 2):
            state_prop, logdet = self._forward_lf(step, state_prop, training)
            sumlogdet += logdet
            if self._verbose:
                data = _get_metrics(state_prop, sumlogdet)
                metrics = _update_metrics(data, step+1)

        # -- Flip momentum --------------------------------------------------
        state_prop = State(state_prop.x, -1. * state_prop.v, state_prop.beta)

        # -- Backward for second half of trajectory -------------------------
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
        sumlogdet = tf.zeros((state.x.shape[0],))
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
            metrics = _update_metrics(data, step + 1)
            metrics = _stack_metrics()

        return state_prop, metrics

    def _transition_kernel_backward(
            self,
            state: State,
            training: bool = None
    ):
        """Run the augmented leapfrog sampler in the forward direction."""
        state_prop = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = tf.zeros((state.x.shape[0],))
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
        # -- Setup -------------------------------------------------
        lf_fn = self._forward_lf if forward else self._backward_lf
        state_prop = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = tf.zeros((state.x.shape[0],))
        metrics = self._init_metrics(state_prop)

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
        if len(mask) == 2:
            mask, _ = mask

        x, v = inputs
        shape = (x.shape[0], -1)
        m = tf.reshape(mask, shape)
        idxs = tf.where(m)
        _x = tf.reshape(tf.gather_nd(x, idxs), shape)
        _x = tf.concat([tf.math.cos(_x), tf.math.sin(_x)], axis=-1)
        if not self.config.separate_networks:
            S, T, Q = self.xnet((_x, v), training)
        else:
            xnet = self.xnet[step]
            S, T, Q = xnet((x, v), training)

        return S, T, Q

    def _call_vnet(self, inputs, step, training=None):
        """Call `self.xnet` to get Sx, Tx, Qx for updating `x`."""
        if self.config.hmc:
            return [tf.zeros_like(inputs[0]) for _ in range(3)]

        step = 0 if not self.config.separate_networks else step
        return self.vnet[step](inputs, training)

    def _convert_to_cartesian(self, x: tf.Tensor, mask: tf.Tensor):
        """Convert `x` from an angle to [cos(x), sin(x)]."""
        if mask.shape[0] == 2:
            mask, _ = mask

        xcos = mask * tf.math.cos(x)
        xsin = mask * tf.math.sin(x)
        if self.config.use_conv_net:
            xcos = tf.reshape(xcos, self.lattice_shape)
            xsin = tf.reshape(xsin, self.lattice_shape)

        x = tf.stack([xcos, xsin], axis=-1)

        return x

    def _call_xnet(
            self,
            inputs: tuple,
            mask: tf.Tensor,
            step: int,
            training: bool = None,
            first: bool = False
    ):
        """Call `self.xnet` to get Sx, Tx, Qx for updating `x`."""
        if self.config.hmc:
            return [tf.zeros_like(inputs[0]) for _ in range(3)]

        x, v = inputs
        x = self._convert_to_cartesian(x, mask)

        # -- self.xnet is a list of tf.keras.Models -----------
        step = 0 if not self.config.separate_networks else step
        xnet = self.xnet[step]
        # -- only a single xnet -------------------------------
        if callable(xnet):
            return xnet((x, v), training)
        # -- xnets split into even/odd updates-----------------
        if first:
            return xnet[0]((x, v), training)
        return xnet[1]((x, v), training)

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
        S, T, Q = self._call_vnet((x, grad), step, training)

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
        S, T, Q = self._call_vnet((x, grad), step, training)

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
        S, T, Q = self._call_vnet((x, grad), step, training)

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
        sumlogdet = tf.zeros((state.x.shape[0],))
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

        m, mc = masks
        eps = self.xeps[step]
        x = self.normalizer(state.x)

        S, T, Q = self._call_xnet((x, state.v), m, step, training, first)

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
        S, T, Q = self._call_vnet((x, grad), step_r, training)

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
        S, T, Q = self._call_vnet((x, grad), step_r, training)

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
        S, T, Q = self._call_vnet((x, grad), step, training)

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
        sumlogdet = tf.zeros((state.x.shape[0],))

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

        # Call `XNet` using `self._scattered_xnet`
        m, mc = masks
        eps = self.xeps[step]
        x = self.normalizer(state.x)
        S, T, Q = self._call_xnet((x, state.v), m, step, training, first)

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
        ploss = tf.constant(0.)
        if self.plaq_weight > 0:
            dwloops = 2 * (1. - tf.math.cos(wl_prop - wl_init))
            ploss = accept_prob * tf.reduce_sum(dwloops, axis=(1, 2))

            # ==== FIXME: Try using mixed loss??
            if self.config.use_mixed_loss:
                ploss = self.mixed_loss(ploss, self.plaq_weight)
            else:
                ploss = tf.reduce_mean(-ploss / self.plaq_weight, axis=0)

        # Calculate the charge loss
        qloss = tf.constant(0.)
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

        if callable(self.lr):
            return self.lr(step)

        return K.get_value(self.optimizer.lr)

    @tf.function
    def train_step(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, AttrDict]:
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
            tape = hvd.DistributedGradientTape(tape, compression=self._fp16)

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
            return (x[0], x[midpt], x[1]) if key is None else {
                f'{key}': tf.squeeze(x),
                f'{key}_start': x[0],
                f'{key}_mid': x[midpt],
                f'{key}_end': x[-1],
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
            'sumlogdet': metrics.get('sumlogdet', None)
        })
        data.update(**_traj_summ(self.xeps, 'xeps'))
        data.update(**_traj_summ(self.veps, 'veps'))

        if self._verbose and not self.config.hmc:
            mf = metrics.get('forward', None)
            mb = metrics.get('backward', None)
            for (kf, vf), (kb, vb) in zip(mf.items(), mb.items()):
                data.update(**_traj_summ(vf, f'{kf}f'))
                data.update(**_traj_summ(vb, f'{kb}b'))
        data.update(metrics)
        data.update(self.calc_observables(states))

        return states.out.x, data

    def _calc_observables(
            self, state: State
    ):
        """Calculate the observables for a particular state.

        NOTE: We track the error in the plaquette instead of the actual value.
        """
        wloops = self.lattice.calc_wilson_loops(state.x)
        p4x4_obs = self.lattice.calc_plaqs4x4(x=state.x, beta=state.beta)
        p4x4_err = p4x4_obs # - p4x4_exp
        charges = self.lattice.calc_both_charges(x=state.x)
        plaqs = self.lattice.calc_plaqs(wloops=wloops, beta=state.beta)

        return plaqs, charges, p4x4_err

    def calc_observables(
            self,
            states: MonteCarloStates
    ):
        """Calculate observables."""
        _, q_init, _ = self._calc_observables(states.init)
        plaqs, q_out, p4x4 = self._calc_observables(states.out)
        dqsin = tf.math.abs(q_out.sinQ - q_init.sinQ)
        dqint = tf.math.abs(q_out.intQ - q_init.intQ)

        observables = AttrDict({
            'dq_int': dqint,
            'dq_sin': dqsin,
            'charges': q_out.intQ,
            'sin_charges': q_out.sinQ,
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
            #  'params': params
        }

    def _get_time(self, i, tile=1):
        """Format the current leapfrog step as:
        ```
        [cos(2pi * step/num_steps), sin(2pi * step/num_steps)]
        ```
        for step in [0, 1, ..., num_steps], and reshape so that each chain in
        our batch of inputs gets a copy.
        """
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
        masks = []
        zeros = np.zeros(self.lattice_shape, dtype=np.float32)

        def rolled_reshape(m, ax, shape=None):
            if shape is None:
                shape = (self.batch_size, -1)

            return sum([tf.roll(m, i, ax).reshape(shape) for i in range(4)])

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
