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
# pylint:disable=unused-import
from __future__ import absolute_import, division, print_function

import os
import json
import time

from typing import NoReturn, Dict, Tuple, Union, Optional
from math import pi

import numpy as np
import tensorflow as tf

from config import (BIN_DIR, GaugeDynamicsConfig, lrConfig, NetWeights,
                    NetworkConfig, State, MonteCarloStates)
from dynamics.base_dynamics import BaseDynamics
#  from network.gauge_network import GaugeNetwork
from network.gauge_network_new import GaugeNetwork
from network.gauge_conv_network import ConvolutionConfig, GaugeNetworkConv2D
from utils.attr_dict import AttrDict
from utils.seed_dict import vnet_seeds, xnet_seeds
from lattice.gauge_lattice import GaugeLattice

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False

TIMING_FILE = os.path.join(BIN_DIR, 'timing_file.log')
TFLOAT = tf.keras.backend.floatx()
if TFLOAT == 'float32':
    TF_FLOAT = tf.float32
elif TFLOAT == 'float64':
    TF_FLOAT = tf.float64


INPUTS = Tuple[tf.Tensor, tf.Tensor]


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
    lr_config = lrConfig(**dict(flags.get('lr_config', None)))
    config = GaugeDynamicsConfig(**dict(flags.get('dynamics_config', None)))

    conv_config = flags.get('conv_config', None)
    conv_config.update({
        'input_shape': flags.get('lattice_shape', None)[1:]
    })
    conv_config = ConvolutionConfig(**conv_config)

    net_config = flags.get('network_config', None)
    activation = net_config.pop('activation', 'relu')
    if activation == 'tanh':
        activation_fn = tf.nn.tanh
    elif activation == 'leaky_relu':
        activation_fn = tf.nn.leaky_relu
    else:
        activation_fn = tf.nn.relu

    net_config.update({
        'name': 'GaugeDynamics',
        'activation_fn': activation_fn
    })
    net_config = NetworkConfig(**net_config)

    dynamics = GaugeDynamics(flags, config, net_config, lr_config, conv_config)

    return dynamics


def build_dynamics_old(flags):
    """Build dynamics using parameters from FLAGS."""
    activation = flags.get('activation', 'relu')
    if activation == 'tanh':
        activation_fn = tf.nn.tanh
    elif activation == 'leaky_relu':
        activation_fn = tf.nn.leaky_relu
    else:
        activation_fn = tf.nn.relu

    use_conv_net = flags.get('use_conv_net', False)
    lattice_shape = flags.get('lattice_shape', None)

    conv_config = None
    if use_conv_net:
        conv_config = ConvolutionConfig(
            input_shape=lattice_shape[1:],  # Ignore batch dimension
            filters=flags.get('filters', None),
            sizes=flags.get('sizes', None),
            pool_sizes=flags.get('pool_sizes', None),
            conv_activations=flags.get('conv_activations', None),
            conv_paddings=flags.get('conv_paddings', None),
            use_batch_norm=flags.get('use_batch_norm', None),
            name='ConvolutionBlock2D'
        )

    net_config = NetworkConfig(
        name='GaugeNetwork',
        units=flags.get('units', None),
        activation_fn=activation_fn,
        dropout_prob=flags.get('dropout_prob', 0.),
    )

    config = GaugeDynamicsConfig(
        model_type='GaugeModel',
        eps=flags.eps,
        hmc=flags.hmc,
        use_ncp=flags.get('use_ncp', False),
        num_steps=flags.num_steps,
        separate_networks=flags.get('separate_networks', False),
    )

    lr_config = lrConfig(
        init=hvd.size() * flags.get('lr_init', None),
        decay_rate=flags.get('lr_decay_rate', None),
        decay_steps=flags.get('lr_decay_steps', None),
        warmup_steps=flags.get('warmup_steps', 0),
    )

    flags = AttrDict({
        'horovod': flags.get('horovod', False),
        'plaq_weight': flags.get('plaq_weight', 0.),
        'charge_weight': flags.get('charge_weight', 0.),
        'lattice_shape': flags.get('lattice_shape', None),
        'use_conv_net': flags.get('use_conv_net', False)
    })

    dynamics = GaugeDynamics(flags, config, net_config,
                             lr_config, conv_config=conv_config)

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
            conv_config: Optional[ConvolutionConfig] = None
    ) -> NoReturn:

        self.aux_weight = params.get('aux_weight', 0.)
        self.plaq_weight = params.get('plaq_weight', 10.)
        self.charge_weight = params.get('charge_weight', 0.1)
        self.zero_init = params.get('zero_init', False)
        self._gauge_eq_masks = params.get('gauge_eq_masks', True)
        self.use_conv_net = params.get('use_conv_net', False)
        self.conv_config = conv_config
        self.config = config
        self.net_config = network_config
        self.lr_config = lr_config

        self.lattice_shape = params.get('lattice_shape', None)
        self.lattice = GaugeLattice(self.lattice_shape)
        self.batch_size = self.lattice_shape[0]
        self.xdim = np.cumprod(self.lattice_shape[1:])[-1]

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
        )

    def _build(self, params, config, network_config, lr_config, **kwargs):
        self.config = config
        self.net_config = network_config
        self.eps = self._build_eps(use_log=False)
        self.masks = self._build_masks()
        if self.config.hmc:
            net_weights = NetWeights(0., 0., 0., 0., 0., 0.)
            self.config.use_ncp = False
        else:
            if self.config.use_ncp:
                net_weights = NetWeights(1., 1., 1., 1., 1., 1.)
            else:
                net_weights = NetWeights(0., 1., 1., 1., 1., 1.)

        self.params = self._parse_params(params, net_weights=net_weights)
        if self.config.separate_networks:
            self.xnet_even, self.xnet_odd, self.vnet = self._build_networks()
        else:
            self.xnet, self.vnet = self._build_networks()
        if self._has_trainable_params:
            self.lr_config = lr_config
            self.lr = self._create_lr(lr_config)
            self.optimizer = self._create_optimizer()

    def apply_transition1(
            self,
            inputs: INPUTS,
            training: bool = None,
    ) -> (MonteCarloStates, tf.Tensor, MonteCarloStates):
        """Propose a new state and perform the accept/reject step.

        NOTE: We simulate the dynamics both forward and backward, and use
        sampled Bernoulli masks to compute the actual solutions.
        """
        x, beta = inputs
        # ====
        # sf(b)_init: initial state forward (backward)
        # sf(b)_prop: proposed state forward (backward)
        # pxf(b): acceptance probability in the forward (backward) direction
        # sldf(b): sumlogdet in the forward (backward) direction
        # ====
        sf_init, sf_prop, pxf, sldf = self._transition(inputs, forward=True,
                                                       training=training)
        sb_init, sb_prop, pxb, sldb = self._transition(inputs, forward=False,
                                                       training=training)
        # ====
        # Combine the forward/backward outputs;
        # these comprise the proposed configuration
        # mf_, mb_: forward, backward masks (respectively)
        # ====
        mf_, mb_ = self._get_direction_masks()
        mf = mf_[:, None]
        mb = mb_[:, None]
        v_init = mf * sf_init.v + mb * sb_init.v
        x_prop = mf * sf_prop.x + mb * sb_prop.x
        v_prop = mf * sf_prop.v + mb * sb_prop.v
        sld_prop = mf_ * sldf + mb_ * sldb

        # Compute the acceptance probability
        accept_prob = mf_ * pxf + mb_ * pxb

        # ma_: accept_mask; mr_: reject mask
        ma_, mr_ = self._get_accept_masks(accept_prob)
        ma = ma_[:, None]
        mr = mr_[:, None]

        # Construct the output configuration
        v_out = ma * v_prop + mr * v_init
        x_out = self.normalizer(ma * x_prop + mr * x)
        sld_out = ma_ * sld_prop  # NOTE: initial sumlogdet = 0

        state_init = State(x=x, v=v_init, beta=beta)
        state_prop = State(x=x_prop, v=v_prop, beta=beta)
        state_out = State(x=x_out, v=v_out, beta=beta)

        mc_states = MonteCarloStates(state_init, state_prop, state_out)
        sld_states = MonteCarloStates(0., sld_prop, sld_out)

        return mc_states, sld_states

    def transition_kernel_sep_nets(
            self,
            state: State,
            forward: bool,
            training: bool = None,
    ):
        """Transition kernel of the augmented leapfrog integrator.

        Returns:
            state_prop (State): Proposed state output from integrator.
            accept_prob (tf.Tensor): (batch-wise) Acceptance probability.
            sumlogdet (tf.Tensor): Total log determinant of the Jacobian
        """
        lf_fn = self._forward_lf if forward else self._backward_lf
        state_prop = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = tf.zeros((self.batch_size,), dtype=TF_FLOAT)
        for step in tf.range(tf.constant(self.config.num_steps)):
            if step % 2 == 0:
                setattr(self, 'xnet', self.xnet_even)
            else:
                setattr(self, 'xnet', self.xnet_odd)

            state_prop, logdet = lf_fn(step, state_prop, training)
            sumlogdet += logdet

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)

        return state_prop, accept_prob, sumlogdet

    def transition_kernel(
            self,
            state: State,
            forward: bool,
            training: bool = None,
    ):
        """Transition kernel of the augmented leapfrog integrator.

        Returns:
            state_prop (State): Proposed state output from integrator.
            accept_prob (tf.Tensor): (batch-wise) Acceptance probability.
            sumlogdet (tf.Tensor): Total log determinant of the Jacobian
        """
        if self.config.separate_networks and not self.config.hmc:
            return self.transition_kernel_sep_nets(state, forward, training)
        return super().transition_kernel(state, forward, training)

    def get_config(self):
        return {
            'config': self.config,
            'network_config': self.net_config,
            'lr_config': self.lr_config,
            'params': self.params
        }

    def _get_network(self, step):
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

    def _build_networks(self):
        """Build the nets that parameterize the aumgmented LF integrator."""
        if self.config.hmc:
            print('USING HMC NETWORKS')
            return self._build_hmc_networks()
        if self.use_conv_net:
            print('USING CONV NETS')
            return self._build_conv_networks()
        if self.config.separate_networks:
            print('USING SEPARATE NETWORKS')
            return self._build_separate_networks()

        print('USING GENERIC NETWORKS')
        return self._build_generic_networks()

    def _build_conv_networks(self):
        xnet = GaugeNetworkConv2D(
            conv_config=self.conv_config,
            config=self.net_config,
            xdim=self.xdim,
            factor=2.,
            name='XNet'
        )

        vnet = GaugeNetwork(
            #  conv_config=self.conv_config,
            config=self.net_config,
            xdim=self.xdim,
            factor=1.,
            name='VNet',
        )

        return xnet, vnet

    def _build_separate_networks(self):
        """Build separate networks for the even / odd update steps."""
        xnet_even = GaugeNetwork(self.net_config, self.xdim, name='XNet_even')
        xnet_odd = GaugeNetwork(self.net_config, self.xdim, name='XNet_odd')
        vnet = GaugeNetwork(self.net_config, self.xdim, name='VNet')

        return xnet_even, xnet_odd, vnet

    def _build_generic_networks(self):
        xnet = GaugeNetwork(self.net_config, xdim=self.xdim, name='XNet')
        vnet = GaugeNetwork(self.net_config, xdim=self.xdim, name='VNet')
        return xnet, vnet

    @staticmethod
    def mixed_loss(loss, weight):
        """Returns: tf.reduce_mean(weight / loss - loss / weight)."""
        return tf.reduce_mean((weight / loss) - (loss / weight))

    def calc_losses(self, states: MonteCarloStates, accept_prob: tf.Tensor):
        """Calculate the total loss."""
        dtype = states.init.x.dtype
        # ==== FIXME: Should we stack
        # ==== `states = [states.init.x, states.proposed.x]`
        # ==== and call `self.lattice.calc_plaq_sums(states)`?
        wl_init = self.lattice.calc_wilson_loops(states.init.x)
        wl_prop = self.lattice.calc_wilson_loops(states.proposed.x)

        # Calculate the plaquette loss
        ploss = tf.cast(0., dtype=dtype)
        if self.plaq_weight > 0:
            dwloops = 2 * (1. - tf.math.cos(wl_prop - wl_init))
            ploss = accept_prob * tf.reduce_sum(dwloops, axis=(1, 2))

            # ==== FIXME: Try using mixed loss??
            #  ploss = self.mixed_loss(ploss, self.plaq_weight)
            ploss = tf.reduce_mean(-ploss / self.plaq_weight, axis=0)

        # Calculate the charge loss
        qloss = tf.cast(0., dtype=dtype)
        if self.charge_weight > 0:
            q_init = self.lattice.calc_charges(wloops=wl_init, use_sin=True)
            q_prop = self.lattice.calc_charges(wloops=wl_prop, use_sin=True)
            qloss = accept_prob * (q_prop - q_init) ** 2

            # ==== FIXME: Try using mixed loss??
            #  qloss = self.mixed_loss(qloss, self.charge_weight)
            qloss = tf.reduce_mean(-qloss / self.charge_weight, axis=0)

        return ploss, qloss

    def train_step(self, data):
        """Perform a single training step."""
        x, beta = data
        start = time.time()
        with tf.GradientTape() as tape:
            states, accept_prob, sumlogdet = self(data, training=True)
            ploss, qloss = self.calc_losses(states, accept_prob)
            loss = ploss + qloss
            if self.aux_weight > 0:
                z = tf.random.normal(x.shape, dtype=x.dtype)
                states_, accept_prob_, _ = self((z, beta), training=True)
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
        #  if first_step and HAS_HOROVOD:
        if self.optimizer.iterations == 0 and self.using_hvd:
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

    def _scattered_xnet(self, x, v, t, masks, training=None):
        """Call `self.xnet` on non-zero entries of `x` via `tf.gather_nd`."""
        m, _ = masks
        shape = (self.batch_size, -1)
        m = tf.reshape(m, shape)
        idxs = tf.where(m)
        _x = tf.reshape(tf.gather_nd(x, idxs), shape)
        S, T, Q = self.xnet((v, _x, t), training)

        return S, T, Q

    def _update_v_forward(
                self,
                state: State,
                t: tf.Tensor,
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
        S, T, Q = self.vnet((x, x, t), training)

        transl = self._vtw * T
        scale = self._vsw * (0.5 * self.eps * S)
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
                t: tf.Tensor,
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
            return super()._update_x_forward(state, t, masks, training)
        if self.config.use_ncp:
            return self._update_xf_ncp(state, t, masks, training)

        m, mc = masks
        x = self.normalizer(state.x)
        S, T, Q = self.xnet((state.v, m * x, t), training=training)
        #  S, T, Q = self.xnet((state.v, m * x, t), training=training)
        #  S, T, Q = self._scattered_xnet(x, state.v, t, masks, training)

        transl = self._xtw * T
        transf = self._xqw * (self.eps * Q)
        scale = self._xsw * (self.eps * S)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)
        y = x * expS + self.eps * (state.v * expQ + transl)
        xf = self.normalizer(m * x + mc * y)
        state_out = State(x=xf, v=state.v, beta=state.beta)
        logdet = tf.reduce_sum(mc * scale, axis=1)

        return state_out, logdet

    def _update_v_backward(
                self,
                state: State,
                t: tf.Tensor,
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
        S, T, Q = self.vnet((x, grad, t), training)

        #  x = tf.reshape(x, self.lattice_shape)
        #  grad = tf.reshape(grad, self.lattice_shape)
        #
        #  S, T, Q = self.vnet((x, grad, t), training)

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
                t: tf.Tensor,
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
            return super()._update_x_backward(state, t, masks, training)
        if self.config.use_ncp:
            return self._update_xb_ncp(state, t, masks, training)

        # Call `XNet` using `self._scattered_xnet`
        m, mc = masks
        x = self.normalizer(state.x)

        S, T, Q = self.xnet((state.v, m * x, t), training=training)
        #  S, T, Q = self.xnet((state.v, m * x, t), training=training)
        #  S, T, Q = self._scattered_xnet(x, state.v, t, masks, training)

        scale = self._xsw * (-self.eps * S)
        transl = self._xtw * T
        transf = self._xqw * (self.eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)
        y = expS * (x - self.eps * (state.v * expQ + transl))
        xb = self.normalizer(m * x + mc * y)
        state_out = State(xb, v=state.v, beta=state.beta)
        logdet = tf.reduce_sum(mc * scale, axis=1)

        return state_out, logdet

    def _update_xf_ncp(
                self,
                state: State,
                t: tf.Tensor,
                masks: Tuple[tf.Tensor, tf.Tensor],   # [m, 1. - m]
                training: bool = None
    ):
        """Update the position `x` in the forward leapfrog step.

        NOTE: Non-Compact Projection
        -----------------------------
        1. Update,
                x -> x' = m * x + (1 - m) * y
          where
                y = x * exp(eps * Sx) + eps * (v * Qx + Tx)
        2. Let
                z = f(x): [-pi, pi] -> R, given by z = tan(x / 2)
        3. Then
                x' = m * x + (1 - m) * (2 * arctan(y))
          where
                y = tan(x / 2) * exp(eps * Sx) + eps * (v * Qx + Tx))
        4. With Jacobian:
                J = 1 / {[cos(x/2)]^2 + [exp(eps*Sx) * sin(x/2)]^2}
        """
        m, mc = masks
        x = self.normalizer(state.x)

        S, T, Q = self.xnet((state.v, m * x, t), training=training)
        #  S, T, Q = self.xnet((state.v, m * x, t), training=training)
        #  S, T, Q = self._scattered_xnet(x, state.v, t, masks, training)

        transl = self._xtw * T
        transf = self._xqw * (self.eps * Q)
        scale = self._xsw * (self.eps * S)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        _x = 2 * tf.math.atan(tf.math.tan(state.x/2) * expS)
        _y = _x + self.eps * (state.v * expQ + transl)
        xf = (m * state.x) + (mc * _y)
        state_out = State(x=xf, v=state.v, beta=state.beta)

        cterm = tf.math.cos(state.x / 2) ** 2
        sterm = (expS * tf.math.sin(state.x / 2)) ** 2
        log_jac = tf.math.log(expS / (cterm + sterm))
        logdet = tf.reduce_sum(mc * log_jac, axis=1)

        return state_out, logdet

    def _update_xb_ncp(
                self,
                state: State,
                t: tf.Tensor,
                masks: Tuple[tf.Tensor, tf.Tensor],   # [m, 1. - m]
                training: bool = None
    ):
        """Update the position `x` in the backward leapfrog step."""
        #  if not self.config.use_ncp:
        #      return super()._update_x_backward(state,
        #                                        t, masks, training)
        m, mc = masks
        x = self.normalizer(state.x)
        S, T, Q = self.xnet((state.v, m * x, t), training=training)
        #  S, T, Q = self.xnet((state.v, m * x, t), training=training)
        #  S, T, Q = self._scattered_xnet(x, state.v, t, masks, training)

        #  S, T, Q = self._scattered_xnet(x, state.v, t, masks, training)
        #  shape = (self.batch_size, -1)
        #  m_ = tf.reshape(m, shape)
        #  idxs = tf.where(m_)
        #  x_ = tf.reshape(tf.gather_nd(x, idxs), shape)
        #  S, T, Q = self.xnet((state.v, x_, t), training)

        transl = self._xtw * T
        transf = self._xqw * (self.eps * Q)
        scale = self._xsw * (-self.eps * S)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        # Apply Non-Compact Projection to the product $x \odot S_{x}$
        term1 = 2 * tf.math.atan(expS * tf.math.tan(state.x / 2))
        term2 = expS * self.eps * (state.v * expQ + transl)
        y = term1 - term2
        xb = (m * state.x) + (mc * y)
        state_out = State(x=xb, v=state.v, beta=state.beta)

        cterm = tf.math.cos(state.x / 2) ** 2
        sterm = (expS * tf.math.sin(state.x / 2)) ** 2
        log_jac = tf.math.log(expS / (cterm + sterm))
        logdet = tf.reduce_sum(mc * log_jac, axis=1)

        return state_out, logdet
