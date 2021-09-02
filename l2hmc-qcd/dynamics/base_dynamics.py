"""
dynamics_base.py

Implementes `BaseDynamics`, an abstract base class for the dynamics engine of
the L2HMC sampler.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.

Reference [Robust Parameter Estimation with a Neural Network Enhanced
Hamiltonian Markov Chain Monte Carlo Sampler]
https://infoscience.epfl.ch/record/264887/files/robust_parameter_estimation.pdf

Author: Sam Foreman (github: @saforem2)
Date: 6/30/2020
"""
from __future__ import absolute_import, annotations, division, print_function
from dataclasses import dataclass

import os
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

import utils.file_io as io
from utils.logger import Logger

logger = Logger()

try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
    NUM_RANKS = hvd.size()
except (ImportError, ModuleNotFoundError):
    HAS_HOROVOD = False
    NUM_RANKS = 1

from dynamics.config import (TF_FLOAT, DynamicsConfig, MonteCarloStates,
                             NetWeights, State)
from network.config import ConvolutionConfig, LearningRateConfig, NetworkConfig
from utils.attr_dict import AttrDict
from utils.learning_rate import WarmupExponentialDecay

__all__ = ['BaseDynamics', 'State', 'MonteCarloStates', 'NetWeights']

OPTIMIZERS_MAP = {
    'adam': tf.keras.optimizers.Adam,
    'Nadam': tf.keras.optimizers.Nadam,
    'sgd': tf.keras.optimizers.SGD,
}

# pylint:disable=invalid-name

def identity(x: tf.Tensor) -> tf.Tensor:
    """Returns x"""
    return x


class hmcNetwork:
    def __call__(self, inputs, *args):
        return [tf.zeros_like(inputs[0]) for _ in range(3)]

#  class hmcNetworks:
#      def __init__(self):
#          self._xnet = lambda inputs, *args: [
#              tf.zeros_like(inputs[0]) for _ in range(3)
#          ]
#          self._vnet = lambda inputs, *args: [
#              tf.zeros_like(inputs[0]) for _ in range(3)
#          ]
#
#      def xnet(self, inputs, *args):
#          return [tf.zeros_like(inputs[0]) for _ in range(3)]
#



Inputs = Tuple[tf.Tensor, tf.Tensor]

# pylint:disable=attribute-defined-outside-init
# pylint:disable=invalid-name, too-many-instance-attributes
# pylint:disable=too-many-arguments, too-many-locals, too-many-ancestors
class BaseDynamics(tf.keras.Model):
    """Dynamics object for training the L2HMC sampler."""

    def __init__(
            self,
            params: AttrDict,
            config: DynamicsConfig,
            network_config: NetworkConfig,
            potential_fn: Callable[[tf.Tensor], tf.Tensor],
            lr_config: LearningRateConfig = None,
            normalizer: Callable[[tf.Tensor], tf.Tensor] = None,
            should_build: Optional[bool] = True,
            name: Optional[str] = 'Dynamics',
    ):
        """Initialization method.

        #  name: str = 'Dynamics',

        * NOTE: `normalizer` is meant to be a function that forces the
            configurations to exist in some restricted space. For example,
            in the case of a 2D U(1) lattice gauge model, we restrict the
            configurations `x` to reside in [-pi, pi] (the U(1) gauge group).
            If not specified, `normalizer` will default to `identity` defined
            above, which just returns the input.
        """
        super(BaseDynamics, self).__init__(name=name)
        #  super(BaseDynamics, self).__init__(name=name)
        self._has_trainable_params = False
        #  self._model_type = config.get('model_type', 'BaseDynamics')
        self.params = params
        self.config = config
        self.lr_config = lr_config
        self.net_config = network_config
        self.potential_fn = potential_fn
        self._verbose = config.verbose
        #  self._verbose = config.get('verbose', False)

        loss_scale = config.loss_scale
        #  loss_scale = self.config.get('loss_scale', 1.)
        self._loss_scale = tf.constant(loss_scale, name='loss_scale')
        self.xdim = params.get('xdim', None)
        self.clip_val = params.get('clip_val', 0.)
        self.aux_weight = params.get('aux_weight', 0.)
        self.batch_size = params.get('batch_size', None)

        self.x_shape = (self.batch_size, self.xdim)
        #  self.eps = self._build_eps(use_log=False)
        self.veps = [
            self._build_eps(use_log=False) for _ in
            range(self.config.num_steps)
        ]
        self.xeps = [
            self._build_eps(use_log=False) for _ in
            range(self.config.num_steps)
        ]
        self._lf_step = 0
        self.masks = self._build_masks()
        self.normalizer = normalizer if normalizer is not None else identity

        if not self.config.hmc:
            self.xnet, self.vnet = self._build_networks()

        #  if self._has_trainable_params:
        if not self.config.hmc and not self.config.eps_fixed:
            self.lr = self._create_lr(lr_config)
            self.optimizer = self._create_optimizer(self.config.optimizer)

        if self.config.hmc:
            self.net_weights = NetWeights(0., 0., 0., 0., 0., 0.)
            self.xnet, self.vnet = self._build_hmc_networks()
            if self.config.eps_fixed:
                self._has_trainable_params = False

        #  if should_build:
        #      if not self.config.hmc:
        #          self.xnet, self.vnet = self._build_networks()
        #      if self._has_trainable_params:
        #          self.lr = self._create_lr(lr_config)
        #          self.optimizer = self._create_optimizer(
        #              self.config.optimizer
        #          )

    def call(
            self,
            inputs: Inputs,
            training: bool = None,
    ):
        """Calls the model on new inputs.

        In this case `call` just reapplies all ops in the graph to the new
        inputs (e.g. build a new computational graph from the provided inputs).

        NOTE: Custom implementation calls `self.apply_transition` and returns a
        list of the states (init, proposed, out), accept probability, and the
        sumlogdet states (init, proposed, out):
            [mc_states: MonteCarloStates,
             accept_prob: tf.Tensor,
             sld_states: MonteCarloStates]

        Arguments:
            inputs: A tuple of `tf.Tensor`'s (x, beta).
            training: Boolean or boolean scalar tensor, indicating whether to
                run the `Network` in training mode or inference mode.

        Returns:
            A tensor if there is a single output, or a list of tensors if there
            are more than one outputs.
        """
        return self.apply_transition(inputs, training=training)

    def calc_losses(
            self, states: MonteCarloStates, accept_prob: tf.Tensor
    ):
        """Calculate the total loss."""
        raise NotImplementedError

    @staticmethod
    def calc_esjd(
            x: tf.Tensor, y: tf.Tensor, accept_prob: tf.Tensor
    ) -> tf.Tensor:
        """Calculate the expected squared jump distance, ESJD."""
        return accept_prob * tf.reduce_sum((x - y) ** 2, axis=1) + 1e-4

    def _mixed_loss(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            accept_prob: tf.Tensor,
            scale: float
    ):
        """Compute the mixed loss as: scale / esjd - esjd / scale."""
        if scale is None:
            scale = self._loss_scale

        esjd = self.calc_esjd(x, y, accept_prob)
        loss = tf.reduce_mean(scale / esjd) - tf.reduce_mean(esjd / scale)

        return loss

    def _get_lr(self, step=None) -> (tf.Tensor):
        if step is None:
            step = self.optimizer.iterations

        #  if isinstance(self.lr, callable):
        if callable(self.lr):
            return self.lr(step)

        return K.get_value(self.optimizer.lr)

    def train_step(  # pylint:disable=arguments-differ
            self,
            inputs: Inputs,
    ):
        """Perform a single training step.

        Returns:
            x_out (tf.Tensor): Next `x` configuration in chain.
            metrics (AttrDict): Dictionary of various metrics for logging.
        """
        raise NotImplementedError

    def test_step(  # pylint:disable=arguments-differ
            self,
            inputs: Inputs,
    ):
        """Perform a single inference step.

        Returns:
            x_out (tf.Tensor): Next `x` configuration in chain.
            metrics (AttrDict): Dictionary of various metrics for logging.
        """

        raise NotImplementedError

    def _forward(
            self,
            inputs: Inputs,
            training: bool = None,
    ):
        """Propose a new state by running the leapfrog integrator forward."""
        state_init, state_prop, data = self._transition(inputs, forward=True,
                                                        training=training)
        sumlogdet = data.get('sumlogdet', None)
        accept_prob = data.get('accept_prob', None)
        ma_, mr_ = self._get_accept_masks(accept_prob)
        ma = ma_[:, None]
        mr = mr_[:, None]
        x_out = state_prop.x * ma + state_init.x * mr
        v_out = state_prop.v * ma + state_init.v * mr
        data.sumlogdet_out = sumlogdet * ma
        state_out = State(x_out, v_out, state_init.beta)
        mc_states = MonteCarloStates(state_init, state_prop, state_out)

        return mc_states, data

    def _backward(
            self,
            inputs: Inputs,
            training: bool = None,
    ):
        """Propose a new state by running the leapfrog integrator forward."""
        state_init, state_prop, data = self._transition(inputs, forward=False,
                                                        training=training)
        sumlogdet = data.get('sumlogdet', None)
        accept_prob = data.get('accept_prob', None)
        ma_, mr_ = self._get_accept_masks(accept_prob)
        ma = ma_[:, None]
        mr = mr_[:, None]
        x_out = state_prop.x * ma + state_init.x * mr
        v_out = state_prop.v * ma + state_init.v * mr
        data.sumlogdet_out = sumlogdet * ma
        state_out = State(x_out, v_out, state_init.beta)
        mc_states = MonteCarloStates(state_init, state_prop, state_out)

        return mc_states, data

    def _random_direction(
            self,
            inputs: Inputs,
            training: bool = None
    ) -> tuple[MonteCarloStates, AttrDict]:
        """Propose a new state and perform the accept/reject step."""
        forward = tf.cast((tf.random.uniform(shape=[]) < 0.5), dtype=tf.bool)
        state_init, state_prop, data = self._transition(inputs,
                                                        forward=forward,
                                                        training=training)
        sumlogdet = data.get('sumlogdet', None)
        accept_prob = data.get('accept_prob', None)
        ma, mr = self._get_accept_masks(accept_prob)
        ma_tensor = ma[:, None]
        mr_tensor = mr[:, None]
        x_out = state_prop.x * ma_tensor + state_init.x * mr_tensor
        v_out = state_prop.v * ma_tensor + state_init.v * mr_tensor
        data.sumlogdet_out = sumlogdet * ma

        state_out = State(x_out, v_out, state_init.beta)
        #  sld_states = MonteCarloStates(0., data.sumlogdet, sumlogdet)
        mc_states = MonteCarloStates(state_init, state_prop, state_out)

        return mc_states, data

    def _hmc_transition(
            self, inputs: Inputs, training: bool = None,
    ) -> tuple[MonteCarloStates, AttrDict]:
        """Propose a new state and perform the accept/reject step."""
        x, v, beta = None, None, None
        if len(inputs) == 2:
            x, beta = inputs
            v = tf.random.normal(x.shape, dtype=x.dtype)
        elif len(inputs) == 3:
            x, v, beta = inputs

        state = State(x, v, beta)
        #  x, beta = inputs
        state, state_prop, data = self._transition(state, forward=True,
                                                   training=training)

        sumlogdet = data.get('sumlogdet', None)
        accept_prob = data.get('accept_prob', None)
        ma_, mr_ = self._get_accept_masks(accept_prob)
        ma = ma_[:, None]
        mr = mr_[:, None]

        # Construct the output configuration
        v_out = ma * state_prop.v + mr * state.v
        x_out = self.normalizer(ma * state_prop.x + mr * state.x)
        sumlogdet = ma_ * sumlogdet  # NOTE: initial sumlogdet = 0

        #  state_init = State(x=x, v=state_init.v, beta=beta)
        state_prop = State(x=state_prop.x, v=state_prop.v, beta=state.beta)
        state_out = State(x=x_out, v=v_out, beta=state.beta)

        mc_states = MonteCarloStates(state, state_prop, state_out)
        #  sld_states = MonteCarloStates(0., sld_prop, sumlogdet)
        data.update({
            'accept_mask': ma_,
            'sumlogdet': sumlogdet,
        })

        return mc_states, data

    def _transition(
            self,
            inputs: tuple[tf.Tensor, ...],
            forward: bool,
            training: bool = None
    ) -> tuple[State, State, AttrDict]:
        """Run the augmented leapfrog integrator."""
        x, v, beta = None, None, None
        if not isinstance(inputs, (tuple, List)):
            raise ValueError(
                f'Expected inputs to be one of: [list, tuple]. \n'
                f'Received: {type(inputs)}'
            )

        if len(inputs) == 2:
            x, beta = inputs
            v = tf.random.normal(x.shape, dtype=x.dtype)

        elif len(inputs) == 3:
            x, v, beta = inputs

        state = State(x=x, v=v, beta=beta)
        state_, data = self.transition_kernel(state, forward, training)

        return state, state_, data

    def apply_transition(
            self, inputs: Inputs, training: Optional[bool] = None,
    ) -> tuple[MonteCarloStates, AttrDict]:
        """Propose a new state and perform the accept/reject step.

        NOTE: We simulate the dynamics both forward and backward, and use
        sampled Bernoulli masks to compute the actual solutions
        """
        if self.config.hmc:
            return self._hmc_transition(inputs, training)

        x, beta = inputs

        sf_init, sf_prop, dataf = self._transition(inputs, forward=True,
                                                   training=training)
        sb_init, sb_prop, datab = self._transition(inputs, forward=False,
                                                   training=training)
        sldf = dataf.get('sumlogdet', None)
        sldb = datab.get('sumlogdet', None)
        pxf = dataf.get('accept_prob', None)
        pxb = datab.get('accept_prob', None)

        # ====
        # Combine the forward / backward outputs;
        # get forward / backward masks:
        mf_, mb_ = self._get_direction_masks()
        mf = mf_[:, None]
        mb = mb_[:, None]
        # reconstruct v_init
        v_init = mf * sf_init.v + mb * sb_init.v
        # construct proposed configuration: `x`, `v`
        x_prop = mf * sf_prop.x + mb * sb_prop.x
        v_prop = mf * sf_prop.v + mb * sb_prop.v
        sld_prop = mf_ * sldf + mb_ * sldb

        # Compute the acceptance probability
        accept_prob = mf_ * pxf + mb_ * pxb

        # get accept `ma_`, reject `mr_` masks
        ma_, mr_ = self._get_accept_masks(accept_prob)
        ma = ma_[:, None]
        mr = mr_[:, None]

        # Construct the output configuration
        v_out = ma * v_prop + mr * v_init
        x_out = self.normalizer(ma * x_prop + mr * x)
        sumlogdet = ma_ * sld_prop  # NOTE: initial sumlogdet = 0

        state_init = State(x=x, v=v_init, beta=beta)
        state_prop = State(x=x_prop, v=v_prop, beta=beta)
        state_out = State(x=x_out, v=v_out, beta=beta)

        mc_states = MonteCarloStates(state_init, state_prop, state_out)
        #  sld_states = MonteCarloStates(0., sld_prop, sumlogdet)

        data = AttrDict({
            'forward': dataf,
            'backward': datab,
            'accept_prob': accept_prob,
            'accept_mask': ma_,
            'sumlogdet': sumlogdet,
            'sumlogdet_prop': sld_prop,
        })

        #  return mc_states, accept_prob, sld_states
        return mc_states, data

    def md_update(
            self,
            inputs: tuple[tf.Tensor, tf.Tensor],
            training: bool = None
    ) -> Tuple[MonteCarloStates[State], MonteCarloStates[float]]:
        """Perform the molecular dynamics (MD) update w/o accept/reject.

        NOTE: We simulate the dynamics both forward and backward, and use
        sampled Bernoulli masks to compute the actual solutions
        """
        x, beta = inputs
        sf_init, sf_prop, dataf = self._transition(inputs, forward=True,
                                                   training=training)
        sb_init, sb_prop, datab = self._transition(inputs, forward=False,
                                                   training=training)

        sldf = dataf.get('sumlogdet', None)
        sldb = datab.get('sumlogdet', None)

        # Decide direction uniformly
        mf_, mb_ = self._get_direction_masks()
        mf = mf_[:, None]
        mb = mb_[:, None]

        v_init = mf * sf_init.v + mb * sb_init.v
        x_prop = mf * sf_prop.x + mb * sb_prop.x
        v_prop = mf * sf_prop.v + mb * sb_prop.v
        sld_prop = mf_ * sldf + mb_ * sldb
        x_prop = self.normalizer(x_prop)

        state_init = State(x=x, v=v_init, beta=beta)
        state_prop = State(x=x_prop, v=v_prop, beta=beta)
        mc_states = MonteCarloStates(init=state_init,
                                     proposed=state_prop,
                                     out=state_prop)
        sld_states = MonteCarloStates(init=0.,
                                      proposed=sld_prop,
                                      out=sld_prop)

        return mc_states, sld_states

    def test_reversibility(
            self,
            x: tf.Tensor = None,
            v: tf.Tensor = None,
            beta: tf.Tensor = None,
            training: bool = None
    ):
        """Test reversibility.

        NOTE:
         1. Run forward then backward
                 (x, v) -> (xf, vf)
                 (xf, vf) -> (xb, vb)
            check that x == xb, v == vb

         2. Run backward then forward
                 (x, v) -> (xb, vb)
                 (xb, vb) -> (xf, vf)
            check that x == xf, v == vf
        """
        if x is None:
            x = tf.random.normal(self.x_shape)
        if v is None:
            v = tf.random.normal(self.x_shape)
        if beta is None:
            beta = tf.constant(1.)

        state = State(x, v, beta)
        dxf, dvf = self._test_reversibility(state,
                                            forward_first=True,
                                            training=training)
        dxb, dvb = self._test_reversibility(state,
                                            forward_first=False,
                                            training=training)
        output = AttrDict({
            'forward': {
                'dx': dxf,
                'dv': dvf,
            },
            'backward': {
                'dx': dxb,
                'dv': dvb,
            }
        })

        return output

    def _test_reversibility(
            self,
            state: State,
            forward_first: bool = False,
            training: bool = None
    ):
        """Helper method for `self.test_reversibility`.

        NOTE:
         - If `forward_first`, test (1.) (forward then backward).
         - Else, test (2.) (backward then forward).
        """
        #  x, beta = data
        #  v = tf.random.normal(tf.shape(x))

        #  Run from (x, v) -> (x1, v1)
        state1, _ = self.transition_kernel(state, forward_first, training)
        # Run from (x1, v1) --> (x2, v2)
        state2, _ = self.transition_kernel(state1, not forward_first, training)
        # If reversible, then x2 == x, v2 == v
        dx = state.x - state2.x
        dv = state.v - state2.v

        return dx, dv

    def _init_metrics(self, state: State) -> (AttrDict):
        """Create logdet/energy metrics for verbose logging."""
        metrics = AttrDict({
            'sumlogdet': 0.,
            'accept_prob': None,
        })
        if not self._verbose:
            return metrics

        kwargs = {
            'dynamic_size': False,
            'element_shape': (state.x.shape[0],),
            'size': self.config.num_steps + 1,
            'clear_after_read': False
        }
        if state.x.shape[0] != self.batch_size:
            logger.warning(f'state.x.shape[0] != self.batch_size')
            logger.warning(f'state.x.shape[0]: {state.x.shape[0]}')
            logger.warning(f'self.batch_size: {self.batch_size}')
            logger.warning(f'Setting self.batch_size to {state.x.shape[0]}')
            self.batch_size = state.x.shape[0]

        logdets = tf.TensorArray(TF_FLOAT, **kwargs)
        energies = tf.TensorArray(TF_FLOAT, **kwargs)
        energies_scaled = tf.TensorArray(TF_FLOAT, **kwargs)

        energy = self.hamiltonian(state)
        logdet = tf.zeros((state.x.shape[0]))
        energy_scaled = energy - logdet
        metrics.update({
            'H': energies.write(0, energy),
            'logdets': logdets.write(0, logdet),
            'Hw': energies_scaled.write(0, energy_scaled),
            # pad with 1st elt (copy) to have same length as TensorArray's
            # defined above (useful for creating ridgeplots)
            'xeps': [self.xeps[0], *self.xeps],
            'veps': [self.veps[0], *self.veps],
        })

        return metrics

    def _update_metrics(
            self,
            metrics: AttrDict,
            step: int,
            state: State,
            sumlogdet: float,
            **kwargs
    ) -> (AttrDict):
        """Write to metrics."""
        midpt = self.config.num_steps // 2
        def _traj_summ(x, key=None):
            if key is not None:
                return {f'{key}': tf.squeeze(x)}
            return (x[0], x[midpt], x[1])

        metrics['sumlogdet'] = sumlogdet
        for key, val in kwargs.items():
            #  if not isinstance(val, tf.TensorArray):
            if isinstance(val, tf.Tensor):
                metrics[key] = val

        if not self._verbose:
            return metrics

        #  tarr_len = self.config.num_steps + 1
        #  logdets = metrics['logdets']
        kwargs = {
            'size': self.config.num_steps+1,
            'element_shape': (state.x.shape[0],),
        }
        logdets = tf.TensorArray(TF_FLOAT, **kwargs).unstack(
            metrics['logdets']
        )
        energies = tf.TensorArray(TF_FLOAT, **kwargs).unstack(
            metrics['H']
        )
        energies_ = tf.TensorArray(TF_FLOAT, **kwargs).unstack(
            metrics['Hw']
        )

        if step == self.config.num_steps + 1:
            for key, val in metrics.items():
                if isinstance(val, tf.TensorArray):
                    tensor = val.stack()
                    metrics[key] = tensor
        else:
            energy = self.hamiltonian(state)
            energies = energies.write(step, energy)
            logdets = logdets.write(step, sumlogdet)
            energies_ = energies_.write(step, energy - sumlogdet)
            metrics['H'] = energies.stack()
            metrics['logdets'] = logdets.stack()
            metrics['Hw'] = energies_.stack()

        return metrics

    def _transition_kernel_forward(
            self,
            state: State,
            training: bool = None
    ):
        """Run the augmented leapfrog sampler in the forward direction."""
        sumlogdet = tf.zeros((state.x.shape[0],))
        state_prop = State(self.normalizer(state.x), state.v, state.beta)
        metrics = self._init_metrics(state_prop)
        state_prop, logdet = self._half_v_update_forward(state_prop, step=0,
                                                         training=training)
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
                    energy = self.hamiltonian(state_prop)
                    metrics['H'] = metrics['H'].write(step+1, energy)
                    metrics['logdets'] = metrics['logdets'].write(step+1,
                                                                  sumlogdet)
                    metrics['Hw'] = metrics['Hw'].write(step+1,
                                                        energy - sumlogdet)

        state_prop, logdet = self._half_v_update_forward(state_prop,
                                                         step, training)
        sumlogdet += logdet
        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)
        metrics['sumlogdet'] = sumlogdet
        metrics['accept_prob'] = accept_prob
        if self._verbose:
            energy = self.hamiltonian(state_prop)
            metrics['H'] = metrics['H'].write(step+1, energy)
            metrics['logdets'] = metrics['logdets'].write(step+1,
                                                          sumlogdet)
            metrics['Hw'] = metrics['Hw'].write(step+1,
                                                energy-sumlogdet)
            metrics['H'] = metrics['H'].stack()
            metrics['logdets'] = metrics['logdets'].stack()
            metrics['Hw'] = metrics['Hw'].stack()

        return state_prop, metrics

    def _transition_kernel_backward(
            self,
            state: State,
            training: bool = None
    ):
        """Run the augmented leapfrog sampler in the forward direction."""
        sumlogdet = tf.zeros((state.x.shape[0],))
        state_prop = State(state.x, state.v, state.beta)
        metrics = self._init_metrics(state_prop)

        state_prop, logdet = self._half_v_update_backward(state_prop,
                                                          0, training)
        sumlogdet += logdet
        for step in range(self.config.num_steps):
            state_prop, logdet = self._full_x_update_backward(state_prop,
                                                              step, training)

            if step < self.config.num_steps - 1:
                state_prop, logdet = self._full_v_update_backward(
                    state_prop, step, training
                )
                sumlogdet += logdet
                if self._verbose:
                    energy = self.hamiltonian(state_prop)
                    metrics['H'] = metrics['H'].write(step+1, energy)
                    metrics['logdets'] = metrics['logdets'].write(step+1,
                                                                  sumlogdet)
                    metrics['Hw'] = metrics['Hw'].write(step+1,
                                                        energy - sumlogdet)
        step = self.config.num_steps - 1
        state_prop, logdet = self._half_v_update_backward(state_prop,
                                                          step, training)
        sumlogdet += logdet

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)
        metrics['sumlogdet'] = sumlogdet
        metrics['accept_prob'] = accept_prob
        if self._verbose:
            energy = self.hamiltonian(state_prop)
            metrics['H'] = metrics['H'].write(step+1, energy)
            metrics['logdets'] = metrics['logdets'].write(step+1,
                                                          sumlogdet)
            metrics['Hw'] = metrics['Hw'].write(step+1,
                                                energy-sumlogdet)
            metrics['H'] = metrics['H'].stack()
            metrics['logdets'] = metrics['logdets'].stack()
            metrics['Hw'] = metrics['Hw'].stack()

        return state_prop, metrics

    def transition_kernel(
            self,
            state: State,
            forward: bool,
            training: Optional[bool] = None,
            #  verbose: Optional[bool] = False,
    ):
        """Transition kernel of the augmented leapfrog integrator."""
        lf_fn = self._forward_lf if forward else self._backward_lf
        state_prop = State(x=state.x, v=state.v, beta=state.beta)
        sumlogdet = tf.zeros((state.x.shape[0],), dtype=TF_FLOAT)
        metrics = self._init_metrics(state_prop)
        for step in range(self.config.num_steps):
            state_prop, logdet = lf_fn(step, state_prop, training)
            sumlogdet += logdet
            if self._verbose:
                energy = self.hamiltonian(state_prop)
                metrics['H'] = metrics['H'].write(step+1, energy)
                metrics['Hw'] = metrics['Hw'].write(step+1, energy - sumlogdet)

                metrics['logdets'] = metrics['logdets'].write(step+1,
                                                              sumlogdet)
            #  metrics = self._update_metrics(metrics, step+1,
            #                                 state_prop, sumlogdet)

        accept_prob = self.compute_accept_prob(state, state_prop, sumlogdet)
        metrics['sumlogdet'] = sumlogdet
        metrics['accept_prob'] = accept_prob
        step = self.config.num_steps - 1
        if self._verbose:
            energy = self.hamiltonian(state_prop)
            metrics['H'] = metrics['H'].write(step+1, energy)
            metrics['logdets'] = metrics['logdets'].write(step+1,
                                                          sumlogdet)
            metrics['Hw'] = metrics['Hw'].write(step+1,
                                                energy-sumlogdet)
            metrics['H'] = metrics['H'].stack()
            metrics['logdets'] = metrics['logdets'].stack()
            metrics['Hw'] = metrics['Hw'].stack()

        return state_prop, metrics

    def compute_accept_prob(
            self, state_init: State, state_prop: State, sumlogdet: tf.Tensor
    ):
        """Compute the acceptance prob. of `state_prop` given `state_init`.

        Returns: tf.Tensor
        """
        h_init = self.hamiltonian(state_init)
        h_prop = self.hamiltonian(state_prop)
        dh = h_init - h_prop + sumlogdet
        prob = tf.exp(tf.minimum(dh, 0.))

        return tf.where(tf.math.is_finite(prob), prob, tf.zeros_like(prob))

    def _forward_lf(self, step: int, state: State, training: bool = None):
        """Run the augmented leapfrog integrator in the forward direction."""
        # === NOTE: m = random mask (half 1s, half 0s); mc = 1. - m
        m, mc = self._get_mask(step)  # pylint: disable=invalid-name
        sumlogdet = tf.zeros((state.x.shape[0],))

        state, logdet = self._update_v_forward(state, step, training)
        sumlogdet += logdet
        state, logdet = self._update_x_forward(state, step,
                                               (m, mc), training, first=True)
        sumlogdet += logdet
        state, logdet = self._update_x_forward(state, step,
                                               (mc, m), training, first=False)
        sumlogdet += logdet
        state, logdet = self._update_v_forward(state, step, training)
        sumlogdet += logdet

        return state, sumlogdet

    def _backward_lf(self, step: int, state: State, training: bool = None):
        """Run the augmented leapfrog integrator in the backward direction."""
        step_r = self.config.num_steps - step - 1
        m, mc = self._get_mask(step_r)
        sumlogdet = tf.zeros((state.x.shape[0]))

        state, logdet = self._update_v_backward(state, step_r, training)
        sumlogdet += logdet

        state, logdet = self._update_x_backward(state, step_r,
                                                (mc, m), training)
        sumlogdet += logdet

        state, logdet = self._update_x_backward(state, step_r,
                                                (m, mc), training)
        sumlogdet += logdet

        state, logdet = self._update_v_backward(state, step_r, training)
        sumlogdet += logdet

        return state, sumlogdet

    def _call_vnet(
            self,
            inputs: Inputs,
            step: int,
            training: bool = None
    ):
        """Call `self.vnet` to get Sv, Tv, Qv for updating `v`."""
        raise NotImplementedError

    def _call_xnet(
            self,
            inputs: Inputs,
            mask: tf.Tensor,
            step: int,
            training: bool = None,
            first: bool = False,
    ):
        """Call `self.xnet` to get Sx, Tx, Qx for updating `x`."""
        raise NotImplementedError

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

        scale = self._vsw * (0.5 * eps * S)
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
        x = self.normalizer(state.x)
        grad = self.grad_potential(x, state.beta)
        eps = self.veps[step]

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
        x = self.normalizer(state.x)
        eps = self.veps[step]
        grad = self.grad_potential(x, state.beta)
        S, T, Q = self._call_vnet((x, grad), step, training)

        transl = self._vtw * T
        scale = self._vsw * (0.5 * eps * S)
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

    def _full_x_update(
            self,
            state: State,
            step: int,
            forward: bool,
            training: bool = None
    ):
        if forward:
            return self._full_x_update_forward(state, step, training)

        return self._full_x_update_backward(state, step, training)

    def _update_x_forward(
                self,
                state: State,
                step: int,
                masks: Tuple[tf.Tensor, tf.Tensor],   # (m, 1. - m)
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
        m, mc = masks
        x = self.normalizer(state.x)
        eps = self.xeps[step]

        S, T, Q = self.xnet((m * x, state.v), training)

        transl = self._xtw * T
        scale = self._xsw * (eps * S)
        transf = self._xqw * (eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        y = x * expS + eps * (state.v * expQ + transl)
        xf = m * x + mc * y

        xf = self.normalizer(xf)

        state_out = State(x=xf, v=state.v, beta=state.beta)
        logdet = tf.reduce_sum(mc * scale, axis=1)

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
        eps = self.veps[step_r]
        S, T, Q = self._call_vnet((x, grad), step_r, training)

        scale = self._vsw * (-0.5 * eps * S)
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
        x = self.normalizer(state.x)
        grad = self.grad_potential(x, state.beta)
        eps = self.veps[step]
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
        x = self.normalizer(state.x)
        eps = self.veps[step]

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
                masks: Tuple[tf.Tensor, tf.Tensor],   # (m, 1. - m)
                training: bool = None,
                first_update: bool = True,
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
        m, mc = masks
        x = self.normalizer(state.x)
        eps = self.xeps[step]
        S, T, Q = self.xnet((m * x, state.v), training)

        scale = self._xsw * (-eps * S)
        transl = self._xtw * T
        transf = self._xqw * (eps * Q)

        expS = tf.exp(scale)
        expQ = tf.exp(transf)

        y = expS * (x - eps * (state.v * expQ + transl))
        xb = self.normalizer(m * x + mc * y)

        state_out = State(x=xb, v=state.v, beta=state.beta)
        logdet = tf.reduce_sum(mc * scale, axis=1)

        return state_out, logdet

    def grad_potential(self, x: tf.Tensor, beta: tf.Tensor):
        """Compute the gradient of the potential function."""
        x = self.normalizer(x)
        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                tape.watch(x)
                pe = self.potential_energy(x, beta)
            grad = tape.gradient(pe, x)
            #  grad = tf.reshape(tape.gradient(pe, x),
            #                    (self.batch_size, -1))
        else:
            grad = tf.gradients(self.potential_energy(x, beta), [x])[0]

        return grad

    def potential_energy(self, x: tf.Tensor, beta: tf.Tensor):
        """Compute the potential energy as beta times the potential fn."""
        x = self.normalizer(x)
        pe = beta * self.potential_fn(x)

        return pe

    @staticmethod
    def kinetic_energy(v: tf.Tensor):
        """Compute the kinetic energy of the momentum as 0.5 * (v ** 2)."""
        return 0.5 * tf.reduce_sum(tf.keras.layers.Flatten()(v) ** 2, axis=-1)

    def hamiltonian(self, state: State):
        """Compute the overall Hamiltonian."""
        kinetic = self.kinetic_energy(state.v)
        potential = self.potential_energy(state.x, state.beta)

        return potential + kinetic

    def _get_time(self, i: int, tile=1):
        """Format the MCMC step as [cos(...), sin(...)]."""
        i = tf.cast(i, dtype=TF_FLOAT)
        trig_t = tf.squeeze([
            tf.cos(2 * np.pi * i / self.config.num_steps),
            tf.sin(2 * np.pi * i / self.config.num_steps),
        ])
        t = tf.tile(tf.expand_dims(trig_t, 0), (tile, 1))

        return t

    def _get_direction_masks(self):
        """Decide direction uniformly."""
        forward_mask = tf.cast(
            tf.random.uniform((self.batch_size,)) > 0.5,
            dtype=TF_FLOAT,
        )
        backward_mask = 1. - forward_mask

        return forward_mask, backward_mask

    @staticmethod
    def _get_accept_masks(accept_prob: tf.Tensor):
        """Create binary array to pick out which idxs are accepted."""
        accept_mask = tf.cast(
            accept_prob > tf.random.uniform(tf.shape(accept_prob)),
            dtype=TF_FLOAT,
        )
        reject_mask = 1. - accept_mask

        return accept_mask, reject_mask

    def _build_masks(self):
        """Construct different binary masks for different time steps."""
        masks = []
        for _ in range(self.config.num_steps):
            # Need to use numpy.random here because tensorflow would
            # generate different random values across different calls.
            _idx = np.arange(self.xdim)
            idx = np.random.permutation(_idx)[:self.xdim // 2]
            mask = np.zeros((self.xdim,))
            mask[idx] = 1.

            mask = tf.constant(mask, dtype=TF_FLOAT)
            masks.append(mask[None, :])

        return masks

    def _get_mask(self, i: int):
        """Retrieve the binary mask for the i-th leapfrog step."""
        m = self.masks[i]
        return m, 1. - m

    def _build_networks(
            self,
            net_config: NetworkConfig = None,
            conv_config: ConvolutionConfig = None,
            log_dir: str = None,
    ):
        """Logic for building the position and momentum networks.

        Returns:
            xnet: tf.keras.models.Model
            vnet: tf.keras.models.Model
        """
        raise NotImplementedError

    def _load_networks(
            self,
            log_dir: str
    ):
        raise NotImplementedError

    @staticmethod
    def _build_hmc_networks():
        # pylint:disable=unused-argument
        xnet = lambda inputs, is_training: [  # noqa: E731
            tf.zeros_like(inputs[0]) for _ in range(3)
        ]
        vnet = lambda inputs, is_training: [  # noqa: E731
            tf.zeros_like(inputs[0]) for _ in range(3)
        ]

        return xnet, vnet

    # pylint:disable=unused-argument
    def _get_network(self, step: int):
        return self.xnet, self.vnet

    def _build_eps(self, use_log: bool = False):
        """Create `self.eps` (i.e. the step size) as a `tf.Variable`.

        Args:
            use_log (bool): If True, initialize `log_eps` as the actual
            `tf.Variable` and set `self.eps = tf.exp(log_eps)`; otherwise, set
            `self.eps` as a `tf.Variable` directly.

        Returns:
            eps: The (trainable) step size to be used in the L2HMC algorithm.
        """
        if use_log:
            init = tf.math.log(tf.constant(self.config.eps, dtype=TF_FLOAT))
        else:
            init = tf.constant(self.config.eps, dtype=TF_FLOAT)

        return tf.Variable(initial_value=init,
                           name='eps', dtype=TF_FLOAT,
                           constraint=tf.keras.constraints.non_neg(),
                           trainable=not self.config.eps_fixed)

    def _create_lr(
            self,
            lr_config: LearningRateConfig = None,
            scale: bool = False,
            auto: bool = True,
    ):
        """Create the learning rate schedule to be used during training."""
        if lr_config is None:
            lr_config = self.lr_config

        if scale:
            tf.print('Scaling learning rate...\n')
            tf.print(f'original lr: {lr_config.lr_init}')
            lr_config.lr_init *= tf.math.sqrt(tf.cast(hvd.size(), TF_FLOAT))
            tf.print(f'new (scaled) lr: {lr_config.lr_init}')

        if auto:
            return lr_config.lr_init

        warmup_steps = lr_config.get('warmup_steps', None)
        if warmup_steps > 0:
            return WarmupExponentialDecay(lr_config, staircase=True,
                                          name='WarmupExponentialDecay')

        decay_rate = lr_config.get('decay_rate', None)
        decay_steps = lr_config.get('decay_steps', None)
        cond1 = (decay_rate is not None and decay_rate > 0)
        cond2 = (decay_steps is not None and decay_steps > 0)
        if cond1 and cond2:
            return tf.keras.optimizers.schedules.ExponentialDecay(
                lr_config.lr_init,
                decay_steps=lr_config.decay_steps,
                decay_rate=lr_config.decay_rate,
                staircase=True,
            )

        return lr_config.lr_init

    def _create_optimizer(self, opt: str = 'adam'):
        """Create the optimizer to be used for backpropagating gradients."""
        opt_fn = OPTIMIZERS_MAP[opt]
        if self.clip_val > 0:
            optimizer = opt_fn(self.lr, clipnorm=self.clip_val)
        else:
            optimizer = opt_fn(self.lr)

        return optimizer

    def save_config(self, config_dir: str):
        """Helper method for saving configuration objects."""
        io.save_dict(self.config, config_dir, name='dynamics_config')
        if self.net_config is not None:
            io.save_dict(self.net_config, config_dir, name='network_config')
        if self.lr_config is not None:
            io.save_dict(self.lr_config, config_dir, name='lr_config')
        io.save_dict(self.params, config_dir, name='dynamics_params')

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
                try:
                    xnets = self.xnet[idx]  # type: tf.keras.models.Model
                    vnet = self.vnet[idx]  # type: tf.keras.models.Model
                    xnets[0].save(xf0)
                    xnets[1].save(xf1)
                    vnet.save(vf)
                except TypeError:
                    continue
        else:
            xnet_paths = os.path.join(models_dir, 'dynamics_xnet')
            vnet_paths = os.path.join(models_dir, 'dynamics_vnet')
            self.xnet.save(xnet_paths)
            self.vnet.save(vnet_paths)


    def _parse_net_weights(self, net_weights: NetWeights):
        self._xsw = net_weights.x_scale
        self._xtw = net_weights.x_translation
        self._xqw = net_weights.x_transformation
        self._vsw = net_weights.v_scale
        self._vtw = net_weights.v_translation
        self._vqw = net_weights.v_transformation

        return net_weights

    def _parse_params(
            self,
            params: dict[str, Any],
            net_weights: NetWeights = None
    ):
        """Set instance attributes from `params`."""
        self.xdim = params.get('xdim', None)
        self.batch_size = params.get('batch_size', None)
        #  self.using_hvd = params.get('horovod', False)
        self.x_shape = (self.batch_size, self.xdim)
        self.clip_val = params.get('clip_val', 0.)
        self.aux_weight = params.get('aux_weight', 0.)

        # Determine if there are any parameters to be trained
        self._has_trainable_params = True
        if self.config.hmc and self.config.eps_fixed:
            self._has_trainable_params = False

        if net_weights is None:
            if self.config.hmc:
                net_weights = NetWeights(*(6 * [0.]))
            else:
                net_weights = NetWeights(*(6 * [1.]))

        self.net_weights = net_weights
        self._xsw = self.net_weights.x_scale
        self._xtw = self.net_weights.x_translation
        self._xqw = self.net_weights.x_transformation
        self._vsw = self.net_weights.v_scale
        self._vtw = self.net_weights.v_translation
        self._vqw = self.net_weights.v_transformation

        return {
            'xdim': self.xdim,
            'batch_size': self.batch_size,
            'x_shape': self.x_shape,
            'clip_val': self.clip_val,
        }
