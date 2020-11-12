"""
config.py

Contains configuration objects for various `Dynamics`.
"""
import os

from collections import namedtuple

import tensorflow as tf

from config import BIN_DIR, TIMING_FILE
from typing import NamedTuple
from utils.attr_dict import AttrDict
from dataclasses import dataclass


TF_FLOAT = tf.keras.backend.floatx()

ACTIVATIONS = {
    'relu': tf.nn.relu,
    'tanh': tf.nn.tanh,
    'leaky_relu': tf.nn.leaky_relu
}


State = namedtuple('State', ['x', 'v', 'beta'])
MonteCarloStates = namedtuple('MonteCarloStates', [
    'init', 'proposed', 'out'
])

NetWeights = namedtuple('NetWeights', [
    'x_scale', 'x_translation', 'x_transformation',
    'v_scale', 'v_translation', 'v_transformation'
])

NET_WEIGHTS_HMC = NetWeights(0., 0., 0., 0., 0., 0.)
NET_WEIGHTS_L2MC = NetWeights(1., 1., 1., 1., 1., 1.)


class DynamicsConfig(AttrDict):
    """Configuration object for `BaseDynamics` object"""

    # pylint:disable=too-many-arguments
    def __init__(
            self,
            eps: float,
            num_steps: int,
            hmc: bool = False,
            model_type: str = None,
            eps_fixed: bool = False,
            aux_weight: float = 0.,
            loss_scale: float = 1.,
            use_mixed_loss: bool = False,
            verbose: bool = False,
            **kwargs,
    ):
        super(DynamicsConfig, self).__init__(
            eps=eps,
            hmc=hmc,
            num_steps=num_steps,
            model_type=model_type,
            eps_fixed=eps_fixed,
            aux_weight=aux_weight,
            loss_scale=loss_scale,
            use_mixed_loss=use_mixed_loss,
            verbose=verbose,
        )
        #  self._custom_update(**kwargs)

    def _custom_update(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


# pylint:disable=too-many-locals
class GaugeDynamicsConfig(AttrDict):
    """Configuration object for `GaugeDynamics` object"""

    # pylint:disable=too-many-arguments
    def __init__(
            self,
            eps: float,                    # step size
            num_steps: int,                # n leapfrog steps per acc/rej
            hmc: bool = False,             # run standard HMC?
            model_type: str = None,        # name for model
            eps_fixed: bool = False,       # Fixed step size?
            aux_weight: float = 0.,        # Weight of aux term in loss fn
            loss_scale: float = 1.,        # Scale loss?
            use_mixed_loss: bool = False,  # Use mixed loss?
            verbose: bool = False,         # Verbose metric logging?
            use_ncp: bool = False,         # Transform x using NCP?
            lattice_shape: tuple = None,   # (batch_size, Lt, Lx, 2)
            plaq_weight: float = 0.,       # Weight of plaq term in loss
            charge_weight: float = 0.,     # Weight of charge term in loss
            zero_init: bool = False,       # Initialize weights as zeros?
            directional_updates: bool = False,  # Use directional updates?
            combined_updates: bool = False,     # Use combined v updates?
            separate_networks: bool = False,    # Use separate nets?
            use_conv_net: bool = False,         # Use conv nets?
            use_scattered_xnet_update: bool = False,  # scattered xupdate?
            use_tempered_traj: bool = False,  # Use tempered trajectory?
            gauge_eq_masks: bool = False,      # Use gauge eq. masks?
            **kwargs,                      # Additional KeywordArguments
    ):
        super(GaugeDynamicsConfig, self).__init__(
            eps=eps,
            hmc=hmc,
            num_steps=num_steps,
            model_type=model_type,
            eps_fixed=eps_fixed,
            aux_weight=aux_weight,
            loss_scale=loss_scale,
            use_mixed_loss=use_mixed_loss,
            verbose=verbose,
            use_ncp=use_ncp,
            lattice_shape=lattice_shape,
            plaq_weight=plaq_weight,
            charge_weight=charge_weight,
            zero_init=zero_init,
            directional_updates=directional_updates,
            combined_updates=combined_updates,
            separate_networks=separate_networks,
            use_conv_net=use_conv_net,
            use_scattered_xnet_update=use_scattered_xnet_update,
            use_tempered_traj=use_tempered_traj,
            gauge_eq_masks=gauge_eq_masks,
        )
        #  self._custom_update(**kwargs)
        #  self._custom_update(**kwargs)

    def _custom_update(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
#
#  @dataclass
#  class DynamicsConfig:
#      """Configuration object for `BaseDynamics` class"""
#      eps: float
#      num_steps: int
#      hmc: bool = False
#      model_type: str = None
#      eps_fixed: bool = False
#      aux_weight: float = 0.
#      loss_scale: float = 1.
#      use_mixed_loss: bool = False
#      verobse: bool = False
#
#
#  @dataclass
#  class GaugeDynamicsConfig:
#      """Configuration object for `GaugeDynamics` object."""
#      eps: float
#      num_steps: int
#      hmc: bool = False
#      model_type: str = None
#      eps_fixed: bool = False
#      aux_weight: float = 0.
#      loss_scale: float = 1.
#      use_mixed_loss: bool = False
#      verbose: bool = False
#      use_ncp: bool = False
#      lattice_shape: tuple = None
#      plaq_weight: float = 0.
#      charge_weight: float = 1.
#      zero_init: bool = False
#      directional_updates: bool = False
#      combined_updates: bool = False
#      separate_networks: bool = False
#      use_conv_net: bool = False
#      use_scattered_xnet_update: bool = False
#      use_tempered_traj: bool = False
#      gauge_eq_masks: bool = False
#
#
