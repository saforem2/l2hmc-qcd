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

# pylint:disable=too-many-instance-attributes

@dataclass
class DynamicsConfig:
    eps: float
    num_steps: int
    clip_val: float = 0.
    hmc: bool = False
    eps_fixed: bool = False
    aux_weight: float = 0.
    loss_scale: float = 1.
    use_mixed_loss: bool = False
    verbose: bool = False


@dataclass
class GaugeDynamicsConfig:
    eps: float                          # step size
    #  xdim: int                        # dimensionality of target distribution
    num_steps: int                      # n leapfrog steps per acc/rej
    hmc: bool = False                   # run standard HMC?
    eps_fixed: bool = False             # Fixed step size?
    aux_weight: float = 0.              # Weight of aux term in loss fn
    loss_scale: float = 1.              # Scale loss?
    use_mixed_loss: bool = False        # Use mixed loss?
    verbose: bool = False               # Verbose metric logging?
    use_ncp: bool = False               # Transform x using NCP?
    x_shape: tuple = None               # (batch_size Lt Lx 2)
    plaq_weight: float = 0.             # Weight of plaq term in loss
    charge_weight: float = 0.           # Weight of charge term in loss
    zero_init: bool = False             # Initialize weights as zeros?
    directional_updates: bool = False   # Use directional updates?
    combined_updates: bool = False      # Use combined v updates?
    separate_networks: bool = False     # Use separate nets?
    use_conv_net: bool = False          # Use conv nets?
    use_scattered_xnet_update: bool = False  # scattered xupdate?
    use_tempered_traj: bool = False     # Use tempered trajectory?
    gauge_eq_masks: bool = False        # Use gauge eq. masks?
    log_dir: str = None                 # `log_dir` containing loadable nets
    optimizer: str = 'adam'             # optimizer to use for backprop
    net_weights: NetWeights = None
