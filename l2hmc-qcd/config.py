"""
config.py
"""
from __future__ import absolute_import, division, print_function

import os

from collections import namedtuple

import numpy as np
import tensorflow as tf

from utils.attr_dict import AttrDict

__author__ = 'Sam Foreman'
__date__ = '07/03/2020'

# pylint:disable=invalid-name

TRAIN_STR = (r"""
  _____          _       _               _     ____  _   _ __  __  ____
 |_   _| __ __ _(_)_ __ (_)_ __   __ _  | |   |___ \| | | |  \/  |/ ___|
   | || '__/ _` | | '_ \| | '_ \ / _` | | |     __) | |_| | |\/| | |
   | || | | (_| | | | | | | | | | (_| | | |___ / __/|  _  | |  | | |___ _ _ _
   |_||_|  \__,_|_|_| |_|_|_| |_|\__, | |_____|_____|_| |_|_|  |_|\____(_|_|_)
                                 |___/
""")

SNAME = 'scale_layer'
TNAME = 'translation_layer'
QNAME = 'transformation_layer'
SCOEFF = 'coeff_scale'
QCOEFF = 'coeff_transformation'

# ----------------------------------------------------------------
# Included below is a catch-all for various structures
# ----------------------------------------------------------------

class DynamicsConfig(AttrDict):
    """Configuration object for `BaseDynamics` object"""

    def __init__(self,
                 eps: float,
                 num_steps: int,
                 hmc: bool = False,
                 model_type: str = None,
                 eps_trainable: bool = True):
        super(DynamicsConfig, self).__init__(
            eps=eps,
            hmc=hmc,
            num_steps=num_steps,
            model_type=model_type,
            eps_trainable=eps_trainable,
        )


class GaugeDynamicsConfig(AttrDict):
    """Configuration object for `GaugeDynamics` object"""

    # pylint:disable=too-many-arguments
    def __init__(self,
                 eps: float,
                 num_steps: int,
                 hmc: bool = False,
                 use_ncp: bool = False,
                 model_type: str = None,
                 eps_trainable: bool = True,
                 separate_networks: bool = False):
        super(GaugeDynamicsConfig, self).__init__(
            eps=eps,
            hmc=hmc,
            use_ncp=use_ncp,
            num_steps=num_steps,
            model_type=model_type,
            eps_trainable=eps_trainable,
            separate_networks=separate_networks
        )


class NetworkConfig(AttrDict):
    """Configuration object for network of `Dynamics` object"""

    def __init__(self,
                 units: list,
                 name: str = None,
                 dropout_prob: float = 0.,
                 activation_fn: callable = tf.nn.relu):
        super(NetworkConfig, self).__init__(
            name=name,
            units=units,
            dropout_prob=dropout_prob,
            activation_fn=activation_fn
        )


class lrConfig(AttrDict):
    """Configuration object for specifying learning rate schedule."""
    def __init__(self,
                 init: float,
                 decay_steps: int,
                 decay_rate: float,
                 warmup_steps: int = 0):
        super(lrConfig, self).__init__(
            init=init,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            warmup_steps=warmup_steps
        )


#  DynamicsConfig = namedtuple('DynamicsConfig', [
#      'eps',
#      'hmc',
#      'num_steps',
#      'model_type',
#      'eps_trainable',
#      'separate_networks',
#      'use_ncp',
#  ])
#
#  NetworkConfig = namedtuple('NetworkConfig', [
#      'type',
#      'units',
#      'dropout_prob',
#      'activation_fn'
#  ])

#  lrConfig = namedtuple('lrConfig', [
#      'init',
#      'decay_steps',
#      'decay_rate',
#      'warmup_steps',
#  ])

NAMES = [
    'step', 'dt', 'loss', 'ploss', 'qloss',
    'px', 'eps', 'beta', 'sumlogdet', '|dq|', 'plaq_err',
]
HSTR = ''.join(["{:^12s}".format(name) for name in NAMES])
SEP = '-' * len(HSTR)
HEADER = '\n'.join([SEP, HSTR, SEP])

#  DynamicsConfig = {
#      'eps': None,
#      'hmc': False,
#      'num_steps': None,
#      'model_type': None,
#      'input_shape': None,
#      'eps_trainable': True,
#      'separate_networks': False,
#  }
#
#  NetConfig = AttrDict({
#      'type': None,
#      'units': None,
#      'dropout_prob': 0.,
#      'activation_fn': tf.nn.relu,
#  })

#  DynamicsConfig = namedtuple('DynamicsConfig', [
#      'eps',
#      'hmc',
#      'num_steps',
#      'model_type',
#      'input_shape',
#      'net_weights',
#      'eps_trainable',
#  ])


# State is an object for grouping the position/momentum
# configurations together with the value of `beta`.
State = namedtuple('State', ['x', 'v', 'beta'])
lfData = namedtuple('LFdata', ['init', 'proposed', 'prob'])
EnergyData = namedtuple('EnergyData', ['init', 'proposed', 'out'])
Energy = namedtuple('Energy', ['potential', 'kinetic', 'hamiltonian'])
MonteCarloStates = namedtuple('MonteCarloStates', ['init', 'proposed', 'out'])

# generic object for representing a `weight` matrix in the neural net
# contains both the weight matrix and the bias term
Weights = namedtuple('Weights', ['w', 'b'])

NetWeights = namedtuple('NetWeights', [
    'x_scale', 'x_translation', 'x_transformation',
    'v_scale', 'v_translation', 'v_transformation'
])

TrainData = namedtuple('TrainData', ['loss', 'px', 'eps'])

ObsData = namedtuple('ObsData', [
    'actions', 'plaqs', 'charges',  # 'charge_diffs'
])

BootstrapData = namedtuple('BootstrapData', ['mean', 'err', 'means_bs'])

l2hmcFn = namedtuple('l2hmcFn', ['v1', 'x1', 'x2', 'v2'])
l2hmcFns = namedtuple('l2hmcFns', ['scale', 'translation', 'transformation'])

PI = np.pi
TWO_PI = 2 * PI

NET_WEIGHTS_HMC = NetWeights(0., 0., 0., 0., 0., 0.)
NET_WEIGHTS_L2HMC = NetWeights(0., 1., 1., 1., 1., 1.)

TF_FLOAT = tf.float32
TF_INT = tf.int32
NP_FLOAT = np.float32
NP_INT = np.int32

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_DIR)
BIN_DIR = os.path.join(BASE_DIR, 'bin')
GAUGE_LOGS_DIR = os.path.join(BASE_DIR, 'gauge_logs')
TEST_LOGS_DIR = os.path.join(BASE_DIR, 'test_logs')
BIN_DIR = os.path.join(BASE_DIR, 'bin')

DEFAULT_FLAGS = AttrDict({
    'log_dir': None,
    'eager_execution': False,
    'restore': False,
    'inference': True,
    'run_steps': 500,
    'horovod': False,
    'rand': True,
    'eps': 0.1,
    'num_steps': 2,
    'hmc': False,
    'eps_fixed': False,
    'beta_init': 1.,
    'beta_final': 3.,
    'train_steps': 50,
    'save_steps': 5,
    'print_steps': 1,
    'logging_steps': 1,
    'hmc_start': True,
    'hmc_steps': 50,
    'dropout_prob': 0.1,
    'warmup_lr': True,
    'warmup_steps': 10,
    'lr_init': 0.001,
    'lr_decay_steps': 1000,
    'lr_decay_rate': 0.96,
    'plaq_weight': 10.,
    'charge_weight': 0.1,
    'separate_networks': False,
    'network_type': 'GaugeNetwork',
    'lattice_shape': [128, 16, 16, 2],
    'units': [512, 256, 256, 256, 512],
})


#  COLORS = 5000 * ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
MARKERS = 5000 * ['o', 's', 'x', 'v', 'h', '^', 'p', '<', 'd', '>', 'o']
LINESTYLES = 5000 * ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']
COLORS = [  # from seaborn `bright` style
    (0.00784313725490196, 0.24313725490196078, 1.0),
    (1.0, 0.48627450980392156, 0.0),
    (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
    (0.9098039215686274, 0.0, 0.043137254901960784),
    (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
    (0.6235294117647059, 0.2823529411764706, 0.0),
    (0.9450980392156862, 0.2980392156862745, 0.7568627450980392),
    (0.6392156862745098, 0.6392156862745098, 0.6392156862745098),
    (1.0, 0.7686274509803922, 0.0),
    (0.0, 0.8431372549019608, 1.0)
]


header = ("{:^12s}" + 8 * "{:^10s}").format(
    "STEP", "t/STEP", "% ACC", "EPS", "BETA",
    "ACTIONS", "PLAQS", "(EXACT)", "dQ"
)
dash0 = (len(header) + 1) * '='
dash1 = (len(header) + 1) * '-'
RUN_HEADER = dash0 + '\n' + header + '\n' + dash1

try:
    import memory_profiler  # noqa: F401

    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

try:
    import matplotlib.pyplot as plt  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import psutil  # noqa: F401

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

