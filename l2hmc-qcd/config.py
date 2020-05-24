from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np

from collections import namedtuple

# ----------------------------------------------------------------
# Included below is a catch-all for various structures
# (namedtuples) that are used project wide in various locations.
# ----------------------------------------------------------------
DynamicsConfig = namedtuple('DynamicsConfig', [
    'num_steps', 'eps', 'input_shape', 'hmc',
    'eps_trainable', 'net_weights',
    'model_type',
])

# State is an object for grouping the position/momentum
# configurations together with the value of `beta`.
State = namedtuple('State', ['x', 'v', 'beta'])
MonteCarloStates = namedtuple('MonteCarloStates', ['init', 'proposed', 'out'])
LFdata = namedtuple('LFdata', ['init', 'proposed', 'prob'])
EnergyData = namedtuple('EnergyData', ['init', 'proposed', 'out'])
Energy = namedtuple('Energy', ['potential', 'kinetic', 'hamiltonian'])

# generic object for representing a `weight` matrix in the neural net
# contains both the weight matrix and the bias term
Weights = namedtuple('Weights', ['w', 'b'])

NetWeights = namedtuple('NetWeights', [
    'x_scale', 'x_translation', 'x_transformation',
    'v_scale', 'v_translation', 'v_transformation'
])

NET_WEIGHTS_HMC = NetWeights(0., 0., 0., 0., 0., 0.)
NET_WEIGHTS_L2HMC = NetWeights(1., 1., 1., 1., 1., 1.)

TrainData = namedtuple('TrainData', ['loss', 'px', 'eps'])

ObsData = namedtuple('ObsData', [
    'actions', 'plaqs', 'charges',  # 'charge_diffs'
])

BootstrapData = namedtuple('BootstrapData', ['mean', 'err', 'means_bs'])

l2hmcFn = namedtuple('l2hmcFn', ['v1', 'x1', 'x2', 'v2'])
l2hmcFns = namedtuple('l2hmcFns', ['scale', 'translation', 'transformation'])

TF_FLOAT = tf.float32
TF_INT = tf.int32
NP_FLOAT = np.float32
NP_INT = np.int32

PI = np.pi
TWO_PI = 2 * PI

#  TF_FLOAT = tf.float64
#  TF_INT = tf.int64
#  NP_FLOAT = np.float64
#  NP_INT = np.int64

#  GLOBAL_SEED = np.random.randint(1e6)
#
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))

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
    import horovod.tensorflow as hvd  # noqa: F401
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False

try:
    import psutil  # noqa: F401
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from comet_ml import Experiment  # noqa: F401
    HAS_COMET = True
except ImportError:
    HAS_COMET = False
