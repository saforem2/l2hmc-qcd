"""
config.py
"""
# pylint:disable=too-many-arguments
from __future__ import absolute_import, division, print_function

import os

from collections import namedtuple

from colorama import Fore

import numpy as np
import tensorflow as tf

from utils.attr_dict import AttrDict

__author__ = 'Sam Foreman'
__date__ = '07/03/2020'

SNAME = 'scale_layer'
TNAME = 'translation_layer'
QNAME = 'transformation_layer'
SCOEFF = 'coeff_scale'
QCOEFF = 'coeff_transformation'

# ----------------------------------------------------------------
# Included below is a catch-all for various structures
# ----------------------------------------------------------------

CBARS = {
    'black': Fore.BLACK,
    'red': Fore.RED,
    'green': Fore.GREEN,
    'yellow': Fore.YELLOW,
    'blue': Fore.BLUE,
    'magenta': Fore.MAGENTA,
    'cyan': Fore.CYAN,
    'white': Fore.WHITE,
    'reset': Fore.RESET,
}


NAMES = [
    'step', 'dt', 'loss', 'ploss', 'qloss',
    'px', 'eps', 'beta', 'sumlogdet', '|dq|', 'plaq_err',
]
HSTR = ''.join(["{:^12s}".format(name) for name in NAMES])
SEP = '-' * len(HSTR)
HEADER = '\n'.join([SEP, HSTR, SEP])

# State is an object for grouping the position/momentum
# configurations together with the value of `beta`.
#  lfData = namedtuple('LFdata', ['init', 'proposed', 'prob'])
#  EnergyData = namedtuple('EnergyData', ['init', 'proposed', 'out'])
#  Energy = namedtuple('Energy', ['potential', 'kinetic', 'hamiltonian'])

# generic object for representing a `weight` matrix in the neural net
# contains both the weight matrix and the bias term
Weights = namedtuple('Weights', ['w', 'b'])

State = namedtuple('State', ['x', 'v', 'beta'])
MonteCarloStates = namedtuple('MonteCarloStates', ['init', 'proposed', 'out'])

NetWeights = namedtuple('NetWeights', [
    'x_scale', 'x_translation', 'x_transformation',
    'v_scale', 'v_translation', 'v_transformation'
])

TrainData = namedtuple('TrainData', ['loss', 'px', 'eps'])

ObsData = namedtuple('ObsData', [
    'actions', 'plaqs', 'charges',  # 'charge_diffs'
])

BootstrapData = namedtuple('BootstrapData', ['mean', 'err', 'means_bs'])

#  l2hmcFn = namedtuple('l2hmcFn', ['v1', 'x1', 'x2', 'v2'])
#  l2hmcFns = namedtuple('l2hmcFns',
#                        ['scale', 'translation', 'transformation'])

PI = np.pi
TWO_PI = 2 * PI

NET_WEIGHTS_HMC = NetWeights(0., 0., 0., 0., 0., 0.)
NET_WEIGHTS_L2HMC = NetWeights(0., 1., 1., 1., 1., 1.)

TF_FLOAT = tf.float32
TF_INT = tf.int32
NP_INT = np.int32
TF_FLOATS = {
    'float16': tf.float16,
    'float32': tf.float32,
    'float64': tf.float64,
}
TF_INTS = {
    'int8': tf.int8,
    'int16': tf.int16,
    'int32': tf.int32,
    'int64': tf.int64,
}
NP_FLOATS = {
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
}


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = ROOT_DIR
BASE_DIR = ROOT_DIR
BIN_DIR = os.path.join(PROJECT_DIR, 'bin')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')
PLOTS_DIR = os.path.join(PROJECT_DIR, 'plots')
DOC_DIR = os.path.join(PROJECT_DIR, 'doc')
TEX_FIGURES_DIR = os.path.join(DOC_DIR, 'figures')
GAUGE_LOGS_DIR = os.path.join(LOGS_DIR, 'GaugeModel_logs')
HMC_LOGS_DIR = os.path.join(GAUGE_LOGS_DIR, 'hmc_logs')

FIGSIZE = (4, 3)
COLORS = 5000 * ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
MARKERS = 5000 * ['o', 's', '^', '<', 'd', 'v', 'h', '>', 'p', 'x', '+', '*']
LINESTYLES = 5000 * ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']
#  COLORS = [  # from seaborn `bright` style
#    (0.00784313725490196, 0.24313725490196078, 1.0),
#    (1.0, 0.48627450980392156, 0.0),
#    (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
#    (0.9098039215686274, 0.0, 0.043137254901960784),
#    (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
#    (0.6235294117647059, 0.2823529411764706, 0.0),
#    (0.9450980392156862, 0.2980392156862745, 0.7568627450980392),
#    (0.6392156862745098, 0.6392156862745098, 0.6392156862745098),
#    (1.0, 0.7686274509803922, 0.0),
#    (0.0, 0.8431372549019608, 1.0)
#  ]

# pylint:disable=invalid-name
