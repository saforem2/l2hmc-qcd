"""
config.py
"""
# pylint:disable=too-many-arguments, invalid-name
from __future__ import absolute_import, division, print_function

import os
import json

from collections import namedtuple


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

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = ROOT_DIR
BASE_DIR = ROOT_DIR
BIN_DIR = os.path.join(PROJECT_DIR, 'bin')
TIMING_FILE = os.path.join(BIN_DIR, 'timing_file.log')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')
PLOTS_DIR = os.path.join(PROJECT_DIR, 'plots')
DOC_DIR = os.path.join(PROJECT_DIR, 'doc')
TEX_FIGURES_DIR = os.path.join(DOC_DIR, 'figures')
GAUGE_LOGS_DIR = os.path.join(LOGS_DIR, 'GaugeModel_logs')
HMC_LOGS_DIR = os.path.join(GAUGE_LOGS_DIR, 'hmc_logs')
TRAIN_CONFIGS_FILE = os.path.join(BIN_DIR, 'train_configs.json')
with open(TRAIN_CONFIGS_FILE, 'rt') as f:
    TRAIN_CONFIGS = json.load(f)

TRAIN_CONFIGS = AttrDict(TRAIN_CONFIGS)

PI = np.pi
TWO_PI = 2 * PI

FLOATS = {
    'float16': {
        'numpy': np.float16,
        'tensorflow': tf.float16,
    },
    'float32': {
        'numpy': np.float32,
        'tensorflow': tf.float32,
    },
    'float64': {
        'numpy': np.float64,
        'tensorflow': tf.float64,
    },
}

TF_FLOATS = {
    'float16': tf.float16,
    'float32': tf.float32,
    'float64': tf.float64,
}

TF_FLOAT = FLOATS[tf.keras.backend.floatx()]['tensorflow']
NP_FLOAT = FLOATS[tf.keras.backend.floatx()]['numpy']

NetWeights = namedtuple('NetWeights', [
    'x_scale', 'x_translation', 'x_transformation',
    'v_scale', 'v_translation', 'v_transformation'
])

Weights = namedtuple('Weights', ['w', 'b'])

State = namedtuple('State', ['x', 'v', 'beta'])

MonteCarloStates = namedtuple('MonteCarloStates',
                              ['init', 'proposed', 'out'])

TrainData = namedtuple('TrainData', ['loss', 'px', 'eps'])

ObsData = namedtuple('ObsData', [
    'actions', 'plaqs', 'charges',  # 'charge_diffs'
])

FIGSIZE = (4, 3)
COLORS = 5000 * ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
MARKERS = 5000 * ['o', 's', '^', '<', 'd', 'v', 'h', '>', 'p', 'x', '+', '*']
LINESTYLES = 5000 * ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']

NAMES = [
    'step', 'dt', 'loss', 'ploss', 'qloss',
    'px', 'eps', 'beta', 'sumlogdet', '|dq|', 'plaq_err',
]

HSTR = ''.join(["{:^12s}".format(name) for name in NAMES])
SEP = '-' * len(HSTR)
HEADER = '\n'.join([SEP, HSTR, SEP])
