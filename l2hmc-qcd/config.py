from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np


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


#  TF_FLOAT = tf.float64
#  NP_FLOAT = np.float64
#  TF_INT = tf.int64
#  NP_INT = np.int64
#
TF_FLOAT = tf.float32
TF_INT = tf.int32
NP_FLOAT = np.float32
NP_INT = np.int32

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))

COLORS = 5000 * ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
MARKERS = 5000 * ['o', 's', 'x', 'v', 'h', '^', 'p', '<', 'd', '>', 'o']
LINESTYLES = 5000 * ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']

GLOBAL_SEED = 9

header = ("{:^12s}" + 8 * "{:^10s}").format(
    "STEP", "t/STEP", "% ACC", "EPS", "BETA",
    "ACTIONS", "PLAQS", "(EXACT)", "dQ"
)
dash0 = (len(header) + 1) * '='
dash1 = (len(header) + 1) * '-'
RUN_HEADER = dash0 + '\n' + header + '\n' + dash1
