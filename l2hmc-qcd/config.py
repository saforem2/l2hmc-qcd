import tensorflow as tf
import numpy as np
import os

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from comet_ml import Experiment
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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))

COLORS = 5000 * ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
MARKERS = 5000 * ['o', 's', 'x', 'v', 'h', '^', 'p', '<', 'd', '>', 'o']
LINESTYLES = 5000 * ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']

GLOBAL_SEED = 42


h_str = ("{:^12s}{:^10s}{:^10s}{:^10s}{:^10s}"
         "{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}")
h_strf = h_str.format("STEP", "LOSS", "t/STEP", "% ACC", "EPS",
                      "BETA", "ACTION", "PLAQ", "(EXACT)", "dQ", "LR")
dash0 = (len(h_strf) + 1) * '-'
dash1 = (len(h_strf) + 1) * '-'
TRAIN_HEADER = dash0 + '\n' + h_strf + '\n' + dash1

header = ("{:^12s}{:^10s}{:^10s}{:^10s}"
          "{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}")

header = header.format("STEP", "t/STEP", "% ACC", "EPS", "BETA",
                       "ACTIONS", "PLAQS", "(EXACT)", "dQ")
dash0 = (len(header) + 1) * '='
dash1 = (len(header) + 1) * '-'
RUN_HEADER = dash0 + '\n' + header + '\n' + dash1

# ---------------------------------------
# Default parameters for gauge_model.py
# ---------------------------------------
PARAMS = {
    'space_size': 8,
    'time_size': 8,
    'link_type': 'U1',
    'dim': 2,
    'num_samples': 128,
    'rand': True,
    'num_steps': 5,
    'eps': 0.2,
    'loss_scale': 0.1,
    'lr_init': 0.001,
    'lr_decay_steps': 1000,
    'lr_decay_rate': 0.96,
    'warmup_lr': False,
    'fixed_beta': False,
    'hmc_beta': None,
    'hmc_eps': None,
    'beta_init': 2.0,
    'beta_final': 5.0,
    'inference': False,
    'beta_inference': None,
    'charge_weight_inference': None,
    'train_steps': 5000,
    'run_steps': 10000,
    'trace': False,
    'save_steps': 2500,
    'print_steps': 1,
    'logging_steps': 10,
    'network_arch': 'conv2D',
    'num_hidden': 100,
    'summaries': True,
    'plot_lf': False,
    'loop_net_weights': False,
    'hmc': False,
    'run_hmc': False,
    'eps_fixed': False,
    'metric': 'cos_diff',
    'inverse_loss': True,
    'nnehmc_loss': False,
    'std_weight': 1.0,
    'aux_weight': 1.0,
    'charge_weight': 1.0,
    'profiler': False,
    'gpu': False,
    'use_bn': False,
    'horovod': False,
    'comet': False,
    'dropout_prob': 0.5,
    'save_samples': False,
    'save_lf': True,
    'clip_value': 0.0,
    'restore': False,
    'theta': False,
    'num_intra_threads': 0,
    'float64': False,
    'data_format': 'channels_last',
    'using_hvd': False,
    '_plot': True
}
