import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))

COLORS = 5000 * ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
MARKERS = 5000 * ['o', 's', 'x', 'v', 'h', '^', 'p', '<', 'd', '>', 'o']
LINESTYLES = 5000 * ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']

GLOBAL_SEED = 42

########################################
# Default parameters for gauge_model.py
########################################
PARAMS = {
    # --------------------- Lattice parameters ----------------------------
    'time_size': 8,
    'space_size': 8,
    'link_type': 'U1',
    'dim': 2,
    'num_samples': 56,
    'rand': False,
    'data_format': 'channels_last',
    # --------------------- Leapfrog parameters ---------------------------
    'num_steps': 5,
    'eps': 0.25,
    'loss_scale': 1.,
    # --------------------- Learning rate parameters ----------------------
    'lr_init': 1e-3,
    'lr_decay_steps': 1000,
    'lr_decay_rate': 0.96,
    # --------------------- Annealing rate parameters ---------------------
    'annealing': True,
    #  'annealing_steps': 200,
    #  'annealing_factor': 0.97,
    #  'beta': 2.,
    'beta_init': 2.,
    'beta_final': 4.,
    # --------------------- Training parameters ---------------------------
    'train_steps': 10000,
    'save_steps': 1000,
    'logging_steps': 50,
    'print_steps': 1,
    'training_samples_steps': 1000,
    'training_samples_length': 500,
    # --------------------- Model parameters ------------------------------
    #  'conv_net': True,
    'network_arch': 'conv3D',
    'hmc': False,
    'eps_trainable': True,
    'metric': 'cos_diff',
    #  'aux': True,
    'std_weight': 1.,
    'aux_weight': 1.,
    'charge_weight': 1.,
    #  'charge_loss': True,
    'summaries': True,
    'clip_grads': False,
    'clip_value': None,
}
