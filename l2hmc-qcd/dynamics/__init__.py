from collections import namedtuple

import os
import sys

modulepath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(modulepath)


DynamicsConfig = namedtuple('DynamicsConfig', [
    'num_steps', 'eps', 'input_shape',
    'hmc', 'eps_trainable', 'net_weights',
    'model_type',
])

State = namedtuple('State', ['x', 'v', 'beta'])
MonteCarloStates = namedtuple('MonteCarloStates', ['init', 'proposed', 'out'])

