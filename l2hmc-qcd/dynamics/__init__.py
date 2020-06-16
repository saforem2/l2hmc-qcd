from collections import namedtuple


DynamicsConfig = namedtuple('DynamicsConfig', [
    'num_steps', 'eps', 'input_shape',
    'hmc', 'eps_trainable', 'net_weights',
    'model_type',
])

State = namedtuple('State', ['x', 'v', 'beta'])
MonteCarloStates = namedtuple('MonteCarloStates', ['init', 'proposed', 'out'])

