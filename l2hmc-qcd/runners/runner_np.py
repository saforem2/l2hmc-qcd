"""
runner_np.py

Implements `RunnerNP` object responsible for performing tensorflow-independent
inference on a trained model.

Author: Sam Foreman (github: @saforem2)
Date: 01/09/2020
"""
import os
import time
import pickle

import pandas as pd

import utils.file_io as io

from config import NetWeights, State, Weights, Energy
from lattice.lattice import GaugeLattice, u1_plaq_exact, calc_plaqs_diffs
from plotters.data_utils import therm_arr, bootstrap
from dynamics.dynamics_np import DynamicsNP
from utils.file_io import timeit
from utils.parse_inference_args_np import parse_args as parse_inference_args

try:
    import autograd.numpy as np
except ImportError:
    HAS_AUTOGRAD = False
    import numpy as np

# pylint: disable=no-member
# pylint: disable=inconsistent-return-statements
# pylint: disable=no-else-return
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments, invalid-name


NET_WEIGHTS_HMC = NetWeights(0, 0, 0, 0, 0, 0)
NET_WEIGHTS_L2HMC = NetWeights(1, 1, 1, 1, 1, 1)

HEADER = ("{:^13s}" + 9 * "{:^12s}").format(
    "STEP", "t/STEP", "% ACC", "𝞭x_out", "𝞭x_prop",
    "exp(𝞭H)", "sumlogdet", "𝞭𝜙", "𝞭x_r", "𝞭v_r",
)
SEPERATOR = len(HEADER) * '-'


def _update_dict(_dict, data, force=True):
    """Update `_dict` using `key, value` pairs from `data`."""
    for key, val in data.items():
        try:
            _dict[key].append(val)
        except KeyError:
            if force:
                _dict[key] = [val]
            else:
                continue

    return _dict


def _strf(x):
    """Format the number x as a string."""
    if np.allclose(x - np.around(x), 0):
        xstr = f'{int(x)}'
    else:
        xstr = f'{x:.1}'.replace('.', '')
    return xstr


def load_pkl(pkl_file):
    """Load from `pkl_file`."""
    with open(pkl_file, 'rb') as f:
        tmp = pickle.load(f)
    return tmp


def sum_squared_diff(x, y):
    """Calculate the Euclidean distance between `x1` and `x2`."""
    return np.sqrt(np.sum((x - y) ** 2))


def cos_metric(x, y):
    """Calculate the difference between x1, x2 using gauge metric."""
    return np.mean(1. - np.cos(x - y), axis=-1)


def create_lattice(params):
    """Create `GaugeLattice` object from `params`."""
    return GaugeLattice(time_size=params['time_size'],
                        space_size=params['space_size'],
                        dim=params['dim'], link_type='U1',
                        batch_size=params['batch_size'])


def _load_rp(run_dirs, idx=0):
    rp_file = os.path.join(run_dirs[idx], 'run_params.pkl')
    if os.path.isfile(rp_file):
        run_params = load_pkl(rp_file)
        return run_params
    else:
        idx += 1
        _load_rp(run_dirs, idx)


def _get_eps(log_dir):
    """Get the step size `eps` by looking for it in `log_dir` ."""
    try:
        in_file = os.path.join(log_dir, 'eps_np.pkl')
        eps_dict = load_pkl(in_file)
        eps = eps_dict['eps']
    except FileNotFoundError:
        run_dirs = io.get_run_dirs(log_dir)
        rp_file = os.path.join(run_dirs[0], 'run_params.pkl')
        if os.path.isfile(rp_file):
            run_params = load_pkl(rp_file)
        else:
            rp_file = os.path.join(run_dirs[-1], 'run_params.pkl')
            if os.path.isfile(rp_file):
                run_params = load_pkl(rp_file)
            else:
                raise FileNotFoundError('Unable to load run_params.')
        eps = run_params['eps']
    return eps


def _update_params(params, **kwargs):
    """Update params with new values for `eps`, `num_steps`, `batch_size`."""
    #  def _update_params(params, eps=None, num_steps=None, batch_size=None):
    for key, val in kwargs.items():
        if val is not None:
            if key in ['num_steps', 'batch_size']:
                params[key] = int(val)
            else:
                params[key] = val

    eps = kwargs.get('eps', None)
    if eps is None:
        eps = _get_eps(params['log_dir'])

    params['eps'] = float(eps)

    return params


def reduced_weight_matrix(W, n=10):
    """Use the first `n` singular vals to reconstruct `W`."""
    U, S, V = np.linalg.svd(W, full_matrices=True)
    S_ = np.zeros((W.shape[0], W.shape[1]))
    S_[:W.shape[0], :W.shape[0]] = np.diag(S)
    S_ = S_[:, :n]
    V = V[:n, :]
    W_ = U @ S_ @ V

    return W_


def get_reduced_weights(weights, n=10):
    """Keep the first n singular vals of each weight matrix in weights."""
    for key, val in weights['xnet']['GenericNet'].items():
        if 'layer' in key:
            W, b = val
            W_ = reduced_weight_matrix(W, n=n)
            weights['xnet']['GenericNet'][key] = Weights(W_, b)
    for key, val in weights['vnet']['GenericNet'].items():
        if 'layer' in key:
            W, b = val
            W_ = reduced_weight_matrix(W, n=n)
            weights['vnet']['GenericNet'][key] = Weights(W_, b)

    return weights


def create_dynamics(log_dir,
                    potential_fn,
                    x_dim,
                    model_type='GaugeModel',
                    **kwargs):
    """Create `DynamicsNP` object for running dynamics imperatively."""
    params = load_pkl(os.path.join(log_dir, 'parameters.pkl'))
    #  params = _update_params(params, eps, num_steps, batch_size)

    # Update params using command line arguments to override defaults
    params = _update_params(params, **kwargs)
    #  params['model_type'] = model_type

    # load saved weights from `.pkl` file:
    with open(os.path.join(log_dir, 'weights.pkl'), 'rb') as f:
        weights = pickle.load(f)

    # to run with reduced weight matrix keeping first `num_singular_values`
    # singular values from the SVD decomposition: W = U * S * V^{H}
    num_singular_values = kwargs.get('num_singular_values', -1)
    if num_singular_values > 0:
        io.log(f'Keeping the first {num_singular_values} singular values!')
        weights = get_reduced_weights(weights, n=num_singular_values)

    dynamics = DynamicsNP(potential_fn,
                          x_dim=x_dim,
                          weights=weights,
                          model_type=model_type,
                          **params)
    io.log(f'Dynamics._model_type: {dynamics._model_type}\n')

    mask_file = os.path.join(log_dir, 'dynamics_mask.pkl')
    if os.path.isfile(mask_file):
        with open(mask_file, 'rb') as f:
            masks = pickle.load(f)
        dynamics.set_masks(masks)

    return dynamics


def _calc_energies(dynamics, x, v, beta):
    """Calculate the potential/kinetic energies and the Hamiltonian."""
    potential_energy = dynamics.potential_energy(x, beta)
    kinetic_energy = dynamics.kinetic_energy(v)
    hamiltonian = dynamics.hamiltonian(x, v, beta)

    return potential_energy, kinetic_energy, hamiltonian


def calc_energies(dynamics, x_init, outputs, beta):
    """Calculate initial, proposed, and output energies."""
    pe_init, ke_init, h_init = _calc_energies(dynamics, x_init,
                                              outputs['v_init'], beta)

    pe_prop, ke_prop, h_prop = _calc_energies(dynamics,
                                              outputs['x_proposed'],
                                              outputs['v_proposed'], beta)

    pe_out, ke_out, h_out = _calc_energies(dynamics,
                                           outputs['x_out'],
                                           outputs['v_out'], beta)

    outputs = {
        'potential_init': pe_init,
        'potential_proposed': pe_prop,
        'potential_out': pe_out,
        'kinetic_init': ke_init,
        'kinetic_proposed': ke_prop,
        'kinetic_out': ke_out,
        'hamiltonian_init': h_init,
        'hamiltonian_proposed': h_prop,
        'hamiltonian_out': h_out,
        'exp_energy_diff': np.exp(h_init - h_out),
    }

    return outputs


def _check_param(dynamics, param=None):
    """Check param against it's value as an attribute of `dynamics`."""
    attr = getattr(dynamics, str(param), None)
    if param is None:
        param = attr
    if param != attr:
        setattr(dynamics, str(param), param)

    return dynamics, param


def _init_dicts():
    """Initialize dictionaries to store inference data."""
    energy_keys = [
        'potential_init', 'potential_proposed', 'potential_out',
        'kinetic_init', 'kinetic_proposed', 'kinetic_out',
        'hamiltonian_init', 'hamiltonian_proposed', 'hamiltonian_out',
        'exp_energy_diff'
    ]
    data_keys = [
        'plaqs', 'actions', 'charges',
        'dx_proposed', 'dx_out',
        'sumlogdet_proposed', 'sumlogdet_out',
        'accept_prob', 'rand_num', 'mask_a',
        'forward', 'xdiff_r', 'vdiff_r',
        #  'xdiff_r0', 'xdiff_r1', 'vdiff_r0', 'vdiff_r1',
    ]
    reverse_keys = ['xdiff_fb', 'xdiff_bf', 'vdiff_fb', 'vdiff_bf']
    run_data = {key: [] for key in data_keys}
    energy_data = {key: [] for key in energy_keys}
    reverse_data = {key: [] for key in reverse_keys}

    return run_data, energy_data, reverse_data


def _get_run_str(run_params, init='rand'):
    run_steps = run_params.get('run_steps', None)
    beta = run_params.get('beta', None)
    net_weights = run_params.get('net_weights', None)
    eps = run_params.get('eps', None)
    num_steps = run_params.get('num_steps', None)
    batch_size = run_params.get('batch_size', None)
    mix_samplers = run_params.get('mix_samplers', False)
    direction = run_params.get('direction', 'rand')
    zero_masks = run_params.get('zero_masks', False)
    num_singular_values = run_params.get('num_singular_values', -1)

    nw_str = ''.join(
        (_strf(i).replace('.', '') for i in net_weights)
    )
    beta_str = f'{beta}'.replace('.', '')
    eps_str = f'{eps:.3g}'.replace('.', '')
    run_str = (f'lf{num_steps}'
               f'_bs{batch_size}'
               f'_steps{run_steps}'
               f'_beta{beta_str}'
               f'_eps{eps_str}'
               f'_nw{nw_str}'
               f'_{init}')

    if zero_masks:
        run_str += f'_zero_masks'

    if mix_samplers:
        run_str += f'_mix_samplers'

    if direction != 'rand':
        run_str += f'_{direction}'

    if num_singular_values > 0:
        run_str += f'_nsv{num_singular_values}'

    time_strs = io.get_timestr()
    timestr = time_strs['timestr']
    run_str += f'__{timestr}'

    return run_str


def _inference_setup(log_dir, dynamics, run_params):
    """Setup for inference run."""
    #  , init='rand', skip=True):
    #  run_steps = run_params['run_steps']
    #  beta = run_params['beta']
    #  net_weights = run_params['net_weights']
    #  num_singular_values = run_params.get('num_singular_values', -1)
    eps = run_params.get('eps', None)
    #  print(f'\n\n eps: {eps}\n dynamics.eps: {dynamics.eps}\n\n')
    num_steps = run_params.get('num_steps', None)
    batch_size = run_params.get('batch_size', None)
    init = run_params.get('init', 'rand')

    #  batch_size = dynamics.batch_size

    dynamics, batch_size = _check_param(dynamics, batch_size)
    dynamics, num_steps = _check_param(dynamics, num_steps)
    dynamics, eps = _check_param(dynamics, eps)
    run_params.update({
        'eps': eps,
        'num_steps': num_steps,
        'batch_size': batch_size,
    })

    init = str(init).lower()
    if init == 'rand' or init is None:
        init = 'rand'
        samples = np.random.randn(batch_size, dynamics.x_dim)
    if init == 'zeros':
        samples = np.zeros((batch_size, dynamics.x_dim))
    if init == 'ones':
        samples = np.ones((batch_size, dynamics.x_dim))
    else:
        init = 'rand'
        io.log(f'init: {init}\n')
        samples = np.random.randn(batch_size, dynamics.x_dim)

    run_str = _get_run_str(run_params)
    run_dir = os.path.join(log_dir, 'runs_np', run_str)
    io.check_else_make_dir(run_dir)

    run_params['run_str'] = run_str
    run_params['run_dir'] = run_dir

    return samples, run_params, run_dir


def reverse_dynamics(dynamics, state, net_weights, forward_first=True):
    """Check reversibility of dynamics by running either:
        1. backward(forward(state)) if `forward_first=True`
        2. forward(backward(state)) if `forward_first=False`

    Args:
        dynamics (`DynamicsNP` object): Dynamics on which to run.
        state (`State`, namedtuple): Initial state (x, v, beta).
        net_weights (`NetWeights`, namedtuple): NetWeights multiplicative
            scaling factor.
        forward_first (bool): Whether the forward direction should be ran
            first.

    Returns:
        state_bf (`State` object): Resultant state (xbf, vbf, beta).
    """
    ff = forward_first
    xprop1, vprop1, px1, _ = dynamics.transition_kernel(*state,
                                                        net_weights,
                                                        forward=ff)
    xprop1 = np.mod(xprop1, 2 * np.pi)
    state_prop1 = State(x=xprop1, v=vprop1, beta=state.beta)

    xprop2, vprop2, px2, _ = dynamics.transition_kernel(*state_prop1,
                                                        net_weights,
                                                        forward=(not ff))
    xprop2 = np.mod(xprop2, 2 * np.pi)
    state_prop2 = State(x=xprop2, v=vprop2, beta=state.beta)

    return state_prop2


def check_reversibility_np(dynamics,
                           state,
                           net_weights,
                           step=None,
                           out_file=None,
                           to_csv=False):
    """Check reversibility explicitly.

    Args:
        dynamics (`DynamicsNP` object): Dynamics on which to run.
        state (`State`, namedtuple): Initial state (x, v, beta).
        net_weights (`NetWeights`, namedtuple): NetWeights multiplicative
            scaling factor.

    Returns:
        diff_fb (tuple): Tuple (x, v) of the sum squared differences between
            input and output state obtained by running the dynamics via
            forward(backward(state)).
        str_fb (tuple): String representation of `diff_fb`.
        diff_bf (tuple): Tuple (x, v) of the sum squared difference between
            input and output state obtained by running the dynamics via
            forward(backward(state)).
        str_bf (tuple): String representation of `diff_bf`.
    """
    state_bf = reverse_dynamics(dynamics, state, net_weights,
                                forward_first=True)
    state_fb = reverse_dynamics(dynamics, state, net_weights,
                                forward_first=False)

    xdiff_bf = state.x - state_bf.x
    vdiff_bf = state.v - state_bf.v

    xdiff_fb = state.x - state_fb.x
    vdiff_fb = state.v - state_fb.v
    diff_dict = {
        'xdiff_bf': np.squeeze(xdiff_bf.flatten()),
        'vdiff_bf': np.squeeze(vdiff_bf.flatten()),
        'xdiff_fb': np.squeeze(xdiff_fb.flatten()),
        'vdiff_fb': np.squeeze(vdiff_fb.flatten()),
    }

    if to_csv:
        diff_df = pd.DataFrame(diff_dict)
        if out_file is not None:
            header = (step == 0)
            diff_df.to_csv(out_file, mode='a', header=header, index=False)

    return diff_dict


def inference_step(step, x_init, dynamics, lattice, **run_params):
    """Run a single inference step."""
    x_init = np.mod(x_init, 2 * np.pi)
    beta = run_params.get('beta', None)
    run_steps = run_params.get('run_steps', None)
    net_weights = run_params.get('net_weights', None)
    #  plaq_exact = u1_plaq_exact(beta)

    start_time = time.time()
    output = dynamics.apply_transition(x_init, beta, net_weights)
    time_diff = time.time() - start_time

    x_out = np.mod(output['x_out'], 2 * np.pi)
    output['x_out'] = x_out

    observables = lattice.calc_observables_np(samples=x_out)
    plaq_diff = calc_plaqs_diffs(observables['plaqs'], beta)
    #  plaq_diff = plaq_exact - observables['plaqs']

    #  dx_prop = cos_metric(output['x_proposed'], x_init)
    observables['accept_prob'] = output['accept_prob']
    observables['dx_out'] = cos_metric(x_out, x_init)
    observables['dx_proposed'] = cos_metric(output['x_proposed'], x_init)
    observables['xdiff_r'] = output['xdiff_r'].mean(axis=-1)
    observables['vdiff_r'] = output['vdiff_r'].mean(axis=-1)

    #  xdiff_r = output['xdiff_r'].reshape(lattice.batch_size,
    #                                      *lattice.links_shape)
    #  vdiff_r = output['vdiff_r'].reshape(lattice.batch_size,
    #                                      *lattice.links_shape)
    #
    #  observables['xdiff_r0'] = xdiff_r[..., 0].reshape(lattice.batch_size, -1)
    #  observables['xdiff_r1'] = xdiff_r[..., 1].reshape(lattice.batch_size, -1)
    #  observables['vdiff_r0'] = vdiff_r[..., 0].reshape(lattice.batch_size, -1)
    #  observables['vdiff_r1'] = vdiff_r[..., 1].reshape(lattice.batch_size, -1)
    #  observables['xdiff_r0'] = output['xdiff_r0']
    #  observables['xdiff_r1'] = output['xdiff_r1']
    #  observables['vdiff_r0'] = output['vdiff_r0']
    #  observables['vdiff_r1'] = output['vdiff_r1']

    edata = calc_energies(dynamics, x_init, output, beta)

    #  xdiff = (observables['xdiff_r0'] + observables['xdiff_r1']) / 2
    #  vdiff = (observables['vdiff_r0'] + observables['vdiff_r1']) / 2
    data_str = (f"{step:>6g}/{run_steps:<6g} "
                f"{time_diff:^11.4g} "
                f"{output['accept_prob'].mean():^11.4g} "
                f"{observables['dx_out'].mean():^11.4g} "
                f"{observables['dx_proposed'].mean():^11.4g} "
                #  f"{output['forward']:^11.4g} "
                f"{edata['exp_energy_diff'].mean():^11.4g} "
                f"{output['sumlogdet_out'].mean():^11.4g} "
                f"{plaq_diff.mean():^11.4g} "
                f"{observables['xdiff_r'].mean():^11.4g} "
                f"{observables['vdiff_r'].mean():^11.4g}")

    outputs = {
        'data_str': data_str,
        'energy_data': edata,
        'dynamics_output': output,
        'observables': observables,
    }

    return outputs


def update_data(run_data, energy_data, outputs):
    """Update data using `outputs`."""
    for key, val in outputs['observables'].items():
        try:
            run_data[key].append(val)
        except KeyError:
            run_data[key] = [val]

    #  for key, val in outputs['dynamics_output'].items():
    #      if key in list(run_data.keys()):
    #          try:
    #              run_data[key].append(val)
    #          except KeyError:
    #              io.log(f'KeyError: {key}\n')
    #  for key in run_data.keys():
    #      try:
    #          run_data[key].append(outputs['dynamics_output'][key])
    #      except KeyError:
    #          io.log(f'KeyError: {key}\n')
    #          continue

    run_data['sumlogdet_out'].append(
        outputs['dynamics_output']['sumlogdet_out']
    )
    run_data['sumlogdet_proposed'].append(
        outputs['dynamics_output']['sumlogdet_proposed']
    )
    run_data['rand_num'].append(outputs['dynamics_output']['rand_num'])
    run_data['forward'].append(outputs['dynamics_output']['forward'])
    run_data['mask_a'].append(outputs['dynamics_output']['mask_a'])
    #  run_data['xdiff_r0'].append(outputs['dynamics_output']['xdiff_r0'])
    #  run_data['xdiff_r1'].append(outputs['dynamics_output']['xdiff_r1'])
    #  run_data['vdiff_r0'].append(outputs['dynamics_output']['vdiff_r0'])
    #  run_data['vdiff_r1'].append(outputs['dynamics_output']['vdiff_r1'])

    for key, val in outputs['energy_data'].items():
        try:
            energy_data[key].append(val)
        except KeyError:
            energy_data[key] = [val]

    return run_data, energy_data


def _run_np(steps, nws, dynamics, lattice, samples, run_params, data):
    """Run inference with different net_weights."""
    run_data = data.get('run_data', None)
    energy_data = data.get('energy_data', None)
    data_strs = data.get('data_strs', None)

    io.log('\n\n' + SEPERATOR + '\n')
    io.log(f"RUNNING INFERENCE WITH:\n "
           f"\tnet_weights: {run_params['net_weights']}"
           f"\tsteps: {steps}\n")

    _run_params = run_params.copy()
    _run_params['net_weights'] = nws
    _run_params['run_steps'] = steps
    samples = np.mod(samples, 2 * np.pi)
    for step in range(steps):
        if step % 100 == 0:
            io.log(SEPERATOR + '\n' + HEADER + '\n' + SEPERATOR)

        outputs = inference_step(step, samples,
                                 dynamics, lattice,
                                 **_run_params)
        samples = np.mod(outputs['dynamics_output']['x_out'], 2 * np.pi)
        outputs['dynamics_output']['x_out'] = samples
        run_data, energy_data = update_data(run_data, energy_data, outputs)
        if step % run_params['print_steps'] == 0:
            data_strs.append(outputs['data_str'])
            io.log(outputs['data_str'])

    data = {
        'energy_data': energy_data,
        'run_data': run_data,
        'data_strs': data_strs,
    }
    io.log(SEPERATOR + '\n\n' + 'Back to original sampler...\n')
    io.log(SEPERATOR + '\n' + HEADER + '\n' + SEPERATOR)

    return samples, data


def _run_hmc_np(steps, dynamics, lattice, samples, run_params, **data):
    """Run HMC for a few steps intermittently during L2HMC inference run."""
    run_data = data.get('run_data', None)
    energy_data = data.get('energy_data', None)
    data_strs = data.get('data_strs', None)
    io.log('\n\n' + SEPERATOR + '\n'
           + f'RUNNING GENERIC HMC FOR {steps} steps...')

    # Create copy of `run_params` with `net_weights` set to all zeros
    hmc_params = run_params.copy()
    hmc_params['net_weights'] = NetWeights(0, 0, 0, 0, 0, 0)
    hmc_params['run_steps'] = steps

    for step in range(steps):
        if step % 100 == 0:
            io.log(SEPERATOR + '\n' + HEADER + '\n' + SEPERATOR)

        samples_init = np.mod(samples, 2 * np.pi)
        outputs = inference_step(step, samples_init, dynamics,
                                 lattice, **hmc_params)
        samples = outputs['dynamics_output']['x_out']
        if energy_data is not None and run_data is not None:
            run_data, energy_data = update_data(run_data,
                                                energy_data,
                                                outputs)
            if data_strs is not None:
                data_strs.append(outputs['data_str'])
        #  run_data, energy_data = update_data(run_data, energy_data, outputs)

        io.log(outputs['data_str'])

    data = {
        'energy_data': energy_data,
        'run_data': run_data,
        'data_strs': data_strs,
    }

    io.log(SEPERATOR + '\n\n' + 'Back to running L2HMC...\n')
    io.log(SEPERATOR + '\n' + HEADER + '\n' + SEPERATOR)

    return samples, data


def _get_reverse_data(run_dir):
    """Load reverse data from `reversibility_results.csv` in `run_dir`."""
    rdata_file = os.path.join(run_dir, 'reversibility_results.csv')
    if os.path.isfile(rdata_file):
        return pd.read_csv(rdata_file)

    return None


@timeit
def run_inference_np(log_dir, dynamics, lattice, run_params, save=True):
    """Run inference imperatively w/ numpy using `dynamics` object.

    Args:
        log_dir (str): Path to `log_dir` containing trained model on which to
            run inference.
        dynamics (dynamicsNP object): Dynamics engine for running the sampler.
        lattice (GaugeLattice object): Lattice object on which the model is
            defined.
        run_params (dict): Dictionary of parameters to use for inference run.
    """
    #  init = run_params.get('init', 'rand')
    mix_samplers = run_params.get('mix_samplers', False)
    samples, run_params, run_dir = _inference_setup(log_dir,
                                                    dynamics,
                                                    run_params)
    run_params['direction'] = dynamics.direction

    beta = run_params['beta']
    run_steps = run_params['run_steps']
    net_weights = run_params['net_weights']

    if run_params.get('mix_samplers', False):
        switch_steps = 2000
        run_steps_alt = 500
        run_params['switch_steps'] = switch_steps
        run_params['run_steps_alt'] = run_steps_alt

        # if running HMC, mix in L2HMC
        if net_weights == NetWeights(0, 0, 0, 0, 0, 0):
            nws = NetWeights(1, 1, 1, 1, 1, 1)
        # if running L2HMC, mix in HMC
        if net_weights == NetWeights(1, 1, 1, 1, 1, 1):
            nws = NetWeights(0, 0, 0, 0, 0, 0)
        else:
            # Switch each value of `net_weights`
            nws = NetWeights(
                *tuple(np.array([not i for i in net_weights], dtype=float))
            )

        run_params_alt = {
            'switch_steps': switch_steps,
            'run_steps_alt': run_steps_alt,
            'net_weights_alt': nws,
        }
        io.save_dict(run_params_alt, run_params['run_dir'], 'run_params_alt')

    io.save_params(run_params, run_dir, name='run_params_')
    reverse_file = os.path.join(run_dir, 'reversibility_results.csv')
    run_data, energy_data, reverse_data = _init_dicts()
    samples = np.mod(samples, 2 * np.pi)

    hstr = SEPERATOR + '\n' + HEADER + '\n' + SEPERATOR

    data_strs = []
    samples_arr = []
    data_strs.append(hstr)
    for step in range(run_steps):
        samples_init = np.mod(samples, 2 * np.pi)
        outputs = inference_step(step, samples_init,
                                 dynamics, lattice, **run_params)
        samples = np.mod(outputs['dynamics_output']['x_out'], 2 * np.pi)
        run_data, energy_data = update_data(run_data, energy_data, outputs)
        samples_arr.append(samples)

        if step % 100 == 0:
            io.log(hstr)

        if mix_samplers and step % switch_steps == 0:
            _data = {
                'run_data': run_data,
                'energy_data': energy_data,
                'data_strs': data_strs,
            }
            samples, _ = _run_np(run_steps_alt, nws,
                                 dynamics, lattice,
                                 samples, run_params, _data)

        if step % run_params['reverse_steps'] == 0:
            v_out = outputs['dynamics_output']['v_out']
            state = State(x=samples, v=v_out, beta=beta)
            reverse_output = check_reversibility_np(dynamics, state,
                                                    net_weights, step,
                                                    reverse_file)
            for key, val in reverse_output.items():
                reverse_data[key].extend(val)

        if step % run_params['print_steps'] == 0:
            io.log(outputs['data_str'])
            data_strs.append(outputs['data_str'])

    data_dict = {
        'run_data': run_data,
        'energy_data': energy_data,
        'reverse_data': reverse_data,
        'samples_arr': samples_arr,
    }

    if save:
        save_inference_data(run_dir, run_params, data_dict, data_strs)

    outputs = {
        'run_params': run_params,
        'data': data_dict,
        'reverse_data': reverse_data,
    }

    return outputs


def summarize_run(run_data, run_params):
    """Summarize the `quality` of a given inference run and save results."""
    beta = run_params['beta']
    #  run_dir = run_params['run_dir']

    #  run_steps = run_params['run_steps']
    #  batch_size = run_params['batch_size']

    def therm_(key):
        return therm_arr(np.array(run_data[key]), ret_steps=False)

    obs_keys = ['charges', 'accept_prob', 'dx_out']
    data = {
        key: therm_(key) for key in obs_keys
    }
    #  data['plaqs_diffs'] = u1_plaq_exact(beta) - therm_('plaqs')
    data['plaqs_diffs'] = calc_plaqs_diffs(therm_('plaqs'), beta)
    #  data['plaqs_diffs'] = u1_plaq_exact(beta) - therm_('plaqs')
    data['tunneling_events'] = data['charges'][1:] - data['charges'][:-1]


def save_direction_data(run_dir, run_data):
    """Save directionality data to `.txt` file in `run_dir`."""
    forward_arr = run_data.get('forward', None)
    if forward_arr is None:
        io.log(f'`run_data` has no `forward` item. Returning.')
        return

    forward_arr = np.array(forward_arr)
    num_steps = len(forward_arr)
    steps_f = forward_arr.sum()
    steps_b = num_steps - steps_f
    percent_f = steps_f / num_steps
    percent_b = steps_b / num_steps

    direction_file = os.path.join(run_dir, 'direction_results.txt')
    with open(direction_file, 'w') as f:
        f.write(f'forward steps: {steps_f}/{num_steps}, {percent_f}\n')
        f.write(f'backward steps: {steps_b}/{num_steps}, {percent_b}\n')

    return


def save_inference_data(run_dir, run_params, data_dict, data_strs):
    """Save all inference data to `run_dir`."""
    samples_arr = data_dict.get('samples_arr', None)
    run_data = data_dict.get('run_data', None)
    energy_data = data_dict.get('energy_data', None)
    reverse_data = data_dict.get('reverse_data', None)

    if samples_arr is not None:
        samples_arr_file = os.path.join(run_dir, 'samples.pkl')
        io.save_pkl(samples_arr, samples_arr_file)
        #  np.savez_compressed(samples_arr_file, samples_arr)

    max_rdata = {}
    for key, val in reverse_data.items():
        #  max_rdata[key] = np.max(val, axis=1)
        max_rdata[key] = [np.max(val)]
    max_rdata_df = pd.DataFrame(max_rdata)
    out_file = os.path.join(run_dir, 'max_reversibility_results.csv')
    io.log(f'Saving `max` reversibility data to {out_file}.')
    max_rdata_df.to_csv(out_file)

    if 'forward' in run_data:
        save_direction_data(run_dir, run_data)

    io.save_dict(run_params, run_dir, name='run_params')
    run_history_file = os.path.join(run_dir, 'run_history.txt')
    io.log(f'Writing run history to: {run_history_file}...')
    with open(run_history_file, 'w') as f:
        for s in data_strs:
            f.write(f'{s}\n')

    def _pkl_dump(data, pkl_file, name=None):
        if name is not None:
            io.log(f'Saving {name} to {pkl_file}...')
        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)

    run_data_file = os.path.join(run_dir, 'run_data.pkl')
    energy_data_file = os.path.join(run_dir, 'energy_data.pkl')
    reverse_data_file = os.path.join(run_dir, 'reverse_data.pkl')

    _pkl_dump(run_data, run_data_file)
    _pkl_dump(energy_data, energy_data_file)
    _pkl_dump(reverse_data, reverse_data_file)

    observables_dir = os.path.join(run_dir, 'observables')
    io.check_else_make_dir(observables_dir)
    for k, v in run_data.items():
        out_file = os.path.join(observables_dir, f'{k}.pkl')
        io.log(f'Saving {k} to {out_file}...')
        with open(out_file, 'wb') as f:
            pickle.dump(np.array(v), f)
