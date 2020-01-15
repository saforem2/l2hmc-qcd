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

from config import NetWeights

import utils.file_io as io

from lattice.lattice import GaugeLattice, u1_plaq_exact
from plotters.plot_utils import get_run_dirs
from dynamics.dynamics_np import DynamicsRunner
from utils.file_io import timeit
from utils.parse_inference_args_np import parse_args as parse_inference_args

HAS_AUTOGRAD = False
try:
    import autograd.numpy as np
except ImportError:
    HAS_AUTOGRAD = False

    import numpy as np

# pylint: disable=no-member


HEADER = ("{:^13s}" + 7 * "{:^12s}").format(
    "STEP", "t/STEP", "% ACC", "∆x", "∆xf", "∆xb", "exp(∆H)", "∆ø"
)
SEPERATOR = len(HEADER) * '-'

RUN_DATA = {
    'plaqs': [],
    'actions': [],
    'charges': [],
    'dxf': [],
    'dxb': [],
    'dx': [],
    'accept_prob': [],
    'px': [],
    'mask_f': [],
    'mask_b': [],
}

ENERGY_DATA = {
    'potential_init': [],
    'kinetic_init': [],
    'hamiltonian_init': [],
    'potential_proposed': [],
    'kinetic_proposed': [],
    'hamiltonian_proposed': [],
    'potential_out': [],
    'kinetic_out': [],
    'hamiltonian_out': [],
}


def load_pkl(pkl_file):
    """Load from `pkl_file`."""
    with open(pkl_file, 'rb') as f:
        tmp = pickle.load(f)
    return tmp


def _create_lattice(params):
    """Create `GaugeLattice` object from `params`."""
    return GaugeLattice(time_size=params['time_size'],
                        space_size=params['space_size'],
                        dim=params['dim'], link_type='U1',
                        batch_size=params['batch_size'])


def _get_eps(log_dir):
    """Get the step size `eps` by looking for it in `log_dir` ."""
    try:
        run_dirs = get_run_dirs(log_dir)
        run_dir = run_dirs[0]
        rp_file = os.path.join(run_dir, 'run_params.pkl')
        run_params = load_pkl(rp_file)
        eps = run_params['eps']
    except:
        try:
            eps_dict = load_pkl(os.path.join(log_dir, 'eps_np.pkl'))
            eps = eps_dict['eps']
        except FileNotFoundError:
            raise

    return eps


def _update_params(params, eps=None, num_steps=None, batch_size=None):
    """Update params with new values for `eps`, `num_steps`, `batch_size`."""
    if num_steps is not None:
        params['num_steps'] = int(num_steps)
    if batch_size is not None:
        params['batch_size'] = int(batch_size)

    if eps is None:
        eps = _get_eps(params['log_dir'])

    params['eps'] = float(eps)

    return params


def create_dynamics(log_dir, hmc=False, eps=None,
                    num_steps=None, batch_size=None):
    """Create `DynamicsNP` object for running dynamics imperatively."""
    params_file = os.path.join(log_dir, 'parameters.pkl')
    params = load_pkl(params_file)
    params = _update_params(params, eps, num_steps, batch_size)

    weights_file = os.path.join(log_dir, 'weights.pkl')
    with open(weights_file, 'rb') as f:
        weights = pickle.load(f)

    lattice = _create_lattice(params)

    zero_masks = params.get('zero_masks', False)

    dynamics = DynamicsRunner(lattice.calc_actions_np,
                              weights=weights,
                              hmc=hmc,
                              x_dim=lattice.x_dim,
                              eps=params['eps'],
                              num_steps=params['num_steps'],
                              batch_size=params['batch_size'],
                              zero_masks=zero_masks)

    return dynamics, lattice


# pylint:disable=invalid-name
def calc_dx(x1, x2):
    """Calculate the difference between x1, x2 using gauge metric."""
    return np.mean(1. - np.cos(x1 - x2), axis=-1)


def _calc_energies(dynamics, x, v, beta):
    """Calculate the potential/kinetic energies and the Hamiltonian."""
    pe = dynamics.potential_energy(x, beta)
    ke = dynamics.kinetic_energy(v)
    h = dynamics.hamiltonian(x, v, beta)

    return pe, ke, h


def calc_energies(dynamics, x_init, outputs, beta):
    """Calculate initial, proposed, and output energies."""
    pe_init, ke_init, h_init = _calc_energies(dynamics,
                                              x_init,
                                              outputs['v_init'], beta)

    pe_prop, ke_prop, h_prop = _calc_energies(dynamics,
                                              outputs['x_proposed'],
                                              outputs['v_proposed'], beta)

    pe_out, ke_out, h_out = _calc_energies(dynamics,
                                           outputs['x_out'],
                                           outputs['v_out'], beta)

    outputs = {
        'potential_init': pe_init,
        'kinetic_init': ke_init,
        'hamiltonian_init': h_init,
        'potential_proposed': pe_prop,
        'kinetic_proposed': ke_prop,
        'hamiltonian_proposed': h_prop,
        'potential_out': pe_out,
        'kinetic_out': ke_out,
        'hamiltonian_out': h_out,
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


# pylint: disable=too-many-locals
def _inference_setup(log_dir, dynamics, run_params, init=None):
    """Setup for inference run."""
    run_steps = run_params['run_steps']
    beta = run_params['beta']
    net_weights = run_params['net_weights']
    eps = run_params.get('eps', None)
    num_steps = run_params.get('num_steps', None)
    batch_size = dynamics.batch_size

    dynamics, batch_size = _check_param(dynamics, batch_size)
    dynamics, num_steps = _check_param(dynamics, num_steps)
    dynamics, eps = _check_param(dynamics, eps)
    run_params.update({
        'eps': eps,
        'num_steps': num_steps,
        'batch_size': batch_size,
    })

    #  if num_steps is None:
    #      num_steps = dynamics.num_steps
    #  if num_steps != dynamics.num_steps:
    #      dynamics.num_steps = num_steps
    #
    #  if eps is None:
    #      eps = dynamics.eps
    #  if eps != dynamics.eps:
    #      dynamics.eps = eps
    #
    #  run_params['eps'] = eps
    #  run_params['num_steps'] = num_steps

    init = str(init).lower()
    if init == 'rand' or init is None:
        init = 'rand'
        samples = np.random.randn(batch_size, dynamics.x_dim)
    if init == 'zeros':
        samples = np.zeros(batch_size, dynamics.x_dim)
    if init == 'ones':
        samples = np.ones(batch_size, dynamics.x_dim)
    else:
        init = 'rand'
        io.log(f'init: {init}\n')
        samples = np.random.randn(batch_size, dynamics.x_dim)
        #  raise ValueError("InvalidArgument: `init` must be one of "
        #                   "'rand', 'zeros', or 'ones'.")

    nw_str = ''.join((str(int(i)) for i in net_weights))
    beta_str = f'{beta}'.replace('.', '')
    eps_str = f'{eps:.3g}'.replace('.', '')
    run_str = (f'lf{num_steps}_'
               f'steps{run_steps}_'
               f'beta{beta_str}_'
               f'eps{eps_str}_'
               f'nw{nw_str}_'
               f'{init}')

    runs_dir = os.path.join(log_dir, 'runs_np')
    io.check_else_make_dir(runs_dir)
    if os.path.isdir(os.path.join(runs_dir, run_str)):
        io.log(f'Existing run found! Creating new run_dir...')
        timestrs = io.get_timestr()
        run_str += f"_{timestrs['hour_str']}"

    run_params['run_str'] = run_str
    run_dir = os.path.join(runs_dir, run_str)
    io.check_else_make_dir(run_dir)

    return samples, run_params, run_dir


@timeit
def run_inference_np(log_dir, dynamics, lattice, run_params, init=None):
    """Run inference using `dynamics` object."""
    samples, run_params, run_dir = _inference_setup(log_dir, dynamics,
                                                    run_params, init=init)
    samples = np.mod(samples, 2 * np.pi)
    run_steps = run_params['run_steps']
    beta = run_params['beta']
    net_weights = run_params['net_weights']

    run_data = {
        'plaqs': [],
        'actions': [],
        'charges': [],
        'dxf': [],
        'dxb': [],
        'dx': [],
        'accept_prob': [],
        'px': [],
        'mask_f': [],
        'mask_b': [],
    }

    energy_data = {
        'potential_init': [],
        'kinetic_init': [],
        'hamiltonian_init': [],
        'potential_proposed': [],
        'kinetic_proposed': [],
        'hamiltonian_proposed': [],
        'potential_out': [],
        'kinetic_out': [],
        'hamiltonian_out': [],
    }

    data_strs = []
    plaq_exact = u1_plaq_exact(beta)

    start_time = time.time()
    for step in range(run_steps):
        if step % 100 == 0:
            io.log(SEPERATOR)
            io.log(HEADER)
            io.log(SEPERATOR)

        t0 = time.time()
        samples_init = np.mod(samples, 2 * np.pi)
        output = dynamics.apply_transition(samples, beta, net_weights,
                                           model_type='GaugeModel')
        dt = time.time() - t0
        samples = np.mod(output['x_out'], 2 * np.pi)
        obs = lattice.calc_observables_np(samples=samples)
        for k, v in obs.items():
            try:
                run_data[k].append(v)
            except KeyError:
                run_data[k] = [v]

        xf = np.mod(output['xf'], 2*np.pi) * output['mask_f'][:, None]
        xb = np.mod(output['xb'], 2*np.pi) * output['mask_b'][:, None]

        xf0 = samples_init * output['mask_f'][:, None]
        xb0 = samples_init * output['mask_b'][:, None]

        dxf = calc_dx(xf0, xf)
        dxb = calc_dx(xb0, xb)
        dx = calc_dx(samples_init, samples)

        run_data['dx'].append(dx)
        run_data['dxf'].append(dxf)
        run_data['dxb'].append(dxb)
        run_data['accept_prob'].append(output['accept_prob'])

        plaq_diff = plaq_exact - obs['plaqs']

        edata = calc_energies(dynamics, samples_init, output, beta)
        for k, v in edata.items():
            energy_data[k].append(v)

        exp_dH = np.exp(edata['hamiltonian_init'] - edata['hamiltonian_out'])
        px = output['accept_prob']
        run_data['px'].append(px)

        data_str = (f"{step:>6g}/{run_steps:<6g} "
                    f"{dt:^11.4g} "
                    f"{px.mean():^11.4g} "
                    f"{dx.mean():^11.4g} "
                    f"{dxf.mean():^11.4g} "
                    f"{dxb.mean():^11.4g} "
                    f"{exp_dH.mean():^11.4g} "
                    f"{plaq_diff.mean():^11.4g}")

        io.log(data_str)
        data_strs.append(data_str)

    io.log(HEADER)
    io.log(f'Time to complete: {time.time() - start_time:.4g}s')
    io.log(HEADER)

    save_inference_data(run_dir, run_params, run_data, energy_data, data_strs)

    return run_params, run_data, energy_data


def save_inference_data(run_dir, run_params, run_data, energy_data, data_strs):
    """Save all inference data to `run_dir`."""
    io.save_dict(run_params, run_dir, name='run_params')
    run_history_file = os.path.join(run_dir, 'run_history.txt')
    io.log(f'Writing run history to: {run_history_file}...')
    with open(run_history_file, 'w') as f:
        for s in data_strs:
            f.write(f'{s}\n')

    run_data_file = os.path.join(run_dir, 'run_data.pkl')
    io.log(f'Saving run_data to {run_data_file}...')
    with open(run_data_file, 'wb') as f:
        pickle.dump(run_data, f)

    energy_data_file = os.path.join(run_dir, 'energy_data.pkl')
    io.log(f'Saving energy_data to {energy_data_file}...')
    with open(energy_data_file, 'wb') as f:
        pickle.dump(energy_data, f)

    observables_dir = os.path.join(run_dir, 'observables')
    io.check_else_make_dir(observables_dir)
    for k, v in run_data.items():
        out_file = os.path.join(observables_dir, f'{k}.pkl')
        io.log(f'Saving {k} to {out_file}...')
        with open(out_file, 'wb') as f:
            pickle.dump(np.array(v), f)

