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

import config  # pylint: disable=unused-import
from config import NetWeights, Weights, State

import utils.file_io as io

from lattice.lattice import GaugeLattice, u1_plaq_exact
from plotters.plot_utils import get_run_dirs
from dynamics.dynamics_np import DynamicsNP
from utils.file_io import timeit
from utils.parse_inference_args_np import parse_args as parse_inference_args

HAS_AUTOGRAD = False
try:
    import autograd.numpy as np
except ImportError:
    HAS_AUTOGRAD = False

    import numpy as np

# pylint: disable=no-member


HEADER = ("{:^13s}" + 8 * "{:^12s}").format(
    "STEP", "t/STEP", "% ACC", "∆x", "∆xf", "∆xb", "exp(∆H)", "sumlogdet", "∆ø"
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
    #  'px': [],
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


def sum_squared_diff(x1, x2):
    """Calculate the Euclidean distance between `x1` and `x2`."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def cos_metric(x1, x2):
    """Calculate the difference between x1, x2 using gauge metric."""
    return np.mean(1. - np.cos(x1 - x2), axis=-1)


def _create_lattice(params):
    """Create `GaugeLattice` object from `params`."""
    return GaugeLattice(time_size=params['time_size'],
                        space_size=params['space_size'],
                        dim=params['dim'], link_type='U1',
                        batch_size=params['batch_size'])

# pylint: disable=inconsistent-return-statements, no-else-return
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
        run_dirs = get_run_dirs(log_dir)
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

    dynamics = DynamicsNP(lattice.calc_actions_np,
                              weights=weights,
                              hmc=hmc,
                              x_dim=lattice.x_dim,
                              eps=params['eps'],
                              num_steps=params['num_steps'],
                              batch_size=params['batch_size'],
                              zero_masks=zero_masks)

    mask_file = os.path.join(log_dir, 'dynamics_mask.pkl')
    if os.path.isfile(mask_file):
        with open(mask_file, 'rb') as f:
            masks = pickle.load(f)

        dynamics.set_masks(masks)

    return dynamics, lattice


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


# pylint: disable=too-many-locals
def _inference_setup(log_dir, dynamics, run_params, init='rand', skip=True):
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

    nw_str = ''.join((_strf(i).replace('.', '') for i in net_weights))
    beta_str = f'{beta}'.replace('.', '')
    eps_str = f'{eps:.3g}'.replace('.', '')
    run_str = (f'lf{num_steps}_'
               f'bs{batch_size}_'
               f'steps{run_steps}_'
               f'beta{beta_str}_'
               f'eps{eps_str}_'
               f'nw{nw_str}_'
               f'{init}')

    runs_dir = os.path.join(log_dir, 'runs_np')
    io.check_else_make_dir(runs_dir)
    existing_flag = False
    if os.path.isdir(os.path.join(runs_dir, run_str)):
        rd = os.path.join(runs_dir, run_str)
        rp_file = os.path.join(rd, 'run_params.pkl')
        rd_file = os.path.join(rd, 'run_data.pkl')
        ed_file = os.path.join(rd, 'energy_data.pkl')
        rp_exists = os.path.isfile(rp_file)
        rd_exists = os.path.isfile(rd_file)
        ed_exists = os.path.isfile(ed_file)
        if rp_exists and rd_exists and ed_exists:
            existing_flag = True

        if not skip:
            io.log(f'Existing run found! Creating new run_dir...')
            timestrs = io.get_timestr()
            run_str += f"_{timestrs['hour_str']}"

    run_params['run_str'] = run_str
    run_dir = os.path.join(runs_dir, run_str)
    io.check_else_make_dir(run_dir)

    return samples, run_params, run_dir, existing_flag


def _run_fb_np(dynamics, state, net_weights):
    """Check reversibility of dynamics by running backward(forward(state).
    
    Args:
        dynamics (`DynamicsNP` object): Dynamics on which to run.
        state (`State`, namedtuple): Initial state (x, v, beta).
        net_weights (`NetWeights`, namedtuple): NetWeights multiplicative
            scaling factor.

    Returns:
        state_fb (`State` object): Resultant state (xfb, vfb, beta).
    """
    xb, vb, _, _ = dynamics.transition_kernel(*state,
                                              net_weights,
                                              forward=False)
    state_b = State(x=xb, v=vb, beta=state.beta)
    xfb, vfb, _, _ = dynamics.transition_kernel(*state_b,
                                                net_weights,
                                                forward=True)
    state_fb = State(x=xfb, v=vfb, beta=state.beta)

    return state_fb


def _run_bf_np(dynamics, state, net_weights):
    """Check reversibility of dynamics by running forward(backward(state)).
    Args:
        dynamics (`DynamicsNP` object): Dynamics on which to run.
        state (`State`, namedtuple): Initial state (x, v, beta).
        net_weights (`NetWeights`, namedtuple): NetWeights multiplicative
            scaling factor.

    Returns:
        state_bf (`State` object): Resultant state (xbf, vbf, beta).
    """
    xf, vf, _, _ = dynamics.transition_kernel(*state,
                                              net_weights,
                                              forward=True)
    state_f = State(x=xf, v=vf, beta=state.beta)
    xbf, vbf, _, _ = dynamics.transition_kernel(*state_f,
                                                net_weights,
                                                forward=False)
    state_bf = State(x=xbf, v=vbf, beta=state.beta)

    return state_bf

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
    xprop1, vprop1, _, _ = dynamics.transition_kernel(*state,
                                                      net_weights,
                                                      forward=forward_first)
    state_prop1 = State(x=xprop1, v=vprop1, beta=state.beta)
    xprop2, vprop2, _, _ = dynamics.transition_kernel(*state_prop1,
                                                      net_weights,
                                                      forward=(not
                                                               forward_first))
    state_prop2 = State(x=xprop2, v=vprop2, beta=state.beta)

    return state_prop2


def check_reversibility_np(dynamics,
                           state,
                           net_weights,
                           step=None,
                           out_file=None):
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
    #  state_fb = _run_fb_np(dynamics, state, net_weights)
    #  state_bf = _run_bf_np(dynamics, state, net_weights)

    xdiff_fb = sum_squared_diff(state.x, state_fb.x)
    vdiff_fb = sum_squared_diff(state.v, state_fb.v)
    diff_fb = (xdiff_fb, vdiff_fb)

    xdiff_bf = sum_squared_diff(state.x, state_bf.x)
    vdiff_bf = sum_squared_diff(state.v, state_bf.v)
    diff_bf = (xdiff_bf, vdiff_bf)

    batch_size = state.x.shape[0]
    if out_file is not None:
        if step is not None:
            io.log_and_write(f'step: {step}', out_file)
        avg_str_fb = f' < dxfb, dvfb > : ({xdiff_fb:.5g}, {vdiff_fb:.5g})'
        avg_str_bf = f' < dxbf, dvbf > : ({xdiff_bf:.5g}, {vdiff_bf:.5g})'
        io.log_and_write(avg_str_fb, out_file)
        io.log_and_write(avg_str_bf, out_file)
        out_str = ''
        formatter = {'float_kind': lambda x: f'{x:.3g}'}

        if batch_size <= 1:
            dxfb = state_fb.x - state.x
            dxfb_str = np.array2string(dxfb, max_line_width=80,
                                       precision=4, separator=', ',
                                       prefix='      ', formatter=formatter)
            dvfb = state_fb.v - state.v
            dvfb_str = np.array2string(dvfb, max_line_width=80,
                                       precision=4, separator=', ',
                                       prefix='      ', formatter=formatter)

            dxbf = state_bf.x - state.x
            dxbf_str = np.array2string(dxbf, max_line_width=80,
                                       precision=4, separator=', ',
                                       prefix='      ')
            dvbf = state_bf.v - state.v
            dvbf_str = np.array2string(dvbf, max_line_width=80,
                                       precision=4, separator=', ',
                                       prefix='      ', formatter=formatter)

            out_str += (f' * dxfb:\n  {dxfb_str}\n'
                        f' * dvfb:\n  {dvfb_str}\n'
                        f' * dxbf:\n  {dxbf_str}\n'
                        f' * dvbf:\n  {dvbf_str}\n')
            io.log_and_write(out_str, out_file)
            io.log_and_write(80 * '-', out_file)
            return diff_fb, diff_bf

        for i in range(batch_size):
            dxfb = state_fb.x[i] - state.x[i]
            formatter = {'float_kind': lambda x: f'{x:.3g}'}
            dxfb_str = np.array2string(dxfb, max_line_width=80,
                                       precision=4, separator=', ',
                                       prefix='      ', formatter=formatter)
            dvfb = state_fb.v[i] - state.v[i]
            dvfb_str = np.array2string(dvfb, max_line_width=80,
                                       precision=4, separator=', ',
                                       prefix='      ', formatter=formatter)

            dxbf = state_bf.x[i] - state.x[i]
            dxbf_str = np.array2string(dxbf, max_line_width=80,
                                       precision=4, separator=', ',
                                       prefix='      ')
            dvbf = state_bf.v[i] - state.v[i]
            dvbf_str = np.array2string(dvbf, max_line_width=80,
                                       precision=4, separator=', ',
                                       prefix='      ', formatter=formatter)

            out_str += (f' - chain idx: {i}:\n'
                        f'   * dxfb:\n {dxfb_str}\n'
                        f'   * dvfb:\n {dvfb_str}\n'
                        f'   * dxbf:\n {dxbf_str}\n'
                        f'   * dvfb:\n {dvbf_str}\n')
        io.log_and_write(out_str, out_file)
        io.log_and_write(80 * '-', out_file)

    return diff_fb, diff_bf


@timeit
def run_inference_np(log_dir, dynamics, lattice, run_params, **kwargs):
    """Run inference using `dynamics` object."""
    init = kwargs.get('init', 'rand')
    skip = kwargs.get('skip', True)
    reverse_steps = kwargs.get('reverse_steps', 1000)
    samples, run_params, run_dir, existing_flag = _inference_setup(log_dir,
                                                                   dynamics,
                                                                   run_params,
                                                                   init=init,
                                                                   skip=skip)
    reverse_file = os.path.join(run_dir, 'reversibility_results.txt')

    if skip and existing_flag:
        io.log(f'Existing run found! Loading data...')
        run_params = load_pkl(os.path.join(run_dir, 'run_params.pkl'))
        run_data = load_pkl(os.path.join(run_dir, 'run_data.pkl'))
        energy_data = load_pkl(os.path.join(run_dir, 'energy_data.pkl'))

    else:
        samples = np.mod(samples, 2 * np.pi)
        beta = run_params['beta']
        run_steps = run_params['run_steps']
        net_weights = run_params['net_weights']

        data_keys = ['plaqs', 'actions', 'charges', 'dxf', 'dxb',
                     'dx', 'accept_prob', 'mask_f', 'mask_b']

        reverse_keys = ['xdiff_fb', 'xdiff_bf', 'vdiff_fb', 'vdiff_bf']

        energy_keys = ['potential_init', 'potential_proposed', 'potential_out',
                       'kinetic_init', 'kinetic_proposed', 'kinetic_out',
                       'hamiltonian_init', 'hamiltonian_proposed',
                       'hamiltonian_out', 'exp_energy_diff']

        run_data = {key: [] for key in data_keys}
        energy_data = {key: [] for key in energy_keys}
        reverse_data = {key: [] for key in reverse_keys}

        data_strs = []
        plaq_exact = u1_plaq_exact(beta)

        start_time = time.time()
        for step in range(run_steps):
            if step % 100 == 0:
                io.log(SEPERATOR + '\n' + HEADER + '\n' + SEPERATOR)
            if step % reverse_steps == 0:
                v_rand = np.random.randn(*samples.shape)
                state = State(x=samples, v=v_rand, beta=beta)
                diff_fb, diff_bf = check_reversibility_np(dynamics, state,
                                                          net_weights, step,
                                                          reverse_file)
                reverse_data['xdiff_fb'].append(diff_fb[0])
                reverse_data['xdiff_bf'].append(diff_bf[0])
                reverse_data['vdiff_fb'].append(diff_fb[1])
                reverse_data['vdiff_bf'].append(diff_bf[1])

            start_time = time.time()
            samples_init = np.mod(samples, 2 * np.pi)
            output = dynamics.apply_transition(samples_init,
                                               beta, net_weights,
                                               model_type='GaugeModel')
            sld = output['sumlogdet_out']
            time_diff = time.time() - start_time
            samples = np.mod(output['x_out'], 2 * np.pi)
            obs = lattice.calc_observables_np(samples=samples)
            for k, v in obs.items():
                try:
                    run_data[k].append(v)
                except KeyError:
                    run_data[k] = [v]

            xf0 = samples_init * output['mask_f'][:, None]
            xb0 = samples_init * output['mask_b'][:, None]

            xf1 = np.mod(output['xf'], 2*np.pi) * output['mask_f'][:, None]
            xb1 = np.mod(output['xb'], 2*np.pi) * output['mask_b'][:, None]

            dxf = cos_metric(xf0, xf1)
            dxb = cos_metric(xb0, xb1)
            dx_out = cos_metric(samples_init, samples)

            run_data['dx'].append(dx_out)
            run_data['dxf'].append(dxf)
            run_data['dxb'].append(dxb)
            run_data['accept_prob'].append(output['accept_prob'])
            plaq_diff = plaq_exact - obs['plaqs']

            edata = calc_energies(dynamics, samples_init, output, beta)
            for k, v in edata.items():
                energy_data[k].append(v)

            data_str = (f"{step:>6g}/{run_steps:<6g} "
                        f"{time_diff:^11.4g} "
                        f"{output['accept_prob'].mean():^11.4g} "
                        f"{dx_out.mean():^11.4g} "
                        f"{dxf.mean():^11.4g} "
                        f"{dxb.mean():^11.4g} "
                        f"{edata['exp_energy_diff'].mean():^11.4g} "
                        f"{sld.mean():^11.4g} "
                        f"{plaq_diff.mean():^11.4g}")

            io.log(data_str)
            data_strs.append(data_str)

        data_dict = {
            'run_data': run_data,
            'energy_data': energy_data,
            'reverse_data': reverse_data,
        }
        save_inference_data(run_dir, run_params, data_dict, data_strs)
    outputs = {
        'run_params': run_params,
        'data': data_dict,
        #  'run_data': run_data,
        #  'energy_data': energy_data,
        'reverse_data': reverse_data,
        'existing_flag': existing_flag,
    }

    return outputs


def save_inference_data(run_dir, run_params, data_dict, data_strs):
    """Save all inference data to `run_dir`."""
    run_data = data_dict.get('run_data', None)
    energy_data = data_dict.get('energy_data', None)
    reverse_data = data_dict.get('reverse_data', None)

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

    #  io.log(f'Saving run_data to {run_data_file}...')
    #  with open(run_data_file, 'wb') as f:
    #      pickle.dump(run_data, f)
    #
    #  energy_data_file = os.path.join(run_dir, 'energy_data.pkl')
    #  io.log(f'Saving energy_data to {energy_data_file}...')
    #  with open(energy_data_file, 'wb') as f:
    #      pickle.dump(energy_data, f)
    #
    #  reverse_data_file = os.path.join(run_dir, 'reverse_data.pkl')
    #  io.log(f'Saving reversibility data to {reverse_data_file}...')

    observables_dir = os.path.join(run_dir, 'observables')
    io.check_else_make_dir(observables_dir)
    for k, v in run_data.items():
        out_file = os.path.join(observables_dir, f'{k}.pkl')
        io.log(f'Saving {k} to {out_file}...')
        with open(out_file, 'wb') as f:
            pickle.dump(np.array(v), f)
