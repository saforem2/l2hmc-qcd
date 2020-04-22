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
import autograd.numpy as np

from config import NetWeights, State, Weights
from runners import HSTR
from .run_data import RunData, strf
from utils.file_io import timeit
from lattice.lattice import calc_plaqs_diffs, GaugeLattice
from dynamics.dynamics_np import DynamicsNP, convert_to_angle


#  from plotters.data_utils import therm_arr, bootstrap
#  from plotters.inference_plots import calc_tunneling_rate
# pylint: disable=no-member
# pylint: disable=protected-access
# pylint: disable=inconsistent-return-statements
# pylint: disable=no-else-return
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes


def cos_metric(x, y):
    """Calculate the difference between x1, x2 using gauge metric."""
    return np.mean(1. - np.cos(x - y), axis=-1)


def create_lattice(params):
    """Create `GaugeLattice` object from `params`."""
    return GaugeLattice(time_size=params['time_size'],
                        space_size=params['space_size'],
                        dim=params['dim'], link_type='U1',
                        batch_size=params['batch_size'])


def _get_eps(log_dir):
    """Get the step size `eps` by looking for it in `log_dir` ."""
    try:
        in_file = os.path.join(log_dir, 'eps_np.pkl')
        eps_dict = io.load_pkl(in_file)
        eps = eps_dict['eps']
    except FileNotFoundError:
        run_dirs = io.get_run_dirs(log_dir)
        rp_file = os.path.join(run_dirs[0], 'run_params.pkl')
        if os.path.isfile(rp_file):
            run_params = io.load_pkl(rp_file)
        else:
            rp_file = os.path.join(run_dirs[-1], 'run_params.pkl')
            if os.path.isfile(rp_file):
                run_params = io.load_pkl(rp_file)
            else:
                raise FileNotFoundError('Unable to load run_params.')
        eps = run_params['eps']
    return eps


def _update_params(params, **kwargs):
    """Update params with new values for `eps`, `num_steps`, `batch_size`."""
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
    params = io.load_pkl(os.path.join(log_dir, 'parameters.pkl'))

    # Update params using command line arguments to override defaults
    params = _update_params(params, **kwargs)

    # load saved weights from `.pkl` file:
    xw_file = os.path.join(log_dir, 'xnet_weights.pkl')
    vw_file = os.path.join(log_dir, 'vnet_weights.pkl')
    weights = {
        'xnet': io.load_pkl(xw_file),
        'vnet': io.load_pkl(vw_file),
    }

    # to run with reduced weight matrix keeping first `num_singular_values`
    # singular values from the SVD decomposition: W = U * S * V^{H}
    num_singular_values = kwargs.get('num_singular_values', -1)
    if num_singular_values > 0:
        io.log(f'Keeping the first {num_singular_values} singular values!')
        weights = get_reduced_weights(weights, n=num_singular_values)

    dynamics = DynamicsNP(x_dim=x_dim,
                          params=params,
                          weights=weights,
                          model_type=model_type,
                          potential_fn=potential_fn)

    mask_file = os.path.join(log_dir, 'dynamics_mask.pkl')
    if os.path.isfile(mask_file):
        with open(mask_file, 'rb') as f:
            masks = pickle.load(f)
        dynamics.set_masks(masks)

    #  out_file = os.path.join(log_dir, 'dynamics', 'dynamics_np_dict.txt')
    #  io.write_dict(dynamics.__dict__, out_file)

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


def _get_run_str(run_params, init='rand'):
    """Get `run_str` for naming `run_dir`."""
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
        (strf(i).replace('.', '') for i in net_weights)
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
    eps = run_params.get('eps', None)
    num_steps = run_params.get('num_steps', None)
    batch_size = run_params.get('batch_size', None)
    init = run_params.get('init', 'rand')

    dynamics, batch_size = _check_param(dynamics, batch_size)
    dynamics, num_steps = _check_param(dynamics, num_steps)
    dynamics, eps = _check_param(dynamics, eps)
    run_params.update({
        'eps': eps,
        'num_steps': num_steps,
        'batch_size': batch_size,
    })

    shape = (batch_size, dynamics.x_dim)
    init = str(init).lower()
    if init in ['rand', 'normal'] or init is None:
        init = 'rand'
        samples = np.random.uniform(-np.pi, np.pi, size=shape)
    if init == 'uniform':
        samples = np.random.uniform(-np.pi, np.pi, size=shape)
    if init == 'zeros':
        samples = np.zeros((shape))
    if init == 'ones':
        samples = np.ones((shape))
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
    xprop1, vprop1, _, _ = dynamics.transition_kernel(*state,
                                                      net_weights,
                                                      forward=ff)

    state_prop1 = State(x=convert_to_angle(xprop1), v=vprop1, beta=state.beta)

    xprop2, vprop2, _, _ = dynamics.transition_kernel(*state_prop1,
                                                      net_weights,
                                                      forward=(not ff))

    state_prop2 = State(x=convert_to_angle(xprop2), v=vprop2, beta=state.beta)

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


def plaq_loss(output, lattice, beta, eps=1e-4,
              plaq_weight=0.1, charge_weight=0.1):
    """Calculate the plaquette and charge losses."""
    def mixed_loss(weight, val):
        return weight / val - val / weight

    ps_out = lattice.calc_plaq_sums_np(output['x_out'])
    ps_init = lattice.calc_plaq_sums_np(output['x_init'])
    ps_prop = lattice.calc_plaq_sums_np(output['x_proposed'])

    plaqs_init = np.sum(np.cos(ps_init), axis=(1, 2)) / lattice.num_plaqs
    plaqs_prop = np.sum(np.cos(ps_prop), axis=(1, 2)) / lattice.num_plaqs
    plaqs_out = np.sum(np.cos(ps_out), axis=(1, 2)) / lattice.num_plaqs
    dplaqs = 2. * (1. - np.cos(ps_prop - ps_init))

    charges_init = np.sum(np.sin(ps_init), axis=(1, 2)) / (2 * np.pi)
    charges_prop = np.sum(np.sin(ps_prop), axis=(1, 2)) / (2 * np.pi)
    charges_out = np.sum(np.sin(ps_prop), axis=(1, 2)) / (2 * np.pi)

    ploss_ = output['accept_prob'] * np.sum(dplaqs, axis=(1, 2)) + eps
    qloss_ = output['accept_prob'] * (charges_prop - charges_init) ** 2 + eps

    ploss = mixed_loss(plaq_weight, ploss_)
    qloss = mixed_loss(charge_weight, qloss_)

    outputs = {
        'plaq_loss': ploss,
        'charge_loss': qloss,
        'charges': charges_out,
        'dplaqs': ps_out - ps_init,
        #  'dcharges': charges_out - charges_init,
        'dcharges': np.around(charges_out) - np.around(charges_init),
        'plaqs_diffs': calc_plaqs_diffs(plaqs_out, beta),
        'plaqs_init': plaqs_init,
        'plaqs_proposed': plaqs_prop,
        'plaqs_out': plaqs_out,
        'charges_init': charges_init,
        'charges_out': charges_out,
        'charges_prop': charges_prop,
    }

    return outputs


def _calc_lattice_observables(output, lattice, beta, eps=1e-4, pw=0.1, qw=0.1):
    """Calculate lattice observables."""
    observables = plaq_loss(output, lattice, beta, eps, pw, qw)

    output = {
        'accept_prob': output['accept_prob'],
        'xdiff_r': output['xdiff_r'].mean(axis=-1),
        'vdiff_r': output['vdiff_r'].mean(axis=-1),
        'dplaqs': observables['plaqs_diffs'].mean(),
        'dcharges': observables['charges_diffs'].mean(),
        'plaq_loss': observables['plaq_loss'],
        'charge_loss': observables['charge_loss'],
        'plaqs_diffs': observables['plaqs_diffs_out'].mean()
    }

    return output, observables


def inference_step(step, x_init, dynamics, lattice, **run_params):
    """Run a single inference step."""
    if dynamics._model_type == 'GaugeModel':
        x_init = convert_to_angle(x_init)

    beta = run_params.get('beta', None)
    run_steps = run_params.get('run_steps', None)
    net_weights = run_params.get('net_weights', None)
    symplectic_check = run_params.get('symplectic_check', False)

    start_time = time.time()
    output = dynamics.apply_transition(x_init, beta, net_weights)
    time_diff = time.time() - start_time

    if dynamics._model_type == 'GaugeModel':
        output['x_out'] = convert_to_angle(output['x_out'])
        output['x_init'] = convert_to_angle(output['x_init'])
        output['x_proposed'] = convert_to_angle(output['x_proposed'])

    observables = plaq_loss(output, lattice, beta, eps=1e-4,
                            plaq_weight=0.1, charge_weight=0.1)
    edata = calc_energies(dynamics, x_init, output, beta)

    data_str = (f"{step:>4g}/{run_steps:<5g} "
                f"{time_diff:^10.4g} "
                f"{output['accept_prob'].mean():^10.4g} "
                f"{output['xdiff_r'].mean():^10.4g} "
                f"{output['vdiff_r'].mean():^10.4g} "
                f"{output['sumlogdet_out'].mean():^10.4g} "
                f"{edata['exp_energy_diff'].mean():^10.4g} "
                f"{observables['plaq_loss'].mean():^10.4g} "
                f"{observables['charge_loss'].mean():^10.4g} "
                f"{observables['dplaqs'].mean():^10.4g} "
                f"{np.sum(observables['dcharges'] ** 2):^10.4g} "
                f"{observables['plaqs_diffs'].mean():^10.4g} ")

    outputs = {
        'data_str': data_str,
        'dynamics_output': output,
        'observables': observables,
        'energy_data': edata,
    }

    if symplectic_check:
        state1 = State(x=x_init, v=output['v_init'], beta=beta)
        volume_diffs = dynamics.volume_transformation(state1,
                                                      net_weights,
                                                      output['forward'])
        outputs['volume_diffs'] = volume_diffs

    return outputs


def update_data(run_data, energy_data, volume_diffs, outputs):
    """Update data using `outputs`."""
    for key, val in outputs['observables'].items():
        try:
            run_data[key].append(val)
        except KeyError:
            run_data[key] = [val]

    run_data['sumlogdet_out'].append(
        outputs['dynamics_output']['sumlogdet_out']
    )
    run_data['sumlogdet_proposed'].append(
        outputs['dynamics_output']['sumlogdet_proposed']
    )
    run_data['rand_num'].append(outputs['dynamics_output']['rand_num'])
    run_data['forward'].append(outputs['dynamics_output']['forward'])
    run_data['mask_a'].append(outputs['dynamics_output']['mask_a'])

    for key, val in outputs['energy_data'].items():
        try:
            energy_data[key].append(val)
        except KeyError:
            energy_data[key] = [val]

    #  if volume_diffs is not None:
    if 'volume_diffs' in outputs:
        for key, val in outputs['volume_diffs'].items():
            try:
                volume_diffs[key].append(val)
            except KeyError:
                volume_diffs[key] = [val]

    return run_data, energy_data, volume_diffs


def _get_reverse_data(run_dir):
    """Load reverse data from `reversibility_results.csv` in `run_dir`."""
    rdata_file = os.path.join(run_dir, 'reversibility_results.csv')
    if os.path.isfile(rdata_file):
        return pd.read_csv(rdata_file)

    return None


def map_points(dynamics, run_params):
    """Map how points are transformed by the dynamics sampler."""
    x_init = np.arange(0, 2 * np.pi, 0.01)
    v_init = np.arange(0, 2 * np.pi, 0.01)
    beta = run_params['beta']
    net_weights = run_params['net_weights']
    samples_dict = {
        'x_init': [],
        'x_proposed': [],
        'x_out': [],
        'v_init': [],
        'v_proposed': [],
        'v_out': [],
    }
    for x0, v0 in zip(x_init, v_init):
        forward = (np.random.uniform() < 0.5)
        state0 = State(x=x0, v=v0, beta=beta)
        samples_dict['x_init'].append(x0)
        samples_dict['v_init'].append(v0)

        x_prop, v_prop, px, _ = dynamics.transition_kernel(*state0,
                                                           net_weights,
                                                           forward=forward)
        x_prop = convert_to_angle(x_prop)

        samples_dict['x_proposed'].append(x_prop)
        samples_dict['v_proposed'].append(v_prop)

        mask_a, mask_r, _ = dynamics._get_accept_masks(px)
        x_out = x_prop * mask_a[:, None] + x0 * mask_r[:, None]
        v_out = v_prop * mask_a[:, None] + v0 * mask_r[:, None]

        samples_dict['x_out'].append(x_out)
        samples_dict['v_out'].append(v_out)

    out_dir = os.path.join(run_params['run_dir'], 'mapped_samples')
    io.check_else_make_dir(out_dir)
    for key, val in samples_dict.items():
        out_file = os.path.join(out_dir, f'{key}.pkl')
        io.log(f'Saving {key} to {out_file}.')
        io.save_pkl(np.array(val), out_file)

    return samples_dict


@timeit
def run_inference_np(log_dir, dynamics, lattice, run_params):
    """Run inference imperatively w/ numpy using `dynamics` object.
    Args:
        log_dir (str): Path to `log_dir` containing trained model on which to
            run inference.
        dynamics (dynamicsNP object): Dynamics engine for running the sampler.
        lattice (GaugeLattice object): Lattice object on which the model is
            defined.
        run_params (dict): Dictionary of parameters to use for inference run.
    """
    samples, run_params, run_dir = _inference_setup(log_dir,
                                                    dynamics,
                                                    run_params)
    run_params['direction'] = dynamics.direction

    # Object for holding data generated during inference run
    run_data = RunData(run_params)

    if run_params.get('mix_samplers', False):
        switch_steps = 2000
        run_steps_alt = 500
        net_weights = run_params['net_weights']
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
    samples = convert_to_angle(samples)

    for step in range(run_params['run_steps']):
        samples_init = convert_to_angle(samples)
        outputs = inference_step(step, samples_init,
                                 dynamics, lattice, **run_params)
        samples = convert_to_angle(outputs['dynamics_output']['x_out'])
        run_data.update(step, samples, outputs)

        if step % 100 == 0:
            io.log(HSTR)

    return run_data
