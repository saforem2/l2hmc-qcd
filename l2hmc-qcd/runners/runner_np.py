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
from lattice.lattice import GaugeLattice, u1_plaq_exact
from plotters.plot_utils import get_run_dirs
from dynamics.dynamics_np import DynamicsNP
from utils.file_io import timeit
from utils.parse_inference_args_np import parse_args as parse_inference_args

try:
    import deepdish as dd
    HAS_DEEPDISH = True
except ImportError:
    HAS_DEEPDISH = False


try:
    import autograd.numpy as np
except ImportError:
    HAS_AUTOGRAD = False
    import numpy as np

# pylint: disable=no-member
# pylint: disable=inconsistent-return-statements
# pylint: disable=no-else-return
# pylint: disable=too-many-locals
# pylint:disable=too-many-arguments


NET_WEIGHTS_HMC = NetWeights(0, 0, 0, 0, 0, 0)
NET_WEIGHTS_L2HMC = NetWeights(1, 1, 1, 1, 1, 1)

HEADER = ("{:^13s}" + 6 * "{:^12s}").format(
    "STEP", "t/STEP", "% ACC", "𝞭x_out", "𝞭x_prop",
    "exp(𝞭H)", "sumlogdet", "𝞭𝜙"
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


class RunnerNP:
    def __init__(self,
                 dynamics: DynamicsNP,
                 params: dict,
                 run_params: dict,
                 observables_op: callable=None) -> None:
        """Initialization method.

        Args:
            dynamics (dynamicsNP object): Dynamics object.
        """
        self.dynamics = dynamics
        self.params = params
        self._observables_op = observables_op
        self.log_dir = params.get('log_dir', None)
        self._setup(run_params)
        self._setup_directories()
        self.run_data, self.energy_data, self.reverse_data = self._init_data()

        self._model_type = params.get('model_type', 'GaugeModel')
        metric = 'cos_diff' if self._model_type == 'GaugeModel' else 'l2'
        self.metric_fn = self._create_metric_fn(metric)

    def _create_metric_fn(self, metric):
        if metric == 'l1':
            def metric_fn(x1, x2):
                return np.abs(x1 - x2)
        elif metric == 'l2':
            def metric_fn(x1, x2):
                return (x1 - x2) ** 2
        elif metric in ['cos', 'cos_diff']:
            def metric_fn(x1, x2):
                return 1. - np.cos(x1 - x2)
        else:
            raise ValueError(f'metric: {metric}. Expected one of:\n'
                             '\t`l1`, `l2`, or `cos_diff`.')

        return metric_fn

    def _init_data(self):
        """Initialize dictionaries to store inference data."""
        rkeys = ['dx_bf', 'dv_bf',
                 'dx_fb', 'dv_fb']
        ekeys = [
            'potential_init', 'potential_proposed', 'potential_out',
            'kinetic_init', 'kinetic_proposed', 'kinetic_out',
            'hamiltonian_init', 'hamiltonian_proposed', 'hamiltonian_out',
            'exp_energy_diff'
        ]

        dkeys = ['dx_proposed', 'dx_out',
                 'sumlogdet_prop', 'sumlogdet_out',
                 'accept_prob', 'rand_num', 'mask_a', 'forward']

        if self._model_type == 'GaugeModel':
            dkeys += ['plaqs', 'actions', 'charges']

        run_data = {key: [] for key in dkeys}
        energy_data = {key: [] for key in ekeys}
        reverse_data = {key: [] for key in rkeys}

        return run_data, energy_data, reverse_data

    def _setup(self, run_params):
        """Setup run_params and dynamics."""
        self._eps = run_params.get('eps', None)
        self._beta = run_params.get('beta', 1.)
        self._init = run_params.get('init', 'rand')
        self._num_steps = run_params.get('num_steps', 1)
        self._batch_size = run_params.get('batch_size', 1)
        self._print_steps = run_params.get('print_steps', 1)
        self._run_steps = run_params.get('run_steps', 10000)
        self._mix_samplers = run_params.get('mix_samplers', False)
        self._reverse_steps = run_params.get('reverse_steps', 1000)
        self._model_type = run_params.get('model_type', 'GaugeModel')
        self._net_weights = run_params.get('net_weights', NET_WEIGHTS_L2HMC)
        self._update_attrs({
            'eps': self._eps,
            'num_steps': self._num_steps,
            'batch_size': self._batch_size,
        })

        run_str = self._get_run_str()
        self._run_dir = os.path.join(self.log_dir, 'runs_np', run_str)
        io.check_else_make_dir(self._run_dir)
        self._reverse_file = os.path.join(self._run_dir, 'reversibility.csv')

        #  self._header = ("{:^13s}" + 6 * "{:^12s}").format("STEP", "t/STEP",
        self._header = (7 * "{:^12s}").format('step', 'dt', '% accept',
                                              '𝞭x (out)', '𝞭x (prop)',
                                              'exp(𝞭H)', 'sumlogdet')
        self._separator = len(self._header) * '-'
        self._divider = (self._separator
                         + '\n' + self._header
                         + '\n' + self._separator)

        if self._model_type == 'GaugeModel':
            self._header += "{:^12s}".format("𝞭𝜙")

        if self._mix_samplers:
            self._switch_steps = run_params.get('switch_steps', 2000)
            self._run_steps_alt = run_params.get('run_steps_alt', 500)

            if self._net_weights == NET_WEIGHTS_L2HMC:
                self._net_weights_alt = NET_WEIGHTS_HMC
            if self._net_weights == NET_WEIGHTS_HMC:
                self._net_weights_alt = NET_WEIGHTS_L2HMC
            if self._net_weights == NetWeights(0., 1., 0., 0., 0., 0.):
                self._net_weights_alt = NET_WEIGHTS_HMC

            run_params.update({
                'net_weights_alt': self._net_weights_alt,
            })

        run_params.update({
            'eps': self._eps,
            'run_str': run_str,
            'run_dir': self._run_dir,
            'num_steps': self._num_steps,
            'batch_size': self._batch_size,
            'model_type': self._model_type,
            'print_steps': self._print_steps,
            'mix_samplers': self._mix_samplers,
            'reverse_steps': self._reverse_steps,
            'direction': self.dynamics.direction,
        })

        self.run_params = run_params

    def _update_attrs(self, params=None):
        """Check param against its value as an attribute of `dynamics`.

        Explicitly:

            1. If no param is passed, use the current value of
              `dynamics.param`.

            2. If param is passed but is not equal to `dynamics.param`, set
              `dynamics.param = param`.
        """
        for key, param in params.items():
            attr = getattr(self.dynamics, key, None)
            if param is None:
                param = attr
            if param != attr:
                setattr(self.dynamics, key, param)

    def _get_run_str(self):
        nw_str = ''.join(
            (_strf(i).replace('.', '') for i in self._net_weights)
        )
        beta_str = f'{self._beta}'.replace('.', '')
        eps_str = f'{self._eps:.3g}'.replace('.', '')
        run_str = (f'lf{self._num_steps}_'
                   f'bs{self._batch_size}_'
                   f'steps{self._run_steps}_'
                   f'beta{beta_str}_'
                   f'eps{eps_str}_'
                   f'nw{nw_str}_'
                   f'_{self._init}')

        if self._mix_samplers:
            run_str += f'_mix_samplers'

        time_strs = io.get_timestr()
        timestr = time_strs['timestr']
        run_str += f'_{timestr}'

        return run_str

    def _init_samples(self, init='rand'):
        """Initialize samples array."""
        init = str(init).lower()
        if init == 'rand' or init is None:
            init = 'rand'
            samples = np.random.randn(self.batch_size, self.dynamics.x_dim)
        if init == 'zeros':
            samples = np.zeros((self.batch_size, self.dynamics.x_dim))
        if init == 'ones':
            samples = np.ones((self.batch_size, self.dynamics.x_dim))
        else:
            init = 'rand'
            io.log(f'init: {init}\n')
            samples = np.random.randn(self.batch_size, self.dynamics.x_dim)

        samples = np.mod(samples, 2 * np.pi)

        return samples

    def _calc_energies(self, state, sumlogdet=0.):
        """Calculate the energies of `state`."""
        potential_energy = self.dynamics.potential_energy(state.x, state.beta)
        kinetic_energy = self.dynamics.kinetic_energy(state.v)
        hamiltonian = potential_energy + kinetic_energy

        return Energy(potential=potential_energy,
                      kinetic=kinetic_energy,
                      hamiltonian=hamiltonian)

    def _update_data(self, outputs):
        """Update data structures using new values in `outputs`.

        Args:
            outputs (dict): Dictionary of outputs from running a single
                inference step.
        """
        if 'energy_data' in outputs:
            self.energy_data = _update_dict(self.energy_data,
                                            outputs['energy_data'],
                                            force=True)
        if 'observables' in outputs:
            self.run_data = _update_dict(self.run_data,
                                         outputs['observables'],
                                         force=True)

        if 'dynamics_output' in outputs:
            self.run_data = _update_dict(self.run_data,
                                         outputs['dynamics_output'],
                                         force=False)

    def reverse_dynamics(self, state, f_first=True):
        """Check reversibility of dynamics by running either:
            1. backward(forward(state)) if `f_first=True`
            2. forward(backward(state)) if `f_first=False`

        Args:
            state (State object): Initial state (x, v, beta).
            f_first (bool): Whether the forward direction should be ran first.

        Returns:
            state_new (State object): New, resultant state.
        """
        x1, v1, _, _ = self.dynamics.transition_kernel(*state,
                                                       self._net_weights,
                                                       forward=f_first)
        state1 = State(x=x1, v=v1, beta=state.beta)

        x2, v2, _, _ = self.dynamics.transition_kernel(*state1,
                                                       self._net_weights,
                                                       forward=(not f_first))
        state2 = State(x=x2, v=v2, beta=state.beta)

        return state2

    def check_reversibility(self, step, state):
        """Run reversibility checker."""
        state_bf = self.reverse_dynamics(state, f_first=True)
        state_fb = self.reverse_dynamics(state, f_first=False)

        dxdv = {
            'dx_bf': np.squeeze((state.x - state_bf.x).flatten()),
            'dv_bf': np.squeeze((state.v - state_bf.v).flatten()),
            'dx_fb': np.squeeze((state.x - state_fb.x).flatten()),
            'dv_fb': np.squeeze((state.v - state_fb.v).flatten()),
        }
        dxdv_df = pd.DataFrame(dxdv)
        dxdv_df.to_csv(self._reverse_file, mode='a',
                       header=(step == 0), index=False)

        for key, val in dxdv.items():
            self.reverse_data[key].extend(val)

        return dxdv

    def inference_step(self, step, x, **run_params):
        """Run a single inference step."""
        beta = run_params.get('beta', self._beta)
        run_steps = run_params.get('run_steps', self._run_steps)
        net_weights = run_params.get('net_weights', self._net_weights)

        if self._model_type == 'GaugeModel':
            x = np.mod(x, 2 * np.pi)
            plaq_exact = u1_plaq_exact(beta)

        start_time = time.time()
        output = self.dynamics.apply_transition(x, beta, net_weights,
                                                self._model_type)
        time_diff = time.time() - start_time
        x_out = output['x_out']

        if self._model_type == 'GaugeModel':
            x_out = np.mod(output['x_out'], 2 * np.pi)
            output['x_out'] = x_out
            if self._observables_op is not None:
                observables = self._observables_op(samples=x_out)
                plaq_diff = plaq_exact - observables['plaqs']

        edata = self.calc_energies(x, output)

        dx_prop = self.metric_fn(output['x_proposed'], x)
        dx_out = self.metric_fn(x_out, x)
        output['dx_out'] = dx_out
        output['dx_proposed'] = dx_prop

        data_str = (f"{step:>6g}/{self._run_steps:<6g} "
                    f"{time_diff:^11.4g} "
                    f"{output['accept_prob'].mean():^11.4g} "
                    f"{dx_out.mean():^11.4g} "
                    f"{dx_prop.mean():^11.4g} "
                    f"{edata['exp_energy_diff'].mean():^11.4g} "
                    f"{output['sumlogdet_out'].mean():^11.4g} ")

        if self._model_type == 'GaugeModel':
            data_str += f"{plaq_diff.mean():^11.4g} "

        outputs = {
            'data_str': data_str,
            'energy_data': edata,
            'dynamics_output': output,
            'observables': observables,
        }

        return outputs

    def run_alt_sampler(self, steps, samples, data_strs):
        """Run inference using the alternative sampler."""
        io.log(self._divider)
        io.log(f"RUNNING INFERENCE WITH:\n "
               f"\t net_weights: {self.run_params['net_weights_alt']}"
               f"\t steps: {steps}\n")

        if self._model_type == 'GaugeModel':
            samples = np.mod(samples, 2 * np.pi)

        for step in range(steps):
            if step % 100 == 0:
                io.log(self._divider)

            outputs = self.inference_step(step, samples,
                                          run_steps=steps,
                                          beta=self._beta,
                                          net_weights=self._net_weights_alt)
            samples = outputs['dynamics_output']['x_out']

            if self._model_type == 'GaugeModel':
                samples = np.mod(samples, 2 * np.pi)
                outputs['dynamics_output']['x_out'] = samples

            self._update_data(outputs)

            if step % self._print_steps == 0:
                data_strs.append(outputs['data_str'])
                io.log(outputs['data_str'])

        io.log(self._divider)
        io.log('\n...Back to original sampler!\n')

        return samples, data_strs

    def run_inference(self):
        """Run inference for `self._run_steps` steps."""
        samples = self._init_samples(init=self._init)
        if self._model_type == 'GaugeModel':
            samples = np.mod(samples, 2 * np.pi)

        io.log(self._header)
        data_strs = []
        for step in range(self._run_steps):
            samples_in = samples
            if self._model_type == 'GaugeModel':
                samples_in = np.mod(samples, 2 * np.pi)
            outputs = self.inference_step(step, samples_in)
            samples = outputs['dynamics_output']['x_out']
            self._update_data(outputs)

            if step % 100 == 0:
                io.log(self._divider)

            if self._mix_samplers and step % self._switch_steps == 0:
                samples, data_strs = self._run_alt_sampler(self._run_steps_alt,
                                                           samples, data_strs)

            if step % self._reverse_steps == 0:
                v_out = outputs['dynamics_output']['v_out']
                state = State(x=samples, v=v_out, beta=self._beta)
                _ = self.check_reversibility(step, state)

            if step % self._print_steps == 0:
                io.log(outputs['data_str'])
                data_strs.append(outputs['data_str'])

        self._save_inference_data(data_strs)
        out_data = {
            'run_data': self.run_data,
            'energy_data': self.energy_data,
            'reverse_data': self.reverse_data,
        }

        return out_data

    def save_direction_data(self):
        """Save directionality data to `.txt` file in `run_dir`."""
        forward_arr = self.run_data.get('forward', None)
        if forward_arr is None:
            io.log(f'`run_data` has no `forward` entry. Returning.')
            return

        forward_arr = np.array(forward_arr)
        num_steps = len(forward_arr)
        steps_f = forward_arr.sum()
        steps_b = num_steps - steps_f
        percent_f = steps_f / num_steps
        percent_b = steps_b / num_steps

        direction_file = os.path.join(self._run_dir, 'direction_results.txt')
        with open(direction_file, 'w') as f:
            f.write(f'forward steps: {steps_f}/{num_steps}, {percent_f}\n')
            f.write(f'backward steps: {steps_b}/{num_steps}, {percent_b}\n')

        return

    def save_inference_data(self, data_strs):
        """Save all inference data to `self._run_dir`."""
        max_rdata = {}
        for key, val in self.reverse_data.items():
            max_rdata[key] = np.max(val, axis=1)

        max_rdata_df = pd.DataFrame(max_rdata)
        out_file = os.path.join(self._run_dir, 'max_reversibility_results.csv')
        io.log(f'Saving `max` reversibility data to: {out_file}.')
        max_rdata_df.to_csv(out_file)

        if 'forward' in self.run_data:
            self.save_direction_data()

        io.save_dict(self.run_params, self._run_dir, name='run_params')
        run_history_file = os.path.join(self._run_dir, 'run_history.txt')
        io.log(f'Writing run history to: {run_history_file}...')
        with open(run_history_file, 'w') as f:
            for s in data_strs:
                f.write(f'{s}\n')

        def _pkl_dump(data, pkl_file, name=None):
            if name is not None:
                io.log(f'Saving {name} to {pkl_file}...')
            with open(pkl_file, 'wb') as f:
                pickle.dump(data, f)

        run_data_file = os.path.join(self._run_dir, 'run_data.pkl')
        energy_data_file = os.path.join(self._run_dir, 'energy_data.pkl')
        reverse_data_file = os.path.join(self._run_dir, 'reverse_data.pkl')

        _pkl_dump(self.run_data, run_data_file)
        _pkl_dump(self.energy_data, energy_data_file)
        _pkl_dump(self.reverse_data, reverse_data_file)

        observables_dir = os.path.join(self._run_dir, 'observables')
        io.check_else_make_dir(observables_dir)
        for k, v in self.run_data.items():
            out_file = os.path.join(observables_dir, f'{k}.pkl')
            io.log(f'Saving {k} to {out_file}...')
            with open(out_file, 'wb') as f:
                pickle.dump(np.array(v), f)












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
    #  except:  # noqa: E722 pylint:disable=bare-except
    #      eps_dict = load_pkl(os.path.join(log_dir, 'eps_np.pkl'))
    #      eps = eps_dict['eps']

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
    #  if num_steps is not None:
    #      params['num_steps'] = int(num_steps)
    #  if batch_size is not None:
    #      params['batch_size'] = int(batch_size)

    eps = kwargs.get('eps', None)
    if eps is None:
        eps = _get_eps(params['log_dir'])

    params['eps'] = float(eps)

    return params


def create_dynamics(log_dir,
                    potential_fn,
                    x_dim,
                    **kwargs):
                    #  hmc=False,
                    #  eps=None,
                    #  num_steps=None,
                    #  batch_size=None,
                    #  model_type=None,
                    #  direction='rand'):
    """Create `DynamicsNP` object for running dynamics imperatively."""
    params = load_pkl(os.path.join(log_dir, 'parameters.pkl'))
    #  params = _update_params(params, eps, num_steps, batch_size)
    for key, val in kwargs.items():
        if val is not None:
            if key in ['num_steps', 'batch_size']:
                params[key] = int(val)
            else:
                params[key] = val

    eps_ = kwargs.get('eps', None)
    eps = _get_eps(log_dir) if eps_ is None else eps_

    #  params = _update_params(params, kwargs)

    with open(os.path.join(log_dir, 'weights.pkl'), 'rb') as f:
        weights = pickle.load(f)

    dynamics = DynamicsNP(potential_fn,
                          x_dim=x_dim,
                          weights=weights,
                          **params)
                          #  hmc=hmc,
                          #  x_dim=x_dim,
                          #  eps=params['eps'],
                          #  num_steps=params['num_steps'],
                          #  batch_size=params['batch_size'],
                          #  #  zero_masks=params.get('zero_masks', False),
                          #  model_type=model_type,
                          #  direction=direction)

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
        'sumlogdet_prop', 'sumlogdet_out',
        'accept_prob', 'rand_num', 'mask_a',
        'forward'
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

    time_strs = io.get_timestr()
    timestr = time_strs['timestr']
    run_str += f'__{timestr}'

    return run_str


def _inference_setup(log_dir, dynamics, run_params, init='rand', skip=True):
    """Setup for inference run."""
    run_steps = run_params['run_steps']
    beta = run_params['beta']
    net_weights = run_params['net_weights']
    eps = run_params.get('eps', None)
    print(f'\n\n eps: {eps}\n dynamics.eps: {dynamics.eps}\n\n')
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

    run_str = _get_run_str(run_params)
    #  nw_str = ''.join((_strf(i).replace('.', '') for i in net_weights))
    #  beta_str = f'{beta}'.replace('.', '')
    #  eps_str = f'{eps:.3g}'.replace('.', '')
    #  run_str = (f'lf{num_steps}_'
    #             f'bs{batch_size}_'
    #             f'steps{run_steps}_'
    #             f'beta{beta_str}_'
    #             f'eps{eps_str}_'
    #             f'nw{nw_str}_'
    #             f'{init}')

    runs_dir = os.path.join(log_dir, 'runs_np')
    io.check_else_make_dir(runs_dir)
    existing_flag = False
    if os.path.isdir(os.path.join(runs_dir, run_str)):
        run_dir = os.path.join(runs_dir, run_str)
        rp_file = os.path.join(run_dir, 'run_params.pkl')
        rd_file = os.path.join(run_dir, 'run_data.pkl')
        ed_file = os.path.join(run_dir, 'energy_data.pkl')
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
    run_params['run_dir'] = run_dir

    return samples, run_params, run_dir, existing_flag


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
    dx_bf = state.x - state_bf.x
    dv_bf = state.v - state_bf.v

    dx_fb = state.x - state_fb.x
    dv_fb = state.v - state_fb.v

    dx_dict = {
        'dx_bf': np.squeeze(dx_bf.flatten()),
        'dx_fb': np.squeeze(dx_fb.flatten()),
        'dv_bf': np.squeeze(dv_bf.flatten()),
        'dv_fb': np.squeeze(dv_fb.flatten()),
    }
    dx_df = pd.DataFrame(dx_dict)
    if out_file is not None:
        header = (step == 0)
        dx_df.to_csv(out_file, mode='a', header=header, index=False)

    return (dx_fb, dv_fb), (dx_bf, dv_bf)


def inference_step(step, x_init, dynamics, lattice, **run_params):
    """Run a single inference step."""
    x_init = np.mod(x_init, 2 * np.pi)
    beta = run_params.get('beta', None)
    run_steps = run_params.get('run_steps', None)
    net_weights = run_params.get('net_weights', None)
    plaq_exact = u1_plaq_exact(beta)

    start_time = time.time()
    output = dynamics.apply_transition(x_init, beta, net_weights,
                                       model_type='GaugeModel')
    time_diff = time.time() - start_time

    x_out = np.mod(output['x_out'], 2 * np.pi)
    output['x_out'] = x_out

    observables = lattice.calc_observables_np(samples=x_out)
    plaq_diff = plaq_exact - observables['plaqs']

    dx_prop = cos_metric(output['x_proposed'], x_init)
    dx_out = cos_metric(x_out, x_init)
    observables['dx_out'] = dx_out
    observables['dx_proposed'] = dx_prop
    observables['accept_prob'] = output['accept_prob']

    edata = calc_energies(dynamics, x_init, output, beta)

    data_str = (f"{step:>6g}/{run_steps:<6g} "
                f"{time_diff:^11.4g} "
                f"{output['accept_prob'].mean():^11.4g} "
                f"{dx_out.mean():^11.4g} "
                f"{dx_prop.mean():^11.4g} "
                #  f"{output['forward']:^11.4g} "
                f"{edata['exp_energy_diff'].mean():^11.4g} "
                f"{output['sumlogdet_out'].mean():^11.4g} "
                f"{plaq_diff.mean():^11.4g}")

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

    run_data['sumlogdet_out'].append(
        outputs['dynamics_output']['sumlogdet_out']
    )
    run_data['sumlogdet_prop'].append(
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
    _run_params['run_stteps'] = steps
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
def run_inference_np(log_dir, dynamics, lattice, run_params, **kwargs):
    """Run inference imperatively w/ numpy using `dynamics` object.

    Args:
        log_dir (str): Path to `log_dir` containing trained model on which to
            run inference.
        dynamics (dynamicsNP object): Dynamics engine for running the sampler.
        lattice (GaugeLattice object): Lattice object on which the model is
            defined.
        run_params (dict): Dictionary of parameters to use for inference run.
    """
    init = kwargs.get('init', 'rand')
    skip = kwargs.get('skip', True)
    mix_samplers = kwargs.get('mix_samplers', False)
    io.log(f'MIX_SAMPLERS: {mix_samplers}\n')
    reverse_steps = kwargs.get('reverse_steps', 1000)
    print_steps = kwargs.get('print_steps', 1)
    samples, run_params, run_dir, existing_flag = _inference_setup(log_dir,
                                                                   dynamics,
                                                                   run_params,
                                                                   init=init,
                                                                   skip=skip)
    run_params['print_steps'] = print_steps
    run_params['mix_samplers'] = mix_samplers
    run_params['reverse_steps'] = reverse_steps
    run_params['direction'] = dynamics.direction

    beta = run_params['beta']
    run_steps = run_params['run_steps']
    net_weights = run_params['net_weights']

    if mix_samplers:
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

    data_strs = []
    for step in range(run_steps):
        samples_init = np.mod(samples, 2 * np.pi)
        outputs = inference_step(step, samples_init,
                                 dynamics, lattice, **run_params)
        samples = outputs['dynamics_output']['x_out']
        run_data, energy_data = update_data(run_data, energy_data, outputs)

        if step % 100 == 0:
            io.log(SEPERATOR + '\n' + HEADER + '\n' + SEPERATOR)

        if mix_samplers and step % switch_steps == 0:
            _data = {
                'run_data': run_data,
                'energy_data': energy_data,
                'data_strs': data_strs,
            }
            samples, data = _run_np(run_steps_alt, nws,
                                    dynamics, lattice,
                                    samples, run_params, _data)

        if step % reverse_steps == 0:
            v_out = outputs['dynamics_output']['v_out']
            state = State(x=samples, v=v_out, beta=beta)
            diff_fb, diff_bf = check_reversibility_np(dynamics, state,
                                                      net_weights, step,
                                                      reverse_file)
            reverse_data['xdiff_fb'].extend(diff_fb[0])
            reverse_data['xdiff_bf'].extend(diff_bf[0])
            reverse_data['vdiff_fb'].extend(diff_fb[1])
            reverse_data['vdiff_bf'].extend(diff_bf[1])

        if step % print_steps == 0:
            io.log(outputs['data_str'])
            data_strs.append(outputs['data_str'])

    data_dict = {
        'run_data': run_data,
        'energy_data': energy_data,
        'reverse_data': reverse_data,
    }
    save_inference_data(run_dir, run_params, data_dict, data_strs)

    outputs = {
        'run_params': run_params,
        'data': data_dict,
        'reverse_data': reverse_data,
        'existing_flag': existing_flag,
    }

    return outputs


def summarize_run(run_data):
    charges = np.array(run_data['charges'])
    accept_prob = np.array(run_data['accept_prob'])
    dx_out = np.array(run_data['dx_out'])


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
    run_data = data_dict.get('run_data', None)
    energy_data = data_dict.get('energy_data', None)
    reverse_data = data_dict.get('reverse_data', None)

    max_rdata = {}
    for key, val in reverse_data.items():
        max_rdata[key] = np.max(val, axis=1)
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
