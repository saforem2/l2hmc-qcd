"""
runner_np.py
Implements `RunnerNP` object responsible for performing tensorflow-independent
inference on a trained model.
Author: Sam Foreman (github: @saforem2)
Date: 01/09/2020
"""
import os
import time
from collections import namedtuple

#  import pandas as pd

import utils.file_io as io
import autograd.numpy as np

from config import NetWeights, State, Weights
from runners import HSTR
from .run_data import RunData, strf
from utils.file_io import timeit
from utils.attr_dict import AttrDict
from lattice.lattice import calc_plaqs_diffs, GaugeLattice
from dynamics.dynamics import MonteCarloStates
from dynamics.dynamics_np import (DynamicsNP, convert_to_angle,
                                  DynamicsParamsNP, DynamicsConfigNP)


# pylint: disable=no-member
# pylint: disable=protected-access
# pylint: disable=inconsistent-return-statements
# pylint: disable=no-else-return
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
NET_WEIGHTS_HMC = NetWeights(0., 0., 0., 0., 0., 0.)
NET_WEIGHTS_L2HMC = NetWeights(1., 1., 1., 1., 1., 1.)
PI = np.pi
TWO_PI = 2 * PI

RunParams = namedtuple('RunParams', [
    'eps', 'beta', 'run_steps', 'num_steps', 'batch_size',
    'init', 'print_steps', 'mix_samplers',
    'num_singular_values', 'net_weights', 'network_type'
])

Observables = namedtuple('Observables', ['plaqs', 'avg_plaqs', 'charges'])
Energy = namedtuple('Energy', ['potential', 'kinetic', 'hamiltonian'])


def project_angle(x):
    return x - TWO_PI * np.floor((x + PI) / TWO_PI)


def charge_as_int(q):
    return np.floor(q + 0.5)


def project_angle_fft(x, n=10):
    """use the fourier series representation `x` to approx `project_angle`.
    note: because `project_angle` suffers a discontinuity, we approximate `x`
    with its fourier series representation in order to have a differentiable
    function when computing the loss.
    args:
        x (array-like): array to be projected.
        n (int): number of terms to keep in fourier series.
    """
    y = np.zeros(x.shape, dtype=x.dtype)
    for n in range(1, n):
        y += (-2 / n) * ((-1) ** n) * np.sin(n * x)
    return y


def _get_eps(log_dir):
    """Get the step size `eps` by looking for it in `log_dir` ."""
    try:
        in_file = os.path.join(log_dir, 'eps_np.z')
        eps_dict = io.loadz(in_file)
        eps = eps_dict['eps']
    except FileNotFoundError:
        run_dirs = io.get_run_dirs(log_dir)
        rp_file = os.path.join(run_dirs[0], 'run_params.z')
        if os.path.isfile(rp_file):
            run_params = io.loadz(rp_file)
        else:
            rp_file = os.path.join(run_dirs[-1], 'run_params.z')
            if os.path.isfile(rp_file):
                run_params = io.loadz(rp_file)
            else:
                raise FileNotFoundError('Unable to load run_params.')
        eps = run_params['eps']
    return eps


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
    for key, val in weights['xnet'].items():
        if 'layer' in key:
            W, b = val
            W_ = reduced_weight_matrix(W, n=n)
            weights['xnet'][key] = Weights(W_, b)
    for key, val in weights['vnet'].items():
        if 'layer' in key:
            W, b = val
            W_ = reduced_weight_matrix(W, n=n)
            weights['vnet'][key] = Weights(W_, b)

    return weights


class RunConfig:
    """Configuration object for running inference."""
    def __init__(self,
                 run_params,
                 log_dir=None,
                 extra_params=None,
                 model_type=None,
                 from_trained_model=True):
        self.model_type = model_type
        if from_trained_model:
            if log_dir is None:
                if extra_params is None:
                    log_dir = os.getcwd()
                else:
                    log_dir = extra_params.get('log_dir', os.getcwd())

            extra_params = self.find_params(log_dir)

        self.log_dir = log_dir
        self.run_params = self.set_attrs(run_params, extra_params)
        if from_trained_model:
            self.weights = self.load_weights(self.log_dir)
        else:
            self.weights = {
                'xnet': None,
                'vnet': None,
            }

        self.run_str = self.get_run_str()
        self.run_dir = os.path.join(self.log_dir, 'runs_np', self.run_str)
        io.check_else_make_dir(self.run_dir)

        self.dynamics_params = DynamicsParamsNP(
            eps=self.eps,
            num_steps=self.num_steps,
            input_shape=self.input_shape,
            net_weights=self.net_weights,
            network_type=self.network_type,
            weights=self.weights,
            model_type=self.model_type,
        )

    @staticmethod
    def check_log_dir(params, log_dir=None):
        """Check if"""
        log_dir = params.get('log_dir', None)
        if log_dir is None:
            raise OSError('`log_dir` not specified.')

        return log_dir

    @staticmethod
    def find_params(log_dir=None):
        """Try and locate the parameters file to load from."""
        names = ['parameters.pkl', 'parameters.z', 'params.pkl', 'params.z']
        if log_dir is None:
            d = os.getcwd()
        else:
            d = log_dir
        #  dirs = [self.log_dir, os.getcwd()] if log_dir is None else [log_dir]
        #  for d in dirs:
        files = [os.path.join(d, name) for name in names]
        for f in files:
            if os.path.isfile(f):
                return io.loadz(f)

    def load_weights(self, log_dir=None):
        """Load weights from `log_dir`."""
        if log_dir is None:
            log_dir = self.log_dir

        if self.net_weights == NET_WEIGHTS_HMC:
            weights = {
                'xnet': None,
                'vnet': None,
            }
        else:
            weights = {
                'xnet': io.loadz(os.path.join(log_dir, 'xnet_weights.z')),
                'vnet': io.loadz(os.path.join(log_dir, 'vnet_weights.z')),
            }

        return weights

    # pylint: disable=attribute-defined-outside-init
    def set_attrs(self, run_params, extra_params):
        run_params = AttrDict(run_params._asdict())
        run_params.update({
            k: v for k, v in extra_params.items() if k not in run_params
        })

        if run_params.get('eps', None) is None:
            run_params['eps'] = _get_eps(self.log_dir)

        dim = run_params.get('dim', None)
        time_size = run_params.get('time_size', None)
        space_size = run_params.get('space_size', None)
        xdim = time_size * space_size * dim
        input_shape = (run_params.batch_size, xdim)

        run_params['xdim'] = xdim
        run_params['input_shape'] = input_shape

        self.eps = run_params.eps
        self.xdim = run_params.xdim
        self.beta = run_params.beta
        self.init = run_params.init
        self.run_steps = run_params.run_steps
        self.num_steps = run_params.num_steps
        self.batch_size = run_params.batch_size
        self.net_weights = run_params.net_weights
        self.print_steps = run_params.print_steps
        self.input_shape = run_params.input_shape
        #  self.input_shape = (self.batch_size, self.xdim)
        self.network_type = run_params.network_type
        self.mix_samplers = run_params.mix_samplers
        self.num_singular_values = run_params.num_singular_values

        return run_params

    def _update_params(self, params=None, **kwargs):
        """Update params with new values from **kwargs."""
        if params is None:
            params = {}

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

    def get_run_str(self):
        nw_str = ''.join((strf(i).replace('.', '') for i in self.net_weights))
        beta_str = f'{self.beta}'.replace('.', '')
        eps_str = f'{self.eps:.3g}'.replace('.', '')
        run_str = (f'lf{self.num_steps}'
                   f'_bs{self.batch_size}'
                   f'_steps{self.run_steps}'
                   f'_beta{beta_str}'
                   f'_eps{eps_str}'
                   f'_nw{nw_str}'
                   f'_init{self.init}')

        if self.mix_samplers:
            run_str += f'_mix_samplers'

        if self.num_singular_values > 0:
            run_str += f'_nsv{self.num_singular_values}'

        time_strs = io.get_timestr()
        timestr = time_strs['timestr']
        run_str += f'__{timestr}'

        return run_str


class RunnerNP:
    """Responsible for running inference using `numpy` from trained model."""
    def __init__(self,
                 run_params,
                 log_dir=None,
                 model_type=None,
                 extra_params=None,
                 from_trained_model=True):
        self.config = RunConfig(run_params,
                                log_dir=log_dir,
                                model_type=model_type,
                                extra_params=extra_params,
                                from_trained_model=from_trained_model)
        #  self.config = RunConfig(run_params, log_dir=log_dir,
        #                          model_type=model_type)
        #  self.train_params = self.config.extra_params
        pw = self.config.run_params.get('plaq_weight', 0.1)
        qw = self.config.run_params.get('charge_weight', 0.1)
        self._plaq_weight = max((pw, 0.1))
        self._charge_weight = max((qw, 0.1))

        self.log_dir = self.config.log_dir
        self.lattice = self.create_lattice(self.config.run_params)
        self._potential_fn = self.lattice.calc_actions_np
        self.dynamics = self.create_dynamics(self._potential_fn,
                                             self.config.dynamics_params)

    def create_lattice(self, params):
        """Craete `GaugeLattice` object."""
        return GaugeLattice(params['time_size'], params['space_size'],
                            dim=2, link_type='U1', rand=True,
                            batch_size=self.config.batch_size)

    def create_dynamics(self, potential_fn=None, params=None, masks='rand'):
        """Create `DynamicsNP` object."""
        if potential_fn is None:
            potential_fn = self._potential_fn
        if params is None:
            params = self.config.dynamics_params

        # to run with reduced weight matrix keeping first `num_singular_values`
        # singular values from the SVD decomposition: W = U * S * V^{H}
        nsv = self.config.num_singular_values
        if nsv > 0:
            io.log(f'Keeping the first {nsv} singular values!')
            weights = get_reduced_weights(params.weights, n=nsv)
            params._update(weights=weights)

        dynamics = DynamicsNP(potential_fn, params)

        masks_file = os.path.join(self.log_dir, 'dynamics_mask.z')
        if os.path.isfile(masks_file):
            dynamics.set_masks(io.loadz(masks_file))

        return dynamics

    def create_hmc_dynamics(self, potential_fn=None, params=None):
        """Create `DynamicsNP` object for running generic HMC."""
        if potential_fn is None:
            potential_fn = self._potential_fn

        if params is None:
            params = self.config.dynamics_params

        weights = {
            'xnet': None,
            'vnet': None,
        }
        hmc_params = DynamicsParamsNP(**params._asdict())
        hmc_params._replace(weights=weights)
        hmc_params._replace(net_weights=NET_WEIGHTS_HMC)
        return self.create_dynamics(potential_fn,
                                    params=hmc_params)

    def _calc_observables(self, x):
        """Calculate observables from `x`."""
        plaqs = self.lattice.calc_plaq_sums_np(x)
        avg_plaqs = np.sum(np.cos(plaqs), axis=(1, 2)) / self.lattice.num_plaqs
        plaqs_proj = project_angle(plaqs)
        charges = np.sum(plaqs_proj, axis=(1, 2)) / TWO_PI

        return Observables(plaqs, avg_plaqs, charges)

    def _calc_observables1(self, x, n_fft=0, loop_sizes=None):
        """"Calculate quantities of interest from `x`."""
        if loop_sizes is None:
            loop_sizes = [1]

        p_arr = []
        pavg_arr = []
        q_arr = []
        for loop_size in loop_sizes:
            p = self.lattice.calc_plaq_sums_np(x, loop_size)
            pavg = np.mean(p, axis=(1, 2))
            #  nlp = self.lattice.num_plaqs
            #  pavg = np.sum(np.cos(plaqs), axis=(1, 2)) / nlp
            pp = project_angle(p)
            q = np.sum(pp, axis=(1, 2)) / TWO_PI

            #  if n_fft > 0:  # project angle using FFT
            #      pp = project_angle_fft(p, n=n_fft)
            #  else:
            #      pp = np.sin(p)

            p_arr.append(p)
            q_arr.append(q)
            pavg_arr.append(pavg)

        obs = Observables(np.array(p_arr), np.array(pavg_arr), np.array(q_arr))

        return obs

    def calc_losses(self, mc_observables, accept_prob):
        """Calc charge and plaquette losses.

        Args:
            mc_observables (MonteCarloStates): Namedtuple object containing
                initial, proposed, and output values of Observables.
            accept_prob (array-like): Array of shape (batch_size,) containing
                the acceptance probabilities.

        Returns:
            plaq_loss (array-like): Batch-wise plaquette losses.
            charge_loss (array-like): Batch-wise charge losses.
        """
        obs_init = mc_observables.init  # initial values of observables
        obs_prop = mc_observables.proposed  # proposed values of observables
        # individual plaquette differences
        dp_ = 2. * (1. - np.cos(obs_prop.plaqs - obs_init.plaqs))

        # sum plaquette differences over all plaquettes, get expected val
        dp = accept_prob * np.sum(dp_, axis=(1, 2))
        # get expected value of the charge difference squared
        dq = accept_prob * (obs_prop.charges - obs_init.charges) ** 2
        #  dq = np.sum(dq, axis=0)

        # scale by weight values
        ploss = -dp / self._plaq_weight
        qloss = -dq / self._charge_weight

        return ploss, qloss

    def calc_observables_and_losses(self, mc_states, accept_prob):
        """Calculate observables and losses.

        Args:
            mc_states (MonteCarloStates): Namedtuple object containing initial,
                proposed, and output states from the LeapFrog sampler.
            accept_prob (array-like): Array of shape (batch_size,) containing
                the acceptance probabilities.

        Returns:
            plaq_loss (array-like): The batch-wise plaquette loss.
            charge_loss (array-like): The batch-wise charge loss.
            mc_observables (MonteCarloStates): Namedtuple object containing
                initial, proposed, and output values of Observables.
        """
        obs_init = self._calc_observables(mc_states.init.x)
        obs_out = self._calc_observables(mc_states.out.x)
        obs_prop = self._calc_observables(mc_states.proposed.x)
        mc_observables = MonteCarloStates(obs_init, obs_prop, obs_out)
        ploss, qloss = self.calc_losses(mc_observables, accept_prob)

        return ploss, qloss, mc_observables

    def calc_changes(self, mc_observables):
        """Calculate the changes in the plaquettes and top charges."""
        p_init = mc_observables.init.plaqs
        p_out = mc_observables.out.plaqs
        q_init = mc_observables.init.charges
        q_out = mc_observables.out.charges

        dp = 2. * (1. - np.cos(p_out - p_init))
        dq = np.abs(np.around(q_out) - np.around(q_init))

        return dp, dq

    def _calc_energies(self, state):
        potential = self.dynamics.potential_energy(state.x, state.beta)
        kinetic = self.dynamics.kinetic_energy(state.v)
        hamiltonian = potential + kinetic

        return Energy(potential, kinetic, hamiltonian)

    def calc_energies(self, mc_states):
        energy_init = self._calc_energies(mc_states.init)
        energy_proposed = self._calc_energies(mc_states.proposed)
        energy_out = self._calc_energies(mc_states.out)
        mc_energies = MonteCarloStates(energy_init,
                                       energy_proposed,
                                       energy_out)
        exp_energy_diff = np.exp(energy_out.hamiltonian
                                 - energy_init.hamiltonian)

        return mc_energies, exp_energy_diff

    def inference_step(self, step: int, x: np.ndarray) -> dict:
        start_time = time.time()
        # mc_s (MonteCarloStates): initial, proposed, output STATE
        # sld_s (MonteCarloStates): initial, proposed, output SUMLOGDET
        # state_diff_r (State): State(xdiff_r, vdiff_r, state.beta)
        mc_s, px, sld_s, state_diff_r = self.dynamics(x, self.config.beta)
        step_time = time.time() - start_time

        mc_energies, exp_energy_diff = self.calc_energies(mc_s)

        ploss, qloss, mc_obs = self.calc_observables_and_losses(mc_s, px)
        dp, dq = self.calc_changes(mc_obs)
        plaqs_diffs = calc_plaqs_diffs(mc_obs.out.avg_plaqs, self.config.beta)

        data_str = (f" {step:>5g}/{self.config.run_steps:<5g}"
                    f"{step_time:^12.3g}"
                    f"{px.mean():^12.3g}"
                    f"{state_diff_r.x.mean():^12.3g}"
                    f"{state_diff_r.v.mean():^12.3g}"
                    f"{sld_s.out.mean():^12.4g}"
                    f"{exp_energy_diff.mean():^12.4g}"
                    f"{ploss.mean():^12.4g}"
                    f"{qloss.mean():^12.3g}"
                    f"{dp.mean():^12.3g}"
                    f"{dq.sum():^12.3g}"
                    f"{plaqs_diffs.mean():^12.3g}")

        outputs = AttrDict({
            'data_str': data_str,
            'mc_states': mc_s,    # MonteCarloStates (State; init, prop, out)
            'accept_prob': px,    # acceptance probabilities
            'sld_states': sld_s,  # MonteCarloStates (sumlogdet)
            'state_diff_r': state_diff_r,  # State(x=xdiff_r, v=vdiff_r, _)
            'mc_energies': mc_energies,    # MonteCarloStates (Energy)
            'exp_energy_diff': exp_energy_diff,  # exp(H_out - H_init)
            'plaq_loss': ploss,          # plaquette loss
            'charge_loss': qloss,        # charge loss
            'mc_observables': mc_obs,    # MonteCarloStates (Observables)
            'plaq_change': dp,           # 2 * (1 - cos(p_out - p_init))
            'charge_change': dq,         # q_out - q_init
            'plaqs_diffs': plaqs_diffs,  # difference b/t actual and expected
        })

        return mc_s.out.x, self._expand_outputs(outputs)

    def _expand_outputs(self, data):
        return {
            'data_str': data['data_str'],
            'accept_prob': data['accept_prob'],
            'sumlogdet_proposed': data['sld_states'].proposed,
            'sumlogdet_out': data['sld_states'].out,
            'exp_energy_diff': data['exp_energy_diff'],
            'plaq_loss': data['plaq_loss'],
            'charge_loss': data['charge_loss'],
            'plaqs': data['mc_observables'].out.plaqs,
            'charges': data['mc_observables'].out.charges,
            'plaqs_diffs': data['plaqs_diffs'],
            'plaq_change': data['plaq_change'],
            'charge_change': data['charge_change'],
            'xdiff_r': data['state_diff_r'].x,
            'vdiff_r': data['state_diff_r'].v,
            'potential_init': data['mc_energies'].init.potential,
            'potential_proposed': data['mc_energies'].proposed.potential,
            'potential_out': data['mc_energies'].out.potential,
            'kinetic_init': data['mc_energies'].init.kinetic,
            'kinetic_proposed': data['mc_energies'].proposed.kinetic,
            'kinetic_out': data['mc_energies'].out.kinetic,
            'hamiltonian_init': data['mc_energies'].init.hamiltonian,
            'hamiltonian_proposed': data['mc_energies'].proposed.hamiltonian,
            'hamiltonian_out': data['mc_energies'].out.hamiltonian,
        }

    def _init_x(self) -> np.ndarray:
        if self.init == 'zeros':
            x = np.zeros(self.config.input_shape)
        if self.init == 'ones':
            x = np.ones(self.config.input_shape)
        else:
            if self.model_type == 'GaugeModel':
                x = np.random.uniform(-np.pi, np.pi,
                                      size=self.config.input_shape)
            else:
                x = np.random.randn(*self.config.input_shape)

        return x

    def inference(self, x=None, run_steps=None):
        """Run inference."""
        if x is None:
            x = self._init_x()

        if run_steps is None:
            run_steps = self.config.run_steps

        run_data = RunData(self.config)
        for step in range(run_steps):
            x, outputs = self.inference_step(step, x)
            run_data.update(step, x, outputs)
            if step % 1000 == 0:
                io.log(HSTR)

        return run_data
