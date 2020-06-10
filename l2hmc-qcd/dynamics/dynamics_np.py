"""
dynamics_np.py
Numpy implementation of the dynamics engine for the L2HMC sampler, allowing for
inference to be run on a trained model without the need for the `tf.Graph`
object.
Author: Sam Foreman (github: @saforem2)
Date: 01/07/2020
"""
# pylint: disable=too-many-locals
# pylint: disable=useless-object-inheritance
# pylint: disable=too-many-instance-attributes,
# pylint: disable=invalid-name, too-many-arguments
# pylint:disable=no-member
from __future__ import absolute_import, division, print_function
import os

import autograd.numpy as np

from autograd import elementwise_grad
from collections import namedtuple

from config import NP_FLOAT, State, NetWeights
from network.layers import linear, relu
from .dynamics import MonteCarloStates
import utils.file_io as io
#  from network.generic_net import GenericNetNP
from network.gauge_network import GaugeNetworkNP

# TODO: Put all `namedtuple` objects in `dynamics/__init__.py`

NET_WEIGHTS_HMC = NetWeights(0., 0., 0., 0., 0., 0.)
NET_WEIGHTS_L2HMC = NetWeights(1., 1., 1., 1., 1., 1.)

DynamicsParamsNP = namedtuple('DynamicsConfigNP', [
    'num_steps', 'eps', 'input_shape',
    'net_weights', 'network_type',
    'weights', 'model_type',
])


def reduced_weight_matrix(W, n=10):
    """Use the first n singular vals to reconstruct the original matrix W."""
    U, S, V = np.linalg.svd(W)
    W_ = np.matrix(U[:, :n]) * np.diag(S[:n]) * np.matrix(V[:n, :])

    return W_


def convert_to_angle(x):
    """Restrict `x` to be in the range -pi <= x < pi."""
    x = np.mod(x, 2 * np.pi)
    x -= np.floor(x / (2 * np.pi) + 0.5) * 2 * np.pi
    #  x = np.mod(x + np.pi, 2 * np.pi) - np.pi
    return x


class DynamicsConfigNP:
    """DynamicsConfigNP object."""
    def __init__(self, dynamics_params_np):
        self._params = dynamics_params_np
        self.set_attrs(self._params)

    # pylint: disable=attribute-defined-outside-init
    def set_attrs(self, params=None):
        if params is None:
            params = self._params

        self.eps = params.eps
        self.num_steps = int(params.num_steps)
        self.model_type = params.model_type

        self.net_weights = params.net_weights
        self.hmc = bool(self.net_weights == NET_WEIGHTS_HMC)

        self.weights = params.weights
        self.network_type = params.network_type

        self.batch_size = params.input_shape[0]
        if len(params.input_shape) == 2:
            self.lattice_shape = None
            self.xdim = params.input_shape[1]
        elif len(params.input_shape) == 4:
            self.lattice_shape = params.input_shape[1:]
            self.xdim = np.cumprod(self.lattice_shape)[-1]

    def save(self, out_dir):
        """Save the config to `out_dir` as both `.txt` and `.z` files."""
        io.savez(self._params, os.path.join(out_dir, 'dynamics_params_np.z'))
        io.savez(self.__dict__, os.path.join(out_dir, 'dynamics_config_np.z'))

        txt_file = os.path.join(out_dir, 'dynamics_params_np.txt')
        with open(txt_file, 'w') as f:
            for key, val in self.__dict__.items():
                f.write(f'{key}: {val}\n')

    @staticmethod
    def load(fpath):
        """Load `params` from `fpath`."""
        if os.path.isdir(fpath):
            params_file = os.path.join(fpath, 'dynamics_params_np.z')
            if os.path.isfile(params_file):
                params = io.loadz(params_file)
        elif os.path.isfile(fpath):
            params = io.loadz(fpath)

        else:
            raise FileNotFoundError('Incorrect fpath specified.')

        return params

    def restore(self, fpath):
        """Restore params from `fpath`."""
        params = self.load(fpath)
        self.set_attrs(params)


class DynamicsNP(object):
    """Implements tools for running tensorflow-independent inference."""
    def __init__(self, potential_fn, dynamics_params,
                 model_type=None, separate_nets=True):
        """Init.
        Args:
            potential_fn (callable): Potential energy function.
            dynamics_params (DynamicsParamsNP): Configuration object.
            model_type (str): Model type. (if 'GaugeModel', run extra ops)
        """
        self.params = dynamics_params
        self.config = DynamicsConfigNP(dynamics_params)
        self._potential_fn = potential_fn
        self._model_type = self.config.model_type

        self._activation_fn = relu
        self.xdim = int(self.config.xdim)
        self.eps = float(self.config.eps)
        self.num_steps = int(self.config.num_steps)

        self.masks = self.build_masks()
        self.xnet, self.vnet = self.build_networks(self.config.weights)

        self._xsw = self.config.net_weights.x_scale
        self._xtw = self.config.net_weights.x_translation
        self._xqw = self.config.net_weights.x_transformation
        self._vsw = self.config.net_weights.v_scale
        self._vtw = self.config.net_weights.v_translation
        self._vqw = self.config.net_weights.v_transformation

    def _setup_direction(self):
        if self.direction == 'forward':
            forward = True
        elif self.direction == 'backward':
            forward = False
        else:
            forward = None

        return forward

    def __call__(self, x, beta):
        return self.apply_transition(x, beta)

    def fix_state(self, state):
        if self.config.model_type == 'GaugeModel':
            state = state._replace(x=convert_to_angle(state.x))
        return state

    def apply_transition(self, x, beta):
        forward = (np.random.uniform() < 0.5)
        v_init = np.random.randn(*x.shape)
        state_init = self.fix_state(State(x, v_init, beta))
        state_prop, px, sld_prop = self.transition_kernel(state_init, forward)

        mask_a, mask_r = self._get_accept_masks(px)
        x_out = state_prop.x * mask_a[:, None] + state_init.x * mask_r[:, None]
        v_out = state_prop.v * mask_a[:, None] + state_init.v * mask_r[:, None]
        sld_out = sld_prop * mask_a

        state_out = State(x_out, v_out, state_init.beta)
        mc_states = MonteCarloStates(state_init, state_prop, state_out)
        state_diff_r = self.check_reversibility(mc_states, forward)
        sld_states = MonteCarloStates(np.zeros_like(sld_out),
                                      sld_prop, sld_out)

        return mc_states, px, sld_states, state_diff_r

    def l2_metric(self, x1, x2):
        """"Calculate np.sqrt((x1 - x2) ** 2)."""
        if self._model_type == 'GaugeModel':
            x1 = np.array([np.cos(x1), np.sin(x1)])
            x2 = np.array([np.cos(x2), np.sin(x2)])
            dx = np.sqrt(np.sum((x1 - x2) ** 2, axis=0))
        else:
            dx = np.sqrt((x1 - x2) ** 2)

        #  dx = np.sqrt(np.sum((x1 - x2) ** 2, axis=0))
        return dx

    def check_reversibility(self, mc_states, forward):
        """Check reversibility.
        NOTE: `s_init = State(x_init, v_init, beta)`,
        Explicitly:
            1. Run MD:
                (state_init, d=forward) --> (state_prop, d=forward)
            2. Flip the direction and run MD:
                (state_prop, d=(not forward)) --> (state_r, d=(not forward))
            3. Check differences:
                dx = (state_r.x - state_init.x)
                dv = (state_r.v - state_init.v)
        """
        state_r, _, _ = self.transition_kernel(mc_states.proposed,
                                               forward=(not forward))
        dv = state_r.v - mc_states.init.v
        if self.config.model_type == 'GaugeModel':
            dx = 2. * (1. - np.cos(state_r.x - mc_states.init.x))
        else:
            dx = state_r.x - mc_states.init.x

        dstate_r = State(dx, dv, state_r.beta)

        #  dv = v_r - mc_states.init.v
        #  if self._model_type == 'GaugeModel':
        #      x_r = convert_to_angle(x_r)
        #      dx = 1. - np.cos(x_r - s_init.x)
        #  else:
        #      dx = x_r - s_init

        return dstate_r

    def volume_transformation(self, state, net_weights, eta=1e-3):
        """
        Check that the sampler is 'symplectic' (volume-preserving).
        MD update starting from from `s1 = state_init = (x1, v1, d)`:
            `s1 --> s2 = (x2, v2, d)`
        Want to see if, when starting from a slightly perturbed initial
        state, the (augmented) leapfrog sampler produces a slightly
        perturbed output.
        If the sampler is symplectic, plotting the RMS difference of the
        outputs vs the RMS diff of the inputs should be correlated with a slope
        of 1.
        Explicitly, perturb the initial state `s1`:
            `s1 --> s1 + ds1 = (x1 + dx1, v1 + dv1, d)`
        We know that running MD on `s1` gives `s2`, so we need to check that
        running MD on `s1 + ds1`:
            `s1 + ds1 --> _s2 + _ds2 = (_x2 + dx2, _v2 + dv2, d)`
        if symplectic, we should have `dx2 / dx1 ~ 1` and `dv2 / dv1 ~ 1`.
        Args:
            state1 (State object): Initial (starting state).
            net_weights (NetWeights object): Multiplicative scaling factors for
                network components.
        Returns:
            (dx1, dv1): Tuple consisting of the `perturbation` to the initial
                state.
            (dx2, dv2): Tuple consisting of the `perturbation` of the output
                state, after accept/reject. Computed as:
                    ```
                    dx2 = x2 - _x2
                    dv2 = v2 - _v2
                    ```
        """
        if self._model_type == 'GaugeModel':
            x_mod = convert_to_angle(state.x)
            state = State(x=x_mod, v=state.v, beta=state.beta)

        # Perturb the initial state
        dx_in = eta * np.random.randn(*state.x.shape)
        dv_in = eta * np.random.randn(*state.v.shape)
        x_pert = state.x + dx_in
        v_pert = state.v + dv_in

        if self._model_type == 'GaugeModel':
            x_pert = convert_to_angle(x_pert)
            dx_in = 1. - np.cos(state.x - x_pert)
        else:
            dx_in = state.x - x_pert

        # Randomly choose direction
        forward = (np.random.uniform() < 0.5)

        state_p, _, _ = self.transition_kernel(*state, net_weights,
                                               forward=forward)
        xp = state_p.x
        vp = state_p.v

        state_pert = State(x=x_pert, v=v_pert, beta=state.beta)
        state_pert, _, _ = self.transition_kernel(*state_pert,
                                                  net_weights,
                                                  forward=forward)
        xp_pert = state_pert.x
        vp_pert = state_pert.v
        dv_out = vp_pert - vp
        if self._model_type == 'GaugeModel':
            xp = convert_to_angle(xp)
            xp_pert = convert_to_angle(xp_pert)
            dx_out = 1. - np.cos(xp - xp_pert)

        else:
            dx_out = (xp_pert - xp)

        diffs = {
            'dx_in': dx_in,
            'dv_in': dv_in,
            'dx_out': dx_out,
            'dv_out': dv_out,
        }

        return diffs

    def transition_kernel(self, state, forward):
        """Transition kernel of augmented leapfrog integrator."""
        lf_fn = self._forward_lf if forward else self._backward_lf
        sumlogdet = 0.
        x_, v_ = state.x, state.v  # copy initial state
        state_ = State(x_, v_, state.beta)
        for step in range(self.config.num_steps):
            state_, logdet = lf_fn(state_, step)
            sumlogdet += logdet

        state_ = self.fix_state(state_)
        accept_prob = self._compute_accept_prob(state, state_, sumlogdet)

        return state_, accept_prob, sumlogdet

    def _forward_lf(self, state, step):
        """One forward augmented leapfrog step."""
        t = self._get_time(step, tile=state.x.shape[0])
        m, mc = self._get_mask(step)
        sumlogdet = 0.
        state, logdet = self._update_v_forward(state, t)
        sumlogdet += logdet
        state, logdet = self._update_x_forward(state, t, (m, mc))
        sumlogdet += logdet
        state, logdet = self._update_x_forward(state, t, (mc, m))
        sumlogdet += logdet
        state, logdet = self._update_v_forward(state, t)
        sumlogdet += logdet

        return state, sumlogdet

    def _backward_lf(self, state, step):
        """One backward leapfrog step."""
        step_r = self.config.num_steps - step - 1
        t = self._get_time(step_r, tile=state.x.shape[0])
        m, mc = self._get_mask(step_r)
        sumlogdet = 0.
        state, logdet = self._update_v_backward(state, t)
        sumlogdet += logdet
        state, logdet = self._update_x_backward(state, t, (mc, m))
        sumlogdet += logdet
        state, logdet = self._update_x_backward(state, t, (m, mc))
        sumlogdet += logdet
        state, logdet = self._update_v_backward(state, t)
        sumlogdet += logdet

        return state, sumlogdet

    def _update_v_forward(self, state, t):
        """Update the momentum `v` in the forward leapfrog step."""
        state = self.fix_state(state)
        grad = self.grad_potential(state)
        Sv, Tv, Qv = self.vnet((state.x, grad, t))

        scale = self._vsw * (0.5 * self.config.eps * Sv)
        transf = self._vqw * (self.config.eps * Qv)
        transl = self._vtw * Tv

        exp_s = np.exp(scale)
        exp_q = np.exp(transf)

        v = state.v * exp_s - 0.5 * self.config.eps * (grad * exp_q + transl)

        state_ = State(state.x, v, state.beta)
        logdet = np.sum(scale, axis=1)

        return state_, logdet

    def _update_x_forward(self, state, t, masks):
        """Update the position in the forward leapfrog step."""
        state = self.fix_state(state)
        m, mc = masks
        Sx, Tx, Qx = self.xnet((state.v, m * state.x, t))

        scale = self._xsw * (self.config.eps * Sx)
        transf = self._xqw * (self.config.eps * Qx)
        transl = self._xtw * Tx

        exp_s = np.exp(scale)
        exp_q = np.exp(transf)

        y = state.x * exp_s + self.config.eps * (state.v * exp_q + transl)
        x = m * state.x + mc * y

        state_ = self.fix_state(State(x, state.v, state.beta))
        logdet = np.sum(mc * scale, axis=1)

        return state_, logdet

    def _update_v_backward(self, state, t):
        """Update the momentum `v` in the backward leapfrog step."""
        state = self.fix_state(state)
        grad = self.grad_potential(state)
        Sv, Tv, Qv = self.vnet((state.x, grad, t))
        scale = self._vsw * (-0.5 * self.config.eps * Sv)
        transf = self._vqw * (self.config.eps * Qv)
        transl = self._vtw * Tv
        exp_s = np.exp(scale)
        exp_q = np.exp(transf)
        v = exp_s * (state.v + 0.5 * self.config.eps * (grad * exp_q + transl))
        state_ = State(state.x, v, state.beta)
        logdet = np.sum(scale, axis=1)

        return state_, logdet

    def _update_x_backward(self, state, t, masks):
        """Update the position `x` in the backward leapfrog update."""
        state = self.fix_state(state)
        m, mc = masks
        Sx, Tx, Qx = self.xnet((state.v, m * state.x, t))
        scale = self._xsw * (-self.config.eps * Sx)
        transf = self._xqw * (self.config.eps * Qx)
        transl = self._xtw * Tx
        exp_s = np.exp(scale)
        exp_q = np.exp(transf)
        y = exp_s * (state.x - self.eps * (state.v * exp_q + transl))
        x = m * state.x + mc * y
        state_ = self.fix_state(State(x, state.v, state.beta))
        logdet = np.sum(mc * scale, axis=1)

        return state_, logdet

    def _compute_accept_prob(self, state_init, state_prop, sumlogdet):
        """Compute the probability of accepting the proposed states."""
        h_init = self.hamiltonian(state_init)
        h_prop = self.hamiltonian(state_prop)
        dh = h_init - h_prop + sumlogdet
        prob = np.exp(np.minimum(dh, 0.))
        accept_prob = np.where(np.isfinite(prob), prob, np.zeros_like(prob))

        return accept_prob

    def _get_time(self, i, tile=1):
        """Format time as [cos(...), sin(...)]."""
        trig_t = np.squeeze([
            np.cos(2 * np.pi * i / self.num_steps),
            np.sin(2 * np.pi * i / self.num_steps),
        ])

        t = np.tile(np.expand_dims(trig_t, 0), (tile, 1))

        return t

    @staticmethod
    def _get_accept_masks(accept_prob, accept_mask=None):
        rand_unif = np.random.uniform(size=accept_prob.shape)
        if accept_mask is None:
            #  rand_unif = np.random.uniform(size=accept_prob.shape)
            accept_mask = np.array(accept_prob >= rand_unif, dtype=NP_FLOAT)
        reject_mask = 1. - accept_mask

        return accept_mask, reject_mask

    def _get_direction_masks(self):
        rand_unif = np.random.uniform(size=(self.batch_size,))
        forward_mask = np.array(rand_unif > 0.5, dtype=NP_FLOAT)
        backward_mask = 1. - forward_mask

        return forward_mask, backward_mask

    def _set_direction_masks(self, forward_mask):
        """Set direction masks using `forward_mask`."""
        # pylint:disable=attribute-defined-outside-init
        self.forward_mask = forward_mask
        self.backward_mask = 1. - forward_mask

    @staticmethod
    def hmc_networks():
        """Build hmc networks that output all zeros from the S, T, Q fns."""
        xnet = lambda inputs: [  # noqa: E731
            np.zeros_like(inputs[0]) for _ in range(3)
        ]
        vnet = lambda inputs: [  # noqa: E731
            np.zeros_like(inputs[0]) for _ in range(3)
        ]

        return xnet, vnet

    def build_networks(self, weights, separate_nets=True):
        """Build neural networks."""
        if self.config.hmc:
            xnet, vnet = self.hmc_networks()
        else:
            if self.config.network_type == 'GaugeNetwork':
                xnet = GaugeNetworkNP(weights['xnet'],
                                      activation=self._activation_fn)
                vnet = GaugeNetworkNP(weights['vnet'],
                                      activation=self._activation_fn)
            #  # TODO: Update other networks
            elif self.config.network_type == 'CartesianNet':
                pass
                #  xnet = CartesianNetNP(weights['xnet'],
                #                        activation=self._activation_fn)
                #  vnet = CartesianNetNP(weights['vnet'],
                #                        activation=self._activation_fn)
            else:
                pass
                #  xnet = GenericNetNP(weights['xnet'])
                #  vnet = GenericNetNP(weights['vnet'])

        return xnet, vnet

    def _build_zero_masks(self):
        masks = []
        for _ in range(self.num_steps):
            #  mask = np.zeros((self.x_dim,))
            mask = np.ones((self.x_dim,))
            masks.append(mask[None, :])

        return masks

    def build_masks(self):
        """Build `x` masks used for selecting which idxs of `x` get updated."""
        #  if self.zero_masks:
        #      return self._build_zero_masks()

        masks = []
        for _ in range(self.config.num_steps):
            _idx = np.arange(self.xdim)
            idx = np.random.permutation(_idx)[:self.xdim//2]
            mask = np.zeros((self.xdim,))
            mask[idx] = 1.
            masks.append(mask[None, :])

        return masks

    # pylint: disable=attribute-defined-outside-init
    def set_masks(self, masks):
        """Set `self.masks` to `masks`."""
        self.masks[:self.config.num_steps] = masks[:self.config.num_steps]

    def _get_mask(self, step):
        m = self.masks[step]
        return m, 1. - m

    def grad_potential(self, state):
        """Caclulate the element wise gradient of the potential energy fn."""
        grad_fn = elementwise_grad(self.potential_energy, 0)
        #  if HAS_JAX:
        #      grad_fn = jax.vmap(jax.grad(self.potential_energy, argnums=0))
        #  grad_fn = grad(self.potential_energy, 0)

        return grad_fn(state.x, state.beta)

    def potential_energy(self, x, beta):
        """Potential energy function."""
        return beta * self._potential_fn(x)

    @staticmethod
    def kinetic_energy(v):
        """Kinetic energy function."""
        return 0.5 * np.sum(v ** 2, axis=-1)

    def hamiltonian(self, state):
        """Hamiltonian function, H = PE + KE."""
        pe = self.potential_energy(state.x, state.beta)
        ke = self.kinetic_energy(state.v)


        return pe + ke
