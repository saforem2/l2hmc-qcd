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

import autograd.numpy as np

from autograd import elementwise_grad

from config import NP_FLOAT, State
from network.layers import linear, relu
from network.encoder_net import EncoderNetNP
from network.generic_net import GenericNetNP
from network.cartesian_net import CartesianNetNP
from network.gauge_network import GaugeNetworkNP


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


class DynamicsNP(object):
    """Implements tools for running tensorflow-independent inference."""
    def __init__(self,
                 x_dim,
                 params,
                 weights,
                 potential_fn,
                 model_type=None):
        """Init.
        Args:
            potential_fn (callable): Potential energy function.
            weights (dict): Dictionary of weights, from `log_dir/weights.pkl`
                file where `log_dir` contains the trained model.
            hmc (bool): Run generic HMC (faster)
            model_type (str): Model type. (if 'GaugeModel', run extra ops)
            params (dict): Dictionary of parameters used.
        """
        if model_type is None:
            model_type = 'None'

        self.x_dim = x_dim
        self.potential = potential_fn
        self._model_type = model_type

        self.eps = float(params.get('eps', None))
        self.num_steps = int(params.get('num_steps', None))
        self.batch_size = int(params.get('batch_size', None))
        self.direction = params.get('direction', 'rand')
        self.zero_masks = params.get('zero_masks', False)
        self._input_shape = params.get('_input_shape', None)
        self._network_type = params.get('network_type', None)

        activation = params.get('activation', 'relu')
        if activation == 'relu':
            self._activation_fn = relu
        elif activation == 'tanh':
            self._activation_fn = np.tanh
        else:
            self._activation_fn = linear

        if params.get('hmc', False):
            self.xnet, self.vnet = self.hmc_networks()
        else:
            self.xnet, self.vnet = self.build_networks(weights)

        self._forward = self._setup_direction()
        self.masks = self.build_masks()
        #  self.tau = self.build_time()

    def _setup_direction(self):
        if self.direction == 'forward':
            forward = True
        elif self.direction == 'backward':
            forward = False
        else:
            forward = None

        return forward

    def __call__(self, *args, **kwargs):
        return self.apply_transition(*args, **kwargs)

    def transition_forward(self, x, beta, net_weights,
                           v=None, model_type=None):
        """Propose a new state by running the transition kernel forward.
        Args:
            x (array-like): Input samples.
            beta (float): Inverse temperature (gauge coupling constant).
            net_weights (NetWeights): Tuple of net weights for scaling the
                network functions.
            model_type (str): String specifying the type of model.
        Returns:
            outputs (dict): Dictionary containing the outputs.
        """
        if model_type == 'GaugeModel':
            x = convert_to_angle(x)

        if v is None:
            v = np.random.normal(size=x.shape)

        xf, vf, pxf, sumlogdetf = self.transition_kernel(*State(x, v, beta),
                                                         net_weights,
                                                         forward=True)
        mask_a, mask_r, _ = self._get_accept_masks(pxf)
        x_out = xf * mask_a[:, None] + x * mask_r[:, None]
        v_out = vf * mask_a[:, None] + v * mask_r[:, None]
        sumlogdet_out = sumlogdetf * mask_a

        outputs = {
            'x_init': x,
            'v_init': v,
            'x_proposed': xf,
            'v_proposed': vf,
            'x_out': x_out,
            'v_out': v_out,
            'sumlogdet_out': sumlogdet_out,
        }

        return outputs

    def transition_backward(self, x, beta, net_weights,
                            v=None, model_type=None):
        """Propose a new state by running the transition kernel backward."""
        if model_type == 'GaugeModel':
            x = convert_to_angle(x)

        if v is None:
            v = np.random.normal(size=x.shape)

        xb, vb, pxb, sumlogdetb = self.transition_kernel(*State(x, v, beta),
                                                         net_weights,
                                                         forward=False)
        mask_a, mask_r, _ = self._get_accept_masks(pxb)
        x_out = xb * mask_a[:, None] + x * mask_r[:, None]
        v_out = vb * mask_a[:, None] + v * mask_r[:, None]
        sumlogdet_out = sumlogdetb * mask_a

        outputs = {
            'x_init': x,
            'v_init': v,
            'x_proposed': xb,
            'v_proposed': vb,
            'x_out': x_out,
            'v_out': v_out,
            'sumlogdet_out': sumlogdet_out
        }

        return outputs

    def apply_transition(self, x, beta, net_weights):
        """Propose a new state and perform the accept/reject step."""
        forward = self._forward
        if forward is None:
            forward = (np.random.uniform() < 0.5)

        if self._model_type == 'GaugeModel':
            x = convert_to_angle(x)

        v_init = np.random.normal(size=x.shape)
        state_init = State(x, v_init, beta)
        x_, v_, px, sumlogdet = self.transition_kernel(*state_init,
                                                       net_weights,
                                                       forward=forward)
        if self._model_type == 'GaugeModel':
            x_ = convert_to_angle(x_)

        # Check reversibility using proposed and initial states
        state_prop = State(x_, v_, beta)
        xdiff_r, vdiff_r = self.check_reversibility(state_init, state_prop,
                                                    forward, net_weights)

        mask_a, mask_r, rand_num = self._get_accept_masks(px)
        x_out = x_ * mask_a[:, None] + x * mask_r[:, None]
        v_out = v_ * mask_a[:, None] + v_init * mask_r[:, None]
        sumlogdet_out = sumlogdet * mask_a

        # TODO: Simplify outputs by grouping (x, v, beta) into State(s)
        outputs = {
            'x_init': x,
            'v_init': v_init,
            'x_proposed': x_,
            'v_proposed': v_,
            'x_out': x_out,
            'v_out': v_out,
            'accept_prob': px,
            'sumlogdet_proposed': sumlogdet,
            'sumlogdet_out': sumlogdet_out,
            'forward': forward,
            'mask_a': mask_a,
            'mask_r': mask_r,
            'rand_num': rand_num,
            'xdiff_r': xdiff_r,
            'vdiff_r': vdiff_r,
        }

        return outputs

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

    def check_reversibility(self, s_init, s_prop, forward, net_weights):
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
        x_r, v_r, _, _ = self.transition_kernel(*s_prop,
                                                net_weights,
                                                forward=(not forward))
        dv = v_r - s_init.v
        if self._model_type == 'GaugeModel':
            x_r = convert_to_angle(x_r)
            dx = 1. - np.cos(x_r - s_init.x)
        else:
            dx = x_r - s_init

        return dx, dv

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

        xp, vp, _, _ = self.transition_kernel(*state, net_weights,
                                              forward=forward)

        state_pert = State(x=x_pert, v=v_pert, beta=state.beta)
        xp_pert, vp_pert, _, _ = self.transition_kernel(*state_pert,
                                                        net_weights,
                                                        forward=forward)
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

    def apply_transition_both(self, x, beta, net_weights, model_type=None):
        """Propose a new state and perform the accept/reject step."""
        if model_type == 'GaugeModel':
            x = convert_to_angle(x)

        vf_init = np.random.normal(size=x.shape)
        state_init_f = State(x, vf_init, beta)
        xf, vf, pxf, sumlogdetf = self.transition_kernel(*state_init_f,
                                                         net_weights,
                                                         forward=True)
        vb_init = np.random.normal(size=x.shape)
        state_init_b = State(x, vb_init, beta)
        xb, vb, pxb, sumlogdetb = self.transition_kernel(*state_init_b,
                                                         net_weights,
                                                         forward=False)

        mask_f, mask_b = self._get_direction_masks()
        v_init = (vf_init * mask_f[:, None] + vb_init * mask_b[:, None])
        x_prop = xf * mask_f[:, None] + xb * mask_b[:, None]
        v_prop = vf * mask_f[:, None] + vb * mask_b[:, None]

        accept_prob = pxf * mask_f + pxb * mask_b
        sumlogdet_prop = sumlogdetf * mask_f + sumlogdetb * mask_b

        mask_a, mask_r, rand_num = self._get_accept_masks(accept_prob)
        x_out = x_prop * mask_a[:, None] + x * mask_r[:, None]
        v_out = v_prop * mask_a[:, None] + v_init * mask_r[:, None]
        sumlogdet_out = sumlogdet_prop * mask_a

        # TODO: Simplify outputs via `state_init, state_prop, state_out, ...`
        outputs = {
            'x_init': x,
            'v_init': v_init,
            'x_proposed': x_prop,
            'v_proposed': v_prop,
            'x_out': x_out,
            'v_out': v_out,
            'xf': xf,
            'xb': xb,
            'pxf': pxf,
            'pxb': pxb,
            'accept_prob': accept_prob,
            'sumlogdet_proposed': sumlogdet_prop,
            'sumlogdet_out': sumlogdet_out,
            'mask_f': mask_f,
            'mask_b': mask_b,
            'mask_a': mask_a,
            'mask_r': mask_r,
            'rand_num': rand_num,
        }

        return outputs

    def transition_kernel(self, x_in, v_in, beta, net_weights, forward=True):
        """Transition kernel of augmented leapfrog integrator."""
        lf_fn = self._forward_lf if forward else self._backward_lf
        x_prop, v_prop = x_in, v_in
        sumlogdet = 0.
        for step in range(self.num_steps):
            x_prop, v_prop, logdet = lf_fn(x_prop, v_prop, beta,
                                           step, net_weights)
            sumlogdet += logdet

        accept_prob = self._compute_accept_prob(x_in, v_in,
                                                x_prop, v_prop,
                                                sumlogdet, beta)

        return x_prop, v_prop, accept_prob, sumlogdet

    def _forward_lf(self, x, v, beta, step, net_weights):
        """One forward augmented leapfrog step."""
        t = self._get_time(step, tile=x.shape[0])
        #  t = self._get_time(step)
        mask, mask_inv = self._get_mask(step)

        sumlogdet = 0.

        v, logdet = self._update_v_forward(x, v, beta, t, net_weights)
        sumlogdet += logdet

        x, logdet = self._update_x_forward(x, v, t, net_weights,
                                           (mask, mask_inv))
        sumlogdet += logdet

        x, logdet = self._update_x_forward(x, v, t, net_weights,
                                           (mask_inv, mask))
        sumlogdet += logdet

        v, logdet = self._update_v_forward(x, v, beta, t, net_weights)
        sumlogdet += logdet

        return x, v, sumlogdet

    def _backward_lf(self, x, v, beta, step, net_weights):
        """One backward augmented leapfrog step."""
        step_r = self.num_steps - step - 1
        t = self._get_time(step_r, tile=x.shape[0])
        #  t = self._get_time(step_r)
        mask, mask_inv = self._get_mask(step_r)

        sumlogdet = 0.

        v, logdet = self._update_v_backward(x, v, beta, t, net_weights)
        sumlogdet += logdet

        x, logdet = self._update_x_backward(x, v, t, net_weights,
                                            (mask_inv, mask))
        sumlogdet += logdet

        x, logdet = self._update_x_backward(x, v, t, net_weights,
                                            (mask, mask_inv))
        sumlogdet += logdet

        v, logdet = self._update_v_backward(x, v, beta, t, net_weights)
        sumlogdet += logdet

        return x, v, sumlogdet

    def _update_v_forward(self, x, v, beta, t, net_weights):
        """Update v in the forward leapfrog step."""
        if self._model_type == 'GaugeModel':
            x = convert_to_angle(x)

        grad = self.grad_potential(x, beta)
        Sv, Tv, Qv = self.vnet([x, grad, t])

        scale = net_weights.v_scale * (0.5 * self.eps * Sv)
        transf = net_weights.v_transformation * (self.eps * Qv)
        transl = net_weights.v_translation * Tv

        v = (v * np.exp(scale)
             - 0.5 * self.eps * (grad * np.exp(transf) + transl))
        logdet = np.sum(scale, axis=1)

        return v, logdet

    def _update_x_forward(self, x, v, t, net_weights, masks):
        """Update x in the forward leapfrog step."""
        if self._model_type == 'GaugeModel':
            x = convert_to_angle(x)

        mask, mask_inv = masks
        Sx, Tx, Qx = self.xnet([v, mask * x, t])

        scale = net_weights.x_scale * (self.eps * Sx)
        transf = net_weights.x_transformation * (self.eps * Qx)
        transl = net_weights.x_translation * Tx

        y = x * np.exp(scale) + self.eps * (v * np.exp(transf) + transl)
        x = mask * x + mask_inv * y
        logdet = np.sum(mask_inv * scale, axis=1)

        return x, logdet

    def _update_v_backward(self, x, v, beta, t, net_weights):
        """Update v in the backward lf step. Inverting the forward update."""
        if self._model_type == 'GaugeModel':
            x = convert_to_angle(x)

        grad = self.grad_potential(x, beta)
        Sv, Tv, Qv = self.vnet([x, grad, t])

        scale = net_weights.v_scale * (-0.5 * self.eps * Sv)
        transf = net_weights.v_transformation * (self.eps * Qv)
        transl = net_weights.v_translation * Tv

        v = np.exp(scale) * (
            v + 0.5 * self.eps * (grad * np.exp(transf) + transl)
        )
        logdet = np.sum(scale, axis=1)

        return v, logdet

    def _update_x_backward(self, x, v, t, net_weights, masks):
        """Update x in the backward lf step. Inverting the forward update."""
        if self._model_type == 'GaugeModel':
            x = convert_to_angle(x)

        mask, mask_inv = masks
        Sx, Tx, Qx = self.xnet([v, mask * x, t])

        scale = net_weights.x_scale * (-self.eps * Sx)
        transl = net_weights.x_translation * Tx
        transf = net_weights.x_transformation * (self.eps * Qx)

        exp_scale = np.exp(scale)
        exp_transf = np.exp(transf)
        y = exp_scale * (x - self.eps * (v * exp_transf + transl))
        xb = mask * x + mask_inv * y
        logdet = np.sum(mask_inv * scale, axis=1)

        return xb, logdet

    def _compute_accept_prob(self, xi, vi, xf, vf, sumlogdet, beta):
        """Compute the prob of accepting the proposed state given old state."""
        h_init = self.hamiltonian(xi, vi, beta)
        h_prop = self.hamiltonian(xf, vf, beta)
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

        return accept_mask, reject_mask, rand_unif

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

    def hmc_networks(self):
        """Build hmc networks that output all zeros from the S, T, Q fns."""
        xnet = lambda inputs: [  # noqa: E731
            np.zeros_like(inputs[0]) for _ in range(3)
        ]
        vnet = lambda inputs: [  # noqa: E731
            np.zeros_like(inputs[0]) for _ in range(3)
        ]

        return xnet, vnet

    def build_networks(self, weights):
        """Build neural networks."""
        if self._network_type == 'GaugeNetwork':
            xnet = GaugeNetworkNP(weights['xnet'],
                                  activation=self._activation_fn)
            vnet = GaugeNetworkNP(weights['vnet'],
                                  activation=self._activation_fn)
        elif self._network_type == 'CartesianNet':
            xnet = CartesianNetNP(weights['xnet'],
                                  activation=self._activation_fn)
            vnet = CartesianNetNP(weights['vnet'],
                                  activation=self._activation_fn)
        #  # TODO: Update GenericNetNP to use `self._activation_fn`.
        else:
            xnet = GenericNetNP(weights['xnet'])
            vnet = GenericNetNP(weights['vnet'])

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
        if self.zero_masks:
            return self._build_zero_masks()

        masks = []
        for _ in range(self.num_steps):
            _idx = np.arange(self.x_dim)
            idx = np.random.permutation(_idx)[:self.x_dim//2]
            mask = np.zeros((self.x_dim,))
            mask[idx] = 1.
            masks.append(mask[None, :])

        return masks

    def build_time(self):
        """Convert leapfrog step index into sinusoidal time."""
        #  ts = []
        #  for i in range(self.num_steps):
        #      t = np.array([np.cos(2 * np.pi * i / self.num_steps),
        #                    np.sin(2 * np.pi * i / self.num_steps)])
        #      ts.append(t[None, :])
        #
        #  return ts
        pass

    def set_masks(self, masks):
        """Set `self.masks` to `masks`."""
        self.masks = masks[:self.num_steps]

    def _get_mask(self, step):
        m = self.masks[step]
        return m, 1. - m

    def grad_potential(self, x, beta):
        """Caclulate the element wise gradient of the potential energy fn."""
        grad_fn = elementwise_grad(self.potential_energy, 0)
        #  if HAS_JAX:
        #      grad_fn = jax.vmap(jax.grad(self.potential_energy, argnums=0))
        #  grad_fn = grad(self.potential_energy, 0)

        return grad_fn(x, beta)

    def potential_energy(self, x, beta):
        """Potential energy function."""
        return beta * self.potential(x)

    @staticmethod
    def kinetic_energy(v):
        """Kinetic energy function."""
        return 0.5 * np.sum(v ** 2, axis=-1)

    def hamiltonian(self, x, v, beta):
        """Hamiltonian function, H = PE + KE."""
        return self.kinetic_energy(v) + self.potential_energy(x, beta)
