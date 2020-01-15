"""
dynamics_np.py

Numpy implementation of the dynamics engine for the L2HMC sampler, allowing for
inference to be run on a trained model without the need for the `tf.Graph`
object.


Author: Sam Foreman (github: @saforem2)
Date: 01/07/2020
"""
from __future__ import absolute_import, division, print_function

import config as cfg

from collections import namedtuple

from utils.file_io import timeit  # noqa: F401
from network.activation_functions import ReLU

HAS_AUTOGRAD = False
try:
    import autograd.numpy as np

    from autograd import elementwise_grad

    HAS_AUTOGRAD = True
except ImportError:
    import numpy as np

# pylint: disable=invalid-name
State = cfg.State
Weights = namedtuple('Weights', ['w', 'b'])
NP_FLOAT = cfg.NP_FLOAT

def relu(x):
    return np.where(x >= 0, x, 0)



def linear(x):
    return x


class DenseLayerNP:
    """Implements fully-connected Dense layer using numpy."""
    def __init__(self, weights, activation=linear):
        self.activation = activation
        self.weights = weights
        self.w = weights.w
        self.b = weights.b

    def __call__(self, x):
        return self.activation(np.dot(x, self.w) + self.b)


class GenericNetNP:
    def __init__(self, weights, name=None, activation=relu):
        self.name = name
        self.activation = activation
        self.x_layer = DenseLayerNP(weights['x_layer'])
        self.v_layer = DenseLayerNP(weights['v_layer'])
        self.t_layer = DenseLayerNP(weights['t_layer'])
        self.h_layer = DenseLayerNP(weights['h_layer'])
        self.scale_layer = DenseLayerNP(weights['scale_layer'])

        transl_weight = weights['translation_layer']
        transf_weight = weights['transformation_layer']
        self.translation_layer = DenseLayerNP(transl_weight)
        self.transformation_layer = DenseLayerNP(transf_weight)

        self.coeff_scale = weights['coeff_scale']
        self.coeff_transformation = weights['coeff_transformation']

    def __call__(self, inputs):
        v, x, t = inputs
        v = self.v_layer(v)
        x = self.x_layer(x)
        t = self.t_layer(t)
        h = self.activation(v + x + t)
        h = self.activation(self.h_layer(h))

        S = np.tanh(self.scale_layer(h))
        Q = np.tanh(self.transformation_layer(h))

        scale = S * np.exp(self.coeff_scale)
        translation = self.translation_layer(h)
        transformation = Q * np.exp(self.coeff_transformation)

        return scale, translation, transformation


class DynamicsRunner:
    """Implements tools for running tensorflow-independent inference."""
    def __init__(self, potential_fn, weights, hmc=False, **params):
        self.potential = potential_fn
        if hmc:
            self.xnet, self.vnet = self.hmc_networks()
        else:
            self.xnet, self.vnet = self.build_networks(weights)

        for key, val in params.items():
            setattr(self, key, val)

        self.eps = params.get('eps', None)
        self.x_dim = params.get('x_dim', None)
        self.num_steps = params.get('num_steps', None)
        self.zero_masks = params.get('zero_masks', False)
        self.masks = self.build_masks(self.zero_masks)

    def __call__(self, *args, **kwargs):
        return self.apply_transition(*args, **kwargs)

    def apply_transition(self, x_init, beta, net_weights, model_type=None):
        """Propose a new state and perform the accept/reject step."""
        if model_type == 'GaugeModel':
            x_init = np.mod(x_init, 2 * np.pi)

        vf_init = np.random.normal(size=x_init.shape)
        state_init_f = cfg.State(x_init, vf_init, beta)
        xf, vf, pxf, sumlogdetf = self.transition_kernel(*state_init_f,
                                                         net_weights,
                                                         forward=True)
        vb_init = np.random.normal(size=x_init.shape)
        state_init_b = cfg.State(x_init, vb_init, beta)
        xb, vb, pxb, sumlogdetb = self.transition_kernel(*state_init_b,
                                                         net_weights,
                                                         forward=False)

        # TODO: Instead of running forward and backward simultaneously, 
        # use np.choose([-1, 1]) to determine which direction gets ran
        mask_f, mask_b = self._get_direction_masks()
        v_init = (vf_init * mask_f[:, None] + vb_init * mask_b[:, None])
        x_prop = xf * mask_f[:, None] + xb * mask_b[:, None]
        v_prop = vf * mask_f[:, None] + vb * mask_b[:, None]

        accept_prob = pxf * mask_f + pxb * mask_b
        sumlogdet_prop = sumlogdetf * mask_f + sumlogdetb * mask_b

        mask_a, mask_r = self._get_accept_masks(accept_prob)
        x_out = x_prop * mask_a[:, None] + x_init * mask_r[:, None]
        v_out = v_prop * mask_a[:, None] + v_init * mask_r[:, None]
        sumlogdet_out = sumlogdet_prop * mask_a

        outputs = {
            'x_init': x_init,
            'v_init': v_init,
            'x_proposed': x_prop,
            'v_proposed': v_prop,
            'x_out': x_out,
            'v_out': v_out,
            'xf': xf,
            'xb': xb,
            'accept_prob': accept_prob,
            'sumlogdet_proposed': sumlogdet_prop,
            'sumlogdet_out': sumlogdet_out,
            'mask_f': mask_f,
            'mask_b': mask_b
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
        mask, mask_inv = self._get_mask(step)
        sumlogdet = 0.

        vf1, logdet = self._update_v_forward(x, v, beta, t, net_weights)
        sumlogdet += logdet

        xf1, logdet = self._update_x_forward(x, vf1, t,
                                             net_weights,
                                             (mask, mask_inv))
        sumlogdet += logdet

        xf2, logdet = self._update_x_forward(xf1, vf1, t,
                                             net_weights,
                                             (mask_inv, mask))
        sumlogdet += logdet

        vf2, logdet = self._update_v_forward(xf2, vf1, beta, t, net_weights)
        sumlogdet += logdet

        return xf2, vf2, sumlogdet

    def _backward_lf(self, x, v, beta, step, net_weights):
        """One backward augmented leapfrog step."""
        step_r = self.num_steps - step - 1
        t = self._get_time(step_r, tile=x.shape[0])
        mask, mask_inv = self._get_mask(step_r)

        sumlogdet = 0.

        vb1, logdet = self._update_v_backward(x, v, beta, t, net_weights)
        sumlogdet += logdet

        xb1, logdet = self._update_x_backward(x, vb1, t,
                                              net_weights,
                                              (mask_inv, mask))
        sumlogdet += logdet

        xb2, logdet = self._update_x_backward(xb1, vb1, t,
                                              net_weights,
                                              (mask, mask_inv))
        sumlogdet += logdet

        vb2, logdet = self._update_v_backward(xb2, vb1, beta, t, net_weights)
        sumlogdet += logdet

        return xb2, vb2, sumlogdet

    def _update_v_forward(self, x, v, beta, t, net_weights):
        """Update v in the forward leapfrog step."""
        dU_dx = self.grad_potential(x, beta)
        Sv, Tv, Qv = self.vnet([x, dU_dx, t])

        transl = net_weights.v_translation * Tv
        scale = net_weights.v_scale * (0.5 * self.eps * Sv)
        transf = net_weights.v_transformation * (self.eps * Qv)

        exp_scale = np.exp(scale)
        exp_transf = np.exp(transf)

        vf = v * exp_scale - 0.5 * self.eps * (dU_dx * exp_transf + transl)
        logdet = np.sum(scale, axis=1)

        return vf, logdet

    def _update_x_forward(self, x, v, t, net_weights, masks):
        """Update x in the forward leapfrog step."""
        mask, mask_inv = masks
        Sx, Tx, Qx = self.xnet([v, mask * x, t])
        scale = net_weights.x_scale * (self.eps * Sx)
        transl = net_weights.x_translation * Tx
        transf = net_weights.x_transformation * (self.eps * Qx)

        y = x * np.exp(scale) + self.eps * (v * np.exp(transf) + transl)
        xf = mask * x + mask_inv * y
        logdet = np.sum(mask_inv * scale)

        return xf, logdet

    def _update_v_backward(self, x, v, beta, t, net_weights):
        """Update v in the backward lf step. Inverting the forward update."""
        dU_dx = self.grad_potential(x, beta)
        Sv, Tv, Qv = self.vnet([x, dU_dx, t])

        scale = net_weights.v_scale * (-0.5 * self.eps * Sv)
        transl = net_weights.v_translation * Tv
        transf = net_weights.v_transformation * (self.eps * Qv)

        exp_scale = np.exp(scale)
        exp_transf = np.exp(transf)

        half_eps = 0.5 * self.eps
        vb = exp_scale * (v + half_eps * (dU_dx * exp_transf + transl))
        logdet = np.sum(scale, axis=1)

        return vb, logdet

    def _update_x_backward(self, x, v, t, net_weights, masks):
        """Update x in the backward lf step. Inverting the forward update."""
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

    def _get_accept_masks(self, accept_prob):
        rand_unif = np.random.uniform(size=accept_prob.shape)
        accept_mask = np.array(accept_prob > rand_unif, dtype=NP_FLOAT)
        reject_mask = 1. - accept_mask

        return accept_mask, reject_mask

    def _get_direction_masks(self):
        rand_unif = np.random.uniform(size=(self.batch_size,))
        forward_mask = np.array(rand_unif > 0.5, dtype=NP_FLOAT)
        backward_mask = 1. - forward_mask

        return forward_mask, backward_mask

    def hmc_networks(self):
        """Build hmc networks that output all zeros from the S, T, Q fns."""
        xnet = lambda inputs: [
            np.zeros_like(inputs[0]) for _ in range(3)
        ]
        vnet = lambda inputs: [
            np.zeros_like(inputs[0]) for _ in range(3)
        ]

        return xnet, vnet

    def build_networks(self, weights):
        if 'xnet' in weights:
            xnet_weights = weights['xnet']
            if 'GenericNet' in xnet_weights:
                xnet_weights = xnet_weights['GenericNet']
        if 'vnet' in weights:
            vnet_weights = weights['vnet']
            if 'GenericNet' in vnet_weights:
                vnet_weights = vnet_weights['GenericNet']

        xnet = GenericNetNP(xnet_weights, name='xnet')
        vnet = GenericNetNP(vnet_weights, name='vnet')

        return xnet, vnet

    def build_masks(self, zero_masks=False):
        masks = []
        for _ in range(self.num_steps):
            idx = np.random.permutation(np.arange(self.x_dim))[:self.x_dim//2]
            mask = np.zeros((self.x_dim,))
            if not zero_masks:
                mask[idx] = 1.
            masks.append(mask[None, :])

        return masks

    def _get_mask(self, step):
        m = self.masks[step]
        return m, 1. - m

    def grad_potential(self, x, beta):
        if HAS_AUTOGRAD:
            grad_fn = elementwise_grad(self.potential_energy, 0)
            #  grad_fn = grad(self.potential_energy, 0)
        else:
            raise ModuleNotFoundError('Unable to load autodiff library. '
                                      'Exiting.')

        return grad_fn(x, beta)

    def potential_energy(self, x, beta):
        return beta * self.potential(x)

    def kinetic_energy(self, v):
        return 0.5 * np.sum(v ** 2, axis=1)

    def hamiltonian(self, x, v, beta):
        return (self.kinetic_energy(v) + self.potential_energy(x, beta))
