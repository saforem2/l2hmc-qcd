"""
dynamics_np.py

Numpy implementation of the dynamics engine for the L2HMC sampler, allowing for
inference to be run on a trained model without the need for the `tf.Graph`
object.


Author: Sam Foreman (github: @saforem2)
Date: 01/07/2020
"""
from __future__ import absolute_import, division, print_function

from collections import namedtuple

from config import State, NP_FLOAT

from utils.file_io import timeit  # noqa: F401
from network.generic_net_np import GenericNetNP

HAS_AUTOGRAD = False
try:
    import autograd.numpy as np

    from autograd import elementwise_grad

    HAS_AUTOGRAD = True
except ImportError:
    import numpy as np

# pylint: disable=invalid-name, too-many-arguments
Weights = namedtuple('Weights', ['w', 'b'])

def reduced_weight_matrix(W, n=10):
    """Use the first n singular vals to reconstruct the original matrix W."""
    U, S, V = np.linalg.svd(W)
    W_ = np.matrix(U[:, :n]) * np.diag(S[:n]) * np.matrix(V[:n, :])

    return W_


def get_reduced_weights(weights, n=10):
    """Keep only the first `n` singular values for the weight matrices."""

# pylint: disable=too-many-instance-attributes,too-many-locals
# pylint: disable=useless-object-inheritance
class DynamicsNP(object):
    """Implements tools for running tensorflow-independent inference."""
    def __init__(self, potential_fn, weights, hmc=False, **params):
        self._model_type = params.get('model_type', None)
        self.potential = potential_fn
        if hmc:
            self.xnet, self.vnet = self.hmc_networks()
        else:
            self.xnet, self.vnet = self.build_networks(weights)

        self.hmc = hmc

        self.x_dim = params.get('x_dim', None)
        self._input_shape = params.get('_input_shape', None)
        self.batch_size = params.get('batch_size', None)
        self.eps = params.get('eps', None)
        self.x_dim = params.get('x_dim', None)
        self.num_steps = params.get('num_steps', None)
        self.zero_masks = params.get('zero_masks', False)
        self.masks = self.build_masks()
        self.direction = params.get('direction', 'rand')
        if self.direction == 'forward':
            self._forward = True
        elif self.direction == 'backward':
            self._forward = False
        else:
            self._forward = None

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
            x = np.mod(x, 2 * np.pi)

        if v is None:
            v = np.random.normal(size=x.shape)

        xf, vf, pxf, sumlogdetf = self.transition_kernel(*State(x, v, beta),
                                                         net_weights,
                                                         forward=True)
        mask_a, mask_r, rand_num = self._get_accept_masks(pxf)
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
            x = np.mod(x, 2 * np.pi)

        if v is None:
            v = np.random.normal(size=x.shape)

        xb, vb, pxb, sumlogdetb = self.transition_kernel(*State(x, v, beta),
                                                         net_weights,
                                                         forward=False)
        mask_a, mask_r, rand_num = self._get_accept_masks(pxb)
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

    def apply_transition(self, x, beta, net_weights, model_type=None):
        """Propose a new state and perform the accept/reject step."""
        forward = self._forward
        if forward is None:
            forward = (np.random.uniform() < 0.5)

        if model_type == 'GaugeModel':
            x = np.mod(x, 2 * np.pi)

        v_init = np.random.normal(size=x.shape)
        state_init = State(x, v_init, beta)
        x_, v_, px, sumlogdet = self.transition_kernel(*state_init,
                                                       net_weights,
                                                       forward=forward)
        mask_a, mask_r, rand_num = self._get_accept_masks(px)
        x_out = x_ * mask_a[:, None] + x * mask_r[:, None]
        v_out = v_ * mask_a[:, None] + v_init * mask_r[:, None]
        sumlogdet_out = sumlogdet * mask_a

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
        }

        return outputs

    def apply_transition_both(self, x, beta, net_weights, model_type=None):
        """Propose a new state and perform the accept/reject step."""
        if model_type == 'GaugeModel':
            x = np.mod(x, 2 * np.pi)

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

        # TODO: Instead of running forward and backward simultaneously,
        # use np.choose([-1, 1]) to determine which direction gets ran
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
            x = np.mod(x, 2 * np.pi)

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
            x = np.mod(x, 2 * np.pi)

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
        self.forward_mask = forward_mask
        self.backward_mask = 1. - forward_mask
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

    def _build_zero_masks(self):
        masks = []
        for  _ in range(self.num_steps):
            #  mask = np.zeros((self.x_dim,))
            mask = np.ones((self.x_dim,))
            masks.append(mask[None, :])

        return masks

    def build_masks(self):
        """Build `x` masks used for selecting which idxs of `x` get updated."""
        if self.zero_masks:
            masks = self._build_zero_masks()

        else:
            masks = []
            for _ in range(self.num_steps):
                _idx = np.arange(self.x_dim)
                idx = np.random.permutation(_idx)[:self.x_dim//2]
                mask = np.zeros((self.x_dim,))
                mask[idx] = 1.
                masks.append(mask[None, :])

        return masks

    def set_masks(self, masks):
        """Set `self.masks` to `masks`."""
        for idx, mask in enumerate(masks):
            print(f'Setting mask for {idx}...')
            self.masks[idx] = mask
        #  self.masks = masks

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
