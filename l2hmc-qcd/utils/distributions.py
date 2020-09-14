"""
Example distributions.
"""
from __future__ import absolute_import, division, print_function

import collections

from typing import Callable, Optional

from scipy.stats import multivariate_normal, ortho_group
import tensorflow_probability as tfp

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from config import NP_FLOATS, TF_FLOATS


tfd = tfp.distributions

TF_FLOAT = TF_FLOATS[tf.keras.backend.floatx()]
NP_FLOAT = NP_FLOATS[tf.keras.backend.floatx()]

# pylint:disable=invalid-name
# pylint:disable=unused-argument


def w1(z):
    """Transformation."""
    return tf.math.sin(2. * np.pi * z[0] / 4.)


def w2(z):
    """Transformation."""
    return 3. * tf.exp(-0.5 * (((z[0] - 1.) / 0.6)) ** 2)


def w3(z):
    """Transformation."""
    return 3. * (1 + tf.exp(-(z[0] - 1.) / 0.3)) ** (-1)


def plot_samples2D(
        samples: np.ndarray,
        title: str = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs,
):
    """Plot collection of 2D samples.

    NOTE: **kwargs are passed to `ax.plot(...)`.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    _ = ax.plot(samples[:, 0], samples[:, 1], **kwargs)
    if title is not None:
        _ = ax.set_title(title, fontsize='x-large')

    return fig, ax


def meshgrid(x, y=None):
    """Create a mesgrid of dtype 'float32'."""
    if y is None:
        y = x

    [gx, gy] = np.meshgrid(x, y, indexing='ij')
    gx, gy = np.float32(gx), np.float32(gy)
    grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
    return grid.T.reshape(x.size, y.size, 2)


# pylint:disable=too-many-arguments
def contour_potential(
        potential_fn: Callable,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        xlim: Optional[float] = 5.,
        ylim: Optional[float] = 5.,
        cmap: Optional[str] = 'inferno',
        dtype: Optional[str] = 'float32'
):
    """Plot contours of `potential_fn`."""
    if isinstance(xlim, (tuple, list)):
        x0, x1 = xlim
    else:
        x0 = -xlim
        x1 = xlim
    if isinstance(ylim, (tuple, list)):
        y0, y1 = ylim
    else:
        y0 = -ylim
        y1 = ylim
    grid = np.mgrid[x0:x1:100j, y0:y1:100j]
    #  grid_2d = meshgrid(np.arange(x0, x1, 0.05), np.arange(y0, y1, 0.05))
    grid_2d = grid.reshape(2, -1).T
    cmap = plt.get_cmap(cmap)
    if ax is None:
        _, ax = plt.subplots()
    try:
        pdf1e = np.exp(-potential_fn(grid_2d))
    except Exception as e:
        pdf1e = np.exp(-potential_fn(tf.cast(grid_2d, dtype)))

    z = pdf1e.reshape(100, 100)
    _ = ax.contourf(grid[0], grid[1], z, cmap=cmap, levels=8)
    if title is not None:
        ax.set_title(title, fontsize='x-large')
    plt.tight_layout()

    return ax


def two_moons_potential(z):
    """two-moons like potential."""
    z = tf.transpose(z)
    term1 = 0.5 * ((tf.linalg.norm(z, axis=0) - 2.) / 0.4) ** 2
    logterm1 = tf.exp(-0.5 * ((z[0] - 2.) / 0.6) ** 2)
    logterm2 = tf.exp(-0.5 * ((z[0] + 2.) / 0.6) ** 2)
    output = term1 - tf.math.log(logterm1 + logterm2)

    return output


def sin_potential(z):
    """Sin-like potential."""
    z = tf.transpose(z)
    x, y = z
    return 0.5 * ((y - w1(z)) / 0.4) ** 2 + 0.1 * tf.math.abs(x)


def sin_potential1(z):
    """Modified sin potential."""
    z = tf.transpose(z)
    logterm1 = tf.math.exp(-0.5 * ((z[1] - w1(z)) / 0.35) ** 2)
    logterm2 = tf.math.exp(-0.5 * ((z[1] - w1(z) + w2(z)) / 0.35) ** 2)
    term3 = 0.1 * tf.math.abs(z[0])
    output = -1. * tf.math.log(logterm1 + logterm2) + term3

    return output


def sin_potential2(z):
    """Modified sin potential."""
    z = tf.transpose(z)
    logterm1 = tf.math.exp(-0.5 * ((z[1] - w1(z)) / 0.4) ** 2)
    logterm2 = tf.math.exp(-0.5 * ((z[1] - w1(z) + w3(z)) / 0.35) ** 2)
    term3 = 0.1 * tf.math.abs(z[0])
    output = -1. * tf.math.log(logterm1 + logterm2) + term3

    return output


def quadratic_gaussian(x, mu, S):
    """Simple quadratic Gaussian (normal) distribution."""
    x = tf.cast(x, dtype=TF_FLOAT)
    return tf.linalg.diag_part(0.5 * ((x - mu) @ S) @ tf.transpose((x - mu)))


def random_tilted_gaussian(dim, log_min=-2., log_max=2.):
    """Implements a randomly tilted Gaussian (Normal) distribution."""
    mu = np.zeros((dim,))
    R = ortho_group.rvs(dim)
    sigma = np.diag(
        np.exp(np.log(10.) * np.random.uniform(log_min, log_max, size=(dim,)))
    )
    S = R.T.dot(sigma).dot(R)
    return Gaussian(mu, S)


def gen_ring(r=1., var=1., nb_mixtures=2):
    """Generate a ring of Gaussian distributions."""
    base_points = []
    for t in range(nb_mixtures):
        c = np.cos(2 * np.pi * t / nb_mixtures)
        s = np.sin(2 * np.pi * t / nb_mixtures)
        base_points.append(np.array([r * c, r * s]))

    #  v = np.array(base_points)
    sigmas = [var * np.eye(2) for t in range(nb_mixtures)]
    pis = [1. / nb_mixtures] * nb_mixtures
    pis[0] += 1. - sum(pis)

    return GMM(base_points, sigmas, pis)


def distribution_arr(x_dim, num_distributions):
    """Create array describing likelihood of drawing from distributions."""
    if num_distributions > x_dim:
        pis = [1. / num_distributions] * num_distributions
        pis[0] += 1 - sum(pis)
    if x_dim == num_distributions:
        big_pi = round(1.0 / num_distributions, x_dim)
        pis = num_distributions * [big_pi]
    else:
        big_pi = (1.0 / num_distributions) - x_dim * 1e-16
        pis = num_distributions * [big_pi]
        small_pi = (1. - sum(pis)) / (x_dim - num_distributions)
        pis.extend((x_dim - num_distributions) * [small_pi])

    return np.array(pis, dtype=NP_FLOAT)


def ring_of_gaussians(num_modes, sigma, r=1.):
    """Create ring of Gaussians for GaussianMixtureModel.

    Args:
        num_modes (int): Number of modes for distribution.
        sigma (float): Standard deviation of each mode.
        x_dim (int): Spatial dimensionality in which the distribution exists.
        radius (float): Radius from the origin along which the modes are
            located.

    Returns:
        distribution (GMM object): Gaussian mixture distribution.
        mus (np.ndarray): Array of the means of the distribution.
        covs (np.array): Covariance matrices.
        distances (np.ndarray): Array of the differences between different
            modes.
    """
    distribution = gen_ring(r=r, var=sigma, nb_mixtures=num_modes)
    mus = np.array(distribution.mus)
    covs = np.array(distribution.sigmas)
    diffs = mus[1:] - mus[:-1, :]
    distances = [np.sqrt(np.dot(d, d.T)) for d in diffs]

    return distribution, mus, covs, distances


def lattice_of_gaussians(num_modes, sigma, x_dim=2, size=None):
    """Create lattice of Gaussians for GaussianMixtureModel.

    Args:
        num_modes (int): Number of modes for distribution.
        sigma (float): Standard deviation of each mode.
        x_dim (int): Spatial dimensionality in which the distribution exists.
        size (int): Spatial extent of lattice.

    Returns:
        distribution (GMM object): Gaussian mixture distribution.
        covs (np.array): Covariance matrices.
        mus (np.ndarray): Array of the means of the distribution.
        pis (np.ndarray): Array of relative probabilities for each mode. Must
            sum to 1.
    """
    if size is None:
        size = int(np.sqrt(num_modes))

    mus = np.array([(i, j) for i in range(size) for j in range(size)])
    covs = np.array([sigma * np.eye(x_dim) for _ in range(num_modes)])
    pis = [1. / num_modes] * num_modes
    pis[0] += 1. - sum(pis)

    distribution = GMM(mus, covs, pis)

    return distribution, mus, covs, pis


class Gaussian:
    """Implements a standard Gaussian distribution."""

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.i_sigma = np.linalg.inv(np.copy(sigma))

    def get_energy_function(self):
        """Return the potential energy function of the Gaussian."""
        def fn(x, *args, **kwargs):
            S = tf.constant(tf.cast(self.i_sigma, TF_FLOAT))
            mu = tf.constant(tf.cast(self.mu, TF_FLOAT))

            return quadratic_gaussian(x, mu, S)

        return fn

    def get_samples(self, n):
        """Get `n` samples from the distribution."""
        C = np.linalg.cholesky(self.sigma)
        x = np.random.randn(n, self.sigma.shape[0])
        return x.dot(C.T)

    def log_density(self, x):
        """Return the log_density of the distribution."""
        return multivariate_normal(mean=self.mu, cov=self.sigma).logpdf(x)


class TiltedGaussian(Gaussian):
    """Implements a tilted Gaussian."""

    def __init__(self, dim, log_min, log_max):
        self.R = ortho_group.rvs(dim)
        rand_unif = np.random.uniform(log_min, log_max, size=(dim,))
        self.diag = np.diag(np.exp(np.log(10.) * rand_unif))
        S = self.R.T.dot(self.diag).dot(self.R)
        self.dim = dim
        Gaussian.__init__(self, np.zeros((dim,)), S)

    def get_samples(self, n):
        """Get `n` samples from the distribution."""
        x = np.random.randn(200, self.dim)
        x = x.dot(np.sqrt(self.diag))
        x = x.dot(self.R)
        return x


class RoughWell:
    """Implements a rough well distribution."""

    def __init__(self, dim, eps, easy=False):
        self.dim = dim
        self.eps = eps
        self.easy = easy

    def get_energy_function(self):
        """Return the potential energy function of the distribution."""
        def fn(x, *args, **kwargs):
            n = tf.reduce_sum(tf.square(x), 1)
            eps2 = self.eps * self.eps
            if not self.easy:
                out = (0.5 * n
                       + self.eps * tf.reduce_sum(tf.math.cos(x/eps2), 1))
            else:
                out = (0.5 * n
                       + self.eps * tf.reduce_sum(tf.cos(x / self.eps), 1))
            return out

        return fn

    def get_samples(self, n):
        """Get `n` samples from the distribution."""
        # we can approximate by a gaussian for eps small enough
        return np.random.randn(n, self.dim)


class GaussianFunnel:
    """Gaussian funnel distribution."""
    def __init__(self, dim=2, clip=6., sigma=2.):
        self.dim = dim
        self.sigma = sigma
        self.clip = 4 * self.sigma

    def get_energy_function(self):
        """Returns the (potential) energy function of the distribution."""
        def fn(x):
            v = x[:, 0]
            log_p_v = tf.square(v / self.sigma)
            s = tf.exp(v)
            sum_sq = tf.reduce_sum(tf.square(x[:, 1:]), axis=1)
            n = tf.cast(tf.shape(x)[1] - 1, TF_FLOAT)
            E = 0.5 * (log_p_v + sum_sq / s + n * tf.math.log(2.0 * np.pi * s))
            s_min = tf.exp(-self.clip)
            s_max = tf.exp(self.clip)
            E_safe1 = 0.5 * (
                log_p_v + sum_sq / s_max + n * tf.math.log(2. * np.pi * s_max)
            )
            E_safe2 = 0.5 * (
                log_p_v + sum_sq / s_min + n * tf.math.log(2.0 * np.pi * s_min)
            )
            #  E_safe = tf.minimum(E_safe1, E_safe2)

            E_ = tf.where(tf.greater(v, self.clip), E_safe1, E)
            E_ = tf.where(tf.greater(-self.clip, v), E_safe2, E_)

            return E_
        return fn

    def get_samples(self, n):
        """Get `n` samples from the distribution."""
        samples = np.zeros((n, self.dim))
        for t in range(n):
            v = self.sigma * np.random.randn()
            s = np.exp(v / 2)
            samples[t, 0] = v
            samples[t, 1:] = s * np.random.randn(self.dim-1)

        return samples

    def log_density(self, x):
        """Return the log density of the distribution."""
        v = x[:, 0]
        log_p_v = np.square(v / self.sigma)
        s = np.exp(v)
        sum_sq = np.square(x[:, 1:]).sum(axis=1)
        n = tf.shape(x)[1] - 1
        return 0.5 * (
            log_p_v + sum_sq / s + (n / 2) * tf.math.log(2 * np.pi * s)
        )


class GMM:
    """Implements a Gaussian Mixutre Model distribution."""

    def __init__(self, mus, sigmas, pis):
        assert len(mus) == len(sigmas)
        assert sum(pis) == 1.0

        self.mus = mus
        self.sigmas = sigmas
        self.pis = pis

        self.nb_mixtures = len(pis)
        self.k = mus[0].shape[0]
        self.i_sigmas = []
        self.constants = []

        for i, sigma in enumerate(sigmas):
            self.i_sigmas.append(tf.cast(np.linalg.inv(sigma), TF_FLOAT))
            det = np.sqrt((2 * np.pi) ** self.k * np.linalg.det(sigma))
            det = tf.cast(det, TF_FLOAT)
            self.constants.append(tf.cast(pis[i] / det, dtype=TF_FLOAT))

    def get_energy_function(self):
        """Get the energy function of the distribution."""
        def fn(x):
            V = tf.concat([
                tf.expand_dims(-quadratic_gaussian(x, self.mus[i],
                                                   self.i_sigmas[i])
                               + tf.math.log(self.constants[i]), axis=1)
                for i in range(self.nb_mixtures)
            ], axis=1)

            return -1.0 * tf.math.reduce_logsumexp(V, axis=1)
        return fn

    def get_samples(self, n):
        """Get `n` samples from the distribution."""
        categorical = np.random.choice(self.nb_mixtures, size=(n,), p=self.pis)
        counter_samples = collections.Counter(categorical)
        samples = []
        for k, v in counter_samples.items():
            samples.append(
                np.random.multivariate_normal(
                    self.mus[k], self.sigmas[k], size=(v,)
                )
            )

        samples = np.concatenate(samples, axis=0)
        np.random.shuffle(samples)

        return samples

    def log_density(self, x):
        """Returns the log density of the distribution."""
        exp_arr = [
            self.pis[i] * multivariate_normal(
                mean=self.mus[i], cov=self.sigmas[i]
            ).pdf(x) for i in range(self.nb_mixtures)
        ]

        return np.log(sum(exp_arr))


class GaussianMixtureModel:
    """Gaussian mixture model, using tensorflow-probability."""
    def __init__(self, mus, sigmas, pis):
        self.mus = tf.convert_to_tensor(mus, dtype=tf.float32)
        self.sigmas = tf.convert_to_tensor(sigmas, dtype=tf.float32)
        self.pis = tf.convert_to_tensor(pis, dtype=tf.float32)

        self.dist = tfd.Mixture(
            cat=tfd.Categorical(probs=self.pis),
            components=[
                tfd.MultivariateNormalDiag(loc=m, scale_diag=s)
                for m, s in zip(self.mus, self.sigmas)
            ]
        )

    def get_energy_function(self):
        """Get the energy function (log probability) of the distribution."""
        def f(x):
            return -1 * self.dist.log_prob(x)
        return f

    def plot_contours(self, num_pts=500):
        """Plot contours of the target distribution."""
        grid = meshgrid(np.linspace(np.min(self.mus) - 1,
                                    np.max(self.mus) + 1,
                                    num_pts, dtype=np.float32))
        fig, ax = plt.subplots()
        ax.contour(grid[..., 0], grid[..., 1], self.dist.prob(grid))
        ax.set_title('Gaussian Mixture Model', fontsize='large')
        return fig, ax
