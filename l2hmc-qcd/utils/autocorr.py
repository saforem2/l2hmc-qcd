"""
Various helper functions for computing the autocorrelation and integrated
autocorrelation time from time series data.

Author: Sam Foreman (twitter/github @saforem2)

==============================================================================
* NOTE:
------------------------------------------------------------------------------
    * The most reliable combination of methods seems to be:
        - autocorr_func_1d: For obtaining the normalized autocorrelation
            function of a 1-D time series.
        - integrated_time: For obtaining the integrated autocorrelation time
            using the autocorrelation function generated using
            `autocorr_func_1d`.
==============================================================================
"""
# pylint:disable=invalid-name
import logging
import numpy as np


def autocorr_fast(X, kappa=500):
    """Compute the autocorrelation curve of X using FFT's."""
    # The autocorrelation has to be truncated at some point so there are enough
    # data points constructing each lag. Let kappa be the cutoff.
    X = X - np.mean(X)
    N = len(X)
    fvi = np.fft.fft(X, n=2*N)
    autocorr = np.real(np.fft.ifft(fvi * np.conjugate(fvi))[:N])
    autocorr /= N - np.arange(N)
    autocorr /= autocorr[0]
    autocorr = autocorr[:kappa]
    return autocorr

def autocorr(X):
    """Alternative method for calculating the autocorrelation spectrum."""
    result = np.correlate(X, X, mode='full')
    result /= result[result.argmax()]
    return result[result.size//2:]

def autocovariance(X, tau=0):
    """Compute the autocovariance of X[t] and X[t + tau]."""
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    dT, dN, dX = np.shape(X)

    s = 0.
    for t in range(dT - tau):
        x1 = X[t, :, :]
        x2 = X[t + tau, :, :]

        s += np.sum(x1 * x2) / dN

    return s / (dT - tau)

def acl_spectrum(X, scale):
    """Compute the autocorrelation spectrum of X."""
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    n = X.shape[0]
    return np.array([autocovariance(X / scale, tau=t) for t in range(n-1)])

def calc_ESS(acl_spectrum):
    """Compute the expected sample size from an autocorrelation spectrum."""
    acl_spectrum = acl_spectrum * (acl_spectrum > 0.05)

    return 1. / (1. + 2 * np.sum(acl_spectrum[1:]))

def calc_iat(X, kappa=500):
    """Calculates the autocorrelation curve and integrated autocorrelation time
    (IAC) of X."""
    # autocorr is the UNintegrated autocorrelation curve
    autocorr = autocorr_fast(X, kappa)
    tau = 1 + 2 * np.sum(autocorr)
    return tau, autocorr


###############################################################################
# NOTE: The methods below, namely the `integrated_time` method seems to give
# the most reliable estimate of the integrated autocorrelation time.
###############################################################################
def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`."""
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_gw2010(y, c=5.0):
    """Following the approach described in Goodman & Weare (2010)."""
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2. * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2. * np.cumsum(f) - 1.
    window = auto_window(taus, c)
    return taus[window]

def autocorr_func_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series.
    Args:
        x: The series as a 1-D numpy array.
    Returns:
        array: The autocorrelation function of the time series.
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4 * n
    acf /= acf[0]

    return acf

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def integrated_time(x, c=5, tol=50, quiet=False):
    """Estimate the integrated autocorrelation time of a time series.
    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ to
    determine a reasonable window size.

    Args:
        x: The time series. If multidimensional, set the time axis using the
            ``axis`` keyword argument and the function will be computed for
            every other axis.
        c (Optional[float]): The step size for the window search. (default:
            ``5``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust the estimate. (default: ``50``)
        quiet (Optional[bool]): This argument controls the behavior when the
            chain is too short. If ``True``, give a warning instead of raising
            an :class:`AutocorrError`. (default: ``False``)
    Returns:
        float or array: An estimate of the integrated autocorrelation time of
            the time series ``x`` computed along the axis ``axis``.
    Raises
        AutocorrError: If the autocorrelation time can't be reliably estimated
            from the chain and ``quiet`` is ``False``. This normally means
            that the chain is too short.
    """
    x = np.atleast_1d(x)
    if len(x.shape) == 1:
        x = x[:, np.newaxis, np.newaxis]
    if len(x.shape) == 2:
        x = x[:, :, np.newaxis]
    if len(x.shape) != 3:
        raise ValueError("invalid dimensions")

    n_t, n_w, n_d = x.shape
    tau_est = np.empty(n_d)
    windows = np.empty(n_d, dtype=int)

    # Loop over parameters
    for d in range(n_d):
        f = np.zeros(n_t)
        for k in range(n_w):
            f += autocorr_func_1d(x[:, k, d])
        f /= n_w
        taus = 2.0*np.cumsum(f)-1.0
        windows[d] = auto_window(taus, c)
        tau_est[d] = taus[windows[d]]

    # Check convergence
    flag = tol * tau_est > n_t

    _flag = None
    # Warn or raise in the case of non-convergence
    if np.any(flag):
        msg = (
            "The chain is shorter than {0} times the integrated "
            "autocorrelation time for {1} parameter(s). Use this estimate "
            "with caution and run a longer chain!\n"
        ).format(tol, np.sum(flag))
        msg += "N/{0} = {1:.0f};\ntau: {2}".format(tol, n_t/tol, tau_est)
        if not quiet:
            raise AutocorrError(tau_est, msg)
        else:
            _flag = 1
        #  logging.warning(msg)
        print(msg)

    return tau_est, _flag

class AutocorrError(Exception):
    """Raised if the chain is too short to estimate an autocorrelation time.
    The current estimate of the autocorrelation time can be accessed via the
    ``tau`` attribute of this exception.
    """
    def __init__(self, tau, *args, **kwargs):
        self.tau = tau

        super(AutocorrError, self).__init__(*args, **kwargs)
