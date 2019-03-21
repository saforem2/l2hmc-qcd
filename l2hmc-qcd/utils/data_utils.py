import os
import pickle
import numpy as np
#  from .jackknife import block_resampling, jackknife_err
from sklearn.model_selection import KFold

def compute_ac_spectrum(samples_history, target_mean, target_var=None):
    """Compute autocorrelation spectrum using equation 15 from the L2HMC paper.
    
    ********** UNABLE TO GET WORKING CORRECTLY FOR U(1) GAUGE MODEL **********

    Args:
        samples_history: Numpy array of shape [T, B, D], where T is the total
            number of time steps, B is the batch size, and D is the
            dimensionality of sample space.
        target_mean: 1D Numpy array of the mean of target (true) distribution.
        target_covar: 2D Numpy array representing a symmetric matrix for
            variance.

    Returns:
        Autocorrelation spectrum, Numpy array of shape [T-1].
    """
    pass

def next_power_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def compute_autocorrelation_fn(data, norm=True, fft=False):
    """Compute the autocorrelation function of data.

    Args:
        norm (bool): Normalize autocorrelation function to have max val 1.
        fft (bool): Compute autocorr function using fft/ifft (faster if data is
            large)
    """
    data = np.atleast_1d(data)

    if len(data.shape) != 1:
        raise ValueError('Invalid dimensions for 1D autocorrelation function.')

    if not fft:
        acf = np.correlate(data, data, mode='full')
        if norm:
            acf /= acf[acf.argmax()]

        return acf[acf.size//2:]

    # if fft, compute autocorrelation function using FFT / IFFT
    n = next_power_two(len(data))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(data - np.mean(data), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(data)].real
    acf /= 4*n
    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


def block_resampling(data, num_blocks):
    """ Block-resample data to return num_blocks samples of original data. """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    num_samples = data.shape[0]

    if num_samples < 1:
        raise ValueError("Data must have at least one sample.")
    if num_blocks < 1:
        raise ValueError("Number of resampled blocks must be greater than or"
                         "equal to 1.")
    if num_samples < num_blocks:
        num_blocks = max(2, num_samples)

    kf = KFold(n_splits=num_blocks)
    resampled_data = []
    for i, j in kf.split(data):
        resampled_data.append(data[i])
    return resampled_data


def jackknife(x, func, num_blocks=100):
    """Jackknife estimate of the estimator function."""
    n = len(x)
    block_size = n // num_blocks
    idx = np.arange(0, n, block_size)
    return np.sum(func(x[idx!=i]) for i in range(n))/float(n)


def jackknife_var(x, func, num_blocks=100):
    """Jackknife estimate of the variance of the estimator function."""
    n = len(x)
    block_size = n // num_blocks
    idx = np.arange(0, n, block_size)
    j_est = jackknife(x, func)
    return (n - 1) / (n + 0.) * np.sum(
        func(x[idx!=i]) - j_est**2.0 for i in range(n)
    )


def jackknife_err(y_i, y_full, num_blocks):
    if isinstance(y_i, list):
        y_i = np.array(y_i)
    if isinstance(y_full, list):
        y_full = np.array(y_full)
    try:
        err = np.sqrt(np.sum((y_i - y_full)**2) / (num_blocks-1)*(num_blocks))
    except ValueError:
        print(f"y_i.shape: {y_i.shape}, y_full.shape: {y_full.shape}")
        raise
    return err


def calc_avg_vals_errors(data, num_blocks=100):
    """Calculate average values and errors of `data` using block jackknife
    resampling method.

    Args:
        data (array-like): Array containing data for which statistics are
            desired.
        num_blocks (int): Number of blocks to use for block jackknife
            resampling.
    Returns:
        avg_val: The block jackknifed average of `data`
        error: The standard error obtained from the block jaccknifed resampling
            of `data`.
    """
    arr = np.array(data)
    avg_val = np.mean(arr)
    avg_val_rs = []
    arr_rs = block_resampling(arr, num_blocks)
    for block in arr_rs:
        avg_val_rs.append(np.mean(block))
    error = jackknife_err(y_i=avg_val_rs,
                          y_full=avg_val,
                          num_blocks=num_blocks)
    return avg_val, error


def load_data(data_dir):
    """
    Load all data from `.npy` and `.pkl` files contained in data_dir into
    numpy arrays and dictionaries respectively.

    Args:
        data_dir (directory, str):
            Directory containing data to load.

    Returns:
        arrays_dict (dict):
            Dictionary containing data loaded from `.npy` files.
            keys (str): String containing the filename without extension. 
            values (np.ndarray): Array containing data contained in file.
        pkl_dict (dict):
            Dictionary containing data load from `.pkl` files.
            keys (str): String containing the filename without extension.
            values (dict): Dictionary loaded in from file.
    """
    if not data_dir.endswith('/'):
        data_dir += '/'

    files = os.listdir(data_dir)
    if files == []:
        print(f"data_dir is empty. exiting!")
        raise ValueError

    np_files = []
    pkl_files = []
    for file in files:
        if file.endswith('.npy'):
            np_files.append(data_dir + file)
        if file.endswith('.pkl'):
            pkl_files.append(data_dir + file)

    #  np_load = lambda file: np.load(data_dir + file)
    def get_name(file): 
        return file.split('/')[-1][:-4]

    arrays_dict = {}
    for file in np_files:
        key = file.split('/')[-1][:-4]
        #  key = get_name(file)
        arrays_dict[key] = np.load(file)

    pkl_dict = {}
    for file in pkl_files:
        key = get_name(file)
        with open(file, 'rb') as f:
            pkl_dict[key] = pickle.load(f)

    return arrays_dict, pkl_dict
