from __future__ import absolute_import, division, print_function

import os
import pickle

from config import HAS_HOROVOD, HAS_MATPLOTLIB
from update import set_precision
from plotters.plot_utils import plot_histogram, plot_acl_spectrum

import numpy as np
import tensorflow as tf

import scipy.stats as stats
from scipy.stats import sem

import utils.file_io as io

from utils.data_utils import block_resampling
from utils.distributions import GMM

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt

if HAS_HOROVOD:
    import horovod.tensorflow as hvd

if float(tf.__version__.split('.')[0]) <= 2:
    tf.logging.set_verbosity(tf.logging.INFO)


def create_config(params):
    """Helper method for creating a tf.ConfigProto object."""
    config = tf.ConfigProto(allow_soft_placement=True)

    if params.get('gpu', False):
        # Horovod: pin GPU to be used to process local rank 
        # (one GPU per process)
        config.gpu_options.allow_growth = True
        #  config.allow_soft_placement = True
        if HAS_HOROVOD and params['horovod']:
            config.gpu_options.visible_device_list = str(hvd.local_rank())

    if HAS_MATPLOTLIB:
        params['_plot'] = True

    if params['theta']:
        params['_plot'] = False
        io.log("Training on Theta @ ALCF...")
        params['data_format'] = 'channels_last'
        os.environ["KMP_BLOCKTIME"] = str(0)
        os.environ["KMP_AFFINITY"] = (
            "granularity=fine,verbose,compact,1,0"
        )
        # NOTE: KMP affinity taken care of by passing -cc depth to aprun call
        OMP_NUM_THREADS = 62
        config.allow_soft_placement = True
        config.intra_op_parallelism_threads = OMP_NUM_THREADS
        config.inter_op_parallelism_threads = 0

    return config, params


def parse_flags(FLAGS, log_file=None):
    """Parse command line FLAGS and create dictionary of parameters."""
    hmc = getattr(FLAGS, 'hmc', False)
    if not hmc:
        return

    try:
        kv_pairs = FLAGS.__dict__.items()
    except AttributeError:
        kv_pairs = FLAGS.items()

    params = {
        k: v for k, v in kv_pairs
    }

    params['log_dir'] = io.create_log_dir(FLAGS, log_file=log_file)
    params['summaries'] = not getattr(FLAGS, 'no_summaries', False)
    if 'no_summaries' in params:
        del params['no_summaries']

    save_steps = getattr(FLAGS, 'save_steps', None)
    train_steps = getattr(FLAGS, 'train_steps', 5000)
    if save_steps is None and train_steps is not None:
        params['train_steps'] = train_steps
        params['save_steps'] = train_steps // 4

    float64 = getattr(FLAGS, 'float64', False)
    if float64:
        io.log(f'INFO: Setting floating point precision to `float64`.')
        set_precision('float64')

    return params


def load_params(pkl_file=None, log_file=None):
    if pkl_file is None:
        alt_file = os.path.join(os.getcwd(), 'params.pkl')
        if os.path.isfile(alt_file):
            pkl_file = alt_file

    if not os.path.isfile(pkl_file):
        raise FileNotFoundError(
            'Unable to locate the `.pkl` file containing params. Exiting.'
        )

    with open(pkl_file, 'rb') as f:
        params = pickle.load(f)

    return params


def _load_distribution_params(_dir):
    """Load `mus`, `sigmas` and `pis` to reconstruct GMM distribution."""
    mus_file = os.path.join(_dir, 'mus.pkl')
    sigmas_file = os.path.join(_dir, 'sigmas.pkl')
    pis_file = os.path.join(_dir, 'pis.pkl')

    def _load(in_file):
        with open(in_file, 'rb') as f:
            tmp = pickle.load(f)
        return tmp

    mus = _load(mus_file)
    sigmas = _load(sigmas_file)
    pis = _load(pis_file)

    return mus, sigmas, pis


def recreate_distribution(_dir):
    mus, sigmas, pis = _load_distribution_params(_dir)
    return GMM(mus, sigmas, pis)


def autocovariance(x, tau=0):
    dT, dN, dX = np.shape(x)
    xs = np.mean(x)
    s = 0.
    for t in range(dT - tau):
        x1 = x[t, :, :] - xs
        x2 = x[t+tau, :, :] - xs
        s += np.sum(x1 * x2) / dN
    return s / (dT - tau)


def acl_spectrum(x):
    n = x.shape[0]
    autocovs = np.array([autocovariance(x, tau=t) for t in range(n-1)])
    autocovs /= np.max(autocovs)

    return autocovs


def ESS(A):
    A = A * (A > 0.05)
    return 1. / (1. + 2 * np.sum(A[1:]))


def _bootstrap_replicate_1d(data, func):
    """Generate a bootstrap replicate data."""
    bs_sample = np.random.choice(data, len(data))

    return bs_sample, func(bs_sample)


def bootstrap_replicates_1d(data, func, num_replicates=1000):
    return [_bootstrap_replicate_1d(data, func) for _ in range(num_replicates)]


def bootstrap_resample(x, n=None):
    """Use bootstrap resampling to obtain (re) sampled elements of `x`.

    Args:
        x (array-like): Data to resample.
        n (int, optional): Length of resampled array, equal to len(x) if n is
            None.

    Returns:
        x_rs: The resampled array.
    """
    if n is None:
        n = len(x)

    resample_i = np.floor(np.random.rand(n)*len(x)).astype(int)
    x_rs = x[resample_i]

    return x_rs


def error_analysis(samples, n=None, bs_iters=500):
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)

    samplesT = samples.transpose((-1, 1, 0))
    samples_rs = []
    means_arr = []
    #  errs_arr = []
    for component in samplesT:
        x_arr = []
        for _ in range(bs_iters):
            x_rs = [bootstrap_resample(x, n) for x in component]
            x_arr.append(x_rs)
        samples_rs.append(x_arr)
        x_arr = np.array(x_arr)
        means = np.mean(x_arr, axis=-1, dtype=np.float64)
        means_arr.append(means)
        #  samples_rs.append(x_rs)

    samples_rs = np.array(samples_rs)

    means = np.mean(means_arr, axis=(1, 2), dtype=np.float64)
    errs = np.std(means_arr, axis=(1, 2), dtype=np.float64)
    #  means_ = samples_rs.mean(axis=-1, dtype=np.float64)
    #  errs = np.std(means_, axis=-1, dtype=np.float64)
    #  errs = sem(means_, axis=-1)
    #  means = np.mean(means_, axis=-1, dtype=np.float64)

    return means, errs, means_arr, samples_rs


def alt_error_analysis(samples, num_blocks=None):
    """Calc the mean and std error using blocked jackknife resampling. 

    Args:
        samples (array-like): Array of samples on which the error analysis will
            be performed. 

                `samples.shape = (chain_length, batch_size, dim)`,

            where `dim` is the dimensionality of the target distribution.
        num_blocks (int): Number of blocks to use in blocked jackknife
            resampling.

    Returns:
        means (list): List containing the average value of each `component`
            in samples. For example, if the target distribution is
            two-dimensional, means would be `[x_avg, y_avg]`. 
        errs (list): List containing the standard error of each `component`
            in samples.

    NOTE:
        As an example, if `samples.shape = (1e4, 128, 2)` and 
        `num_blocks = None`, then `chain_length = 1e4`, `batch_size = 128` and
        `dim = 2`.

        We want to calculate the average and error for each component in `dim`,
        in this case we want `x_avg, x_err` and `y_avg, y_err`.

        This is done by first calculating the values `x_avg`, and `x_err` and
        then repeating the same calculation for `y_avg`, and `y_err`.

        For simplicity, we describe the calculation of `x_avg` and `x_err`.

        Let `samplesT = samples.transpose((-1, 1, 0))` with 
        `samplesT.shape = (2, 128, 1e4)`.

        Define `X, Y = samplesT` so that `X.shape = Y.shape = (128, 1e4)`.

        Then

        ```
        num_blocks = X.shape[1] // 50   # 1e4 // 50 = 200
        for x in X:
            x_rs = block_resampling(x, num_blocks)  # x_rs.shape = (200, 9950)
            x_rs_mean = x_rs.mean()
            means, errs = [], []
            for block in x_rs:
                block_mean = block.mean()
                denom = (num_blocks - 1) * num_blocks
                err = np.sqrt(np.sum((block_mean - x_rs_mean)**2) / denom)
                means.append(block_mean)
                errs.append(err)
        ```
    """
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)

    if num_blocks is None:
        num_blocks = samples.shape[0] // 50

    denom = (num_blocks - 1) * num_blocks

    def mean_(x):
        return np.mean(x, dtype=np.float64)

    def err_(x, xm):
        return np.sqrt(np.sum((mean_(x) - xm) ** 2) / denom)

    samplesT = samples.transpose((-1, 1, 0))
    # samplesT.shape = (dim, batch_size, chain_length)

    means_arr = []
    errs_arr = []
    for component in samplesT:  # loop over dimensionality (e.g. x, y, z)
        m_arr = []
        e_arr = []
        for x in component:  # loop over samples in batch
            x_rs = block_resampling(x, num_blocks)
            m_arr.extend([mean_(xb) for xb in x_rs])
            e_arr.extend([sem(xb) for xb in x_rs])
            #  m_rs = [mean_(xb) for xb in x_rs]
            #  e_rs = [err_(xb, x_rs_mean) for xb in x_rs]
        means_arr.append(m_arr)
        errs_arr.append(e_arr)

    means = np.array(means_arr).mean(axis=1, dtype=np.float64)
    errs = np.array(errs_arr).mean(axis=1, dtype=np.float64)

    return means, errs


def bootstrap(data, n_boot=10000, ci=68):
    boot_dist = []
    for i in range(int(n_boot)):
        resampler = np.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(np.mean(sample, axis=0))
    b = np.array(boot_dist)
    s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50. - ci / 2.)
    s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50. + ci / 2.)

    mean = np.mean(b)
    err = max(mean - s1.mean(), s2.mean() - mean)

    return mean, err, b


def write_means(samples, means, errs, means_file, tag=None):
    with open(means_file, 'a') as f:
        if tag is not None:
            _ = f.write('\n' + 80 * '-' + '\n')
            _ = f.write(tag + '\n' + '\n')

        _ = f.write(f'samples.shape: {samples.shape}\n')
        _ = f.write('Component averages:\n')
        for mean, err in zip(means, errs):
            _ = f.write(f'  {mean:.5g} +/- {err:.5g}\n')

        _ = f.write(80 * '-' + '\n')


def _pickle_dump(data, out_file, name=None):
    if name is None:
        name = 'data'

    io.log(f'Saving {name} to {out_file}')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


def save_inference_data(samples, px, run_dir, fig_dir,
                        acl=True, bs_iters=100, ignore_first=0.1):
    """Save inference data and estimate the 'average' for each coordinate.

    By comparing, for example, the average `x` location (`x_mean_obs`) across
    all `samples` and comparing this to the true value dictated from the target
    distribution (`x_mean_true`), we can try and determine if there is an
    inherent bias in the trained sampler.

    Args:
        samples (array-like): Array of sample configurations generated from an
            inference run.
        px (array-like): Array of acceptance probabilities generated from an
            inference run.
        run_dir (str): String specifying the location of the `run_dir` used for
            the inference run.
        fig_dir (str): String specifying the location of the `fig_dir`, i.e.
            where to save matplotlib figures.
        acl (bool): Boolean indicating whether or not to perform the
            `autocorrelation` analysis on samples
        bs_iters (int): Number of bootstrap replicates to use when estimating
            the standard error of the mean(s). Note that a larger value will
            produce better statistics (at a relatively steep computational
            cost).
        ignore_first (float): Percentage of total length of chain to 'ignore'
            when calculting statistics due to thermalization. For example, if
            `ignore_first = 0.1`, the first 10% of `samples` will be discarded
            to account for thermalization.

    Returns:
        None
    """
    if not isinstance(px, np.ndarray):
        px = np.array(px)
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)

    samples_out_file = os.path.join(run_dir, f'samples_out.pkl')
    px_out_file = os.path.join(run_dir, 'px_out.pkl')
    _pickle_dump(samples, samples_out_file, name='samples')
    _pickle_dump(px, px_out_file, name='probs')

    #  warmup_steps = samples.shape[0] // 20
    warmup_steps = int(ignore_first * samples.shape[0])
    samples_therm = samples[warmup_steps:]

    io.log(f'INFO: Ignoring first {warmup_steps} steps for thermalization...')
    #  io.log(f'INFO: Using (naive) `np.mean` and `np.std`:')
    x_means = samples_therm[:, :, 0].mean(axis=0)
    x_mean = np.mean(x_means)
    x_std = np.std(x_means)
    x_sem = sem(x_means)

    y_means = samples_therm[:, :, 1].mean(axis=0)
    y_mean = np.mean(y_means)
    y_std = np.std(y_means)
    y_sem = sem(y_means)
    #  x_mean, y_mean = samples_therm.mean(axis=(0, 1))
    #  x_std, y_std = np.std(samples_therm, axis=(0, 1))

    #  x_sem = sem(samples_therm[:, :, 0],)
    #  y_sem = sem(samples_therm[:, :, 1])
    #  x_std, y_std = samples[warmup_steps:].std(axis=(0, 1))

    means_naive = (x_mean, y_mean)
    stds_naive = (x_std, y_std)
    sems_naive = (x_sem, y_sem)
    std_str_naive = f'({x_std:.5g}, {y_std:.5g})'
    sem_str_naive = f'({x_sem:.5g}, {y_sem:.5g})'
    mean_str_naive = f'({x_mean:.5g}, {y_mean:.5g})'
    err_str_naive = f'{std_str_naive} (std),  {sem_str_naive} (sem)'

    '''
    means, errs, means_, samples_rs = error_analysis(samples[warmup_steps:],
                                                     n=None,
                                                     bs_iters=bs_iters)
    '''
    samples_x = samples_therm[:, :, 0]
    samples_y = samples_therm[:, :, 1]
    x_mean_bs, x_err_bs, x_means_arr = bootstrap(samples_x,
                                                 bs_iters, ci=68)
    y_mean_bs, y_err_bs, y_means_arr = bootstrap(samples_y,
                                                 bs_iters, ci=68)

    means = (x_mean_bs, y_mean_bs)
    errs = (x_err_bs, y_err_bs)
    means_arr = (x_means_arr, y_means_arr)

    # labels to be used in histogram plot
    label_strs = [f'mean: {i:.4g} +/- {j:.4g}' for i, j in zip(means, errs)]

    err_str = f'({errs[0]:.5g}, {errs[1]:.5g})'
    mean_str = f'({means[0]:.5g}, {means[1]:.5g})'

    means_strs = f'(x, y): {mean_str} +/- {err_str}'
    means_strs_naive = f'(x, y): {mean_str_naive} +/- {err_str_naive}'

    io.log(f'Using {bs_iters} bootstrap replications:\n  {means_strs}\n')
    io.log(f'Using np.mean and std/sem:\n  {means_strs_naive}\n')

    means_file = os.path.join(run_dir, 'means.txt')
    write_means(samples, means, errs, means_file)
    write_means(samples, means_naive,
                stds_naive, means_file,
                tag='std[i] = np.std(samples_therm[:, :, i].mean(axis=0))')
    write_means(samples, means_naive,
                sems_naive, means_file,
                tag=('sem[i] = scipy.stats.sem('
                     'samples_therm[:, :, i].mean(axis=0))'))

    if acl:
        spectrum = acl_spectrum(samples)
        ess = ESS(spectrum)

        spectrum_file = os.path.join(run_dir, 'acl_spectrum.pkl')
        ess_file = os.path.join(run_dir, 'ess.txt')
        with open(spectrum_file, 'wb') as f:
            pickle.dump(spectrum, f)

        with open(ess_file, 'w') as f:
            _ = f.write(f'ESS: {ess}')

    out_files = [os.path.join(fig_dir, 'x_mean_histogram.pdf'),
                 os.path.join(fig_dir, 'y_mean_histogram.pdf')]
    xlabels = ['mean, x', 'mean, y']
    hist_kwargs = {
        'bins': 32,
        'density': True,
        'stacked': True,
        'label': None,
        'out_file': None,
        'xlabel': None
    }
    if HAS_MATPLOTLIB:
        for idx, x in enumerate(means_arr):
            hist_kwargs.update({
                'label': label_strs[idx],
                'out_file': out_files[idx],
                'xlabel': xlabels[idx]
            })
            fig, ax = plt.subplots()
            _ = plot_histogram(x.flatten(), ax=ax, **hist_kwargs)

        if acl:
            acl_kwargs = {
                'out_file': os.path.join(fig_dir,
                                         'autocorrelation_spectrum.pdf')
            }
            fig, ax = plot_acl_spectrum(spectrum, **acl_kwargs)
