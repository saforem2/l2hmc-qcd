"""
Augmented Hamiltonian Monte Carlo Sampler using the L2HMC algorithm, applied
to a U(1) lattice gauge theory model.

==============================================================================
* TODO:
-----------------------------------------------------------------------------
    * Look at thermalization times for L2HMC vs generic HMC.
    * Find out how large of a lattice is feasible for running on local laptop.

==============================================================================
* COMPLETED:
-----------------------------------------------------------------------------
==============================================================================
"""
from __future__ import absolute_import, division, print_function

# pylint: disable=no-member, too-many-arguments, invalid-name
import os
import sys
import time
import pickle

import numpy as np
import tensorflow as tf

try:
    import horovod.tensorflow as hvd

    HAS_HOROVOD = True

except ImportError:
    HAS_HOROVOD = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True

except ImportError:
    HAS_MATPLOTLIB = False

MAKE_SUMMARIES = False

import utils.file_io as io

from globals import (
    COLORS, FILE_PATH, GLOBAL_SEED, MARKERS, NP_FLOAT, PARAMS, TF_FLOAT
)
from collections import Counter, OrderedDict
from lattice.lattice import GaugeLattice, u1_plaq_exact
from dynamics.gauge_dynamics import GaugeDynamics
from utils.parse_args import parse_args
from utils.tf_logging import activation_summary, variable_summaries

from scipy.stats import sem
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorflow.core.protobuf import rewriter_config_pb2

if HAS_MATPLOTLIB:
    from utils.plot_helper import plot_multiple_lines

###########################################
# Set global seed for numpy and tensorflow
###########################################
np.random.seed(GLOBAL_SEED)

if '2.' not in tf.__version__:
    tf.set_random_seed(GLOBAL_SEED)
    tf.logging.set_verbosity(tf.logging.INFO)


##############################################################################
#                       Helper functions
##############################################################################
def check_else_make_dir(d):
    """Check if directory already exists, otherwise create it."""
    if HAS_HOROVOD:
        if hvd.rank() != 0:
            return
    if os.path.isdir(d):
        io.log(f"Directory {d} already exists... Continuing.")
    else:
        try:
            io.log(f"Creating directory {d}")
            os.makedirs(d)
        except OSError:
            raise


def tf_accept(x1, x2, px):
    """Helper function for determining if x is accepted given _x."""
    mask = (px - tf.random_uniform(tf.shape(px)) > 0.)
    return tf.where(mask, x2, x1)


def calc_fourier_coeffs(f, T, N, return_complex=False):
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:

        f(t) ~= a0/2+ sum_{k=1}^{N} (a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T))

    If return_complex is set to True, it returns instead the coefficients
    {c0,c1,c2,...}
    such that:

        f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)

    where we define c_{-n} = complex_conjugate(c_{n})

    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.

    Args:
        f: The periodic function, a callable.
        T: The period of the function f so that f(0) == f(T)
        N: The degree of the resulting approximation (number of Fourier
            coefficient) to return.
        return_complex: Return Fourier coefficients of complex representation
            of Fourier series. (Deafult: False)
    Returns:
        if return_complex:
            c (np.ndarray): Has shape = (N+1,), i.e. the first N+1 Fourier
                coefficients multiplying the complex exponential representation
                of the series.
        else:
            a0 (float): Coefficient of the 0th order term.
            a (np.ndarray): Has shape = (N+1,), i.e. the first N+1 Fourier
                coefficients multiplying the `cos` (even) part of the series.
            b (np.ndarray): Has shape = (N+1,), i.e. the first N+1 Fourier
                coefficients multiplying the `sin` (odd) part of the series.
    """
    # From Shanon theorem we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal.
    f_sample = 2 * N
    # we also need to use an integer sampling frequency, or the
    # points will not be equispaced between 0 and 1. We then add +2 to f_sample
    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.rfft(f(t)) / t.size

    if return_complex:
        return y

    y *= 2
    return y[0].real, y[1:-1].real, -y[1:-1].imag


def calc_fourier_series(a0, a, b, t, T):
    """Calculates the Fourier series with period T at times t.

    Example:
        f(t) ~ a0/2 + Σ_{k=1}^{N} [ a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) ]
    """
    p2 = 2 * np.pi
    tmp = np.ones_like(t) * a0 / 2.
    for k, (ak, bk) in enumerate(zip(a, b)):
        tmp += (ak * tf.cos(p2 * (k+1) * t / T)
                + bk * tf.sin(p2 * (k+1) * t / T))
    return tmp


def project_angle_slow(x, num_components=50):
    a0, a, b = calc_fourier_coeffs(project_angle, 2 * np.pi, num_components)
    y_fft = calc_fourier_series(a0, a, b, x, 2 * np.pi)
    return y_fft


def create_dynamics(samples, **kwargs):
    """Initialize dynamics object.

    Args:
        lattice: GaugeLattice object (defined in
            `lattice.lattice.GaugeLattice').
        samples (array-like): Array of samples, each sample representing a
            gauge configuration (of link variables).
        **kwargs (optional, dictionary): Keyword-arguments.
            'eps' (float): Step size to use in leapfrog integrator  .
            'hmc' (bool): Whether or not to use generic HMC.
            'network_arch':
    """
    batch_size = samples.shape[0]
    time_size = samples.shape[1]
    space_size = samples.shape[2]
    dim = len(samples.shape[2:])

    dynamics_kwargs = {
        'eps': kwargs.get('eps', 0.1),
        'hmc': kwargs.get('hmc', False),
        'network_arch': kwargs.get('network_arch', 'conv3d'),
        'num_steps': kwargs.get('num_steps', 5),
        'eps_trainable': kwargs.get('eps_trainable', True),
        'data_format': kwargs.get('data_format', 'channels_last'),
        'rand': kwargs.get('rand', False),
        'link_type': kwargs.get('link_type')
    }

    lattice = GaugeLattice(time_size=time_size,
                           space_size=space_size,
                           dim=dim,
                           link_type=dynamics_kwargs['link_type'],
                           num_samples=batch_size,
                           rand=dynamics_kwargs['rand'])

    potential_fn = lattice.get_potential_fn(samples)

    dynamics = GaugeDynamics(lattice=lattice,
                             potential_fn=potential_fn,
                             **dynamics_kwargs)

    return dynamics, potential_fn


# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes
class GaugeModel:
    """Wrapper class implementing L2HMC algorithm on lattice gauge models."""
    def __init__(self,
                 params=None,
                 sess=None,
                 config=None,
                 log_dir=None,
                 restore=False,
                 build_graph=True):
        """Initialization method."""
        np.random.seed(GLOBAL_SEED)
        tf.set_random_seed(GLOBAL_SEED)
        tf.enable_resource_variables()
        # ------------------------------------------------------------------
        # Create instance attributes from key, val pairs in `params`.
        # ------------------------------------------------------------------
        self._create_attrs(params)
        if not restore:
            #  self._create_attrs(params)
            # --------------------------------------------------------------
            # Create necessary directories for holding checkpoints, data, etc.
            # --------------------------------------------------------------
            if (self.using_hvd and hvd.rank() == 0) or not self.using_hvd:
                self.is_chief = True
                self._create_dir_structure(log_dir)
                # ---------------------------------------------------------
                # Write relevant instance attributes to .txt file.
                # ---------------------------------------------------------
                self._write_run_parameters(_print=True)
        else:
            self._create_dir_structure(log_dir, params, restore=True)

        # ------------------------------------------------------------------
        # Create lattice object.
        # ------------------------------------------------------------------
        with tf.name_scope('lattice'):
            self.lattice = self._create_lattice()

            #  self._create_tensors()
            self.batch_size = self.lattice.samples.shape[0]
            self.x_dim = self.lattice.num_links
            self.samples = tf.convert_to_tensor(self.lattice.samples,
                                                dtype=TF_FLOAT)
        # ------------------------------------------------------------------
        # Create placeholders for input data.
        # ------------------------------------------------------------------
        with tf.name_scope('inputs'):
            if not tf.executing_eagerly():
                self.x = tf.placeholder(dtype=TF_FLOAT,
                                        shape=(None, self.x_dim),
                                        name='x')

                self.beta = tf.placeholder(dtype=TF_FLOAT,
                                           shape=(),
                                           name='beta')
            else:
                self.x = self.lattice.samples
                self.beta = self.beta_init
        # ------------------------------------------------------------------
        # Create dynamics object responsible for performing L2HMC sampling.
        # ------------------------------------------------------------------
        with tf.name_scope('dynamics'):
            self.dynamics, self.potential_fn = self._create_dynamics(
                lattice=self.lattice,
                samples=self.samples,
                hmc=self.hmc,
                kwargs=None
            )
        # ------------------------------------------------------------------
        # Create metric function for measuring `distance` between configs.
        # ------------------------------------------------------------------
        with tf.name_scope('metric_fn'):
            self.metric_fn = self._create_metric_fn(self.metric)
        # ------------------------------------------------------------------
        # Create operations for calculating plaquette observables.
        # ------------------------------------------------------------------
        with tf.name_scope('plaq_observables'):
            with tf.name_scope('plaq_sums'):
                self.plaq_sums_op = self.lattice.calc_plaq_sums(self.x)
                #  self.plaq_sums_op = self._calc_plaq_sums(self.x)
            with tf.name_scope('actions'):
                self.actions_op = self.lattice.calc_actions(self.x)
                #  self.actions_op = self._calc_total_actions(self.x)
            with tf.name_scope('avg_plaqs'):
                self.plaqs_op = self.lattice.calc_plaqs(self.x)
                #  self.plaqs_op = self._calc_avg_plaqs(self.x)
                self.avg_plaq_op = tf.reduce_mean(self.plaqs_op,
                                                  name='avg_plaq')
            with tf.name_scope('top_charges'):
                self.charges_op = self.lattice.calc_top_charges(self.x,
                                                                fft=False)
                #  self.charges_op = self._calc_top_charges(self.x, fft=False)
        # ------------------------------------------------------------------
        # If restore, load from most recently saved checkpoint in `log_dir`.
        # ------------------------------------------------------------------
        if restore:
            self._restore_model(sess, config)
        # ------------------------------------------------------------------
        # Otherwise, build graph.
        # ------------------------------------------------------------------
        else:
            if not tf.executing_eagerly():
                if build_graph:
                    self.build_graph(sess, config)

    def _parse_params(self, params):
        """Parse key, value pairs from params and set class attributes."""
    # --------------------- LATTICE PARAMETERS -------------------------------
        self.time_size = params.get('time_size', 8)
        self.space_size = params.get('space_size', 8)
        self.link_type = params.get('link_type', 'U1')
        self.dim = params.get('dim', 2)
        self.num_samples = params.get('num_samples', 6)
        self.rand = params.get('rand', False)
    # --------------------- LEAPFROG PARAMETERS ------------------------------
        self.num_steps = params.get('num_steps', 5)
        self.eps = params.get('eps', 0.1)
        self.loss_scale = params.get('loss_scale', 1.)
    # --------------------- LEARNING RATE PARAMETERS -------------------------
        self.lr_init = params.get('lr_init', 1e-3)
        self.lr_decay_steps = params.get(
            'lr_decay_steps', 1000
        )
        self.lr_decay_rate = params.get(
            'lr_decay_rate', 0.98
        )
    # --------------------- ANNEALING RATE PARAMETERS ------------------------
        self.annealing = params.get('annealing', True)
        self.beta_init = params.get('beta_init', 2.)
        self.beta_final = params.get('beta_final', 4.)
    # --------------------- TRAINING PARAMETERS ------------------------------
        self.train_steps = params.get('train_steps', 5000)
        self.save_steps = params.get('save_steps', 1000)
        self.logging_steps = params.get('logging_steps', 50)
        self.print_steps = params.get('print_steps', 1)
        self.training_samples_steps = params.get(
            'training_samples_steps', 1000
        )
        self.training_samples_length = params.get(
            'training_samples_length', 100
        )
    # --------------------- MODEL PARAMETERS ---------------------------------
        self.network_arch = params.get('network_arch', 'conv3D')
        self.data_format = params.get('data_format', 'channels_last')
        self.hmc = params.get('hmc', False)
        self.eps_trainable = params.get('eps_trainable', True)
        self.metric = params.get('metric', 'cos_diff')
        #  self.aux = params.get('aux', True)
        self.std_weight = params.get('std_weight', 1.)
        self.aux_weight = params.get('aux_weight', 1.)
        self.charge_weight = params.get('charge_weight', 1.)
        self.summaries = params.get('summaries', False)
        self.clip_grads = params.get('clip_grads', False)
        self.clip_value = params.get('clip_value', 1.)
        self.using_hvd = params.get('using_hvd', False)

    def _create_attrs(self, params):
        """Parse key value pairs from params and set as class attributes."""
        if params is None:
            print('Using default parameters...')
            params = PARAMS
        else:
            self._parse_params(params)

        self.loss_weights = {}

        self.params = params

        for key, val in params.items():
            if 'weight' in key:
                self.loss_weights[key] = val
            else:
                setattr(self, key, val)

        if not self.clip_grads:
            self.clip_value = None

        #  self.training_samples_dict = {}
        self.losses_arr = []
        self.accept_prob_arr = []
        self.eps_arr = []
        self.charges_arr = []
        self.charges_dict = {}
        self.actions_dict = {}
        self.plaqs_dict = {}
        self.charge_diff_dict = {}
        self.train_data_dict = {
            'loss': {},
            'actions': {},
            'plaqs': {},
            'charges': {},
            'charge_diff': {},
            'accept_prob': {}
        }
        self._current_state = {
            'samples': None,
            'eps': self.eps,
            'step': 0,
            'beta': self.beta_init,
            'lr': self.lr_init
        }

        self.condition1 = not self.using_hvd  # condition1: NOT using horovod
        self.condition2 = False               # condition2: Initially False
        # If we're using horovod, we have     --------------------------------
        # to make sure all file IO is done    --------------------------------
        # only from rank 0                    --------------------------------
        if self.using_hvd:                    # If we are using horovod:
            if hvd.rank() == 0:               # AND rank == 0:
                self.condition2 = True        # condition2: True
        #  io.log('\n')
        #  io.log(80*'-')
        #  io.log(f'self.condition1: {self.condition1}')
        #  io.log(f'self.condition2: {self.condition2}')
        #  io.log(f'self.condition1 or self.condition2: '
        #         f'{self.condition1 or self.condition2}')
        #  io.log('\n')
        #  io.log(80*'-')

        self.safe_write = self.condition1 or self.condition2

    def _create_dir_structure(self, log_dir, restore=False):
        """Create self.files and directory structure."""
        if not self.is_chief:
            return

        project_dir = os.path.abspath(os.path.dirname(FILE_PATH))
        if log_dir is None:
            #  if (self.using_hvd and hvd.rank() == 0) or not self.using_hvd:
            if self.is_chief:
                root_log_dir = os.path.join(project_dir, 'gauge_logs_graph')
            else:
                return
        else:
            root_log_dir = os.path.join(project_dir, log_dir)

        check_else_make_dir(root_log_dir)

        if not self.using_hvd or (self.using_hvd and hvd.rank() == 0):
            if restore:
                self.log_dir = root_log_dir
            else:

                run_num = io.get_run_num(root_log_dir)
                log_dir = os.path.abspath(os.path.join(root_log_dir,
                                                       f'run_{run_num}'))
                check_else_make_dir(log_dir)

                self.log_dir = log_dir
                if self.using_hvd:
                    io.log('\n')
                    io.log(f"Successfully created and assigned "
                           f"`self.log_dir` on {hvd.rank()}.")
                    io.log(f"self.log_dir: {self.log_dir}")
                    io.log('\n')
        else:
            return

        self.info_dir = os.path.join(self.log_dir, 'run_info')
        self.figs_dir = os.path.join(self.log_dir, 'figures')
        self.eval_dir = os.path.join(self.log_dir, 'eval_info')
        self.samples_dir = os.path.join(self.eval_dir, 'samples')
        self.train_eval_dir = os.path.join(self.eval_dir, 'training')
        self.train_samples_dir = os.path.join(self.train_eval_dir, 'samples')
        self.obs_dir = os.path.join(self.log_dir, 'observables')
        self.train_obs_dir = os.path.join(self.obs_dir, 'training')
        self.ckpt_file = os.path.join(self.log_dir, 'gauge_model.ckpt')

        dirs = [self.info_dir, self.figs_dir, self.eval_dir, self.samples_dir,
                self.train_eval_dir, self.train_samples_dir, self.obs_dir,
                self.train_obs_dir]

        io.make_dirs(dirs)

        self.files = {
            'parameters_file': os.path.join(self.info_dir, 'parameters.txt'),
            'run_info_file': os.path.join(self.info_dir, 'run_info.txt'),
            'data_pkl_file': os.path.join(self.info_dir, 'data.pkl'),
            'samples_pkl_file': os.path.join(self.info_dir, 'samples.pkl'),
            'params_pkl_file': os.path.join(self.info_dir, 'parameters.pkl'),
            'current_state_file': os.path.join(
                self.info_dir, 'current_state.pkl'
            ),
            'train_data_file': os.path.join(
                self.info_dir, 'train_data.pkl'
            ),
            'actions_out_file': os.path.join(
                self.train_obs_dir, 'actions_training.pkl'
            ),
            'plaqs_out_file': os.path.join(
                self.train_obs_dir, 'plaqs_training.pkl'
            ),
            'charges_out_file': os.path.join(
                self.train_obs_dir, 'charges_training.pkl'
            ),
            'charge_diff_out_file': os.path.join(
                self.train_obs_dir, 'charge_diff_training.pkl'
            ),
        }

    def _restore_model(self, sess, config):
        """Restore model from previous run contained in `log_dir`."""
        #  if not hasattr(self, 'using_hvd'):
        #      self.using_hvd = params.get('using_hvd', False)
        #
        if self.using_hvd:
            if hvd.rank() != 0:
                return

        #  if not hasattr(self, 'hmc'):
        #      self.hmc = params.get('hmc', False)

        if self.hmc:
            io.log(f"ERROR: self.hmc: {self.hmc}. "
                   "No model to restore. Exiting.")
            sys.exit(1)

        assert os.path.isdir(self.log_dir), (f"log_dir: {self.log_dir} does "
                                             "not exist.")

        #  run_info_dir = os.path.join(self.log_dir, 'run_info')
        #  assert os.path.isdir(run_info_dir), (f"run_info_dir: {run_info_dir}"
        #                                       " does not exist.")

        with open(self.files['params_pkl_file'], 'rb') as f:
            self.params = pickle.load(f)

        with open(self.files['train_data_file'], 'rb') as f:
            self.train_data_dict = pickle.load(f)

        with open(self.files['current_state_file'], 'rb') as f:
            self._current_state = pickle.load(f)

        self._create_attrs(self.params)

        #  self.lr.assign(self._current_state['lr'])

        self.lattice = self._create_lattice()
        self.samples = tf.convert_to_tensor(self.lattice.samples,
                                            dtype=TF_FLOAT)

        kwargs = {
            'hmc': self.hmc,
            'eps': self._current_state['eps'],
            'network_arch': self.network_arch,
            'beta_init': self._current_state['beta'],
            'num_steps': self.num_steps,
            'eps_trainable': self.eps_trainable
        }
        self.dynamics, self.potential_fn = self._create_dynamics(self.lattice,
                                                                 self.samples,
                                                                 **kwargs)
        self._create_tensors()

        self.build_graph(sess, config)

        self.saver = tf.train.Saver(max_to_keep=3)

        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            io.log('Restoring previous model from: '
                   f'{ckpt.model_checkpoint_path}')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            io.log('Model restored.\n', nl=False)
            self.global_step = tf.train.get_global_step()

        sys.stdout.flush()
        return

    def _save_model(self, samples=None):
        """Save run `data` to `files` in `log_dir` using `checkpointer`"""
        if HAS_HOROVOD and self.using_hvd:
            if hvd.rank() != 0:
                return

        if samples is not None:
            with open(self.files['samples_pkl_file'], 'wb') as f:
                pickle.dump(samples, f)

        with open(self.files['train_data_file'], 'wb') as f:
            pickle.dump(self.train_data_dict, f)
        with open(self.files['current_state_file'], 'wb') as f:
            pickle.dump(self._current_state, f)

        #  if not tf.executing_eagerly():
        #      #  ckpt_prefix = os.path.join(self.log_dir, 'ckpt')
        #      io.log(f'INFO: Saving checkpoint to: {self.ckpt_file}')
        #      self.saver.save(self.sess,
        #                      self.ckpt_file,
        #                      global_step=self._current_state['step'])
        #      self.writer.flush()
        #  else:
        #      saved_path = self.checkpoint.save(
        #          file_prefix=os.path.join(self.log_dir, 'ckpt')
        #      )
        #      io.log(f"\n Saved checkpoint to: {saved_path}")

        if not self.hmc:
            if tf.executing_eagerly():
                self.dynamics.position_fn.save_weights(
                    os.path.join(self.log_dir, 'position_model_weights.h5')
                )
                self.dynamics.momentum_fn.save_weights(
                    os.path.join(self.log_dir, 'momentum_model_weights.h5')
                )

        self.writer.flush()

    def _write_run_parameters(self, _print=False):
        """Write model parameters out to human readable .txt file."""
        if not self.is_chief or self.info_dir is None:
            return

        s0 = 'Parameters'
        sep_str = 80 * '-'
        strings = []
        for key, val in self.params.items():
            strings.append(f'{key}: {val}')

        if _print:
            io.log(sep_str)
            for key, val in self.params.items():
                io.log(f'{key}: {val}')

        params_pkl_file = os.path.join(self.info_dir, 'parameters.pkl')
        io.log(f"Saving parameters to: {params_pkl_file}.")
        with open(params_pkl_file, 'wb') as f:
            pickle.dump(self.params, f)

        io.write(s0, self.files['parameters_file'], 'w')
        io.write(sep_str, self.files['parameters_file'], 'a')
        _ = [io.write(s, self.files['parameters_file'], 'a') for s in strings]
        io.write(sep_str, self.files['parameters_file'], 'a')

    def _create_lattice(self):
        """Create GaugeLattice object."""
        return GaugeLattice(time_size=self.time_size,
                            space_size=self.space_size,
                            dim=self.dim,
                            link_type=self.link_type,
                            num_samples=self.num_samples,
                            rand=self.rand)

    def _create_dynamics(self, lattice, samples, **kwargs):
        """Initialize dynamics object."""
        dynamics_kwargs = {  # default values of keyword arguments
            'eps': self.eps,
            'hmc': self.hmc,
            'network_arch': self.network_arch,
            'num_steps': self.num_steps,
            'eps_trainable': self.eps_trainable,
            'data_format': self.data_format,
            'use_bn': self.use_bn
        }

        dynamics_kwargs.update(kwargs)  # update dynamics_kwargs using kwargs

        potential_fn = lattice.get_potential_fn(samples)

        dynamics = GaugeDynamics(lattice=lattice,
                                 potential_fn=potential_fn,
                                 **dynamics_kwargs)  # updated default_kwargs

        return dynamics, potential_fn

    def _create_tensors(self):
        """Initialize tensors (and placeholders if executing in graph mode).

        NOTE: UNNECESSARY, TO BE REMOVED IN FUTURE UPDATE.
        """
        self.batch_size = self.lattice.samples.shape[0]
        self.x_dim = self.lattice.num_links
        self.samples = tf.convert_to_tensor(self.lattice.samples,
                                            dtype=TF_FLOAT)

        if not tf.executing_eagerly():
            self.x = tf.placeholder(TF_FLOAT, (None, self.x_dim), name='x')
            self.beta = tf.placeholder(TF_FLOAT, shape=(), name='beta')
        else:
            self.beta = self.beta_init

    @staticmethod
    def _create_metric_fn(metric):
        """Create metric fn for measuring the distance between two samples."""
        if metric == 'l1':
            def metric_fn(x1, x2):
                return tf.abs(x1 - x2)

        elif metric == 'l2':
            def metric_fn(x1, x2):
                return tf.square(x1 - x2)

        elif metric == 'cos':
            def metric_fn(x1, x2):
                return tf.abs(tf.cos(x1) - tf.cos(x2))

        elif metric == 'cos2':
            def metric_fn(x1, x2):
                return tf.square(tf.cos(x1) - tf.cos(x2))

        elif metric == 'cos_diff':
            def metric_fn(x1, x2):
                return 1. - tf.cos(x1 - x2)
        else:
            raise AttributeError(f"metric={metric}. Expected one of: 'l1', "
                                 f"'l2', 'cos', 'cos2', or 'cos_diff'.")

        return metric_fn

    def _calc_std_loss(self, x_tup, z_tup, p_tup, **weights):
        """Calculate standard contribution to loss.

        NOTE: In contrast to the original paper where the L2 difference was
        used, we are now using 1 - cos(x1 - x2).

        args:
            x_tup: Tuple of (x, x_proposed) configurations.
            z_tup: Tuple of (z, z_proposed) configurations.
            p_tup: Tuple of (x, z) acceptance probabilities.
            px: Acceptance probability of x_propsed given x.
            weights: dictionary of weights giving relative weight of each term
                in loss function.

        Returns:
            std_loss
        """
        eps = 1e-4
        aux_weight = weights.get('aux_weight', 1.)
        std_weight = weights.get('std_weight', 1.)

        x, x_proposed = x_tup
        z, z_proposed = z_tup
        px, pz = p_tup

        ls = self.loss_scale
        with tf.name_scope('std_loss'):
            with tf.name_scope('x_loss'):
                x_std_loss = tf.reduce_sum(self.metric_fn(x, x_proposed), 1)
                x_std_loss *= px
                x_std_loss = tf.add(x_std_loss, eps, name='x_std_loss')

            with tf.name_scope('z_loss'):
                z_std_loss = tf.reduce_sum(self.metric_fn(z, z_proposed), 1)
                z_std_loss *= pz * aux_weight
                z_std_loss = tf.add(z_std_loss, eps, name='z_std_loss')

            with tf.name_scope('tot_loss'):
                std_loss = (ls * (1. / x_std_loss + 1. / z_std_loss)
                            - (x_std_loss + z_std_loss) / ls) * std_weight

                std_loss = tf.reduce_mean(std_loss, axis=0, name='std_loss')

        tf.add_to_collection('losses', std_loss)
        return std_loss

    def _calc_charge_loss(self, x_tup, z_tup, p_tup, **weights):
        """Calculate contribution to total loss from charge difference.

        NOTE: This is an additional term introduced to the loss function that
        measures the difference in the topological charge between the initial
        configuration and the proposed configuration.
        """
        eps = 1e-4
        aux_weight = weights.get('aux_weight', 1.)
        charge_weight = weights.get('charge_weight', 1.)

        if charge_weight == 0:
            return 0.

        x, x_proposed = x_tup
        z, z_proposed = z_tup
        px, pz = p_tup

        ls = self.loss_scale
        with tf.name_scope('charge_loss'):
            with tf.name_scope('x_loss'):
                x_dq_fft = self.lattice.calc_top_charges_diff(x, x_proposed,
                                                              fft=True)
                xq_loss = px * x_dq_fft + eps

            with tf.name_scope('z_loss'):
                z_dq_fft = self.lattice.calc_top_charges_diff(z, z_proposed,
                                                              fft=True)
                zq_loss = aux_weight * (pz * z_dq_fft) + eps

            with tf.name_scope('tot_loss'):
                #  charge_loss = (ls * (1. / xq_loss + 1. / zq_loss)
                #                 - (xq_loss + zq_loss) / ls) * charge_weight
                charge_loss = ls * (charge_weight * (xq_loss + zq_loss))
                charge_loss = tf.reduce_mean(charge_loss, axis=0,
                                             name='charge_loss')

        tf.add_to_collection('losses', charge_loss)
        return charge_loss

    # pylint: disable=too-many-locals
    def _calc_loss(self, x, beta, **weights):
        """Create operation for calculating the loss.

        Args:
            x: Input tensor of shape (self.num_samples, self.lattice.num_links)
                containing batch of GaugeLattice links variables.
            beta (float): Inverse coupling strength.

        Returns:
            loss (float): Operation for calculating the total loss.
            px (array-like): Acceptance probabilities from Metropolis-Hastings
                accept/reject step. Has shape: (self.num_samples,)
            x_out: Output samples obtained after Metropolis-Hastings
                accept/reject step.

        NOTE:
            If proposed configuration is accepted following Metropolis-Hastings
            accept/reject step, x_ and x_out are equivalent.
        """
        with tf.name_scope('x_update'):
            x_proposed, _, px, x_out = self.dynamics(x, beta)
        with tf.name_scope('z_update'):
            z = tf.random_normal(tf.shape(x), name='z')  # Auxiliary variable
            z_proposed, _, pz, _ = self.dynamics(z, beta)

        with tf.name_scope('top_charge_diff'):
            x_dq = tf.cast(self.lattice.calc_top_charges_diff(x, x_out,
                                                              fft=False),
                           dtype=tf.int32)
            #  x_dq = tf.cast(self._calc_top_charges_diff(x, x_out, fft=False),
            #                 dtype=tf.int32)

        # Add eps for numerical stability; following released impl
        # NOTE:
        #  std_loss: "standard" loss
        #  charge_loss: loss contribution from the difference in top. charge
        x_tup = (x, x_proposed)
        z_tup = (z, z_proposed)
        p_tup = (px, pz)
        with tf.name_scope('calc_loss'):
            with tf.name_scope('std_loss'):
                std_loss = self._calc_std_loss(x_tup, z_tup, p_tup, **weights)
            with tf.name_scope('charge_loss'):
                charge_loss = self._calc_charge_loss(x_tup, z_tup, p_tup,
                                                     **weights)

            total_loss = tf.add(std_loss, charge_loss, name='total_loss')

        tf.add_to_collection('losses', total_loss)
        return total_loss, x_out, px, x_dq

    def _calc_loss_and_grads(self, x, beta, **weights):
        """Calculate loss and gradients.

        Args:
            x: Tensor object representing batch of GaugeLattice link variables.
            beta: Inverse coupling strength.

        Returns:
            loss: Operation for calculating the total loss.
            grads: Gradients from loss function.
            x_out: New samples obtained after Metropolis-Hastings accept/reject
                step.
            accept_prob: Acceptance probabilities used in Metropolis-Hastings
                accept/reject step.
        """
        #  with tf.name_scope('grads'):
        if tf.executing_eagerly():
            with tf.name_scope('grads'):
                with tf.GradientTape() as tape:
                    loss, x_out, accept_prob, x_dq = self._calc_loss(x, beta,
                                                                     **weights)
                grads = tape.gradient(loss, self.dynamics.trainable_variables)
        else:
            #  loss, x_out, accept_prob = self._calc_loss(x, beta, **weights)
            loss, x_out, accept_prob, x_dq = self._calc_loss(x, beta,
                                                             **weights)
            if self.summaries:
                loss_averages_op = self._add_loss_summaries(loss)
                control_deps = [loss_averages_op]
            else:
                control_deps = []

            with tf.name_scope('grads'):
                with tf.control_dependencies(control_deps):
                    grads = tf.gradients(loss,
                                         self.dynamics.trainable_variables)
                    if self.clip_grads:
                        grads, _ = tf.clip_by_global_norm(grads,
                                                          self.clip_value)

        return loss, grads, x_out, accept_prob, x_dq

    def _create_sampler(self):
        """Create operation for generating new samples using dynamics engine.

        NOTE: This method is to be used when running generic HMC to create
            operations for dealing with `dynamics.apply_transition` without
            building unnecessary operations for calculating loss.
        """
        with tf.name_scope('sampler'):
            _, _, px, self.x_out = self.dynamics(self.x, self.beta)
            self.px = px

            x_dq = self.lattice.calc_top_charges_diff(self.x, self.x_out,
                                                      fft=False)
            self.charge_diffs_op = tf.reduce_sum(x_dq) / self.num_samples

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in  GaugeModel.

        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.

        Args:
            total_loss: Total loss from self._calc_loss()
        Returns:
            loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total
        # loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss;
        # do the same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of
            # the loss as the original loss name.
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op

    def _add_activation_summaries(layers, name_scope):
        with tf.name_scope(name_scope):
            for layer in layers:
                activation_summary(layer)

    def _create_summaries(self):
        """Create summary objects for logging in TensorBoard."""
        if not self.is_chief:
            return

        ld = self.log_dir
        self.summary_writer = tf.contrib.summary.create_file_writer(ld)

        grads_and_vars = zip(self.grads, self.dynamics.trainable_variables)

        with tf.name_scope('loss'):
            tf.summary.scalar('loss', self.loss_op)

        with tf.name_scope('learning_rate'):
            tf.summary.scalar('learning_rate', self.lr)

        with tf.name_scope('step_size'):
            tf.summary.scalar('step_size', self.dynamics.eps)

        with tf.name_scope('tunneling_events'):
            tf.summary.scalar('tunneling_events_per_sample',
                              self.charge_diffs_op)

        with tf.name_scope('avg_plaq'):
            tf.summary.scalar('avg_plaq', self.avg_plaq_op)

        for var in tf.trainable_variables():
            if 'batch_normalization' not in var.op.name:
                tf.summary.histogram(var.op.name, var)

        #  with tf.name_scope('activations'):
        #      conv_layers = [self.dynamics.conv_x1, self.dynamics.conv_x2,
        #                     self.dynamics.conv_v1, self.dynamics.conv_v2]
        #      activation_summary(self.dynamics.conv_x1)
        #      activation_summary(conv)
        with tf.name_scope('summaries'):
            for grad, var in grads_and_vars:
                try:
                    layer, _type = var.name.split('/')[-2:]
                    name = layer + '_' + _type[:-2]
                except:
                    name = var.name[:-2]

                if 'batch_norm' not in name:
                    variable_summaries(var, name)
                    variable_summaries(grad, name + '/gradients')
                    tf.summary.histogram(name + '/gradients', grad)

        self.summary_op = tf.summary.merge_all(name='summary_op')

    def _log_write_graph_creation_time(self, **times):
        if not self.is_chief:
            return 0

        if self.files is None:
            return

        def log_and_write(s, f):
            """Print string `s` to std out and also write to file `f`."""
            io.log(s)
            io.write(s, f)
            return

        t_diff_loss = times.get('t_diff_loss', 0.)
        t_diff_train = times.get('t_diff_train', 0.)
        t_diff_summaries = times.get('t_diff_summaries', 0.)
        t_diff_graph = times.get('t_diff_graph', 0.)

        sep_str = 80 * '-'
        log_and_write(sep_str, self.files['run_info_file'])
        log_and_write(f"Building graph... (started at: {time.ctime()})",
                      self.files['run_info_file'])
        log_and_write("  Creating loss...", self.files['run_info_file'])
        log_and_write(f"    done. took: {t_diff_loss:4.3g} s.",
                      self.files['run_info_file'])
        log_and_write(f"  Creating gradient operations...",
                      self.files['run_info_file'])
        log_and_write(f"    done. took: {t_diff_train:4.3g} s",
                      self.files['run_info_file'])
        log_and_write("  Creating summaries...",
                      self.files['run_info_file'])
        log_and_write(f'    done. took: {t_diff_summaries:4.3g} s to create.',
                      self.files['run_info_file'])
        log_and_write(f'done. Graph took: {t_diff_graph:4.3g} s to build.',
                      self.files['run_info_file'])
        log_and_write(sep_str, self.files['run_info_file'])

    def create_opt(self, lr_init=None):
        """Create learning rate and optimizer."""
        if lr_init is None:
            lr_init = self.lr_init

        with tf.name_scope('global_step'):
            self.global_step = tf.train.get_or_create_global_step()
            self.global_step.assign(1)

        with tf.name_scope('learning_rate'):
            self.lr = tf.train.exponential_decay(lr_init,
                                                 self.global_step,
                                                 self.lr_decay_steps,
                                                 self.lr_decay_rate,
                                                 staircase=True,
                                                 name='learning_rate')
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            if self.using_hvd:
                self.optimizer = hvd.DistributedOptimizer(self.optimizer)

    def _create_hmc_sess(self, sess=None, config=None):
        """Create and initialize session objects for generic HMC sampler."""
        if self.hmc:
            if config is None:
                self.config = tf.ConfigProto()
            else:
                self.config = config
            if sess is None:
                self.sess = tf.Session(config=self.config)
            else:
                self.sess = sess
            if self.space_size > 8:
                off = rewriter_config_pb2.RewriterConfig.OFF
                graph_options = self.config.graph_options
                rewrite_options = graph_options.rewrite_options
                rewrite_options.arithmetic_optimization = off

            self._create_sampler()
            self.sess.run(tf.global_variables_initializer())
        else:
            raise AttributeError('_create_hmc_sess should only be run for '
                                 'generic HMC sampler.')

    def _create_sess(self, sess=None, config=None):
        """Create session objects for L2HMC sampler."""
        if config is None:
            self.config = tf.ConfigProto()
            if self.space_size > 8:
                off = rewriter_config_pb2.RewriterConfig.OFF
                graph_options = self.config.graph_options
                rewrite_options = graph_options.rewrite_options
                rewrite_options.arithmetic_optimization = off

        else:
            self.config = config

        if sess is None:
            if self.using_hvd:
                hooks = [
                    # Horovod: BroadcastGlobalVariablesHook broadcasts initial
                    # variable states from rank 0 to all other processes. This
                    # is necessary to ensure consistent initialization of all
                    # workers when training is started with random weights or
                    # restored from a checkpoint.
                    hvd.BroadcastGlobalVariablesHook(0),

                    # Horovod: adjust number of steps based on number of GPUs.
                    #  tf.train.StopAtStepHook(
                    #      last_step=self.train_steps // hvd.size()
                    #  ),

                    #  tf.train.LoggingTensorHook(tensors={'step': global_step,
                    #                                      'loss': loss},
                    #                             every_n_iter=10),
                ]
                checkpoint_dir = self.log_dir if hvd.rank() == 0 else None
            else:
                hooks = []
                checkpoint_dir = self.log_dir

            # The MonitoredTrainingSession takes care of session
            # initialization, restoring from a checkpoint, saving to a
            # checkpoint, and closing when done or an error occurs.

            #  self.sess = tf.Session(config=self.config)
            self.sess = tf.train.MonitoredTrainingSession(
                checkpoint_dir=checkpoint_dir,
                hooks=hooks,
                config=config,
                save_summaries_secs=None,
                save_summaries_steps=None
            )
        else:
            self.sess = sess

    # pylint:disable=too-many-statements
    def build_graph(self, sess=None, config=None):
        """Build graph for TensorFlow."""
        start_time = time.time()

        if self.hmc:  # if running generic HMC, all we need is the sampler
            self._create_hmc_sess(sess, config)
            return

        # Create global_step, learning_rate, optimizer
        self.create_opt()

        with tf.name_scope('loss'):
            t0_loss = time.time()
            output = self._calc_loss_and_grads(x=self.x, beta=self.beta,
                                               **self.loss_weights)
            self.loss_op, self.grads, self.x_out, self.px, x_dq = output
            self.charge_diffs_op = tf.reduce_sum(x_dq) / self.num_samples
            t_diff_loss = time.time() - t0_loss

        with tf.name_scope('train'):
            t0_train = time.time()
            grads_and_vars = zip(self.grads, self.dynamics.trainable_variables)
            self.train_op = self.optimizer.apply_gradients(
                grads_and_vars, global_step=self.global_step, name='train_op'
            )
            t_diff_train = time.time() - t0_train

        if self.summaries:
            t0_summaries = time.time()
            self._create_summaries()
            t_diff_summaries = time.time() - t0_summaries

        else:
            t_diff_summaries = 0.

        # Create and initialize session objects for L2HMC sampler.
        self._create_sess(sess, config)

        times = {
            't_diff_loss': t_diff_loss,
            't_diff_train': t_diff_train,
            't_diff_summaries': t_diff_summaries,
            't_diff_graph': time.time() - start_time
        }
        self._log_write_graph_creation_time(**times)

    def update_beta(self, step):
        """Returns new beta to follow annealing schedule."""
        temp = ((1. / self.beta_init - 1. / self.beta_final)
                * (1. - step / float(self.train_steps))
                + 1. / self.beta_final)
        new_beta = 1. / temp

        return new_beta

    def train_profiler(self, train_steps, **kwargs):
        """Wrapper around training loop for profiling graph execution."""
        builder = tf.profiler.ProfileOptionBuilder
        opt = builder(builder.time_and_memory()).order_by('micros').build()
        opt2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()

        # Collect traces of steps 10~20, dump the whole profile (with traces of
        # step 10~20) at step 20. The dumped profile can be used for further
        # profiling with command line interface or Web UI.
        profile_dir = os.path.join(self.log_dir, 'profile_info')
        io.check_else_make_dir(profile_dir)
        with tf.contrib.tfprof.ProfileContext(profile_dir,
                                              trace_steps=range(10, 20),
                                              dump_steps=[20]) as pctx:
            #  Run online profiling with 'op' view and 'opt' options at step
            #  15, 18, 20.
            pctx.add_auto_profiling('op', opt, [15, 18, 20])
            #  Run online profiling with 'scope' view and 'opt2' options at
            #  step 20.
            pctx.add_auto_profiling('scope', opt2, [20])
            # High level API, such as slim, Estimator, etc.
            kwargs['trace'] = True

            self.train(train_steps, **kwargs)

    def train_step(self, step, samples_np):
        """Perform a single training step.

        Args:
            step (int): Current training step.
            samples_np (np.ndarray): Array of input samples.

        Returns:
            outputs (tuple):
                Resulting output tuple consisting of:

                    (loss, samples_out, accept_prob, new_eps, avg_actions,
                    avg_plaqs, top_charges, new_lr, charge_diff)

                Where new eps is the new step size, top_charges is an array of
                the topological charge of each sample in samples, and new_lr is
                the updated learning rate.
        """
        start_time = time.time()
        beta_np = self.update_beta(step)

        fd = {
            self.x: samples_np,
            self.beta: beta_np
        }

        ops = [
            self.train_op,         # apply gradients
            self.loss_op,          # calculate loss
            self.x_out,            # get new samples
            self.px,               # calculate accept. prob
            self.dynamics.eps,     # evaluate current step size
            self.actions_op,       # calculate avg. actions
            self.plaqs_op,         # calculate avg. plaquettes
            self.charges_op,       # calculate top. charges
            self.lr,               # evaluate learning rate
            self.charge_diffs_op,  # change in top charge / num_samples
        ]

        outputs = self.sess.run(ops, feed_dict=fd)

        dt = time.time() - start_time

        data_str = (f"{step:>5g}/{self.train_steps:<6g} "
                    f"{outputs[1]:^9.4g} "              # loss value
                    f"{dt:^9.4g} "                      # time / step
                    f"{np.mean(outputs[3]):^9.4g}"      # accept prob
                    f"{outputs[4]:^9.4g} "              # step size
                    f"{beta_np:^9.4g} "                 # beta
                    f"{np.mean(outputs[5]):^9.4g} "     # avg. actions
                    f"{np.mean(outputs[6]):^9.4g} "     # avg. plaqs.
                    f"{u1_plaq_exact(beta_np):^9.4g} "  # exact plaq.
                    f"{outputs[9]:^9.4g} "              # charge diff
                    f"{outputs[8]:^9.4g}")              # learning rate

        return outputs, data_str

    def log_step(self, step, samples_np, beta_np, trace=False):
        """Perform a single logging step and update summaries.

        Args:
            step (int): Current training step.
            samples_np (np.ndarray): Array of samples.
            beta_np (float): Current beta value.
            trace (bool): Whether or not to trace inputs through graph (for
                optimization purposes).

        Returns:
            None
        """
        if not self.is_chief:
            return

        io.log(self.train_header)

        if trace:  # save the timeline to a chrome trace format
            options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE
            )
            run_metadata = tf.RunMetadata()

            fetched_tl = timeline.Timeline(run_metadata.step_stats)
            chrome_tr = fetched_tl.generate_chrome_trace_format()
            profile_dir = os.path.join(self.log_dir,
                                       'profile_info')
            io.check_else_make_dir(profile_dir)
            tl_file = os.path.join(profile_dir,
                                   f'timeline_{step}.json')
            with open(tl_file, 'w') as f:
                f.write(chrome_tr)

        else:
            options = None
            run_metadata = None

        if self.summaries:
            summary_str = self.sess.run(
                self.summary_op, feed_dict={
                    self.x: samples_np,
                    self.beta: beta_np,
                    #  self.lr: lr_np
                }, options=options, run_metadata=run_metadata
            )
            self.writer.add_summary(summary_str,
                                    global_step=step)
            if trace:
                tag = f'metadata_step_{step}'
                self.writer.add_run_metadata(run_metadata,
                                             tag=tag,
                                             global_step=step)
        self.writer.flush()

    # pylint: disable=too-many-statements, too-many-branches
    def train(self, train_steps, **kwargs):
        """Train the L2HMC sampler for `train_steps`.

        Args:
            train_steps: Integer specifying the number of training steps to
                perform.
            pre_train: Boolean that when True, creates `self.saver`, and
                `self.writer` objects and finalizes the graph to ensure no
                additional operations are created during training.
            trace: Boolean that when True performs a full trace of the training
                procedure.
            **kwargs: Dictionary of additional keyword arguments. Possible
                key, value pairs:
                    'samples_init': Array of samples used to start training.
                        (default: None)
                    'beta_init': Initial value of beta used in annealing
                        schedule. (Default: None)
                    'trace': Trace training loop for profiling. (Default:
                        False)
        """
        #  if self.using_hvd:
        #      self.train_steps = train_steps // hvd.size()

        samples_init = kwargs.get('samples_init', None)
        beta_init = kwargs.get('beta_init', None)
        trace = kwargs.get('trace', False)

        #  if self.condition1 or self.condition2:
        if self.is_chief:
            try:
                self.writer = tf.summary.FileWriter(self.log_dir,
                                                    self.sess.graph)
            except AttributeError:
                pass

        h_str = ("{:^12s}{:^10s}{:^10s}{:^10s}{:^10s}"
                 "{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}")
        h_strf = h_str.format("STEP", "LOSS", "t/STEP", "% ACC", "EPS",
                              "BETA", "ACTION", "PLAQ", "(EXACT)", "dQ", "LR")
        dash0 = (len(h_strf) + 1) * '-'
        dash1 = (len(h_strf) + 1) * '-'
        self.train_header = dash0 + '\n' + h_strf + '\n' + dash1
        self.sess.graph.finalize()

        start_time = time.time()

        if beta_init is None:
            beta_np = self.beta_init
        else:
            beta_np = beta_init

        if samples_init is None:
            samples_np = np.reshape(
                np.array(self.lattice.samples, dtype=NP_FLOAT),
                (self.num_samples, self.x_dim)
            )
        else:
            samples_np = samples_init
            assert samples_np.shape == self.x.shape

        initial_step = self._current_state['step']

        data_str = (f"{0:>5g}/{self.train_steps:<6g} "   # step / train_steps
                    f"{0.:^9.4g} "                       # loss
                    f"{0.:^9.4g} "                       # time / step
                    f"{0.:^9.4g} "                       # accept prob
                    f"{self.eps:^9.4g} "                 # initial eps
                    f"{self.beta_init:^9.4g} "           # initial beta
                    f"{0.:^9.4g} "                       # avg. action
                    f"{0.:^9.4g} "                       # avg. plaq
                    f"{u1_plaq_exact(beta_np):^9.4g} "   # exact plaq
                    f"{0.:^9.4g} "                       # tunneling events
                    f"{self.lr_init:^9.4g}")             # learning rate

        try:
            io.log(self.train_header)
            try:
                io.write(self.train_header, self.files['run_info_file'], 'a')
            except AttributeError:
                pass

            for step in range(initial_step, train_steps):
                outputs, data_str = self.train_step(step, samples_np)

                loss_np = outputs[1]
                samples_np = np.mod(outputs[2], 2 * np.pi)
                px_np = outputs[3]
                eps_np = outputs[4]
                actions_np = outputs[5]
                plaqs_np = outputs[6]
                charges_np = outputs[7]
                lr_np = outputs[8]
                charge_diff = outputs[9]

                self._current_state['samples'] = samples_np
                self._current_state['step'] = step
                self._current_state['beta'] = beta_np
                self._current_state['lr'] = lr_np
                self._current_state['eps'] = eps_np

                key = (step, beta_np)

                self.charges_dict[key] = charges_np
                self.charge_diff_dict[key] = charge_diff

                self.train_data_dict['loss'][key] = loss_np
                self.train_data_dict['actions'][key] = actions_np
                self.train_data_dict['plaqs'][key] = plaqs_np
                self.train_data_dict['charges'][key] = charges_np
                self.train_data_dict['charge_diff'][key] = charge_diff
                self.train_data_dict['accept_prob'][key] = px_np

                if step % self.print_steps == 0:
                    io.log(data_str)
                    try:
                        io.write(data_str, self.files['run_info_file'], 'a')
                    except AttributeError:
                        continue

                # Intermittently run sampler and save samples to pkl file.
                # We can calculate observables from these samples to
                # evaluate the samplers performance while we continue training.
                if (step + 1) % self.training_samples_steps == 0:
                    #  if self.condition1 or self.condition2:
                    if self.is_chief:
                        t0 = time.time()
                        io.log(80 * '-')
                        self.run(self.training_samples_length,
                                 current_step=step+1,
                                 beta=self.beta_final)
                        io.log(f"  done. took: {time.time() - t0}.")
                        io.log(80 * '-')
                        io.log(self.train_header)

                if (step + 1) % self.save_steps == 0:
                    #  if self.condition1 or self.condition2:
                    if self.is_chief:
                        self._save_model(samples=samples_np)

                if step % self.logging_steps == 0:
                    self.log_step(step, samples_np, beta_np, trace)

            step = self.sess.run(self.global_step)
            train_time = time.time() - start_time
            train_time_str = f"Time to complete training: {train_time:.3g}."
            io.log(train_time_str)
            #  if self.condition1 or self.condition2:
            if self.is_chief:
                try:
                    self._save_model(samples=samples_np)
                    self._plot_charge_diff()
                    io.write(train_time_str, self.files['run_info_file'], 'a')
                except AttributeError:
                    pass

        except (KeyboardInterrupt, SystemExit):
            io.log("\nKeyboardInterrupt detected! \n", nl=False)
            io.log("Saving current state and exiting.\n", nl=False)
            io.log(data_str)
            io.write(data_str, self.files['run_info_file'], 'a')
            #  if self.condition1 or self.condition2:
            if self.is_chief:
                try:
                    self._save_model(samples=samples_np)
                    self._plot_charge_diff()
                except AttributeError:
                    pass

    def run_step(self, step, run_steps, inputs):
        """Perform a single run step.

        Args:
            step (int): Current step
            run_steps (int): Total number of run steps to perform.
            inputs (tuple): Tuple consisting of (samples_np, beta_np, eps)
                where samples_np is the input batch of samples, beta_np is the
                input value of beta, and eps is the step size.

        Returns:
            outputs (tuple): Tuple of outputs consisting of (new samples,
                accept_prob, avg. actions, top. charges, charge differences).
        """
        samples_np, beta_np, eps, plaq_exact = inputs

        start_time = time.time()

        fd = {
            self.x: samples_np,
            self.beta: beta_np,
        }

        outputs = self.sess.run([
            self.x_out,
            self.px,
            self.actions_op,
            self.plaqs_op,
            self.charges_op,
            self.charge_diffs_op,
        ], feed_dict=fd)

        dt = (time.time() - start_time)  # / (norm_factor)

        eval_str = (f'{step:>5g}/{run_steps:<6g} '
                    f'{dt:^9.4g} '                      # time / step
                    f'{np.mean(outputs[1]):^9.4g} '     # accept. prob
                    f'{eps:^9.4g} '                     # step size
                    f'{beta_np:^9.4g} '                 # beta val
                    f'{np.mean(outputs[2]):^9.4g} '     # avg. actions
                    f'{np.mean(outputs[3]):^9.4g} '     # avg. plaquettes
                    f'{plaq_exact:^9.4g} '              # exact plaquette val
                    f'{outputs[5]:^9.4g} ')             # top. charge diff

        return outputs, eval_str

    # pylint: disable=inconsistent-return-statements, too-many-locals
    def run(self,
            run_steps,
            current_step=None,
            beta=None,
            therm_frac=10):
        """Run the simulation to generate samples and calculate observables.

        Args:
            run_steps: Number of steps to run the sampler for.
            ret: Boolean value indicating if the generated samples should be
                returned. If ret is False, the samples are saved to a `.pkl`
                file and then deleted.
            current_step: Integer passed when the sampler is ran intermittently
                during the training procedure, as a way to create unique file
                names each time the sampler is ran. By running the sampler
                during the training procedure, we are able to monitor the
                performance during training.
            beta: Float value indicating the inverse coupling constant that the
                sampler should be ran at.
        Returns:
            observables: Tuple of observables dictionaries containing
                (samples_history, actions_dict, plaqs_dict, charges_dict,
                charge_diff_dict).
        """
        if not self.is_chief:
            return

        if not isinstance(run_steps, int):
            run_steps = int(run_steps)

        if beta is None:
            beta = self.beta_final

        eps = self.sess.run(self.dynamics.eps)
        plaq_exact = u1_plaq_exact(beta)

        # start with randomly generated samples
        samples_np = np.random.randn(*(self.batch_size, self.x_dim))

        charges_arr = []
        eval_strings = []
        run_data = {
            'accept_probs': {},
            'actions': {},
            'plaqs': {},
            'charges': {},
            'charge_diffs': {},
        }
        #  actions_dict = {}
        #  plaqs_dict = {}
        #  charges_dict = {}
        #  charge_diff_dict = {}

        self.dynamics.trainable = False

        if current_step is None:
            txt_file = f'eval_info_steps_{run_steps}_beta_{beta}.txt'
            eval_file = os.path.join(self.eval_dir, txt_file)
        else:
            txt_file = (f'eval_info_{current_step}_TRAIN_'
                        f'steps_{run_steps}_beta_{beta}.txt')
            eval_file = os.path.join(self.train_eval_dir, txt_file)

        io.log(f"Running sampler for {run_steps} steps at beta = {beta}...")

        header = ("{:^12s}{:^10s}{:^10s}{:^10s}"
                  "{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}")
        header = header.format("STEP", "t/STEP", "% ACC", "EPS", "BETA",
                               "ACTIONS", "PLAQS", "(EXACT)", "dQ")
        dash0 = (len(header) + 1) * '='
        dash1 = (len(header) + 1) * '-'
        header_str = dash0 + '\n' + header + '\n' + dash1
        io.log(header_str)
        io.write(header_str, eval_file, 'a')

        start_time = time.time()
        try:
            for step in range(run_steps):
                inputs = (samples_np, beta, eps, plaq_exact)
                outputs, eval_str = self.run_step(step, run_steps, inputs)

                samples_np = np.mod(outputs[0], 2 * np.pi)
                px = outputs[1]
                actions_np = outputs[2]
                plaqs_np = outputs[3]
                charges_np = outputs[4]
                charge_diffs_np = outputs[5]

                key = (step, beta)
                run_data['accept_probs'][key] = px
                run_data['actions'][key] = actions_np
                run_data['plaqs'][key] = plaqs_np
                run_data['charges'][key] = charges_np
                run_data['charge_diffs'][key] = charge_diffs_np
                charges_arr.append(charges_np)
                #  actions_dict[key] = actions_np
                #  plaqs_dict[key] = plaqs_np
                #  charges_dict[key] = charges_np
                #  charge_diff_dict[key] = charge_diffs_np

                if step % self.print_steps == 0:
                    io.log(eval_str)

                eval_strings.append(eval_str)

                if step % 100 == 0:
                    io.log(header_str)

        except (KeyboardInterrupt, SystemExit):
            io.log("\nKeyboardInterrupt detected! \n", nl=False)
            io.log("Saving current state and exiting.\n", nl=False)

        #  import pdb
        #  pdb.set_trace()
        #  io.write(eval_strings, eval_file, 'a')
        _ = [io.write(s, eval_file, 'a') for s in eval_strings]
        stats = self.calc_observables_stats(run_data, therm_frac)

        charges_arr = np.array(charges_arr)
        _args = (run_steps, current_step, beta, therm_frac)

        self._save_run_info(run_data, stats, _args)
        self.make_plots(run_data, charges_arr, beta, current_step)
        io.log(f'\n Time to complete run: {time.time() - start_time} seconds.')
        io.log(80*'-' + '\n', nl=False)

        self.dynamics.trainable = True

        return run_data, stats

    def make_plots(self, run_data, charges_arr, beta, current_step):
        """Wrapper function that calls all individual plotting functions."""
        t0 = time.time()
        if HAS_MATPLOTLIB:
            self._plot_charge_diff()
            self._plot_observables(run_data, beta, current_step)
            self._plot_top_charges(charges_arr, beta, current_step)
            self._plot_top_charge_probs(charges_arr, beta, current_step)
        io.log(f'\n Took {time.time() - t0:4.3g} s to create and save plots.')

    # pylint:disable=no-self-use
    def calc_observables_stats(self, run_data, therm_frac=10):
        """Calculate statistics from `data`.

        Args:
            observables: Tuple of dictionaries containing (samples_history,
            actions_dict, plaqs_dict, charges_dict, charge_diff_dict).
            therm_frac: Fraction of data to ignore due to thermalization.

        Returns:
            stats: Tuple containing (actions_stats, plaqs_stats, charges_stats,
                charge_probabilities). Where actions_stats, plaqs_stats, and
                charges_stats are tuples of the form (avg_val, stderr), and
                charge_probabilities is a dictionary of the form
                {charge_val: charge_val_frequency}.
        """
        #  samples_history = observables[0]
        actions_dict = run_data['actions']
        plaqs_dict = run_data['plaqs']
        charges_dict = run_data['charges']

        actions_arr = np.array(list(actions_dict.values()))
        plaqs_arr = np.array(list(plaqs_dict.values()))
        charges_arr = np.array(list(charges_dict.values()), dtype=int)
        suscept_arr = np.array(charges_arr ** 2)

        num_steps = actions_arr.shape[0]
        therm_steps = num_steps // therm_frac

        actions_arr = actions_arr[therm_steps:, :]
        plaqs_arr = plaqs_arr[therm_steps:, :]
        charges_arr = charges_arr[therm_steps:, :]

        actions_mean = np.mean(actions_arr, axis=0)
        plaqs_mean = np.mean(plaqs_arr, axis=0)
        charges_mean = np.mean(charges_arr, axis=0)
        suscept_mean = np.mean(suscept_arr, axis=0)

        actions_err = sem(actions_arr)  # using scipy.stats.sem
        plaqs_err = sem(plaqs_arr)
        charges_err = sem(charges_arr)
        suscept_err = sem(suscept_arr)

        charge_probabilities = {}
        counts = Counter(list(charges_arr.flatten()))
        total_counts = np.sum(list(counts.values()))
        for key, val in counts.items():
            charge_probabilities[key] = val / total_counts

        charge_probabilities = OrderedDict(sorted(charge_probabilities.items(),
                                                  key=lambda k: k[0]))

        actions_stats = (actions_mean, actions_err)
        plaqs_stats = (plaqs_mean, plaqs_err)
        charges_stats = (charges_mean, charges_err)
        suscept_stats = (suscept_mean, suscept_err)

        stats = {
            'actions_stats': actions_stats,
            'plaqs_stats': plaqs_stats,
            'charges_stats': charges_stats,
            'suscept_stats': suscept_stats,
            'charge_probs': charge_probabilities
        }

        return stats

    def _plot_observables(self, run_data, beta, current_step=None):
        """Plot observables stored in `observables`."""
        if not HAS_MATPLOTLIB or (self.using_hvd and hvd.rank() != 0):
            return

        io.log("Plotting observables...")
        #  samples_history = observables[0]

        actions_arr = np.array(list(run_data['actions'].values()))
        plaqs_arr = np.array(list(run_data['plaqs'].values()))
        charges_arr = np.array(list(run_data['charges'].values()))
        charge_diff_arr = np.array(list(run_data['charge_diffs'].values()))

        num_steps = actions_arr.shape[0]
        steps_arr = np.arange(num_steps)

        out_dir, key = self._get_plot_dir(charges_arr, beta, current_step)

        title_str = (r"$\beta = $"
                     f"{beta}, {num_steps} {key} steps, "
                     f"{self.num_samples} samples")
        actions_plt_file = os.path.join(out_dir, 'total_actions_vs_step.png')
        plaqs_plt_file = os.path.join(out_dir, 'avg_plaquettes_vs_step.png')
        charges_plt_file = os.path.join(out_dir, 'top_charges_vs_step.png')
        charge_diff_plt_file = os.path.join(out_dir, 'charge_diff_vs_step.png')

        ######################
        # Total actions plots
        ######################
        kwargs = {
            'out_file': actions_plt_file,
            'markers': False,
            'lines': True,
            'alpha': 0.6,
            'title': title_str,
            'legend': False,
            'ret': False,
        }
        plot_multiple_lines(steps_arr, actions_arr.T, x_label='Step',
                            y_label='Total action', **kwargs)

        ###########################
        # Average plaquettes plots
        ###########################
        kwargs['out_file'] = None
        kwargs['ret'] = True
        _, ax = plot_multiple_lines(steps_arr, plaqs_arr.T, x_label='Step',
                                    y_label='Avg. plaquette', **kwargs)

        _ = ax.axhline(y=u1_plaq_exact(beta),
                       color='#CC0033', ls='-', lw=2.5, label='exact')

        _ = ax.plot(steps_arr, plaqs_arr.T.mean(axis=0),
                    color='k', label='average', alpha=0.75)
        plt.tight_layout()

        plt.savefig(plaqs_plt_file, dpi=200, bbox_inches='tight')

        ###########################
        # Topological charge plots
        ###########################
        kwargs['out_file'] = charges_plt_file
        kwargs['markers'] = True
        kwargs['lines'] = False
        kwargs['alpha'] = 1.
        kwargs['ret'] = False
        plot_multiple_lines(steps_arr, charges_arr.T, x_label='Step',
                            y_label='Topological charge', **kwargs)

        ##################################################
        # Tunneling events (change in top. charge) plots
        ##################################################
        _, ax = plt.subplots()
        ax.plot(steps_arr, charge_diff_arr,
                marker='.', ls='', fillstyle='none', color='C0')
        ax.set_xlabel('Steps', fontsize=14)
        ax.set_ylabel('Number of tunneling events', fontsize=14)
        ax.set_title(title_str, fontsize=16)
        io.log(f"Saving figure to: {charge_diff_plt_file}")
        plt.tight_layout()
        plt.savefig(charge_diff_plt_file, dpi=200, bbox_inches='tight')
        io.log('done.')

        return 1

    def _plot_charge_diff(self):
        """Plot num. of tunneling events vs. training step after training."""
        if not HAS_MATPLOTLIB or (self.using_hvd and hvd.rank() != 0):
            return
        #  charge_diff_keys = np.array(list(self.tunn_events_dict.keys()))
        charge_diff_vals = np.array(list(self.charge_diff_dict.values()))
        _, ax = plt.subplots()
        ax.plot(charge_diff_vals, marker='o', ls='', alpha=0.6,
                label=f'total across {self.num_samples} samples')
        ax.set_xlabel('Training step', fontsize=14)
        ax.set_ylabel(r"$\delta Q_{\mathrm{top}} / N$", fontsize=14)
        ax.set_title(rf"""$N =$ {self.num_samples} samples / batch""",
                     fontsize=16)

        #  title_str = (f'Number of tunneling events vs. '
        #               f'training step for {self.num_samples} samples')
        #  ax.set_title(title_str, fontsize=16)
        plt.tight_layout()
        out_file = os.path.join(self.figs_dir,
                                'tunneling_events_vs_training_step.png')
        io.log(f"Saving figure to: {out_file}.")
        plt.savefig(out_file, dpi=200, bbox_inches='tight')

    def _plot_top_charges(self, charges, beta, current_step=None):
        """Plot top. charge history using samples generated from `self.run`."""
        if not HAS_MATPLOTLIB or (self.using_hvd and hvd.rank() != 0):
            return

        io.log("Plotting topological charge vs. step for each sample...")

        out_dir, key = self._get_plot_dir(charges, beta, current_step)
        io.check_else_make_dir(out_dir)
        out_dir = os.path.join(out_dir, 'top_charge_plots')
        io.check_else_make_dir(out_dir)

        run_steps = charges.shape[0]
        title_str = (r"$\beta = $"
                     f"{beta}, {run_steps} {key} steps")

        t0 = time.time()
        # if we have more than 10 samples per batch, only plot first 10
        for idx in range(min(self.num_samples, 5)):
            _, ax = plt.subplots()
            _ = ax.plot(charges[:, idx],
                        marker=MARKERS[idx],
                        color=COLORS[idx],
                        ls='',
                        #  fillstyle='none',
                        alpha=0.6,
                        label=f'sample {idx}')
            _ = ax.legend(loc='best')
            _ = ax.set_xlabel('Step', fontsize=14)
            _ = ax.set_ylabel('Topological charge', fontsize=14)
            _ = ax.set_title(title_str, fontsize=16)
            plt.tight_layout()
            out_file = os.path.join(out_dir, f'top_charge_vs_step_{idx}.png')
            io.log(f"  Saving top. charge plot to {out_file}.")
            plt.savefig(out_file, dpi=200, bbox_inches='tight')

        io.log(f'done. took: {time.time() - t0:.4g}')
        plt.close('all')

    def _plot_top_charge_probs(self, charges, beta, current_step=None):
        """Create scatter plot of frequency of topological charge values."""
        if not HAS_MATPLOTLIB or (self.using_hvd and hvd.rank() != 0):
            return

        io.log("Plotting top. charge probability vs. value")
        out_dir, key = self._get_plot_dir(charges, beta, current_step)
        io.check_else_make_dir(out_dir)
        out_dir = os.path.join(out_dir, 'top_charge_probs')
        io.check_else_make_dir(out_dir)

        run_steps = charges.shape[0]
        title_str = (r"$\beta = $"
                     f"{beta}, {run_steps} {key} steps")

        charges = np.array(charges, dtype=int)
        # if we have more than 10 samples per batch, only plot first 10
        for idx in range(min(self.num_samples, 5)):
            counts = Counter(charges[:, idx])
            total_counts = np.sum(list(counts.values()))
            _, ax = plt.subplots()
            ax.plot(list(counts.keys()),
                    np.array(list(counts.values()) / total_counts),
                    marker=MARKERS[idx],
                    color=COLORS[idx],
                    ls='',
                    label=f'sample {idx}')
            _ = ax.legend(loc='best')
            _ = ax.set_xlabel('Topological charge', fontsize=14)
            _ = ax.set_ylabel('Probability', fontsize=14)
            _ = ax.set_title(title_str, fontsize=16)
            plt.tight_layout()
            out_file = os.path.join(out_dir,
                                    f'top_charge_prob_vs_val_{idx}.png')
            io.log(f'Saving figure to: {out_file}.')
            plt.savefig(out_file, dpi=200, bbox_inches='tight')#, rasterize=True)
            #  _ = plt.savefig(out_file, dpi=400, bbox_inches='tight')
            plt.close('all')

        all_counts = Counter(list(charges.flatten()))
        total_counts = np.sum(list(counts.values()))
        _, ax = plt.subplots()
        ax.plot(list(all_counts.keys()),
                np.array(list(all_counts.values()) / (total_counts *
                                                      self.num_samples)),
                marker='o',
                color='C0',
                ls='',
                alpha=0.6,
                label=f'total across {self.num_samples} samples')
        _ = ax.legend(loc='best')
        _ = ax.set_xlabel('Topological charge', fontsize=14)
        _ = ax.set_ylabel('Probability', fontsize=14)
        #  _ = ax.set_title(title_str, fontsize=16)
        out_file = os.path.join(out_dir,
                                f'TOP_CHARGE_FREQUENCY_VS_VAL_TOTAL.png')
        io.log(f'Saving figure to: {out_file}.')
        plt.savefig(out_file, dpi=200, bbox_inches='tight')#, rasterize=True)
        #  _ = plt.savefig(out_file, dpi=400, bbox_inches='tight')
        plt.close('all')

    def _get_plot_dir(self, charges, beta, current_step=None):
        """Returns directory where plots of observables are to be saved."""
        #  if not HAS_MATPLOTLIB or (self.using_hvd and hvd.rank() != 0):
        if self.using_hvd and hvd.rank() != 0:
            return

        run_steps = charges.shape[0]

        if current_step is not None:  # i.e. sampler evaluated DURING training
            figs_dir = os.path.join(self.figs_dir, 'training')
            io.check_else_make_dir(figs_dir)
            out_dirname = f'{current_step}_TRAIN_{run_steps}_steps_beta_{beta}'
            title_str_key = 'train'
        else:                         # i.e. sampler evaluated AFTER training
            figs_dir = self.figs_dir
            title_str_key = 'eval'
            out_dirname = f'{run_steps}_steps_beta_{beta}'

        out_dir = os.path.join(figs_dir, out_dirname)
        io.check_else_make_dir(out_dir)

        return out_dir, title_str_key

    def _save_run_info(self, run_data, stats, _args):
        """Save samples and observables generated from `self.run` call."""
        #  if not HAS_MATPLOTLIB or (self.using_hvd and hvd.rank() != 0):
        if self.using_hvd and hvd.rank() != 0:
            return

        run_steps, current_step, beta, therm_frac = _args
        therm_steps = run_steps // therm_frac

        training = bool(current_step is not None)
        #  training = True if current_step is not None else False
        #  if current_step is None:
        #      training = False
        #  else:
        #      training = True
        #  import pdb
        #  pdb.set_trace()

        def save_data(data, out_file, name=None):
            out_dir = os.path.dirname(out_file)
            io.log(f"Saving {name} to {out_file}...")
            io.check_else_make_dir(out_dir)
            if out_file.endswith('pkl'):
                with open(out_file, 'wb') as f:
                    pickle.dump(data, f)
            if out_file.endswith('.npy'):
                np.save(out_file, np.array(data))

        files = self._get_run_files(*_args)

        try:
            for key, val in run_data.items():
                save_data(val, files[key], name=key)

            for key, val in stats.items():
                save_data(val, files[key], name=key)
        except KeyError:
            io.log(f"Unable to log {key}: {files[key]}")
            import pdb
            pdb.set_trace()

        actions_arr = np.array(
            list(run_data['actions'].values())
        )[therm_steps:, :]

        plaqs_arr = np.array(
            list(run_data['plaqs'].values())
        )[therm_steps:, :]

        charges_arr = np.array(
            list(run_data['charges'].values()),
            dtype=np.int32
        )[therm_steps:, :]

        charges_squared_arr = charges_arr ** 2

        actions_avg = np.mean(actions_arr)
        actions_err = sem(actions_arr, axis=None)

        plaqs_avg = np.mean(plaqs_arr)
        plaqs_err = sem(plaqs_arr, axis=None)

        q_avg = np.mean(charges_arr)
        q_err = sem(charges_arr, axis=None)

        q2_avg = np.mean(charges_squared_arr)
        q2_err = sem(charges_squared_arr, axis=None)

        _est_key = '  \nestimate +/- stderr'

        ns = self.num_samples
        suscept_k1 = f'  \navg. over all {ns} samples < Q >'
        suscept_k2 = f'  \navg. over all {ns} samples < Q^2 >'
        actions_k1 = f'  \navg. over all {ns} samples < action >'
        plaqs_k1 = f'  \navg. over all {ns} samples < plaq >'

        suscept_stats_strings = {
            suscept_k1: f'{q_avg:.4g} +/- {q_err:.4g}',
            suscept_k2: f'{q2_avg:.4g} +/- {q2_err:.4g}\n',
            _est_key: {}
        }

        actions_stats_strings = {
            actions_k1: f'{actions_avg:.4g} +/- {actions_err:.4g}\n',
            _est_key: {}

        }
        plaqs_stats_strings = {
            'exact_plaq': f'{u1_plaq_exact(beta):.4g}\n',
            plaqs_k1: f'{plaqs_avg:.4g} +/- {plaqs_err:.4g}\n',
            _est_key: {}
        }

        def format_stats(avgs, errs, name=None):
            return [
                f'{name}: {a:.6g} +/- {e:.6}' for (a, e) in zip(avgs, errs)
            ]

        keys = [
            f'sample {idx}' for idx in range(len(stats['suscept_stats'][0]))
        ]

        suscept_vals = format_stats(stats['suscept_stats'][0],
                                    stats['suscept_stats'][1],
                                    '< Q^2 >')

        actions_vals = format_stats(stats['actions_stats'][0],
                                    stats['actions_stats'][1],
                                    '< action >')

        plaqs_vals = format_stats(stats['plaqs_stats'][0],
                                  stats['plaqs_stats'][1],
                                  '< plaq >')

        for k, v in zip(keys, suscept_vals):
            suscept_stats_strings[_est_key][k] = v

        for k, v in zip(keys, actions_vals):
            actions_stats_strings[_est_key][k] = v

        for k, v in zip(keys, plaqs_vals):
            plaqs_stats_strings[_est_key][k] = v

        def accumulate_strings(d):
            all_strings = []
            for k1, v1 in d.items():
                if isinstance(v1, dict):
                    for k2, v2 in v1.items():
                        all_strings.append(f'{k2} {v2}')
                else:
                    all_strings.append(f'{k1}: {v1}\n')

            return all_strings

        actions_strings = accumulate_strings(actions_stats_strings)
        plaqs_strings = accumulate_strings(plaqs_stats_strings)
        suscept_strings = accumulate_strings(suscept_stats_strings)

        charge_probs_strings = []
        for k, v in stats['charge_probs'].items():
            charge_probs_strings.append(f'  probability[Q = {k}]: {v}\n')

        if training:
            str0 = (f'Topological suscept. stats after {current_step} '
                    f'training steps. Chain ran for {run_steps} steps at '
                    f'beta = {beta}.')
            str1 = (f'Total action stats after {current_step} '
                    f'training steps. Chain ran for {run_steps} steps at '
                    f'beta = {beta}.')
            str2 = (f'Average plaquette stats after {current_step} '
                    f'training steps. Chain ran for {run_steps} steps at '
                    f'beta = {beta}.')
            str3 = (f'Topological charge probabilities after '
                    f'{current_step} training steps. '
                    f'Chain ran for {run_steps} steps at beta = {beta}.')
            therm_str = ''
        else:
            str0 = (f'Topological suscept. stats for '
                    f'{run_steps} steps, at beta = {beta}.')
            str1 = (f'Total action stats for '
                    f'{run_steps} steps, at beta = {beta}.')
            str2 = (f'Average plaquette stats for '
                    f'{run_steps} steps, at beta = {beta}.')
            str3 = (f'Topological charge probabilities for '
                    f'{run_steps} steps, at beta = {beta}.')
            therm_str = (
                f'Ignoring first {therm_steps} steps for thermalization.'
            )

        sep_str0 = (1 + max(len(str0), len(therm_str))) * '-'
        sep_str1 = (1 + max(len(str1), len(therm_str))) * '-'
        sep_str2 = (1 + max(len(str2), len(therm_str))) * '-'
        sep_str3 = (1 + max(len(str3), len(therm_str))) * '-'

        io.log(f"Writing statistics to: {files['statistics_txt']}")

        def log_and_write(sep_str, str0, therm_str, stats_strings, file):
            io.log(sep_str)
            io.log(str0)
            io.log(therm_str)
            io.log('')
            _ = [io.log(s) for s in stats_strings]
            io.log(sep_str)
            io.log('')

            io.write(sep_str, file, 'a')
            io.write(str0, file, 'a')
            io.write(therm_str, file, 'a')
            _ = [io.write(s, file, 'a') for s in stats_strings]
            io.write('\n', file, 'a')

        log_and_write(sep_str0, str0, therm_str, suscept_strings,
                      files['statistics_txt'])
        log_and_write(sep_str1, str1, therm_str, actions_strings,
                      files['statistics_txt'])
        log_and_write(sep_str2, str2, therm_str, plaqs_strings,
                      files['statistics_txt'])
        log_and_write(sep_str3, str3, therm_str, charge_probs_strings,
                      files['statistics_txt'])

    def _get_run_files(self, *_args):
        """Create dir and files for storing observables from `self.run`."""
        #  if not HAS_MATPLOTLIB or (self.using_hvd and hvd.rank() != 0):
        if self.using_hvd and hvd.rank() != 0:
            return

        run_steps, current_step, beta, _ = _args

        observables_dir = os.path.join(self.eval_dir, 'observables')
        io.check_else_make_dir(observables_dir)

        if current_step is None:                     # running AFTER training
            obs_dir = os.path.join(observables_dir,
                                   f'steps_{run_steps}_beta_{beta}')
            io.check_else_make_dir(obs_dir)

            npy_file = f'samples_history_steps_{run_steps}_beta_{beta}.npy'
            samples_file_npy = os.path.join(self.samples_dir, npy_file)

        else:                                        # running DURING training
            obs_dir = os.path.join(observables_dir, 'training')
            io.check_else_make_dir(obs_dir)
            obs_dir = os.path.join(obs_dir,
                                   f'{current_step}_TRAIN_'
                                   f'steps_{run_steps}_beta_{beta}')

            npy_file = (f'samples_history_{current_step}_TRAIN_'
                        f'steps_{run_steps}_beta_{beta}.npy')
            samples_file_npy = os.path.join(self.train_samples_dir, npy_file)

        statistics_file_txt = os.path.join(
            obs_dir, f'statistics_steps_{run_steps}_beta_{beta}.txt'
        )

        def file_name(name):
            out_file = f'{name}_steps_{run_steps}_beta_{beta}.pkl'
            return os.path.join(obs_dir, out_file)

        files = {
            'samples_npy': samples_file_npy,
            'statistics_txt': statistics_file_txt,
            'accept_probs': file_name('accept_probs'),
            'actions': file_name('actions'),
            'plaqs': file_name('plaqs'),
            'charges': file_name('charges'),
            'charge_diffs': file_name('charge_diffs'),
            'actions_stats': file_name('actions_stats'),
            'plaqs_stats': file_name('plaqs_stats'),
            'charges_stats': file_name('charges_stats'),
            'suscept_stats': file_name('suscept_stats'),
            'charge_probs': file_name('charge_probs')
        }

        return files


def create_config(FLAGS, params):
    """Create tensorflow config."""
    config = tf.ConfigProto()
    if FLAGS.time_size > 8:
        off = rewriter_config_pb2.RewriterConfig.OFF
        config.graph_options.rewrite_options.arithmetic_optimization = off

    if FLAGS.gpu:
        io.log("Using gpu for training.")
        params['data_format'] = 'channels_first'
        os.environ["KMP_BLOCKTIME"] = str(0)
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
        # Horovod: pin GPU to be used to process local rank (one GPU per
        # process)
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        if HAS_HOROVOD and FLAGS.horovod:
            config.gpu_options.visible_device_list = str(hvd.local_rank())
    else:
        params['data_format'] = 'channels_last'

    if HAS_MATPLOTLIB:
        params['_plot'] = True

    if FLAGS.theta:
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


# pylint: disable=too-many-statements, too-many-branches, too-many-locals
def main(FLAGS):
    """Main method for creating/training/running L2HMC for U(1) gauge model."""
    if HAS_HOROVOD and FLAGS.horovod:
        io.log("INFO: USING HOROVOD")
        hvd.init()

    if FLAGS.use_bn:
        io.log("Using batch_norm...")

    params = {}
    for key, val in FLAGS.__dict__.items():
        params[key] = val

    if FLAGS.hmc:
        params['eps_trainable'] = False
        #  beta1 = params.get('beta', 4.)
        #  beta2 = params.get('beta_init', 4.)
        #  beta3 = params.get('beta_final', 4.)
        #  beta = max((beta1, beta2, beta3))

        #  params['beta'] = beta
        #  params['beta_init'] = beta
        #  params['beta_final'] = beta

    #  if HAS_HOROVOD and FLAGS.horovod:
        #  params['lr_init'] *= hvd.size()
        #  params['train_steps'] /= hvd.size()
        #  params['lr_decay_steps'] /= hvd.size()

    #  if FLAGS.horovod:
    #      io.log('Number of CPUs: %d' % hvd.size())

    #  io.log('\n\n\n')
    #  io.log(len(str(model.log_dir))*'~')
    #  if not model.using_hvd:
    #      io.log(f"model.log_dir: {model.log_dir}")
    #  io.log(len(str(model.log_dir))*'~')
    #  io.log('\n\n\n')

    #  if not FLAGS.horovod or (FLAGS.horovod and hvd.rank() == 0):
    #  if not model.using_hvd or (model.using_hvd and hvd.rank() == 0):
    #      io.save_params_to_pkl_file(params, model.info_dir)

    #  if FLAGS.horovod:
    #      if hvd.rank() == 0:
    #          io.save_params_to_pkl_file(params, model.info_dir)
    #  else:
    #      io.save_params_to_pkl_file(params, model.info_dir)
    run_steps = FLAGS.run_steps
    condition1 = FLAGS.horovod and hvd.rank() == 0
    condition2 = not FLAGS.horovod

    config, params = create_config(FLAGS, params)

    model = GaugeModel(params=params,
                       config=config,
                       sess=None,
                       log_dir=FLAGS.log_dir,
                       restore=FLAGS.restore)

    if FLAGS.hmc:
        betas = np.arange(model.beta_init, model.beta_final, 1)
        if condition1 or condition2:
            model.run(run_steps, beta=model.beta_final)
            for beta in betas:
                model.run(int(5e4), beta=beta)

        model.sess.close()
        tf.reset_default_graph()

    else:
        if FLAGS.profiler:
            model.train_profiler(params['train_steps'], trace=True)
        else:
            if FLAGS.restore:
                # pylint: disable=protected-access
                if model._current_state['step'] <= model.train_steps:
                    model.train(model.train_steps,
                                samples_init=model._current_state['samples'],
                                beta_init=model._current_state['beta'],
                                trace=FLAGS.trace)
                else:
                    io.log('Model restored but training completed. '
                           'Preparing to run the trained sampler...')
            else:
                io.log(f"Training began at: {time.ctime()}")
                model.train(params['train_steps'], samples_init=None,
                            beta_init=None, trace=FLAGS.trace)

        betas = np.arange(model.beta_init, model.beta_final, 1)
        if condition1 or condition2:
            model.run(run_steps, beta=model.beta_final)
            for beta in betas:
                model.run(int(1e4), beta=beta)
            #  if FLAGS.long_run:
            #      model.run(run_steps, beta=model.beta_final+1)

        model.sess.close()
        tf.reset_default_graph()

        ###########################################################
        # Create separate HMC instance for performance comparison
        ###########################################################
        io.log('\n')
        io.log(80*'=')
        io.log('Running generic HMC using params from trained model for '
               'performance comparison.')
        io.log(80*'=' + '\n')
        hmc_params = model.params
        hmc_params['eps'] = model._current_state['eps']
        hmc_params['hmc'] = True
        hmc_params['beta_init'] = model.beta_init
        hmc_params['beta_final'] = model.beta_final

        hmc_config, hmc_params = create_config(FLAGS, hmc_params)
        hmc_log_dir = os.path.join(model.log_dir, 'HMC')

        hmc_model = GaugeModel(params=hmc_params,
                               sess=None,
                               config=hmc_config,
                               log_dir=hmc_log_dir,
                               restore=False,
                               build_graph=True)

        if condition1 or condition2:
            hmc_model.run(run_steps, beta=hmc_model.beta_final)

            for beta in betas:
                hmc_model.run(int(1e4), beta=beta)

        hmc_model.sess.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
