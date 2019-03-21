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
import shlex
import pickle
import argparse

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


#  import utils.gauge_model_helpers as helpers
import utils.file_io as io

from globals import COLORS, FILE_PATH, GLOBAL_SEED, MARKERS, PARAMS
from collections import Counter, OrderedDict
from lattice.lattice import GaugeLattice, u1_plaq_exact
from utils.tf_logging import variable_summaries

from scipy.stats import sem
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorflow.core.protobuf import rewriter_config_pb2

from dynamics.gauge_dynamics import GaugeDynamics

if HAS_MATPLOTLIB:
    from utils.plot_helper import plot_multiple_lines

np.random.seed(GLOBAL_SEED)

if '2.' not in tf.__version__:
    tf.set_random_seed(GLOBAL_SEED)
    tf.logging.set_verbosity(tf.logging.INFO)


##############################################################################
#                       Helper functions
##############################################################################
def check_else_make_dir(d):
    if HAS_HOROVOD:
        if hvd.rank() != 0:
            return
    if os.path.isdir(d):
        io.log(f"Directory {d} already exists... Continuing.")
    else:
        try:
            io.log(f"Creating directory {d}")
            os.makedirs(d)
        except OSError as e:
            raise

def tf_accept(x, _x, px):
    """Helper function for determining if x is accepted given _x."""
    mask = (px - tf.random_uniform(tf.shape(px)) > 0.)
    return tf.where(mask, _x, x)


def project_angle(x):
    """Returns the projection of an angle `x` from [-4pi, 4pi] to [-pi, pi]."""
    return x - 2 * np.pi * tf.floor((x + np.pi) / (2 * np.pi))


def project_angle_approx(x, N=5):
    """Use the fourier series representation `x` to approx `project_angle`.

    NOTE: Because `project_angle` suffers a discontinuity, we approximate `x`
    with its Fourier series representation in order to have a differentiable
    function when computing the loss.
    
    Args:
        x (array-like): Array to be projected.
        N (int): Number of terms to keep in Fourier series.
    """
    y = np.zeros(x.shape)
    for n in range(1, N):
        y += (-2 / n) * ((-1) ** n) * tf.sin(n * x)
    return y


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

def project_angle_fft(x, num_components=50):
    a0, a, b = calc_fourier_coeffs(project_angle, 2 * np.pi, num_components)
    y_fft = calc_fourier_series(a0, a, b, x, 2 * np.pi)
    return y_fft


# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes
class GaugeModel(object):
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
        if not restore:
            self._create_attrs(params)
            # --------------------------------------------------------------
            # Create necessary directories for holding checkpoints, data, etc.
            # --------------------------------------------------------------
            if (self.using_hvd and hvd.rank() == 0) or not self.using_hvd:

                self._create_dir_structure(log_dir)
                # ---------------------------------------------------------
                # Write relevant instance attributes to .txt file.
                # ---------------------------------------------------------
                self._write_run_parameters(_print=True)

        # ------------------------------------------------------------------
        # Create lattice object.
        # ------------------------------------------------------------------
        with tf.name_scope('lattice'):
            self.lattice = self._create_lattice()

            #  self._create_tensors()
            self.batch_size = self.lattice.samples.shape[0]
            self.x_dim = self.lattice.num_links
            self.samples = tf.convert_to_tensor(self.lattice.samples,
                                                dtype=tf.float32)
        # ------------------------------------------------------------------
        # Create placeholders for input data.
        # ------------------------------------------------------------------
        with tf.name_scope('inputs'):
            if not tf.executing_eagerly():
                self.x = tf.placeholder(dtype=tf.float32,
                                        shape=(None, self.x_dim),
                                        name='x')

                self.beta = tf.placeholder(dtype=tf.float32,
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
                self.plaq_sums_op = self._calc_plaq_sums(self.x)
            with tf.name_scope('actions'):
                self.actions_op = self._calc_total_actions(self.x)
            with tf.name_scope('avg_plaqs'):
                self.plaqs_op = self._calc_avg_plaqs(self.x)
                self.avg_plaq_op = tf.reduce_mean(self.plaqs_op,
                                                  name='avg_plaq')
            with tf.name_scope('top_charges'):
                self.charges_op = self._calc_top_charges(self.x, fft=False)
        # ------------------------------------------------------------------
        # If restore, load from most recently saved checkpoint in `log_dir`.
        # ------------------------------------------------------------------
        if restore:
            self._restore_model(log_dir, sess, config)
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
        #  self.charge_loss = params.get('charge_loss', False)
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
        io.log('\n')
        io.log(80*'-')
        io.log(f'self.condition1: {self.condition1}')
        io.log(f'self.condition2: {self.condition2}')
        io.log(f'self.condition1 or self.condition2: '
               f'{self.condition1 or self.condition2}')
        io.log('\n')
        io.log(80*'-')

        self.safe_write = self.condition1 or self.condition2

    def _create_dir_structure(self, log_dir):
        """Create self.files and directory structure."""
        if self.using_hvd or HAS_HOROVOD:
            io.log('\n')
            io.log(80 * '-')
            io.log(f"Calling _create_dir_structure from {hvd.rank()}...")
            if hvd.rank() != 0:
                io.log("Returning...")
                io.log(80 * '-')
                io.log('\n')
                return
        #  log_dir = os.path.join(root_log_dir, f'run_{run_num}')
        #  if log_dir is None:
        #      root_log_dir = os.path.join(project_dir, 'gauge_logs_graph')
        #  else:
        #      root_log_dir = os.path.join(project_dir, log_dir)
        #      self.log_dir = log_dir
        #      io.check_else_make_dir(self.log_dir)
        #  if self.safe_write:
        project_dir = os.path.abspath(os.path.dirname(FILE_PATH))
        if log_dir is None:
            if (self.using_hvd and hvd.rank() == 0) or not self.using_hvd:
                root_log_dir = os.path.join(project_dir, 'gauge_logs_graph')
            else:
                return
        else:
            root_log_dir = os.path.join(project_dir, log_dir)

        check_else_make_dir(root_log_dir)
        #  io.check_else_make_dir(root_log_dir)

        #  if self.log_dir is not None:
        #      io.log("self.log_dir already exists, returning.")
        #  project_dir = os.path.abspath(os.path.dirname(FILE_PATH))
        #  root_log_dir = os.path.abspath(os.path.join(project_dir, log_dir))
        #  if not os.path.exists(log_dir):
        #      try:
        #          os.makedirs(log_dir)
        #      except OSError as e:
        #          if e.errno == errno.EEXIST and os.path.isdir(log_dir):
        #              return
        #          else:
        #              raise

        #  if self.condition1 or self.condition2:
        #  if self.safe_write:

        if not self.using_hvd or (self.using_hvd and hvd.rank() == 0):
            run_num = io.get_run_num(root_log_dir)
            log_dir = os.path.abspath(os.path.join(root_log_dir,
                                                   f'run_{run_num}'))
            #  io.check_else_make_dir(log_dir)
            check_else_make_dir(log_dir)

            self.log_dir = log_dir
            if self.using_hvd:
                io.log('\n')
                io.log(f"Successfully created and assigned `self.log_dir` on "
                       f"{hvd.rank()}.")
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

    def _restore_model(self, log_dir, sess, config):
        """Restore model from previous run contained in `log_dir`."""
        if self.using_hvd:
            if hvd.rank() != 0:
                return

        if self.hmc:
            io.log(f"ERROR: self.hmc: {self.hmc}. "
                   "No model to restore. Exiting.")
            sys.exit(1)

        assert os.path.isdir(log_dir), (f"log_dir: {log_dir} does not exist.")

        run_info_dir = os.path.join(log_dir, 'run_info')
        assert os.path.isdir(run_info_dir), (f"run_info_dir: {run_info_dir}"
                                             " does not exist.")

        with open(self.files['params_pkl_file'], 'rb') as f:
            self.params = pickle.load(f)

        with open(self.files['train_data_file'], 'rb') as f:
            self.train_data_dict = pickle.load(f)

        with open(self.files['current_state_file'], 'rb') as f:
            self._current_state = pickle.load(f)

        self._create_attrs(self.params)

        self.global_step.assign(self._current_state['step'])
        self.lr.assign(self._current_state['lr'])

        self.lattice = self._create_lattice()
        self.samples = tf.convert_to_tensor(self.lattice.samples,
                                            dtype=tf.float32)

        kwargs = {
            'hmc': self.hmc,
            'eps': self._current_state['eps'],
            'network_arch': self.network_arch,
            'beta_init': self.data['beta'],
            'num_steps': self.num_steps,
            'eps_trainable': self.eps_trainable
        }
        self.dynamics, self.potential_fn = self._create_dynamics(self.lattice,
                                                                 self.samples,
                                                                 **kwargs)

        self._create_dynamics(self.lattice, self.samples, kwargs)

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
        if self.using_hvd or HAS_HOROVOD:
            io.log('\n')
            io.log(80 * '-')
            io.log(f"Calling _write_run_parameters from {hvd.rank()}...")
            if hvd.rank() != 0:
                io.log("Returning...")
                io.log(80 * '-')
                io.log('\n')
                return
        #  if self.using_hvd:
        #      if hvd.rank() != 0:
        #          return
        if self.info_dir is None:
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
        default_kwargs = {
            'eps': self.eps,
            'hmc': self.hmc,
            'network_arch': self.network_arch,
            'num_steps': self.num_steps,
            'eps_trainable': self.eps_trainable,
            'data_format': self.data_format,
        }

        default_kwargs.update(kwargs)  # update default_kwargs using kwargs

        potential_fn = lattice.get_energy_function(samples)

        dynamics = GaugeDynamics(lattice=lattice,
                                 potential_fn=potential_fn,
                                 **default_kwargs)  # updated default_kwargs

        return dynamics, potential_fn

    def _create_tensors(self):
        """Initialize tensors (and placeholders if executing in graph mode).

        NOTE: UNNECESSARY, TO BE REMOVED IN FUTURE UPDATE.
        """
        self.batch_size = self.lattice.samples.shape[0]
        self.x_dim = self.lattice.num_links
        self.samples = tf.convert_to_tensor(self.lattice.samples,
                                            dtype=tf.float32)

        if not tf.executing_eagerly():
            self.x = tf.placeholder(tf.float32, (None, self.x_dim), name='x')
            self.beta = tf.placeholder(tf.float32, shape=(), name='beta')
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

    def _calc_plaq_sums(self, x):
        """Calculate plaquette sums.

        Explicitly, calculate the sum of the links around each plaquette in the
        lattice for each sample in samples.

        Args:
            samples (N, D): Tensor of shape where N is the batch size and D is
                the number of links on the lattice (flattened)

        Returns:
            plaq_sums (N, Lx, Lt): Sum of link variables around each plaquette.

            NOTE: Lx, Lt re self.lattice.space_size, time_size respectively.
        """
        x = tf.reshape(x, shape=(self.batch_size, *self.lattice.links.shape))

        with tf.name_scope('calc_plaq_sums'):
            plaq_sums = (x[:, :, :, 0]
                         - x[:, :, :, 1]
                         - tf.roll(x[:, :, :, 0], shift=-1, axis=2)
                         + tf.roll(x[:, :, :, 1], shift=-1, axis=1))
        return plaq_sums

    def _calc_total_actions(self, x):
        """Calculate total action for each sample in samples."""
        with tf.name_scope('calc_total_actions'):
            total_actions = tf.reduce_sum(
                1. - tf.cos(self._calc_plaq_sums(x)), axis=(1, 2),
                name='total_actions'
            )
        return total_actions

    def _calc_avg_plaqs(self, x):
        """Calculate average plaquette values for each sample in samples."""
        num_plaqs = self.lattice.num_plaquettes
        with tf.name_scope('calc_avg_plaqs'):
            avg_plaqs = tf.reduce_sum(tf.cos(self._calc_plaq_sums(x)),
                                      (1, 2), name='avg_plaqs') / num_plaqs
        return avg_plaqs

    def _calc_top_charges(self, x, fft=False):
        """Calculate topological charges for each sample in samples."""
        with tf.name_scope('calc_top_charges'):
            if fft:
                ps_proj = project_angle_approx(self._calc_plaq_sums(x))
            else:
                ps_proj = project_angle(self._calc_plaq_sums(x))

            top_charges = (tf.reduce_sum(ps_proj, axis=(1, 2),
                                         name='top_charges')) / (2 * np.pi)
            #  top_charges = (
            #      0.1 + (tf.reduce_sum(projector(self._calc_plaq_sums(x)),
            #                           axis=(1, 2), name='top_charges')
            #             / (2 * np.pi))
            #  )

        return top_charges

    def _calc_top_charges_diff(self, x1, x2, fft=False):
        """Calculate difference in topological charges between x1 and x2."""
        with tf.name_scope('calc_top_charges_diff'):
            dq = tf.abs(self._calc_top_charges(x1, fft)
                        - self._calc_top_charges(x2, fft))
            #  dq = tf.abs(tf.cast(self._calc_top_charges(x1, fft), tf.float32)
            #  - tf.cast(self._calc_top_charges(x2, fft), tf.float32))
        return dq

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
        eps = 1e-3
        aux_weight = weights.get('aux_weight', 1.)
        std_weight = weights.get('std_weight', 1.)
        charge_weight = weights.get('charge_weight', 1.)

        with tf.name_scope('x_update'):
            x_, _, px, x_out = self.dynamics(x, beta)
            #  xn = x_[0]                  # dynamics update:    x  -->  xn
            #  xf = tf.squeeze(x_out)      # accept/reject:      xn -->  xf
        with tf.name_scope('z_update'):
            z = tf.random_normal(tf.shape(x), name='z')  # Auxiliary variable
            z_, _, pz, _ = self.dynamics(z, beta)
            #  zn = z_[0]

        with tf.name_scope('top_charge_diff'):
            x_dq = tf.cast(self._calc_top_charges_diff(x, x_out, fft=False),
                           dtype=tf.int32)

        # Add eps for numerical stability; following released impl
        with tf.name_scope('calc_loss'):
            with tf.name_scope('std_loss'):
                with tf.name_scope('x_loss'):
                    x_std_loss = (
                        tf.reduce_sum(self.metric_fn(x, x_), axis=1) * px + eps
                    )

                with tf.name_scope('z_loss'):
                    z_std_loss = aux_weight * (
                        tf.reduce_sum(self.metric_fn(z, x_), axis=1) * pz + eps
                    )

                with tf.name_scope('tot_loss'):
                    ls = self.loss_scale
                    std_loss = (ls * (1. / x_std_loss + 1. / z_std_loss)
                                - (x_std_loss + z_std_loss) / ls)
                    std_loss *= std_weight

            with tf.name_scope('charge_loss'):
                with tf.name_scope('x_loss'):
                    x_dq_fft = self._calc_top_charges_diff(x, x_, fft=True)
                    xq_loss = px * x_dq_fft + eps
                with tf.name_scope('z_loss'):
                    z_dq_fft = self._calc_top_charges_diff(z, x_, fft=True)
                    zq_loss = aux_weight * (pz * z_dq_fft + eps)

                with tf.name_scope('tot_loss'):
                    charge_loss = charge_weight * (xq_loss + zq_loss)

            loss = tf.reduce_mean(std_loss + charge_loss, axis=0, name='loss')

        return loss, x_out, px, x_dq

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
            loss, x_out, accept_prob, x_dq = self._calc_loss(x, beta,
                                                             **weights)
            with tf.name_scope('grads'):
                grads = tf.gradients(loss, self.dynamics.variables)
                if self.clip_grads:
                    grads, _ = tf.clip_by_global_norm(grads, self.clip_value)
                    #  grads = tf.check_numerics(
                    #      grads, 'check_numerics caught bad gradients'
                    #  )

        return loss, grads, x_out, accept_prob, x_dq

    def _create_sampler(self):
        """Create operation for generating new samples using dynamics engine.

        NOTE: This method is to be used when running generic HMC to create
            operations for dealing with `dynamics.apply_transition` without
            building unnecessary operations for calculating loss.
        """
        with tf.name_scope('sampler'):
            #  inputs = (self.x, self.beta)
            #  _, _, self.px, self.x_out = self.dynamics(self.x, self.beta)
            _, _, px, x_out = self.dynamics(self.x, self.beta)
            #  xn = x_[0]                  # dynamics update:    x  -->  xn
            self.px = px
            self.x_out = tf.squeeze(x_out)      # accept/reject:      xn -->  xf

            x_dq = self._calc_top_charges_diff(self.x, self.x_out, fft=False)
            self.charge_diff_op = tf.reduce_sum(x_dq) / self.num_samples
            #  self.charge_diff_op = x_dq) / self.num_samples

    def _create_summaries(self):
        """Create summary objects for logging in TensorBoard."""
        if self.using_hvd:
            if hvd.rank() != 0:
                return

        ld = self.log_dir
        self.summary_writer = tf.contrib.summary.create_file_writer(ld)

        grads_and_vars = zip(self.grads, self.dynamics.trainable_variables)

        with tf.name_scope('loss'):
            tf.summary.scalar('loss', self.loss_op)

        with tf.name_scope('step_size'):
            tf.summary.scalar('step_size', self.dynamics.eps)

        with tf.name_scope('tunneling_events'):
            tf.summary.scalar('tunneling_events_per_sample',
                              self.charge_diff_op)
        with tf.name_scope('avg_plaq'):
            tf.summary.scalar('avg_plaq', self.avg_plaq_op)

        with tf.name_scope('summaries'):
            for grad, var in grads_and_vars:
                try:
                    layer, _type = var.name.split('/')[-2:]
                    name = layer + '_' + _type[:-2]
                except:
                    name = var.name[:-2]
                variable_summaries(var, name)
                variable_summaries(grad, name + '_gradient')

        self.summary_op = tf.summary.merge_all(name='summary_op')


    def _log_write_graph_creation_time(self, **times):
        if self.using_hvd or HAS_HOROVOD:
            io.log('\n')
            io.log(80 * '-')
            io.log(f"Calling _write_run_parameters from {hvd.rank()}...")
            if hvd.rank() != 0:
                io.log("Returning...")
                io.log(80 * '-')
                io.log('\n')
                return

        if self.files is None:
            return

        def log_and_write(s, f):
            """Print string `s` to std out and also write to file `f`."""
            if self.using_hvd:
                if hvd.rank() != 0:
                    return
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

    # pylint:disable=too-many-statements
    def build_graph(self, sess=None, config=None):
        """Build graph for TensorFlow."""

        start_time = time.time()

        if self.hmc:  # if running generic HMC, all we need is the sampler
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
            return

        with tf.name_scope('global_step'):
            self.global_step = tf.train.get_or_create_global_step()
            self.global_step.assign(1)

        with tf.name_scope('learning_rate'):
            self.lr = tf.train.exponential_decay(self.lr_init,
                                                 self.global_step,
                                                 self.lr_decay_steps,
                                                 self.lr_decay_rate,
                                                 staircase=True,
                                                 name='learning_rate')
        with tf.name_scope('optimizer'):
            if self.using_hvd:
                #  self.optimizer = tf.train.AdamOptimizer(self.lr * hvd.size())
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                self.optimizer = hvd.DistributedOptimizer(self.optimizer)
            else:
                self.optimizer = tf.train.AdamOptimizer(self.lr)

        with tf.name_scope('loss'):
            t0_loss = time.time()

            output = self._calc_loss_and_grads(x=self.x, beta=self.beta,
                                               **self.loss_weights)

            self.loss_op, self.grads, self.x_out, self.px, x_dq = output
            self.charge_diff_op = tf.reduce_sum(x_dq) / self.num_samples
            #  self.charge_diff_op = tf.reduce_mean(x_dq)

            t_diff_loss = time.time() - t0_loss

        with tf.name_scope('train'):
            t0_train = time.time()

            grads_and_vars = zip(self.grads, self.dynamics.variables)
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


        times = {
            't_diff_loss': t_diff_loss,
            't_diff_train': t_diff_train,
            't_diff_summaries': t_diff_summaries,
            't_diff_graph': time.time() - start_time
        }
        self._log_write_graph_creation_time(**times)

        #  self.sess.run(tf.global_variables_initializer())
        #  if self.using_hvd:
        #      self.sess.run(hvd.broadcast_global_variables(0))

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

        #  if self.condition1 or self.condition2:
        #      self.saver = tf.train.Saver(max_to_keep=2)
            #  self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        samples_init = kwargs.get('samples_init', None)
        beta_init = kwargs.get('beta_init', None)
        trace = kwargs.get('trace', False)

        #  if self.condition1 or self.condition2:
        if self.condition1 or self.condition2:
            try:
                self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
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

        #  Move attribute look ups outside loop to improve performance
        #  loss_op = self.loss_op
        #  train_op = self.train_op
        #  summary_op = self.summary_op
        #  x_out = self.x_out
        #  px = self.px
        #  learning_rate = self.learning_rate
        #  dynamics = self.dynamics
        #  x = self.x
        #  dynamics_beta = self.dynamics.beta
        start_time = time.time()

        if beta_init is None:
            beta_np = self.beta_init
        else:
            beta_np = beta_init

        if samples_init is None:
            samples_np = np.reshape(
                np.array(self.lattice.samples, dtype=np.float32),
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

            #  while not self.sess.should_stop():
            for step in range(initial_step, train_steps):
                start_step_time = time.time()

                beta_np = self.update_beta(step)

                fd = {self.x: samples_np,
                      self.beta: beta_np}

                outputs = self.sess.run([
                    self.train_op,         # apply gradients
                    self.loss_op,          # calculate loss
                    self.x_out,            # get new samples
                    self.px,               # calculate accept. prob
                    self.dynamics.eps,     # evaluate current step size
                    self.actions_op,       # calculate avg. actions
                    self.plaqs_op,         # calculate avg. plaquettes
                    self.charges_op,       # calculate top. charges
                    self.lr,               # evaluate learning rate
                    self.charge_diff_op,   # change in top charge / num_samples 
                ], feed_dict=fd)

                loss_np = outputs[1]
                samples_np = np.mod(outputs[2], 2 * np.pi)
                #  samples_np = outputs[2]
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
                    data_str = (f"{step:>5g}/{self.train_steps:<6g} "
                                f"{loss_np:^9.4g} "
                                f"{time.time() - start_step_time:^9.4g} "
                                f"{np.mean(px_np):^9.4g}"
                                f"{eps_np:^9.4g} "
                                f"{beta_np:^9.4g} "
                                f"{np.mean(actions_np):^9.4g} "
                                f"{np.mean(plaqs_np):^9.4g} "
                                f"{u1_plaq_exact(beta_np):^9.4g} "
                                f"{charge_diff:^9.4g} "
                                f"{lr_np:^9.4g}")

                    io.log(data_str)
                    try:
                        io.write(data_str, self.files['run_info_file'], 'a')
                    except AttributeError:
                        continue

                # Intermittently run sampler and save samples to pkl file.
                # We can calculate observables from these samples to
                # evaluate the samplers performance while we continue training.
                if (step + 1) % self.training_samples_steps == 0:
                    if self.condition1 or self.condition2:
                        t0 = time.time()
                        io.log(80 * '-')
                        self.run(self.training_samples_length,
                                 current_step=step+1,
                                 beta=self.beta_final)
                        io.log(f"  done. took: {time.time() - t0}.")
                        io.log(80 * '-')
                        io.log(self.train_header)

                if (step + 1) % self.save_steps == 0:
                    if self.condition1 or self.condition2:
                        self._save_model(samples=samples_np)
                        #  helpers.write_run_data(self.files['run_info_file'],
                        #                         self.data)

                if step % self.logging_steps == 0:
                    if self.using_hvd:
                        if hvd.rank() != 0:
                            continue

                    io.log(self.train_header)

                    if trace:
                        # This saves the timeline to a chrome trace format:
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
                        if self.condition1 or self.condition2:
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
                    if self.condition1 or self.condition1:
                        self.writer.flush()

            train_time = time.time() - start_time
            io.log("Training complete!")
            io.log(f"Time to complete training: {train_time:.3g}.")
            step = self.sess.run(self.global_step)
            if self.condition1 or self.condition2:
                try:
                    self._save_model(samples=samples_np)
                except AttributeError:
                    pass
                try:
                    self._plot_charge_diff()
                except:
                    pass

        except (KeyboardInterrupt, SystemExit):
            io.log("\nKeyboardInterrupt detected! \n", nl=False)
            io.log("Saving current state and exiting.\n", nl=False)
            io.log(data_str)
            io.write(data_str, self.files['run_info_file'], 'a')
            if self.condition1 or self.condition2:
                try:
                    self._save_model(samples=samples_np)
                except AttributeError:
                    pass
                try:
                    self._plot_charge_diff()
                except:
                    pass

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
        if self.using_hvd:        # if using horovod, make sure we only perform
            if hvd.rank() != 0:   # file IO on rank 0.
                return

        if beta is None:
            beta = self.beta_final

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

        eps = self.sess.run(self.dynamics.eps)

        # start with randomly generated samples
        samples = np.random.randn(*(self.batch_size, self.x_dim))
        #  samples = np.random.randn(*self.samples.shape)
        #  samples_history = []
        charges_arr = []
        actions_dict = {}
        plaqs_dict = {}
        charges_dict = {}
        charge_diff_dict = {}
        plaq_exact = u1_plaq_exact(beta)

        start_time = time.time()
        eval_strings = []
        try:
            for step in range(run_steps):
                t0 = time.time()

                fd = {
                    self.x: samples,
                    self.beta: beta,
                }

                outputs = self.sess.run([
                    self.x_out,
                    self.px,
                    self.actions_op,
                    self.plaqs_op,
                    self.charges_op,
                    self.charge_diff_op,
                ], feed_dict=fd)

                samples = np.mod(outputs[0], 2 * np.pi)
                #  samples = outputs[0]
                px = outputs[1]
                actions_np = outputs[2]
                plaqs_np = outputs[3]
                charges_np = outputs[4]
                charge_diff = outputs[5]

                charges_arr.append(charges_np)

                #  try:
                #      tunneling_events = np.sum(
                #          np.abs(charges_arr[-1] - charges_arr[-2])
                #      )
                #
                #  except IndexError:
                #      tunneling_events = 0

                #  samples_history.append(np.squeeze(samples))

                key = (step, beta)
                actions_dict[key] = actions_np
                plaqs_dict[key] = plaqs_np
                charges_dict[key] = charges_np
                charge_diff_dict[key] = charge_diff

                #  if step % 10 == 0:
                tt = (time.time() - t0)  # / (norm_factor)
                eval_str = (f'{step:>5g}/{run_steps:<6g} '
                            f'{tt:^9.4g} '
                            f'{np.mean(px):^9.4g} '
                            f'{eps:^9.4g} '
                            f'{beta:^9.4g} '
                            f'{np.mean(actions_np):^9.4g} '
                            f'{np.mean(plaqs_np):^9.4g} '
                            f'{plaq_exact:^9.4g} '
                            f'{charge_diff:^9.4g} ')

                io.log(eval_str)

                eval_strings.append(eval_str)
                #  log('\n')
                #  log('top_charges: ', nl=False)
                #  log(charges_np)
                #  log('\n')

                #  io.write('accept_prob:', eval_file, 'a', nl=False)
                #  io.write(str(px), eval_file, 'a', nl=True)
                #  io.write('top. charges:', eval_file, 'a', nl=False)
                #  io.write(str(charges_np), eval_file, 'a', nl=True)
                #  io.write('', eval_file, 'a')

                if step % 100 == 0:
                    io.log(header_str)

        except (KeyboardInterrupt, SystemExit):
            io.log("\nKeyboardInterrupt detected! \n", nl=False)
            io.log("Saving current state and exiting.\n", nl=False)

        #  io.write(eval_strings, eval_file, 'a')
        _ = [io.write(s, eval_file, 'a') for s in eval_strings]
        observables = (actions_dict, plaqs_dict,
                       charges_dict, charge_diff_dict)
        stats = self.calc_observables_stats(observables, therm_frac)
        charges_arr = np.array(charges_arr)
        _args = (run_steps, current_step, beta, therm_frac)

        self._save_run_info(observables, stats, _args)
        self.make_plots(observables, charges_arr, beta, current_step)
        io.log(f'\n Time to complete run: {time.time() - start_time} seconds.')
        io.log(80*'-' + '\n', nl=False)

        return observables, stats

    def make_plots(self, observables, charges_arr, beta, current_step):
        """Wrapper function that calls all individual plotting functions."""
        t0 = time.time()
        if HAS_MATPLOTLIB:
            self._plot_charge_diff()
            self._plot_observables(observables, beta, current_step)
            self._plot_top_charges(charges_arr, beta, current_step)
            self._plot_top_charge_probs(charges_arr, beta, current_step)
        io.log(f'\n Took {time.time() - t0:4.3g} s to create and save plots.')

    # pylint:disable=no-self-use
    def calc_observables_stats(self, observables, therm_frac=10):
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
        actions_dict = observables[0]
        plaqs_dict = observables[1]
        charges_dict = observables[2]
        #  charge_diff_dict = observables[4]

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

        stats = (actions_stats, plaqs_stats, charges_stats,
                 suscept_stats, charge_probabilities)

        return stats

    def _plot_observables(self, observables, beta, current_step=None):
        """Plot observables stored in `observables`."""
        if not HAS_MATPLOTLIB or (self.using_hvd and hvd.rank() != 0):
            return

        io.log("Plotting observables...")
        #  samples_history = observables[0]
        actions_dict = observables[0]
        plaqs_dict = observables[1]
        charges_dict = observables[2]
        charge_diff_dict = observables[3]

        actions_arr = np.array(list(actions_dict.values()))
        plaqs_arr = np.array(list(plaqs_dict.values()))
        charges_arr = np.array(list(charges_dict.values()))
        charge_diff_arr = np.array(list(charge_diff_dict.values()))

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

        plt.savefig(plaqs_plt_file, dpi=400, bbox_inches='tight')

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
        plt.savefig(charge_diff_plt_file, dpi=400, bbox_inches='tight')
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
        ax.set_title(rf"""$N = #{self.num_samples} samples / batch""",
                     fontsize=16)

        #  title_str = (f'Number of tunneling events vs. '
        #               f'training step for {self.num_samples} samples')
        #  ax.set_title(title_str, fontsize=16)
        out_file = os.path.join(self.figs_dir,
                                'tunneling_events_vs_training_step.png')
        print(f"Saving figure to: {out_file}.")
        plt.savefig(out_file, dpi=400, bbox_inches='tight')

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
            out_file = os.path.join(out_dir, f'top_charge_vs_step_{idx}.png')
            io.log(f"  Saving top. charge plot to {out_file}.")
            plt.savefig(out_file, dpi=400, bbox_inches='tight')
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
            out_file = os.path.join(out_dir,
                                    f'top_charge_prob_vs_val_{idx}.png')
            io.log(f'Saving figure to: {out_file}.')
            _ = plt.savefig(out_file, dpi=400, bbox_inches='tight')
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
        _ = plt.savefig(out_file, dpi=400, bbox_inches='tight')
        plt.close('all')

    def _get_plot_dir(self, charges, beta, current_step=None):
        """Returns directory where plots of observables are to be saved."""
        if not HAS_MATPLOTLIB or (self.using_hvd and hvd.rank() != 0):
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

    def _save_run_info(self, observables, stats, _args):
        """Save samples and observables generated from `self.run` call."""
        if not HAS_MATPLOTLIB or (self.using_hvd and hvd.rank() != 0):
            return

        actions_dict = observables[0]
        plaqs_dict = observables[1]
        charges_dict = observables[2]
        charge_diff_dict = observables[3]

        run_steps, current_step, beta, therm_frac = _args
        therm_steps = run_steps // therm_frac

        if current_step is None:
            training = False
        else:
            training = True

        actions_arr = np.array(list(actions_dict.values()))[therm_steps:, :]
        plaqs_arr = np.array(list(plaqs_dict.values()))[therm_steps:, :]

        charges_arr = np.array(list(charges_dict.values()),
                               dtype=np.int32)[therm_steps:, :]

        charges_squared_arr = charges_arr ** 2

        actions_stats = stats[0]
        plaqs_stats = stats[1]
        charges_stats = stats[2]
        suscept_stats = stats[3]
        charge_probs = stats[4]

        files = self._get_run_files(*_args)

        actions_file = files[2]
        plaqs_file = files[3]
        charges_file = files[4]
        charge_diff_file = files[5]
        actions_stats_file = files[6]
        plaqs_stats_file = files[7]
        charges_stats_file = files[8]
        suscept_stats_file = files[9]
        charge_probs_file = files[10]
        statistics_txt_file = files[11]

        def save_data(data, out_file, name=None):
            out_dir = os.path.dirname(out_file)
            io.check_else_make_dir(out_dir)
            try:
                if out_file.endswith('pkl'):
                    io.log(f"Saving {str(name)} to {out_file} "
                           "using `pkl.dump`.")
                    with open(out_file, 'wb') as f:
                        pickle.dump(data, f)
                if out_file.endswith('.npy'):
                    io.log(f"Saving {str(name)} to {out_file} "
                           f"using `np.save`.")
                    np.save(out_file, np.array(data))
            except:
                import pdb

                pdb.set_trace()
                #  raise IOError(f'Unable to save {name} to {out_file}.')

        save_data(actions_dict, actions_file, name='actions')
        save_data(plaqs_dict, plaqs_file, name='plaqs')
        save_data(charges_dict, charges_file, name='charges')
        save_data(charge_diff_dict, charge_diff_file, name='tunneling_events')
        save_data(actions_stats, actions_stats_file, name='actions_stats')
        save_data(plaqs_stats, plaqs_stats_file, name='plaqs_stats')
        save_data(charges_stats, charges_stats_file, name='charges_stats')
        save_data(suscept_stats, suscept_stats_file, name='suscept_stats')
        save_data(charge_probs, charge_probs_file, name='charge_probs')

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
            plaqs_k1: f'{plaqs_avg:.4g} +/- {plaqs_err:.4g}\n',
            _est_key: {}
        }

        def format_stats(avgs, errs, name=None):
            return [
                f'{name}: {a:.6g} +/- {e:.6}' for (a, e) in zip(avgs, errs)
            ]

        keys = [f'sample {idx}' for idx in range(len(suscept_stats[0]))]

        suscept_vals = format_stats(suscept_stats[0], suscept_stats[1],
                                    '< Q^2 >')
        actions_vals = format_stats(actions_stats[0], actions_stats[1],
                                    '< action >')
        plaqs_vals = format_stats(plaqs_stats[0], plaqs_stats[1],
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
        for k, v in charge_probs.items():
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

        io.log(f"Writing statistics to: {statistics_txt_file}")

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
                      statistics_txt_file)
        log_and_write(sep_str1, str1, therm_str, actions_strings,
                      statistics_txt_file)
        log_and_write(sep_str2, str2, therm_str, plaqs_strings,
                      statistics_txt_file)
        log_and_write(sep_str3, str3, therm_str, charge_probs_strings,
                      statistics_txt_file)


    def _get_run_files(self, *_args):
        """Create dir and files for storing observables from `self.run`."""
        if not HAS_MATPLOTLIB or (self.using_hvd and hvd.rank() != 0):
            return
        run_steps, current_step, beta, _ = _args

        observables_dir = os.path.join(self.eval_dir, 'observables')
        io.check_else_make_dir(observables_dir)

        if current_step is None:                     # running AFTER training
            obs_dir = os.path.join(observables_dir,
                                   f'steps_{run_steps}_beta_{beta}')
            io.check_else_make_dir(obs_dir)

            npy_file = f'samples_history_steps_{run_steps}_beta_{beta}.npy'
            json_file = f'samples_history_steps_{run_steps}_beta_{beta}.json'
            samples_file_npy = os.path.join(self.samples_dir, npy_file)
            samples_file_json = os.path.join(self.samples_dir, json_file)

        else:                                        # running DURING training
            obs_dir = os.path.join(observables_dir, 'training')
            io.check_else_make_dir(obs_dir)
            obs_dir = os.path.join(obs_dir,
                                   f'{current_step}_TRAIN_'
                                   f'steps_{run_steps}_beta_{beta}')

            npy_file = (f'samples_history_{current_step}_TRAIN_'
                        f'steps_{run_steps}_beta_{beta}.npy')
            json_file = (f'samples_history_{current_step}_TRAIN_'
                         f'steps_{run_steps}_beta_{beta}.json')
            samples_file_npy = os.path.join(self.train_samples_dir, npy_file)
            samples_file_json = os.path.join(self.train_samples_dir, json_file)

        actions_file = os.path.join(
            obs_dir, f'actions_steps_{run_steps}_beta_{beta}.pkl'
        )
        plaqs_file = os.path.join(
            obs_dir, f'plaqs_steps_{run_steps}_beta_{beta}.pkl'
        )
        charges_file = os.path.join(
            obs_dir, f'charges_steps_{run_steps}_beta_{beta}.pkl'
        )
        charge_diff_file = os.path.join(
            obs_dir, f'charge_diff_{run_steps}_beta_{beta}.pkl'
        )
        actions_stats_file = os.path.join(
            obs_dir, f'actions_stats_steps_{run_steps}_beta_{beta}.pkl'
        )
        plaqs_stats_file = os.path.join(
            obs_dir, f'plaqs_stats_steps_{run_steps}_beta_{beta}.pkl'
        )
        charges_stats_file = os.path.join(
            obs_dir, f'charges_stats_steps_{run_steps}_beta_{beta}.pkl'
        )
        suscept_stats_file = os.path.join(
            obs_dir, f'suscept_stats_steps_{run_steps}_beta_{beta}.pkl'
        )
        charge_probs_file = os.path.join(
            obs_dir, f'charge_probs_steps_{run_steps}_beta_{beta}.pkl'
        )
        statistics_txt_file = os.path.join(
            obs_dir, f'statistics_steps_{run_steps}_beta_{beta}.txt'
        )

        files = (samples_file_npy, samples_file_json, actions_file, plaqs_file,
                 charges_file, charge_diff_file, actions_stats_file,
                 plaqs_stats_file, charges_stats_file, suscept_stats_file,
                 charge_probs_file, statistics_txt_file)

        return files


# pylint: disable=too-many-statements, too-many-branches
def main(FLAGS):
    """Main method for creating/training U(1) gauge model from command line."""
    if HAS_HOROVOD and FLAGS.horovod:
        io.log("INFO: USING HOROVOD")
        hvd.init()

    #  params = PARAMS  # use default parameters if no command line args passed
    params = {}

    for key, val in FLAGS.__dict__.items():
        params[key] = val

    if FLAGS.hmc:
        params['eps_trainable'] = False
        beta1 = params.get('beta', 4.)
        beta2 = params.get('beta_init', 4.)
        beta3 = params.get('beta_final', 4.)
        beta = max((beta1, beta2, beta3))

        params['beta'] = beta
        params['beta_init'] = beta
        params['beta_final'] = beta

    config = tf.ConfigProto()
    if FLAGS.time_size > 8:
        off = rewriter_config_pb2.RewriterConfig.OFF
        config.graph_options.rewrite_options.arithmetic_optimization = off

    if FLAGS.gpu:
        print("Using gpu for training.")
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
        #  os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
        # NOTE: KMP affinity taken care of by passing -cc depth to aprun call
        OMP_NUM_THREADS = 62
        config.allow_soft_placement = True
        config.intra_op_parallelism_threads = OMP_NUM_THREADS
        config.inter_op_parallelism_threads = 1

    #  if HAS_HOROVOD and FLAGS.horovod:
        #  params['lr_init'] *= hvd.size()
        #  params['train_steps'] /= hvd.size()
        #  params['lr_decay_steps'] /= hvd.size()

    model = GaugeModel(params=params,
                       config=config,
                       sess=None,
                       log_dir=FLAGS.log_dir,
                       restore=FLAGS.restore)

    io.log('\n\n\n')
    io.log(80*'~')
    if not model.using_hvd:
        io.log(f"model.log_dir: {model.log_dir}")
    io.log(80*'~')
    io.log('\n\n\n')

    #  if not FLAGS.horovod or (FLAGS.horovod and hvd.rank() == 0):
    #  if not model.using_hvd or (model.using_hvd and hvd.rank() == 0):
    #      io.save_params_to_pkl_file(params, model.info_dir)

    if FLAGS.horovod:
        io.log('Number of CPUs: %d' % hvd.size())

    #  if FLAGS.horovod:
    #      if hvd.rank() == 0:
    #          io.save_params_to_pkl_file(params, model.info_dir)
    #  else:
    #      io.save_params_to_pkl_file(params, model.info_dir)

    io.log(f"Training began at: {time.ctime()}")

    if not FLAGS.hmc:
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
                model.train(params['train_steps'], samples_init=None,
                            beta_init=None, trace=FLAGS.trace)

    try:
        run_steps = 5e4
        model.run(run_steps, beta=model.beta_final)
        model.run(run_steps, beta=model.beta_final - 1)
        #  run_steps_grid = [100, 500, 1000, 2500, 5000, 10000]
        #  run_steps_grid = [20000, 50000]
        #  run_steps = 50000
        #  run_steps_grid = [50000]
        #  betas = [model.beta_final - 1, model.beta_final]
        #  betas = [model.beta_final]
        #  for steps in run_steps_grid:
        #      for beta1 in betas:
        #          model.run(steps, beta=beta1)

    except (KeyboardInterrupt, SystemExit):
        io.log("\nKeyboardInterrupt detected! \n")

        import pdb

        pdb.set_trace()


# =============================================================================
#  * NOTE:
#      - if action == 'store_true':
#          The argument is FALSE by default. Passing this flag will cause the
#          argument to be ''stored true''.
#      - if action == 'store_false':
#          The argument is TRUE by default. Passing this flag will cause the
#          argument to be ''stored false''.
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('L2HMC model using U(1) lattice gauge theory for target '
                     'distribution.'),
        fromfile_prefix_chars='@',
    )
# ========================= Lattice parameters ===============================
    parser.add_argument("--space_size", type=int, default=8,
                        required=False, dest="space_size",
                        help="Spatial extent of lattice. (Default: 8)")

    parser.add_argument("--time_size", type=int, default=8,
                        required=False, dest="time_size",
                        help="Temporal extent of lattice. (Default: 8)")

    parser.add_argument("--link_type", type=str, required=False,
                        default='U1', dest="link_type",
                        help="Link type for gauge model. (Default: U1)")

    parser.add_argument("--dim", type=int, required=False,
                        default=2, dest="dim",
                        help="Dimensionality of lattice (Default: 2)")

    parser.add_argument("--num_samples", type=int, default=10,
                        required=False, dest="num_samples",
                        help=("Number of samples (batch size) to use for "
                              "training. (Default: 2)"))

    parser.add_argument("--rand", action="store_true",
                        required=False, dest="rand",
                        help=("Start lattice from randomized initial "
                              "configuration. (Default: False)"))

# ========================= Leapfrog parameters ==============================

    parser.add_argument("-n", "--num_steps", type=int,
                        default=5, required=False, dest="num_steps",
                        help=("Number of leapfrog steps to use in (augmented) "
                              "HMC sampler. (Default: 5)"))

    parser.add_argument("--eps", type=float, default=0.1,
                        required=False, dest="eps",
                        help=("Step size to use in leapfrog integrator. "
                              "(Default: 0.1)"))

    parser.add_argument("--loss_scale", type=float, default=1.,
                        required=False, dest="loss_scale",
                        help=("Scaling factor to be used in loss function. "
                              "(lambda in Eq. 7 of paper). (Default: 1.)"))

# ========================= Learning rate parameters ==========================

    parser.add_argument("--lr_init", type=float, default=1e-3,
                        required=False, dest="lr_init",
                        help=("Initial value of learning rate. "
                              "(Deafult: 1e-3)"))

    parser.add_argument("--lr_decay_steps", type=int, default=500,
                        required=False, dest="lr_decay_steps",
                        help=("Number of steps after which to decay learning "
                              "rate. (Default: 500)"))

    parser.add_argument("--lr_decay_rate", type=float, default=0.96,
                        required=False, dest="lr_decay_rate",
                        help=("Learning rate decay rate to be used during "
                              "training. (Default: 0.96)"))

# ========================= Annealing rate parameters ========================

    parser.add_argument("--annealing", action="store_true",
                        required=False, dest="annealing",
                        help=("Flag that when passed will cause the model "
                              "to perform simulated annealing during "
                              "training. (Default: False)"))

    #  parser.add_argument("--annealing_steps", type=float, default=200,
    #                      required=False, dest="annealing_steps",
    #                      help=("Number of steps after which to anneal
    #                      beta."))
    #
    #  parser.add_argument("--annealing_factor", type=float, default=0.97,
    #                      required=False, dest="annealing_factor",
    #                      help=("Factor by which to anneal beta."))
    #
    #  parser.add_argument("-b", "--beta", type=float,
    #                      required=False, dest="beta",
    #                      help=("Beta (inverse coupling constant) used in "
    #                            "gauge model. (Default: 8.)"))

    parser.add_argument("--beta_init", type=float, default=1.,
                        required=False, dest="beta_init",
                        help=("Initial value of beta (inverse coupling "
                              "constant) used in gauge model when annealing. "
                              "(Default: 1.)"))

    parser.add_argument("--beta_final", type=float, default=8.,
                        required=False, dest="beta_final",
                        help=("Final value of beta (inverse coupling "
                              "constant) used in gauge model when annealing. "
                              "(Default: 8.)"))

# ========================== Training parameters ==============================

    parser.add_argument("--train_steps", type=int, default=1000,
                        required=False, dest="train_steps",
                        help=("Number of training steps to perform. "
                              "(Default: 1000)"))

    parser.add_argument("--trace", action="store_true",
                        required=False, dest="trace",
                        help=("Flag that when passed will create trace during "
                              "training loop."))

    parser.add_argument("--save_steps", type=int, default=50,
                        required=False, dest="save_steps",
                        help=("Number of steps after which to save the model "
                              "and current values of all parameters. "
                              "(Default: 50)"))

    parser.add_argument("--print_steps", type=int, default=1,
                        required=False, dest="print_steps",
                        help=("Number of steps after which to display "
                              "information about the loss and various "
                              "other quantities (Default: 1)"))

    parser.add_argument("--logging_steps", type=int, default=50,
                        required=False, dest="logging_steps",
                        help=("Number of steps after which to write logs for "
                              "tensorboard. (Default: 50)"))

    parser.add_argument("--training_samples_steps", type=int, default=1000,
                        required=False, dest="training_samples_steps",
                        help=("Number of intermittent steps after which "
                              "the sampler is evaluated at `beta_final`. "
                              "This allows us to monitor the performance of "
                              "the sampler during training. (Default: 500)"))

    parser.add_argument("--training_samples_length", type=int, default=500,
                        required=False, dest="training_samples_length",
                        help=("Number of steps to run sampler for when "
                              "evaluating the sampler during training. "
                              "(Default: 100)"))

# ========================== Model parameters ================================

    parser.add_argument('--network_arch', type=str, default='conv3D',
                        required=False, dest='network_arch',
                        help=("String specifying the architecture to use for "
                              "the neural network. Must be one of: "
                              "`'conv3D', 'conv2D', 'generic'`. "
                              "(Default: conv3D)"))

    parser.add_argument('--summaries', action="store_true",
                        required=False, dest="summaries",
                        help=("Flag that when passed creates "
                              "summaries of gradients and variables for "
                              "monitoring in tensorboard. (Default: False)"))

    #  parser.add_argument("--conv_net", action="store_true",
    #                      required=False, dest="conv_net",
    #                      help=("Whether or not to use convolutional "
    #                            "neural network for pre-processing lattice "
    #                            "configurations (prepended to generic FC net "
    #                            "as outlined in paper). (Default: False)"))

    parser.add_argument("--hmc", action="store_true",
                        required=False, dest="hmc",
                        help=("Use generic HMC (without augmented leapfrog "
                              "integrator described in paper). Used for "
                              "comparing against L2HMC algorithm. "
                              "(Default: False)"))

    parser.add_argument("--eps_trainable", action="store_true",
                        required=False, dest="eps_trainable",
                        help=("Flag that when passed will allow the step size "
                              "`eps` to be a trainable parameter."))

    parser.add_argument("--metric", type=str, default="cos_diff",
                        required=False, dest="metric",
                        help=("Metric to use in loss function. "
                              "(Default: `l2`, choices: [`l2`, `l1`, `cos`])"))

    #  parser.add_argument("--charge_loss", action="store_true",
    #                      required=False, dest="charge_loss",
    #                      help=("Flag then when passed will modify the loss "
    #                            "function to include an additional term that "
    #                            "measures the difference in topological charge "
    #                            "between the original and proposed sample."))

    #  parser.add_argument("--aux", action="store_true",
    #                      required=False, dest="aux",
    #                      help=("Include auxiliary function `q` for calculating "
    #                            "expected squared jump distance conditioned on "
    #                            "initialization distribution. (Default: False)"))

    parser.add_argument("--std_weight", type=float, default=1.,
                        required=False, dest="std_weight",
                        help=("Multiplicative factor used to weigh relative "
                              "strength of stdiliary term in loss function. "
                              "(Default: 1.)"))

    parser.add_argument("--aux_weight", type=float, default=1.,
                        required=False, dest="aux_weight",
                        help=("Multiplicative factor used to weigh relative "
                              "strength of auxiliary term in loss function. "
                              "(Default: 1.)"))

    parser.add_argument("--charge_weight", type=float, default=1.,
                        required=False, dest="charge_weight",
                        help=("Multiplicative factor used to weigh relative "
                              "strength of chargeiliary term in loss function. "
                              "(Default: 1.)"))

    parser.add_argument("--clip_grads", action="store_true",
                        required=False, dest="clip_grads",
                        help=("Flag that when passed will clip gradients by "
                              "global norm using `--clip_value` command line "
                              "argument. If `--clip_value` is not passed, "
                              "it defaults to 100."))

    parser.add_argument("--clip_value", type=float, default=1.,
                        required=False, dest="clip_value",
                        help=("Clip value, used for clipping value of "
                              "gradients by global norm. (Default: 1.)"))

    parser.add_argument("--log_dir", type=str, default=None,
                        required=False, dest="log_dir",
                        help=("Log directory to use from previous run. "
                              "If this argument is not passed, a new "
                              "directory will be created. (Default: None)"))

    parser.add_argument("--restore", action="store_true",
                        required=False, dest="restore",
                        help=("Restore model from previous run. "
                              "If this argument is passed, a `log_dir` "
                              "must be specified and passed to `--log_dir` "
                              "argument. (Default: False)"))

    parser.add_argument("--profiler", action="store_true",
                        required=False, dest='profiler',
                        help=("Flag that when passed will profile the graph "
                              "execution using `TFProf`. (Default: False)"))

    parser.add_argument("--gpu", action="store_true",
                        required=False, dest="gpu",
                        help=("Flag that when passed indicates we're training "
                              "using an NVIDIA GPU."))

    parser.add_argument("--theta", action="store_true",
                        required=False, dest="theta",
                        help=("Flag that when passed indicates we're training "
                              "on theta @ ALCf."))

    parser.add_argument("--horovod", action="store_true",
                        required=False, dest="horovod",
                        help=("Flag that when passed uses Horovod for "
                              "distributed training on multiple nodes."))

    parser.add_argument("--num_intra_threads", type=int, default=0,
                        required=False, dest="num_intra_threads",
                        help=("Number of intra op threads to use for "
                              "tf.ConfigProto.intra_op_parallelism_threads"))

    parser.add_argument("--num_inter_threads", type=int, default=0,
                        required=False, dest="num_intra_threads",
                        help=("Number of intra op threads to use for "
                              "tf.ConfigProto.intra_op_parallelism_threads"))

    if sys.argv[1].startswith('@'):
        args = parser.parse_args(shlex.split(open(sys.argv[1][1:]).read(),
                                             comments=True))
    else:
        args = parser.parse_args()

    main(args)
