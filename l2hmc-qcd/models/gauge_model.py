"""
gauge_model.py

Implements GaugeModel class responsible for building computation graph used in
tensorflow.

Author: Sam Foreman (github: @saforem2)
Date: 04/12/2019
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False

import utils.file_io as io

from globals import GLOBAL_SEED, TF_FLOAT
from lattice.lattice import GaugeLattice
from dynamics.gauge_dynamics import GaugeDynamics

from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorflow.core.protobuf import rewriter_config_pb2

np.random.seed(GLOBAL_SEED)

if float(tf.__version__.split('.')[0]) <= 2:
    tf.set_random_seed(GLOBAL_SEED)
    tf.logging.set_verbosity(tf.logging.INFO)


def check_log_dir(log_dir):
    """Check log_dir for existing checkpoints."""
    assert os.path.isdir(log_dir)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    if os.path.isdir(ckpt_dir):
        pass


class GaugeModel:
    def __init__(self, params=None):
        # -------------------------------------------------------
        # Create attributes from (key, val) pairs in params
        # -------------------------------------------------------
        self.loss_weights = {}
        self.params = params
        for key, val in self.params.items():
            if 'weight' in key:
                self.loss_weights[key] = val
            else:
                setattr(self, key, val)

        # -------------------------------------------------------
        # Create lattice
        # -------------------------------------------------------
        self.lattice, self.samples = self._create_lattice()

        self.batch_size = self.lattice.samples.shape[0]
        self.x_dim = self.lattice.num_links

        # -------------------------------------------------------
        # Create input placeholders
        # -------------------------------------------------------
        self.x, self.beta = self._create_inputs()

        # -------------------------------------------------------
        # Create dynamics engine
        # -------------------------------------------------------
        self.dynamics, self.potential_fn = self._create_dynamics(
            self.lattice,
            self.samples
        )

        # -------------------------------------------------------
        # Create metric function used in loss
        # -------------------------------------------------------
        self.metric_fn = self._create_metric_fn(self.metric)

        # -------------------------------------------------------
        # Create operations for calculating plaquette observables.
        # -------------------------------------------------------
        obs_ops = self._create_observables()
        self.plaq_sums_op = obs_ops['plaq_sums']
        self.actions_op = obs_ops['actions']
        self.plaqs_op = obs_ops['plaqs']
        self.avg_plaqs_op = obs_ops['avg_plaqs']
        self.charges_op = obs_ops['charges']

        # -------------------------------------------------------
        # Create optimizer, build graph, create / init. saver
        # -------------------------------------------------------
        if self.hmc:
            self.create_sampler()
        else:
            self._create_optimizer()
            self.build()
            self.init_saver()

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=3)

    def save(self, sess, checkpoint_dir):
        """Save model checkpoint to checkpoint directory."""
        io.log(f"INFO: Saving model to: {checkpoint_dir}")
        self.saver.save(sess, checkpoint_dir, self.global_step)
        io.log("Model saved.")

    def load(self, sess, checkpoint_dir):
        latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            io.log(f"INFO: Loading model from {latest_ckpt}...\n")
            self.saver.restore(sess, latest_ckpt)
            io.log("Model loaded.")

    def _create_lattice(self):
        """Create GaugeLattice object."""
        with tf.name_scope('lattice'):
            lattice = GaugeLattice(time_size=self.time_size,
                                   space_size=self.space_size,
                                   dim=self.dim,
                                   link_type=self.link_type,
                                   num_samples=self.num_samples,
                                   rand=self.rand)

            assert lattice.samples.shape[0] == self.num_samples

            samples = tf.convert_to_tensor(lattice.samples, dtype=TF_FLOAT)

        return lattice, samples

    def _create_observables(self):
        """Create operations for calculating lattice observables."""
        obs_ops = {}
        with tf.name_scope('plaq_observables'):
            with tf.name_scope('plaq_sums'):
                obs_ops['plaq_sums'] = self.lattice.calc_plaq_sums(self.x)

            with tf.name_scope('actions'):
                obs_ops['actions'] = self.lattice.calc_actions(self.x)

            with tf.name_scope('avg_plaqs'):
                plaqs = self.lattice.calc_plaqs(self.x)
                avg_plaqs = tf.reduce_mean(plaqs, name='avg_plaqs')

                obs_ops['plaqs'] = plaqs
                obs_ops['avg_plaqs'] = avg_plaqs

            with tf.name_scope('top_charges'):
                obs_ops['charges'] = self.lattice.calc_top_charges(self.x,
                                                                   fft=False)

        #  return plaq_sums_op, actions_op, plaqs_op, charges_op
        return obs_ops

    def _create_inputs(self):
        """Create input placeholders `x` and `beta`.  """
        with tf.name_scope('inputs'):
            if not tf.executing_eagerly():
                x = tf.placeholder(dtype=TF_FLOAT,
                                   shape=(self.batch_size, self.x_dim),
                                   name='x')
                beta = tf.placeholder(dtype=TF_FLOAT,
                                      shape=(),
                                      name='beta')
            else:
                x = self.lattice.samples,
                beta = self.beta_init

        return x, beta

    def _create_dynamics(self, lattice, samples, **kwargs):
        """Initialize dynamics object."""
        with tf.name_scope('dynamics'):
            dynamics_kwargs = {  # default values of keyword arguments
                'eps': self.eps,
                'hmc': self.hmc,
                'network_arch': self.network_arch,
                'num_steps': self.num_steps,
                'eps_trainable': self.eps_trainable,
                'data_format': self.data_format,
                'use_bn': self.use_bn
            }

            dynamics_kwargs.update(kwargs)
            potential_fn = lattice.get_potential_fn(samples)
            dynamics = GaugeDynamics(lattice=lattice,
                                     potential_fn=potential_fn,
                                     **dynamics_kwargs)

        return dynamics, potential_fn

    @staticmethod
    def _create_metric_fn(metric):
        """Create metric fn for measuring the distance between two samples."""
        with tf.name_scope('metric_fn'):
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
                    return tf.abs(tf.cos(x1) - tf.cos(x2))

            elif metric == 'cos_diff':
                def metric_fn(x1, x2):
                    return 1. - tf.cos(x1 - x2)
            else:
                raise AttributeError(f"metric={metric}. Expected one of: "
                                     "'l1', 'l2', 'cos', 'cos2' or 'cos_diff'")

        return metric_fn

    def calc_loss(self, x, beta, **weights):
        """Create operation for calculating the loss.

        Args:
            x: Input tensor of shape (self.num_samples,
                self.lattice.num_links) containing batch of GaugeLattice links
                variables.
            beta (float): Inverse coupling strength.

        Returns:
            loss (float): Operation responsible for calculating the total loss.
            px (np.ndarray): Array of acceptance probabilities from
                Metropolis-Hastings accept/reject step. Has shape:
                (self.num_samples,)
            x_out: Output samples obtained after Metropolis-Hastings
                accept/reject step.

        NOTE: If proposed configuration is accepted following
            Metropolis-Hastings accept/reject step, x_proposed and x_out are
            equivalent.
        """
        with tf.name_scope('x_update'):
            x_proposed, _, px, x_out = self.dynamics(x, beta)
        with tf.name_scope('z_update'):
            z = tf.random_normal(tf.shape(x), name='z')  # Auxiliary variable
            z_proposed, _, pz, _ = self.dynamics(z, beta)

        with tf.name_scope('top_charge_diff'):
            x_dq = tf.cast(
                self.lattice.calc_top_charges_diff(x, x_out, fft=False),
                dtype=tf.int32
            )

        # Add eps for numerical stability; following released implementation
        # NOTE:
        #  std:_loss: 'standard' loss
        #  charge_loss: Contribution from the difference in topological charge
        #    betweween the initial and proposed configurations  to the total
        #     loss.
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

    def _calc_std_loss(self, x_tup, z_tup, p_tup, **weights):
        """Calculate standard contribution to loss.

        NOTE: In contrast to the original paper where the L2 difference was
        used, we are now using 1 - cos(x1 - x2).

        Args:
            x_tup: Tuple of (x, x_proposed) configurations.
            z_tup: Tuple of (z, z_proposed) configurations.
            p_tup: Tuple of (x, z) acceptance probabilities.
            weights: dictionary of weights giving relative weight of each term
                in the loss function.

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

        Args:
            x_tup: Tuple of (x, x_proposed) configurations.
            z_tup: Tuple of (z, z_proposed) configurations.
            p_tup: Tuple of (x, z) acceptance probabilities.
            weights: dictionary of weights giving relative weight of each term
                in the loss function.

        Returns:
            std_loss
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

    def calc_loss_and_grads(self, x, beta, **weights):
        """Calculate loss its gradient with respect to all trainable variables.

        Args:
            x: Placeholder (tensor object)representing batch of GaugeLattice
                link variables.
            beta: Placeholder (tensor object) representing inverse coupling
            strength.

        Returns:
            loss: Operation for calculating the total loss.
            grads: Tensor containing the gradient of the loss function with
                respect to all trainable variables.
            x_out: Operation for obtaining new samples (i.e. output of
                augmented L2HMC algorithm.)
            accept_prob: Operation for calculating acceptance probabilities
                used in Metropolis-Hastings accept reject.
            x_dq: Operation for calculating the topological charge difference
                between the initial and proposed configurations.
        """
        if tf.executing_eagerly():
            with tf.name_scope('grads'):
                with tf.GradientTape() as tape:
                    loss, x_out, accept_prob, x_dq = self.calc_loss(x, beta,
                                                                    **weights)
                grads = tape.gradient(loss, self.dynamics.trainable_variables)
        else:
            loss, x_out, accept_prob, x_dq = self.calc_loss(x, beta, **weights)

            with tf.name_scope('grads'):
                grads = tf.gradients(loss, self.dynamics.trainable_variables)
                if self.clip_grads:
                    grads, _ = tf.clip_by_global_norm(grads, self.clip_value)

        return loss, grads, x_out, accept_prob, x_dq

    def create_sampler(self):
        """Create operation for generating new samples using dynamics engine.

        NOTE: This method is to be used when running generic HMC to create
        operations for generating new sampler.
        """
        with tf.name_scope('sampler'):
            _, _, self.px, self.x_out = self.dynamics(self.x, self.beta)
            x_dq = self.lattice.calc_top_charges_diff(self.x, self.x_out,
                                                      fft=False)

            self.charge_diffs_op = tf.reduce_sum(x_dq) / self.num_samples

    def _create_optimizer(self, lr_init=None):
        """Create learning rate and optimizer."""
        if self.hmc:
            return

        if lr_init is None:
            lr_init = self.lr_init

        with tf.name_scope('global_step'):
            self.global_step = tf.train.get_or_create_global_step()
            #  self.global_step.assign(1)

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

    def build(self):
        """Build Tensorflow graph."""
        with tf.name_scope('loss'):
            output = self.calc_loss_and_grads(x=self.x,
                                              beta=self.beta,
                                              **self.loss_weights)
            self.loss_op, self.grads, self.x_out, self.px, x_dq = output
            self.charge_diffs_op = tf.reduce_sum(x_dq) / self.num_samples

        with tf.name_scope('train'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #  ! no update ops in the default graph
            io.log("update_ops: ", update_ops)
            # Use the update ops of the model itself
            io.log("model.updates: ", self.dynamics.updates)

            grads_and_vars = zip(self.grads, self.dynamics.trainable_variables)
            with tf.control_dependencies(self.dynamics.updates):
                self.train_op = self.optimizer.apply_gradients(
                    grads_and_vars,
                    global_step=self.global_step,
                    name='train_op'
                )
