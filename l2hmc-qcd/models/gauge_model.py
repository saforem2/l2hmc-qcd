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
            if 'weight' in key and key != 'charge_weight':
                self.loss_weights[key] = val
            else:
                setattr(self, key, val)

        io.log(80 * '-')
        io.log(f'Args received by `GaugeModel`:')
        for key, val in params.items():
            io.log(f'{key}: {val}')
        io.log(80 * '-')

        # -------------------------------------------------------
        # Create lattice
        # -------------------------------------------------------
        self.lattice, self.samples = self._create_lattice()

        self.batch_size = self.lattice.samples.shape[0]
        self.x_dim = self.lattice.num_links

        # -------------------------------------------------------
        # Create input placeholders:
        #   (x, beta, charge_weight, net_weights)
        # -------------------------------------------------------
        inputs = self._create_inputs()
        self.x = inputs['x']
        self.beta = inputs['beta']
        self.charge_weight = inputs['charge_weight']
        self.net_weights = inputs['net_weights']

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
        # Create operations for calculating plaquette observables
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
        """Create input paceholders (if not executing eagerly).

        Returns:
            outputs: Dictionary with the following entries:
                x: Placeholder for input lattice configuration with 
                    shape = (batch_size, x_dim) where x_dim is the number of
                    links on the lattice and is equal to lattice.time_size *
                    lattice.space_size * lattice.dim.
                beta: Placeholder for inverse coupling constant.
                charge_weight: Placeholder for the charge_weight (i.e. alpha_Q,
                    the multiplicative factor that scales the topological
                    charge term in the modified loss function) .
                net_weights: Array of placeholders, each of which is a
                    multiplicative constant used to scale the effects of the
                    various S, Q, and T functions from the original paper.
                    net_weights[0] = 'scale_weight', multiplies the S fn.
                    net_weights[1] = 'transformation_weight', multiplies the Q
                    fn.  net_weights[2] = 'translation_weight', multiplies the
                    T fn.
        """
        with tf.name_scope('inputs'):
            if not tf.executing_eagerly():
                x = tf.placeholder(dtype=TF_FLOAT,
                                   shape=(self.batch_size, self.x_dim),
                                   name='x_placeholder')
                beta = tf.placeholder(dtype=TF_FLOAT,
                                      shape=(),
                                      name='beta')
                charge_weight = tf.placeholder(dtype=TF_FLOAT,
                                               shape=(),
                                               name='charge_weight')
                net_weights = [
                    tf.placeholder(
                        dtype=TF_FLOAT, shape=(), name='scale_weight'
                    ),
                    tf.placeholder(
                        dtype=TF_FLOAT, shape=(), name='transformation_weight'
                    ),
                    tf.placeholder(
                        dtype=TF_FLOAT, shape=(), name='translation_weight'
                    )
                ]
            else:
                x = tf.convert_to_tensor(
                    self.lattice.samples.reshape((self.batch_size, self.x_dim))
                )
                beta = tf.convert_to_tensor(self.beta_init)
                charge_weight = tf.convert_to_tensor(0.)
                net_weights = tf.convert_to_tensor([1., 1., 1.])

        outputs = {
            'x': x,
            'beta': beta,
            'charge_weight': charge_weight,
            'net_weights': net_weights
        }

        return outputs

    def _create_dynamics(self, lattice, samples, **kwargs):
        """Initialize dynamics object."""
        with tf.name_scope('dynamics'):
            # default values of keyword arguments
            dynamics_kwargs = {
                'eps': self.eps,
                'hmc': self.hmc,
                'network_arch': self.network_arch,
                'num_steps': self.num_steps,
                'eps_trainable': self.eps_trainable,
                'data_format': self.data_format,
                'use_bn': self.use_bn,
                'num_hidden': self.num_hidden,
                #  'scale_weight': self.scale_weight,
                #  'transformation_weight': self.transformation_weight,
                #  'translation_weight': self.translation_weight
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
            elif metric == 'tan_cos':
                def metric_fn(x1, x2):
                    cos_diff = 1. - tf.cos(x1 - x2)
                    return tf.tan(np.pi * cos_diff / 2)
            else:
                raise AttributeError(f"""metric={metric}. Expected one of:
                                     'l1', 'l2', 'cos', 'cos2', 'cos_diff', or
                                     'tan_cos'.""")

        return metric_fn

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
            # Define update operations for batch normalization    
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # Define optimizer
            with tf.control_dependencies(self.update_ops):
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                if self.using_hvd:
                    self.optimizer = hvd.DistributedOptimizer(self.optimizer)

    def _calc_std_loss(self, config_init, config_proposed, accept_prob):
        """Calculate the (individual) standard contribution to the loss fn."""
        eps = 1e-4
        with tf.name_scope('_std_loss'):
            summed_diff = tf.reduce_sum(
                self.metric_fn(config_init, config_proposed), axis=1
            )

        return summed_diff * accept_prob + eps

    def calc_std_loss(self, inputs, **weights):
        """Calculate the standard contribution to the loss.

        Explicitly: This is calculated as the expectation value of the jump
        loss over both the target and initialization distributions.

        Args: 
            x_tup: Tuple containing (x_init, x_proposed)
            z_tup: Tuple containing (z_init, z_proposed) (aux. variable)
            p_tup: Tuple containing (px, pz) acceptance probabilities.
            **weights: Dictionary of weights used as multiplicative scaling
                factors to control the contribution from individual terms to
                the total loss. 
        Returns:
            std_loss: Tensorflow operation responsible for calculating the
                standard contribution to the loss function.
        """
        aux_weight = weights.get('aux_weight', 1.)
        std_weight = weights.get('std_weight', 1.)

        x_init = inputs.get('x_init', None)
        x_proposed = inputs.get('x_proposed', None)

        z_init = inputs.get('z_init', None)
        z_proposed = inputs.get('z_proposed', None)

        px = inputs.get('px', None)
        pz = inputs.get('pz', None)

        with tf.name_scope('x_std_loss'):
            x_loss = self._calc_std_loss(x_init, x_proposed, px)
            x_loss_tot = tf.reduce_mean((self.loss_scale / x_loss
                                        - x_loss / self.loss_scale),
                                        axis=0, name='x_std_loss_tot')
            tf.add_to_collection('losses', x_loss_tot)

        if aux_weight > 0 and z_init is not None:
            with tf.name_scope('z_std_loss'):
                z_loss = self._calc_std_loss(z_init, z_proposed, pz)
                z_loss_tot = tf.reduce_mean((self.loss_scale / z_loss
                                            - z_loss / self.loss_scale),
                                            axis=0, name='z_std_loss_tot')
                tf.add_to_collection('losses', z_loss_tot)
        else:
            z_loss_tot = 0.

        with tf.name_scope('std_loss'):
            loss_tot = tf.multiply(std_weight,
                                   (x_loss_tot + aux_weight * z_loss_tot),
                                   name='std_loss_tot')
            tf.add_to_collection('losses', loss_tot)

        return loss_tot

    def _calc_charge_loss(self, config_init, config_proposed, accept_prob):
        """Calculate the (individual) top. charge contribution to the loss fn

        Args:
            config_init: Initial configuration.
            config_proposed: Proposed configuration.
            accept_prob: Likelihood of accepting the proposed configuration,
                usd to calculate the expectation value of the topological
                charge difference. 
        Returns:
            charge_loss: Individual contribution to the loss function from the
                topological charge difference between the initial and proposed
                configurations.
        """
        with tf.name_scope('charge_diff'):
            charge_diff = self.lattice.calc_top_charges_diff(config_init,
                                                             config_proposed,
                                                             fft=True)
        return accept_prob * charge_diff

    def calc_charge_loss(self, inputs, **weights):
        """Calculate the (individual) top. charge contribution to the loss fn

        Args:
            inputs: Dictionary containing initial and proposed configurations
                sampled from both the target ('x_init', 'x_proposed') and
                initialization ('z_init', 'z_proposed') distributions, as well
                as the calculated acceptance probabilities ('px', and 'pz').
            weights: Dictionary containing various multiplicative weights used
                to scale the contribution from individual terms to the total
                loss function
        Returns:
            charge_loss: Total contribution to the loss function from the
                topological charge difference between initial and proposed
                configuratons.
        """
        x_init = inputs.get('x_init', None)
        x_proposed = inputs.get('x_proposed', None)

        z_init = inputs.get('z_init', None)
        z_proposed = inputs.get('z_proposed', None)

        px = inputs.get('px', None)
        pz = inputs.get('pz', None)

        aux_weight = weights.get('aux_weight', 1.)

        # Calculate the difference in topological charge between the initial
        # and proposed configurations multiplied by the probability of
        # acceptance to get the expected value of the difference in top. charge
        with tf.name_scope('x_charge_loss'):
            x_charge_loss = self._calc_charge_loss(x_init, x_proposed, px)
            tf.add_to_collection('losses', x_charge_loss)

        with tf.name_scope('z_charge_loss'):
            if aux_weight > 0 and z_init is not None:
                z_charge_loss = self._calc_charge_loss(z_init, z_proposed, pz)
            else:
                z_charge_loss = 0.
            tf.add_to_collection('losses', x_charge_loss)

        with tf.name_scope('total_charge_loss'):
            charge_loss = self.charge_weight * (x_charge_loss + z_charge_loss)
            charge_loss = tf.reduce_mean(charge_loss, axis=0,
                                         name='charge_loss')

            tf.add_to_collection('losses', charge_loss)

        return charge_loss

    def calc_loss(self, x, beta, net_weights, **weights):
        """Create operation for calculating the loss.

        Args:
            x: Input tensor of shape (self.num_samples,
                self.lattice.num_links) containing batch of GaugeLattice links
                variables.
            beta (float): Inverse coupling strength.
            net_weights: Tuple containing multiplicative weights that scale the
                contributions from the scale (S), transformation (Q) and
                translation (T) functions.
            weights: Dictionary containing various multiplicative weights used
                to scale the contribution from individual terms to the total
                loss function

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
            x_dynamics_output = self.dynamics(x, beta, net_weights,
                                              while_loop=self.while_loop,
                                              #  v_in=None,  # v_in
                                              save_lf=self.save_lf)
            #  x_proposed = tf.mod(dynamics_output['x_proposed'], 2 * np.pi)
            #  x_out = tf.mod(x_dynamics_output['x_out'], 2 * np.pi)
            x_proposed = x_dynamics_output['x_proposed']
            px = x_dynamics_output['accept_prob']
            x_out = x_dynamics_output['x_out']

        # Auxiliary variable
        if weights['aux_weight'] > 0:
            with tf.name_scope('z_update'):
                z = tf.random_normal(tf.shape(x), seed=GLOBAL_SEED, name='z')
                z_dynamics_output = self.dynamics(
                    z, beta, net_weights,
                    while_loop=self.while_loop,
                    #  v_in=None,
                    save_lf=False
                )
                z_proposed = z_dynamics_output['x_proposed']
                #  z_proposed = tf.mod(z_dynamics_output['x_proposed'],
                #                      2 * np.pi)
                pz = z_dynamics_output['accept_prob']
        else:
            z = tf.zeros(x.shape, dtype=TF_FLOAT, name='z')
            z_proposed = tf.zeros(x.shape, dtype=TF_FLOAT)
            pz = tf.zeros(px.shape, dtype=TF_FLOAT)

        with tf.name_scope('top_charge_diff'):
            x_dq = tf.cast(
                self.lattice.calc_top_charges_diff(x, x_out, fft=False),
                dtype=tf.int32
            )

        # NOTE:
        #  std:_loss: 'standard' loss
        #  charge_loss: Contribution from the difference in topological charge
        #    betweween the initial and proposed configurations  to the total
        #     loss.
        inputs = {
            'x_init': x,
            'x_proposed': x_proposed,
            'px': px,
            'z_init': z,
            'z_proposed': z_proposed,
            'pz': pz
        }
        with tf.name_scope('calc_loss'):
            with tf.name_scope('std_loss'):
                std_loss = self.calc_std_loss(inputs, **weights)
            with tf.name_scope('charge_loss'):
                charge_loss = self.calc_charge_loss(inputs, **weights)

            total_loss = tf.add(std_loss, charge_loss, name='total_loss')
            tf.add_to_collection('losses', total_loss)

        return total_loss, x_dq, x_dynamics_output

    def calc_loss_and_grads(self, x, beta, net_weights, **weights):
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
        # TODO: Fix eager execution logic to deal with self.lf_out
        with tf.name_scope('grads'):
            if tf.executing_eagerly():
                with tf.GradientTape() as tape:
                    loss, x_dq, dynamics_output = self.calc_loss(
                        x, beta, net_weights, **weights
                    )
                grads = tape.gradient(loss, self.dynamics.trainable_variables)
            else:
                loss, x_dq, dynamics_output = self.calc_loss(x, beta,
                                                             net_weights,
                                                             **weights)
                with tf.name_scope('grads'):
                    grads = tf.gradients(loss,
                                         self.dynamics.trainable_variables)
                    if self.clip_grads:
                        grads, _ = tf.clip_by_global_norm(grads,
                                                          self.clip_value)

        return loss, grads, x_dq, dynamics_output

    def create_sampler(self):
        """Create operation for generating new samples using dynamics engine.

        NOTE: This method is to be used when running generic HMC to create
        operations for generating new sampler.
        """
        with tf.name_scope('sampler'):
            output = self.dynamics(self.x, self.beta,
                                   save_lf=True, train=False)
            self.x_out = output['x_out']
            self.px = output['accept_prob']
            if self.save_lf:
                self.lf_out_f = output['lf_out_f']
                self.pxs_out_f = output['accept_probs_f']
                self.lf_out_b = output['lf_out_b']
                self.pxs_out_b = output['accept_probs_b']
                self.masks_f = output['forward_mask']
                self.masks_b = output['backward_mask']
                self.logdets_f = output['logdets_f']
                self.logdets_b = output['logdets_b']
                self.sumlogdet_f = output['sumlogdet_f']
                self.sumlogdet_b = output['sumlogdet_b']

    def build(self):
        """Build Tensorflow graph."""
        with tf.name_scope('output'):
            #  self.loss_op, self.grads, self.x_out, self.px, x_dq = output
            loss, grads, x_dq, dynamics_output = self.calc_loss_and_grads(
                x=self.x, beta=self.beta,
                net_weights=self.net_weights,
                **self.loss_weights
            )
            #  self.loss_op = outputs[0]
            #  self.grads = outputs[1]
            self.loss_op = loss
            self.grads = grads
            self.x_out = dynamics_output['x_out']
            self.px = dynamics_output['accept_prob']
            self.charge_diffs_op = tf.reduce_sum(x_dq) / self.num_samples
            if self.save_lf:
                self.lf_out_f = dynamics_output['lf_out_f']
                self.lf_out_b = dynamics_output['lf_out_b']
                self.pxs_out_f = dynamics_output['accept_probs_f']
                self.pxs_out_b = dynamics_output['accept_probs_b']
                self.masks_f = dynamics_output['forward_mask']
                self.masks_b = dynamics_output['backward_mask']
                self.logdets_f = dynamics_output['logdets_f']
                self.logdets_b = dynamics_output['logdets_b']
                self.sumlogdet_f = dynamics_output['sumlogdet_f']
                self.sumlogdet_b = dynamics_output['sumlogdet_b']

        with tf.name_scope('train'):
            # --------------------------------------------------------
            #  TODO:
            #
            #  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #  ! no update ops in the default graph
            #  io.log("update_ops: ", update_ops)
            #  Use the update ops of the model itself
            #  io.log("model.updates: ", self.dynamics.updates)
            # --------------------------------------------------------
            grads_and_vars = zip(self.grads, self.dynamics.trainable_variables)
            with tf.control_dependencies(self.dynamics.updates):
                self.train_op = self.optimizer.apply_gradients(
                    grads_and_vars,
                    global_step=self.global_step,
                    name='train_op'
                )
