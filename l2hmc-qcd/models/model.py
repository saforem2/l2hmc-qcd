"""
gauge_model.py

Implements GaugeModel class responsible for building computation graph used in
tensorflow.

----------------------------------------------------------
TODO (taken from [1.]):

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
! no update ops in the default graph
io.log("update_ops: ", update_ops)
Use the update ops of the model itself
io.log("model.updates: ", self.dynamics.updates)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PREVIOUS (WORKING but diverging grads as of 07/15/2019):
```
with tf.control_dependencies(self.dynamics.updates):
    self.train_op = self.optimizer.apply_gradients(
        grads_and_vars,
        global_step=self.global_step,
        name='train_op'
    )
```
    self.train_op = tf.group(minimize_op,
                             self.dynamics.updates)
    try:
        self.train_op = self._append_update_ops(train_op)
    update_ops = [self.dynamics.updates,
                  tf.get_collection(tf.GraphKeys.UPDATE_OPS)]
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                         *[i for i in self.dynamics.updates])
----------------------------------------------------------

Author: Sam Foreman (github: @saforem2)
Date: 04/12/2019
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import control_flow_ops as control_flow_ops

import os

import numpy as np
import tensorflow as tf
try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False

import utils.file_io as io

from variables import GLOBAL_SEED, TF_FLOAT, PARAMS
from lattice.lattice import GaugeLattice
from dynamics.dynamics import GaugeDynamics
from dynamics.nnehmc_dynamics import nnehmcDynamics
from utils.horovod_utils import configure_learning_rate


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
        if params is None:
            params = PARAMS  # default parameters, defined in `variables.py`

        self.params = params
        for key, val in self.params.items():
            if 'weight' in key and key != 'charge_weight':
                self.loss_weights[key] = val
            else:
                setattr(self, key, val)

        self.charge_weight_np = params['charge_weight']

        self.lattice, self.samples = self._create_lattice()
        self.batch_size = self.lattice.samples.shape[0]
        self.x_dim = self.lattice.num_links

        inputs = self._create_inputs()
        self.x = inputs['x']
        self.beta = inputs['beta']
        self.charge_weight = inputs['charge_weight']
        self.net_weights = inputs['net_weights']
        self.train_phase = inputs['train_phase']

        self.dynamics, self.potential_fn = self._create_dynamics(
            self.lattice,
            self.samples
        )

        # metric function used when calculating the loss
        self.metric_fn = self._create_metric_fn(self.metric)

        obs_ops = self._create_observables()
        self.plaq_sums_op = obs_ops['plaq_sums']
        self.actions_op = obs_ops['actions']
        self.plaqs_op = obs_ops['plaqs']
        self.avg_plaqs_op = obs_ops['avg_plaqs']
        self.charges_op = obs_ops['charges']

        self._build_sampler()

        self._create_lr()
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
        obs_ops = {}
        with tf.name_scope('observables'):
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
                                   name='x')
                beta = tf.placeholder(dtype=TF_FLOAT,
                                      shape=(),
                                      name='beta')

                charge_weight = tf.placeholder(dtype=TF_FLOAT,
                                               shape=(),
                                               name='charge_weight')

                scale_weight = tf.placeholder(dtype=TF_FLOAT,
                                              shape=(),
                                              name='scale_weight')
                transf_weight = tf.placeholder(dtype=TF_FLOAT,
                                               shape=(),
                                               name='transformation_weight')
                transl_weight = tf.placeholder(dtype=TF_FLOAT,
                                               shape=(),
                                               name='translation_weight')

                net_weights = [scale_weight, transf_weight, transl_weight]

                train_phase = tf.placeholder(tf.bool, name='is_training')

            else:
                x = tf.convert_to_tensor(
                    self.lattice.samples.reshape((self.batch_size,
                                                  self.x_dim))
                )
                beta = tf.convert_to_tensor(self.beta_init)
                charge_weight = tf.convert_to_tensor(0.)
                net_weights = tf.convert_to_tensor([1., 1., 1.])
                train_phase = True

        names = ['x', 'beta', 'charge_weight', 'train_phase', 'net_weights']
        tensors = [x, beta, charge_weight, train_phase, net_weights]
        outputs = dict(zip(names, tensors))

        for name, tensor in outputs.items():
            if name != 'net_weights':
                tf.add_to_collection('inputs', tensor)
            else:
                [tf.add_to_collection('inputs', t) for t in tensor]

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
                'eps_trainable': not self.eps_fixed,
                #  'data_format': self.data_format,
                'use_bn': self.use_bn,
                'dropout_prob': self.dropout_prob,
                'num_hidden': self.num_hidden,
            }

            dynamics_kwargs.update(kwargs)
            potential_fn = lattice.get_potential_fn(samples)
            if self.nnehmc_loss:
                dynamics = nnehmcDynamics(lattice=lattice,
                                          potential_fn=potential_fn,
                                          **dynamics_kwargs)
            else:
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

            elif metric == 'sin_diff':
                def metric_fn(x1, x2):
                    return 1. - tf.sin(x1 - x2)

            elif metric == 'tan_cos':
                def metric_fn(x1, x2):
                    cos_diff = 1. - tf.cos(x1 - x2)
                    return tf.tan(np.pi * cos_diff / 2)
            else:
                raise AttributeError(f"""metric={metric}. Expected one of:
                                     'l1', 'l2', 'cos', 'cos2', 'cos_diff', or
                                     'tan_cos'.""")

        return metric_fn

    def _create_lr(self, lr_init=None):
        """Create learning rate."""
        if self.hmc:
            return

        if lr_init is None:
            lr_init = self.lr_init

        with tf.name_scope('global_step'):
            self.global_step = tf.train.get_or_create_global_step()

        with tf.name_scope('learning_rate'):
            # HOROVOD: When performing distributed training, it can be usedful
            # to "warmup" the learning rate gradually, done using the
            # `configure_learning_rate` method below..
            if self.warmup_lr:
                # lr_init has already been multiplied by num_workers, so to
                # get back to the original `lr_init` parsed from the
                # command line, divide once by `num_workers`.
                num_workers = hvd.size()
                if self.using_hvd:
                    lr_init /= num_workers

                lr_warmup = lr_init / 10
                warmup_steps = int(0.1 * self.train_steps)
                self.lr = configure_learning_rate(lr_warmup,
                                                  lr_init,
                                                  self.lr_decay_steps,
                                                  self.lr_decay_rate,
                                                  self.global_step,
                                                  warmup_steps)
                #  lr_warmup = lr_init / num_workers
                #  _train_steps = self.train_steps // num_workers
                #  warmup_steps = int(0.1 * _train_steps)
            else:
                self.lr = tf.train.exponential_decay(lr_init,
                                                     self.global_step,
                                                     self.lr_decay_steps,
                                                     self.lr_decay_rate,
                                                     staircase=False,
                                                     name='learning_rate')

    def _create_optimizer(self):
        """Create learning rate and optimizer."""
        if self.lr is None:
            self._create_lr()

        #  with tf.control_dependencies(update_ops):
        with tf.name_scope('optimizer'):
            #  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            if self.using_hvd:
                self.optimizer = hvd.DistributedOptimizer(self.optimizer)

    def _calc_std_loss(self, inputs, **weights):
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

        x_init = inputs['x_init']
        x_proposed = inputs['x_proposed']
        z_init = inputs['z_init']
        z_proposed = inputs['z_proposed']
        px = inputs['px']
        pz = inputs['pz']

        ls = self.loss_scale
        with tf.name_scope('std_loss'):
            with tf.name_scope('x_loss'):
                x_std_loss = tf.reduce_sum(
                    self.metric_fn(x_init, x_proposed), axis=1
                )
                x_std_loss *= px
                x_std_loss = tf.add(x_std_loss, eps, name='x_std_loss')
                tf.add_to_collection('losses', x_std_loss)

            with tf.name_scope('z_loss'):
                z_std_loss = tf.reduce_sum(
                    self.metric_fn(z_init, z_proposed), axis=1
                )
                z_std_loss *= pz * aux_weight
                z_std_loss = tf.add(z_std_loss, eps, name='z_std_loss')
                tf.add_to_collection('losses', z_std_loss)

            with tf.name_scope('tot_loss'):
                loss = (ls * (1. / x_std_loss + 1. / z_std_loss)
                        - (x_std_loss + z_std_loss) / ls) * std_weight

                std_loss = tf.reduce_mean(loss, axis=0, name='std_loss')
                tf.add_to_collection('losses', std_loss)

        return std_loss

    def _calc_charge_loss(self, inputs, **weights):
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
        aux_weight = weights.get('aux_weight', 1.)
        charge_weight = weights.get('charge_weight', 1.)

        if charge_weight == 0:
            return 0.

        x_init = inputs['x_init']
        x_proposed = inputs['x_proposed']
        z_init = inputs['z_init']
        z_proposed = inputs['z_proposed']
        px = inputs['px']
        pz = inputs['pz']

        # Calculate the difference in topological charge between the initial
        # and proposed configurations multiplied by the probability of
        # acceptance to get the expected value of the difference in topological
        # charge
        with tf.name_scope('charge_loss'):
            with tf.name_scope('x_loss'):
                x_dq_fft = self.lattice.calc_top_charges_diff(x_init,
                                                              x_proposed,
                                                              fft=True)
                xq_loss = px * x_dq_fft
                tf.add_to_collection('losses', xq_loss)

            with tf.name_scope('z_loss'):
                if aux_weight > 0:
                    z_dq_fft = self.lattice.calc_top_charges_diff(z_init,
                                                                  z_proposed,
                                                                  fft=True)
                else:
                    z_dq_fft = tf.zeros_like(x_dq_fft)

                zq_loss = aux_weight * pz * z_dq_fft

                tf.add_to_collection('losses', zq_loss)

            with tf.name_scope('tot_loss'):
                # Each of the loss terms is scaled by the `loss_scale` which
                # introduces a universal multiplicative factor that scales the
                # value of the loss
                charge_loss = - charge_weight * (xq_loss + zq_loss)
                charge_loss = tf.reduce_mean(charge_loss, axis=0,
                                             name='charge_loss')
                tf.add_to_collection('losses', charge_loss)

        return charge_loss

    def _calc_nnehmc_loss(self, x_dynamics_out, z_dynamics_out, **weights):
        """Implements the loss from the NNEHMC paper."""
        old_hamil_x = x_dynamics_out['old_hamil']
        new_hamil_x = x_dynamics_out['new_hamil']

        eta = 1.

        _px = tf.exp(tf.minimum(
            (old_hamil_x - new_hamil_x), 0.
        ), name='hmc_px')

        hmc_px = tf.where(tf.is_finite(_px), _px, tf.zeros_like(_px))

        if weights['aux_weight'] > 0.:
            old_hamil_z = z_dynamics_out['old_hamil']
            new_hamil_z = z_dynamics_out['new_hamil']

            _pz = tf.exp(tf.minimum(
                (old_hamil_z - new_hamil_z), 0.
            ), name='hmc_pz')

            hmc_pz = tf.where(tf.is_finite(_pz), _pz, tf.zeros_like(_pz))

        else:
            hmc_pz = tf.zeros_like(_px)

        loss = - eta * (hmc_px + weights['aux_weight'] * hmc_pz)

        nnehmc_loss = tf.reduce_mean(loss, axis=0, name='nnehmc_loss')
        tf.add_to_collection('losses', nnehmc_loss)

        return nnehmc_loss

    def _append_update_ops(self, train_op):
        """Returns `train_op` appending `UPDATE_OPS` collection if prsent."""
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            io.log(f'Update ops: {update_ops}')
            return control_flow_ops(train_op, *update_ops)
        return train_op

    def _build_run_ops(self):
        """Build run_ops dict containing grouped operations for inference."""
        run_ops = {
            'x_out': self.x_out,
            'px': self.px,
            'actions_op': self.actions_op,
            'plaqs_op': self.plaqs_op,
            'avg_plaqs_op': self.avg_plaqs_op,
            'charges_op': self.charges_op,
            'charge_diffs_op': self.charge_diffs_op,
        }

        if self.save_lf:
            run_ops.update({
                'lf_out_f': self.lf_out_f,
                'pxs_out_f': self.pxs_out_f,
                'masks_f': self.masks_f,
                'logdets_f': self.logdets_f,
                'sumlogdet_f': self.sumlogdet_f,
                'lf_out_b': self.lf_out_b,
                'pxs_out_b': self.pxs_out_b,
                'masks_b': self.masks_b,
                'logdets_b': self.logdets_b,
                'sumlogdet_b': self.sumlogdet_b
            })

        run_ops['dynamics_eps'] = self.dynamics.eps

        return run_ops

    def _build_train_ops(self):
        """Build train_ops dict containing grouped operations for training."""
        if self.hmc:
            train_ops = {}

        else:
            train_ops = {
                'train_op': self.train_op,
                'loss_op': self.loss_op,
                'x_out': self.x_out,
                'px': self.px,
                'dynamics.eps': self.dynamics.eps,
                'actions_op': self.actions_op,
                'plaqs_op': self.plaqs_op,
                'charges_op': self.charges_op,
                'charge_diffs_op': self.charge_diffs_op,
                'lr': self.lr
            }

        return train_ops

    def calc_loss(self, x, beta, net_weights, train_phase, **weights):
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
            x_dynamics_output = self.dynamics.apply_transition(x, beta,
                                                               net_weights,
                                                               train_phase,
                                                               self.save_lf)
            x_proposed = x_dynamics_output['x_proposed']
            #  x_proposed = tf.mod(x_dynamics_output['x_proposed'], 2 * np.pi)
            px = x_dynamics_output['accept_prob']
            x_out = x_dynamics_output['x_out']
            #  x_out = tf.mod(x_dynamics_output['x_out'], 2 * np.pi)

        # Auxiliary variable
        with tf.name_scope('z_update'):
            z = tf.random_normal(tf.shape(x), seed=GLOBAL_SEED, name='z')
            z_dynamics_output = self.dynamics.apply_transition(z, beta,
                                                               net_weights,
                                                               train_phase,
                                                               save_lf=False)
            z_proposed = z_dynamics_output['x_proposed']
            pz = z_dynamics_output['accept_prob']
            #  z_proposed, _, pz, _ = self.dynamics(z, beta)

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
        inputs = {
            'x_init': x,
            'x_proposed': x_proposed,
            'z_init': z,
            'z_proposed': z_proposed,
            'px': px,
            'pz': pz
        }

        with tf.name_scope('calc_loss'):
            with tf.name_scope('std_loss'):
                std_loss = self._calc_std_loss(inputs, **weights)

            with tf.name_scope('charge_loss'):
                charge_loss = self._calc_charge_loss(inputs, **weights)

            if self.nnehmc_loss:
                with tf.name_scope('nnehmc_loss'):
                    nnehmc_loss = self._calc_nnehmc_loss(x_dynamics_output,
                                                         z_dynamics_output,
                                                         **weights)

                    total_loss = std_loss + charge_loss + nnehmc_loss
            else:
                total_loss = tf.add(std_loss, charge_loss, name='total_loss')

            tf.add_to_collection('losses', total_loss)

        return total_loss, x_dq, x_dynamics_output

    def calc_loss_and_grads(self,
                            x,
                            beta,
                            net_weights,
                            train_phase,
                            **weights):
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
                    loss, x_dq, dynamics_output = self.calc_loss(x,
                                                                 beta,
                                                                 net_weights,
                                                                 train_phase,
                                                                 **weights)

                grads = tape.gradient(loss, self.dynamics.trainable_variables)
                if self.clip_value > 0.:
                    grads, _ = tf.clip_by_global_norm(grads, self.clip_value)
        else:
            loss, x_dq, dynamics_output = self.calc_loss(x, beta,
                                                         net_weights,
                                                         train_phase,
                                                         **weights)
            with tf.name_scope('grads'):
                grads = tf.gradients(loss, self.dynamics.trainable_variables)
                if self.clip_value > 0.:
                    grads, _ = tf.clip_by_global_norm(grads, self.clip_value)

        return loss, grads, x_dq, dynamics_output

    def _build_sampler(self):
        """Build TensorFlow graph."""
        with tf.name_scope('l2hmc_sampler'):
            #  self.loss_op, self.grads, self.x_out, self.px, x_dq = output
            loss, grads, x_dq, dynamics_output = self.calc_loss_and_grads(
                x=self.x,
                beta=self.beta,
                net_weights=self.net_weights,
                train_phase=self.train_phase,
                **self.loss_weights
            )
            self.loss_op = loss
            self.grads = grads
            self.x_out = dynamics_output['x_out']
            self.px = dynamics_output['accept_prob']
            self.charge_diffs_op = tf.reduce_sum(x_dq) / self.num_samples

            if self.save_lf:
                op_keys = ['masks_f', 'masks_b',
                           'lf_out_f', 'lf_out_b',
                           'pxs_out_f', 'pxs_out_b',
                           'logdets_f', 'logdets_b',
                           'sumlogdet_f', 'sumlogdet_b']
                for key in op_keys:
                    try:
                        op = dynamics_output[key]
                        setattr(self, key, op)
                    except KeyError:
                        continue

    def build(self):
        """Build Tensorflow graph."""
        run_ops = self._build_run_ops()

        # ------------------------------
        # Ref. [1.] in TODO (line 8)
        # ------------------------------
        with tf.name_scope('train'):
            grads_and_vars = zip(self.grads, self.dynamics.trainable_variables)
            with tf.control_dependencies(self.dynamics.updates):
                self.train_op = self.optimizer.apply_gradients(
                    grads_and_vars,
                    global_step=self.global_step,
                    name='train_op'
                )
            train_ops = self._build_train_ops()

        self.ops_dict = {
            'run_ops': run_ops,
            'train_ops': train_ops
        }

        for key, val in self.ops_dict.items():
            [tf.add_to_collection(key, op) for op in list(val.values())]
