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
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
import tensorflow as tf

from lattice.lattice import GaugeLattice
from utils.horovod_utils import warmup_lr
from dynamics.dynamics import GaugeDynamics
from config import GLOBAL_SEED, TF_FLOAT, TF_INT, PARAMS, HAS_HOROVOD
#  from tensorflow.python.ops import control_flow_ops as control_flow_ops

if HAS_HOROVOD:
    import horovod.tensorflow as hvd


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
            elif key == 'charge_weight':
                pass
            else:
                setattr(self, key, val)

        self.eps_trainable = not self.eps_fixed
        self.charge_weight_np = params['charge_weight']

        with tf.name_scope('init'):
            self.lattice = self._create_lattice()
            self.batch_size = self.lattice.samples.shape[0]
            self.x_dim = self.lattice.num_links

            # build input placeholders for tensors
            inputs = self._create_inputs()
            _ = [setattr(self, key, val) for key, val in inputs.items()]

            self.dynamics = self._create_dynamics()

            # metric function used when calculating the loss
            self.metric_fn = self._create_metric_fn(self.metric)

        # build operations for calculating lattice observables
        obs_ops = self._create_observables()
        _ = [setattr(self, key, val) for key, val in obs_ops.items()]

        # Calculate loss and collect outputs from L2HMC step
        loss, dynamics_out, grads = self.calc_loss(self.x, self.beta,
                                                   self.net_weights,
                                                   self.train_phase,
                                                   **self.loss_weights)
        self.loss_op = loss
        self.grads = grads
        self._parse_dynamics_output(dynamics_out)

        with tf.name_scope('run_ops'):
            run_ops = self._build_run_ops()

        if not self.hmc:
            with tf.name_scope('train'):
                self.lr = self._create_lr()
                self.optimizer = self._create_optimizer()
                self.train_op = self._apply_grads()
                train_ops = self._build_train_ops()
        else:
            train_ops = {}

        self.ops_dict = {
            'run_ops': run_ops,
            'train_ops': train_ops
        }

        for key, val in self.ops_dict.items():
            _ = [tf.add_to_collection(key, op) for op in list(val.values())]

    def _create_lattice(self):
        """Create GaugeLattice object."""
        with tf.name_scope('lattice'):
            lattice = GaugeLattice(time_size=self.time_size,
                                   space_size=self.space_size,
                                   dim=self.dim,
                                   link_type=self.link_type,
                                   num_samples=self.num_samples,
                                   rand=self.rand)

        return lattice

    def _create_dynamics(self, **kwargs):
        """Initialize dynamics object."""
        with tf.name_scope('create_dynamics'):
            # default values of keyword arguments
            keys = ['eps', 'hmc', 'network_arch',
                    'num_steps', 'use_bn', 'dropout_prob',
                    'num_hidden1', 'num_hidden2', 'zero_translation']
            dynamics_kwargs = {k: getattr(self, k) for k in keys}

            dynamics_kwargs['eps_trainable'] = not self.eps_fixed
            dynamics_kwargs['num_filters'] = self.lattice.space_size
            dynamics_kwargs['x_dim'] = self.lattice.num_links
            #  dynamics_kwargs['links_shape'] = self.lattice.links.shape
            dynamics_kwargs['batch_size'] = self.num_samples
            dynamics_kwargs['_input_shape'] = (self.num_samples,
                                               *self.lattice.links.shape)

            dynamics_kwargs.update(kwargs)
            samples = self.lattice.samples_tensor
            potential_fn = self.lattice.get_potential_fn(samples)
            dynamics = GaugeDynamics(potential_fn=potential_fn,
                                     **dynamics_kwargs)

        return dynamics

    def _create_observables(self):
        """Create operations for calculating lattice observables."""
        with tf.name_scope('observables'):
            plaq_sums = self.lattice.calc_plaq_sums(self.x)
            actions = self.lattice.calc_actions(plaq_sums=plaq_sums)
            plaqs = self.lattice.calc_plaqs(plaq_sums=plaq_sums)
            avg_plaqs = tf.reduce_mean(plaqs, name='avg_plaqs')
            charges = self.lattice.calc_top_charges(plaq_sums=plaq_sums)

        obs_ops = {
            'plaq_sums_op': plaq_sums,
            'actions_op': actions,
            'plaqs_op': plaqs,
            'avg_plaqs_op': avg_plaqs,
            'charges_op': charges,
        }

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
                train_phase: Boolean placeholder indicating if the model is
                    curerntly being trained. 
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
                _ = [tf.add_to_collection('inputs', t) for t in tensor]

        return outputs

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
                kwargs = {
                    'target_lr': lr_init,
                    'warmup_steps': int(0.1 * self.train_steps),
                    'global_step': self.global_step,
                    'decay_steps': self.lr_decay_steps,
                    'decay_rate': self.lr_decay_rate,
                }
                lr = warmup_lr(**kwargs)
            else:
                lr = tf.train.exponential_decay(lr_init, self.global_step,
                                                self.lr_decay_steps,
                                                self.lr_decay_rate,
                                                staircase=False,
                                                name='learning_rate')
        return lr

    def _create_optimizer(self):
        """Create learning rate and optimizer."""
        if not hasattr(self, 'lr'):
            self._create_lr()

        #  with tf.control_dependencies(update_ops):
        with tf.name_scope('optimizer'):
            #  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer = tf.train.AdamOptimizer(self.lr)
            if self.using_hvd:
                optimizer = hvd.DistributedOptimizer(optimizer)

        return optimizer

    def _build_run_ops(self):
        """Build run_ops dict containing grouped operations for inference."""
        keys = ['x_out', 'px', 'actions_op',
                'plaqs_op', 'avg_plaqs_op',
                'charges_op', 'charge_diffs_op']
        run_ops = {k: getattr(self, k) for k in keys}

        if self.save_lf:
            keys = ['lf_out', 'pxs_out', 'masks',
                    'logdets', 'sumlogdet', 'fns_out']

            fkeys = [k + '_f' for k in keys]
            bkeys = [k + '_b' for k in keys]

            run_ops.update({k: getattr(self, k) for k in fkeys})
            run_ops.update({k: getattr(self, k) for k in bkeys})

        run_ops['dynamics_eps'] = self.dynamics.eps

        return run_ops

    def _build_train_ops(self):
        """Build train_ops dict containing grouped operations for training."""
        if self.hmc:
            train_ops = {}

        else:
            keys = ['train_op', 'loss_op', 'x_out',
                    'px', 'actions_op', 'plaqs_op',
                    'charges_op', 'charge_diffs_op', 'lr']
            train_ops = {k: getattr(self, k) for k in keys}

            train_ops['dynamics.eps'] = self.dynamics.eps

        return train_ops

    def run_dynamics(self, x, z, beta, net_weights, train_phase):
        """Run dynamics by applying transition.

        Args:
            x (tf.placeholder): Input configuration.
            z (tf.Tensor): Input configuration drawn randomly from
                initialization distribution, contributes to loss function
                (encourages typical moves to be large; 2nd term in Eq. 8 of
                original paper)
            beta (tf.placeholder): Inverse coupling constant (inverse
                temperature) used for simulated annealing.
            net_weights (array-like): Array of `net_weights`, which are
                multiplicative factors used to scale the contributions from
                each of the S, T, Q functions:
                    `net_weights = [s_weight, t_weight, q_weight]`
            train_phase (tf.placeholder): Boolean indicating whether running
                `training` or `inference`.

        Returns:
            x_dynamics: Output from `self.dynamics.apply_transition`
                method, when called on samples drawn from target distribution.
            z_dynamics: Output from `self.dynamics.apply_transition`
                method, when called on samples drawn from initialization
                distribution.

        NOTE: We are drawing from both the target and initialization
            distributions.
        """
        with tf.name_scope('main_update'):
            x_dynamics = self.dynamics.apply_transition(x, beta,
                                                        net_weights,
                                                        train_phase,
                                                        self.save_lf)

        with tf.name_scope('aux_update'):
            z_dynamics = self.dynamics.apply_transition(z, beta,
                                                        net_weights,
                                                        train_phase,
                                                        save_lf=False)

        return x_dynamics, z_dynamics

    def _calc_std_loss(self, inputs, **weights):
        """Calculate standard contribution to loss.

        NOTE: In contrast to the original paper where the L2 difference was
        used, we are now using 1 - cos(x1 - x2).

        Args:
            inputs (dict): Dictionary of input data containing initial
                configurations, proposed configurations, and acceptance
                probabilities (drawn from both the target (`x`) and
                initialization distributions (`z`).
            weights (dict): Dictionary of weights giving relative weight of
                each term in the loss function.

        Returns:
            std_loss
        """
        eps = 1e-4
        aux_weight = weights.get('aux_weight', 1.)
        std_weight = weights.get('std_weight', 1.)

        if std_weight == 0.:  # don't bother calculating loss if weight is 0
            return 0.

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
                #  x_std_loss_avg = tf.reduce_sum(x_std_loss, axis=0,
                #                                 name='x_std_loss_avg')

                # Add eps for numerical stability
                x_std_loss = tf.add(x_std_loss, eps, name='x_std_loss')
                #  tf.add_to_collection('losses', x_std_loss)

            with tf.name_scope('z_loss'):
                z_std_loss = tf.reduce_sum(
                    self.metric_fn(z_init, z_proposed), axis=1
                )
                z_std_loss *= pz * aux_weight
                #  z_std_loss_avg = tf.reduce_mean(z_std_loss, axis=0,
                #                                  name='z_std_loss_avg')
                z_std_loss = tf.add(z_std_loss, eps, name='z_std_loss')
                #  tf.add_to_collection('losses', z_std_loss)

            with tf.name_scope('tot_loss'):
                loss = (ls * (1. / x_std_loss + 1. / z_std_loss)
                        - (x_std_loss + z_std_loss) / ls) * std_weight

                std_loss = tf.reduce_mean(loss, axis=0, name='std_loss')
                #  tf.add_to_collection('losses', std_loss)

        return std_loss  # , x_std_loss_avg, z_std_loss_avg

    def _calc_charge_loss(self, inputs, **weights):
        """Calculate contribution to total loss from charge difference.

        NOTE: This is an additional term introduced to the loss function that
        measures the difference in the topological charge between the initial
        configuration and the proposed configuration.

        Args:
            inputs (dict): Dictionary of input data containing initial
                configurations, proposed configurations, and acceptance
                probabilities (drawn from both the target (`x`) and
                initialization distributions (`z`).
            weights (dict): Dictionary of weights giving relative weight of
                each term in the loss function.

        Returns:
            `charge_loss` if `self.charge_weight_np == 0` else, `0.`
        """
        aw = weights.get('aux_weight', 1.)
        qw = weights.get('charge_weight', 1.)

        if qw == 0. or self.charge_weight_np == 0.:
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
                x_dq = self.lattice.calc_top_charges_diff(x_init, x_proposed)
                xq_loss = px * x_dq
                #  xq_loss_avg = tf.reduce_mean(xq_loss, axis=0,
                #                               name='xq_loss_avg')

            with tf.name_scope('z_loss'):
                if aw > 0:
                    z_dq = self.lattice.calc_top_charges_diff(z_init,
                                                              z_proposed)
                else:
                    z_dq = tf.zeros_like(x_dq)

                zq_loss = aw * pz * z_dq
                #  zq_loss_avg = tf.reduce_mean(zq_loss, axis=0,
                #                               name='zq_loss_avg')

            with tf.name_scope('tot_loss'):
                # Each of the loss terms is scaled by the `loss_scale` which
                # introduces a universal multiplicative factor that scales the
                # value of the loss
                charge_loss = - qw * (xq_loss + zq_loss) / self.loss_scale
                charge_loss = tf.reduce_mean(charge_loss, axis=0,
                                             name='charge_loss')
                #  tf.add_to_collection('losses', charge_loss)

        return charge_loss  # , xq_loss_avg, zq_loss_avg

    def _calc_loss(self, x, beta, net_weights, train_phase, **weights):
        """Create operation for calculating the loss, when running graph mode.

        Args:
            x (tf.placeholder): Input configuration.
            z (tf.Tensor): Input configuration drawn randomly from
                initialization distribution, contributes to loss function
                (encourages typical moves to be large; 2nd term in Eq. 8 of
                original paper)
            beta (tf.placeholder): Inverse coupling constant (inverse
                temperature) used for simulated annealing.
            net_weights (array-like): Array of `net_weights`, which are
                multiplicative factors used to scale the contributions from
                each of the S, T, Q functions:
                    `net_weights = [s_weight, t_weight, q_weight]`
            train_phase (tf.placeholder): Boolean indicating whether running
                `training` or `inference`.
            weights (dict): Dictionary containing multiplicative weights used
                to scale the contribution from different terms in the loss
                function.

        Returns:
            loss (float): Operation responsible for calculating the total loss.
            x_dynamics: Output from `self.dynamics.apply_transition`
                method, when called on samples drawn from target distribution.

        NOTE: If proposed configuration is accepted following
            Metropolis-Hastings accept/reject step, x_proposed and x_out are
            equivalent.
        """
        with tf.name_scope('dynamics'):
            with tf.name_scope('sample_init_distribution'):
                z = tf.random_normal(tf.shape(x), dtype=TF_FLOAT,
                                     seed=GLOBAL_SEED, name='z')

            x_dynamics, z_dynamics = self.run_dynamics(x, z, beta,
                                                       net_weights,
                                                       train_phase)

        inputs = {
            'x_init': x,
            'z_init': z,
            'px': x_dynamics['accept_prob'],
            'pz': z_dynamics['accept_prob'],
            'x_proposed': x_dynamics['x_proposed'],
            'z_proposed': z_dynamics['x_proposed'],
        }

        with tf.name_scope('loss'):
            std_loss = self._calc_std_loss(inputs, **weights)
            if self.charge_weight_np > 0:
                charge_loss = self._calc_charge_loss(inputs, **weights)
                loss = tf.add(std_loss, charge_loss, name='total_loss')
            else:
                loss = tf.identity(std_loss, name='total_loss')

        return loss, x_dynamics

    def _calc_loss_eager(self, x, beta, net_weights, train_phase, **weights):
        """Calculate the loss, using eager execution.

        Args:
            x (array-like): Input configuration.
            z (array-like): Input configuration drawn randomly from
                initialization distribution, contributes to loss function
                (encourages typical moves to be large; 2nd term in Eq. 8 of
                original paper)
            beta (float): Inverse coupling constant (inverse temperature) used
                for simulated annealing.
            net_weights (array-like): Array of `net_weights`, which are
                multiplicative factors used to scale the contributions from
                each of the S, T, Q functions:
                    `net_weights = [s_weight, t_weight, q_weight]`
            train_phase (bool): Boolean indicating whether running `training`
                or `inference`.
            weights (dict): Dictionary containing multiplicative weights used
                to scale the contribution from different terms in the loss
                function.

        Returns:
            loss (float): Operation responsible for calculating the total loss.
            x_dynamics: Output from `self.dynamics.apply_transition`
                method, when called on samples drawn from target distribution.

        NOTE: If proposed configuration is accepted following
            Metropolis-Hastings accept/reject step, x_proposed and x_out are
            equivalent.
        """
        with tf.GradientTape() as tape:
            output = self.calc_loss(x, beta,
                                    net_weights,
                                    train_phase,
                                    **weights)
            loss = output[0]
            dynamics_output = output[1]

        grads = tape.gradient(loss, self.dynamics.trainable_variables)
        if self.clip_value > 0.:
            grads, _ = tf.clip_by_global_norm(grads, self.clip_value)

        return loss, dynamics_output, grads

    def calc_loss(self, x, beta, net_weights, train_phase, **weights):
        """Calculate loss with gradients and get output from MD update.

        Args:
            x (array-like): Input configuration.
            z (array-like): Input configuration drawn randomly from
                initialization distribution, contributes to loss function
                (encourages typical moves to be large; 2nd term in Eq. 8 of
                original paper)
            beta (float): Inverse coupling constant (inverse temperature) used
                for simulated annealing.
            net_weights (array-like): Array of `net_weights`, which are
                multiplicative factors used to scale the contributions from
                each of the S, T, Q functions:
                    `net_weights = [s_weight, t_weight, q_weight]`
            train_phase (bool): Boolean indicating whether running `training`
                or `inference`.
            weights (dict): Dictionary containing multiplicative weights used
                to scale the contribution from different terms in the loss
                function.

        Returns:
            loss (float): Operation responsible for calculating the total loss.
            dynamics_out (dict): Output from `self.dynamics.apply_transition`
                method, when called on samples drawn from target distribution.
            grads (tf.gradients): Gradients used for backprop accumulated
                from the calculation of the loss function.
        """
        if tf.executing_eagerly():
            loss, dynamics_out, grads = self._calc_loss_eager(x, beta,
                                                              net_weights,
                                                              train_phase,
                                                              **weights)
        else:
            loss, dynamics_out = self._calc_loss(x, beta, net_weights,
                                                 train_phase, **weights)
            # calculate gradients for back prop.
            grads = self.calc_grads(loss)

        return loss, dynamics_out, grads

    def calc_grads(self, loss):
        """Calculate gradients using automatic differentiation on the loss.

        NOTE: IF `--clip_value x` command line argument was passed to `main.py`
            with a float `x > 0.`, gradient clipping (by global norm, `x`) will
            be performed.

        Args:
            loss: Tensorflow operation used to calculate the loss function to
                be minimized.

        Returns:
            grads (tf.gradients): Gradients used for backprop accumulated from
                the calculation of the loss function.
        """
        with tf.name_scope('grads'):
            grads = tf.gradients(loss, self.dynamics.trainable_variables)
            if self.clip_value > 0.:
                grads, _ = tf.clip_by_global_norm(grads, self.clip_value)

        return grads

    def _apply_grads(self):
        """Apply backpropagated gradients using `self.optimizer`.

        TODO: [1.] (line 8)

        Returns:
            train_op: Operation used for running a single training step.
        """
        grads_and_vars = zip(self.grads, self.dynamics.trainable_variables)
        with tf.control_dependencies([self.loss_op, *self.dynamics.updates]):
            train_op = self.optimizer.apply_gradients(grads_and_vars,
                                                      self.global_step,
                                                      'train_op')

        return train_op

    def _parse_dynamics_output(self, dynamics_output):
        """Parse output dictionary from `self.dynamics.apply_transition`.

        This method is responsible for creating instance attributes from the
        dictionary returned from the `self.dynamics.apply_transition` method.
        """
        self.x_out = dynamics_output['x_out']
        self.px = dynamics_output['accept_prob']

        with tf.name_scope('top_charge_diff'):
            x_in = dynamics_output['x_in']
            x_out = dynamics_output['x_out']
            x_dq = tf.cast(
                self.lattice.calc_top_charges_diff(x_in, x_out),
                dtype=TF_INT
            )
            self.charge_diffs_op = tf.reduce_sum(x_dq) / self.num_samples

        if self.save_lf:
            op_keys = ['masks_f', 'masks_b',          # directional masks
                       'lf_out_f', 'lf_out_b',        # MD output
                       'pxs_out_f', 'pxs_out_b',      # accept probs
                       'logdets_f', 'logdets_b',      # individ. logdets
                       'fns_out_f', 'fns_out_b',      # S, T, Q fns
                       'sumlogdet_f', 'sumlogdet_b']  # sumlogdet
            for key in op_keys:
                try:
                    op = dynamics_output[key]
                    setattr(self, key, op)
                except KeyError:
                    continue

            # Collect outputs from S, T, Q fns for logging in tensorboard
            with tf.name_scope('l2hmc_fns'):
                self.l2hmc_fns = {
                    'out_fns_f': self._extract_l2hmc_fns(self.fns_out_f),
                    'out_fns_b': self._extract_l2hmc_fns(self.fns_out_b),
                }

    def _extract_l2hmc_fns(self, fns):
        """Method for extracting each of the Q, S, T functions as tensors."""
        if not self.save_lf:
            return

        # fns has shape: (num_steps, 4, 3, num_samples, lattice.num_links)
        fnsT = tf.transpose(fns, perm=[2, 1, 0, 3, 4], name='fns_transposed')

        out_fns = {}
        names = ['scale', 'translation', 'transformation']
        subnames = ['v1', 'x1', 'x2', 'v2']
        for idx, name in enumerate(names):
            out_fns[name] = {}
            for subidx, subname in enumerate(subnames):
                out_fns[name][subname] = fnsT[idx][subidx]

        return out_fns
