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
import functools

import numpy as np
import tensorflow as tf
import utils.file_io as io

from lattice.lattice import GaugeLattice
from utils.horovod_utils import warmup_lr
from dynamics.dynamics import GaugeDynamics
from config import GLOBAL_SEED, TF_FLOAT, TF_INT, PARAMS, HAS_HOROVOD
from tensorflow.python.ops import control_flow_ops as control_flow_ops

if HAS_HOROVOD:
    import horovod.tensorflow as hvd


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def doublewrap(function):
    """A decorator decorator.

    This allows for the the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """A decorator for functions that define TensorFlow operations.

    The wrapped function will only be executed once. Subsequent calls to it
    will directly return the result so that the operations are added to the
    graph only once.

    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute) or self.attribute is None:
            with tf.name_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
            return getattr(self, attribute)
    return decorator


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

        # Create self.lattice using @lattice.setter property method
        lattice_keys = ('time_size', 'space_size', 'dim',
                        'link_type', 'num_samples', 'rand')
        self.lattice = {key: getattr(self, key) for key in lattice_keys}
        self.batch_size = self.lattice.samples.shape[0]
        self.x_dim = self.lattice.num_links

        # build input placeholders for tensors
        inputs = self._create_inputs()
        _ = [setattr(self, key, val) for key, val in inputs.items()]

        # Create self.dynamics using @dynamics.setter property method
        dynamics_keys = ('eps', 'hmc', 'network_arch', 'num_steps',
                         'eps_trainable', 'use_bn', 'dropout_prob',
                         'num_hidden1', 'num_hidden2', 'zero_translation')
        self.dynamics = {key: getattr(self, key) for key in dynamics_keys}

        # metric function used when calculating the loss
        self.metric_fn = self._create_metric_fn(self.metric)

        # Create ops to calc lattice observables using @property methods
        self._plaq_sums_op = None
        self._actions_op = None
        self._plaqs_op = None
        self._avg_plaqs_op = None
        self._charges_op = None

        self._build_sampler()
        run_ops = self._build_run_ops()
        #  self._collect_inputs()

        if self.hmc:
            train_ops = {}
        else:
            self._create_lr()
            self._create_optimizer()
            self._apply_grads()
            train_ops = self._build_train_ops()

        self.ops_dict = {
            'run_ops': run_ops,
            'train_ops': train_ops
        }

        for key, val in self.ops_dict.items():
            _ = [tf.add_to_collection(key, op) for op in list(val.values())]

    @property
    def lattice(self):
        return self._lattice

    @lattice.setter
    def lattice(self, kwargs):
        """Create GaugeLattice object."""
        with tf.name_scope('lattice'):
            self._lattice = GaugeLattice(**kwargs)
            assert self._lattice.samples.shape[0] == self.num_samples

    @property
    def plaq_sums_op(self):
        if self._plaq_sums_op is None:
            with tf.name_scope('observables'):
                self._plaq_sums_op = self.lattice.calc_plaq_sums(self.x)

        return self._plaq_sums_op

    @property
    def actions_op(self):
        """Operation for calculating the total action."""
        if self._actions_op is None:
            with tf.name_scope('observables'):
                self._actions_op = self.lattice.calc_actions(self.x)

        return self._actions_op

    @property
    def plaqs_op(self):
        """Operation for calculating the plaquette sum for each plaquette."""
        if self._plaqs_op is None:
            with tf.name_scope('observables'):
                self._plaqs_op = self.lattice.calc_plaqs(self.x)

        return self._plaqs_op

    @property
    def avg_plaqs_op(self):
        """Operation for calculating the average plaquette."""
        if self._avg_plaqs_op is None:
            with tf.name_scope('observables'):
                self._avg_plaqs_op = tf.reduce_mean(self._plaqs_op)

        return self._avg_plaqs_op

    @property
    def charges_op(self):
        """Operation for calculating the topological charge."""
        if self._charges_op is None:
            with tf.name_scope('observables'):
                self._charges_op = self.lattice.calc_top_charges(self.x,
                                                                 fft=False)

        return self._charges_op

    @property
    def dynamics(self):
        return self._dynamics

    @dynamics.setter
    def dynamics(self, kwargs):
        """Create GaugeDynamics o object.

        Args:
            lattice: Lattice object.
            samples: Initial value of samples (configurations) to use.
        """
        if hasattr(self, '_dynamics'):
            raise AttributeError(
                'GaugeDynamics object has already been created.'
            )

        with tf.name_scope('dynamics'):
            samples = self.lattice.samples_tensor
            potential_fn = self.lattice.get_potential_fn(samples)
            self._dynamics = GaugeDynamics(lattice=self.lattice,
                                           potential_fn=potential_fn,
                                           **kwargs)

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
                self.lr = warmup_lr(**kwargs)
            else:
                self.lr = tf.train.exponential_decay(lr_init,
                                                     self.global_step,
                                                     self.lr_decay_steps,
                                                     self.lr_decay_rate,
                                                     staircase=False,
                                                     name='learning_rate')

    def _create_optimizer(self):
        """Create learning rate and optimizer."""
        if not hasattr(self, 'lr'):
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

                # Add eps for numerical stability
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
            px = x_dynamics_output['accept_prob']
            x_out = x_dynamics_output['x_out']

        # Auxiliary variable
        with tf.name_scope('z_update'):
            z = tf.random_normal(tf.shape(x),
                                 dtype=TF_FLOAT,
                                 seed=GLOBAL_SEED,
                                 name='z')
            z_dynamics_output = self.dynamics.apply_transition(z, beta,
                                                               net_weights,
                                                               train_phase,
                                                               save_lf=False)
            z_proposed = z_dynamics_output['x_proposed']
            pz = z_dynamics_output['accept_prob']

        with tf.name_scope('top_charge_diff'):
            x_dq = tf.cast(
                self.lattice.calc_top_charges_diff(x, x_out, fft=False),
                dtype=TF_INT
            )

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
                std_loss = tf.cast(self._calc_std_loss(inputs, **weights),
                                   dtype=TF_FLOAT)

            with tf.name_scope('charge_loss'):
                charge_loss = self._calc_charge_loss(inputs, **weights)

            if self.nnehmc_loss:
                with tf.name_scope('nnehmc_loss'):
                    nnehmc_loss = self._calc_nnehmc_loss(x_dynamics_output,
                                                         z_dynamics_output,
                                                         **weights)

                    total_loss = std_loss + charge_loss + nnehmc_loss
            else:
                total_loss = tf.add(std_loss,
                                    charge_loss,
                                    name='total_loss')

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
                           'fns_out_f', 'fns_out_b',
                           'sumlogdet_f', 'sumlogdet_b']
                for key in op_keys:
                    try:
                        op = dynamics_output[key]
                        setattr(self, key, op)
                    except KeyError:
                        continue

                with tf.name_scope('l2hmc_fns'):
                    self.l2hmc_fns = {
                        'out_fns_f': self.extract_l2hmc_fns(self.fns_out_f),
                        'out_fns_b': self.extract_l2hmc_fns(self.fns_out_b),
                    }

    def extract_l2hmc_fns(self, fns):
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

    def _apply_grads(self):
        """Build Tensorflow graph."""
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
