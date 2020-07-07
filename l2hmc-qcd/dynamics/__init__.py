from collections import namedtuple
from utils.attr_dict import AttrDict

import os
import sys

from config import (
    State, MonteCarloStates,
    NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC
)

modulepath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(modulepath)


class DynamicsParams:

    def __init__(self, params, dynamics_config, network_config):
        self.config = dynamics_config
        self.network_config = network_config
        self.params = self.parse_params(params)

    # pylint:disable=too-many-statements
    def parse_params(self, params):
        """Set instance attributes from `params`."""
        #  self.params = AttrDict(params)
        params = AttrDict(params)
        attrs = AttrDict({})

        hmc = self.config.hmc
        attrs.net_weights = NET_WEIGHTS_HMC if hmc else NET_WEIGHTS_L2HMC
        attrs.xsw = attrs.net_weights.x_scale
        attrs.xtw = attrs.net_weights.x_translation
        attrs.xqw = attrs.net_weights.x_transformation
        attrs.vsw = attrs.net_weights.v_scale
        attrs.vtw = attrs.net_weights.v_translation
        attrs.vqw = attrs.net_weights.v_transformation

        attrs.separate_networks = params.get('separate_networks', False)

        lattice_shape = params.get('lattice_shape', None)
        if lattice_shape is not None:
            batch_size, time_size, space_size, dim = lattice_shape
        else:
            batch_size = params.get('batch_size', None)
            time_size = params.get('time_size', None)
            space_size = params.get('space_size', None)
            dim = params.get('dim', 2)
            lattice_shape = (batch_size, time_size, space_size, dim)

        attrs.batch_size = batch_size
        attrs.lattice_shape = lattice_shape
        attrs.xdim = time_size * space_size * dim
        attrs.x_shape = (batch_size, attrs.xdim)

        attrs.plaq_weight = params.get('plaq_weight', 0.)
        attrs.charge_weight = params.get('charge_weight', 0.)

        attrs.print_steps = params.get('print_steps', 10)
        attrs.run_steps = params.get('run_steps', int(1e3))
        attrs.logging_steps = params.get('logging_steps', 50)
        attrs.save_run_data = params.get('save_run_data', True)
        attrs.save_train_data = True
        attrs.save_steps = params.get('save_steps', None)

        attrs.should_compile = True
        eager_execution = params.get('eager_execution', False)
        if tf.executing_eagerly() or eager_execution:
            attrs.should_compile = False

            # Determine if there are any parameters to be trained
        attrs.has_trainable_params = True
        if attrs.config.hmc and not attrs.config.eps_trainable:
            attrs.has_trainable_params = False

        # If there exist parameters to be optimized, setup optimizer
        if attrs.has_trainable_params:
            attrs.lr_init = params.get('lr_init', None)
            attrs.warmup_lr = params.get('warmup_lr', False)
            attrs.warmup_steps = params.get('warmup_steps', None)
            attrs.using_hvd = params.get('horovod', False)
            attrs.lr_decay_steps = params.get('lr_decay_steps', None)
            attrs.lr_decay_rate = params.get('lr_decay_rate', None)

            attrs.train_steps = params.get('train_steps', None)
            attrs.beta_init = params.get('beta_init', None)
            attrs.beta_final = params.get('beta_final', None)
            beta = params.get('beta', None)
            if attrs.beta_init == attrs.beta_final or beta is None:
                attrs.beta = attrs.beta_init
                attrs.betas = tf.convert_to_tensor(
                    tf.cast(attrs.beta * np.ones(attrs.train_steps),
                            dtype=TF_FLOAT)
                )
            else:
                if attrs.train_steps is not None:
                    attrs.betas = get_betas(
                        attrs.train_steps, attrs.beta_init, attrs.beta_final
                    )

        return attrs


