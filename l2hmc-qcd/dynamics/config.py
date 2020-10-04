"""
config.py

Contains configuration objects for various `Dynamics`.
"""
import tensorflow as tf
from utils.attr_dict import AttrDict


ACTIVATIONS = {
    'relu': tf.nn.relu,
    'tanh': tf.nn.tanh,
    'leaky_relu': tf.nn.leaky_relu
}


class DynamicsConfig(AttrDict):
    """Configuration object for `BaseDynamics` object"""

    # pylint:disable=too-many-arguments
    def __init__(self,
                 eps: float,
                 num_steps: int,
                 hmc: bool = False,
                 model_type: str = None,
                 eps_fixed: bool = False,
                 aux_weight: float = 0.,
                 loss_scale: float = 1.,
                 use_mixed_loss: bool = False,
                 verbose: bool = False):
        super(DynamicsConfig, self).__init__(
            eps=eps,
            hmc=hmc,
            num_steps=num_steps,
            model_type=model_type,
            eps_fixed=eps_fixed,
            aux_weight=aux_weight,
            loss_scale=loss_scale,
            use_mixed_loss=use_mixed_loss,
            verbose=verbose,
        )


# pylint:disable=too-many-locals
class GaugeDynamicsConfig(AttrDict):
    """Configuration object for `GaugeDynamics` object"""

    # pylint:disable=too-many-arguments
    def __init__(self,
                 eps: float,                    # step size
                 num_steps: int,                # n leapfrog steps per acc/rej
                 hmc: bool = False,             # run standard HMC?
                 model_type: str = None,        # name for model
                 eps_fixed: bool = False,       # Fixed step size?
                 aux_weight: float = 0.,        # Weight of aux term in loss fn
                 loss_scale: float = 1.,        # Scale loss?
                 use_mixed_loss: bool = False,  # Use mixed loss?
                 verbose: bool = False,         # Verbose metric logging?
                 use_ncp: bool = False,         # Transform x using NCP?
                 lattice_shape: tuple = None,   # (batch_size, Lt, Lx, 2)
                 plaq_weight: float = 0.,       # Weight of plaq term in loss
                 charge_weight: float = 0.,     # Weight of charge term in loss
                 zero_init: bool = False,       # Initialize weights as zeros?
                 directional_updates: bool = False,  # Use directional updates?
                 separate_networks: bool = False,    # Use separate nets?
                 use_conv_net: bool = False,         # Use conv nets?
                 use_scattered_xnet_update: bool = False,  # scattered xupdate?
                 use_tempered_traj: bool = False,  # Use tempered trajectory?
                 gauge_eq_masks: bool = False):    # Use gauge eq. masks?
        super(GaugeDynamicsConfig, self).__init__(
            eps=eps,
            hmc=hmc,
            num_steps=num_steps,
            model_type=model_type,
            eps_fixed=eps_fixed,
            aux_weight=aux_weight,
            loss_scale=loss_scale,
            use_mixed_loss=use_mixed_loss,
            verbose=verbose,
            use_ncp=use_ncp,
            lattice_shape=lattice_shape,
            plaq_weight=plaq_weight,
            charge_weight=charge_weight,
            zero_init=zero_init,
            directional_updates=directional_updates,
            separate_networks=separate_networks,
            use_conv_net=use_conv_net,
            use_scattered_xnet_update=use_scattered_xnet_update,
            use_tempered_traj=use_tempered_traj,
            gauge_eq_masks=gauge_eq_masks,
        )
