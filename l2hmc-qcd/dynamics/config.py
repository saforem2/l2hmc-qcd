"""
config.py

Contains configuration objects for various `Dynamics`.
"""
from utils.attr_dict import AttrDict


class GaugeDynamicsConfig(AttrDict):
    """Configuration object for `GaugeDynamics` object"""

    # pylint:disable=too-many-arguments
    def __init__(self,
                 eps: float,                    # step size
                 num_steps: int,                # n leapfrog steps per acc/rej
                 hmc: bool = False,             # run standard HMC?
                 use_ncp: bool = False,         # Transform x using NCP?
                 model_type: str = None,        # name for model
                 eps_fixed: bool = False,       # Fixed step size?
                 lattice_shape: tuple = None,   # (batch_size, Lt, Lx, 2)
                 aux_weight: float = 0.,        # Weight of aux term in loss fn
                 plaq_weight: float = 0.,       # Weight of plaq term in loss
                 charge_weight: float = 0.,     # Weight of charge term in loss
                 zero_init: bool = False,       # Initialize weights as zeros?
                 directional_updates: bool = False,  # Use directional updates?
                 separate_networks: bool = False,    # Use separate nets?
                 use_conv_net: bool = False,    # Use conv nets?
                 use_mixed_loss: bool = False,  # Use mixed loss?
                 use_scattered_xnet_update: bool = False,  # scattered xupdate?
                 use_tempered_traj: bool = False,  # Use tempered trajectory?
                 gauge_eq_masks: bool = False):    # Use gauge eq. masks?
        super(GaugeDynamicsConfig, self).__init__(
            eps=eps,
            hmc=hmc,
            use_ncp=use_ncp,
            num_steps=num_steps,
            model_type=model_type,
            eps_fixed=eps_fixed,
            lattice_shape=lattice_shape,
            aux_weight=aux_weight,
            plaq_weight=plaq_weight,
            charge_weight=charge_weight,
            zero_init=zero_init,
            directional_updates=directional_updates,
            separate_networks=separate_networks,
            use_conv_net=use_conv_net,
            use_mixed_loss=use_mixed_loss,
            use_scattered_xnet_update=use_scattered_xnet_update,
            use_tempered_traj=use_tempered_traj,
            gauge_eq_masks=gauge_eq_masks,
        )


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
                 use_mixed_loss: bool = False):
        super(DynamicsConfig, self).__init__(
            eps=eps,
            hmc=hmc,
            num_steps=num_steps,
            model_type=model_type,
            eps_fixed=eps_fixed,
            aux_weight=aux_weight,
            loss_scale=loss_scale,
            use_mixed_loss=use_mixed_loss
        )

