"""
config.py

Contains configuration objects for networks.
"""
from typing import List, Optional

import tensorflow as tf

from utils.attr_dict import AttrDict


class LearningRateConfig(AttrDict):
    """Configuration object for specifying learning rate schedule."""
    def __init__(self,
                 lr_init: float,
                 decay_steps: int,
                 decay_rate: float,
                 warmup_steps: int = 0):
        super(LearningRateConfig, self).__init__(
            lr_init=lr_init,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            warmup_steps=warmup_steps
        )


class NetworkConfig(AttrDict):
    """Configuration object for network of `Dynamics` object"""

    def __init__(self,
                 units: list,
                 name: str = None,
                 dropout_prob: float = 0.,
                 activation_fn: callable = tf.nn.relu):
        super(NetworkConfig, self).__init__(
            name=name,
            units=units,
            dropout_prob=dropout_prob,
            activation_fn=activation_fn
        )


class ConvolutionConfig(AttrDict):
    """Defines a configuration object for passing to `ConvolutionBlock`."""
    def __init__(
            self,
            input_shape: List[int],  # expected input shape
            filters: List[int],      # number of filters to use
            sizes: List[int],        # filter sizes to use
            pool_sizes: Optional[List[int]] = None,  # MaxPooling sizes
            conv_activations: Optional[List[str]] = None,  # Activation fns
            conv_paddings: Optional[List[str]] = None,  # Paddings to use
            use_batch_norm: Optional[bool] = False,  # Use batch normalization?
            name: Optional[str] = None,  # Name of model
    ):
        super(ConvolutionConfig, self).__init__(
            input_shape=input_shape,
            filters=filters,
            sizes=sizes,
            pool_sizes=pool_sizes,
            conv_activations=conv_activations,
            conv_paddings=conv_paddings,
            use_batch_norm=use_batch_norm,
            name=name,
        )
