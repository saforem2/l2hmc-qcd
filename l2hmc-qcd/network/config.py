"""
config.py

Contains configuration objects for networks.
"""
from typing import List, Optional, Callable

import tensorflow as tf

from utils.attr_dict import AttrDict
from dataclasses import dataclass

#
#  @dataclass
#  class LearningRateConfig:
#      lr_init: float
#      decay_steps: int = int(5e3)
#      decay_rate: float = 0.99
#      warmup_steps: int = 0
#
#
#  @dataclass
#  class NetworkConfig:
#      """Configuration object for model network."""
#      units: list
#      name: str = None
#      dropout_prob: float = 0.
#      activation_fn: callable = tf.nn.relu
#      use_batch_norm: bool = False
#
#
#  @dataclass
#  class ConvolutionConfig:
#      """Configuration object for convolutional block of network."""
#      input_shape: List
#      filters: List
#      sizes: List
#      pool_sizes: List = None
#      conv_activations: List = None
#      conv_paddings: List = None
#      use_batch_norm: bool = True
#      name: str = None
#

#  pylint:disable=too-many-arguments
@dataclass
class LearningRateConfig:
    lr_init: float
    decay_steps: int
    decay_rate: float
    warmup_steps: int = 0


@dataclass
class NetworkConfig:
    units: list
    dropout_prob: float = 0.
    activation_fn: callable = tf.nn.relu
    use_batch_norm: bool = True


@dataclass
class ConvolutionConfig:
    input_shape: tuple
    filters: tuple
    sizes: tuple
    pool_sizes: tuple
    conv_activations: tuple = None
    conv_paddings: tuple = None
    use_batch_norm: bool = True


#  class LearningRateConfig(AttrDict):
#      """Configuration object for specifying learning rate schedule."""
#
#      def __init__(
#              self,
#              lr_init: float,
#              decay_steps: int,
#              decay_rate: float,
#              warmup_steps: Optional[int] = 0
#      ):
#          super(LearningRateConfig, self).__init__(
#              lr_init=lr_init,
#              decay_steps=decay_steps,
#              decay_rate=decay_rate,
#              warmup_steps=warmup_steps
#          )
#

#  class NetworkConfig(AttrDict):
#      """Configuration object for network of `Dynamics` object"""
#
#      def __init__(
#              self,
#              units: list,
#              name: Optional[str] = None,              # Name of network
#              dropout_prob: Optional[float] = 0.,      # Dropout probability
#              activation_fn: Optional[Callable] = tf.nn.relu,
#              use_batch_norm: Optional[bool] = False,
#      ):
#          super(NetworkConfig, self).__init__(
#              name=name,
#              units=units,
#              dropout_prob=dropout_prob,
#              activation_fn=activation_fn,
#              use_batch_norm=use_batch_norm
#          )
#
#
#  class ConvolutionConfig(AttrDict):
#      """Defines a configuration object for passing to `ConvolutionBlock`."""
#
#      def __init__(
#              self,
#              input_shape: List[int],  # expected input shape
#              filters: List[int],      # number of filters to use
#              sizes: List[int],        # filter sizes to use
#              pool_sizes: Optional[List[int]] = None,
#              conv_activations: Optional[List[str]] = None,
#              conv_paddings: Optional[List[str]] = None,
#              use_batch_norm: Optional[bool] = False,
#              name: Optional[str] = None,
#      ):
#          super(ConvolutionConfig, self).__init__(
#              input_shape=input_shape,
#              filters=filters,
#              sizes=sizes,
#              pool_sizes=pool_sizes,
#              conv_activations=conv_activations,
#              conv_paddings=conv_paddings,
#              use_batch_norm=use_batch_norm,
#              name=name,
#          )
