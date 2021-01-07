"""
config.py

Contains configuration objects for networks.
"""
from dataclasses import dataclass
import tensorflow as tf



@dataclass
class LearningRateConfig:
    """Learning rate configuration object."""
    lr_init: float
    decay_steps: int
    decay_rate: float
    warmup_steps: int = 0


@dataclass
class NetworkConfig:
    """Network configuration object."""
    units: list
    dropout_prob: float = 0.
    activation_fn: callable = tf.nn.relu
    use_batch_norm: bool = True


@dataclass
class ConvolutionConfig:
    """Convolutional network configuration."""
    input_shape: tuple
    filters: tuple
    sizes: tuple
    pool_sizes: tuple
    conv_activations: tuple = None
    conv_paddings: tuple = None
    use_batch_norm: bool = True
