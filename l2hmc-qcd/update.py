import tensorflow as tf
import numpy as np
import config


def set_precision(precision):
    """Set floating point precision project-wide."""
    if precision == 'float64':
        config.TF_FLOAT = tf.float64
        config.NP_FLOAT = np.float64
        config.TF_INT = tf.int64
        config.NP_INT = np.int64
    elif precision == 'float32':
        config.TF_FLOAT = tf.float32
        config.NP_FLOAT = np.float32
        config.TF_INT = tf.int32
        config.NP_INT = np.int32


def set_seed(seed):
    config.GLOBAL_SEED = seed
