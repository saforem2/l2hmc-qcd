"""
Helper functions to simplify tensorflow interface.
"""
import tensorflow as tf

def clip_gradients(grads_and_vars, clip_ratio):
    """Clip gradients by global norm."""
    gradients, variables = zip(*grads_and_vars)
    clipped, _ = tf.clip_by_global_norm(gradients, clip_ratio)
    return zip(clipped, variables)
