from config import TF_FLOAT
import tensorflow as tf


def cast(f):
    return tf.cast(f, TF_FLOAT)


def warmup_lr(**kwargs):
    """Create a dynamic learning rate schedule that slowly warms up."""
    global_step = kwargs.get('global_step')
    target_lr = kwargs.get('target_lr', 1e-3)
    decay_rate = kwargs.get('decay_rate', 0.96)
    decay_steps = kwargs.get('decay_steps', 1000)
    warmup_steps = kwargs.get('warmup_steps', 1000)

    learning_rate = tf.train.exponential_decay(target_lr, global_step,
                                               decay_steps, decay_rate,
                                               staircase=True,
                                               name='learning_rate')

    def warmup(global_step):
        return tf.cast(target_lr * (global_step / warmup_steps),
                       dtype=TF_FLOAT)

    learning_rate = tf.cond(global_step < warmup_steps,
                            lambda: warmup(global_step),
                            lambda: learning_rate)

    return learning_rate
