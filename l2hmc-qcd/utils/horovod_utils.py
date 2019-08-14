from config import TF_FLOAT
import tensorflow as tf


def cast(f):
    return tf.cast(f, TF_FLOAT)


def configure_learning_rate(*args):
    """Implements gradual learning rate warmup:
        
        `lr = lr_init / hvd.size()` --> `lr = lr_init`

    where `lr_init` is the learning rate of the model optimizer at the start of
    training.


    Based on: horovod.keras.LearningRateWarmupCallback class.
    https://github.com/horovod/horovod/blob/master/horovod/keras/callbacks.py

    This technique was described in the paper "Accurate, Large Minibatch SGD:
        Training ImageNet in 1 Hour". See https://arxiv.org/pdf/1706.02677.pdf
        for details.

    Math recap:
                                                batch
        epoch               = full_epochs + ---------------
                                            steps_per_epoch

                               lr     size - 1
        lr'(epoch)          = ---- * (-------- * epoch + 1)
                              size     warmup

                               lr
        lr'(epoch = 0)      = ----
                              size
        lr'(epoch = warmup) = lr

    Args:
        lr_warmup: Initial (smaller) value of the learning rate.
        lr_init: Target value of the learning rate. (lr_warmup --> lr_init)
        decay_steps: Learning rate decay steps following warmup.
        global_step: Tensorflow global step object. 
        warmup_steps: Number of steps over which to warmup the learning rate.
    """
    #  for i in args:
    #      if isinstance(i, float):
    #          i = tf.cast(i, dtype=TF_FLOAT)

    lr_warmup = args[0]
    lr_init, decay_steps, decay_rate, global_step, warmup_steps = args[1:]

    learning_rate = tf.train.exponential_decay(lr_init, global_step,
                                               decay_steps, decay_rate,
                                               staircase=False,
                                               name='learning_rate')
    if warmup_steps > 0:
        def warmup_decay(lr1, global_step, warmup_steps, lr2):
            p = global_step / warmup_steps
            diff = lr2 - lr1
            res = lr1 + (diff * p)
            return tf.cast(res, TF_FLOAT)

    learning_rate = tf.cond(global_step < warmup_steps,
                            lambda: warmup_decay(lr_warmup, global_step,
                                                 warmup_steps,
                                                 lr_init),
                            lambda: learning_rate)

    return learning_rate


def warmup_lr(**kwargs):
    target_lr = kwargs.get('target_lr', 1e-3)
    warmup_steps = kwargs.get('warmup_steps', 1000)
    global_step = kwargs.get('global_step')
    decay_steps = kwargs.get('decay_steps', 1000)
    decay_rate = kwargs.get('decay_rate', 0.96)

    learning_rate = tf.train.exponential_decay(target_lr, global_step,
                                               decay_steps, decay_rate,
                                               staircase=False,
                                               name='learning_rate')

    def warmup(step):
        return step * target_lr / warmup_steps

    learning_rate = tf.cond(global_step < warmup_steps,
                            lambda: warmup(global_step),
                            lambda: learning_rate)

    return learning_rate

