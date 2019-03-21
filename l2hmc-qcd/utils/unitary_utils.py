import numpy as np

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

from math import sqrt


def complex_matmul(_c, _r):
    return tf.complex(tf.real(_c) * _r, tf.imag(_c) * _r)


def refl_c(in_, normal_):
    normal_rk2 = tf.expand_dims(normal_, 1)
    scale = 2 * tf.matmul(in_, tf.conj(normal_rk2))
    return in_ - tf.matmul(scale, tf.transpose(normal_rk2))


def complex_abs_sq(z):
    return tf.real(z) * tf.real(z) + tf.imag(z) * tf.imag(z)


def normalize_c(in_):
    norm = tf.sqrt(tf.reduce_sum(complex_abs_sq(in_)))
    scale = 1.0 / (norm + 1e-5)
    return complex_mul_real(in_, scale)


def get_complex_variable(name, scope, shape):
    re = vs.get_variable(name + '_re', shape=shape)
    im = vs.get_variable(name + '_im', shape=shape)
    return tf.complex(re, im, name=name)


def complex_mul_unitary(vec_in, out_size, scope=None):
    shape = vec_in.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError('Argument `vec_in` must be a batch of vectors'
                         ' (2D Tensor)')
    #  in_size = shape[1]
    fft_scale = 1.0 / sqrt(out_size)
    with vs.variable_scope(scope or 'ULinear') as _s:
        diag0 = get_complex_variable('diag0', _s, shape=[out_size])
        diag1 = get_complex_variable('diag1', _s, shape=[out_size])
        diag2 = get_complex_variable('diag2', _s, shape=[out_size])
        refl0 = get_complex_variable('refl0', _s, shape=[out_size])
        refl1 = get_complex_variable('refl1', _s, shape=[out_size])
        perm0 = tf.constant(np.random.permutation(out_size),
                            name='perm0', dtype=tf.int32)
        out_ = vec_in * diag0
        refl0 = normalize_c(refl0)
        refl1 = normalize_c(refl1)
        out_ = refl_c(math_ops.batch_fft(out_) * fft_scale, refl0)
        out_ = diag1 * tf.transpose(tf.gather(tf.transpose(out_), perm0))
        out_ = diag2 * refl_c(math_ops.batch_ifft(out_) * fft_scale, refl1)

        return out_


def modReLU(in_c, bias, scope=None):
    with vs.variable_scope(scope or 'ULinear'):
        n = tf.abs(in_c)
        scale = 1.0 / (n + 1e-5)
    return complex_mul_real(in_c, (nn_ops.relu(n + bias) * scale))
