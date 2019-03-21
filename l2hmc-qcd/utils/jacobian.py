"""
Code borrowed from:
https://stackoverflow.com/questions/48878053/tensorflow-gradient-with-respect-to-matrix
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe


def _map(f, x, dtype=None, parallel_iterations=10):
    """
    Apply f to each of the elements in x using the specified number of parallel
    iterations.

    Important points:
        1.) By "elements in x", we mean that we will be applying f to 
        x[0], ... , x[tf.shape(x)[0] - 1].
        2.) The output size of f(x[i]) can be arbitrary. However, if the dtype
        of that output is different than the dtype of x, you need to specify
        that as an additional argument.
    """
    if dtype is None:
        dtype = x.dtype

    n = tf.shape(x)[0]
    loop_vars = [
        tf.constant(0, n.dtype),
        tf.TensorArray(dtype, size=n),
    ]

    _, fx = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j + 1, result.write(j, f(x[j]))),
        loop_vars,
        parallel_iterations=parallel_iterations
    )
    return fx.stack()


def jacobian(fx, x, parallel_iterations=10):
    """Given a tensor fx, which is a function of x, vectorize fx (via
    tf.reshape(fx, [-1])) and then compute the Jacobian of each entry of fx
    with respect to x. 

    Specifically, if x has shape (n1, n2, ..., np) and fx has L entries
    (tf.size(fx) = L), then the output will be (L, n1, n2, ..., np), where
    output[i] will be (n1, n2, ..., np) with each entry denoting the gradient
    of output[i] wrt the corresponding element of x.
    """
    def _grads(fxi, x):
        x = np.array(x)
        if tf.executing_eagerly():
            grad_fn = tfe.gradients_function(fxi)
            grad = grad_fn(x[0])[0]
        else:
            grad_fn = tf.gradients(fxi, x)
            grad = grad_fn(x[0])[0]
        return grad

    return _map(_grads(fx, x),
               #  lambda fxi: tf.gradients(fxi, x)[0],
               tf.reshape(fx, [-1]),
               dtype=x.dtype,
               parallel_iterations=parallel_iterations)

