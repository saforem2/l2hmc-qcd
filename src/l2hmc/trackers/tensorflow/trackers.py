"""
l2hmc/trackers/tensorflow/trackers.py
Contains various utilities for tracking / logging metrics in TensorBoard
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
from typing import DefaultDict, Mapping, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import backend as K
from l2hmc.configs import flatten_dict
# from l2hmc.utils.logger import get_pylogger


# log = get_pylogger(__name__)
# from l2hmc import get_logger
# log = get_logger(__name__)
log = logging.getLogger(__name__)

Layer = tf.keras.layers.Layer
Model = tf.keras.models.Model
Optimizer = tf.keras.optimizers.Optimizer
Tensor = tf.Tensor
Array = np.ndarray
Scalar = Union[float, int, bool]
ArrayLike = Union[Tensor, Array, Scalar]

tfComplex = [tf.dtypes.complex64, tf.dtypes.complex128]


def log_step(tag: str, step: int) -> None:
    iter_tag = '/'.join([tag.split('/')[0]] + ['iter'])
    tf.summary.scalar(iter_tag, step, step=step)


def check_tag(tag: str) -> str:
    tags = tag.split('/')
    return '/'.join(tags[1:]) if len(tags) > 2 and (tags[0] == tags[1]) else tag


def log_item(
        tag: str,
        val: float | int | bool | list | np.ndarray | tf.Tensor | None,
        step: Optional[int] = None,
):
    if val is None:
        return

    if step is not None:
        log_step(tag, step)

    tag = check_tag(tag)
    if isinstance(val, (Tensor, Array)):
        if (
            (isinstance(val, Tensor) and val.dtype in tfComplex)
            or (isinstance(val, Array) and np.iscomplexobj(val))
        ):
            log_item(tag=f'{tag}.real', val=tf.math.real(val), step=step)
            log_item(tag=f'{tag}.imag', val=tf.math.imag(val), step=step)
        elif hasattr(val, 'shape') and len(getattr(val, 'shape', [])) > 0:
            tf.summary.scalar(f'{tag}/avg', tf.reduce_mean(val), step=step)
            tf.summary.histogram(tag, val, step=step)
        else:
            tf.summary.scalar(tag, val, step=step)
    elif isinstance(val, list):
        log_list(val, step=step, prefix=tag)

    elif (
            isinstance(val, (float, int, np.floating, np.integer, bool))
            or len(val.shape) == 0
    ):
        tf.summary.scalar(tag, val, step=step)

    # elif isinstance(val, (tf.Tensor, np.ndarray)):
    #     if (
    #         (isinstance(val, tf.Tensor) and val.dtype in tfComplex)
    #         or isinstance(val, np.ndarray) and np.iscomplexobj(val)
    #     ):
    #         log_item(
    #             tag=f'{tag}.imag',
    #             val=tf.math.imag(val),
    #             step=step
    #         )
    #         log_item(
    #             tag=f'{tag}.real',
    #             val=tf.math.real(val),
    #             step=step
    #         )
    #     elif hasattr(val, 'shape') and len(getattr(val, 'shape', [])) > 0:
    #         tf.summary.scalar(f'{tag}/avg', tf.reduce_mean(val), step=step)
    #         tf.summary.histogram(tag, val, step=step)
    #     else:
    #         tf.summary.scalar(f'{tag}', val, step=step)
    # else:
    #     if hasattr(val, 'shape') and len(getattr(val, 'shape', [])) > 0:
    #         tf.summary.histogram(tag, val, step=step)
    #         tf.summary.scalar(f'{tag}/avg', tf.reduce_mean(val), step=step)
    #     else:
    #         try:
    #             tf.summary.scalar(tag, val, step=step)
    #         except Exception as e:
    #             log.exception(e)
    #             log.warning(f'Unexpected type encountered for: {tag}')
    #             log.warning(f'{tag}.type: {type(val)}')


def log_dict(
        d: dict | DefaultDict | Mapping,
        step: int,
        prefix: Optional[str] = None
):
    """Create tensorboard summaries for all items in `d`"""
    for key, val in d.items():
        pre = key if prefix is None else f'{prefix}/{key}'
        if isinstance(val, dict):
            log_dict(val, step=step, prefix=pre)
        elif isinstance(val, list):
            log_list(val, step, prefix=prefix)
        else:
            log_item(pre, val, step=step)


def log_list(x, step, prefix: Optional[str] = None):
    for idx, t in enumerate(x):
        name = getattr(t, 'name', getattr(t, '__name__', None))
        if name is None:
            name = f'{idx}'
        tag = name if prefix is None else f'{prefix}/{name}'
        assert tag is not None
        log_item(tag, t, step=step)


def log_model_weights1(
        step: int,
        model: tf.keras.Model | tf.keras.layers.Layer,
        prefix: Optional[str] = None,
):
    # assert isinstance(model, Model)
    # log_list(model.variables, step=step, prefix=prefix)
    prefix = f'model/{prefix}' if prefix is not None else 'model'
    name = getattr(model, 'name', None)
    if name is not None:
            prefix += f'/{name}'
    # prefix = f'{prefix}/{name}' if name is not None else f'model/prefix'
    log_list(
        model.trainable_variables,
        step=step,
        prefix=f'{prefix}/trainable_vars'
    )
    layers = getattr(model, 'layers', [])
    if len(layers) > 0:
        for layer in model.layers:
            weights = layer.get_weights()
            log_list(
                weights,
                step=step,
                prefix=f'{prefix}/{layer.name}.weights'
            )
    else:
        weights = model.get_weights()
        log_list(
            weights,
            step=step,
            prefix=f'{prefix}/{model.name}.weights'
        )


def log_model_weights(
        step: int,
        model: tf.keras.Model | tf.keras.layers.Layer,
        prefix: Optional[str] = None,
        sep: Optional[str] = None,
):
    weights = model.weights
    wdict = {w.name: w for w in weights}
    # if pre is None:
    #     wdict = {w.name: w for w in weights}
    # else:
    #     wdict = {
    #         f'{pre}/{w.name}': w
    #         for w in weights
    #     }
    if sep is not None:
        wdict.update({
            k.replace('/', sep): v
            for k, v in wdict.items()
        })

    log_dict(wdict, step, prefix=prefix)


def format_weight_name(name: str) -> str:
    return name.rstrip(':0').replace('kernel', 'weight')


def update_summaries(
        step: int,
        metrics: Optional[dict[str, Tensor]] = None,
        model: Optional[Model] = None,
        optimizer: Optional[Optimizer] = None,
        prefix: Optional[str] = None,
) -> None:
    if metrics is not None and isinstance(metrics, dict):
        log_dict(metrics, step, prefix=prefix)

    if model is not None:
        weights = {
            format_weight_name(w.name): w for w in model.weights
        }
        
        assert isinstance(weights, (dict, DefaultDict, Mapping))
        log_dict(
            flatten_dict(weights),
            step=step,
            prefix='model'
        )

    if optimizer is not None:
        ostr = 'optimizer'
        opre = ostr if prefix is None else '/'.join([ostr, prefix])
        lpre = 'lr' if prefix is None else '/'.join(['lr', prefix])
        tf.summary.scalar(lpre, K.get_value(optimizer.lr), step=step)
        log_list(optimizer.variables(), step=step, prefix=opre)


def update_summaries1(
        step: int,
        metrics: Optional[dict] = None,
        # model: Optional[Model | Layer | dict[str, Model | Layer]] = None,
        model: Optional[Model | Layer] = None,
        weights: Optional[dict] = None,
        optimizer: Optional[Optimizer] = None,
        prefix: Optional[str] = None,
        # name: Optional[str] = None,
        job_type: Optional[str] = None,
        sep: Optional[str] = None,
):
    """"Create summary objects."""
    if metrics is not None:
        log_dict(metrics, step, prefix=job_type)

    if weights is not None:
        log.info('Caught weights!')
        if isinstance(weights, dict):
            log_dict(weights, step=step, prefix=prefix)

    if model is not None:
        weights = {w.name: w for w in model.weights}
        if weights != {}:
            log_dict(weights, step, prefix='network')

        if isinstance(model, Model):
            log_model_weights(
                step=step,
                model=model,
                sep=sep,
                prefix=prefix,
            )
        # if isinstance(model, dict):
        #     for prefix, val in model.items():
        #         if isinstance(val, Model):
        #             log_model_weights(
        #                 step=step,
        #                 model=val,
        #                 sep=sep
        #             )
        #             wdict = {w.name: w for w in val.weights}
        # elif isinstance(model, (Model, Layer)):
        #     wdict = {w.name: w for w in model.weights}
        # else:
        #     raise TypeError

        # ------------------------------------------------------
        # if isinstance(model, Model):
        #     assert isinstance(model, Model)
        #     wlist = model.weights
        #     wdict = {w.name: w for w in wlist}
        #     log_dict(wdict, step, prefix=model.name)
        #
        # if isinstance(model, dict):
        #     prefix = 'model'  # /{subname}/{layer.name}.weights'
        #     if name is not None:
        #         prefix += f'/{name}'
        #     for subname, submodel in model.items():
        #         if isinstance(submodel, dict):
        #             update_summaries(
        #                 step=step,
        #                 model=submodel,
        #                 name=subname,
        #             )
        #         else:
        #             layers = getattr(submodel, 'layers', [])
        #             if len(layers) > 0:
        #                 for layer in layers:
        #                     weights = layer.get_weights()
        #                     pre = (
        #                         f'{prefix}/{subname}/{layer.name}/weights'
        #                     )
        #                     log_list(
        #                         weights,
        #                         step=step,
        #                         prefix=pre,
        #                         # prefix=f'{prefix}/{subname}/{layer.name}.weights',
        #                         # prefix=f'model/{subname}/{layer.name}.weights'
        #                     )
        #             else:
        #                 weights = submodel.get_weights()
        #                 log_list(
        #                     weights,
        #                     step=step,
        #                     prefix=f'{prefix}/{subname}/weights'
        #                 )
        #     # --------------------------------------------------------------
        # elif isinstance(model, (Model, Layer)):
        #     log_model_weights(step, model, prefix='model')
        # --------------------------------------------------------------
        # if isinstance(model, dict):
        #     for subname, submodel in model.items():
        #         update_summaries(
        #             step,
        #             model=submodel,
        #             prefix=f'{subname}',
        #             # prefix=f'{prefix}/{name}',
        #         )
        #         # if isinstance(submodel, Model):
        #         #     log_model_weights(step, submodel, f'{prefix}/{name}')
        # elif isinstance(model, (Model, tf.keras.layers.Layer)):
        #     log_model_weights(step, model, prefix)
        # --------------------------------------------------------------

        # pstr = 'model'
        # prefix = f'{pstr}'
        # prefix = f'model/{prefix}' if prefix is not None else 'model'
        # name = getattr(model, 'name', None)
        # if name is not None:
        #     prefix += f'/{name}'
        # prefix = f'model/{prefix}'
        # prefix = 'model' if prefix is None else prefix
        # log_list(model.variables, step=step, prefix='model')
        # # weights = model.trainable_variables()
        # log_list(
        #     model.trainable_variables,
        #     step=step,
        #     prefix='model/trainable_vars'
        # )
        # for layer in model.layers:
        #     weights = layer.get_weights()
        #     log_list(
        #         weights,
        #         step=step,
        #         prefix=f'model/{layer.name}.weights'
        #     )
        # -------------------------------
        # weights = {
        #     'model/iter': step,
        #     'model/variables': model.variables,
        # }
        #     w = layer.get_weights()
        #     log_list(
        #         w,
        #         step=step,
        #         prefix=f'model/{layer.name}.weights',
        #     )
        #     log_item(
        #         tag=f'model/{layer.name}.bias',
        #         val=b,
        #         step=step,
        #     )
        #     log_item(w, step=step, prefix=f'model/{layer.name}.weight')
        #     log_list(b, step=step, prefix=f'model/{layer.name}.bias')
        #     weights[layer] = wb
        # log_dict(weights, step=step, prefix='model')

        # log_dict(weights, step=step, prefix='weights')
        # log_list(weights, step=step, prefix=f'model/{layer}')

    if optimizer is not None:
        ostr = 'optimizer'
        opre = ostr if prefix is None else '/'.join([ostr, prefix])
        lpre = 'lr' if prefix is None else '/'.join(['lr', prefix])
        tf.summary.scalar(lpre, K.get_value(optimizer.lr), step=step)
        # assert isinstance(optimizer, tf.python.keras.optimizers.Optimizer)
        log_list(optimizer.variables(), step=step, prefix=opre)
