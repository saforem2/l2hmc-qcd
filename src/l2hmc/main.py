"""
main.py

Contains entry point for training Dynamics.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
import warnings

import hydra
from omegaconf import DictConfig


log = logging.getLogger(__name__)


def train_tensorflow(cfg: DictConfig) -> dict:
    import tensorflow as tf
    tf.keras.backend.set_floatx('float32')  # or 'float64 for double precision
    assert tf.keras.backend.floatx() == tf.float32
    import horovod.tensorflow as hvd
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        gpu = gpus[hvd.local_rank()]
        tf.config.experimental.set_visible_devices(gpu, 'GPU')

    from l2hmc.scripts.tensorflow.main import main as main_tf
    output = main_tf(cfg)

    return output


def train_pytorch(cfg: DictConfig) -> dict:
    import torch
    if cfg.precision == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    from l2hmc.scripts.pytorch.main import main as main_pt
    return main_pt(cfg)


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    width = cfg.get('width', None)
    if width is not None and os.environ.get('COLUMNS', None) is None:
        os.environ['COLUMNS'] = str(width)
    elif os.environ.get('COLUMNS', None) is not None:
        cfg.update({'width': int(os.environ.get('COLUMNS', 235))})

    framework = cfg.get('framework', None)
    assert framework is not None, (
        'Framework must be specified, one of: [pytorch, tensorflow]'
    )

    if cfg.get('ignore_warnings'):
        warnings.filterwarnings('ignore')

    if framework in ['tf', 'tensorflow']:
        _ = train_tensorflow(cfg)
    elif framework in ['pt', 'pytorch']:
        _ = train_pytorch(cfg)


if __name__ == '__main__':
    main()
