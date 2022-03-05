"""
main.py

Contains entry point for training Dynamics.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def train_tensorflow(cfg: DictConfig) -> dict:
    import tensorflow as tf
    import horovod.tensorflow as hvd
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        gpu = gpus[hvd.local_rank()]
        tf.config.experimental.set_visible_devices(gpu, 'GPU')

    from l2hmc.main_tensorflow import main as main_tf
    output = main_tf(cfg)

    return output


def train_pytorch(cfg: DictConfig) -> dict:
    from l2hmc.main_pytorch import main as main_pt
    output = main_pt(cfg)
    return output


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # log.info(OmegaConf.create(cfg).pretty())
    # rich.print(OmegaConf.to_container(cfg, resolve=True))
    framework = cfg.get('framework', None)
    width = max((150, int(cfg.get('width', os.environ.get('COLUMNS', 150)))))
    cfg.update({'width': width})
    assert framework is not None, (
        'Framework must be specified, one of: [pytorch, tensorflow]'
    )
    if framework in ['tf', 'tensorflow']:
        _ = train_tensorflow(cfg)
    elif framework in ['pt', 'pytorch']:
        _ = train_pytorch(cfg)


if __name__ == '__main__':
    main()
