"""
main.py

Contains entry point for training Dynamics.
"""
from __future__ import absolute_import, print_function, division, annotations
from omegaconf import DictConfig, OmegaConf
import os
import hydra
import logging
from l2hmc.common import train

log = logging.getLogger(__name__)


def train_tensorflow(cfg: DictConfig) -> dict:
    from l2hmc.main_tensorflow import train
    output = train(cfg)

    return output


def train_pytorch(cfg: DictConfig) -> dict:
    from l2hmc.main_pytorch import train
    output = train(cfg)
    return output


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))
    print(OmegaConf.to_yaml(cfg))
    _ = train(cfg)


if __name__ == '__main__':
    main()
