"""
main_tensorflow.py

Main entry-point for training L2HMC Dynamics w/ TensorFlow
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from l2hmc.utils.hvd_init import IS_CHIEF

from l2hmc.common import (
    analyze_dataset,
    get_timestamp,
    save_logs,
    setup_annealing_schedule,
)
from l2hmc.configs import (
    ConvolutionConfig,
    DynamicsConfig,
    InputSpec,
    LossConfig,
    NetWeights,
    NetworkConfig,
    Steps,
)
from l2hmc.dynamics.tensorflow.dynamics import Dynamics
from l2hmc.lattice.tensorflow.lattice import Lattice
from l2hmc.loss.tensorflow.loss import LatticeLoss
from l2hmc.network.tensorflow.network import NetworkFactory
from l2hmc.trainers.tensorflow.trainer import Trainer
from l2hmc.utils.console import is_interactive
log = logging.getLogger(__name__)


def setup(cfg: DictConfig) -> dict:
    steps = Steps(**cfg.steps)
    loss_cfg = LossConfig(**cfg.loss)
    net_weights = NetWeights(**cfg.net_weights)
    network_cfg = NetworkConfig(**cfg.network)
    dynamics_cfg = DynamicsConfig(**cfg.dynamics)
    schedule = setup_annealing_schedule(cfg)
    conv_cfg = cfg.get('conv', None)
    if conv_cfg is not None:
        conv_cfg = (
            ConvolutionConfig(**cfg.conv)
            if len(cfg.conv.keys()) > 0
            else None
        )

    xdim = dynamics_cfg.xdim
    xshape = dynamics_cfg.xshape
    input_spec = InputSpec(xshape=xshape,
                           vnet={'v': [xdim, ], 'x': [xdim, ]},
                           xnet={'v': [xdim, ], 'x': [xdim, 2]})
    net_factory = NetworkFactory(input_spec=input_spec,
                                 net_weights=net_weights,
                                 network_config=network_cfg,
                                 conv_config=conv_cfg)
    lattice = Lattice(tuple(xshape))
    dynamics = Dynamics(config=dynamics_cfg,
                        potential_fn=lattice.action,
                        network_factory=net_factory)
    loss_fn = LatticeLoss(lattice=lattice, loss_config=loss_cfg)
    optimizer = tf.keras.optimizers.Adam()
    trainer = Trainer(steps=steps,
                      loss_fn=loss_fn,
                      schedule=schedule,
                      dynamics=dynamics,
                      optimizer=optimizer,
                      aux_weight=loss_cfg.aux_weight)
    return {
        'lattice': lattice,
        'loss_fn': loss_fn,
        'dynamics': dynamics,
        'trainer': trainer,
        'optimizer': optimizer,
    }


def train(cfg: DictConfig) -> dict:
    objs = setup(cfg)
    trainer = objs['trainer']  # type: Trainer
    kwargs = {
        'save_x': cfg.get('save_x', False),
        'width': cfg.get('width', 150),
        'compile': cfg.get('compile', True),
        'jit_compile': cfg.get('jit_compile', False),
    }

    train_output = trainer.train(**kwargs)
    output = {
        'setup': setup,
        'train': train_output,
    }
    if IS_CHIEF:
        outdir = Path(cfg.get('outdir', os.getcwd()))
        # day = get_timestamp('%Y-%m-%d')
        # time = get_timestamp('%H-%M-%S')
        # outdir = outdir.joinpath('tensorflow', day, time)
        train_dir = outdir.joinpath('train')

        train_dataset = train_output['history'].get_dataset()
        nchains = min((cfg.dynamics.xshape[0], cfg.dynamics.nleapfrog))

        analyze_dataset(train_dataset,
                        name='train',
                        nchains=nchains,
                        outdir=train_dir,
                        lattice=objs['lattice'],
                        xarr=train_output['xarr'],
                        title='Training: TensorFlow')

        _ = kwargs.pop('save_x', False)
        tfrac = cfg.get('therm_frac', 0.2)
        eval_dir = outdir.joinpath('eval')
        eval_output = trainer.eval(**kwargs)
        eval_dataset = eval_output['history'].get_dataset(therm_frac=tfrac)
        analyze_dataset(eval_dataset,
                        name='eval',
                        nchains=nchains,
                        outdir=eval_dir,
                        lattice=objs['lattice'],
                        xarr=eval_output['xarr'],
                        title='Evaluating: TensorFlow')

        if not is_interactive():
            tdir = train_dir.joinpath('logs')
            edir = eval_dir.joinpath('logs')
            tdir.mkdir(exist_ok=True, parents=True)
            edir.mkdir(exist_ok=True, parents=True)
            log.info(f'Saving train logs to: {tdir.as_posix()}')
            save_logs(logdir=tdir,
                      tables=train_output['tables'],
                      summaries=train_output['summaries'])
            log.info(f'Saving eval logs to: {edir.as_posix()}')
            save_logs(logdir=edir,
                      tables=eval_output['tables'],
                      summaries=eval_output['summaries'])

        output.update({'eval': eval_output})

    return output


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    log.info(f'Working directory: {os.getcwd()}')
    log.info(OmegaConf.to_yaml(cfg))
    _ = train(cfg)


if __name__ == '__main__':
    main()
