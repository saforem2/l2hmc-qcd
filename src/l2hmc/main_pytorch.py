"""
main_pytorch.py

Contains entry-point for training and inference.
"""
from __future__ import absolute_import, annotations, division, print_function
from omegaconf import DictConfig, OmegaConf
import os
import hydra
import logging
from pathlib import Path
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from accelerate import Accelerator

from l2hmc.dynamics.pytorch.dynamics import Dynamics
from l2hmc.lattice.pytorch.lattice import Lattice
from l2hmc.network.pytorch.network import NetworkFactory
from l2hmc.trainers.pytorch.trainer import Trainer
from l2hmc.loss.pytorch.loss import LatticeLoss


from l2hmc.common import (
    analyze_dataset, save_logs, setup_annealing_schedule,
    get_timestamp,
)
from l2hmc.utils.console import is_interactive

from l2hmc.configs import (
    ConvolutionConfig,
    DynamicsConfig,
    InputSpec,
    LossConfig,
    NetWeights,
    NetworkConfig,
    Steps,
)

log = logging.getLogger(__name__)


def setup(cfg: DictConfig) -> dict:
    accelerator = Accelerator()
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
    network_factory = NetworkFactory(input_spec=input_spec,
                                     net_weights=net_weights,
                                     network_config=network_cfg,
                                     conv_config=conv_cfg)
    lattice = Lattice(tuple(xshape))
    dynamics = Dynamics(config=dynamics_cfg,
                        potential_fn=lattice.action,
                        network_factory=network_factory)
    loss_fn = LatticeLoss(lattice=lattice, loss_config=loss_cfg)

    optimizer = torch.optim.Adam(dynamics.parameters())
    dynamics = dynamics.to(accelerator.device)
    dynamics, optimizer = accelerator.prepare(dynamics, optimizer)
    trainer = Trainer(steps=steps,
                      loss_fn=loss_fn,
                      dynamics=dynamics,
                      schedule=schedule,
                      optimizer=optimizer,
                      accelerator=accelerator,
                      dynamics_config=dynamics_cfg,
                      # evals_per_step=nlf,
                      aux_weight=loss_cfg.aux_weight)

    return {
        'lattice': lattice,
        'loss_fn': loss_fn,
        'dynamics': dynamics,
        'trainer': trainer,
        'optimizer': optimizer,
        'accelerator': accelerator,
    }


@record
def train(cfg: DictConfig) -> dict:
    objs = setup(cfg)
    trainer = objs['trainer']  # type: Trainer
    accelerator = objs['accelerator']  # type: Accelerator
    kwargs = {
        'save_x': cfg.get('save_x', False),
        'width': cfg.get('width', None),
    }

    # outdir = Path(cfg.get('outdir', os.getcwd()))
    # day = get_timestamp('%Y-%m-%d')
    # time = get_timestamp('%H-%M-%S')
    # outdir = outdir.joinpath('pytorch').joinpath(day, time)
    # train_dir = outdir.joinpath('train')
    train_output = trainer.train(**kwargs)
    output = {
        'setup': setup,
        'train': train_output,
    }
    if accelerator.is_local_main_process:
        outdir = Path(cfg.get('outdir', os.getcwd()))
        # day = get_timestamp('%Y-%m-%d')
        # time = get_timestamp('%H-%M-%S')
        # outdir = outdir.joinpath('pytorch', day, time)
        train_dir = outdir.joinpath('train')

        train_dset = train_output['history'].get_dataset()
        nchains = min((cfg.dynamics.xshape[0], cfg.dynamics.nleapfrog))

        analyze_dataset(train_dset,
                        name='train',
                        nchains=nchains,
                        outdir=train_dir,
                        lattice=objs['lattice'],
                        xarr=train_output['xarr'],
                        title='Training: PyTorch')

        _ = kwargs.pop('save_x', False)
        therm_frac = cfg.get('therm_frac', 0.2)
        eval_dir = outdir.joinpath('eval')
        eval_output = trainer.eval(**kwargs)
        eval_dset = eval_output['history'].get_dataset(
            therm_frac=therm_frac
        )
        analyze_dataset(eval_dset,
                        name='eval',
                        nchains=nchains,
                        outdir=eval_dir,
                        lattice=objs['lattice'],
                        xarr=eval_output['xarr'],
                        title='Evaluating: PyTorch')

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
