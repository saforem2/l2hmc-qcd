"""
main_pytorch.py

Contains entry-point for training and inference.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from pathlib import Path

from accelerate import Accelerator
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch.distributed.elastic.multiprocessing.errors import record

from l2hmc.common import analyze_dataset, save_logs
from l2hmc.configs import (
    ConvolutionConfig,
    AnnealingSchedule,
    DynamicsConfig,
    InputSpec,
    LossConfig,
    NetWeights,
    LearningRateConfig,
    NetworkConfig,
    Steps,
)
from l2hmc.dynamics.pytorch.dynamics import Dynamics
from l2hmc.lattice.pytorch.lattice import Lattice
from l2hmc.loss.pytorch.loss import LatticeLoss
from l2hmc.network.pytorch.network import NetworkFactory
from l2hmc.trainers.pytorch.trainer import Trainer
from l2hmc.utils.console import is_interactive


log = logging.getLogger(__name__)


def load_from_ckpt(
        dynamics: Dynamics,
        optimizer: torch.optim.Optimizer,
        cfg: DictConfig,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict]:
    outdir = Path(cfg.get('outdir', os.getcwd()))
    ckpts = list(outdir.joinpath('train', 'checkpoints').rglob('*.tar'))
    if len(ckpts) > 0:
        latest = max(ckpts, key=lambda p: p.stat().st_ctime)
        if latest.is_file():
            log.info(f'Loading from checkpoint: {latest}')
            ckpt = torch.load(latest)
        else:
            raise FileNotFoundError(f'No checkpoints found in {outdir}')
    else:
        raise FileNotFoundError(f'No checkpoints found in {outdir}')

    dynamics.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    dynamics.assign_eps({
        'xeps': ckpt['xeps'],
        'veps': ckpt['veps'],
    })

    return dynamics, optimizer, ckpt


def setup(cfg: DictConfig) -> dict:
    accelerator = Accelerator()
    steps = instantiate(cfg.steps)                  # type: Steps
    loss_cfg = instantiate(cfg.loss)                # type: LossConfig
    network_cfg = instantiate(cfg.network)          # type: NetworkConfig
    lr_cfg = instantiate(cfg.learning_rate)         # type: LearningRateConfig
    dynamics_cfg = instantiate(cfg.dynamics)        # type: DynamicsConfig
    net_weights = instantiate(cfg.net_weights)      # type: NetWeights
    schedule = instantiate(cfg.annealing_schedule)  # type: AnnealingSchedule
    schedule.setup(steps)

    try:
        ccfg = instantiate(cfg.get('conv', None))   # type: ConvolutionConfig
    except TypeError:
        ccfg = None  # type: ignore

    xdim = dynamics_cfg.xdim
    xshape = dynamics_cfg.xshape
    lattice = Lattice(tuple(xshape))
    input_spec = InputSpec(xshape=xshape,
                           vnet={'v': [xdim, ], 'x': [xdim, ]},
                           xnet={'v': [xdim, ], 'x': [xdim, 2]})
    net_factory = NetworkFactory(input_spec=input_spec,
                                 net_weights=net_weights,
                                 network_config=network_cfg,
                                 conv_config=ccfg)
    dynamics = Dynamics(config=dynamics_cfg,
                        potential_fn=lattice.action,
                        network_factory=net_factory)
    loss_fn = LatticeLoss(lattice=lattice, loss_config=loss_cfg)
    optimizer = torch.optim.Adam(dynamics.parameters())
    try:
        dynamics, optimizer, ckpt = load_from_ckpt(dynamics, optimizer, cfg)
    except FileNotFoundError:
        pass

    dynamics = dynamics.to(accelerator.device)
    dynamics, optimizer = accelerator.prepare(dynamics, optimizer)
    trainer = Trainer(steps=steps,
                      loss_fn=loss_fn,
                      lr_config=lr_cfg,
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
