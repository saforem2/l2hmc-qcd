"""
main_pytorch.py

Contains entry-point for training and inference.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from pathlib import Path
from typing import Any, Optional

from accelerate import Accelerator
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
# from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard.writer import SummaryWriter

from l2hmc.common import analyze_dataset, save_logs
from l2hmc.configs import (
    AnnealingSchedule,
    ConvolutionConfig,
    DynamicsConfig,
    InputSpec,
    LearningRateConfig,
    LossConfig,
    NetWeights,
    NetworkConfig,
    Steps,
)
from l2hmc.dynamics.pytorch.dynamics import Dynamics
from l2hmc.lattice.pytorch.lattice import Lattice
from l2hmc.loss.pytorch.loss import LatticeLoss
from l2hmc.network.pytorch.network import NetworkFactory
from l2hmc.trainers.pytorch.trainer import Trainer
from l2hmc.utils.console import is_interactive
import wandb
from wandb.util import generate_id


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
        dynamics, optimizer, _ = load_from_ckpt(dynamics, optimizer, cfg)
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
        'schedule': schedule,
        'optimizer': optimizer,
        'accelerator': accelerator,
    }


def update_wandb_config(
        cfg: DictConfig,
        id: Optional[str] = None,
        debug: Optional[bool] = None,
        # job_type: Optional[str] = None,
        wbconfig: Optional[dict[Any, Any] | list[Any] | str] = None,
        wbdir: Optional[os.PathLike] = None,
) -> None:
    group = [
        'pytorch',
        'gpu' if torch.cuda.is_available() else 'cpu',
        'DDP' if torch.cuda.device_count() > 1 else 'local'
    ]
    if debug:
        group.append('debug')

    cfg.wandb.setup.update({'group': '/'.join(group)})
    if id is not None:
        cfg.wandb.setup.update({'id': id})

    # if job_type is not None:
    #     cfg.wandb.setup.update({'job_type': job_type})

    if wbdir is not None:
        cfg.wandb.setup.update({'dir': Path(wbdir).as_posix()})

    if wbconfig is not None:
        cfg.wandb.setup.update({'config': wbconfig})


def get_summary_writer(
        cfg: DictConfig,
        trainer: Trainer,
        job_type: str
):
    """Returns SummaryWriter object for tracking summaries."""
    outdir = Path(cfg.get('outdir', os.getcwd()))
    jobdir = outdir.joinpath(job_type)
    summary_dir = jobdir.joinpath('summaries')
    summary_dir.mkdir(exist_ok=True, parents=True)

    writer = None
    if trainer.accelerator.is_local_main_process:
        writer = SummaryWriter(summary_dir.as_posix())

    return writer


def eval(
        cfg: DictConfig,
        trainer: Trainer,
        job_type: str,
        run: Optional[Any] = None,
        # writer: Optional[Any] = None,
        jobdir: Optional[os.PathLike] = None,
) -> dict:
    """Evaluate model (nested as `trainer.model`)"""
    nchains = cfg.get('nchains', -1)
    therm_frac = cfg.get('therm_frac', 0.2)
    if jobdir is None:
        jobdir = Path(cfg.get('outdir', os.getcwd())).joinpath(job_type)

    assert jobdir is not None
    jobdir = Path(jobdir)
    writer = get_summary_writer(cfg, trainer, job_type=job_type)
    eval_output = trainer.eval(run=run,
                               writer=writer,
                               job_type=job_type,
                               width=cfg.get('width', None))

    eval_dset = eval_output['history'].get_dataset(therm_frac=therm_frac)
    _ = analyze_dataset(eval_dset,
                        nchains=nchains,
                        outdir=jobdir,
                        title=f'{job_type}: PyTorch')

    if not is_interactive():
        edir = jobdir.joinpath('logs')
        edir.mkdir(exist_ok=True, parents=True)
        log.info(f'Saving {job_type} logs to: {edir.as_posix()}')
        save_logs(logdir=edir,
                  run=run,
                  job_type=job_type,
                  tables=eval_output['tables'],
                  summaries=eval_output['summaries'])

    if writer is not None:
        writer.close()

    return eval_output


def train(
        cfg: DictConfig,
        trainer: Trainer,
        run: Optional[Any] = None,
        # writer: Optional[SummaryWriter] = None,
        jobdir: Optional[os.PathLike] = None,
) -> dict:
    if jobdir is None:
        jobdir = Path(cfg.get('outdir', os.getcwd())).joinpath('train')
    assert jobdir is not None
    jobdir = Path(jobdir)
    writer = get_summary_writer(cfg, trainer, job_type='train')
    train_output = trainer.train(run=run,
                                 writer=writer,
                                 train_dir=jobdir,
                                 width=cfg.get('width', None))
    if trainer.accelerator.is_local_main_process:
        train_dset = train_output['history'].get_dataset()
        _ = analyze_dataset(train_dset,
                            outdir=jobdir,
                            prefix='train',
                            title='Training: PyTorch',
                            nchains=cfg.get('nchains', None))

        if not is_interactive():
            tdir = jobdir.joinpath('logs')
            tdir.mkdir(exist_ok=True, parents=True)
            log.info(f'Saving train logs to: {tdir.as_posix()}')
            save_logs(logdir=tdir,
                      tables=train_output['tables'],
                      summaries=train_output['summaries'])

    if writer is not None:
        writer.close()

    return train_output


# @record
def main(cfg: DictConfig) -> dict:
    outputs = {}
    objs = setup(cfg)
    trainer = objs['trainer']  # type: Trainer

    nchains = min((cfg.dynamics.xshape[0], cfg.dynamics.nleapfrog))
    width = max((150, int(cfg.get('width', os.environ.get('COLUMNS', 150)))))
    cfg.update({'width': width, 'nchains': nchains})

    id = generate_id() if trainer.accelerator.is_local_main_process else None
    wbconfig = OmegaConf.to_container(cfg, resolve=True)
    outdir = Path(cfg.get('outdir', os.getcwd()))
    debug = any([s in outdir.as_posix() for s in ['debug', 'test']])
    update_wandb_config(cfg,
                        id=id,
                        debug=debug,
                        # job_type='train',
                        wbconfig=wbconfig)

    run = None
    if trainer.accelerator.is_local_main_process:
        run = wandb.init(**cfg.wandb.setup)

    # Train model
    outputs['train'] = train(cfg, trainer, run=run)
    if trainer.accelerator.is_local_main_process:
        # Evaluate trained model following training and update 'job_type''
        # cfg.wandb.setup.update({
        #     # 'job_type': 'eval',
        #     'tags': [f'beta={cfg.annealing_schedule.beta_final:1.2f}'],
        # })
        jobs = ['eval', 'hmc']
        for job in jobs:
            log.warning(f'Running {job}')
            outputs[job] = eval(cfg=cfg,
                                run=run,
                                job_type=job,
                                trainer=trainer)

    return outputs


@hydra.main(config_path='./conf', config_name='config')
def launch(cfg: DictConfig) -> None:
    log.info(f'Working directory: {os.getcwd()}')
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    _ = main(cfg)


if __name__ == '__main__':
    launch()
