"""
utils/pytorch.py

Contains utilities for use in PyTorch
"""
from __future__ import absolute_import, division, print_function, annotations
from omegaconf import DictConfig
from pathlib import Path
import os
import logging
from torch.utils.tensorboard.writer import SummaryWriter

from l2hmc.dynamics.pytorch.dynamics import Dynamics
import torch


log = logging.getLogger(__name__)
# from l2hmc import get_logger
# log = get_logger(__name__)


def get_summary_writer(
        cfg: DictConfig,
        job_type: str
):
    """Returns SummaryWriter object for tracking summaries."""
    outdir = Path(cfg.get('outdir', os.getcwd()))
    jobdir = outdir.joinpath(job_type)
    summary_dir = jobdir.joinpath('summaries')
    summary_dir.mkdir(exist_ok=True, parents=True)

    return SummaryWriter(summary_dir.as_posix())


def load_from_ckpt(
        dynamics: Dynamics,
        optimizer: torch.optim.Optimizer,
        cfg: DictConfig,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict]:
    outdir = Path(cfg.get('outdir', os.getcwd()))
    if not (
        ckpts := list(outdir.joinpath('train', 'checkpoints').rglob('*.tar'))
    ):
        raise FileNotFoundError(f'No checkpoints found in {outdir}')

    latest = max(ckpts, key=lambda p: p.stat().st_ctime)
    if not latest.is_file():
        raise FileNotFoundError(f'No checkpoints found in {outdir}')
    log.info(f'Loading from checkpoint: {latest}')
    ckpt = torch.load(latest)
    dynamics.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    dynamics.assign_eps({
        'xeps': ckpt['xeps'],
        'veps': ckpt['veps'],
    })

    return dynamics, optimizer, ckpt


# def update_wandb_config(
#         cfg: DictConfig,
#         tag: Optional[str] = None,
#         debug: Optional[bool] = None,
# ) -> DictConfig:
#     group = [
#         'pytorch',
#         'gpu' if torch.cuda.is_available() else 'cpu',
#         'DDP' if torch.cuda.device_count() > 1 else 'local'
#     ]
#     if debug:
#         group.append('debug')

#     cfg.wandb.setup.update({'group': '/'.join(group)})
#     if tag is not None:
#         cfg.wandb.setup.update({'id': tag})

#     cfg.wandb.setup.update({
#         'tags': [
#             f'{cfg.framework}',
#             f'nlf-{cfg.dynamics.nleapfrog}',
#             f'beta_final-{cfg.annealing_schedule.beta_final}',
#             f'{cfg.dynamics.latvolume[0]}x{cfg.dynamics.latvolume[1]}',
#             f'{cfg.dynamics.group}',
#         ]
#     })

#     return cfg
