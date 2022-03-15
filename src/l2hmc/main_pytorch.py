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
# from accelerate.utils import extract_model_from_parallel
import hydra
from hydra.utils import instantiate
from l2hmc.common import analyze_dataset, save_logs
from l2hmc.configs import (
    AnnealingSchedule,
    ConvolutionConfig,
    DynamicsConfig,
    HERE,
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
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import wandb
from wandb.util import generate_id
from l2hmc import utils


log = logging.getLogger(__name__)


Tensor = torch.Tensor


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
        wbdir: Optional[os.PathLike] = None,
) -> DictConfig:
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

    if wbdir is not None:
        cfg.wandb.setup.update({'dir': Path(wbdir).as_posix()})

    cfg.wandb.setup.update({
        'tags': [
            f'{cfg.framework}',
            f'nlf-{cfg.dynamics.nleapfrog}',
            f'beta_final-{cfg.annealing_schedule.beta_final}',
            f'{cfg.dynamics.xshape[1]}x{cfg.dynamics.xshape[2]}',
        ]
    })

    return cfg


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


def get_jobdir(cfg: DictConfig, job_type: str) -> Path:
    jobdir = Path(cfg.get('outdir', os.getcwd())).joinpath(job_type)
    jobdir.mkdir(exist_ok=True, parents=True)
    assert jobdir is not None
    return jobdir


def eval(
        cfg: DictConfig,
        trainer: Trainer,
        job_type: str,
        run: Optional[Any] = None,
        nchains: Optional[int] = None,
        eps: Tensor = None,
) -> dict:
    """Evaluate model (nested as `trainer.model`)"""
    nchains = -1 if nchains is None else nchains
    therm_frac = cfg.get('therm_frac', 0.2)

    jobdir = get_jobdir(cfg, job_type=job_type)
    writer = get_summary_writer(cfg, trainer, job_type=job_type)
    output = trainer.eval(run=run,
                          writer=writer,
                          nchains=nchains,
                          job_type=job_type,
                          eps=eps)
    dataset = output['history'].get_dataset(therm_frac=therm_frac)
    if trainer.accelerator.is_local_main_process:
        _ = analyze_dataset(dataset,
                            run=run,
                            save=True,
                            outdir=jobdir,
                            nchains=nchains,
                            job_type=job_type,
                            title=f'{job_type}: PyTorch')

        if not is_interactive():
            edir = jobdir.joinpath('logs')
            edir.mkdir(exist_ok=True, parents=True)
            log.info(f'Saving {job_type} logs to: {edir.as_posix()}')
            save_logs(logdir=edir,
                      run=run,
                      job_type=job_type,
                      tables=output['tables'],
                      summaries=output['summaries'])

    if writer is not None:
        writer.close()

    if run is not None:
        dQint = dataset.data_vars.get('dQint').values
        drop = int(0.1 * len(dQint))
        dQint = dQint[drop:]
        run.summary[f'dQint.{job_type}'] = dQint
        run.summary[f'dQint_{job_type}.mean'] = dQint.mean()

    return output


def train(
        cfg: DictConfig,
        trainer: Trainer,
        run: Optional[Any] = None,
        nchains: Optional[int] = None,
) -> dict:
    nchains = -1 if nchains is None else nchains
    jobdir = get_jobdir(cfg, job_type='train')
    writer = get_summary_writer(cfg, trainer, job_type='train')
    output = trainer.train(run=run,
                           writer=writer,
                           train_dir=jobdir)

    if trainer.accelerator.is_local_main_process:
        dset = output['history'].get_dataset()
        _ = analyze_dataset(dset,
                            run=run,
                            save=True,
                            outdir=jobdir,
                            nchains=nchains,
                            job_type='train',
                            title='Training: PyTorch')

        if not is_interactive():
            tdir = jobdir.joinpath('logs')
            tdir.mkdir(exist_ok=True, parents=True)
            log.info(f'Saving train logs to: {tdir.as_posix()}')
            save_logs(logdir=tdir,
                      run=run,
                      job_type='train',
                      rows=output['rows'],
                      tables=output['tables'],
                      summaries=output['summaries'])

    if writer is not None:
        writer.close()

    return output


# @record
def main(cfg: DictConfig) -> dict:
    outputs = {}
    objs = setup(cfg)
    trainer = objs['trainer']  # type: Trainer

    batch_size = cfg.dynamics.xshape[0]
    nchains = max((1, batch_size // 4))
    cfg.update({'nchains': nchains})

    id = generate_id() if trainer.accelerator.is_local_main_process else None
    outdir = Path(cfg.get('outdir', os.getcwd()))
    debug = any([s in outdir.as_posix() for s in ['debug', 'test']])
    cfg = update_wandb_config(cfg, id=id, debug=debug)

    run = None
    if trainer.accelerator.is_local_main_process:
        run = wandb.init(**cfg.wandb.setup)
        wandb.define_metric('dQint_eval', summary='mean')
        run.log_code(HERE)
        cfg_dict = OmegaConf.to_container(cfg,
                                          resolve=True,
                                          throw_on_missing=True)
        run.config.update(cfg_dict)
        run.config.update({'outdir': outdir.as_posix()})
        utils.print_config(cfg, resolve=True)

    # ----------------------------------------------------------
    # 1. Train model
    # 2. Evaluate trained model
    # 3. Run generic HMC as baseline w/ same trajectory length
    # ----------------------------------------------------------
    outputs['train'] = train(cfg, trainer, run=run)     # [1.]
    if trainer.accelerator.is_local_main_process:
        eps = torch.tensor(0.05)
        for job in ['eval', 'hmc']:                     # [2.], [3.]
            log.warning(f'Running {job}')
            nchains = max((1, batch_size // 8))
            if job == 'hmc':
                outputs[job] = eval(cfg=cfg,
                                    run=run,
                                    eps=eps,            # Use fixed step size
                                    job_type=job,
                                    nchains=nchains,
                                    trainer=trainer)
            else:
                outputs[job] = eval(cfg=cfg,
                                    run=run,
                                    job_type=job,
                                    nchains=nchains,
                                    trainer=trainer)
    if run is not None:
        run.finish()

    return outputs


@hydra.main(config_path='./conf', config_name='config')
def launch(cfg: DictConfig) -> None:
    # log.info(f'Working directory: {os.getcwd()}')
    # log.info(OmegaConf.to_yaml(cfg, resolve=True))
    _ = main(cfg)


if __name__ == '__main__':
    launch()
