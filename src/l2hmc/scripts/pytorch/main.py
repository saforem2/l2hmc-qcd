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
    HERE,
    Steps,
    InputSpec,
    get_jobdir,
    LossConfig,
    NetWeights,
    NetworkConfig,
    DynamicsConfig,
    AnnealingSchedule,
    ConvolutionConfig,
    LearningRateConfig,
)
from l2hmc.dynamics.pytorch.dynamics import Dynamics
# from l2hmc.lattice.pytorch.lattice import Lattice
from l2hmc.lattice.u1.pytorch.lattice import LatticeU1
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

    lattice = None
    xdim = dynamics_cfg.xdim
    group = dynamics_cfg.group
    xshape = dynamics_cfg.xshape
    latvolume = dynamics_cfg.latvolume
    log.warning(f'xdim: {dynamics_cfg.xdim}')
    log.warning(f'group: {dynamics_cfg.group}')
    log.warning(f'xshape: {dynamics_cfg.xshape}')
    log.warning(f'latvolume: {dynamics_cfg.latvolume}')
    if group == 'U1':
        lattice = LatticeU1(dynamics_cfg.nchains, tuple(latvolume))
    elif group == 'SU3':
        log.error('LatticeSU3 not implemented for pytorch!! (yet)')
        # lattice = LatticeSU3(dynamics_cfg.nchains, tuple(latvolume), c1=c1)
    else:
        log.info(dynamics_cfg)
        raise ValueError('Unexpected value encountered in `dynamics.group`')
    assert lattice is not None
    input_spec = InputSpec(xshape=list(xshape),
                           vnet={'v': [xdim, ], 'x': [xdim, ]},
                           xnet={'v': [xdim, ], 'x': [xdim, 2]})
    net_factory = NetworkFactory(input_spec=input_spec,
                                 net_weights=net_weights,
                                 network_config=network_cfg,
                                 conv_config=ccfg)
    dynamics = Dynamics(config=dynamics_cfg,
                        potential_fn=lattice.action,
                        network_factory=net_factory)
    optimizer = torch.optim.Adam(dynamics.parameters())
    loss_fn = LatticeLoss(lattice=lattice, loss_config=loss_cfg)
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
        tag: Optional[str] = None,
        debug: Optional[bool] = None,
) -> DictConfig:
    group = [
        'pytorch',
        'gpu' if torch.cuda.is_available() else 'cpu',
        'DDP' if torch.cuda.device_count() > 1 else 'local'
    ]
    if debug:
        group.append('debug')

    cfg.wandb.setup.update({'group': '/'.join(group)})
    if tag is not None:
        cfg.wandb.setup.update({'id': tag})

    cfg.wandb.setup.update({
        'tags': [
            f'{cfg.framework}',
            f'nlf-{cfg.dynamics.nleapfrog}',
            f'beta_final-{cfg.annealing_schedule.beta_final}',
            f'{cfg.dynamics.latvolume[0]}x{cfg.dynamics.latvolume[1]}',
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


# -------------------------------------------------------------
# TODO: Gather and separate duplicate file I/O related methods
# -------------------------------------------------------------


def evaluate(
        cfg: DictConfig,
        trainer: Trainer,
        job_type: str,
        run: Optional[Any] = None,
        nchains: Optional[int] = None,
        eps: Optional[Tensor] = None,
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
    if run is not None:
        dQint = dataset.data_vars.get('dQint').values
        drop = int(0.1 * len(dQint))
        dQint = dQint[drop:]
        run.summary[f'dQint_{job_type}'] = dQint
        run.summary[f'dQint_{job_type}.mean'] = dQint.mean()

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
        save_logs(run=run,
                  logdir=edir,
                  job_type=job_type,
                  tables=output['tables'],
                  summaries=output['summaries'])

    if writer is not None:
        writer.close()

    return output


def train(
        cfg: DictConfig,
        trainer: Trainer,
        run: Optional[Any] = None,
        nchains: Optional[int] = None,
) -> dict:
    nchains = 16 if nchains is None else nchains
    jobdir = get_jobdir(cfg, job_type='train')
    writer = get_summary_writer(cfg, trainer, job_type='train')

    # ------------------------------------------
    # NOTE: cfg.profile will be False by default
    # ------------------------------------------
    if cfg.profile:
        from torch.profiler import profile, ProfilerActivity  # type: ignore
        activities = [ProfilerActivity.CUDA, ProfilerActivity.CPU]
        with profile(record_shapes=True, activities=activities) as prof:
            assert cfg.steps.nepoch * cfg.steps.nera < 100
            output = trainer.train(run=run, writer=writer, train_dir=jobdir)

        log.info(prof.key_averages().table(sort_by="cpu_time_total"))
        tracefile = Path(os.getcwd()).joinpath('trace.json').as_posix()
        prof.export_chrome_trace(tracefile)

    else:
        output = trainer.train(run=run,
                               writer=writer,
                               train_dir=jobdir)

    if trainer.accelerator.is_local_main_process:
        dset = output['history'].get_dataset()
        jobdir.mkdir(exist_ok=True, parents=True)
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
            save_logs(run=run,
                      logdir=tdir,
                      job_type='train',
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

    nchains = max((1, cfg.dynamics.nchains // 4))
    cfg.update({'nchains': nchains})

    tag = generate_id() if trainer.accelerator.is_local_main_process else None
    outdir = Path(cfg.get('outdir', os.getcwd()))
    debug = any([s in outdir.as_posix() for s in ['debug', 'test']])
    cfg = update_wandb_config(cfg, tag=tag, debug=debug)

    run = None
    if trainer.accelerator.is_local_main_process:
        run = wandb.init(**cfg.wandb.setup)
        wandb.define_metric('dQint_eval', summary='mean')
        assert run is not None and run is wandb.run
        run.log_code(HERE.as_posix())
        cfg_dict = OmegaConf.to_container(cfg,
                                          resolve=True,
                                          throw_on_missing=True)
        run.config.update(cfg_dict)
        # run.config.update({'outdir': outdir.as_posix()})
        utils.print_config(cfg, resolve=True)

    # ----------------------------------------------------------
    # 1. Train model
    # 2. Evaluate trained model
    # 3. Run generic HMC as baseline w/ same trajectory length
    # ----------------------------------------------------------
    should_train = (cfg.steps.nera > 0 and cfg.steps.nepoch > 0)
    if should_train:
        outputs['train'] = train(cfg, trainer, run=run)  # [1.]

    if trainer.accelerator.is_local_main_process:
        # batch_size = cfg.dynamics.xshape[0]
        nchains = max((4, cfg.dynamics.nchains // 8))
        if should_train and cfg.steps.test > 0:
            log.warning('Evaluating trained model')
            outputs['eval'] = evaluate(cfg,
                                       run=run,
                                       job_type='eval',
                                       nchains=nchains,
                                       trainer=trainer)
        if cfg.steps.test > 0:
            log.warning('Running generic HMC')
            eps = torch.tensor(0.05)
            outputs['hmc'] = evaluate(cfg=cfg,
                                      run=run,
                                      eps=eps,
                                      job_type='hmc',
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
