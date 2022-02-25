"""
main_pytorch.py

Contains entry-point for training and inference.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from pathlib import Path

from accelerate import Accelerator
# from accelerate.utils import extract_model_from_parallel
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
# from torchinfo import summary as model_summary
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


# @record
def train(cfg: DictConfig) -> dict:
    objs = setup(cfg)
    trainer = objs['trainer']  # type: Trainer
    accelerator = objs['accelerator']  # type: Accelerator
    # dynamics = extract_model_from_parallel(objs['dynamics'])

    outdir = Path(cfg.get('outdir', os.getcwd()))
    train_dir = outdir.joinpath('train')
    train_dir.mkdir(exist_ok=True, parents=True)
    train_summary_dir = train_dir.joinpath('summaries')
    train_summary_dir.mkdir(exist_ok=True, parents=True)
    # state = objs['dynamics'].random_state(1.0)
    # x = torch.tensor(state.x, requires_grad=True)
    # model_summary(dynamics, depth=5)  # , input_data=[(x, state.beta)])

    id = None
    group = None
    train_run = None
    train_writer = None

    schedule = objs['schedule']  # type: AnnealingSchedule
    beta_init = schedule.beta_init
    beta_final = schedule.beta_final
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    nchains = min((cfg.dynamics.xshape[0], cfg.dynamics.nleapfrog))
    width = max((150, int(cfg.get('width', os.environ.get('COLUMNS', 150)))))
    if accelerator.is_local_main_process:
        id = generate_id()
        gnames = ['pytorch']
        # wandb.tensorboard.patch(root_logdir=outdir.as_posix())
        wcuda = torch.cuda.is_available()
        gnames.append('gpu') if wcuda else gnames.append('cpu')
        if torch.cuda.device_count() > 1:
            gnames.append('DDP')

        if 'debug' in train_dir.as_posix():
            gnames.append('debug')

        group = '/'.join(gnames)
        train_run = wandb.init(id=id,
                               group=group,
                               resume='allow',
                               config=wandb_cfg,
                               # magic=True,
                               sync_tensorboard=True,
                               # pytorch=True,
                               job_type='train',
                               entity=cfg.wandb.setup.entity,
                               project=cfg.wandb.setup.project,
                               tags=[f'beta_init={beta_init:1.2f}',
                                     f'beta_final={beta_final:1.2f}'],
                               settings=wandb.Settings(start_method='thread'))
        train_writer = SummaryWriter(train_summary_dir.as_posix())
        # run.watch(objs['dynamics'], objs['loss_fn'], log='all')

    train_output = trainer.train(run=train_run,
                                 train_dir=train_dir,
                                 writer=train_writer,
                                 width=width)
    output = {
        'setup': objs,
        'outdir': outdir,
        'train': train_output,
    }
    if accelerator.is_local_main_process:
        train_dset = train_output['history'].get_dataset()
        _ = analyze_dataset(train_dset,
                            prefix='train',
                            nchains=nchains,
                            outdir=train_dir,
                            # xarr=train_output['xarr'],
                            title='Training: PyTorch')
        if not is_interactive():
            tdir = train_dir.joinpath('logs')
            tdir.mkdir(exist_ok=True, parents=True)
            log.info(f'Saving train logs to: {tdir.as_posix()}')
            save_logs(logdir=tdir,
                      tables=train_output['tables'],
                      summaries=train_output['summaries'])

    if train_writer is not None:
        train_writer.close()

    if train_run is not None:
        train_run.finish()

    if accelerator.is_local_main_process:
        therm_frac = cfg.get('therm_frac', 0.2)
        eval_dir = outdir.joinpath('eval')
        eval_summary_dir = eval_dir.joinpath('summaries')
        eval_summary_dir.mkdir(exist_ok=True, parents=True)

        eval_run = wandb.init(id=id,
                              group=group,
                              resume='allow',
                              config=wandb_cfg,
                              # magic=True,
                              sync_tensorboard=True,
                              # pytorch=True,
                              job_type='eval',
                              tags=[f'beta={beta_final:1.2f}'],
                              entity=cfg.wandb.setup.entity,
                              project=cfg.wandb.setup.project,
                              settings=wandb.Settings(start_method='thread'))

        eval_writer = SummaryWriter(eval_summary_dir.as_posix())

        eval_output = trainer.eval(run=eval_run,
                                   width=width,
                                   writer=eval_writer)

        eval_dset = eval_output['history'].get_dataset(therm_frac=therm_frac)
        _ = analyze_dataset(eval_dset,
                            prefix='eval',
                            nchains=nchains,
                            outdir=eval_dir,
                            # lattice=objs['lattice'],
                            # xarr=eval_output['xarr'],
                            title='Evaluating: PyTorch')

        if not is_interactive():
            edir = eval_dir.joinpath('logs')
            edir.mkdir(exist_ok=True, parents=True)
            log.info(f'Saving eval logs to: {edir.as_posix()}')
            save_logs(logdir=edir,
                      tables=eval_output['tables'],
                      summaries=eval_output['summaries'])

        eval_run.finish()
        eval_writer.close()

        # erun.finish()
        output.update({'eval': eval_output})

    return output


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    log.info(f'Working directory: {os.getcwd()}')
    log.info(OmegaConf.to_yaml(cfg))
    _ = train(cfg)


if __name__ == '__main__':
    main()
