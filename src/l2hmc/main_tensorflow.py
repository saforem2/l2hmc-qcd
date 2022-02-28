"""
main_tensorflow.py

Main entry-point for training L2HMC Dynamics w/ TensorFlow
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from pathlib import Path
from typing import Any, Optional
import rich

import tensorflow as tf
import horovod.tensorflow as hvd

import wandb
from wandb.util import generate_id

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from l2hmc.common import analyze_dataset, save_logs
from l2hmc.configs import InputSpec
from l2hmc.dynamics.tensorflow.dynamics import Dynamics
from l2hmc.lattice.tensorflow.lattice import Lattice
from l2hmc.loss.tensorflow.loss import LatticeLoss
from l2hmc.network.tensorflow.network import NetworkFactory
from l2hmc.trainers.tensorflow.trainer import Trainer
from l2hmc.utils.console import is_interactive


RANK = hvd.rank()
SIZE = hvd.size()

log = logging.getLogger(__name__)


def setup(cfg: DictConfig) -> dict:
    steps = instantiate(cfg.steps)
    loss_cfg = instantiate(cfg.loss)
    network_cfg = instantiate(cfg.network)
    lr_cfg = instantiate(cfg.learning_rate)
    dynamics_cfg = instantiate(cfg.dynamics)
    net_weights = instantiate(cfg.net_weights)
    schedule = instantiate(cfg.annealing_schedule)
    schedule.setup(steps)

    try:
        conv_cfg = instantiate(cfg.get('conv', None))
    except TypeError:
        conv_cfg = None

    xdim = dynamics_cfg.xdim
    xshape = dynamics_cfg.xshape
    lattice = Lattice(tuple(xshape))
    input_spec = InputSpec(xshape=xshape,
                           vnet={'v': [xdim, ], 'x': [xdim, ]},
                           xnet={'v': [xdim, ], 'x': [xdim, 2]})
    net_factory = NetworkFactory(input_spec=input_spec,
                                 net_weights=net_weights,
                                 network_config=network_cfg,
                                 conv_config=conv_cfg)
    dynamics = Dynamics(config=dynamics_cfg,
                        potential_fn=lattice.action,
                        network_factory=net_factory)
    loss_fn = LatticeLoss(lattice=lattice, loss_config=loss_cfg)
    optimizer = tf.keras.optimizers.Adam(cfg.learning_rate.lr_init)
    trainer = Trainer(steps=steps,
                      rank=RANK,
                      loss_fn=loss_fn,
                      lr_config=lr_cfg,
                      schedule=schedule,
                      dynamics=dynamics,
                      optimizer=optimizer,
                      aux_weight=loss_cfg.aux_weight)
    return {
        'lattice': lattice,
        'loss_fn': loss_fn,
        'schedule': schedule,
        'dynamics': dynamics,
        'trainer': trainer,
        'optimizer': optimizer,
        'rank': RANK,
    }


def update_wandb_config(
        cfg: DictConfig,
        id: Optional[str] = None,
        debug: Optional[bool] = None,
        # job_type: Optional[str] = None,
        wbconfig: Optional[dict | list | str] = None,
        wbdir: Optional[os.PathLike] = None,
) -> None:
    """Updates config using runtime information for W&B."""
    group = ['tensorflow', 'horovod' if SIZE > 1 else 'local']
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
) -> dict:
    """Returns SummaryWriter object for tracking summaries."""
    outdir = Path(cfg.get('outdir', os.getcwd()))
    jobdir = outdir.joinpath(job_type)
    sdir = jobdir.joinpath('summaries')
    sdir.mkdir(exist_ok=True, parents=True)

    writer = None
    if trainer.rank == 0:
        writer = tf.summary.create_file_writer(sdir.as_posix())  # type: ignore
        writer.set_as_default()

    return {'dir': jobdir, 'writer': writer}


def eval(
        cfg: DictConfig,
        trainer: Trainer,
        job_type: str,
        run: Optional[Any] = None,
) -> dict:
    therm_frac = cfg.get('therm_frac', 0.2)
    nchains = cfg.get('nchains', -1)
    # job_type = cfg.wandb.setup.get('job_type', 'eval')
    # objs = _setup(cfg, trainer, job_type=job_type)
    objs = get_summary_writer(cfg, trainer, job_type=job_type)

    eval_output = trainer.eval(run=run,
                               writer=objs['writer'],
                               hmc=(job_type == 'hmc'),
                               width=cfg.get('width', None))
    eval_dset = eval_output['history'].get_dataset(therm_frac=therm_frac)
    _ = analyze_dataset(eval_dset,
                        nchains=nchains,
                        outdir=objs['dir'],
                        title=f'{job_type}: PyTorch')
    if not is_interactive():
        edir = objs['dir'].joinpath('logs')
        edir.mkdir(exist_ok=True, parents=True)
        log.info(f'Saving {job_type} logs to: {edir.as_posix()}')
        save_logs(logdir=edir,
                  tables=eval_output['tables'],
                  summaries=eval_output['summaries'])

    if objs['writer'] is not None:
        objs['writer'].close()

    return eval_output


def train(
        cfg: DictConfig,
        trainer: Trainer,
        run: Optional[Any] = None,
        **kwargs,
) -> dict:
    objs = get_summary_writer(cfg, trainer, job_type='train')
    # objs = _setup(cfg, trainer, job_type='train')
    train_output = trainer.train(run=run,
                                 writer=objs['writer'],
                                 train_dir=objs['dir'],
                                 width=cfg.get('width', None),
                                 **kwargs)
    if trainer.rank == 0:
        dset = train_output['history'].get_dataset()
        _ = analyze_dataset(dset,
                            outdir=objs['dir'],
                            prefix='train',
                            title='Training: TensorFlow',
                            nchains=cfg.get('nchains', -1))
        if not is_interactive():
            tdir = objs['dir'].joinpath('logs')
            tdir.mkdir(exist_ok=True, parents=True)
            log.info(f'Saving train logs to: {tdir.as_posix()}')
            save_logs(logdir=tdir,
                      tables=train_output['tables'],
                      summaries=train_output['summaries'])

    if objs['writer'] is not None:
        objs['writer'].close()

    return train_output


def main(cfg: DictConfig) -> dict:
    objs = setup(cfg)
    trainer = objs['trainer']  # type: Trainer
    nchains = min((cfg.dynamics.xshape[0], cfg.dynamics.nleapfrog))
    width = max((150, int(cfg.get('width', os.environ.get('COLUMNS', 150)))))
    cfg.update({'width': width, 'nchains': nchains})
    id = generate_id() if trainer.rank == 0 else None
    wbconfig = OmegaConf.to_container(cfg, resolve=True)
    outdir = Path(cfg.get('outdir', os.getcwd()))
    debug = any([s in outdir.as_posix() for s in ['debug', 'test']])
    update_wandb_config(cfg,
                        id=id,
                        debug=debug,
                        # job_type='train',
                        wbconfig=wbconfig)
    run = None
    if trainer.rank == 0:
        run = wandb.init(**cfg.wandb.setup)

    kwargs = {
        'compile': cfg.get('compile', True),
        'jit_compile': cfg.get('jit_compile', False),
    }
    train_output = train(cfg, trainer, run=run, **kwargs)
    hmc_output = None
    eval_output = None

    if trainer.rank == 0:
        cfg.wandb.setup.update({
            # 'job_type': 'eval',
            'tags': [f'beta={cfg.annealing_schedule.beta_final:1.2f}'],
        })
        log.warning('Evaluating trained model')
        eval_output = eval(cfg=cfg, trainer=trainer, job_type='eval', run=run)

        # cfg.wandb.setup.update({'job_type': 'hmc'})
        log.warning('Running generic HMC for comparison')
        hmc_output = eval(cfg=cfg, trainer=trainer, job_type='hmc', run=run)

    return {
        'train': train_output,
        'eval': eval_output,
        'hmc': hmc_output,
    }


@hydra.main(config_path='./conf', config_name='config')
def launch(cfg: DictConfig) -> None:
    log.info(f'Working directory: {os.getcwd()}')
    # rich.print(OmegaConf.to_container(cfg, resolve=True))
    # log.info(OmegaConf.to_yaml(cfg))
    _ = main(cfg)


if __name__ == '__main__':
    launch()
