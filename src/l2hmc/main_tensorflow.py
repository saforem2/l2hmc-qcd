"""
main_tensorflow.py

Main entry-point for training L2HMC Dynamics w/ TensorFlow
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from pathlib import Path
from typing import Any, Optional

import tensorflow as tf
import horovod.tensorflow as hvd

import wandb
from wandb.util import generate_id

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from l2hmc.common import analyze_dataset, save_logs
from l2hmc.configs import InputSpec, HERE
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
    # clipnorm = cfg.get('clipnorm', 10.0)

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
        # wbdir: Optional[os.PathLike] = None,
) -> DictConfig:
    """Updates config using runtime information for W&B."""
    group = [
        'tensorflow',
        'gpu' if len(tf.config.list_physical_devices('GPU')) > 0 else 'cpu'
        'horovod' if SIZE > 1 else 'local'
    ]
    if debug:
        group.append('debug')

    cfg.wandb.setup.update({'group': '/'.join(group)})
    if id is not None:
        cfg.wandb.setup.update({'id': id})

    cfg.wandb.setup.update({
        'tags': [
            f'{cfg.framework}',
            f'nlf-{cfg.dynamics.nleapfrog}',
            f'beta_final-{cfg.annealing_schedule.beta_final}',
            f'{cfg.dynamics.xshape[1]}x{cfg.dynamics.xshape[2]}',
        ]
    })

    return cfg


def get_summary_writer(cfg: DictConfig, job_type: str):
    """Returns SummaryWriter object for tracking summaries."""
    outdir = Path(cfg.get('outdir', os.getcwd()))
    jobdir = outdir.joinpath(job_type)
    sdir = jobdir.joinpath('summaries')
    sdir.mkdir(exist_ok=True, parents=True)

    writer = None
    if RANK == 0:
        writer = tf.summary.create_file_writer(sdir.as_posix())  # type: ignore

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
) -> dict:
    nchains = cfg.get('nchains', -1)
    therm_frac = cfg.get('therm_frac', 0.2)
    jobdir = get_jobdir(cfg, job_type=job_type)
    writer = get_summary_writer(cfg, job_type=job_type)
    if writer is not None:
        writer.set_as_default()

    eval_output = trainer.eval(run=run,
                               writer=writer,
                               job_type=job_type,
                               width=cfg.get('width', None))
    eval_dset = eval_output['history'].get_dataset(therm_frac=therm_frac)
    _ = analyze_dataset(eval_dset,
                        outdir=jobdir,
                        nchains=nchains,
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
        writer.close()  # type: ignore

    return eval_output


def train(
        cfg: DictConfig,
        trainer: Trainer,
        run: Optional[Any] = None,
        **kwargs,
) -> dict:
    jobdir = get_jobdir(cfg, job_type='train')
    writer = get_summary_writer(cfg, job_type='train')
    width = int(cfg.get('width', os.environ.get('COLUMNS', 150)))
    if writer is not None:
        writer.set_as_default()

    output = trainer.train(run=run,
                           writer=writer,
                           train_dir=jobdir,
                           width=width,
                           **kwargs)
    if RANK == 0:
        dset = output['history'].get_dataset()
        _ = analyze_dataset(dset,
                            outdir=jobdir,
                            job_type='train',
                            title='Training: TensorFlow',
                            nchains=cfg.get('nchains', -1))
        if not is_interactive():
            tdir = jobdir.joinpath('logs')
            tdir.mkdir(exist_ok=True, parents=True)
            log.info(f'Saving train logs to: {tdir.as_posix()}')
            save_logs(logdir=tdir,
                      run=run,
                      job_type='train',
                      tables=output['tables'],
                      summaries=output['summaries'])

    if writer is not None:
        writer.close()  # type: ignore

    return output


def main(cfg: DictConfig) -> dict:
    outputs = {}
    objs = setup(cfg)
    trainer = objs['trainer']  # type: Trainer

    nchains = min((cfg.dynamics.xshape[0], cfg.dynamics.nleapfrog))
    width = max((150, int(cfg.get('width', os.environ.get('COLUMNS', 150)))))
    cfg.update({'width': width, 'nchains': nchains})

    id = generate_id() if trainer.rank == 0 else None
    outdir = Path(cfg.get('outdir', os.getcwd()))
    debug = any([s in outdir.as_posix() for s in ['debug', 'test']])
    cfg = update_wandb_config(cfg, id=id, debug=debug)

    run = None
    if RANK == 0:
        run = wandb.init(**cfg.wandb.setup)
        run.log_code(HERE)
        run.config.update(OmegaConf.to_container(cfg,
                                                 resolve=True,
                                                 throw_on_missing=True))
    # ----------------------------------------------------------
    # 1. Train model
    # 2. Evaluate trained model
    # 3. Run generic HMC as baseline w/ same trajectory length
    # ----------------------------------------------------------
    outputs['train'] = train(cfg, trainer, run=run)      # [1.]

    if RANK == 0:
        if run is not None:
            run.config.update({'eval_beta': cfg.annealing_schedule.beta_final})
        for job in ['eval', 'hmc']:                      # [2.], [3.]
            log.warning(f'Running {job}')
            outputs[job] = eval(cfg=cfg,
                                run=run,
                                job_type=job,
                                trainer=trainer)
    if run is not None:
        run.save()

    return outputs


@hydra.main(config_path='./conf', config_name='config')
def launch(cfg: DictConfig) -> None:
    # log.info(f'Working directory: {os.getcwd()}')
    _ = main(cfg)


if __name__ == '__main__':
    launch()
