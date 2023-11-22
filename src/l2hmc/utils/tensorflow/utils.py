"""
tensorflow.py

Contains various utilities for use with TensorFlow
"""
from __future__ import absolute_import, annotations, division, print_function
import os
from pathlib import Path
import logging
from typing import Any, Optional

from omegaconf import DictConfig
import tensorflow as tf

from l2hmc import get_logger
from l2hmc.common import analyze_dataset, save_logs
from l2hmc.configs import get_jobdir
from l2hmc.trainers.tensorflow.trainer import Trainer
from l2hmc.utils.rich import is_interactive


TensorLike = tf.types.experimental.TensorLike


log = logging.getLogger(__name__)
# log = get_logger(__name__)


def get_summary_writer(cfg: DictConfig, job_type: str):
    """Returns SummaryWriter object for tracking summaries."""
    outdir = Path(cfg.get('outdir', os.getcwd()))
    jobdir = outdir.joinpath(job_type)
    sdir = jobdir.joinpath('summaries')
    sdir.mkdir(exist_ok=True, parents=True)

    return tf.summary.create_file_writer(sdir.as_posix())


def evaluate(
        cfg: DictConfig,
        trainer: Trainer,
        job_type: str,
        run: Optional[Any] = None,
        nchains: Optional[int] = 10,
        eps: Optional[TensorLike] = None,
) -> dict:
    assert isinstance(nchains, int)
    assert job_type in {'eval', 'hmc'}
    therm_frac = cfg.get('therm_frac', 0.2)
    jobdir = get_jobdir(cfg, job_type=job_type)
    writer = get_summary_writer(cfg, job_type=job_type)
    if writer is not None:
        writer.set_as_default()

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
                        title=f'{job_type}: TensorFlow')
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
        **kwargs,
) -> dict:
    nchains = 16 if nchains is None else nchains
    jobdir = get_jobdir(cfg, job_type='train')
    writer = get_summary_writer(cfg, job_type='train')
    if writer is not None:
        writer.set_as_default()

    output = trainer.train(run=run,
                           writer=writer,
                           train_dir=jobdir,
                           **kwargs)
    if RANK == 0:
        dset = output['history'].get_dataset()
        _ = analyze_dataset(dset,
                            run=run,
                            save=True,
                            outdir=jobdir,
                            nchains=nchains,
                            job_type='train',
                            title='Training: TensorFlow')
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
        writer.close()  # type: ignore

    return output


# def update_wandb_config(
#         cfg: DictConfig,
#         tag: Optional[str] = None,
#         debug: Optional[bool] = None,
#         # job_type: Optional[str] = None,
# ) -> DictConfig:
#     """Updates config using runtime information for W&B."""
#     framework = 'tensorflow'
#     size = 'horovod' if SIZE > 1 else 'local'
#     device = (
#         'gpu' if len(tf.config.list_physical_devices('GPU')) > 0
#         else 'cpu'
#     )
#     group = [framework, device, size]
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
#             f'{cfg.dynamics.group}'
#         ]
#     })

#     return cfg
