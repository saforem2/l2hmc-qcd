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
try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True

except (ImportError, ModuleNotFoundError):
    HAS_HOROVOD = False

import wandb
from wandb.util import generate_id

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from l2hmc.common import analyze_dataset, save_logs
from l2hmc.configs import InputSpec, HERE, get_jobdir
from l2hmc.dynamics.tensorflow.dynamics import Dynamics
from l2hmc.loss.tensorflow.loss import LatticeLoss
from l2hmc.network.tensorflow.network import NetworkFactory
from l2hmc.trainers.tensorflow.trainer import Trainer
from l2hmc.utils.console import is_interactive
from l2hmc import utils

from l2hmc.lattice.u1.tensorflow.lattice import LatticeU1
from l2hmc.lattice.su3.tensorflow.lattice import LatticeSU3
# from l2hmc.lattice.tensorflow.lattice import LatticeU1
# from lgt.lattice.u1.tensorflow.lattice import LatticeU1
# from lgt.lattice.su3.tensorflow.lattice import LatticeSU3
# from l2hmc.lattice.tensorflow.lattice import Lattice
# from l2hmc.lattice.su3.tensorflow.lattice import LatticeSU3
# from lgt.lattice.u1.tensorflow.lattice import LatticeU1
# from lgt.lattice.su3.tensorflow.lattice import LatticeSU3

log = logging.getLogger(__name__)

RANK = hvd.rank() if HAS_HOROVOD else 0
SIZE = hvd.size() if HAS_HOROVOD else 1

Tensor = tf.Tensor


def load_from_ckpt(
        # dynamics: Dynamics,
        # optimizer: Optimizer,
        # cfg: DictConfig,
):
    pass


def setup(cfg: DictConfig, c1: float = 0.) -> dict:
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
        lattice = LatticeSU3(dynamics_cfg.nchains, tuple(latvolume), c1=c1)
    else:
        log.info(dynamics_cfg)
        raise ValueError('Unexpected value encountered in `dynamics.group`')

    assert lattice is not None
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
        tag: Optional[str] = None,
        debug: Optional[bool] = None,
        # job_type: Optional[str] = None,
) -> DictConfig:
    """Updates config using runtime information for W&B."""
    framework = 'tensorflow'
    size = 'horovod' if SIZE > 1 else 'local'
    device = (
        'gpu' if len(tf.config.list_physical_devices('GPU')) > 0
        else 'cpu'
    )
    group = [framework, device, size]
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


def evaluate(
        cfg: DictConfig,
        trainer: Trainer,
        job_type: str,
        run: Optional[Any] = None,
        nchains: Optional[int] = 10,
        eps: Optional[Tensor] = None,
) -> dict:
    assert isinstance(nchains, int)
    assert job_type in ['eval', 'hmc']
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


def main(cfg: DictConfig) -> dict:
    outputs = {}
    objs = setup(cfg)
    trainer = objs['trainer']  # type: Trainer

    # nchains = min((cfg.dynamics.nchains, cfg.dynamics.nleapfrog))
    # cfg.update({'nchains': nchains})

    tag = generate_id() if trainer.rank == 0 else None
    outdir = Path(cfg.get('outdir', os.getcwd()))
    debug = any([s in outdir.as_posix() for s in ['debug', 'test']])
    cfg = update_wandb_config(cfg, tag=tag, debug=debug)

    run = None
    if RANK == 0:
        run = wandb.init(**cfg.wandb.setup)
        wandb.define_metric('dQint_eval', summary='mean')
        assert run is not None and run is wandb.run
        run.log_code(HERE.as_posix())
        cfg_dict = OmegaConf.to_container(cfg,
                                          resolve=True,
                                          throw_on_missing=True)
        run.config.update(cfg_dict)
        utils.print_config(cfg, resolve=True)
    # ----------------------------------------------------------
    # 1. Train model
    # 2. Evaluate trained model
    # 3. Run generic HMC as baseline w/ same trajectory length
    # ----------------------------------------------------------
    should_train = (cfg.steps.nera > 0 and cfg.steps.nepoch > 0)
    if should_train:
        outputs['train'] = train(cfg, trainer, run=run)      # [1.]

    if RANK == 0:
        nchains = max((4, cfg.dynamics.nchains // 8))
        if should_train and cfg.steps.test > 0:
            log.warning('Evaluating trained model')
            outputs['eval'] = evaluate(cfg,
                                   run=run,
                                   eps=None,
                                   job_type='eval',
                                   nchains=nchains,
                                   trainer=trainer)
        if cfg.steps.test > 0:
            log.warning('Running generic HMC')
            eps = tf.constant(float(cfg.get('eps_hmc', 0.01)))
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
    _ = main(cfg)


if __name__ == '__main__':
    launch()
