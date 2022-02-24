"""
main_tensorflow.py

Main entry-point for training L2HMC Dynamics w/ TensorFlow
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
import wandb
from wandb.util import generate_id

from l2hmc.common import analyze_dataset, save_logs
from l2hmc.configs import InputSpec
from l2hmc.dynamics.tensorflow.dynamics import Dynamics
from l2hmc.lattice.tensorflow.lattice import Lattice
from l2hmc.loss.tensorflow.loss import LatticeLoss
from l2hmc.network.tensorflow.network import NetworkFactory
from l2hmc.trainers.tensorflow.trainer import Trainer
from l2hmc.utils.console import is_interactive
from l2hmc.utils.hvd_init import RANK

log = logging.getLogger(__file__)


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
        'dynamics': dynamics,
        'trainer': trainer,
        'optimizer': optimizer,
        'rank': RANK,
    }


def train(cfg: DictConfig) -> dict:
    objs = setup(cfg)
    trainer = objs['trainer']  # type: Trainer
    outdir = Path(cfg.get('outdir', os.getcwd()))
    train_dir = outdir.joinpath('train')
    train_dir.mkdir(exist_ok=True, parents=True)
    train_summary_dir = train_dir.joinpath('summaries')
    train_summary_dir.mkdir(exist_ok=True, parents=True)
    kwargs = {
        'save_x': cfg.get('save_x', False),
        # 'width': cfg.get('width', os.environ.get('COLUMNS', 150)),
        'compile': cfg.get('compile', True),
        'jit_compile': cfg.get('jit_compile', False),
    }
    id = None
    group = None
    train_run = None
    train_writer = None
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    nchains = min((cfg.dynamics.xshape[0], cfg.dynamics.nleapfrog))
    width = max((150, int(cfg.get('width', os.environ.get('COLUMNS', 150)))))
    if objs['rank'] == 0:
        id = generate_id()
        # wandb.tensorboard.patch(root_logdir=outdir.as_posix(), pytorch=False)
        gpus = tf.config.experimental.list_physical_devices('gpu')
        gnames = ['tensorflow']
        # if len(gpus) > 1:
        #     gnames.append('hvd')

        gnames.append('gpu') if len(gpus) > 0 else gnames.append('cpu')
        if 'debug' in train_dir.as_posix():
            gnames.append('debug')

        group = '/'.join(gnames)
        train_run = wandb.init(id=id,
                               group=group,
                               resume='allow',
                               config=wandb_cfg,
                               # magic=True,
                               job_type='train',
                               sync_tensorboard=True,
                               entity=cfg.wandb.setup.entity,
                               project=cfg.wandb.setup.project,
                               settings=wandb.Settings(start_method='thread'))
        train_writer = tf.summary.create_file_writer(  # type: ignore
            train_summary_dir.as_posix()
        )
        # run.watch(objs['dynamics'], objs['loss_fn'])

    train_output = trainer.train(run=train_run, train_dir=train_dir,
                                 width=width, writer=train_writer, **kwargs)
    output = {
        'setup': objs,
        'outdir': outdir,
        'train': train_output,
    }
    if objs['rank'] == 0:
        # outdir = Path(cfg.get('outdir', os.getcwd()))
        train_dataset = train_output['history'].get_dataset()
        _ = analyze_dataset(train_dataset,
                            prefix='train',
                            nchains=nchains,
                            outdir=train_dir,
                            # lattice=objs['lattice'],
                            # xarr=train_output['xarr'],
                            title='Training: TensorFlow')

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

    if objs['rank'] == 0:
        eval_run = wandb.init(id=id,
                              group=group,
                              resume='allow',
                              config=wandb_cfg,
                              # magic=True,
                              sync_tensorboard=True,
                              # pytorch=True,
                              job_type='eval',
                              entity=cfg.wandb.setup.entity,
                              project=cfg.wandb.setup.project,
                              settings=wandb.Settings(start_method='thread'))

        _ = kwargs.pop('save_x', False)
        eval_dir = outdir.joinpath('eval')
        eval_summary_dir = eval_dir.joinpath('summaries')
        eval_summary_dir.mkdir(exist_ok=True, parents=True)
        eval_dir.mkdir(exist_ok=True, parents=True)

        eval_writer = tf.summary.create_file_writer(  # type: ignore
            eval_summary_dir.as_posix()
        )

        eval_output = trainer.eval(run=eval_run,
                                   writer=eval_writer,
                                   width=width)

        tfrac = cfg.get('therm_frac', 0.2)
        eval_dataset = eval_output['history'].get_dataset(therm_frac=tfrac)
        _ = analyze_dataset(eval_dataset,
                            prefix='eval',
                            nchains=nchains,
                            outdir=eval_dir,
                            # lattice=objs['lattice'],
                            # xarr=eval_output['xarr'],
                            title='Evaluating: TensorFlow')

        if not is_interactive():
            edir = eval_dir.joinpath('logs')
            edir.mkdir(exist_ok=True, parents=True)
            log.info(f'Saving eval logs to: {edir.as_posix()}')
            save_logs(logdir=edir,
                      tables=eval_output['tables'],
                      summaries=eval_output['summaries'])

        eval_run.finish()
        eval_writer.close()

        output.update({'eval': eval_output})

    return output


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    log.info(f'Working directory: {os.getcwd()}')
    log.info(OmegaConf.to_yaml(cfg))
    _ = train(cfg)


if __name__ == '__main__':
    main()
