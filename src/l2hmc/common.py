"""
l2hmc/common.py

Contains methods intended to be shared across frameworks.
"""
from __future__ import absolute_import, annotations, division, print_function
import datetime
import logging
import os
from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig
from rich.table import Table
import xarray as xr

from l2hmc.configs import (
    DynamicsConfig,
    InputSpec,
    LossConfig,
    NetWeights,
    NetworkConfig,
    Steps,
)
from l2hmc.utils.console import console, is_interactive
from l2hmc.utils.plot_helpers import plot_dataArray, plot_plaqs

log = logging.getLogger(__name__)


def get_timestamp(fstr=None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


def setup_pytorch(configs: dict) -> dict:
    import torch
    from accelerate import Accelerator
    from l2hmc.dynamics.pytorch.dynamics import Dynamics
    from l2hmc.lattice.pytorch.lattice import Lattice
    from l2hmc.network.pytorch.network import NetworkFactory
    from l2hmc.trainers.pytorch.trainer import Trainer
    from l2hmc.loss.pytorch.loss import LatticeLoss

    steps = configs['steps']
    loss_config = configs['loss_config']
    net_weights = configs['net_weights']
    network_config = configs['network_config']
    dynamics_config = configs['dynamics_config']

    xdim = dynamics_config.xdim
    xshape = dynamics_config.xshape

    input_spec = InputSpec(xshape=xshape,
                           vnet={'v': [xdim, ], 'x': [xdim, ]},
                           xnet={'v': [xdim, ], 'x': [xdim, 2]})
    network_factory = NetworkFactory(input_spec=input_spec,
                                     net_weights=net_weights,
                                     network_config=network_config)
    lattice = Lattice(tuple(xshape))
    dynamics = Dynamics(config=dynamics_config,
                        potential_fn=lattice.action,
                        network_factory=network_factory)
    loss_fn = LatticeLoss(lattice=lattice, loss_config=loss_config)
    accelerator = Accelerator()

    optimizer = torch.optim.Adam(dynamics.parameters())
    dynamics = dynamics.to(accelerator.device)
    dynamics, optimizer = accelerator.prepare(dynamics, optimizer)
    trainer = Trainer(steps=steps,
                      loss_fn=loss_fn,
                      dynamics=dynamics,
                      optimizer=optimizer,
                      accelerator=accelerator)

    return {
        'lattice': lattice,
        'loss_fn': loss_fn,
        'dynamics': dynamics,
        'trainer': trainer,
        'optimizer': optimizer,
        'accelerator': accelerator,
    }


def setup_tensorflow(configs: dict) -> dict:
    try:
        import tensorflow as tf
        import horovod.tensorflow as hvd
        hvd.init()
        # RANK = hvd.rank()  # type: ignore
        # SIZE = hvd.size()  # type: ignore
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 0:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            gpu = gpus[hvd.local_rank()]
            tf.config.experimental.set_visible_devices(gpu, 'GPU')

    except (ImportError, ModuleNotFoundError):
        pass
        # RANK = 0
        # SIZE = 1

    from l2hmc.dynamics.tensorflow.dynamics import Dynamics
    from l2hmc.lattice.tensorflow.lattice import Lattice
    from l2hmc.loss.tensorflow.loss import LatticeLoss
    from l2hmc.network.tensorflow.network import NetworkFactory
    from l2hmc.trainers.tensorflow.trainer import Trainer
    import tensorflow as tf

    steps = configs['steps']
    loss_config = configs['loss_config']
    net_weights = configs['net_weights']
    network_config = configs['network_config']
    dynamics_config = configs['dynamics_config']

    xdim = dynamics_config.xdim
    xshape = dynamics_config.xshape

    input_spec = InputSpec(xshape=xshape,
                           vnet={'v': [xdim, ], 'x': [xdim, ]},
                           xnet={'v': [xdim, ], 'x': [xdim, 2]})
    network_factory = NetworkFactory(input_spec=input_spec,
                                     net_weights=net_weights,
                                     network_config=network_config)
    lattice = Lattice(tuple(xshape))
    dynamics = Dynamics(config=dynamics_config,
                        potential_fn=lattice.action,
                        network_factory=network_factory)
    loss_fn = LatticeLoss(lattice=lattice, loss_config=loss_config)
    optimizer = tf.keras.optimizers.Adam()
    trainer = Trainer(steps=steps,
                      loss_fn=loss_fn,
                      dynamics=dynamics,
                      optimizer=optimizer)

    return {
        'lattice': lattice,
        'loss_fn': loss_fn,
        'dynamics': dynamics,
        'trainer': trainer,
        'optimizer': optimizer,
    }


def setup_common(cfg: DictConfig) -> dict:
    steps = Steps(**cfg.steps)
    loss_config = LossConfig(**cfg.loss)
    net_weights = NetWeights(**cfg.net_weights)
    network_config = NetworkConfig(**cfg.network)
    dynamics_config = DynamicsConfig(**cfg.dynamics)

    return {
        'steps': steps,
        'loss_config': loss_config,
        'net_weights': net_weights,
        'network_config': network_config,
        'dynamics_config': dynamics_config,
    }


def plot_dataset(
    dataset: xr.Dataset,
    nchains: int = 10,
    outdir: os.PathLike = None,
    title: str = None,
) -> None:
    if outdir is None:
        outdir = Path(os.getcwd()).joinpath('plots', 'training')
    else:
        outdir = Path(outdir)

    for key, val in dataset.data_vars.items():
        if key == 'x':
            continue

        fig, _, _ = plot_dataArray(val,
                                   key=key,
                                   title=title,
                                   num_chains=nchains)
        outfile = outdir.joinpath(f'{key}.svg')
        outfile.parent.mkdir(exist_ok=True, parents=True)
        log.info(f'Saving figure to: {outfile.as_posix()}')
        fig.savefig(outfile.as_posix(), dpi=500, bbox_inches='tight')


def save_logs(
        tables: dict[str, Table],
        logdir: os.PathLike = None
) -> None:
    if logdir is None:
        logdir = Path(os.getcwd()).joinpath('logs', 'training')
    else:
        logdir = Path(logdir)

    cfile = logdir.joinpath('console.txt').as_posix()
    text = console.export_text()
    with open(cfile, 'w') as f:
        f.write(text)

    table_dir = logdir.joinpath('tables')
    tdir = table_dir.joinpath('txt')
    hdir = table_dir.joinpath('html')

    hfile = hdir.joinpath('training_table.html')
    hfile.parent.mkdir(exist_ok=True, parents=True)

    tfile = tdir.joinpath('training_table.txt')
    tfile.parent.mkdir(exist_ok=True, parents=True)

    for _, table in tables.items():
        console.print(table)
        html = console.export_html(clear=False)
        # hfile = hdir.joinpath(f'era{era}.html')
        with open(hfile.as_posix(), 'a') as f:
            f.write(html)

        # tfile = tdir.joinpath(f'era{era}.txt')
        text = console.export_text()
        with open(tfile, 'a') as f:
            f.write(text)


def train(cfg: DictConfig) -> dict:
    beta = cfg.get('beta', None)
    if beta is None:
        log.warn('Beta not specified! Using beta = 1.0')
        beta = 1.0

    save_x = cfg.get('save_x', False)
    framework = cfg.get('framework', None)
    width = cfg.get('width', 0)
    assert framework is not None

    kwargs = {'beta': beta, 'save_x': save_x, }
    if width > 0:
        kwargs['width'] = width

    common = setup_common(cfg)
    if framework in ['pytorch', 'torch', 'pt']:
        framework = 'pytorch'
        setup = setup_pytorch(common)
    else:
        if framework in ['tensorflow', 'tf']:
            framework = 'tensorflow'
            setup = setup_tensorflow(common)
            kwargs.update({
                'compile': cfg.get('compile', True),
                'jit_compile': cfg.get('jit_compile', False),
            })
        else:
            raise ValueError(f'Unexpected framework: {framework}')

    log.info(f'Using {framework}, with trainer: {setup["trainer"]}')
    train_output = setup['trainer'].train(**kwargs)
    history = train_output['history']
    dataset = history.get_dataset()
    nchains = min((cfg.dynamics.xshape[0], cfg.dynamics.nleapfrog))
    outdir = Path(cfg.get('outdir', os.getcwd()))
    outdir.mkdir(exist_ok=True, parents=True)
    dirs = {'outdir': outdir}
    for key in ['logs', 'plots', 'data']:
        d = dirs['outdir'].joinpath(key)
        d.mkdir(exist_ok=True, parents=True)
        dirs[key] = d

    plot_dataset(dataset,
                 nchains=nchains,
                 title=framework,
                 outdir=dirs['plots'])
    datafile = dirs['data'].joinpath('history_dataset.nc')
    mode = 'a' if datafile.is_file() else 'w'
    log.info(f'Saving dataset to: {datafile.as_posix()}')
    datafile.parent.mkdir(exist_ok=True, parents=True)
    dataset.to_netcdf(datafile.as_posix(), mode=mode)
    if cfg.get('save_x', False):
        xarr = train_output['xarr']
        beta = cfg.get('beta', None)
        lattice = setup['lattice']
        plaqs_diffs = []
        for x in xarr:
            plaqs_diffs.append(lattice.plaqs_diff(beta=beta, x=x).numpy())

        plaqs_diffs = np.array(plaqs_diffs)
        _ = plot_plaqs(plaqs=plaqs_diffs, nchains=10, outdir=dirs['plots'])
        xfile = dirs['data'].joinpath('xarr.z')
        log.info(f'saving xarr to: {xfile}')
        joblib.dump(xarr, xfile)

    output = {
        'train_output': train_output,
        'setup': setup,
        'dirs': dirs,
        'dataset': dataset,
        'history': history,
    }
    if is_interactive():
        return output

    log.info(
        f'Saving logs and training table to: {dirs["logs"].as_posix()}'
    )
    save_logs(train_output['tables'], logdir=dirs['logs'])

    return output
