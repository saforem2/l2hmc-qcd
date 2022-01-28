"""
l2hmc/common.py

Contains methods intended to be shared across frameworks.
"""
from __future__ import absolute_import, annotations, division, print_function
import os

from omegaconf import DictConfig
from pathlib import Path
import xarray as xr
from l2hmc.utils.plot_helpers import plot_dataArray
from l2hmc.utils.console import console
from rich.table import Table
import datetime

from l2hmc.configs import (
    DynamicsConfig,
    InputSpec,
    LossConfig,
    NetWeights,
    NetworkConfig,
    Steps,
)


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
                           vnet={'v': (xdim,), 'x': (xdim,)},
                           xnet={'v': (xdim,), 'x': (xdim, 2)})
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
    import tensorflow as tf
    from l2hmc.dynamics.tensorflow.dynamics import Dynamics
    from l2hmc.lattice.tensorflow.lattice import Lattice
    from l2hmc.loss.tensorflow.loss import LatticeLoss
    from l2hmc.network.tensorflow.network import NetworkFactory
    from l2hmc.trainers.tensorflow.trainer import Trainer

    steps = configs['steps']
    loss_config = configs['loss_config']
    net_weights = configs['net_weights']
    network_config = configs['network_config']
    dynamics_config = configs['dynamics_config']

    xdim = dynamics_config.xdim
    xshape = dynamics_config.xshape

    input_spec = InputSpec(xshape=xshape,
                           vnet={'v': (xdim,), 'x': (xdim,)},
                           xnet={'v': (xdim,), 'x': (xdim, 2)})
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
    nchains: int = 0,
    outdir: os.PathLike = None,
) -> None:
    if outdir is None:
        outdir = Path(os.getcwd()).joinpath('plots', 'training')
    else:
        outdir = Path(outdir)

    for key, val in dataset.data_vars.items():
        fig, _, _ = plot_dataArray(val,
                                   key=key,
                                   title='PyTorch',
                                   num_chains=nchains)
        outfile = outdir.joinpath(f'{key}.svg')
        outfile.parent.mkdir(exist_ok=True, parents=True)
        console.log(f'Saving figure to: {outfile.as_posix()}')
        fig.savefig(outfile.as_posix(), dpi=500, bbox_inches='tight')


def save_logs(
        tables: dict[str, Table],
        logdir: os.PathLike = None
) -> None:
    if logdir is None:
        logdir = Path(os.getcwd()).joinpath('logs', 'training')
    else:
        logdir = Path(logdir)

    for era, table in tables.items():
        console.print(table)

    cfile = logdir.joinpath('console.txt').as_posix()
    text = console.export_text()
    with open(cfile, 'w') as f:
        f.write(text)

    table_dir = logdir.joinpath('tables')
    tdir = table_dir.joinpath('txt')
    hdir = table_dir.joinpath('html')
    for era, table in tables.items():
        console.print(table)
        html = console.export_html(clear=False)
        hfile = hdir.joinpath(f'era{era}.html')
        hfile.parent.mkdir(exist_ok=True, parents=True)
        with open(hfile.as_posix(), 'w') as f:
            f.write(html)

        tfile = tdir.joinpath(f'era{era}.txt')
        tfile.parent.mkdir(exist_ok=True, parents=True)
        console.save_text(tfile.as_posix())
