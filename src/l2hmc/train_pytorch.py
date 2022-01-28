"""
train_pytorch.py

Main entry point for training the pytorch model of the L2HMC dynamics.
"""
from __future__ import absolute_import, annotations, division, print_function
import os
from pathlib import Path

from accelerate import Accelerator
import hydra
from omegaconf import DictConfig, OmegaConf
from rich.table import Table
import torch
import xarray as xr

from l2hmc.common import plot_dataset, save_logs, setup_common, setup_pytorch
from l2hmc.configs import (
    DynamicsConfig,
    InputSpec,
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
from l2hmc.utils.console import console
from l2hmc.utils.plot_helpers import plot_dataArray


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    console.log(OmegaConf.to_yaml(cfg))
    common = setup_common(cfg)
    setup = setup_pytorch(common)
    output = setup['trainer'].train()
    history = output['history']
    dataset = history.get_dataset()


    nchains = min((cfg.dynamics.xshape[0], cfg.dynamics.nleapfrog))

    day = get_timestamp('%Y-%H-%m')
    time = get_timestamp('%H-%m-%S')
    lfstr = f'nlf{cfg.dynamics.nleapfrog}'
    vstr = f'{cfg.dynamics.xshape[1]}x{cfg.dynamics.xshape[2]}'
    outdir = Path(os.getcwd()).joinpath(
        'output', 'tensorflow',
        vstr, lfstr, day, time,
        'training'
    )
    outdir.mkdir(exist_ok=True, parents=True)

    dirs = {'outdir': outdir}
    for key in ['logs', 'plots', 'data']:
        d = dirs['outdir'].joinpath(key)
        d.mkdir(exist_ok=True, parents=True)
        dirs[key] = d

    plot_dataset(dataset, nchains=nchains, outdir=dirs['plots'])

    console.log(
        f'Saving logs and training table to: {dirs["logs"].as_posix()}'
    )
    save_logs(output['tables'], logdir=dirs['logs'])

    datafile = dirs['data'].joinpath('history_dataset.nc')
    mode = 'a' if datafile.is_file() else 'w'
    console.log(f'Saving dataset to: {datafile.as_posix()}')
    datafile.parent.mkdir(exist_ok=True, parents=True)
    dataset.to_netcdf(datafile.as_posix(), mode=mode)




    # logdir = Path(os.getcwd()).joinpath('')
    # nchains = min((cfg.dynamics.xshape[0], cfg.dynamics.nleapfrog))
    # plotdir = Path(os.getcwd()).joinpath('plots', 'training')
    # plot_dataset(dataset, nchains=nchains, outdir=plotdir)

    # logdir = Path(os.getcwd()).joinpath('logs', 'training')
    # logdir.mkdir(exist_ok=True, parents=True)
    # console.log(f'Saving logs and training table to: {logdir.as_posix()}')
    # save_logs(output['tables'], logdir=logdir)

    # datadir = Path(os.getcwd()).joinpath('data', 'training')
    # datafile = datadir.joinpath('history_dataset.nc')
    # datafile.parent.mkdir(exist_ok=True, parents=True)
    # mode = 'a' if datafile.is_file() else 'w'
    # console.log(f'Saving dataset to: {datafile.as_posix()}')
    # dataset.to_netcdf(datafile.as_posix(), mode=mode)


# @hydra.main(config_path='./conf', config_name='config')
# def main(cfg: DictConfig) -> None:
#     console.log(OmegaConf.to_yaml(cfg))
#     init = setup(cfg)
#     output = init['trainer'].train()
#     # output = trainer.train()
#     history = output['history']
#     dataset = history.get_dataset()  # type: xr.Dataset
#     nchains = min((cfg.dynamics.xshape[0], cfg.dynamics.nleapfrog))

#     plotdir = Path(os.getcwd()).joinpath('plots', 'training')
#     plot_dataset(dataset, nchains=nchains, outdir=plotdir)

#     logdir = Path(os.getcwd()).joinpath('logs', 'training')
#     logdir.mkdir(exist_ok=True, parents=True)
#     console.log(f'Saving logs and training table to: {logdir.as_posix()}')
#     save_logs(output['tables'], logdir=logdir)

#     datadir = Path(os.getcwd()).joinpath('data', 'training')
#     datafile = datadir.joinpath('history_dataset.nc')
#     mode = 'a' if datafile.is_file() else 'w'
#     console.log(f'Saving dataset to: {datafile.as_posix()}')
#     dataset.to_netcdf(datafile.as_posix(), mode=mode)


if __name__ == '__main__':
    main()
