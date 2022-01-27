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
import torch

from configs import (
    DynamicsConfig,
    InputSpec,
    LossConfig,
    NetWeights,
    NetworkConfig,
    Steps,
)
from dynamics.pytorch.dynamics import Dynamics
from lattice.pytorch.lattice import Lattice
from loss.pytorch.loss import LatticeLoss
from network.pytorch.network import NetworkFactory
from trainers.pytorch.trainer import Trainer
from utils.console import console
import utils.plot_helpers as hplt


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    console.log(OmegaConf.to_yaml(cfg))
    steps = Steps(**cfg.steps)
    loss_config = LossConfig(**cfg.loss)
    net_weights = NetWeights(**cfg.net_weights)
    network_config = NetworkConfig(**cfg.network)
    dynamics_config = DynamicsConfig(**cfg.dynamics)

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
    output = trainer.train()
    history = output['history']
    data = history.get_dataset()
    num_chains = min((xshape[0] // 2, dynamics_config.nleapfrog))
    for key, val in data.items():
        fig, _, _ = hplt.plot_dataArray(val,
                                        key=key,
                                        title='PyTorch',
                                        num_chains=num_chains)
        outdir = Path(os.getcwd()).joinpath('plots', 'training')
        outfile = outdir.joinpath(f'{key}.svg')
        outfile.parent.mkdir(exist_ok=True, parents=True)
        console.log(f'Saving figure to: {outfile.as_posix()}')
        fig.savefig(outfile, dpi=500, bbox_inches='tight')

    fout = Path(os.getcwd()).joinpath('logs', 'training', 'console.txt')
    fout.parent.mkdir(exist_ok=True, parents=True)

    table = output['table']
    console.print(table)
    text = console.export_text()
    with open(fout.as_posix(), 'w') as f:
        f.write(text)

    console.print(table)
    html = console.export_html(clear=False)
    fout = Path(os.getcwd()).joinpath('logs', 'training', 'table_export.html')
    with open(fout, 'w') as f:
        f.write(html)

    console.save_text(fout.as_posix())


if __name__ == '__main__':
    main()
