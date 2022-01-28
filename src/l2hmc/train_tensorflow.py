"""
train_tensorflow.py

Main entry point for training the pytorch model of the L2HMC dynamics.
"""
from __future__ import absolute_import, annotations, division, print_function
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf

from configs import (
    DynamicsConfig,
    InputSpec,
    LossConfig,
    NetWeights,
    NetworkConfig,
    Steps,
)
from l2hmc.common import (
    plot_dataset, save_logs, setup_common, setup_tensorflow, get_timestamp
)
from utils.console import console


# def build_dynamics_and_loss(
#         dynamics_config: DynamicsConfig,
#         network_factory: NetworkFactory,
#         loss_config: LossConfig,
# ) -> tuple[Dynamics, LatticeLoss]:
#     """Build Dynamics object."""
#     lattice = Lattice(tuple(dynamics_config.xshape))
#     potential_fn = lattice.action
#     dynamics = Dynamics(
#         config=dynamics_config,
#         potential_fn=potential_fn,
#         network_factory=network_factory,
#     )
#     loss = LatticeLoss(lattice=lattice, loss_config=loss_config)

#     return dynamics, loss


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    console.log(OmegaConf.to_yaml(cfg))
    common = setup_common(cfg)
    setup = setup_tensorflow(common)
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


# @hydra.main(config_path='./conf', config_name='config')
# def main(cfg: DictConfig) -> None:
#     console.log(OmegaConf.to_yaml(cfg))
#     steps = Steps(**cfg.steps)
#     loss_config = LossConfig(**cfg.loss)
#     net_weights = NetWeights(**cfg.net_weights)
#     network_config = NetworkConfig(**cfg.network)
#     dynamics_config = DynamicsConfig(**cfg.dynamics)

#     xdim = dynamics_config.xdim
#     xshape = dynamics_config.xshape
#     input_spec = InputSpec(xshape=xshape,
#                            vnet={'v': (xdim,), 'x': (xdim,)},
#                            xnet={'v': (xdim,), 'x': (xdim, 2)})

#     network_factory = NetworkFactory(input_spec=input_spec,
#                                      net_weights=net_weights,
#                                      network_config=network_config)

#     dynamics, loss_fn = build_dynamics_and_loss(
#         dynamics_config,
#         network_factory=network_factory,
#         loss_config=loss_config
#     )

#     optimizer = tf.keras.optimizers.Adam()
#     trainer = Trainer(steps=steps,
#                       dynamics=dynamics,
#                       loss_fn=loss_fn,
#                       optimizer=optimizer)
#     output = trainer.train(compile=True, jit_compile=False)
#     history = output['history']
#     data = history.get_dataset()
#     num_chains = min((xshape[0] // 2, dynamics_config.nleapfrog))
#     for key, val in data.data_vars.items():
#         fig, _, _ = hplt.plot_dataArray(val,
#                                         key=key,
#                                         title='TensorFlow',
#                                         num_chains=num_chains)
#         outdir = Path(os.getcwd()).joinpath('plots', 'training')
#         outfile = outdir.joinpath(f'{key}.svg')
#         outfile.parent.mkdir(exist_ok=True, parents=True)
#         console.log(f'Saving figure to: {outfile.as_posix()}')
#         fig.savefig(outfile, dpi=500, bbox_inches='tight')

#     logdir = Path(os.getcwd()).joinpath('logs', 'training')
#     logdir.mkdir(exist_ok=True, parents=True)

#     out = {
#         'console': logdir.joinpath('console.txt'),
#         'table_txt': logdir.joinpath('table_export.txt'),
#         'table_html': logdir.joinpath('table_export.html'),
#     }

#     table = output['table']
#     console.print(table)
#     text = console.export_text()
#     with open(out['console'].as_posix(), 'w') as f:
#         f.write(text)

#     console.print(table)
#     html = console.export_html(clear=False)
#     with open(out['table_html'], 'w') as f:
#         f.write(html)

#     console.save_text(out['table_txt'].as_posix())


if __name__ == '__main__':
    main()
