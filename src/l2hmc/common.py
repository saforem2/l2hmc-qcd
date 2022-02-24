"""
l2hmc/common.py

Contains methods intended to be shared across frameworks.
"""
from __future__ import absolute_import, annotations, division, print_function
import datetime
import logging
import os
from pathlib import Path

from omegaconf import DictConfig
from rich.table import Table
import xarray as xr
# from l2hmc.lattice.lattice import BaseLattice

from l2hmc.configs import (
    AnnealingSchedule,
    # DynamicsConfig,
    # InputSpec,
    # LossConfig,
    # NetWeights,
    # NetworkConfig,
    # ConvolutionConfig,
    Steps,
)
from l2hmc.utils.console import console  # , is_interactive
from l2hmc.utils.plot_helpers import (
    plot_dataArray, make_ridgeplots
)

os.environ['AUTOGRAPH_VERBOSITY'] = '0'
log = logging.getLogger(__name__)


def get_timestamp(fstr=None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:

        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


def setup_annealing_schedule(cfg: DictConfig) -> AnnealingSchedule:
    steps = Steps(**cfg.steps)
    beta_init = cfg.get('beta_init', None)
    beta_final = cfg.get('beta_final', None)
    if beta_init is None:
        beta_init = 1.
        log.warn(
            'beta_init not specified!'
            f'using default: beta_init = {beta_init}'
        )
    if beta_final is None:
        beta_final = beta_init
        log.warn(
            'beta_final not specified!'
            f'using beta_final = beta_init = {beta_init}'
        )

    sched = AnnealingSchedule(beta_init, beta_final)
    sched.setup(steps)
    return sched


def save_dataset(
        dataset: xr.Dataset,
        outdir: os.PathLike,
        name: str = None,
) -> None:
    fname = 'dataset.nc' if name is None else f'{name}_dataset.nc'
    datafile = Path(outdir).joinpath(fname)
    mode = 'a' if datafile.is_file() else 'w'
    log.info(f'Saving dataset to: {datafile.as_posix()}')
    datafile.parent.mkdir(exist_ok=True, parents=True)
    dataset.to_netcdf(datafile.as_posix(), mode=mode)


def save_logs(
        tables: dict[str, Table],
        summaries: list[str] = None,
        logdir: os.PathLike = None
) -> None:
    if logdir is None:
        logdir = Path(os.getcwd()).joinpath('logs')
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

    if summaries is not None:
        sfile = logdir.joinpath('summaries.txt').as_posix()
        with open(sfile, 'w') as f:
            f.writelines(summaries)


def make_subdirs(basedir: os.PathLike):
    dirs = {}
    for key in ['logs', 'data', 'plots']:
        d = Path(basedir).joinpath(key)
        d.mkdir(exist_ok=True, parents=True)
        dirs[key] = d

    return dirs


def plot_dataset(
        dataset: xr.Dataset,
        nchains: int = 10,
        outdir: os.PathLike = None,
        title: str = None,
        prefix: str | list = None,
) -> None:
    outdir = Path(outdir) if outdir is not None else Path(os.getcwd())
    outdir = outdir.joinpath('plots')
    if prefix is None:
        if outdir is not None:
            if 'train' in outdir.as_posix():
                prefix = 'train'
            elif 'eval' in outdir.as_posix():
                prefix = 'eval'

    name = []
    if prefix is not None:
        name.append(prefix)

    tag = '/'.join(name)
    pngs = {tag: {}}
    for key, val in dataset.data_vars.items():
        if key == 'x':
            continue

        fig, _, _ = plot_dataArray(val,
                                   key=key,
                                   title=title,
                                   line_labels=False,
                                   num_chains=nchains)
        # try:
        #     wandb.log({f'chart/{name}': fig})
        # except AttributeError:
        #     # log.error(err.name)
        #     log.error(
        #         f'Error logging `chart/{name}` with `wandb.log`, skipping!',
        #     )
        # chart_name = '/'.join([*name, f'{key}_chart'])
        # wandb.log({key_: fig})
        outfile = outdir.joinpath(f'{key}.svg')
        outfile.parent.mkdir(exist_ok=True, parents=True)
        log.info(f'Saving figure to: {outfile.as_posix()}')
        fig.savefig(outfile.as_posix(), dpi=500, bbox_inches='tight')
        pngdir = outdir.joinpath('pngs')
        pngdir.mkdir(exist_ok=True, parents=True)
        outpng = pngdir.joinpath(f'{key}.png')
        fig.savefig(outpng.as_posix(), dpi=500, bbox_inches='tight')
        pngs[tag].update({key: outpng.as_posix()})

    # for tag, fname in pngs.items():
    #     wandb.log({'pngs': wandb.Image(fname)})

    _ = make_ridgeplots(dataset, num_chains=nchains, out_dir=outdir)


def analyze_dataset(
        dataset: xr.Dataset,
        outdir: os.PathLike,
        nchains: int = 16,
        title: str = None,
        prefix: str = 'dataset',
        save: bool = True,
):
    dirs = make_subdirs(outdir)
    plot_dataset(dataset,
                 nchains=nchains,
                 title=title,
                 prefix=prefix,
                 outdir=dirs['plots'])
    if save:
        save_dataset(dataset, outdir=dirs['data'], name=prefix)

    # history = {}
    # if xarr is not None and lattice is not None:
    #     metrics = lattice.calc_metrics(xarr[0])
    #     history = {}
    #     for key, val in metrics.items():
    #         try:
    #             val = val.cpu().numpy()     # type: ignore
    #         except AttributeError:
    #             val = val.numpy()           # type: ignore

    #         history[key] = [val]

    #     for idx, x in enumerate(xarr[1:]):
    #         metrics = lattice.calc_metrics(x)
    #         metrics_np = {}
    #         for key, val in metrics.items():
    #             try:
    #                 val = val.cpu().numpy()  # type: ignore
    #             except AttributeError:
    #                 val = val.numpy()        # type: ignore

    #             try:
    #                 history[key].append(val)  # type: ignore
    #             except KeyError:
    #                 history[key] = [val]      # type: ignore

    #             metrics_np[key] = val

    #         # wandb.log({'lattice': {prefix: metrics_np}}, step=idx+1)
    #         # wandb.log({prefix: {'lattice': metrics_np}}, step=idx)

    #     intQ = np.array(history['intQ'])
    #     sinQ = np.array(history['sinQ'])
    #     dQint = np.abs(intQ[1:] - intQ[:-1])  # type: ignore
    #     dQsin = np.abs(sinQ[1:] - sinQ[:-1])  # type: ignore
    #     history['dQint'] = [intQ[0], *dQint]
    #     history['dQsin'] = [sinQ[0], *dQsin]
    #     # wandb.summary({prefix: {'dQint': history['']}})
    #     # wandb.log({'lattice': {prefix: history}})

    #     xlabel = 'Step'
    #     if prefix == 'train':
    #         xlabel = 'Train Epoch'
    #     elif prefix == 'eval':
    #         xlabel = 'Eval Step'

    #     for key, val in history.items():
    #         val = np.array(val)
    #         pfile = dirs['plots'].joinpath(f'{key}.svg')
    #         fig, _ = plot_chains(y=val, num_chains=nchains,
    #                              label=key, xlabel=xlabel,
    #                              ylabel=key, outfile=pfile)

    #         dfile = dirs['data'].joinpath(f'{key}.z')
    #         log.info(f'Saving {key} to {dfile}')
    #         joblib.dump(val, dfile)

    #     # xfile = dirs['data'].joinpath('xarr.z')
    #     # log.info(f'Saving xarr to: {xfile}')
    #     # joblib.dump(xarr, xfile)

    return dataset


# def train(cfg: DictConfig) -> dict:
#     save_x = cfg.get('save_x', False)
#     framework = cfg.get('framework', None)
#     if framework is None:
#         framework = 'tensorflow'
#         log.warn('Framework not specified. Using TensorFlow.')

#     assert framework is not None

#     kwargs = {'save_x': save_x}
#     width = cfg.get('width', 150)
#     if width > 0:
#         kwargs['width'] = width

#     common = setup_common(cfg)
#     if framework in ['pytorch', 'torch', 'pt']:
#         framework = 'pytorch'
#         setup = setup_pytorch(common)
#     else:
#         if framework in ['tensorflow', 'tf']:
#             framework = 'tensorflow'
#             setup = setup_tensorflow(common)
#             kwargs.update({
#                 'compile': cfg.get('compile', True),
#                 'jit_compile': cfg.get('jit_compile', False),
#             })
#         else:
#             raise ValueError(f'Unexpected framework: {framework}')

#     outdir = Path(cfg.get('outdir', os.getcwd()))
#     RANK = setup.get('rank', None)
#     train_dir = outdir.joinpath('train')
#     eval_dir = outdir.joinpath('eval')
#     nchains = min((cfg.dynamics.xshape[0], cfg.dynamics.nleapfrog))

#     log.info(f'Using {framework}, with trainer: {setup["trainer"]}')

#     train_output = setup['trainer'].train(**kwargs)
#     if RANK == 0:
#         train_history = train_output['history']
#         train_dataset = train_history.get_dataset()
#         analyze_dataset(train_dataset,
#                         outdir=train_dir,
#                         lattice=setup['lattice'],
#                         xarr=train_output['xarr'],
#                         nchains=nchains,
#                         title=framework,
#                         name='train')

#         _ = kwargs.pop('save_x', None)
#         eval_output = setup['trainer'].eval(**kwargs)
#         eval_history = eval_output['history']
#         eval_dataset = eval_history.get_dataset()
#         analyze_dataset(eval_dataset,
#                         name='eval',
#                         outdir=eval_dir,
#                         lattice=setup['lattice'],
#                         xarr=eval_output['xarr'],
#                         nchains=nchains,
#                         title=framework)

#     if not is_interactive() and RANK == 0:
#         tdir = train_dir.joinpath('logs')
#         edir = eval_dir.joinpath('logs')
#         tdir.mkdir(exist_ok=True, parents=True)
#         edir.mkdir(exist_ok=True, parents=True)
#         log.info(f'Saving train logs to: {tdir.as_posix()}')
#         save_logs(logdir=tdir,
#                   tables=train_output['tables'],
#                   summaries=train_output['summaries'])
#         log.info(f'Saving eval logs to: {edir.as_posix()}')
#         save_logs(logdir=edir,
#                   tables=eval_output['tables'],
#                   summaries=eval_output['summaries'])

#     output = {
#         'setup': setup,
#         'train': {
#             'output': train_output,
#             'dataset': train_dataset,
#             'history': train_history,
#         },
#         'eval': {
#             'output': eval_output,
#             'dataset': eval_dataset,
#             'history': eval_history,
#         },
#     }

#     return output
