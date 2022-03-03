"""
l2hmc/common.py

Contains methods intended to be shared across frameworks.
"""
from __future__ import absolute_import, annotations, division, print_function
import datetime
import logging
import os
from pathlib import Path
from typing import Any
import joblib
# import pandas as pd

from omegaconf import DictConfig
from rich.table import Table
import xarray as xr

from l2hmc.configs import (
    AnnealingSchedule,
    Steps,
)
from l2hmc.utils.console import console
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
        job_type: str = None,
) -> None:
    fname = 'dataset.nc' if job_type is None else f'{job_type}_dataset.nc'
    datafile = Path(outdir).joinpath(fname)
    mode = 'a' if datafile.is_file() else 'w'
    log.info(f'Saving dataset to: {datafile.as_posix()}')
    datafile.parent.mkdir(exist_ok=True, parents=True)
    dataset.to_netcdf(datafile.as_posix(), mode=mode)


def save_logs(
        tables: dict[str, Table],
        summaries: list[str] = None,
        logdir: os.PathLike = None,
        job_type: str = None,
        run: Any = None,
) -> None:
    job_type = 'job' if job_type is None else job_type
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

    hfile = hdir.joinpath('table.html')
    hfile.parent.mkdir(exist_ok=True, parents=True)

    tfile = tdir.joinpath('table.txt')
    tfile.parent.mkdir(exist_ok=True, parents=True)

    for _, table in tables.items():
        console.print(table)
        html = console.export_html(clear=False)
        text = console.export_text()
        with open(hfile.as_posix(), 'a') as f:
            f.write(html)
        with open(tfile, 'a') as f:
            f.write(text)

        if run is not None:
            try:
                run.log({f'tables/{job_type}': text})
            except Exception:
                continue

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
        job_type: str = None,
        run: Any = None,
) -> None:
    outdir = Path(outdir) if outdir is not None else Path(os.getcwd())
    outdir = outdir.joinpath('plots')
    job_type = job_type if job_type is not None else f'job-{get_timestamp()}'
    for key, val in dataset.data_vars.items():
        if key == 'x':
            continue

        try:
            fig, _, _ = plot_dataArray(val,
                                       key=key,
                                       title=title,
                                       line_labels=False,
                                       num_chains=nchains)
        except TypeError:
            log.error(f'Unable to `plot_dataArray` for {key}')
            continue
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
        if run is not None:
            try:
                run.log({f'plots/{job_type}.{key}': fig})
            except Exception:
                log.error(f'Unable to create plot for {key}, skipping!')
                pass

        fig.savefig(outpng.as_posix(), dpi=500, bbox_inches='tight')

    _ = make_ridgeplots(dataset, num_chains=nchains, out_dir=outdir)


def analyze_dataset(
        dataset: xr.Dataset,
        outdir: os.PathLike,
        nchains: int = 16,
        title: str = None,
        job_type: str = None,
        save: bool = True,
        run: Any = None,
):
    job_type = job_type if job_type is not None else f'job-{get_timestamp()}'
    dirs = make_subdirs(outdir)
    plot_dataset(dataset,
                 run=run,
                 nchains=nchains,
                 title=title,
                 job_type=job_type,
                 outdir=dirs['plots'])
    if save:
        try:
            save_dataset(dataset, outdir=dirs['data'], job_type=job_type)
        except ValueError:
            for key, val in dataset.data_vars.items():
                fout = Path(dirs['data']).joinpath(f'{key}.z')
                try:
                    joblib.dump(val.values, fout)
                except Exception:
                    log.error(f'Unable to `joblib.dump` {key}, skipping!')

    return dataset
