"""
l2hmc/common.py

Contains methods intended to be shared across frameworks.
"""
from __future__ import absolute_import, annotations, division, print_function
import datetime
import logging
import os
from pathlib import Path
from typing import Any, Optional

import joblib
from omegaconf import DictConfig
import pandas as pd
from rich.table import Table
import wandb
import xarray as xr

from l2hmc.configs import AnnealingSchedule, Steps
from l2hmc.utils.console import console
from l2hmc.utils.plot_helpers import make_ridgeplots, plot_dataArray

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
) -> Path:
    fname = 'dataset.nc' if job_type is None else f'{job_type}_dataset.nc'
    datafile = Path(outdir).joinpath(fname)
    mode = 'a' if datafile.is_file() else 'w'
    log.info(f'Saving dataset to: {datafile.as_posix()}')
    datafile.parent.mkdir(exist_ok=True, parents=True)
    dataset.to_netcdf(datafile.as_posix(), mode=mode)

    return datafile


def table_to_dict(table: Table, data: dict = None) -> dict:
    if data is None:
        return {
            column.header: [
                float(i) for i in list(column.cells)  # type:ignore
            ]
            for column in table.columns
        }
    for column in table.columns:
        try:
            data[column.header].extend([
                float(i) for i in list(column.cells)  # type:ignore
            ])
        except KeyError:
            data[column.header] = [
                float(i) for i in list(column.cells)  # type:ignore
            ]

    return data


def save_logs(
        tables: dict[str, Table],
        summaries: Optional[list[str]] = None,
        job_type: str = None,
        # rows: Optional[dict] = None,  # type:ignore
        logdir: os.PathLike = None,
        run: Optional[Any] = None,
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

    # data = {}
    data = {}
    for idx, table in tables.items():
        if idx == 0:
            data = table_to_dict(table)
        else:
            data = table_to_dict(table, data)

        console.print(table)
        html = console.export_html(clear=False)
        text = console.export_text()
        with open(hfile.as_posix(), 'a') as f:
            f.write(html)
        with open(tfile, 'a') as f:
            f.write(text)

    df = pd.DataFrame.from_dict(data)
    dfile = Path(logdir).joinpath(f'{job_type}_table.csv')
    df.to_csv(dfile.as_posix())

    if run is not None:
        with open(hfile.as_posix(), 'r') as f:
            html = f.read()

        run.log({f'Media/{job_type}': wandb.Html(html)})
        run.log({
            f'DataFrames/{job_type}': wandb.Table(data=df)
        })

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
        # run: Any = None,
) -> None:
    outdir = Path(outdir) if outdir is not None else Path(os.getcwd())
    outdir.mkdir(exist_ok=True, parents=True)
    # outdir = outdir.joinpath('plots')
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

        pngdir = outdir.joinpath('pngs')
        outdir.mkdir(exist_ok=True, parents=True)
        pngdir.mkdir(exist_ok=True, parents=True)

        fsvg = outdir.joinpath(f'{key}.svg')
        fpng = pngdir.joinpath(f'{key}.png')
        if fsvg.is_file():
            fsvg = outdir.joinpath(f'xarray-{key}.svg')
        if fpng.is_file():
            fpng = pngdir.joinpath(f'xarray-{key}.svg')

        fig.savefig(fsvg.as_posix(), dpi=500, bbox_inches='tight')
        fig.savefig(fpng.as_posix(), dpi=500, bbox_inches='tight')

    _ = make_ridgeplots(dataset,
                        outdir=outdir,
                        drop_nans=True,
                        num_chains=nchains)


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
                 nchains=nchains,
                 title=title,
                 job_type=job_type,
                 outdir=dirs['plots'])
    if save:
        try:
            datafile = save_dataset(dataset,
                                    outdir=dirs['data'],
                                    job_type=job_type)
        except ValueError:
            datafile = None
            for key, val in dataset.data_vars.items():
                fout = Path(dirs['data']).joinpath(f'{key}.z')
                try:
                    joblib.dump(val.values, fout)
                except Exception:
                    log.error(f'Unable to `joblib.dump` {key}, skipping!')

        artifact = None
        if job_type is not None and run is not None:
            name = f'{job_type}-{run.id}'
            artifact = wandb.Artifact(name=name, type='result')
            pngdir = Path(dirs['plots']).joinpath('pngs').as_posix()

            artifact.add_dir(pngdir, name=f'{job_type}/plots')
            if datafile is not None:
                artifact.add_file(datafile.as_posix(), name=f'{job_type}/data')

            run.log_artifact(artifact)

    return dataset
