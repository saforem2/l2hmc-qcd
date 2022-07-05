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

import h5py
import joblib

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import pandas as pd
from rich.table import Table
import wandb
import xarray as xr

from l2hmc.configs import AnnealingSchedule, Steps
from l2hmc.utils.rich import get_console, is_interactive
from l2hmc.utils.plot_helpers import (
    make_ridgeplots,
    plot_dataArray,
    set_plot_style
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
        use_hdf5: Optional[bool] = True,
        job_type: Optional[str] = None,
        **kwargs,
) -> Path:
    if use_hdf5:
        fname = 'dataset.h5' if job_type is None else f'{job_type}_data.h5'
        outfile = Path(outdir).joinpath(fname)
        dataset_to_h5pyfile(outfile, dataset=dataset, **kwargs)
    else:
        fname = 'dataset.nc' if job_type is None else f'{job_type}_dataset.nc'
        outfile = Path(outdir).joinpath(fname)
        mode = 'a' if outfile.is_file() else 'w'
        log.info(f'Saving dataset to: {outfile.as_posix()}')
        outfile.parent.mkdir(exist_ok=True, parents=True)
        dataset.to_netcdf(outfile.as_posix(), mode=mode)

    return outfile


def dataset_to_h5pyfile(hfile: os.PathLike, dataset: xr.Dataset, **kwargs):
    f = h5py.File(hfile, 'a')
    for key, val in dataset.data_vars.items():
        arr = val.values
        if len(arr) == 0:
            continue
        if key in list(f.keys()):
            shape = (f[key].shape[0] + arr.shape[0])  # type: ignore
            f[key].resize(shape, axis=0)              # type: ignore
            f[key][-arr.shape[0]:] = arr              # type: ignore
        else:
            maxshape = (None,)
            if len(arr.shape) > 1:
                maxshape = (None, *arr.shape[1:])
            f.create_dataset(key, data=arr, maxshape=maxshape, **kwargs)

    f.close()


def dataset_from_h5pyfile(hfile: os.PathLike) -> dict:
    f = h5py.File(hfile, 'r')
    data = {key: f[key] for key in list(f.keys())}
    f.close()
    return data


def table_to_dict(table: Table, data: Optional[dict] = None) -> dict:
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
        job_type: Optional[str] = None,
        # rows: Optional[dict] = None,  # type:ignore
        logdir: Optional[os.PathLike] = None,
        run: Optional[Any] = None,
) -> None:
    job_type = 'job' if job_type is None else job_type
    if logdir is None:
        logdir = Path(os.getcwd()).joinpath('logs')
    else:
        logdir = Path(logdir)

    # cfile = logdir.joinpath('console.txt').as_posix()
    # text = console.export_text()
    # with open(cfile, 'w') as f:
    #     f.write(text)

    table_dir = logdir.joinpath('tables')
    tdir = table_dir.joinpath('txt')
    hdir = table_dir.joinpath('html')

    hfile = hdir.joinpath('table.html')
    hfile.parent.mkdir(exist_ok=True, parents=True)

    tfile = tdir.joinpath('table.txt')
    tfile.parent.mkdir(exist_ok=True, parents=True)

    data = {}
    console = get_console(record=True)
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
    df.to_csv(dfile.as_posix(), mode='a')

    if run is not None:
        with open(hfile.as_posix(), 'r') as f:
            html = f.read()

        # run.log({f'Media/{job_type}': wandb.Html(html)})
        run.log({
            f'DataFrames/{job_type}': wandb.Table(data=df)
        })

    if summaries is not None:
        sfile = logdir.joinpath('summaries.txt').as_posix()
        with open(sfile, 'a') as f:
            f.writelines(summaries)


def make_subdirs(basedir: os.PathLike):
    dirs = {}
    for key in ['logs', 'data', 'plots']:
        d = Path(basedir).joinpath(key)
        d.mkdir(exist_ok=True, parents=True)
        dirs[key] = d

    return dirs


def save_figure(
        fig: plt.Figure,
        key: str,
        outdir: os.PathLike,
        # run: Optional[Any] = None,
        # arun: Optional[Any] = None,
):
    if fig is None:
        fig = plt.gcf()

    pngdir = Path(outdir).joinpath('pngs')
    svgdir = Path(outdir).joinpath('svgs')
    pngdir.mkdir(parents=True, exist_ok=True)
    svgdir.mkdir(parents=True, exist_ok=True)

    svgfile = svgdir.joinpath(f'{key}.svg')
    pngfile = pngdir.joinpath(f'{key}.png')
    fig.savefig(svgfile.as_posix(), transparent=True, bbox_inches='tight')
    fig.savefig(pngfile.as_posix(), transparent=True, bbox_inches='tight')
    return fig


def make_dataset(metrics: dict) -> xr.Dataset:
    dset = {}
    for key, val in metrics.items():
        if isinstance(val, list):
            import torch
            import tensorflow as tf
            if isinstance(val[0], torch.Tensor):
                val = torch.stack(val).detach().numpy()
            elif isinstance(val[0], tf.Tensor):
                import tensorflow as tf
                val = tf.stack(val).numpy()

        assert isinstance(val, np.ndarray)
        assert len(val.shape) in [1, 2, 3]
        dims = ()
        coords = ()
        if len(val.shape) == 1:
            ndraws = val.shape[0]
            dims = ['draw']
            coords = (np.arange(len(val)))
        elif len(val.shape) == 2:
            val = val.T
            nchains, ndraws = val.shape
            dims = ('chain', 'draw')
            coords = (np.arange(nchains), np.arange(ndraws))
        elif len(val.shape) == 3:
            val = val.T
            nchains, nlf, ndraws = val.shape
            dims = ('chain', 'leapfrog', 'draw')
            coords = (np.arange(nchains), np.arange(nlf), np.arange(ndraws))
        else:
            print(f'val.shape: {val.shape}')
            raise ValueError('Invalid shape encountered')

        assert coords is not None and dims is not None
        dset[key] = xr.DataArray(val, dims=dims, coords=tuple(coords))

    return xr.Dataset(dset)


def plot_dataset(
        dataset: xr.Dataset,
        nchains: Optional[int] = 10,
        outdir: Optional[os.PathLike] = None,
        title: Optional[str] = None,
        job_type: Optional[str] = None,
        run: Optional[Any] = None,
        arun: Optional[Any] = None,
        # run: Any = None,
) -> None:
    outdir = Path(outdir) if outdir is not None else Path(os.getcwd())
    outdir.mkdir(exist_ok=True, parents=True)
    # outdir = outdir.joinpath('plots')
    job_type = job_type if job_type is not None else f'job-{get_timestamp()}'
    for key, val in dataset.data_vars.items():
        if key == 'x':
            continue

        # fig, ax = plt.subplots()
        # _ = val.plot(ax=ax)  # type: ignore
        # xdir = outdir.joinpath('xarr_plots')
        # xdir.mkdir(exist_ok=True, parents=True)
        # fig = save_figure(fig=fig, key=key, outdir=xdir)
        # if arun is not None:
        #     from aim import Figure, Run
        #     assert isinstance(arun, Run)
        #     afig = Figure(fig)
        #     arun.track(afig, name=f'figures/{key}_xarr',
        #                context={'subset': job_type})

        fig, _, _ = plot_dataArray(val,
                                   key=key,
                                   outdir=outdir,
                                   title=title,
                                   line_labels=False,
                                   num_chains=nchains)
        _ = save_figure(fig=fig, key=key, outdir=outdir)

    _ = make_ridgeplots(dataset,
                        outdir=outdir,
                        drop_nans=True,
                        drop_zeros=False,
                        num_chains=nchains)


def analyze_dataset(
        dataset: xr.Dataset,
        outdir: os.PathLike,
        nchains: Optional[int] = None,
        title: Optional[str] = None,
        job_type: Optional[str] = None,
        save: Optional[bool] = True,
        run: Optional[Any] = None,
        arun: Optional[Any] = None,
        use_hdf5: Optional[bool] = True,
):
    job_type = job_type if job_type is not None else f'job-{get_timestamp()}'
    dirs = make_subdirs(outdir)
    if nchains is not None and nchains > 1024:
        nchains_ = nchains // 4
        log.warning(
            f'Reducing `nchains` from: {nchains} -> {nchains_} for plotting'
        )

    plot_dataset(dataset,
                 nchains=nchains,
                 title=title,
                 job_type=job_type,
                 outdir=dirs['plots'])
    if save:
        try:
            datafile = save_dataset(dataset,
                                    use_hdf5=use_hdf5,
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
        if job_type is not None:
            pngdir = Path(dirs['plots']).joinpath('pngs')
            if run is not None:
                name = f'{job_type}-{run.id}'
                artifact = wandb.Artifact(name=name, type='result')

                artifact.add_dir(pngdir.as_posix(), name=f'{job_type}/plots')
                if datafile is not None:
                    artifact.add_file(
                        datafile.as_posix(),
                        name=f'{job_type}/data'
                    )

                run.log_artifact(artifact)

            if arun is not None:
                from aim import Image
                for f in list(pngdir.rglob('*.png')):
                    aimage = Image(
                        Path(f).as_posix(),
                        format='png',
                        quality=100,
                    )
                    arun.track(
                        aimage,
                        name=f'images/{f.stem}',
                        context={'subset': job_type}
                    )

    return dataset


def save_and_analyze_data(
        dataset: xr.Dataset,
        outdir: os.PathLike,
        nchains: Optional[int] = None,
        run: Optional[Any] = None,
        arun: Optional[Any] = None,
        output: Optional[dict] = None,
        job_type: Optional[str] = None,
        framework: Optional[str] = None,
) -> xr.Dataset:
    jstr = f'{job_type}'
    output = {} if output is None else output
    title = (
        jstr if framework is None
        else ': '.join([jstr, f'{framework}'])
    )

    set_plot_style()
    dataset = analyze_dataset(dataset,
                              run=run,
                              arun=arun,
                              save=True,
                              outdir=outdir,
                              nchains=nchains,
                              job_type=job_type,
                              title=title)
    if not is_interactive():
        edir = Path(outdir).joinpath('logs')
        edir.mkdir(exist_ok=True, parents=True)
        log.info(f'Saving {job_type} logs to: {edir.as_posix()}')
        save_logs(run=run,
                  logdir=edir,
                  job_type=job_type,
                  tables=output.get('tables', None),
                  summaries=output.get('summaries'))

    return dataset
