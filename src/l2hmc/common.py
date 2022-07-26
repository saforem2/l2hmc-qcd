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
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import pandas as pd
from rich.table import Table
import wandb
import xarray as xr

from l2hmc.configs import AnnealingSchedule, Steps
from l2hmc.configs import OUTPUTS_DIR
from l2hmc.utils.plot_helpers import make_ridgeplots, plot_dataArray, set_plot_style
from l2hmc.utils.rich import get_console, is_interactive

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


def dict_from_h5pyfile(hfile: os.PathLike) -> dict:
    f = h5py.File(hfile, 'r')
    data = {key: f[key] for key in list(f.keys())}
    f.close()
    return data


def dataset_from_h5pyfile(hfile: os.PathLike) -> xr.Dataset:
    f = h5py.File(hfile, 'r')
    data = {key: f[key] for key in list(f.keys())}
    f.close()

    return xr.Dataset(data)


def load_job_data(
        logdir: os.PathLike,
        jobtype: str
) -> xr.Dataset:
    assert jobtype in ['train', 'eval', 'hmc']
    fpath = Path(logdir).joinpath(
        f'{jobtype}',
        'data',
        f'{jobtype}_data.h5',
    )
    assert fpath.is_file()
    return dataset_from_h5pyfile(fpath)




def load_time_data(
        logdir: os.PathLike,
        jobtype: str
) -> pd.DataFrame:
    assert jobtype in ['train', 'eval', 'hmc']
    fpaths = Path(logdir).rglob(f'step-timer-{jobtype}')
    data = {}
    for idx, fpath in enumerate(fpaths):
        tdata = pd.read_csv(fpath)
        data[f'{idx}'] = tdata

    return pd.DataFrame(data)


def _load_from_dir(
        logdir: os.PathLike,
        to_load: str,
) -> xr.Dataset | pd.DataFrame:
    if to_load in ['train', 'eval', 'hmc']:
        return load_job_data(logdir=logdir, jobtype=to_load)
    if to_load in['time', 'timing']:
        return load_time_data(logdir, jobtype=to_load)
    raise ValueError('Unexpected argument for `to_load`')


def load_from_dir(
        logdir: os.PathLike,
        to_load: str | list[str]
) -> dict[str, xr.Dataset]: 
    assert to_load in ['train', 'eval', 'hmc', 'time', 'timing']
    data = {}
    if isinstance(to_load, list):
        for i in to_load:
            data[i] = _load_from_dir(logdir, to_load)
    elif isinstance(to_load, str):
        data[to_load] = _load_from_dir(logdir, to_load)

    return data


def latvolume_to_str(latvolume: list[int]):
    return 'x'.join([str(i) for i in latvolume])



def check_nonempty(fpath: os.PathLike) -> bool:
    return (
        Path(fpath).is_dir()
        and len(os.listdir(fpath)) > 0
    )


def check_jobdir(fpath: os.PathLike) -> bool:
    jobdir = Path(fpath)
    pdir = jobdir.joinpath('plots')
    ddir = jobdir.joinpath('data')
    ldir = jobdir.joinpath('logs')
    return (
        check_nonempty(pdir)
        and check_nonempty(ddir)
        and check_nonempty(ldir)
    )


def check_if_logdir(fpath: os.PathLike) -> bool:
    logdir = Path(fpath)
    tdir = Path(logdir).joinpath('train')
    edir = Path(logdir).joinpath('eval')
    hdir = Path(logdir).joinpath('hmc')
    return (
        (check_nonempty(tdir) and check_jobdir(tdir))
        and (check_nonempty(edir) and check_jobdir(edir))
        and (check_nonempty(hdir) and check_jobdir(hdir))
    )


def check_if_matching_logdir(
        fpath: os.PathLike,
        config_str: str,
) -> bool:
    return (
        check_if_logdir(fpath)
        and config_str in Path(fpath).as_posix()
    )


def find_logdirs(rootdir) -> list[Path]:
    logdirs = []
    # for path in Path(rootdir).iterdir():
    for root, dirs, files in os.walk(rootdir):
        for dir in dirs:
            if check_if_logdir(dir):
                logdirs.append(dir)
        # is_logdir = check_if_logdir(path)
        # if is_logdir:
        #     logdirs.append(path)
        # elif Path(path).is_dir():
        #     find_logdirs(path)
        # else:
        #     continue
    #     # for subdir in dirs:
    #     if check_if_logdir(path):
    #         logdirs.append(path)

    return logdirs


def find_matching_logdirs(
        rootdir: Optional[os.PathLike] = None,
        beta: Optional[float] = None,
        group: Optional[str] = None,
        nlf: Optional[int] = None,
        merge_directions: Optional[bool] = None,
        framework: Optional[str] = None,
        latvolume: Optional[list[int]] = None,
):
    if rootdir is None:
        rdir = Path(OUTPUTS_DIR).joinpath('runs')
    else:
        rdir = Path(rootdir)

    logdirs = rdir.rglob(f'beta-{beta:.1f}')
    # if beta is not None:
    #     logdirs = [
    #         i for i in logdirs
    #         if f'beta-{beta:.1f}' in Path(i).as_posix()
    #     ]

    if group is not None:
        logdirs = [
            i for i in logdirs
            if group in Path(i).as_posix()
        ]

    if nlf is not None:
        logdirs = [
            i for i in logdirs
            if f'nlf-{nlf}' in Path(i).as_posix()
        ]

    if merge_directions is not None:
        logdirs = [
            i for i in logdirs
            if f'merge_directions-{merge_directions}' in Path(i).as_posix()
        ]

    if framework is not None:
        logdirs = [
            i for i in logdirs
            if framework in Path(i).as_posix()
        ]

    if latvolume is not None:
        logdirs = [
            i for i in logdirs
            if 'x'.join([str(i) for i in latvolume]) in Path(i).as_posix()
        ]

    return logdirs


def filter_runs_by(
        beta: float,
        group: Optional[str] = None,
        nlf: Optional[int] = None,
        merge_directions: Optional[bool] = None,
        framework: Optional[str] = None,
        latvolume: Optional[list[int]] = None,
) -> list[Path]:
    """Filter runs matching specified values.

    Directory structure looks like:

    U1/
    └─ 16x16/
        └─ nlf-8/
            └─ beta-4.0/
                └─ merge_directions-True/
                    ├─ pytorch/
                    │   └─ 2022-07-08/
                    │       ├─ 19-56-30/
                    │       │   ├─ train/
                    │       │   ├─ eval/
                    │       │   ├─ hmc/
                    ├─ tensorflow/
                    │   └─ 2022-07-08/
                    │       ├─ 19-57-05/
                    │       │   ├─ train/
                    │       │   ├─ eval/
                    │       │   ├─ hmc/
    """
    runs_path = Path(OUTPUTS_DIR).joinpath('runs')
    group = 'U1' if group is None else group
    latvolume = [16, 16] if latvolume is None else latvolume
    latstr = 'x'.join([str(i) for i in latvolume])

    bstr = f'beta-{beta:.1f}'
    latdirs = runs_path.joinpath(
        group,
        latstr,
    )
    if nlf is None:
        lfdirs = [
            i for i in os.listdir(latdirs)
            if Path(i).is_dir()
        ]
    else:
        lfdirs = [
            latdirs.joinpath(f'nlf-{nlf}')
        ]

    beta_dirs = []
    for lfdir in lfdirs:
        if (
                bstr in Path(lfdir).as_posix()
                and Path(lfdir).is_dir()
        ):
            beta_dirs.append(lfdir)

    mdirs = []
    if merge_directions is not None:
        for bdir in beta_dirs:
            if (
                    f'merge_directions-{merge_directions}' in bdir
                    and Path(bdir).is_dir()
            ):
                mdirs.append(bdir)






def find_runs_with_matching_options(
        config: dict[str, Any],
        load: Optional[str]
) -> list[Path]:
    """Find runs with options matching those specified in `config`."""
    runs_path = Path(OUTPUTS_DIR).joinpath('runs')
    config_files = runs_path.rglob('config.yaml')
    matches = []
    for f in config_files:
        fpath = Path(f)
        assert fpath.is_file()
        loaded = OmegaConf.load(f)
        checks = []
        for key, val in config.items():
            checks.append((val == loaded.get(key, None)))

        if sum(matches) == len(matches):
            matches.append(fpath)

    return matches


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
        logdir: Optional[os.PathLike] = None,
        run: Optional[Any] = None
) -> None:
    job_type = 'job' if job_type is None else job_type
    if logdir is None:
        logdir = Path(os.getcwd()).joinpath('logs')
    else:
        logdir = Path(logdir)

    # cfile = logdir.joinpath('console.txt').as_posix()
    # text = console.export_text(clear=False)
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
    console = get_console(record=True, width=235)
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
        # with open(hfile.as_posix(), 'r') as f:
        #     html = f.read()

        # run.log({f'Media/{job_type}': wandb.Html(html)})
        run.log({
            f'DataFrames/{job_type}': wandb.Table(data=df)
        })

    if summaries is not None:
        sfile = logdir.joinpath('summaries.txt').as_posix()
        with open(sfile, 'a') as f:
            f.write('\n'.join(summaries))


def make_subdirs(basedir: os.PathLike):
    dirs = {}
    assert Path(basedir).is_dir()
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
            if pngdir.is_dir():
                if run is not None:
                    name = f'{job_type}-{run.id}'
                    artifact = wandb.Artifact(
                        name=name,
                        type='result'
                    )

                    artifact.add_dir(
                        pngdir.as_posix(),
                        name=f'{job_type}/plots'
                    )
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
