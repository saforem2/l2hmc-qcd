"""
l2hmc/common.py

Contains methods intended to be shared across frameworks.
"""
from __future__ import absolute_import, annotations, division, print_function
import datetime
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pandas as pd
from rich.table import Table
import tensorflow as tf
import tensorflow.python.framework.ops as ops
import torch
import wandb
import xarray as xr

from l2hmc.configs import AnnealingSchedule, Steps
from l2hmc.configs import OUTPUTS_DIR
from l2hmc.configs import State
from l2hmc.utils.plot_helpers import (
    make_ridgeplots, plot_dataArray, set_plot_style
)
from l2hmc.utils.rich import get_console, is_interactive

os.environ['AUTOGRAPH_VERBOSITY'] = '0'

log = logging.getLogger(__name__)

# TensorLike = tf.Tensor | ops.EagerTensor | torch.Tensor | np.ndarray
ScalarLike = Union[int, float, bool, np.floating]


def grab_tensor(x: Any) -> np.ndarray | ScalarLike:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (tf.Tensor, ops.EagerTensor)):
        assert (
            hasattr(x, 'numpy')
            and callable(getattr(x, 'numpy'))
        )
        return x.numpy()  # type:ignore

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    else:
        return x


def clear_cuda_cache():
    import gc
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
        torch.clear_autocast_cache()


def get_timestamp(fstr=None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


def check_diff(x, y, name: Optional[str] = None):
    if isinstance(x, State):
        xd = {'x': x.x, 'v': x.v, 'beta': x.beta}
        yd = {'x': y.x, 'v': y.v, 'beta': y.beta}
        check_diff(xd, yd, name='State')

    elif isinstance(x, dict) and isinstance(y, dict):
        for (kx, vx), (ky, vy) in zip(x.items(), y.items()):
            if kx == ky:
                check_diff(vx, vy, name=kx)
            else:
                log.warning('Mismatch encountered!')
                log.warning(f'kx: {kx}')
                log.warning(f'ky: {ky}')
                vy_ = y.get(kx, None)
                if vy_ is not None:
                    check_diff(vx, vy_, name=kx)
                else:
                    log.warning(f'{kx} not in y, skipping!')
                    continue

    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        assert len(x) == len(y)
        for idx in range(len(x)):
            check_diff(x[idx], y[idx], name=f'{name}, {idx}')

    else:
        x = grab_tensor(x)
        y = grab_tensor(y)
        dstr = []
        if name is not None:
            dstr.append(f"'{name}''")

        dstr.append(f'  sum(diff): {np.sum(x - y)}')
        dstr.append(f'  min(diff): {np.min(x - y)}')
        dstr.append(f'  max(diff): {np.max(x - y)}')
        dstr.append(f'  mean(diff): {np.mean(x - y)}')
        dstr.append(f'  std(diff): {np.std(x - y)}')
        dstr.append(f'  np.allclose: {np.allclose(x, y)}')
        log.info('\n'.join(dstr))


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
    sched.setup(
        nera=steps.nera,
        nepoch=steps.nepoch,
        beta_init=beta_init,
        beta_final=beta_final
    )
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
    if to_load in ['time', 'timing']:
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
    contents = os.listdir(logdir)
    contents = os.listdir(logdir)
    in_contents = (
        'train' in contents
        and 'eval' in contents
        and 'hmc' in contents
    )
    non_empty = (
        check_nonempty(logdir.joinpath('train'))
        and check_nonempty(logdir.joinpath('eval'))
        and check_nonempty(logdir.joinpath('hmc'))
    )
    return (in_contents and non_empty)


def check_if_matching_logdir(
        fpath: os.PathLike,
        config_str: str,
) -> bool:
    return (
        check_if_logdir(fpath)
        and config_str in Path(fpath).as_posix()
    )


def find_logdirs(rootdir: os.PathLike) -> list[Path]:
    """Every `logdir` should contain a `config_tree.log` file."""
    return [
        Path(i).parent
        for i in Path(rootdir).rglob('config_tree.log')
        if check_if_logdir(Path(i).parent)
    ]


def _match_beta(logdir, beta: Optional[float] = None) -> bool:
    return (
        beta is not None
        and f'beta-{beta:.1f}' in Path(logdir).as_posix()
    )


def _match_group(logdir, group: Optional[str] = None) -> bool:
    return (
        group is not None
        and group in Path(logdir).as_posix()
    )


def _match_nlf(logdir, nlf: Optional[int] = None) -> bool:
    return (
        nlf is not None
        and f'nlf-{nlf}' in Path(logdir).as_posix()
    )


def _match_merge_directions(
        logdir,
        merge_directions: Optional[bool] = None
) -> bool:
    return (
        merge_directions is not None
        and (
            f'merge_directions-{merge_directions}'
            in Path(logdir).as_posix()
        )
    )


def _match_framework(
        logdir: os.PathLike,
        framework: Optional[str] = None,
) -> bool:
    return (
        framework is not None
        and framework in Path(logdir).as_posix()
    )


def _match_latvolume(
        logdir: os.PathLike,
        latvolume: Optional[list[int]] = None
) -> bool:
    return (
        latvolume is not None
        and (
            'x'.join([str(i) for i in latvolume])
            in Path(logdir).as_posix()
        )
    )


def filter_logdirs(
        logdirs: list,
        beta: Optional[float] = None,
        group: Optional[str] = None,
        nlf: Optional[int] = None,
        merge_directions: Optional[bool] = None,
        framework: Optional[str] = None,
        latvolume: Optional[list[int]] = None,
) -> list[os.PathLike]:
    """Filter logdirs by criteria."""
    matches = []

    for logdir in logdirs:
        if _match_beta(logdir, beta):
            matches.append(logdir)
        if _match_group(logdir, group):
            matches.append(logdir)
        if _match_nlf(logdir, nlf):
            matches.append(logdir)
        if _match_merge_directions(logdir, merge_directions):
            matches.append(logdir)
        if _match_framework(logdir, framework):
            matches.append(logdir)
        if _match_latvolume(logdir, latvolume):
            matches.append(logdir)

    return matches


def find_matching_logdirs(
        rootdir: os.PathLike,
        beta: Optional[float] = None,
        group: Optional[str] = None,
        nlf: Optional[int] = None,
        merge_directions: Optional[bool] = None,
        framework: Optional[str] = None,
        latvolume: Optional[list[int]] = None,
):
    logdirs = find_logdirs(rootdir)
    return filter_logdirs(
        logdirs,
        beta=beta,
        group=group,
        nlf=nlf,
        merge_directions=merge_directions,
        framework=framework,
        latvolume=latvolume
    )


def find_runs_with_matching_options(
        config: dict[str, Any],
        rootdir: Optional[os.PathLike] = None,
        # load: Optional[str]
) -> list[Path]:
    """Find runs with options matching those specified in `config`."""
    if rootdir is None:
        rootdir = Path(OUTPUTS_DIR)

    config_files = [
        i.resolve() for i in Path(rootdir).rglob('*.yaml')
        if (i.is_file and i.name == 'config.yaml')
    ]
    matches = []
    for f in config_files:
        fpath = Path(f)
        assert fpath.is_file()
        loaded = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
        assert isinstance(loaded, dict)
        checks = []
        for key, val in config.items():
            if key in loaded and val == loaded.get(key, None):
                checks.append(1)
            else:
                checks.append(0)
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
) -> None:
    outdir = Path(outdir) if outdir is not None else Path(os.getcwd())
    outdir.mkdir(exist_ok=True, parents=True)
    # outdir = outdir.joinpath('plots')
    job_type = job_type if job_type is not None else f'job-{get_timestamp()}'
    names = ['rainbow', 'viridis_r', 'magma', 'mako', 'turbo', 'spectral']
    cmap = np.random.choice(names, replace=True)

    set_plot_style()
    _ = make_ridgeplots(
        dataset,
        outdir=outdir,
        drop_nans=True,
        drop_zeros=False,
        num_chains=nchains,
        cmap=cmap,
    )
    for key, val in dataset.data_vars.items():
        if key == 'x':
            continue

        fig, _, _ = plot_dataArray(
            val,
            key=key,
            outdir=outdir,
            title=title,
            line_labels=False,
            num_chains=nchains
        )
        _ = save_figure(fig=fig, key=key, outdir=outdir)


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
) -> xr.Dataset:
    """Save plot and analyze resultant `xarray.Dataset`."""
    job_type = job_type if job_type is not None else f'job-{get_timestamp()}'
    dirs = make_subdirs(outdir)
    if nchains is not None and nchains > 1000:
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


def avg_diff(
        y: list[float],
        x: Optional[list[float]] = None,
        *,
        drop: Optional[float | int] = None,
) -> float:
    # yarr = np.array(y)
    # xarr = None
    if x is not None:
        assert len(y) == len(x)
    #     xarr = np.array(x)

    if drop is not None:
        # If passed as an int, interpret as num to drop
        if isinstance(drop, int) and drop > 1.:
            n = drop
        elif isinstance(drop, float) and drop < 1.:
            n = int(drop * len(y))
        else:
            raise ValueError(
                '`drop` must either be an `int > 1` or `float < 1.`'
            )

        y = y[n:]
        if x is not None:
            x = x[n:]

    dy = np.subtract(y[1:], y[:-1]).mean()
    if x is None:
        return dy

    return dy / np.subtract(x[1:], x[:-1]).mean()
