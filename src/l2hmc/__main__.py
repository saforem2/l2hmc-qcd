# -*- coding: utf-8 -*-
"""
main.py

Contains entry point for training Dynamics.
"""
from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals
)
import logging
import os
import wandb
# from ezpz import get_rank
# import sys

import time
from pathlib import Path
from mpi4py import MPI
import json

import hydra
from typing import Optional
from omegaconf.dictconfig import DictConfig
# from l2hmc import get_logger

# sys.setdefaultencoding('utf8')

# os.environ['WANDB_SILENT'] = '1'
comm = MPI.COMM_WORLD

# from l2hmc import logger
# logger = logging.getLogger(__name__)
# log = get_logger(__name__)
# log = logging.getLogger()
# _ = get_logger('root').setLevel(logging.INFO)
# _ = get_logger('wandb').setLevel(logging.CRITICAL)
_ = logging.getLogger('wandb').setLevel(logging.INFO)
_ = logging.getLogger('aim').setLevel(logging.INFO)
_ = logging.getLogger('filelock').setLevel(logging.CRITICAL)
_ = logging.getLogger('matplotlib').setLevel(logging.INFO)
_ = logging.getLogger('PIL.PngImagePlugin').setLevel(logging.CRITICAL)
_ = logging.getLogger('graphviz._tools').setLevel(logging.CRITICAL)
_ = logging.getLogger('graphviz').setLevel(logging.CRITICAL)
_ = logging.getLogger('deepspeed').setLevel(logging.INFO)
_ = logging.getLogger('opinionated').setLevel(logging.INFO)

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def get_experiment(
        cfg: DictConfig,
        keep: Optional[str | list[str]] = None,
        skip: Optional[str | list[str]] = None,
):
    framework = cfg.get('framework', None)
    os.environ['RUNDIR'] = os.getcwd()
    if framework in ['tf', 'tensorflow']:
        cfg.framework = 'tensorflow'
        # from l2hmc.utils.dist import setup_tensorflow
        from ezpz import setup_tensorflow
        _ = setup_tensorflow(cfg.precision)
        from l2hmc.experiment.tensorflow.experiment import Experiment
        experiment = Experiment(
            cfg,
            keep=keep,
            skip=skip
        )
        return experiment

    elif framework in ['pt', 'pytorch', 'torch']:
        import torch
        cfg.framework = 'pytorch'
        from ezpz import setup_torch
        _ = setup_torch(
            seed=cfg.seed,
            # precision=cfg.precision,
            backend=cfg.get('backend', 'DDP'),
            port=cfg.get('port', '2345')
        )
        precision = cfg.get('precision', None)
        if precision is not None and precision in {
                'float64', 'fp64', 'f64', '64', 'double',
        }:
            LOG.warning(f'setting default dtype: {precision}')
            torch.set_default_dtype(torch.float64)
        from l2hmc.experiment.pytorch.experiment import Experiment
        experiment = Experiment(cfg, keep=keep, skip=skip)
        return experiment

    raise ValueError(
        'Framework must be specified, one of: [pytorch, tensorflow]'
    )


def run(
        cfg: DictConfig,
        overrides: Optional[list[str]] = None
) -> str:
    from l2hmc.utils.plot_helpers import set_plot_style
    set_plot_style()
    import matplotlib.pyplot as plt
    import opinionated
    plt.style.use(opinionated.STYLES['opinionated_min'])
    # --- [0.] Setup ------------------------------------------------------
    if overrides is not None:
        from l2hmc.configs import get_config
        cfg.update(get_config(overrides))
    ex = get_experiment(cfg)
    if ex.trainer._is_orchestrator:
        try:
            from omegaconf import OmegaConf
            from rich import print_json
            conf = OmegaConf.structured(ex.config)
            cdict = OmegaConf.to_container(conf)
            print_json(json.dumps(cdict))
        except Exception as e:
            LOG.exception(e)
            LOG.warning('Continuing!')
    should_train: bool = (
        ex.config.steps.nera > 0
        and ex.config.steps.nepoch > 0
    )
    nchains_eval = max(2, int(ex.config.dynamics.xshape[0] // 4))
    # TODO -----------------------------------------------------------------
    # - [ ] Add logic for running distributed inference + HMC
    #     - [ ] If we're training across N devices (CPU, GPU, TPU),
    #           we would like to run an indepdent evaluation + HMC process
    #           on each of them, average the model improvement over these
    # ----------------------------------------------------------------------
    # --- [1.] Train model -------------------------------------------------
    if should_train:
        tstart = time.time()
        _ = ex.train()
        LOG.info(f'Training took: {time.time() - tstart:.5f}s')
        # --- [2.] Evaluate trained model ----------------------------------
        if ex.trainer._is_orchestrator and ex.config.steps.test > 0:
            LOG.info('Evaluating trained model')
            estart = time.time()
            _ = ex.evaluate(job_type='eval', nchains=nchains_eval)
            LOG.info(f'Evaluation took: {time.time() - estart:.5f}s')
    # --- [3.] Run generic HMC for comparison ------------------------------
    if ex.trainer._is_orchestrator and ex.config.steps.test > 0:
        LOG.info('Running generic HMC for comparison')
        hstart = time.time()
        _ = ex.evaluate(job_type='hmc', nchains=nchains_eval)
        LOG.info(f'HMC took: {time.time() - hstart:.5f}s')
        from l2hmc.utils.plot_helpers import measure_improvement
        # try:
        improvement = measure_improvement(
            experiment=ex,
            title=f'{ex.config.framework}',
        )
        # except ValueError:
        #     import pudb; pudb.set_trace()
        # improvement = comm.gather(improvement, root=0)
        if ex.config.init_wandb and wandb.run is not None:
            wandb.run.log({'model_improvement': improvement})
            # wandb.save(base_path=os.getcwd())
            # logfile = Path(os.getcwd()).joinpath('__main__.log')
            # if logfile.is_file():
            # log.info(f'Uploading {os.getcwd()} to wandb.')
            # wandb.run.upload_file(logfile.as_posix())
            # dirs_ = {
            #     '.hydra',
            #     'eval',
            #     'train',
            #     'hmc',
            #     'network_diagrams',
            #     'pngs',
            # }
            # files_ = {
            #     '__main__.log',
            #     'plots.txt'
            #     'model_improvement.svg',
            #     'model_improvement.txt',
            # }
            # files = {rundir.joinpath(i) for i in files_}
            # dirs = {rundir.joinpath(i) for i in dirs_}
            # for f in files:
            #     if f.is_file():
            #         log.info(f'Uploading {f.as_posix()} to wandb')
            #         artifact.add_file(f.as_posix())
            # for d in dirs:
            #     if d.is_dir():
            #         log.info(f'Uploading {d.as_posix()} to wandb')
            #         artifact.add_dir(d)

            # else:
            #     log.info(f'Unable to find {logfile.as_posix()}')
            # if ex.run is not None and ex.run is wandb.run:
        LOG.critical(f'Model improvement: {improvement:.8f}')
        if wandb.run is not None:
            LOG.critical(f'🚀 {wandb.run.name}')
            LOG.critical(f'🔗 {wandb.run.url}')
            LOG.critical(f'📂/: {wandb.run.dir}')
            artifact = wandb.Artifact('logdir', type='directory')
            rundir = Path(os.getcwd())
            # contents = [
            #     rundir.joinpath(i) for i in os.listdir(rundir.as_posix())
            #     if 'wandb' not in i and
            # ]
            # for c in contents:
            #     if 'wandb' in c.as_posix():
            #         continue
            #     if c.is_dir():
            #         try:
            #             artifact.add_dir(c.as_posix())
            #         except ValueError:
            #             LOG.exception(f'Unable to add_dir: {c.as_posix()}')
            #             continue
            #     elif c.is_file():
            #         try:
            #             artifact.add_file(c.as_posix())
            #         except ValueError:
            #             LOG.exception(f'Unable to add_dir: {c.as_posix()}')
            #             continue
            dirs_ = ('pngs', 'network_diagrams', '.hydra')
            files_ = (
                '__main__.log',
                'main_debug.log',
                'main.log',
                'model_improvement.svg',
                'model_improvement.txt',
                'plots.txt',
            )
            for file in files_:
                if (fpath := rundir.joinpath(file)).is_file():
                    LOG.info(f'Adding {file} to W&B artifact...')
                    artifact.add_file(fpath.as_posix())
            for dir_ in dirs_:
                if (dpath := rundir.joinpath(dir_)).is_dir():
                    LOG.info(f'Adding {dir_} to W&B artifact...')
                    artifact.add_dir(dpath.as_posix())
            # artifact.remove(rundir.joinpath('wandb').as_posix())
            LOG.info(f'Logging {artifact} to  W&B')
            wandb.run.log_artifact(artifact)
    if ex.trainer._is_orchestrator:
        try:
            ex.visualize_model()
        except Exception:
            # log.exception(e)
            LOG.error('Unable to make visuals for model, continuing!')
        LOG.critical(f"experiment dir: {Path(ex._outdir).as_posix()}")
    return Path(ex._outdir).as_posix()


def build_experiment(overrides: Optional[str | list[str]] = None):
    import warnings
    warnings.filterwarnings('ignore')
    from l2hmc.configs import get_config
    if isinstance(overrides, str):
        overrides = [overrides]
    cfg = get_config(overrides)
    return get_experiment(cfg=cfg)


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig):
    output = run(cfg)
    fw = cfg.get('framework', None)
    be = cfg.get('backend', None)
    # comm.Barrier()
    if (
            str(fw).lower() in {'pt', 'torch', 'pytorch'}
            and str(be).lower() == 'ddp'
    ):
        from l2hmc.utils.dist import cleanup
        cleanup()
    return output


if __name__ == '__main__':
    import sys
    # import warnings
    # warnings.filterwarnings('ignore')
    # import wandb
    # from ezpz import get_rank
    # wandb.require(experiment='service')
    start = time.time()
    outdir = main()
    end = time.time()
    LOG.info(f'Run completed in: {end - start:4.4f} s')
    if outdir is not None:
        LOG.info(f'Run located in: {outdir}')
    if wandb.run is not None:
        wandb.finish()
    # if get_rank() == 0:
    #     os._exit(0)
    sys.exit(0)
