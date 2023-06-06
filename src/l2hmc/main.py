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

import time
from pathlib import Path
from mpi4py import MPI
import json

import hydra
from typing import Optional
from omegaconf.dictconfig import DictConfig
from l2hmc import get_logger

# sys.setdefaultencoding('utf8')

# os.environ['WANDB_SILENT'] = '1'
comm = MPI.COMM_WORLD

# from l2hmc import logger
# logger = logging.getLogger(__name__)
# log = get_logger(__name__)
# log = logging.getLogger()
_ = get_logger('wandb').setLevel(logging.CRITICAL)
_ = get_logger('aim').setLevel(logging.CRITICAL)
_ = get_logger('filelock').setLevel(logging.CRITICAL)
_ = get_logger('matplotlib').setLevel(logging.CRITICAL)
_ = get_logger('PIL.PngImagePlugin').setLevel(logging.CRITICAL)
_ = get_logger('graphviz._tools').setLevel(logging.CRITICAL)
_ = get_logger('graphviz').setLevel(logging.CRITICAL)
_ = get_logger('deepspeed').setLevel(logging.INFO)

log = get_logger(__name__)


def get_experiment(
        cfg: DictConfig,
        keep: Optional[str | list[str]] = None,
        skip: Optional[str | list[str]] = None,
):
    framework = cfg.get('framework', None)
    os.environ['RUNDIR'] = str(os.getcwd())
    if framework in ['tf', 'tensorflow']:
        cfg.framework = 'tensorflow'
        from l2hmc.utils.dist import setup_tensorflow
        _ = setup_tensorflow(cfg.precision)
        from l2hmc.experiment.tensorflow.experiment import Experiment
        experiment = Experiment(
            cfg,
            keep=keep,
            skip=skip
        )
        return experiment

    elif framework in ['pt', 'pytorch', 'torch']:
        cfg.framework = 'pytorch'
        from l2hmc.utils.dist import setup_torch
        _ = setup_torch(
            seed=cfg.seed,
            # precision=cfg.precision,
            backend=cfg.get('backend', 'DDP'),
            port=cfg.get('port', '2345')
        )
        from l2hmc.experiment.pytorch.experiment import Experiment
        experiment = Experiment(cfg, keep=keep, skip=skip)
        return experiment

    raise ValueError(
        'Framework must be specified, one of: [pytorch, tensorflow]'
    )


def run(cfg: DictConfig, overrides: Optional[list[str]] = None) -> str:
    from l2hmc.utils.plot_helpers import set_plot_style
    set_plot_style()
    # --- [0.] Setup ------------------------------------------------------
    if overrides is not None:
        from l2hmc.configs import get_config
        cfg.update(get_config(overrides))
    ex = get_experiment(cfg)
    if ex.trainer._is_chief:
        try:
            from omegaconf import OmegaConf
            from rich import print_json
            conf = OmegaConf.structured(ex.config)
            cdict = OmegaConf.to_container(conf)
            print_json(json.dumps(cdict))
        except Exception as e:
            log.exception(e)
            log.warning('Continuing!')
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
        log.info(f'Training took: {time.time() - tstart:.5f}s')
        # --- [2.] Evaluate trained model ----------------------------------
        if ex.trainer._is_chief and ex.config.steps.test > 0:
            log.info('Evaluating trained model')
            estart = time.time()
            _ = ex.evaluate(job_type='eval', nchains=nchains_eval)
            log.info(f'Evaluation took: {time.time() - estart:.5f}s')
    # --- [3.] Run generic HMC for comparison ------------------------------
    if ex.trainer._is_chief and ex.config.steps.test > 0:
        log.info('Running generic HMC for comparison')
        hstart = time.time()
        _ = ex.evaluate(job_type='hmc', nchains=nchains_eval)
        log.info(f'HMC took: {time.time() - hstart:.5f}s')
        from l2hmc.utils.plot_helpers import measure_improvement
        improvement = measure_improvement(
            experiment=ex,
            title=f'{ex.config.framework}',
        )
        # improvement = comm.gather(improvement, root=0)
        if ex.config.init_wandb:
            if ex.run is not None and ex.run is wandb.run:
                ex.run.log({'model_improvement': improvement})
        log.critical(f'Model improvement: {improvement:.8f}')
        if wandb.run is not None:
            log.critical(f'🚀 {wandb.run}')
            log.critical(f'🔗 {wandb.run.url}')
            log.critical(f'📂/: {wandb.run.dir}')
    if ex.trainer._is_chief:
        try:
            ex.visualize_model()
        except Exception:
            # log.exception(e)
            log.error('Unable to make visuals for model, continuing!')
        log.critical(f"experiment dir: {Path(ex._outdir).as_posix()}")
    return Path(ex._outdir).as_posix()


def build_experiment(overrides: Optional[str | list[str]] = None):
    import warnings
    warnings.filterwarnings('ignore')
    from l2hmc.configs import get_config
    if isinstance(overrides, str):
        overrides = [overrides]
    cfg = get_config(overrides)
    exp = get_experiment(cfg=cfg)
    return exp


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig):
    output = run(cfg)
    fw = cfg.get('framework', None)
    be = cfg.get('backend', None)
    if (
            str(fw).lower() in ['pt', 'torch', 'pytorch']
            and str(be).lower() == 'ddp'
    ):
        from l2hmc.utils.dist import cleanup
        cleanup()
    return output


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    import wandb
    # wandb.require(experiment='service')
    start = time.time()
    outdir = main()
    end = time.time()
    log.info(f'Run completed in: {end - start:4.4f} s')
    if outdir is not None:
        log.info(f'Run located in: {outdir}')
    if wandb.run is not None:
        wandb.finish(0)
    # sys.exit(0)
