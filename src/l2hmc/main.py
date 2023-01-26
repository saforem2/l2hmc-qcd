"""
main.py

Contains entry point for training Dynamics.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
import sys
import warnings
import time
from pathlib import Path
from mpi4py import MPI

import hydra
from typing import Optional
from omegaconf.dictconfig import DictConfig

import json
from l2hmc.configs import ExperimentConfig
from l2hmc.utils.rich import print_config
# from l2hmc.utils.logger import get_pylogger
from l2hmc.utils.plot_helpers import set_plot_style

warnings.filterwarnings('ignore')
set_plot_style()

log = logging.getLogger()
logging.getLogger('wandb').setLevel(logging.ERROR)
logging.getLogger('aim').setLevel(logging.ERROR)
logging.getLogger('filelock').setLevel(logging.CRITICAL)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.CRITICAL)
logging.getLogger('graphviz._tools').setLevel(logging.CRITICAL)
logging.getLogger('graphviz').setLevel(logging.CRITICAL)

comm = MPI.COMM_WORLD


def get_experiment(
        cfg: DictConfig,
        keep: Optional[str | list[str]] = None,
        skip: Optional[str | list[str]] = None,
):
    framework = cfg.get('framework', None)
    os.environ['RUNDIR'] = str(os.getcwd())
    if framework in ['tf', 'tensorflow']:
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
        from l2hmc.utils.dist import setup_torch
        _ = setup_torch(
            seed=cfg.seed,
            precision=cfg.precision,
            backend=cfg.get('backend', 'horovod'),
            port=cfg.get('port', '2345')
        )
        from l2hmc.experiment.pytorch.experiment import Experiment
        experiment = Experiment(cfg, keep=keep, skip=skip)
        return experiment

    raise ValueError(
        'Framework must be specified, one of: [pytorch, tensorflow]'
    )


def run(cfg: DictConfig, overrides: Optional[list[str]] = None) -> str:
    # --- [0.] Setup ------------------------------------------------------
    if overrides is not None:
        from l2hmc.configs import get_config
        cfg.update(get_config(overrides))

    ex = get_experiment(cfg)
    # assert isinstance(ex.config, ExperimentConfig)

    if ex.trainer._is_chief:
        # from rich import print
        # log.info(ex.cfg)
        try:
            from omegaconf import OmegaConf
            from rich import print_json
            # print_json(ex.config.to_json())
            conf = OmegaConf.structured(ex.config)
            cdict = OmegaConf.to_container(conf)
            print_json(json.dumps(cdict))
        except Exception as e:
            log.exception(e)
            log.warning('Continuing!')
        # print(ex.config)
        # print_config(ex.cfg, resolve=True)

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

    try:
        ex.visualize_model()
    except Exception as e:
        log.exception(e)

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
    import wandb
    wandb.require(experiment='service')
    start = time.time()
    outdir = main()
    end = time.time()
    log.info(f'Run completed in: {end - start:4.4f} s')
    if outdir is not None:
        log.info(f'Run located in: {outdir}')
    sys.exit(0)
