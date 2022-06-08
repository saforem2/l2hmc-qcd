"""
main_pytorch.py

Contains entry-point for training and inference.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import DictConfig
import torch

from l2hmc.common import save_and_analyze_data
from l2hmc.configs import get_jobdir
from l2hmc.experiment.pytorch.experiment import Experiment
from l2hmc.trainers.pytorch.trainer import Trainer
from l2hmc.utils.pytorch.utils import get_summary_writer


log = logging.getLogger(__name__)


Tensor = torch.Tensor


# -------------------------------------------------------------
# TODO: Gather and separate duplicate file I/O related methods
# -------------------------------------------------------------

def evaluate(
        cfg: DictConfig,
        trainer: Trainer,
        job_type: str,
        run: Optional[Any] = None,
        nchains: Optional[int] = None,
        eps: Optional[float] = None,
        nleapfrog: Optional[int] = None,
) -> dict:
    """Evaluate model (nested as `trainer.model`)"""
    therm_frac = cfg.get('therm_frac', 0.2)

    assert isinstance(nchains, int)
    assert job_type in ['eval', 'hmc']
    jobdir = get_jobdir(cfg, job_type=job_type)
    # if trainer.accelerator.is_local_main_process:
    if trainer.rank == 0:
        writer = get_summary_writer(cfg, job_type=job_type)
    else:
        writer = None

    output = trainer.eval(
        run=run,
        writer=writer,
        nchains=nchains,
        job_type=job_type,
        eps=eps,
        nleapfrog=nleapfrog
    )
    dataset = output['history'].get_dataset(therm_frac=therm_frac)
    if run is not None:
        dQint = dataset.data_vars.get('dQint').values
        drop = int(0.1 * len(dQint))
        dQint = dQint[drop:]
        run.summary[f'dQint_{job_type}'] = dQint
        run.summary[f'dQint_{job_type}.mean'] = dQint.mean()

    _ = save_and_analyze_data(dataset,
                              run=run,
                              outdir=jobdir,
                              output=output,
                              nchains=nchains,
                              job_type=job_type,
                              framework='pytorch')

    return output


def train(
        cfg: DictConfig,
        trainer: Trainer,
        run: Optional[Any] = None,
        # writer: Optional[Any] = None,
        nchains: Optional[int] = None,
) -> dict:
    writer = None
    jobdir = get_jobdir(cfg, job_type='train')
    # if trainer.accelerator.is_local_main_process:
    if trainer.rank == 0:
        writer = get_summary_writer(cfg, job_type='train')

    # ------------------------------------------
    # NOTE: cfg.profile will be False by default
    # ------------------------------------------
    if cfg.profile:
        from torch.profiler import profile, ProfilerActivity  # type: ignore
        activities = [ProfilerActivity.CUDA, ProfilerActivity.CPU]
        with profile(record_shapes=True, activities=activities) as prof:
            assert cfg.steps.nepoch * cfg.steps.nera < 100
            output = trainer.train(run=run, writer=writer, train_dir=jobdir)

        log.info(prof.key_averages().table(sort_by="cpu_time_total"))
        tracefile = Path(os.getcwd()).joinpath('trace.json').as_posix()
        prof.export_chrome_trace(tracefile)

    else:
        output = trainer.train(run=run, writer=writer, train_dir=jobdir)

    # if trainer.accelerator.is_local_main_process:
    if trainer.rank == 0:
        dset = output['history'].get_dataset()
        nchains = int(
            min(cfg.dynamics.nchains, max(16, cfg.dynamics.nchains // 8))
        )
        _ = save_and_analyze_data(dset,
                                  run=run,
                                  outdir=jobdir,
                                  output=output,
                                  nchains=nchains,
                                  job_type='train',
                                  framework='pytorch')

    if writer is not None:
        writer.close()

    return output


# @record
def main(cfg: DictConfig) -> dict:
    outputs = {}
    # config = instantiate(cfg)
    experiment = Experiment(cfg)
    objs = experiment.build()
    run = objs['run']
    trainer = objs['trainer']

    # ----------------------------------------------------------
    # 1. Train model
    # 2. Evaluate trained model
    # 3. Run generic HMC as baseline w/ same trajectory length
    # ----------------------------------------------------------
    should_train = (cfg.steps.nera > 0 and cfg.steps.nepoch > 0)
    if should_train:
        # tw = experiment.get_summary_writer('train')
        outputs['train'] = train(cfg, trainer, run=run)             # [1.]

    if run is not None:
        run.unwatch(objs['dynamics'])

    # if trainer.accelerator.is_local_main_process:
    if trainer.rank == 0:
        nchains = min(cfg.dynamics.nchains, max(16, cfg.dynamics.nchains // 8))
        if should_train and cfg.steps.test > 0:                     # [2.]
            log.warning('Evaluating trained model')
            # ew = experiment.get_summary_writer('eval')
            outputs['eval'] = evaluate(cfg,
                                       run=run,
                                       # writer=ew,
                                       job_type='eval',
                                       nchains=nchains,
                                       trainer=trainer)
        if cfg.steps.test > 0:                                      # [3.]
            log.warning('Running generic HMC')
            eps_default = 1. / cfg.dynamics.nleapfrog
            eps_hmc = cfg.get('eps_hmc', eps_default)
            outputs['hmc'] = evaluate(cfg=cfg,
                                      run=run,
                                      # writer=hw,
                                      eps=eps_hmc,
                                      job_type='hmc',
                                      nchains=nchains,
                                      trainer=trainer)
    if run is not None:
        run.save('./train/*ckpt*')
        run.save('./train/*.h5*')
        run.save('./eval/*.h5*')
        run.save('./hmc/*.h5*')
        run.finish()

    return outputs


@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def launch(cfg: DictConfig) -> None:
    _ = main(cfg)


if __name__ == '__main__':
    launch()
