"""
main_tensorflow.py

Main entry-point for training L2HMC Dynamics w/ TensorFlow
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
from typing import Any, Optional

import hydra
from omegaconf import DictConfig
import tensorflow as tf

from l2hmc.common import save_and_analyze_data
from l2hmc.configs import get_jobdir
from l2hmc.experiment.tensorflow.experiment import Experiment
from l2hmc.trainers.tensorflow.trainer import Trainer
from l2hmc.utils.tensorflow.utils import get_summary_writer

log = logging.getLogger(__name__)

Tensor = tf.Tensor
TensorLike = tf.types.experimental.TensorLike


def load_from_ckpt(
        # dynamics: Dynamics,
        # optimizer: Optimizer,
        # cfg: DictConfig,
):
    pass


def evaluate(
        cfg: DictConfig,
        trainer: Trainer,
        job_type: str,
        run: Optional[Any] = None,
        # writer: Optional[Any] = None,
        nchains: Optional[int] = None,
        eps: Optional[float] = None,
        nsteps: Optional[int] = None,
) -> dict:
    assert isinstance(nchains, int)
    assert job_type in ['eval', 'hmc']
    therm_frac = cfg.get('therm_frac', 0.2)
    jobdir = get_jobdir(cfg, job_type=job_type)

    if trainer.rank == 0:
        writer = get_summary_writer(cfg, job_type=job_type)
        assert writer is not None
        writer.set_as_default()
    else:
        writer = None

    output = trainer.eval(run=run,
                          writer=writer,
                          nchains=nchains,
                          job_type=job_type,
                          eps=eps,
                          nsteps=nsteps)
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
                              framework='tensorflow')

    if writer is not None:
        writer.close()

    return output


def train(
        cfg: DictConfig,
        trainer: Trainer,
        run: Optional[Any] = None,
        # writer: Optional[Any] = None,
        nchains: Optional[int] = None,
        **kwargs,
) -> dict:
    jobdir = get_jobdir(cfg, job_type='train')

    # if writer is not None:
    if trainer.rank == 0:
        writer = get_summary_writer(cfg, job_type='train')
        assert writer is not None
        writer.set_as_default()
    else:
        writer = None

    output = trainer.train(run=run,
                           writer=writer,
                           train_dir=jobdir,
                           **kwargs)
    if trainer.rank == 0:
        dset = output['history'].get_dataset()
        nchains = min(cfg.dynamics.nchains, max(16, cfg.dynamics.nchains // 8))
        _ = save_and_analyze_data(dset,
                                  run=run,
                                  outdir=jobdir,
                                  output=output,
                                  nchains=nchains,
                                  job_type='train',
                                  framework='tensorflow')
    if writer is not None:
        writer.close()

    return output


def main(cfg: DictConfig) -> dict:
    outputs = {}
    # -----------------------------
    # Create experiment from cfg
    # -----------------------------
    # nchains = max((16, cfg.dynamics.nchains // 4))
    # cfg.update({'nchains': nchains})
    experiment = Experiment(cfg)
    objs = experiment.build()
    run = objs['run']
    trainer = objs['trainer']
    # -------------------------------------------------------------
    # 1. Train model
    # 2. Evaluate trained model
    # 3. Run generic HMC as baseline w/ same trajectory length
    # -------------------------------------------------------------
    should_train = (cfg.steps.nera > 0 and cfg.steps.nepoch > 0)
    if should_train:                                                # [1.]
        # tw = (
        #     experiment.get_summary_writer('train') if trainer.rank == 0
        #     else None
        # )
        outputs['train'] = train(cfg=cfg, trainer=trainer, run=run)

    if trainer.rank == 0:
        nchains = max(16, cfg.dynamics.nchains // 8)
        if should_train and cfg.steps.test > 0:                     # [2.]
            log.warning('Evaluating trained model')
            # ew = experiment.get_summary_writer('eval')
            outputs['eval'] = evaluate(cfg,
                                       run=run,
                                       eps=None,
                                       # writer=ew,
                                       job_type='eval',
                                       nchains=nchains,
                                       trainer=trainer)
        if cfg.steps.test > 0:                                      # [3.]
            log.warning('Running generic HMC')
            # eps = tf.constant(float(cfg.get('eps_hmc', 0.118)))
            eps = float(cfg.get('eps_hmc', 0.118))
            # hw = experiment.get_summary_writer('hmc')
            outputs['hmc'] = evaluate(cfg=cfg,
                                      run=run,
                                      eps=eps,
                                      # writer=hw,
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
