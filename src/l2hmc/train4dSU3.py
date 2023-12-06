"""
train4dSU3.py
"""
from __future__ import absolute_import, annotations, division, print_function
import json
import logging
import os
from pathlib import Path
import time
from typing import Optional

import lovely_tensors as lt
import matplotlib.pyplot as plt
import opinionated
import torch
import yaml

from l2hmc.configs import CONF_DIR, OUTPUTS_DIR
from l2hmc.configs import dict_to_list_of_overrides, get_experiment
from l2hmc.experiment.pytorch.experiment import Experiment
import l2hmc.group.su3.pytorch.group as g
from l2hmc.utils.dist import setup_torch
from l2hmc.utils.history import BaseHistory, summarize_dict
from l2hmc.utils.plot_helpers import get_timestamp

lt.monkey_patch()

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['COLORTERM'] = 'truecolor;'
os.environ['MASTER_PORT'] = '5433'
# os.environ['MPLBACKEND'] = 'module://matplotlib-backend-kitty'
# plt.switch_backend('module://matplotlib-backend-kitty')

# log = get_logger(__name__)

RLOG_YAML = CONF_DIR / 'hydra' / 'job_logging' / 'rich.yaml'
with RLOG_YAML.open('r') as stream:
    LOGCONF = dict(yaml.safe_load(stream))

logging.config.dictConfig(LOGCONF)
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

DAY = get_timestamp('%Y-%m-%d')
TIME = get_timestamp('%H-%M-%S')
OUTDIR = OUTPUTS_DIR / "train4dSU3" / f"{DAY}" / f"{TIME}"
OUTDIR.mkdir(exist_ok=True, parents=True)

HMC_DIR = OUTDIR / "hmc"
EVAL_DIR = OUTDIR / "eval"
TRAIN_DIR = OUTDIR / "train"

HMC_DIR.mkdir(exist_ok=True, parents=True)
EVAL_DIR.mkdir(exist_ok=True, parents=True)
TRAIN_DIR.mkdir(exist_ok=True, parents=True)

log.critical(f"{OUTDIR=}")

_ = setup_torch(precision='float64', backend='DDP', seed=4351)

plt.style.use(opinionated.STYLES['opinionated_min'])
# set_plot_style()

# from l2hmc.utils.plot_helpers import (  # noqa
#     # set_plot_style,
#     # plot_scalar,
#     # plot_chains,
#     # plot_leapfrogs
# )


def savefig(fig: plt.Figure, fname: str, outdir: os.PathLike):
    pngfile = Path(outdir).joinpath(f"pngs/{fname}.png")
    svgfile = Path(outdir).joinpath(f"svgs/{fname}.svg")
    pngfile.parent.mkdir(exist_ok=True, parents=True)
    svgfile.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(svgfile, transparent=True, bbox_inches='tight')
    fig.savefig(pngfile, transparent=True, bbox_inches='tight', dpi=300)


# def plot_metrics(metrics: dict, title: Optional[str] = None, **kwargs):
#     from l2hmc.utils.rich import is_interactive
#     outdir = Path(f"./plots-4dSU3/{title}")
#     outdir.mkdir(exist_ok=True, parents=True)
#     for key, val in metrics.items():
#         fig, ax = plot_metric(val, name=key, **kwargs)
#         if title is not None:
#             ax.set_title(title)
#         log.info(f"Saving {key} to {outdir}")
#         savefig(fig, f"{key}", outdir=outdir)
#         # fpath = outdir.joinpath(f"{key}")
#         # plt.savefig(f"{fpath}.svg", bbox_inches='tight')
#         # plt.savefig(f"{fpath}.png", bbox_inches='tight', dpi=300)
#         # log.info(f"Saving {title} {key} plot to {fpath}")
#         if not is_interactive():
#             plt.show()


# def plot_metric(
#         metric: torch.Tensor,
#         name: Optional[str] = None,
#         **kwargs,
# ):
#     assert len(metric) > 0
#     if isinstance(metric[0], (int, float, bool, np.floating)):
#         y = np.stack(metric)
#         return plot_scalar(y, ylabel=name, **kwargs)
#     element_shape = metric[0].shape
#     if len(element_shape) == 2:
#         y = grab_tensor(torch.stack(metric))
#         return plot_leapfrogs(y, ylabel=name)
#     if len(element_shape) == 1:
#         y = grab_tensor(torch.stack(metric))
#         return plot_chains(y, ylabel=name, **kwargs)
#     if len(element_shape) == 0:
#         y = grab_tensor(torch.stack(metric))
#         return plot_scalar(y, ylabel=name, **kwargs)
#     raise ValueError


def HMC(
        experiment: Experiment,
        nsteps: int = 10,
        beta: float = 1.0,
        nlog: int = 1,
        nprint: int = 1,
        x: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
        nleapfrog: Optional[int] = None,
) -> tuple[torch.Tensor, BaseHistory]:
    """Run HMC on `experiment`"""
    history_hmc = BaseHistory()
    # x = state.x
    if x is None:
        state = experiment.trainer.dynamics.random_state(beta=beta)
        x = state.x
    for step in range(nsteps):
        # log.info(f'HMC STEP: {step}')
        tic = time.perf_counter()
        x, metrics_ = experiment.trainer.hmc_step(
            (x, beta),
            eps=eps,
            nleapfrog=nleapfrog
        )
        toc = time.perf_counter()
        metrics = {
            'hmc_step': step,
            'dt': toc - tic,
            **metrics_,
        }
        if step % nlog == 0 or step % nprint == 0:
            avgs = history_hmc.update(metrics)
        if step % nprint == 0:
            summary = summarize_dict(avgs)
            log.info(summary)
    xhmc = experiment.trainer.dynamics.unflatten(x)
    log.info(f"checkSU(x_hmc): {g.checkSU(xhmc)}")
    history_hmc.plot_all(outdir=HMC_DIR)
    return (xhmc, history_hmc)


def eval(
        experiment: Experiment,
        nsteps: int = 10,
        beta: float = 1.0,
        nlog: int = 1,
        nprint: int = 2,
        x: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, BaseHistory]:
    """Run eval on `experiment`"""
    history_eval = BaseHistory()
    if x is None:
        state = experiment.trainer.dynamics.random_state(beta=beta)
        x = state.x

    for step in range(nsteps):
        tic = time.perf_counter()
        x, metrics_ = experiment.trainer.eval_step((x, beta))
        toc = time.perf_counter()
        metrics = {
            'eval_step': step,
            'dt': toc - tic,
            **metrics_,
        }
        if step % nlog == 0 or step % nprint == 0:
            avgs = history_eval.update(metrics)
        if step % nprint == 0:
            summary = summarize_dict(avgs)
            log.info(summary)
    xeval = experiment.trainer.dynamics.unflatten(x)
    log.info(f"checkSU(x_hmc): {g.checkSU(xeval)}")
    history_eval.plot_all(outdir=EVAL_DIR)
    return (xeval, history_eval)


def main() -> tuple[torch.Tensor, dict[str, BaseHistory]]:
    # from l2hmc.experiment.pytorch.experiment import train_step
    # set_plot_style()
    plt.style.use(opinionated.STYLES['opinionated_min'])

    su3conf = Path('./conf/su3-min.yaml')
    with su3conf.open('r') as stream:
        conf = dict(yaml.safe_load(stream))

    log.info(json.dumps(conf, indent=4))
    overrides = dict_to_list_of_overrides(conf)
    ptExpSU3 = get_experiment(overrides=[*overrides], build_networks=True)
    state = ptExpSU3.trainer.dynamics.random_state(6.0)
    assert isinstance(state.x, torch.Tensor)
    assert isinstance(state.beta, torch.Tensor)
    assert isinstance(ptExpSU3, Experiment)
    xhmc, history_hmc = HMC(
        nsteps=10,
        experiment=ptExpSU3,
        beta=state.beta.item(),
        x=state.x,
        eps=0.1,
        nleapfrog=1,
        nlog=1,
        nprint=2,
    )
    # assert isinstance(history_hmc, BaseHistory)
    # plot_metrics(history_hmc, title='HMC', marker='.')
    # ptExpSU3.trainer.dynamics.init_weights(
    #     method='uniform',
    #     min=-1e-16,
    #     max=1e-16,
    #     bias=True,
    #     # xeps=0.001,
    #     # veps=0.001,
    # )
    xeval, history_eval = eval(
        nsteps=10,
        experiment=ptExpSU3,
        beta=6.0,
        x=state.x,
        nlog=1,
        nprint=1,
    )
    # xeval = ptExpSU3.trainer.dynamics.unflatten(xeval)
    # history_eval.plot_all()
    # log.info(f"checkSU(x_eval): {g.checkSU(xeval)}")
    # plot_metrics(history_eval, title='Evaluate', marker='.')

    history_train = BaseHistory()
    x = state.x
    for step in range(50):
        # log.info(f'HMC STEP: {step}')
        tic = time.perf_counter()
        x, metrics_ = ptExpSU3.trainer.train_step(
            (x, state.beta)
        )
        toc = time.perf_counter()
        metrics = {
            'train_step': step,
            'dt': toc - tic,
            **metrics_,
        }
        avgs = history_train.update(metrics)
        summary = summarize_dict(avgs)
        log.info(summary)

    history_train.plot_all(outdir=TRAIN_DIR)
    # histories = {
    #     'train': history_train,
    #     'eval': history_eval,
    #     'hmc': history_hmc,
    # }

    # history = BaseHistory()
    # x = state.x
    # for step in range(20):
    #     # log.info(f'TRAIN STEP: {step}')
    #     tic = time.perf_counter()
    #     x, metrics = ptExpSU3.trainer.train_step((x, state.beta))
    #     toc = time.perf_counter()
    #     if (step > 0 and step % 2 == 0):
    #         print_dict(metrics, grab=True)
    #     if (step > 0 and step % 1 == 0):
    #         for key, val in metrics.items():
    #             try:
    #                 history[key].append(val)
    #             except KeyError:
    #                 history[key] = [val]
    #
    # x = ptExpSU3.trainer.dynamics.unflatten(x)
    # log.info(f"checkSU(x_train): {g.checkSU(x)}")
    # plot_metrics(history, title='train', marker='.')
    #
    # for step in range(20):
    #     log.info(f"train step: {step}")
    #     x, metrics = ptExpSU3.trainer.train_step((x, state.beta))
    #     if step % 5 == 0:
    #         print_dict(metrics, grab=True)

    return (
        x,
        {
            'train': history_train,
            'eval': history_eval,
            'hmc': history_hmc,
        }
    )


if __name__ == '__main__':
    _ = main()
