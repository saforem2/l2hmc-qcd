r"""°°°
# `l2hmc-qcd`

This notebook contains a minimal working example uor the 4D $SU(3)$ model

Uses `torch.complex128` by default
°°°"""
# |%%--%%| <UccbxQm8jb|ony6FlNRK5>
r"""°°°
## Setup
°°°"""
#|%%--%%| <ony6FlNRK5|aIbo82QZ6U>
# %matplotlib inline import matplotlib_inline
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
import lovely_tensors as lt
lt.monkey_patch()
lt.set_config(color=False)
# automatically detect and reload local changes to modules
%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from l2hmc.utils.plot_helpers import FigAxes, set_plot_style
set_plot_style()

# |%%--%%| <aIbo82QZ6U|w7PXUSqerd>
import os
from pathlib import Path
from typing import Optional
from rich import print

import lovely_tensors as lt
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from l2hmc.utils.dist import setup_torch
seed = np.random.randint(2 ** 32)
print(f"seed: {seed}")
_ = setup_torch(precision='float64', backend='DDP', seed=seed)

import l2hmc.group.su3.pytorch.group as g
from l2hmc.utils.rich import get_console
from l2hmc.common import grab_tensor, print_dict
from l2hmc.configs import dict_to_list_of_overrides, get_experiment
from l2hmc.experiment.pytorch.experiment import Experiment, evaluate  # noqa  # noqa
from l2hmc.utils.plot_helpers import set_plot_style

os.environ['COLORTERM'] = 'truecolor'
os.environ['MASTER_PORT'] = '5439'
# os.environ['MPLBACKEND'] = 'module://matplotlib-backend-kitty'
# plt.switch_backend('module://matplotlib-backend-kitty')
console = get_console()


set_plot_style()

from l2hmc.utils.plot_helpers import (  # noqa
    set_plot_style,
    plot_scalar,
    plot_chains,
    plot_leapfrogs
)

def savefig(fig: plt.Figure, fname: str, outdir: os.PathLike):
    pngfile = Path(outdir).joinpath(f"pngs/{fname}.png")
    svgfile = Path(outdir).joinpath(f"svgs/{fname}.svg")
    pngfile.parent.mkdir(exist_ok=True, parents=True)
    svgfile.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(svgfile, transparent=True, bbox_inches='tight')
    fig.savefig(pngfile, transparent=True, bbox_inches='tight', dpi=300)

def plot_metrics(metrics: dict, title: Optional[str] = None, **kwargs):
    outdir = Path(f"./plots-4dSU3/{title}")
    outdir.mkdir(exist_ok=True, parents=True)
    for key, val in metrics.items():
        fig, ax = plot_metric(val, name=key, **kwargs)
        if title is not None:
            ax.set_title(title)
        console.log(f"Saving {key} to {outdir}")
        savefig(fig, f"{key}", outdir=outdir)
        plt.show()

def plot_metric(
        metric: torch.Tensor,
        name: Optional[str] = None,
        **kwargs,
):
    assert len(metric) > 0
    if isinstance(metric[0], (int, float, bool, np.floating)):
        y = np.stack(metric)
        return plot_scalar(y, ylabel=name, **kwargs)
    element_shape = metric[0].shape
    if len(element_shape) == 2:
        y = grab_tensor(torch.stack(metric))
        return plot_leapfrogs(y, ylabel=name)
    if len(element_shape) == 1:
        y = grab_tensor(torch.stack(metric))
        return plot_chains(y, ylabel=name, **kwargs)
    if len(element_shape) == 0:
        y = grab_tensor(torch.stack(metric))
        return plot_scalar(y, ylabel=name, **kwargs)
    raise ValueError


# |%%--%%| <w7PXUSqerd|Mh9n3vG3Zc>
r"""°°°
## Load config + build Experiment
°°°"""
# |%%--%%| <Mh9n3vG3Zc|QvpgSwgrLI>

from rich import print
set_plot_style()

from l2hmc.configs import CONF_DIR
su3conf = Path(f"{CONF_DIR}/su3test.yaml")
with su3conf.open('r') as stream:
    conf = dict(yaml.safe_load(stream))
# overrides = {
#     'framework': 'pytorch',
#     'backend': 'DDP',
#     'dynamics': {
#         'eps': 0.15,
#         'nleapfrog': 8,
#         'nchains': 4,
#         'use_separate_networks': False,
#     },
#     'network': {
#         'units': [32, 32, 32],
#         'activation_fn': 'tanh',
#         # 'use_batch_norm': True,
#     },
#     'learning_rate': {
#         'clip_norm': 100.0,
#         'lr_init': 0.0005,
#     },
#     'loss': {
#         'use_mixed_loss': True,
#     },
#     'net_weights': {
#         'x': {
#             's': 0.0,
#             't': 0.0,
#             'q': 0.0,
#         },
#         'v': {
#             's': 1.0,
#             't': 1.0,
#             'q': 1.0,
#         },
#     }
# }
# conf |= overrides
# conf['dynamics']['eps'] = 0.05
# conf['dynamics']['nchains'] = 8
# conf['loss']['use_mixed_loss'] = True
console.print(conf)
overrides = dict_to_list_of_overrides(conf)

# |%%--%%| <QvpgSwgrLI|rG0f7zmJbO>

ptExpSU3 = get_experiment(overrides=[*overrides], build_networks=True)
console.print(ptExpSU3.config)
state = ptExpSU3.trainer.dynamics.random_state(6.0)
console.print(f"checkSU(state.x): {g.checkSU(state.x)}")
console.print(f"checkSU(state.x): {g.checkSU(g.projectSU(state.x))}")
assert isinstance(state.x, torch.Tensor)
assert isinstance(state.beta, torch.Tensor)
assert isinstance(ptExpSU3, Experiment)

#|%%--%%| <rG0f7zmJbO|99EodnQDPS>
from l2hmc.utils.plot_helpers import set_plot_style
set_plot_style()

from l2hmc.common import get_timestamp
TSTAMP = get_timestamp()
OUTPUT_DIR = Path(f"./outputs/pt4dSU3/{TSTAMP}")
HMC_DIR = OUTPUT_DIR.joinpath('hmc')
EVAL_DIR = OUTPUT_DIR.joinpath('eval')
TRAIN_DIR = OUTPUT_DIR.joinpath('train')
HMC_DIR.mkdir(exist_ok=True, parents=True)
EVAL_DIR.mkdir(exist_ok=True, parents=True)
TRAIN_DIR.mkdir(exist_ok=True, parents=True)

# ptExpSU3.trainer.dynamics.init_weights( method='xavier_uniform',
#     # constant=0.0,
#     # method='uniform',
#     # min=-1e-6,
#     # max=1e-6,
#     # bias=True,
#     # xeps=0.05,
#     # veps=0.05,
# )
# ptExpSU3.trainer.optimizer.zero_grad()
ptExpSU3.trainer.print_grads_and_weights()
console.print(ptExpSU3.config)

# |%%--%%| <99EodnQDPS|2CLIQasWMO>
r"""°°°
## Training
°°°"""
#|%%--%%| <2CLIQasWMO|s1fvvL6d5f>

from l2hmc.utils.history import BaseHistory
history: BaseHistory = BaseHistory()
state = ptExpSU3.trainer.dynamics.random_state(6.0)
dynamics = ptExpSU3.trainer.dynamics
x = state.x
freq = {'print': 2, 'save': 5}
for step in range(500):
    console.print(f'TRAIN STEP: {step}')
    x, metrics = ptExpSU3.trainer.train_step((x, state.beta))
    avg, diff = ptExpSU3.trainer.g.checkSU(dynamics.unflatten(x))
    console.print(f"avg: {avg}")
    console.print(f"diff: {diff}")
    # x = ptExpSU3.trainer.g.projectSU(dynamics.unflatten(x))
    # x = ptExpSU3.trainer.g.projectSU(dynamics.unflatten(x))
    if (step > 0 and step % freq['print'] == 0):
        print_dict(metrics, grab=True)
    if (step > 0 and step % freq['save'] == 0):
        history.update(metrics)

#|%%--%%| <s1fvvL6d5f|v0zGvghU1y>
dataset = history.get_dataset()
history.plot_all(
    outdir=TRAIN_DIR,
    title='Train',
    num_chains=x.shape[0],
)
console.print(f'{TRAIN_DIR.as_posix()}')

# |%%--%%| <v0zGvghU1y|buExHZxW3Y>
r"""°°°
## Evaluation
°°°"""
# |%%--%%| <buExHZxW3Y|gN0SI4JuE9>

state = ptExpSU3.trainer.dynamics.random_state(6.0)
# ptExpSU3.trainer.dynamics.init_weights(
#     # method='zeros',
#     # constant=0.0,
#     # method='uniform',
#     # min=-1e-3,
#     # max=1e-3,
#     # bias=True,
#     xeps=0.05,
#     veps=0.05,
# )
xeval, history_eval = evaluate(
    nsteps=500,
    exp=ptExpSU3,
    beta=6.0,
    # x=state.x,
    job_type='eval',
    nlog=5,
    nprint=2,
    grab=True,
)

#|%%--%%| <gN0SI4JuE9|nqDHiw6xGT>

dataset_eval = history_eval.get_dataset()

#|%%--%%| <nqDHiw6xGT|AxT7V3bOUW>

history_eval.plot_all(outdir=EVAL_DIR, title='Eval')

xeval = ptExpSU3.trainer.dynamics.unflatten(xeval)
console.log(f"checkSU(x_eval): {g.checkSU(xeval)}")
console.log(f"checkSU(x_eval): {g.checkSU(g.projectSU(xeval))}")

# |%%--%%| <AxT7V3bOUW|dqVpW4I8wq>
r"""°°°
## HMC
°°°"""

xhmc, history_hmc = evaluate(
    nsteps=500,
    exp=ptExpSU3,
    beta=6.0,
    x=state.x,
    eps=0.1,
    nleapfrog=8,
    job_type='hmc',
    nlog=5,
    nprint=5,
    grab=True
)

#|%%--%%| <dqVpW4I8wq|xSbTkYW4DH>

import l2hmc.utils.plot_helpers as ph
xhmc = ptExpSU3.trainer.dynamics.unflatten(xhmc)
console.log(f"checkSU(x_hmc): {g.checkSU(xhmc)}")
dataset_hmc = history_hmc.get_dataset()
# ph.plot_dataset(dataset_hmc, outdir=HMC_DIR)
# plot_metrics(history_hmc, title='HMC', marker='.')

#|%%--%%| <xSbTkYW4DH|8mvysCe9S6>

history_hmc.plot_all(title="HMC", outdir=HMC_DIR)

#|%%--%%| <8mvysCe9S6|iEdyJ2pQsW>

import matplotlib.pyplot as plt
pdiff = dataset_eval.plaqs - dataset_hmc.plaqs
pdiff

#|%%--%%| <iEdyJ2pQsW|nj30j4bfwP>

import xarray as xr

fig, ax = plt.subplots(figsize=(12, 4))
(pdiff ** 2).plot(ax=ax)  #, robust=True)
ax.set_title(r"$\left|\delta U_{\mu\nu}\right|^{2}$ (HMC - Eval)")
outfile = Path(EVAL_DIR).joinpath('pdiff.svg')
fig.savefig(outfile.as_posix(), dpi=400, bbox_inches='tight')


#|%%--%%| <nj30j4bfwP|C9ywINGEBl>

# history_hmc.plot_dataArray1(dataset_hmc.plaqs, key='Plaqs (HMC)')
# ph.plot_array(dataset_hmc.plaqs.values, key='Plaqs (HMC)')



dynamics = ptExpSU3.trainer.dynamics
xeval = ptExpSU3.trainer.dynamics.unflatten(xeval)
console.log(f"checkSU(x_train): {g.checkSU(dynamics.unflatten(x))}")
console.log(f"checkSU(x_train): {g.checkSU(g.projectSU(x))}")


#|%%--%%| <C9ywINGEBl|syvbV3dfEx>

x = ptExpSU3.trainer.dynamics.unflatten(x)
console.print(f"checkSU(x_train): {g.checkSU(x)}")
dataset = history.get_dataset()


#|%%--%%| <syvbV3dfEx|jxTZ7Rv68Q>

# matplotlib.use('module://matplotlib-kitty')
import l2hmc.utils.plot_helpers as ph
from pathlib import Path
from l2hmc.common import get_timestamp

tstamp = get_timestamp()
outdir = Path(f"./outputs/pt4dsu3/2023-07-19/{tstamp}")
ph.plot_dataset(dataset, outdir=outdir, title='Training')

#|%%--%%| <jxTZ7Rv68Q|n82AjzDKaI>

import l2hmc.utils.plot_helpers as ph
from l2hmc.common import di

#|%%--%%| <n82AjzDKaI|y1aG6FM7s6>

# fig, ax = plt.subplots()
figax = plot_metric(
    history['plaqs'],
    name='Plaqs (Training)',
    marker='.'
)
fig, ax = figax
fig.savefig('4dSU3-train-plaqs-2023-07-19.svg', bbox_inches='tight')


#|%%--%%| <y1aG6FM7s6|ZhC5oScQOm>

state = ptExpSU3.trainer.dynamics.random_state(6.0)
acc = torch.ones(state.x.shape[0])
m, mb = ptExpSU3.trainer.dynamics._get_mask(0)
loss = torch.tensor([0.])
with torch.autograd.detect_anomaly(check_nan=True):  # flake8: noqa  pyright:ignore
    ptExpSU3.trainer.optimizer.zero_grad()
    # state_vb, logdet_vf = ptExpSU3.trainer.dynamics._update_v_bwd(
    #     step=0, state=state_vf
    # )
    # console.log(f'TRAIN STEP: {step}')
    x, metrics = ptExpSU3.trainer.train_step((state.x, state.beta))
    loss = metrics['loss']
    # loss_xb = ptExpSU3.trainer.calc_loss(state_xb.x, state.x, acc=acc)
    # loss_xf = ptExpSU3.trainer.calc_loss(state_xf.x, state.x, acc=acc)
    # loss_ = loss_xb + loss_xf
    # loss = ptExpSU3.trainer.backward_step(loss_)
    console.print(f"loss: {loss:.5f}")

#|%%--%%| <ZhC5oScQOm|tnSLcg1Iw1>

state = ptExpSU3.trainer.dynamics.random_state(6.0)
acc = torch.ones(state.x.shape[0])
m, mb = ptExpSU3.trainer.dynamics._get_mask(0)
loss = torch.tensor([0.])
step = 0
from l2hmc.configs import State
with torch.autograd.detect_anomaly(check_nan=True):  # flake8: noqa  pyright:ignore
    ptExpSU3.trainer.optimizer.zero_grad()
    # sumlogdet = torch.zeros(state.x.shape[0], device=self.device)
    state_vf1, logdet = ptExpSU3.trainer.dynamics._update_v_fwd(step, state)
    sumlogdet = logdet
    # state_ = State(state.x, vf1, state.beta)
    state_xf1, logdet = ptExpSU3.trainer.dynamics._update_x_fwd(step, state_vf1, m, first=True)
    sumlogdet = sumlogdet + logdet
    state_xf2, logdet = ptExpSU3.trainer.dynamics._update_x_fwd(step, state_xf1, mb, first=False)
    sumlogdet = sumlogdet + logdet
    state_vf2, logdet = ptExpSU3.trainer.dynamics._update_v_fwd(step, state_xf2)
    sumlogdet = sumlogdet + logdet
    # state_, logdet = ptExpSU3.trainer.dynamics._forward_lf(step=0, state=state)
    # state_, logdet = ptExpSU3.trainer.dynamics._update_x_fwd(
    #     step=0, state=state, m=m, first=True
    # )
    # state_xb, logdet_xb = ptExpSU3.trainer.dynamics._update_x_bwd(
    #     step=0, state=state_xf, m=m, first=True
    # )
    # state_vb, logdet_vf = ptExpSU3.trainer.dynamics._update_v_bwd(
    #     step=0, state=state_vf
    # )
    # loss_xb = ptExpSU3.trainer.calc_loss(state_xb.x, state.x, acc=acc)
    loss_ = ptExpSU3.trainer.calc_loss(state_xf1.x, state.x, acc=acc)
    # loss_ = loss_xb + loss_xf
    loss = ptExpSU3.trainer.backward_step(loss_)
    console.print(f"loss: {loss.item():.5f}")

#|%%--%%| <tnSLcg1Iw1|wPjjw6nln2>

state = ptExpSU3.trainer.dynamics.random_state(6.0)
acc = torch.ones(state.x.shape[0])
m, mb = ptExpSU3.trainer.dynamics._get_mask(0)
loss = torch.tensor([0.])
with torch.autograd.detect_anomaly(check_nan=True):  # flake8: noqa  pyright:ignore
    ptExpSU3.trainer.optimizer.zero_grad()
    state_vf, logdet_vf = ptExpSU3.trainer.dynamics._update_v_fwd(
        step=0, state=state
    )
    state_vb, logdet_vf = ptExpSU3.trainer.dynamics._update_v_bwd(
        step=0, state=state_vf
    )
    loss_ = ptExpSU3.trainer.calc_loss(state.x, state_vf.x, acc=acc)
    loss = ptExpSU3.trainer.backward_step(loss_)
    console.print(f"loss: {loss.item():.5f}")

#|%%--%%| <wPjjw6nln2|MIQ9oXakGc>

from torch import autograd
state = ptExpSU3.trainer.dynamics.random_state(6.0)
acc = torch.ones(state.x.shape[0])
m, mb = ptExpSU3.trainer.dynamics._get_mask(0)
loss = torch.tensor([0.])
# ptExpSU3.trainer.dynamics.init_weights(
#     method='uniform',
#     min=-1e-32,
#     max=1e-32,
#     bias=True,
#     xeps=0.001,
#     veps=0.001,
# )

with autograd.detect_anomaly(check_nan=True):
    ptExpSU3.trainer.optimizer.zero_grad()
    state_vf, logdet_vf = ptExpSU3.trainer.dynamics._update_v_fwd(
        step=0, state=state
    )
    state_xf, logdet_xf = ptExpSU3.trainer.dynamics._update_x_fwd(
        step=0, state=state_vf, m=m, first=True
    )
    console.print(f"state_xf.x.shape: {state_xf.x.shape}")
    avg, diff = ptExpSU3.trainer.g.checkSU(state_xf.x)
    console.print(f"avg: {avg}, diff: {diff}")
    loss_ = ptExpSU3.trainer.calc_loss(state.x, state_xf.x, acc=acc)
    loss = ptExpSU3.trainer.backward_step(loss_)
    console.print(f"loss: {loss.item():.5f}")

#|%%--%%| <MIQ9oXakGc|0akO7quOYk>

from torch import autograd
from l2hmc.dynamics.pytorch.dynamics import sigmoid

step = 0
dynamics = ptExpSU3.trainer.dynamics
state = ptExpSU3.trainer.dynamics.random_state(6.0)
acc = torch.ones(state.x.shape[0])
m, mb = ptExpSU3.trainer.dynamics._get_mask(0)
loss = torch.tensor([0.])
ptExpSU3.trainer.dynamics.init_weights(
    method='uniform',
    min=-1e-32,
    max=1e-32,
    bias=True,
    xeps=0.001,
    veps=0.001,
)

with autograd.detect_anomaly(check_nan=True):
    eps = sigmoid(dynamics.veps[step].log())
    force = dynamics.grad_potential(state.x, state.beta)
    s, t, q = dynamics._call_vnet(step, (state.x, force))
    logjac = eps * s / 2.  # jacobian factor, also used in exp_s below
    logdet = dynamics.flatten(logjac).sum(1)
    force = force.reshape_as(state.v)
    exp_s = (logjac.exp()).reshape_as(state.v)
    exp_q = (eps * q).exp().reshape_as(state.v)
    t = t.reshape_as(state.v)
    vf = (exp_s * state.v) - (0.5 * eps * (force * exp_q + t))
    if dynamics.config.group == 'SU3':
        vf = dynamics.g.projectTAH(vf)
    loss_ = ptExpSU3.trainer.calc_loss(state.v, vf, acc=acc)
    loss_.register_hook(lambda grad: grad.clamp_(max=1.0))
    loss_.register_hook(lambda grad: console.print(f"grad: {grad}"))
    ptExpSU3.trainer.optimizer.zero_grad()
    loss_.backward()
    torch.nn.utils.clip_grad.clip_grad_norm(dynamics.parameters(), max_norm=1.0)
    ptExpSU3.trainer.optimizer.step()
    # loss = ptExpSU3.trainer.backward_step(loss_)
    console.print(f"loss: {loss_.item():.5f}")


# |%%--%%| <0akO7quOYk|dWFrH3mveT>

history = {}
x = state.x
for step in range(50):
    console.log(f'TRAIN STEP: {step}')
    x, metrics = ptExpSU3.trainer.train_step((x, state.beta))
    if (step > 0 and step % 2 == 0):
        print_dict(metrics, grab=True)
    if (step > 0 and step % 1 == 0):
        for key, val in metrics.items():
            try:
                history[key].append(val)
            except KeyError:
                history[key] = [val]

x = ptExpSU3.trainer.dynamics.unflatten(x)
console.log(f"checkSU(x_train): {g.checkSU(x)}")
plot_metrics(history, title='train', marker='.')

# |%%--%%| <dWFrH3mveT|oXz2S4kQx8>


