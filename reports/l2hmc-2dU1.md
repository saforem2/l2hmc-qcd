---
title: '`l2hmc`: Example'
jupyter: '2023-04-26'
---


<a href="https://arxiv.org/abs/2105.03418"><img alt="arxiv" src="http://img.shields.io/badge/arXiv-2105.03418-B31B1B.svg" align="left"></a> <a href="https://arxiv.org/abs/2112.01582"><img alt="arxiv" src="http://img.shields.io/badge/arXiv-2112.01582-B31B1B.svg" align="left" style="margin-left:10px;"></a><br>


This notebook will (attempt) to walk through the steps needed to successfully instantiate and "run" an experiment.

For this example, we wish to train the L2HMC sampler for the 2D $U(1)$ lattice gauge model with Wilson action:

$$
S_{\beta}(n) = \beta \sum_{n}\sum_{\mu<\nu}\mathrm{Re}\left[1 - U_{\mu\nu}(n) \right]
$$

This consists of the following steps:
  1. Build an `Experiment` by parsing our configuration object
  2. Train our model using the `Experiment.train()` method
  3. Evaluate our trained model `Experiment.evaluate(job_type='eval')`
  4. Compare our trained models' performance against generic HMC `Experiment.evaluate(job_type='hmc')`
  

<div class="alert alert-block alert-info" style="background:rgba(102, 102, 102, 0.1);color:#666666; border-radius:5px; border: none;">
<span style="font-weight:700; font-size:1.5em;">Evaluating Performance</span>
    
Explicitly, we measure the performance of our model by comparing the _tunneling rate_ $\delta Q$ of our **trained** sampler to that of generic HMC.
    
Explicitly, the tunneling rate is given by:
    
$$
\delta Q = \frac{1}{N_{\mathrm{chains}}}\sum_{\mathrm{chains}} \left|Q_{i+1} - Q_{i}\right|
$$
    
where the difference is between subsequent states in a chain, and the sum is over all $N$ chains (each being ran in parallel, _independently_).
    
Since our goal is to generate _independent configurations_, the more our sampler tunnels between different topological sectors (_tunneling rate_), the more efficient our sampler.
    
</div>


## Imports / Setup

```{python}
#| execution: {iopub.execute_input: '2023-06-09T23:42:28.968461Z', iopub.status.busy: '2023-06-09T23:42:28.968155Z', iopub.status.idle: '2023-06-09T23:42:31.367466Z', shell.execute_reply: '2023-06-09T23:42:31.366834Z', shell.execute_reply.started: '2023-06-09T23:42:28.968444Z'}
#| tags: []
! nvidia-smi | tail --lines -7
```

```{python}
#| execution: {iopub.execute_input: '2023-06-09T23:43:15.724437Z', iopub.status.busy: '2023-06-09T23:43:15.724152Z', iopub.status.idle: '2023-06-09T23:43:16.401756Z', shell.execute_reply: '2023-06-09T23:43:16.401101Z', shell.execute_reply.started: '2023-06-09T23:43:15.724420Z'}
#| tags: []
%load_ext autoreload
%autoreload 2
%matplotlib widget

import os
import warnings

os.environ['MASTER_PORT'] = '3456'
warnings.filterwarnings('ignore')
```

```{python}
#| execution: {iopub.execute_input: '2023-06-09T23:43:26.367331Z', iopub.status.busy: '2023-06-09T23:43:26.366987Z', iopub.status.idle: '2023-06-09T23:43:26.657862Z', shell.execute_reply: '2023-06-09T23:43:26.657185Z', shell.execute_reply.started: '2023-06-09T23:43:26.367315Z'}
#| tags: []
# automatically detect and reload local changes to modules
%load_ext autoreload
%autoreload 2

import os
# --------------------------------------
# BE SURE TO GRAB A FRESH GPU !
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
!echo $CUDA_VISIBLE_DEVICES
# --------------------------------------
```

```{python}
#| execution: {iopub.execute_input: '2023-06-09T23:43:28.249635Z', iopub.status.busy: '2023-06-09T23:43:28.249359Z', iopub.status.idle: '2023-06-09T23:43:28.537298Z', shell.execute_reply: '2023-06-09T23:43:28.536622Z', shell.execute_reply.started: '2023-06-09T23:43:28.249617Z'}
#| tags: []
devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
print(devices)
!getconf _NPROCESSORS_ONLN
```

```{python}
#| execution: {iopub.execute_input: '2023-06-09T23:43:29.598477Z', iopub.status.busy: '2023-06-09T23:43:29.598262Z', iopub.status.idle: '2023-06-09T23:43:29.885289Z', shell.execute_reply: '2023-06-09T23:43:29.884622Z', shell.execute_reply.started: '2023-06-09T23:43:29.598460Z'}
#| tags: []
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
os.environ['AUTOGRAPH_VERBOSITY'] = '10'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
!echo $CUDA_VISIBLE_DEVICES
```

```{python}
#| execution: {iopub.execute_input: '2023-06-09T23:45:14.972232Z', iopub.status.busy: '2023-06-09T23:45:14.971789Z', iopub.status.idle: '2023-06-09T23:45:15.112777Z', shell.execute_reply: '2023-06-09T23:45:15.112329Z', shell.execute_reply.started: '2023-06-09T23:45:14.972216Z'}
#| tags: []
from __future__ import absolute_import, print_function, annotations, division
%matplotlib widget

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import set_matplotlib_formats

from l2hmc.main import build_experiment
from l2hmc.utils.plot_helpers import set_plot_style

set_plot_style()
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams['grid.color'] = '#404040'
sns.set(rc={"figure.dpi":100, 'savefig.dpi':300})
sns.set_context('notebook')
sns.set_style("ticks")
set_matplotlib_formats('retina')
plt.rcParams['figure.figsize'] = [12.4, 4.8]
import seaborn as sns
from l2hmc.utils.plot_helpers import set_plot_style
#console = get_console()
#print(console.is_jupyter)
#if console.is_jupyter:
#    console.is_jupyter = False
#print(console.is_jupyter)
```


# Initialize and Build `Experiment` objects:

- The `l2hmc.main` module provides a function `build_experiment`:

```python
def build_experiment(overrides: list[str]) -> tfExperiment | ptExperiment:
    ...
```

which will:

1. Load the default options from `conf/config.yaml`
2. Override the default options with any values provided in `overrides`
3. Parse these options and build an `ExperimentConfig` which uniquely defines an experiment
3. Instantiate / return an `Experiment` from the `ExperimentConfig`.
   Depending on `framework=pytorch|tensorflow`:
    a. `framework=pytorch` -> `l2hmc.experiment.pytorch.Experiment`
    b. `framework=tensorflow` -> `l2hmc.experiment.tensorflow.Experiment`

```python
>>> train_output = experiment.train()
>>> eval_output = experiment.evaluate(job_type='eval')
>>> hmc_output = experiment.evaluate(job_type='hmc')
```

<div class="alert alert-block alert-info" style="background:rgba(102, 102, 102, 0.2);color:rgb(102,102,102); border-radius:5px; border:none">
<b><u>Overriding Defaults</u></b>

Specifics about the training / evaluation / hmc runs can be flexibly overridden by passing arguments to the training / evaluation / hmc runs, respectively
</div>

```{python}
#| execution: {iopub.execute_input: '2023-06-09T23:44:36.377754Z', iopub.status.busy: '2023-06-09T23:44:36.377410Z', iopub.status.idle: '2023-06-09T23:44:36.422048Z', shell.execute_reply: '2023-06-09T23:44:36.421488Z', shell.execute_reply.started: '2023-06-09T23:44:36.377735Z'}
#| tags: []
import numpy as np

seed = np.random.randint(100000)

DEFAULTS = {
    'seed': f'{seed}',
    'init_aim': False,
    'init_wandb': False,
    'use_wandb': False,
    'restore': False,
    'save': False,
    'use_tb': False,
    'dynamics': {
        'nleapfrog': 10,
        'nchains': 4096,
    },
    'conv': 'none',
    'steps': {
        'log': 10,
        'print': 100,
        'nepoch': 1000,
        'nera': 1,
    },
    'annealing_schedule': {
        'beta_init': 4.0,
        'beta_final': 4.0,
    },
}

outputs = {
    'pytorch': {
        'train': {},
        'eval': {},
        'hmc': {},
    },
    'tensorflow': {
        'train': {},
        'eval': {},
        'hmc': {},
    },
}
```

```{python}
#| execution: {iopub.execute_input: '2023-06-09T23:44:36.519723Z', iopub.status.busy: '2023-06-09T23:44:36.519215Z', iopub.status.idle: '2023-06-09T23:44:36.575913Z', shell.execute_reply: '2023-06-09T23:44:36.575353Z', shell.execute_reply.started: '2023-06-09T23:44:36.519700Z'}
#| tags: []
from l2hmc.configs import dict_to_list_of_overrides
OVERRIDES = dict_to_list_of_overrides(DEFAULTS)
```

```{python}
#| execution: {iopub.execute_input: '2023-06-09T23:44:39.269613Z', iopub.status.busy: '2023-06-09T23:44:39.269287Z', iopub.status.idle: '2023-06-09T23:44:49.854308Z', shell.execute_reply: '2023-06-09T23:44:49.853795Z', shell.execute_reply.started: '2023-06-09T23:44:39.269596Z'}
#| scrolled: true
#| tags: []
# Build PyTorch Experiment
ptExpU1 = build_experiment(
    overrides=[
        *OVERRIDES,
        'framework=pytorch',
        'backend=DDP',
        'precision=fp16',
    ]
)
```

```{python}
#| execution: {iopub.execute_input: '2023-06-09T23:44:49.856274Z', iopub.status.busy: '2023-06-09T23:44:49.855670Z', iopub.status.idle: '2023-06-09T23:44:54.622705Z', shell.execute_reply: '2023-06-09T23:44:54.622223Z', shell.execute_reply.started: '2023-06-09T23:44:49.856256Z'}
#| tags: []
# Build TensorFlow Experiment

import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16')

tfExpU1 = build_experiment(
    overrides=[
        *OVERRIDES,
        'framework=tensorflow',
        'backend=horovod',
        'precision=fp16',
    ]
)
```


## PyTorch

### Training

```{python}
#| execution: {iopub.execute_input: '2023-06-09T16:34:37.256239Z', iopub.status.busy: '2023-06-09T16:34:37.255876Z', iopub.status.idle: '2023-06-09T17:28:39.894248Z', shell.execute_reply: '2023-06-09T17:28:39.832714Z', shell.execute_reply.started: '2023-06-09T16:34:37.256222Z'}
#| scrolled: true
#| tags: []
outputs['pytorch']['train'] = ptExpU1.trainer.train()
```

```{python}
#| execution: {iopub.execute_input: '2023-06-09T17:28:40.057111Z', iopub.status.busy: '2023-06-09T17:28:40.056815Z', iopub.status.idle: '2023-06-09T17:31:22.975110Z', shell.execute_reply: '2023-06-09T17:31:22.974612Z', shell.execute_reply.started: '2023-06-09T17:28:40.057094Z'}
#| scrolled: true
#| tags: []
_ = ptExpU1.save_dataset(job_type='train', nchains=32)
```

### Inference

#### Evaluation

```{python}
#| execution: {iopub.execute_input: '2023-06-09T17:31:22.976160Z', iopub.status.busy: '2023-06-09T17:31:22.976006Z', iopub.status.idle: '2023-06-09T17:40:51.136809Z', shell.execute_reply: '2023-06-09T17:40:51.136343Z', shell.execute_reply.started: '2023-06-09T17:31:22.976145Z'}
#| scrolled: true
#| tags: []
outputs['pytorch']['eval'] = ptExpU1.trainer.eval(
    job_type='eval',
    nprint=500,
    nchains=128,
    eval_steps=2000,
)
_ = ptExpU1.save_dataset(job_type='eval', nchains=32)
```

```{python}
for k, v in ptExpSU3.trainer.dy
```

#### HMC

```{python}
#| execution: {iopub.execute_input: '2023-06-09T17:40:51.324057Z', iopub.status.busy: '2023-06-09T17:40:51.323883Z', iopub.status.idle: '2023-06-09T17:43:28.593296Z', shell.execute_reply: '2023-06-09T17:43:28.592758Z', shell.execute_reply.started: '2023-06-09T17:40:51.324042Z'}
#| scrolled: true
#| tags: []
outputs['pytorch']['hmc'] = ptExpU1.trainer.eval(
    job_type='hmc',
    nprint=500,
    nchains=128,
    eval_steps=2000,
)
_ = ptExpU1.save_dataset(job_type='hmc', nchains=32)
```

## TensorFlow

### Train

```{python}
#| execution: {iopub.execute_input: '2023-06-04T23:24:30.130364Z', iopub.status.busy: '2023-06-04T23:24:30.129972Z', iopub.status.idle: '2023-06-04T23:38:15.085564Z', shell.execute_reply: '2023-06-04T23:38:15.084913Z', shell.execute_reply.started: '2023-06-04T23:24:30.130345Z'}
#| scrolled: true
#| tags: []
outputs['tensorflow']['train'] = tfExpU1.trainer.train()
_ = tfExpU1.save_dataset(job_type='train', nchains=32)
```

### Inference

#### Evaluate

```{python}
#| execution: {iopub.execute_input: '2023-06-05T00:02:46.693758Z', iopub.status.busy: '2023-06-05T00:02:46.677442Z', iopub.status.idle: '2023-06-05T00:08:53.740184Z', shell.execute_reply: '2023-06-05T00:08:53.739550Z', shell.execute_reply.started: '2023-06-05T00:02:46.693726Z'}
#| scrolled: true
#| tags: []
outputs['tensorflow']['eval'] = tfExpU1.trainer.eval(
    job_type='eval',
    nprint=500,
    nchains=128,
    eval_steps=2000,
)
_ = tfExpU1.save_dataset(job_type='eval', nchains=32)
```

#### HMC

```{python}
#| execution: {iopub.execute_input: '2023-06-05T00:08:53.741493Z', iopub.status.busy: '2023-06-05T00:08:53.741292Z', iopub.status.idle: '2023-06-05T00:19:05.197152Z', shell.execute_reply: '2023-06-05T00:19:05.196435Z', shell.execute_reply.started: '2023-06-05T00:08:53.741477Z'}
#| scrolled: true
#| tags: []
outputs['tensorflow']['hmc'] = tfExpU1.trainer.eval(
    job_type='hmc',
    nprint=500,
    nchains=128,
    eval_steps=2000,
)
_ = tfExpU1.save_dataset(job_type='hmc', nchains=32)
```

# Model Performance

Our goal is improving the efficiency of our MCMC sampler.

In particular, we are interested in generating **independent** save_datasetrations which we can then use to calculate expectation values of physical observables.

For our purposes, we are interested in obtaining lattice configurations from distinct _topological charge sectors_, as characterized by a configurations _topological charge_, $Q$.

HMC is known to suffer from _critical slowing down_, a phenomenon in which our configurations remains stuck in some local topological charge sector and fails to produce distinct configurations.

In particular, it is known that the integrated autocorrelation time of the topological charge $\tau$ grows exponentially with decreasing lattice spacing (i.e. continuum limit), making this theory especially problematic.

Because of this, we can assess our models' performance by looking at the _tunneling rate_, i.e. the rate at which our sampler jumps between these different charge sectors.

We can write this quantity as:

$$
\delta Q = |Q^{(i)} - Q^{(i-1)}|
$$

where we look at the difference in the topological charge between sequential configurations.

<div class="alert alert-block alert-info" style="background:rgba(34,139,230,0.1); color: rgb(34,139,230); border: 0px solid; border-radius:5px;">
<b>Note:</b> 
The efficiency of our sampler is directly proportional to the tunneling rate, which is inversely proportional to the integrated autocorrelation time $\tau$, i.e.

&nbsp;

$$
\text{Efficiency} \propto \delta Q \propto \frac{1}{\tau}
$$

Explicitly, this means that the **more efficient** the model $\longrightarrow$

- the **larger** tunneling rate
- the **smaller** integrated autocorrelation time for $Q$
</div>

```{python}
#| execution: {iopub.execute_input: '2023-06-05T00:19:05.201857Z', iopub.status.busy: '2023-06-05T00:19:05.201719Z', iopub.status.idle: '2023-06-05T00:19:07.180876Z', shell.execute_reply: '2023-06-05T00:19:07.180294Z', shell.execute_reply.started: '2023-06-05T00:19:05.201835Z'}
#| tags: []
import xarray as xr

def get_thermalized_configs(
        x: np.ndarray | xr.DataArray,
        drop: int = 5
) -> np.ndarray | xr.DataArray:
    """Drop the first `drop` states across all chains.

    x.shape = [draws, chains]
    """
    if isinstance(x, np.ndarray):
        return np.sort(x)[..., :-drop]
    if isinstance(x, xr.DataArray):
        return x.sortby(
            ['chain', 'draw'],
            ascending=False
        )[..., :-drop]
    raise TypeError
```

# Comparisons

We can measure our models' performance explicitly by looking at the average tunneling rate, $\delta Q_{\mathbb{Z}}$, for our **trained model** and comparing it against generic HMC.

Recall,

$$\delta Q_{\mathbb{Z}} := \big|Q^{(i+1)}_{\mathbb{Z}} - Q^{(i)}_{\mathbb{Z}}\big|$$

where a **higher** value of $\delta Q_{\mathbb{Z}}$ corresponds to **better** tunneling of the topological charge, $Q_{\mathbb{Z}}$.

Note that we can get a concise representation of the data from different parts of our run via:

Note that the data from each of the different parts of our experiment (i.e. `train`, `eval`, and `hmc`) are stored as a dict, e.g.

```python
>>> list(ptExpU1.trainer.histories.keys())
['train', 'eval', 'hmc']
>>> train_history = ptExpU1.trainer.histories['train']
>>> train_dset = train_history.get_dataset()
>>> assert isinstance(train_history, l2hmc.utils.history.BaseHistory)
>>> assert isinstance(train_dset, xarray.Dataset)
```

(see below, for example)

We aggregate the data into the `dsets` dict below, grouped by:

1. **Framework** (`pytorch` / `tensorflow`)
2. **Job type** (`train`, `eval`, `hmc`)

```{python}
#| execution: {iopub.execute_input: '2023-06-05T00:19:07.183321Z', iopub.status.busy: '2023-06-05T00:19:07.183128Z', iopub.status.idle: '2023-06-05T00:19:07.786468Z', shell.execute_reply: '2023-06-05T00:19:07.785897Z', shell.execute_reply.started: '2023-06-05T00:19:07.183306Z'}
#| tags: []
import logging
log = logging.getLogger(__name__)
dsets = {}
fws = ['pt', 'tf']
modes = ['train', 'eval', 'hmc']
for fw in fws:
    dsets[fw] = {}
    for mode in modes:
        hist = None
        if fw == 'pt':
            hist = ptExpU1.trainer.histories.get(mode, None)
        elif fw == 'tf':
            hist = tfExpU1.trainer.histories.get(mode, None)
        if hist is not None:
            log.info(f'Getting dataset for {fw}: {mode}')
            dsets[fw][mode] = hist.get_dataset()
```

```{python}
#| execution: {iopub.execute_input: '2023-06-05T00:19:07.787421Z', iopub.status.busy: '2023-06-05T00:19:07.787276Z', iopub.status.idle: '2023-06-05T00:19:08.386614Z', shell.execute_reply: '2023-06-05T00:19:08.386008Z', shell.execute_reply.started: '2023-06-05T00:19:07.787407Z'}
#| tags: []
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from l2hmc.utils.plot_helpers import COLORS, set_plot_style

set_plot_style()

fig, ax = plt.subplots(figsize=(16, 3), ncols=2)

# ---------------------------------------------
# ---- DROP FIRST 20% FOR THERMALIZATION ------
# ---------------------------------------------
KEEP = int(0.8 * len(dsets['tf']['eval'].draw))
dqpte = get_thermalized_configs(dsets['pt']['eval']['dQint'].astype('int'))
dqpth = get_thermalized_configs(dsets['pt']['hmc']['dQint'].astype('int'))

dqtfe = get_thermalized_configs(dsets['tf']['eval']['dQint'].astype('int'))
dqtfh = get_thermalized_configs(dsets['tf']['hmc']['dQint'].astype('int'))

_ = sns.distplot(
    dqpte.sum('chain'),
    kde=False,
    color=COLORS['blue'],
    label='Eval',
    ax=ax[0]
)
_ = sns.distplot(
    dqpth.sum('chain'),
    kde=False,
    color=COLORS['red'],
    label='HMC',
    ax=ax[0]
)

_ = ax[0].set_title('PyTorch')
_ = ax[0].set_xlabel(
    f'# tunneling events / {dqpte.shape[-1]} configurations'
)
_ = ax[0].legend(loc='best', frameon=False)
plt.legend()

_ = sns.distplot(
    dqtfe.sum('chain'),
    kde=False,
    color=COLORS['blue'],
    label='Eval',
    ax=ax[1]
)
_ = sns.distplot(
    dqtfh.sum('chain'),
    kde=False,
    color=COLORS['red'],
    label='HMC',
    ax=ax[1]
)

_ = ax[1].set_title('TensorFlow')
_ = ax[1].set_xlabel(
    #r"""$\sum_{i=0} \left|\delta Q_{i}\right|$""",
    #fontsize='large',
    f'# tunneling events / {dqpte.shape[-1]} configurations'
)
_ = ax[1].legend(loc='best', frameon=False)
```

## TensorFlow Results

```{python}
#| execution: {iopub.execute_input: '2023-06-05T00:19:08.387797Z', iopub.status.busy: '2023-06-05T00:19:08.387409Z', iopub.status.idle: '2023-06-05T00:19:08.637720Z', shell.execute_reply: '2023-06-05T00:19:08.637141Z', shell.execute_reply.started: '2023-06-05T00:19:08.387778Z'}
#| tags: []
import rich
```

```{python}
#| execution: {iopub.execute_input: '2023-06-05T00:19:08.638688Z', iopub.status.busy: '2023-06-05T00:19:08.638535Z', iopub.status.idle: '2023-06-05T00:19:09.305203Z', shell.execute_reply: '2023-06-05T00:19:09.304680Z', shell.execute_reply.started: '2023-06-05T00:19:08.638673Z'}
#| tags: []
sns.set_context('notebook')
ndraws = len(dsets['tf']['eval']['dQint'].draw)
drop = int(0.1 * ndraws)
keep = int(0.9 * ndraws)

dqe = dsets['tf']['eval']['dQint'][:, -90:]
dqh = dsets['tf']['hmc']['dQint'][:, -90:]

etot = dqe.astype(int).sum()
htot = dqh.astype(int).sum()

fsize = plt.rcParams['figure.figsize']
figsize = (2.5 * fsize[0], fsize[1])
fig, ax = plt.subplots(figsize=figsize, ncols=2)
_ = dqe.astype(int).plot(ax=ax[0])
_ = dqh.astype(int).plot(ax=ax[1])
_ = ax[0].set_title(f'Eval, total: {etot.values}', fontsize='x-large');
_ = ax[1].set_title(f'HMC, total: {htot.values}', fontsize='x-large');
_ = fig.suptitle(fr'TensorFlow Improvement: {100*(etot / htot):3.0f}%', fontsize='x-large')

log.info(f"TensorFlow, EVAL\n dQint.sum('chain'):\n {dqe.astype(int).sum('chain').T}")
log.info(f"dQint.sum(): {dqe.astype(int).sum().T}")
log.info(f"TensorFlow, HMC\n dQint.sum('chain'):\n {dqh.astype(int).sum('chain').T}")
log.info(f"dQint.sum(): {dqh.astype(int).sum().T}")
```

### PyTorch Results

```{python}
#| execution: {iopub.execute_input: '2023-06-05T00:19:09.306297Z', iopub.status.busy: '2023-06-05T00:19:09.305918Z', iopub.status.idle: '2023-06-05T00:19:09.944283Z', shell.execute_reply: '2023-06-05T00:19:09.943718Z', shell.execute_reply.started: '2023-06-05T00:19:09.306281Z'}
#| tags: []
sns.set_context('notebook')
ndraws = len(dsets['pt']['eval']['dQint'].draw)
drop = int(0.1 * ndraws)
keep = int(0.9 * ndraws)

dqe = dsets['pt']['eval']['dQint'][:, -90:]
dqh = dsets['pt']['hmc']['dQint'][:, -90:]

etot = dqe.astype(int).sum()
htot = dqh.astype(int).sum()

fsize = plt.rcParams['figure.figsize']
figsize = (2.5 * fsize[0], 0.8 * fsize[1])
fig, ax = plt.subplots(figsize=figsize, ncols=2)
_ = dqe.astype(int).plot(ax=ax[0])
_ = dqh.astype(int).plot(ax=ax[1])
_ = ax[0].set_title(f'Eval, total: {etot.values}', fontsize='x-large');
_ = ax[1].set_title(f'HMC, total: {htot.values}', fontsize='x-large');
_ = fig.suptitle(fr'PyTorch Improvement: {100*(etot / htot):3.0f}%', fontsize='x-large')

log.info(60 * '-')
log.info(f"PyTorch, EVAL\n dQint.sum('chain'):\n {dqe.astype(int).sum('chain').T.values}")
log.info(f"dQint.sum(): {dqe.astype(int).sum().T.values}")
log.info(60 * '-')
log.info(f"PyTorch, HMC\n dQint.sum('chain'):\n {dqh.astype(int).sum('chain').T.values}")
log.info(f"dQint.sum(): {dqh.astype(int).sum().T.values}")
```

## Comparisons

```{python}
#| execution: {iopub.execute_input: '2023-06-05T00:19:09.945445Z', iopub.status.busy: '2023-06-05T00:19:09.945063Z', iopub.status.idle: '2023-06-05T00:19:13.970371Z', shell.execute_reply: '2023-06-05T00:19:13.969758Z', shell.execute_reply.started: '2023-06-05T00:19:09.945428Z'}
#| tags: []
import matplotlib.pyplot as plt
from l2hmc.utils.plot_helpers import set_plot_style, COLORS

import seaborn as sns
set_plot_style()
plt.rcParams['axes.linewidth'] = 2.0
sns.set_context('notebook')
figsize = plt.rcParamsDefault['figure.figsize']
plt.rcParams['figure.dpi'] = plt.rcParamsDefault['figure.dpi']

for idx in range(4):
    fig, (ax, ax1) = plt.subplots(
        ncols=2,
        #nrows=4,
        figsize=(3. * figsize[0], figsize[1]),
    )
    _ = ax.plot(
        dsets['pt']['eval'].intQ[idx],  # .dQint.mean('chain')[100:],
        color=COLORS['red'],
        ls=':',
        label='Trained',
        lw=1.5,
    );

    _ = ax.plot(
        dsets['pt']['hmc'].intQ[idx],  # .dQint.mean('chain')[100:],
        ls='-',
        label='HMC',
        color='#666666',
        zorder=5,
        lw=2.0,
    );

    _ = ax1.plot(
        dsets['tf']['eval'].intQ[idx],  # .dQint.mean('chain')[-100:],
        color=COLORS['blue'],
        ls=':',
        label='Trained',
        lw=1.5,

    );
    _ = ax1.plot(
        dsets['tf']['hmc'].intQ[idx],  # .dQint.mean('chain')[-100:],
        color='#666666',
        ls='-',
        label='HMC',
        zorder=5,
        lw=2.0,
    );
    _ = ax.set_title('PyTorch', fontsize='x-large')
    _ = ax1.set_title('TensorFlow', fontsize='x-large')
    #_ = ax1.set_ylim(ax.get_ylim())
    _ = ax.grid(True, alpha=0.4)
    _ = ax1.grid(True, alpha=0.4)
    _ = ax.set_xlabel('MD Step', fontsize='large')
    _ = ax1.set_xlabel('MD Step', fontsize='large')
    _ = ax.set_ylabel('dQint', fontsize='large')
    _ = ax.legend(loc='best', framealpha=0.0, ncol=2, labelcolor='#666666')
    _ = ax1.legend(loc='best', framealpha=0.0, ncol=2, labelcolor='#666666')
```

