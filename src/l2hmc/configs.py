"""
config.py

Implements various configuration objects
"""
from __future__ import absolute_import, annotations, division, print_function
from collections import namedtuple
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hydra.core.config_store import ConfigStore
import numpy as np
from omegaconf import DictConfig
from omegaconf import MISSING

# from accelerate.accelerator import Accelerator
# from hydra.utils import instantiate


log = logging.getLogger(__name__)


HERE = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = HERE.parent.parent
CONF_DIR = HERE.joinpath('conf')
LOGS_DIR = PROJECT_DIR.joinpath('logs')
AIM_DIR = HERE.joinpath('.aim')
OUTPUTS_DIR = HERE.joinpath('outputs')

CONF_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
OUTDIRS_FILE = OUTPUTS_DIR.joinpath('outdirs.log')


State = namedtuple('State', ['x', 'v', 'beta'])

MonteCarloStates = namedtuple('MonteCarloStates', ['init', 'proposed', 'out'])


def add_to_outdirs_file(outdir: os.PathLike):
    with open(OUTDIRS_FILE, 'a') as f:
        f.write('\n')
        f.write(Path(outdir).resolve().as_posix())


def get_jobdir(cfg: DictConfig, job_type: str) -> Path:
    jobdir = Path(cfg.get('outdir', os.getcwd())).joinpath(job_type)
    jobdir.mkdir(exist_ok=True, parents=True)
    assert jobdir is not None
    add_to_outdirs_file(jobdir)
    return jobdir


def list_to_str(x: list) -> str:
    if isinstance(x[0], int):
        return '-'.join([str(int(i)) for i in x])
    elif isinstance(x[0], float):
        return '-'.join([f'{i:2.1f}' for i in x])
    else:
        return '-'.join([str(i) for i in x])


@dataclass
class BaseConfig:
    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    def get_config(self) -> dict:
        return asdict(self)

    def asdict(self) -> dict:
        return asdict(self)

    def to_dict(self):
        return deepcopy(self.__dict__)

    def to_file(self, fpath: os.PathLike) -> None:
        with open(fpath, 'w') as f:
            json.dump(self.to_json(), f, indent=4)

    def from_file(self, fpath: os.PathLike) -> None:
        with open(fpath, 'w') as f:
            with open(fpath, 'r') as f:
                config = json.load(f)

        self.__init__(**config)

    def __getitem__(self, key):
        return super().__getattribute__(key)


@dataclass
class Charges:
    intQ: Any
    sinQ: Any


@dataclass
class LatticeMetrics:
    plaqs: Any
    charges: Charges
    p4x4: Any

    def asdict(self) -> dict:
        return {
            'plaqs': self.plaqs,
            'sinQ': self.charges.sinQ,
            'intQ': self.charges.intQ,
            'p4x4': self.p4x4,
        }


@dataclass
class AcceleratorConfig(BaseConfig):
    device_placement: Optional[bool] = True
    split_batches: Optional[bool] = False
    mixed_precision: Optional[str] = 'no'
    cpu: Optional[bool] = False
    deepspeed_plugin: Optional[Any] = None
    # fsdp_plugin: Optional[Any] = None
    rng_types: Optional[list[str]] = None
    log_with: Optional[list[str]] = None
    logging_dir: Optional[str | os.PathLike] = None
    dispatch_batches: Optional[bool] = False
    step_scheduler_with_optimizer: Optional[bool] = True
    kwargs_handlers: Optional[list[Any]] = None


@dataclass
class wandbSetup(BaseConfig):
    id: Optional[str] = None
    group: Optional[str] = None
    save_code: Optional[bool] = True
    sync_tensorboard: Optional[bool] = True
    tags: Optional[list[str]] = None
    mode: Optional[str] = 'online'
    resume: Optional[str] = 'allow'
    entity: Optional[str] = 'l2hmc-qcd'
    project: Optional[str] = 'l2hmc-qcd'
    settings: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        if self.settings is None:
            self.settings = {'start_method': 'thread'}


@dataclass
class wandbConfig(BaseConfig):
    setup: wandbSetup


@dataclass
class U1Config(BaseConfig):
    steps: Steps
    network: NetworkConfig
    dynamics: DynamicsConfig
    loss: LossConfig
    net_weights: NetWeights
    # conv: Optional[ConvolutionConfig] = None
    backend: str = MISSING

    def __post_init__(self):
        self.xshape = self.dynamics.xshape
        xdim = self.dynamics.xdim
        self.input_spec = InputSpec(
            xshape=self.dynamics.xshape,  # type:ignore
            xnet={'x': [xdim, int(2)], 'v': [xdim, ]},
            vnet={'x': [xdim, int(2)], 'v': [xdim, ]}
        )


@dataclass
class NetWeight(BaseConfig):
    """Object for selectively scaling different components of learned fns.

    Explicitly,
     - s: scales the v (x) scaling function in the v (x) updates
     - t: scales the translation function in the update
     - q: scales the force (v) transformation function in the v (x) updates
    """
    s: float = 1.
    t: float = 1.
    q: float = 1.

    def to_dict(self):
        return {'s': self.s, 't': self.t, 'q': self.q}

    def to_str(self):
        return f's{self.s:2.1f}t{self.t:2.1f}q{self.t:2.1f}'


@dataclass
class NetWeights(BaseConfig):
    """Object for selectively scaling different components of x, v networks."""
    x: NetWeight = NetWeight(1., 1., 1.)
    v: NetWeight = NetWeight(1., 1., 1.)

    def to_dict(self):
        return {
            'x': self.x.to_dict(),
            'v': self.v.to_dict(),
        }

    def __post_init__(self):
        if not isinstance(self.x, NetWeight):
            self.x = NetWeight(**self.x)
        if not isinstance(self.v, NetWeight):
            self.v = NetWeight(**self.v)


@dataclass
class LearningRateConfig(BaseConfig):
    """Learning rate configuration object."""
    lr_init: float = 1e-3
    mode: str = 'auto'
    monitor: str = 'loss'
    patience: int = 5
    cooldown: int = 0
    warmup: int = 1000
    verbose: bool = True
    min_lr: float = 1e-6
    factor: float = 0.98
    min_delta: float = 1e-4
    clip_norm: float = 2.0
    # decay_steps: int = -1
    # decay_rate: float = 1.0
    # warmup_steps: int = 100
    # min_lr: float = 1e-5
    # patience: int = 5

    def to_str(self):
        return f'lr-{self.lr_init:3.2f}'


@dataclass
class AnnealingSchedule(BaseConfig):
    beta_init: float
    beta_final: Optional[float] = 1.0
    # steps: Steps
    # TODO: Add methods for specifying different annealing schedules

    def __post_init__(self):
        if self.beta_final is None or self.beta_final < self.beta_init:
            log.warning(
                f'AnnealingSchedule.beta_final must be >= {self.beta_init},'
                f' but received: {self.beta_final}.\n'
                f'Setting self.beta_final to {self.beta_init}'
            )
            self.beta_final = float(self.beta_init)
        assert (
            isinstance(self.beta_final, float)
            and self.beta_final >= self.beta_init
        )

    def setup(self, steps: Steps) -> None:
        if self.beta_final is None:
            self.beta_final = self.beta_init

        betas = np.linspace(self.beta_init, self.beta_final, steps.nera)
        self.betas = {
            str(era): betas[era] for era in range(steps.nera)
        }


@dataclass
class ConvolutionConfig(BaseConfig):
    filters: List[int]
    sizes: List[int]
    pool: List[int]
    # activation: str
    # paddings: list[int]

    def to_str(self):
        outstr = [
            list_to_str(self.filters),
            list_to_str(self.sizes),
            list_to_str(self.pool)
        ]

        return '_'.join(outstr)


@dataclass
class NetworkConfig(BaseConfig):
    units: List[int]
    activation_fn: str
    dropout_prob: float
    use_batch_norm: bool = True
    # conv_config: Optional[ConvolutionConfig] = None

    def to_str(self):
        ustr = ''.join([str(int(i)) for i in self.units])
        outstr = [f'nh-{ustr}_act-{self.activation_fn}']
        if self.dropout_prob > 0:
            outstr.append(f'dp-{self.dropout_prob:2.1f}')
        if self.use_batch_norm:
            outstr.append('bNorm')

        return '_'.join(outstr)


@dataclass
class DynamicsConfig(BaseConfig):
    nchains: int
    group: str
    latvolume: List[int]
    # xshape: List[int]
    nleapfrog: int
    eps: float = 0.01
    eps_hmc: float = 0.01
    use_ncp: bool = True
    verbose: bool = True
    eps_fixed: bool = False
    use_split_xnets: bool = True
    use_separate_networks: bool = True
    merge_directions: bool = True

    def __post_init__(self):
        assert self.group.upper() in ['U1', 'SU3']
        if self.eps_hmc is None:
            # if not specified, use a trajectory length of 1
            self.eps_hmc = 1.0 / self.nleapfrog
        if self.group.upper() == 'U1':
            self.dim = 2
            self.nt, self.nx = self.latvolume
            self.xshape = (self.nchains, self.dim, *self.latvolume)
            assert len(self.xshape) == 4
            assert len(self.latvolume) == 2
            self.xdim = int(np.cumprod(self.xshape[1:])[-1])
        elif self.group.upper() == 'SU3':
            self.dim = 4
            self.link_shape = (3, 3)
            self.nt, self.nx, self.ny, self.nz = self.latvolume
            self.xshape = (
                self.nchains,
                self.dim,
                *self.latvolume,
                *self.link_shape
            )
            assert len(self.xshape) == 8
            assert len(self.latvolume) == 4
            self.xdim = self.nt * self.nx * self.ny * self.nz * self.dim * 8
        else:
            raise ValueError('Expected `group` to be one of `"U1", "SU3"`')

    # def get_xshape(self):
    #     if self.group.upper() == 'U1':
    #         return (self.nchains, self.dim, *self.latvolume)
    #     elif self.group.upper() == 'SU3':
    #         return (
    #             self.nchains,
    #             self.dim,
    #             *self.latvolume,
    #             *self.link_shape
    #         )


@dataclass
class LossConfig(BaseConfig):
    use_mixed_loss: bool = False
    charge_weight: float = 0.01
    plaq_weight: float = 0.
    aux_weight: float = 0.0


@dataclass
class Steps:
    nera: int
    nepoch: int
    test: int
    log: Optional[int] = None
    print: Optional[int] = None
    extend_last_era: Optional[int] = None

    def __post_init__(self):
        if self.extend_last_era is None:
            self.extend_last_era = 1
        self.total = self.nera * self.nepoch
        if self.total < 1000:
            self.log = 5
            self.print = max(1, int(self.nepoch // 10))
        else:
            if self.log is None:
                self.log = max(1, int(self.total // 1000))
                # self.log = max(1, int(self.nepoch // 10))

            if self.print is None:
                self.print = max(1, int(self.nepoch // 10))

        assert isinstance(self.log, int)
        assert isinstance(self.print, int)
        self.log = max(1, int(self.log))
        self.print = max(1, int(self.print))


@dataclass
class InputSpec(BaseConfig):
    xshape: List[int] | Tuple[int]
    xnet: Optional[Dict[str, List[int] | Tuple[int]]] = None
    vnet: Optional[Dict[str, List[int] | Tuple[int]]] = None

    def __post_init__(self):
        if len(self.xshape) == 2:
            self.xdim = self.xshape[-1]
        elif len(self.xshape) > 2:
            self.xdim = np.cumprod(self.xshape[1:])[-1]
        else:
            raise ValueError(f'Invalid `xshape`: {self.xshape}')

        if self.xnet is None:
            self.xnet = {'x': self.xshape, 'v': self.xshape}
        if self.vnet is None:
            self.vnet = {'x': self.xshape, 'v': self.xshape}


@dataclass
class TrainerConfig1:
    steps: Steps
    lr: LearningRateConfig
    dynamics: DynamicsConfig
    schedule: AnnealingSchedule
    compile: bool = True
    aux_weight: float = 0.0
    evals_per_step: int = 1
    compression: Optional[str] = 'none'
    keep: Optional[str | list[str]] = None
    skip: Optional[str | list[str]] = None


@dataclass
class ExperimentConfig:
    framework: str
    seed: int
    steps: Steps
    loss: LossConfig
    network: NetworkConfig
    net_weights: NetWeights
    dynamics: DynamicsConfig
    annealing_schedule: AnnealingSchedule
    learning_rate: LearningRateConfig
    wandb: Any
    # ----- optional below -------------------
    # outdir: Optional[Any] = None
    conv: Optional[ConvolutionConfig] = None
    c1: Optional[float] = 0.0                # rectangle coefficient for SU3
    width: Optional[int] = None
    nchains: Optional[int] = None
    profile: Optional[bool] = False
    debug_mode: Optional[bool] = False
    default_mode: Optional[bool] = True
    print_config: Optional[bool] = True
    precision: Optional[str] = 'float32'
    ignore_warnings: Optional[bool] = True
    compile: Optional[bool] = True
    name: Optional[str] = None

    def __post_init__(self):
        if self.debug_mode:
            self.compile = False

        self.annealing_schedule.setup(self.steps)
        w = int(os.environ.get('COLUMNS', 235))
        self.width = w if self.width is None else self.width
        self.xdim = self.dynamics.xdim
        self.xshape = self.dynamics.xshape
        log.warning(f'xdim: {self.dynamics.xdim}')
        log.warning(f'group: {self.dynamics.group}')
        log.warning(f'xshape: {self.dynamics.xshape}')
        log.warning(f'latvolume: {self.dynamics.latvolume}')


def get_config(overrides: Optional[list[str]] = None):
    from hydra import (
        initialize_config_dir,
        compose
    )
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    overrides = [] if overrides is None else overrides
    with initialize_config_dir(
            CONF_DIR.absolute().as_posix(),
            version_base=None,
    ):
        cfg = compose('config', overrides=overrides)

    return cfg


def get_experiment(overrides: Optional[list[str]] = None):
    cfg = get_config(overrides)
    if cfg.framework == 'pytorch':
        from l2hmc.experiment.pytorch.experiment import Experiment
        return Experiment(cfg)
    elif cfg.framework == 'tensorflow':
        from l2hmc.experiment.tensorflow.experiment import Experiment
        return Experiment(cfg)
    else:
        raise ValueError(
            f'Unexpected value for `cfg.framework: {cfg.framework}'
        )


defaults = [
    {'backend': MISSING}
]

cs = ConfigStore.instance()
cs.store(
    name='experiment_config',
    node=ExperimentConfig,
)
cs.store(
    name='dynamics_config',
    node=DynamicsConfig,
)
cs.store(
    name='network_config',
    node=NetworkConfig
)
cs.store(
    name='loss_config',
    node=LossConfig,
)
cs.store(
    name='net_weights',
    node=NetWeights,
)
cs.store(
    name='annealing_schedule',
    node=AnnealingSchedule,
)
