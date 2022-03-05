"""
config.py

Implements various configuration objects
"""
from __future__ import absolute_import, annotations, division, print_function
from collections import namedtuple
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hydra.core.config_store import ConfigStore
import numpy as np
from omegaconf import MISSING


log = logging.getLogger(__name__)


HERE = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = HERE.parent.parent
CONF_DIR = HERE.joinpath('conf')
LOGS_DIR = PROJECT_DIR.joinpath('logs')


State = namedtuple('State', ['x', 'v', 'beta'])

MonteCarloStates = namedtuple('MonteCarloStates', ['init', 'proposed', 'out'])


def list_to_str(x: list) -> str:
    if isinstance(x[0], int):
        return '-'.join([str(int(i)) for i in x])
    elif isinstance(x[0], float):
        return '-'.join([f'{i:2.1g}' for i in x])
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


defaults = [
    {'backend': MISSING}
]


@dataclass
class ExperimentConfig(BaseConfig):
    framework: str
    steps: Steps
    dynamics: DynamicsConfig
    loss: LossConfig
    network: NetworkConfig
    conv: ConvolutionConfig
    net_weights: NetWeights
    schedule: AnnealingSchedule
    learning_rate: LearningRateConfig

    def __post_init__(self):
        self.schedule.setup(self.steps)


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
            xshape=self.dynamics.xshape,
            xnet={'x': [xdim, int(2)], 'v': [xdim, ]},
            vnet={'x': [xdim, ], 'v': [xdim, ]}
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
        return f's{self.s:2.1g}t{self.t:2.1g}q{self.t:2.1g}'


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
        # if self.x is None:
        #     self.x = NetWeight(s=1., t=1., q=1.)
        # if self.v is None:
        #     self.v = NetWeight(s=1., t=1., q=1.)


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
    # decay_steps: int = -1
    # decay_rate: float = 1.0
    # warmup_steps: int = 100
    # min_lr: float = 1e-5
    # patience: int = 5

    def to_str(self):
        return f'lr-{self.lr_init:3.2g}'


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
class TrainingConfig(BaseConfig):
    lr_config: LearningRateConfig
    annealing_schedule: AnnealingSchedule


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
            outstr.append(f'dp-{self.dropout_prob:2.1g}')
        if self.use_batch_norm:
            outstr.append('bNorm')

        return '_'.join(outstr)


@dataclass
class DynamicsConfig(BaseConfig):
    xshape: List[int]
    nleapfrog: int
    eps: float = 0.01
    use_ncp: bool = True
    verbose: bool = False
    eps_fixed: bool = False
    use_split_xnets: bool = True
    use_separate_networks: bool = True
    merge_directions: bool = False

    def __post_init__(self):
        assert len(self.xshape) == 4
        self.nchains, self.nt, self.nx, self.dim = self.xshape
        self.xdim = int(np.cumprod(self.xshape[1:])[-1])


# @dataclass
# class wandbConfig:
#     entity: str
#     project: str
#     dir: Optional[str] = None
#     name: Optional[str] = None
#     group: Optional[str] = None
#     resume: Optional[str] = None
#     config: Optional[dict] = None
#     job_type: Optional[str] = None
#     tags: Optional[list[str]] = None
#     tensorboard: Optional[str] = None
#     sync_tensorboard: Optional[bool] = None

#     def set_job_type(self, job_type: str) -> None:
#         self.job_type = job_type


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

    def __post_init__(self):
        self.total = self.nera * self.nepoch
        if self.log is None:
            self.log = max(1, int(self.nepoch // 10))

        if self.print is None:
            self.print = max(1, int(self.nepoch // 5))

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


# def register_configs() -> None:
cs = ConfigStore()
cs.store(name='config_schema', node=ExperimentConfig)
cs.store(
    group="steps",
    name="steps",
    node=Steps,
)
cs.store(
    group="dynamics",
    name="dynamics",
    node=DynamicsConfig,
)
cs.store(
    group="loss",
    name="loss",
    node=LossConfig,
)
cs.store(
    group='network',
    name='network',
    node=NetworkConfig,
)
cs.store(
    group='conv',
    name='conv',
    node=ConvolutionConfig,
)
cs.store(
    group="net_weights",
    name="net_weights",
    node=NetWeights,
)
cs.store(
    group="annealing_schedule",
    name="annealing_schedule",
    node=AnnealingSchedule,
)
cs.store(
    group="learning_rate",
    name="learning_rate",
    node=LearningRateConfig,
)
