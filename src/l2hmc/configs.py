"""
config.py

Implements various configuration objects
"""
from __future__ import absolute_import, annotations, division, print_function
from collections import namedtuple
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import NamedTuple, Optional, List, Tuple, Dict
from omegaconf import MISSING

from hydra.core.config_store import ConfigStore
import numpy as np


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


class NetWeight(NamedTuple):
    """Object for selectively scaling different components of learned fns.

    Explicitly,
     - s: scales the v (x) scaling function in the v (x) updates
     - t: scales the translation function in the update
     - q: scales the force (v) transformation function in the v (x) updates
    """
    s: float = 1.
    t: float = 1.
    q: float = 1.

    def to_str(self):
        return f's{self.s:2.1g}t{self.t:2.1g}q{self.t:2.1g}'


@dataclass
class NetWeights(BaseConfig):
    """Object for selectively scaling different components of x, v networks."""
    x: Optional[NetWeight] = None
    v: Optional[NetWeight] = None

    def __post_init__(self):
        if self.x is None:
            self.x = NetWeight(s=1., t=1., q=1.)
        if self.v is None:
            self.v = NetWeight(s=1., t=1., q=1.)


@dataclass
class LearningRateConfig(BaseConfig):
    """Learning rate configuration object."""
    lr_init: float
    decay_steps: int = -1
    decay_rate: float = 1.0
    warmup_steps: int = 100

    def to_str(self):
        return f'lr-{self.lr_init:3.2g}'


@dataclass
class AnnealingSchedule(BaseConfig):
    beta_init: float
    beta_final: float
    steps: Steps
    # TODO: Add methods for specifying different annealing schedules

    def __post_init__(self):
        betas = np.linspace(self.beta_init, self.beta_final, self.steps.nera)
        self.betas = {
            str(era): betas[era] for era in range(self.steps.nera)
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
    log: int = 0
    print: int = 0

    def __post_init__(self):
        self.total = self.nera * self.nepoch
        self.log = self.nepoch // 20 if self.log == 0 else self.log
        self.print = self.nepoch // 10 if self.print == 0 else self.print


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
cs = ConfigStore.instance()
cs.store(
    group="dynamics",
    name="dynamics",
    node=DynamicsConfig,
)
cs.store(
    group="steps",
    name="steps",
    node=Steps,
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
    group="loss",
    name="loss",
    node=LossConfig,
)
# # cs.store(
# #     group="net_weights",
# #     name="net_weights",
# #     node=NetWeights,
# # )
