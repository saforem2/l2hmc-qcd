"""
config.py

Implements various configuration objects
"""
from __future__ import absolute_import, annotations, division, print_function
from collections import namedtuple
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import NamedTuple, Optional

from hydra.core.config_store import ConfigStore
import numpy as np


SRC = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = SRC.parent
LOGS_DIR = PROJECT_DIR.joinpath('logs')


Shape = list[int]

State = namedtuple('State', ['x', 'v', 'beta'])

MonteCarloStates = namedtuple('MonteCarloStates', ['init', 'proposed', 'out'])


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
    x: Optional[NetWeight] = field(default_factory=NetWeight)
    v: Optional[NetWeight] = field(default_factory=NetWeight)

    def __post_init__(self):
        if self.x is None:
            self.x = NetWeight(1., 1., 1.)
            self.v = NetWeight(1., 1., 1.)

    def to_str(self):
        assert self.x is not None and self.v is not None
        return f'xNW-{self.x.to_str()}_vNW-{self.v.to_str()}'


@dataclass
class LearningRateConfig(BaseConfig):
    """Learning rate configuration object."""
    lr_init: float
    warmup_steps: int = 0
    decay_steps: int = -1
    decay_rate: float = 1.0

    def to_str(self):
        return f'lr-{self.lr_init:3.2g}'


def list_to_str(x: list) -> str:
    if isinstance(x[0], int):
        return '-'.join([str(int(i)) for i in x])
    elif isinstance(x[0], float):
        return '-'.join([f'{i:2.1g}' for i in x])
    else:
        return '-'.join([str(i) for i in x])


@dataclass
class ConvolutionConfig(BaseConfig):
    input_shape: list[int]
    filters: list[int]
    sizes: list[int]
    pool: list[int]
    activation: str
    paddings: list[int]

    def to_str(self):
        outstr = [
            list_to_str(self.filters),
            list_to_str(self.sizes),
            list_to_str(self.pool)
        ]

        return '_'.join(outstr)


@dataclass
class NetworkConfig(BaseConfig):
    units: list[int]
    activation_fn: str
    dropout_prob: float
    use_batch_norm: bool = True
    conv_config: Optional[ConvolutionConfig] = None

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
    xshape: list[int]
    nleapfrog: int
    eps: float = 0.01
    use_ncp: bool = True
    verbose: bool = False
    eps_fixed: bool = False
    use_split_xnets: bool = True
    use_separate_networks: bool = True
    merge_directions: bool = False

    def __post_init__(self):
        self.xdim = int(np.cumprod(self.xshape[1:])[-1])


@dataclass
class LossConfig(BaseConfig):
    use_mixed_loss: bool = False
    charge_weight: float = 0.01
    plaq_weight: float = 0.


@dataclass
class Steps:
    nera: int
    nepoch: int
    test: int
    log: int = 10
    print: int = 0

    def __post_init__(self):
        self.total = self.nera * self.nepoch
        if self.print == 0:
            self.print = self.nepoch // 4


@dataclass
class InputShapes(BaseConfig):
    x: list[int]
    v: list[int]


@dataclass
class InputSpec(BaseConfig):
    xshape: list[int]
    xnet: Optional[InputShapes] = None
    vnet: Optional[InputShapes] = None

    def __post_init__(self):
        if len(self.xshape) == 2:
            self.xdim = self.xshape[-1]
        elif len(self.xshape) > 2:
            self.xdim = np.cumprod(self.xshape[1:])[-1]
        else:
            raise ValueError(f'Invalid `xshape`: {self.xshape}')

        if self.xnet is None:
            self.xnet = InputShapes(x=self.xshape, v=self.xshape)
        if self.vnet is None:
            self.vnet = InputShapes(x=self.xshape, v=self.xshape)


@dataclass
class Config:
    steps: Steps
    network: NetworkConfig
    dynamics: DynamicsConfig
    loss: LossConfig
    net_weights: NetWeights
    conv: Optional[ConvolutionConfig] = None

    def __post_init__(self):
        self.xshape = Shape(self.dynamics.xshape)
        xdim = self.dynamics.xdim
        self.input_spec = InputSpec(
            xshape=self.dynamics.xshape,
            xnet=InputShapes(x=[xdim, 2], v=[xdim, ]),
            vnet=InputShapes(x=[xdim, ], v=[xdim, ]),
        )


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="dynamics",
        name="dynamics",
        node=DynamicsConfig,
    )
