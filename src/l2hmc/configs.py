"""
config.py

Implements various configuration objects
"""
from __future__ import absolute_import, annotations, division, print_function
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import asdict, dataclass, field
import json
import numpy as np
import os
from pathlib import Path
from typing import NamedTuple, Optional


SRC = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = SRC.parent
LOGS_DIR = PROJECT_DIR.joinpath('logs')


Shape = tuple[int]

State = namedtuple('State', ['x', 'v', 'beta'])

MonteCarloStates = namedtuple('MonteCarloStates', ['init', 'proposed', 'out'])


@dataclass
class BaseConfig:
    def to_json(self):
        return json.dumps(self.__dict__)

    def to_file(self, fpath: os.PathLike):
        with open(fpath, 'w') as f:
            json.dump(self.to_json(), f, indent=4)


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


@dataclass
class NetWeights:
    """Object for selectively scaling different components of x, v networks."""
    x: Optional[NetWeight] = field(default_factory=NetWeight)
    v: Optional[NetWeight] = field(default_factory=NetWeight)



@dataclass
class LearningRateConfig(BaseConfig):
    """Learning rate configuration object."""
    lr_init: float
    warmup_steps: int = 0
    decay_steps: int = -1
    decay_rate: float = 1.0


@dataclass
class NetworkConfig(BaseConfig):
    units: list[int]
    activation_fn: str
    dropout_prob: float
    use_batch_norm: bool = True


@dataclass
class ConvolutionConfig(BaseConfig):
    input_shape: list[int]
    filters: list[int]
    sizes: list[int]
    pool: list[int]
    activation: str
    paddings: list[int]
    use_batch_norm: bool = False


@dataclass
class DynamicsConfig(BaseConfig):
    xdim: int
    nleapfrog: int
    # hmc: bool = False
    eps: float = 0.01
    eps_fixed: bool = False
    use_ncp: bool = True
    use_split_xnets: bool = True
    use_separate_networks: bool = True


@dataclass
class LossConfig(BaseConfig):
    use_mixed_loss: bool = False
    charge_weight: float = 0.01
    plaq_weight: float = 0.


@dataclass
class Steps:
    train: int
    test: int
    log: int = 1
    save: int = 0

    def __post_init__(self):
        if self.save == 0:
            self.save == int(self.train // 4)


@dataclass
class InputShapes(BaseConfig):
    x: Shape
    v: Shape


@dataclass
class InputSpec(BaseConfig):
    xshape: Shape
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


class BaseNetworkFactory(ABC):
    def __init__(
            self,
            input_spec: InputSpec,
            network_config: NetworkConfig,
            net_weights: Optional[NetWeights] = None,
    ):
        if net_weights is None:
            net_weights = NetWeights(x=NetWeight(1., 1., 1.),  # (s, t, q)
                                     v=NetWeight(1., 1., 1.))

        self.nw = net_weights
        self.input_spec = input_spec
        self.network_config = network_config
        self.config = {
            'net_weights': asdict(self.nw),
            'input_spec': asdict(self.input_spec),
            'network_config': asdict(self.network_config),
        }

    def get_build_configs(self):
        return {
            'xnet': {
                'net_weight': self.nw.x,
                'xshape': self.input_spec.xshape,
                'input_shapes': self.input_spec.xnet,
                'network_config': self.network_config,
            },
            'vnet': {
                'net_weight': self.nw.v,
                'xshape': self.input_spec.xshape,
                'input_shapes': self.input_spec.vnet,
                'network_config': self.network_config,
            }
        }

    @abstractmethod
    def build_networks(self, nleapfrog: int = 0):
        """Build Networks."""
        pass
