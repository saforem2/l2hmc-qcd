"""
configs.py

Implements various configuration objects
"""
from __future__ import absolute_import, annotations, division, print_function
import json
import rich.repr
# import logging
import os

from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Counter, Dict, List, Optional, Sequence

from hydra.core.config_store import ConfigStore
import numpy as np
from omegaconf import DictConfig
import l2hmc.utils.dist as udist

# logger = logging.getLogger(__name__)
from l2hmc import get_logger

logger = get_logger(__name__)


# -- Configure useful Paths -----------------------
HERE = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = HERE.parent.parent
CONF_DIR = HERE.joinpath('conf')
LOGS_DIR = PROJECT_DIR.joinpath('logs')
AIM_DIR = HERE.joinpath('.aim')
OUTPUTS_DIR = HERE.joinpath('outputs')
CHECKPOINTS_DIR = HERE.joinpath('checkpoints')

CONF_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True, parents=True)
OUTDIRS_FILE = OUTPUTS_DIR.joinpath('outdirs.log')


# -- namedtuple objects -------------------------------------------------------
State = namedtuple('State', ['x', 'v', 'beta'])
MonteCarloStates = namedtuple('MonteCarloStates', ['init', 'proposed', 'out'])

FP16_SYNONYMS = ['float16', 'fp16', '16', 'half']
FP32_SYNONYMS = ['float32', 'fp32', '32', 'single']
FP64_SYNONYMS = ['float64', 'fp64', '64', 'double']

ENV_FILTERS = [
    'PS1',
    'LSCOLORS',
    'LS_COLORS',
]

SYNONYMS = {
    'pytorch': [
        'p'
        'pt',
        'torch',
        'pytorch',
    ],
    'tensorflow': [
        't'
        'tf',
        'tflow',
        'tensorflow',
    ],
    'horovod': [
        'h',
        'hv',
        'hvd',
        'horovod',
    ],
    'DDP': [
        'ddp',
    ],
    'deepspeed': [
        'ds',
        'deepspeed',
    ]
}


def dict_to_list_of_overrides(d: dict):
    return [f'{k}={v}' for k, v in flatten_dict(d, sep='.').items()]


def flatten_dict(d: dict, sep: str = '/', pre='') -> dict:
    return {
        pre + sep + k if pre else k: v
        for kk, vv in d.items()
        for k, v in flatten_dict(vv, sep, kk).items()
    } if isinstance(d, dict) else {pre: d}


def add_to_outdirs_file(outdir: os.PathLike):
    with open(OUTDIRS_FILE, 'a') as f:
        f.write(Path(outdir).resolve.as_posix() + '\n')


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
@rich.repr.auto
class BaseConfig(ABC):

    @abstractmethod
    def to_str(self) -> str:
        pass

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    def get_config(self) -> dict:
        return asdict(self)

    def asdict(self) -> dict:
        return asdict(self)

    def to_dict(self) -> dict:
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
            'p4x4': self.p4x4
        }


@dataclass
class EnvConfig:
    # machine: str
    # rank: int
    # local_rank: int
    # world_size: int
    # nhosts: int
    # hostname: str
    # addr: str

    def __post_init__(self):
        import socket
        self.hostname = socket.gethostname()
        dist_env = udist.query_environment()
        self.rank = dist_env['rank']
        self.local_rank = dist_env['local_rank']
        self.world_size = dist_env['world_size']
        self.addr = socket.gethostbyaddr(self.hostname)[0]
        if self.addr.startswith('x3'):
            self.machine = 'Polaris'
            self.nodefile = os.environ.get('PBS_NODEFILE', None)
        elif self.addr.startswith('x1'):
            self.machine = 'Sunspot'
            self.nodefile = os.environ.get('PBS_NODEFILE', None)
        elif self.addr.startswith('thetagpu'):
            self.machine = 'ThetaGPU'
            self.nodefile = os.environ.get('COBALT_NODEFILE', None)
        else:
            self.machine = self.addr
            self.nodefile = None
        self.env = {
            k: v for k, v in dict(os.environ).items()
            if (
                k not in ENV_FILTERS
                and not k.startswith('_ModuleTable')
                and not k.startswith('BASH_FUNC_')
            )
        }


@dataclass
class wandbSetup(BaseConfig):
    id: Optional[str] = None
    group: Optional[str] = None
    save_code: Optional[bool] = True
    sync_tensorboard: Optional[bool] = True
    tags: Optional[Sequence[str]] = None
    mode: Optional[str] = 'online'
    resume: Optional[str] = 'allow'
    entity: Optional[str] = 'l2hmc-qcd'
    project: Optional[str] = 'l2hmc-qcd'
    settings: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        if self.settings is None:
            self.settings = {'start_method': 'thread'}

    def to_str(self) -> str:
        return ''


@dataclass
class wandbConfig(BaseConfig):
    setup: wandbSetup

    def to_str(self) -> str:
        return self.to_json()


@dataclass
class NetWeight(BaseConfig):
    """Object for selectively scaling different components of learned fns.

    Explicitly,
     - s: scales the v (x) scaling function in the v (x) updates
     - t: scales the translation function in the update
     - q: scales the force (v) transformation function in the v (x) updates
    """
    s: float = field(default=1.)
    t: float = field(default=1.)
    q: float = field(default=1.)

    def to_dict(self):
        return {'s': self.s, 't': self.t, 'q': self.q}

    def to_str(self):
        return f's{self.s:2.1f}t{self.t:2.1f}q{self.t:2.1f}'


@dataclass
class NetWeights(BaseConfig):
    """Object for selectively scaling different components of x, v networks."""
    x: NetWeight = NetWeight(1., 1., 1.)
    v: NetWeight = NetWeight(1., 1., 1.)

    def to_str(self):
        return f'nwx-{self.x.to_str()}-nwv-{self.v.to_str()}'

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
class Steps(BaseConfig):
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
        freq = int(self.nepoch // 20)
        self.log = (
            max(1, freq) if self.log is None else self.log
        )
        self.print = (
            max(1, freq) if self.print is None else self.print
        )

        assert isinstance(self.log, int)
        assert isinstance(self.print, int)

    def to_str(self) -> str:
        return f'nera-{self.nera}_nepoch-{self.nepoch}'

    def update(
            self,
            nera: Optional[int] = None,
            nepoch: Optional[int] = None,
            test: Optional[int] = None,
            log: Optional[int] = None,
            print: Optional[int] = None,
            extend_last_era: Optional[int] = None,
    ) -> Steps:
        return Steps(
            nera=(self.nera if nera is None else nera),
            nepoch=(self.nepoch if nepoch is None else nepoch),
            test=(self.test if test is None else test),
            log=(self.log if log is None else log),
            print=(self.print if print is None else print),
            extend_last_era=(
                self.extend_last_era if extend_last_era is None
                else extend_last_era
            )
        )


@dataclass
class ConvolutionConfig(BaseConfig):
    filters: Optional[Sequence[int]] = None
    sizes: Optional[Sequence[int]] = None
    pool: Optional[Sequence[int]] = None
    # activation: str
    # paddings: list[int]

    def __post_init__(self):
        if self.filters is None:
            return

        if self.sizes is None:
            logger.warning('Using default filter size of 2')
            self.sizes = list(len(self.filters) * [2])
        if self.pool is None:
            logger.warning('Using default pooling size of 2')
            self.pool = len(self.filters) * [2]

        assert len(self.filters) == len(self.sizes)
        assert len(self.filters) == len(self.pool)
        assert self.pool is not None

    def to_str(self) -> str:
        if self.filters is None:
            return 'conv-None'

        if len(self.filters) > 0:
            outstr = [
                list_to_str(list(self.filters)),
            ]
            if self.sizes is not None:
                outstr.append(
                    list_to_str(list(self.sizes))
                )
            if self.pool is not None:
                outstr.append(
                    list_to_str(list(self.pool))
                )

            return '-'.join(['conv', '_'.join(outstr)])
        return ''


@dataclass
class NetworkConfig(BaseConfig):
    units: Sequence[int]
    activation_fn: str
    dropout_prob: float
    use_batch_norm: bool = True

    def to_str(self):
        ustr = '-'.join([str(int(i)) for i in self.units])
        dstr = f'dp-{self.dropout_prob:2.1f}'
        bstr = f'bn-{self.use_batch_norm}'
        return '-'.join(['net', '_'.join([ustr, dstr, bstr])])
        # outstr = [f'nh-{ustr}_act-{self.activation_fn}']
        # if self.dropout_prob > 0:
        #     outstr.append(f'dp-{self.dropout_prob:2.1f}')
        # if self.use_batch_norm:
        #     outstr.append('bNorm')

        # return '_'.join(outstr)


@dataclass
class DynamicsConfig(BaseConfig):
    nchains: int
    group: str
    latvolume: List[int]
    nleapfrog: int
    eps: float = 0.01
    eps_hmc: float = 0.01
    use_ncp: bool = True
    verbose: bool = True
    eps_fixed: bool = False
    use_split_xnets: bool = True
    use_separate_networks: bool = True
    merge_directions: bool = True

    def to_str(self) -> str:
        latstr = '-'.join([str(i) for i in self.xshape[1:]])
        lfstr = f'nlf-{self.nleapfrog}'
        splitstr = f'xsplit-{self.use_split_xnets}'
        sepstr = f'sepnets-{self.use_separate_networks}'
        mrgstr = f'merge-{self.merge_directions}'
        return '/'.join([self.group, latstr, lfstr, splitstr, sepstr, mrgstr])

    def __post_init__(self):
        assert self.group.upper() in ['U1', 'SU3']
        # NOTE ---------------------------------------------
        # --------------------------------------------------
        if self.eps_hmc is None:
            # if not specified, use a trajectory length of 1
            self.eps_hmc = 1.0 / self.nleapfrog
        if self.group.upper() == 'U1':
            self.dim = 2
            self.nt, self.nx = self.latvolume
            self.xshape = (self.nchains, self.dim, *self.latvolume)
            self.vshape = (self.nchains, self.dim, *self.latvolume)
            assert len(self.xshape) == 4
            assert len(self.latvolume) == 2
            self.xdim = int(np.cumprod(self.xshape[1:])[-1])
        elif self.group.upper() == 'SU3':
            self.dim = 4
            self.link_shape = (3, 3)
            self.vec_shape = 8
            self.nt, self.nx, self.ny, self.nz = self.latvolume
            # xshape : [Nb, 4, Nt, Nx, Ny, Nz, 3, 3]
            self.xshape = (
                self.nchains,
                self.dim,
                *self.latvolume,
                *self.link_shape
            )
            # vshape : [Nb, 4, Nt, Nx, Ny, Nz, 8]
            self.vshape = (
                self.nchains,
                self.dim,
                *self.latvolume,
                self.vec_shape
            )
            assert len(self.xshape) == 8
            assert len(self.vshape) == 7
            assert len(self.latvolume) == 4
            self.xdim = int(np.cumprod(self.xshape[1:])[-1])
        else:
            raise ValueError('Expected `group` to be one of `"U1", "SU3"`')


@dataclass
class LossConfig(BaseConfig):
    use_mixed_loss: bool = False
    charge_weight: float = 0.01
    rmse_weight: float = 0.0
    plaq_weight: float = 0.0
    aux_weight: float = 0.0

    def to_str(self) -> str:
        return '_'.join([
            f'qw-{self.charge_weight:2.1f}',
            f'pw-{self.plaq_weight:2.1f}',
            f'rw-{self.rmse_weight:2.1f}',
            f'aw-{self.aux_weight:2.1f}',
            f'mixed-{self.use_mixed_loss}',
        ])


@dataclass
class InputSpec(BaseConfig):
    xshape: Sequence[int]
    xnet: Optional[Dict[str, int | Sequence[int]]] = None
    vnet: Optional[Dict[str, int | Sequence[int]]] = None

    def to_str(self):
        return '-'.join([str(i) for i in self.xshape])

    def __post_init__(self):
        if len(self.xshape) == 2:
            self.xdim = self.xshape[-1]
            self.vshape = self.xshape
            self.vdim = self.xshape[-1]
        elif len(self.xshape) > 2:
            # xshape: [Nb, 4, Nt, Nx, Ny, Nz, 3, 3]
            self.xdim: int = np.cumprod(self.xshape[1:])[-1]
            # lat_shape: [Nb, 4, Nt, Nx, Ny, Nz]
            lat_shape = self.xshape[:-2]
            # vdim: 8 = 3 ** 2 - 1
            vd = (self.xshape[-1] ** 2) - 1
            # vshape = [Nb, 4, Nt, Nx, Ny, Nz, 8]
            self.vshape: Sequence[int] = (*lat_shape, vd)
            self.vdim: int = np.cumprod(self.vshape[1:])[-1]
        else:
            raise ValueError(f'Invalid `xshape`: {self.xshape}')

        if self.xnet is None:
            self.xnet = {'x': self.xshape, 'v': self.xshape}
        if self.vnet is None:
            self.vnet = {'x': self.xshape, 'v': self.xshape}


# @dataclass
# class DeepSpeedConfig(BaseConfig):

@dataclass
class FlopsProfiler:
    enabled: bool = False
    profile_step: int = 1
    module_depth: int = -1
    top_modules: int = 1
    detailed: bool = True
    output_file: Optional[os.PathLike | str | Path] = None

    def __post_init__(self):
        pass
        # if self.output_file is None:
        #     self.output_file = Path(os.getcwd()).joinpath(
        #         'ds-flops-profiler.log'
        #     ).resolve().as_posix()


# @dataclass
# class dsOptimizer:
#     type: str = "AdamW"
#     params: dict


# @dataclass
# class DeepSpeedConfig(BaseConfig):
#     fpath: Optional[os.PathLike] = None
#     wall_clock_breakdown: Optional[bool] = None
#     prescale_gradients: Optional[bool] = None
#     flops_profiler:


@dataclass
class OptimizerConfig:
    type: str
    params: Optional[dict] = field(default_factory=dict)


@dataclass
class fp16Config:
    enabled: bool
    auto_cast: bool = True
    fp16_master_weights_and_grads: bool = False
    min_loss_scale: float = 0.


@dataclass
class CommsLogger:
    enabled: bool
    verbose: bool = True
    prof_all: bool = True
    debug: bool = False


@dataclass
class AutoTuning:
    enabled: bool
    arg_mappings: Optional[dict] = field(default_factory=dict)


@dataclass
class ZeroOptimization:
    stage: int


@dataclass
class ExperimentConfig(BaseConfig):
    wandb: Any
    steps: Steps
    framework: str
    loss: LossConfig
    network: NetworkConfig
    conv: ConvolutionConfig
    net_weights: NetWeights
    dynamics: DynamicsConfig
    learning_rate: LearningRateConfig
    annealing_schedule: AnnealingSchedule
    # ----- Optional (w/ defaults) ------------
    # conv: Optional[ConvolutionConfig] = None
    restore: bool = True
    save: bool = True
    c1: float = 0.0
    port: str = '2345'
    compile: bool = True
    profile: bool = False
    init_aim: bool = True
    init_wandb: bool = True
    debug_mode: bool = False
    default_mode: bool = True
    print_config: bool = True
    precision: str = 'float32'
    ignore_warnings: bool = True
    backend: str = 'hvd'
    # ds_config: dict = field(default_factory=dict)
    # ----- Optional (w/o defaults) -----------
    seed: Optional[int] = None
    ds_config_path: Optional[Any] = None
    name: Optional[str] = None
    name: Optional[str] = None
    width: Optional[int] = None
    nchains: Optional[int] = None
    compression: Optional[str] = None

    def __post_init__(self):
        if self.seed is None:
            import numpy as np
            self.seed = np.random.randint(0)
            logger.warning(
                f'No seed specified, using random seed: {self.seed}'
            )
        self.env = EnvConfig()
        self.ds_config = {}
        self.xdim = self.dynamics.xdim
        self.xshape = self.dynamics.xshape
        if self.ds_config_path is None:
            fpath = Path(CONF_DIR).joinpath('ds_config.yaml')
            self.ds_config_path = fpath.resolve().as_posix()

        if self.precision in FP16_SYNONYMS:
            self.precision = 'fp16'
        elif self.precision in FP32_SYNONYMS:
            self.precision = 'float32'
        elif self.precision in FP64_SYNONYMS:
            self.precision = 'float64'

        # self.ds_config = {}
        # if self.ds_config_path is not None:
        #     fpath = Path(self.ds_config_path)
        #     assert fpath.is_file()
        #     with fpath.open('r') as f:
        #         self.ds_config.update(
        #             json.load(f)
        #         )

        #     # assert Path(self.ds_config_path).is_file()
        #     # with open(Path())
        #     # self.ds_config.update({
        #     #     json.load(self.ds_config_path)
        #     # })

        w = int(os.environ.get('COLUMNS', 200))
        self.width = w if self.width is None else self.width
        if self.framework in SYNONYMS['tensorflow']:
            self.backend = 'hvd'
        elif self.framework in SYNONYMS['pytorch']:
            if self.backend is None:
                logger.warning('Backend not specified, using DDP')
                self.backend = 'DDP'

            assert self.backend.lower() in [
                'hvd', 'horovod', 'ddp', 'ds', 'deepspeed',
            ]
        else:
            raise ValueError(
                f'Unexpected value for framework: {self.framework}'
            )

        if self.debug_mode:
            self.compile = False

        self.annealing_schedule.setup(
            nera=self.steps.nera,
            nepoch=self.steps.nepoch,
        )

    def load_ds_config1(self, fpath: Optional[os.PathLike]) -> dict:
        fname = self.ds_config_path if fpath is None else fpath
        assert fname is not None
        cpath = Path(fname)
        ds_config = {}
        if cpath.is_file():
            pass

        return ds_config

    def load_ds_config(self, fpath: Optional[os.PathLike] = None) -> dict:
        fname = self.ds_config_path if fpath is None else fpath
        assert fname is not None
        ds_config_path = Path(fname)
        logger.info(
            f'Loading DeepSpeed Config from: {ds_config_path.as_posix()}'
        )
        if ds_config_path.suffix == '.json':
            with ds_config_path.open('r') as f:
                ds_config = json.load(f)
            return ds_config
        if ds_config_path.suffix == '.yaml':
            import yaml
            with ds_config_path.open('r') as stream:
                ds_config = dict(yaml.safe_load(stream))
            return ds_config
        raise TypeError('Unexpected FileType')

    def set_ds_config(self, ds_config: dict) -> None:
        self.ds_config = ds_config

    def to_str(self) -> str:
        dynstr = self.dynamics.to_str()
        constr = self.conv.to_str()
        netstr = self.network.to_str()
        return '/'.join([dynstr, constr, netstr, self.framework])

    def get_checkpoint_dir(self) -> Path:
        return Path(CHECKPOINTS_DIR).joinpath(self.to_str())

    def rank(self):
        if self.framework in SYNONYMS['pytorch']:
            if self.backend.lower() in SYNONYMS['horovod']:
                import horovod.torch as hvd
                if not hvd.is_initialized():
                    hvd.init()
                return hvd.rank()
            elif self.backend.lower() in SYNONYMS['DDP']:
                return int(os.environ.get('RANK', 0))
            elif self.backend.lower() in SYNONYMS['deepspeed']:
                import torch.distributed as dist
                return dist.get_rank()
        elif self.framework in SYNONYMS['tensorflow']:
            import horovod.tensorflow as hvd
            if not hvd.is_initialized():
                hvd.init()
            return hvd.rank()


@dataclass
class AnnealingSchedule(BaseConfig):
    beta_init: float
    beta_final: Optional[float] = 1.0
    dynamic: bool = False
    # steps: Steps
    # TODO: Add methods for specifying different annealing schedules

    def to_str(self) -> str:
        return f'bi-{self.beta_init}_bf-{self.beta_final}'

    def __post_init__(self):
        if self.beta_final is None or self.beta_final < self.beta_init:
            logger.warning(
                f'AnnealingSchedule.beta_final must be >= {self.beta_init},'
                f' but received: {self.beta_final}.\n'
                f'Setting self.beta_final to {self.beta_init}'
            )
            self.beta_final = float(self.beta_init)
        assert (
            isinstance(self.beta_final, float)
            and self.beta_final >= self.beta_init
        )

    def update(
            self,
            beta_init: Optional[float] = None,
            beta_final: Optional[float] = None,
    ):
        logger.warning('Updating annealing schedule!')
        if beta_init is not None:
            logger.warning(f'annealing_schedule.beta_init = {beta_init:.3f}')
            self.beta_init = beta_init
        if beta_final is not None:
            logger.warning(f'annealing_schedule.beta_final = {beta_final:.3f}')
            self.beta_final = beta_final

    def setup(
            self,
            nera: Optional[int] = None,
            nepoch: Optional[int] = None,
            steps: Optional[Steps] = None,
            beta_init: Optional[float] = None,
            beta_final: Optional[float] = None,
    ) -> dict:
        if nera is None:
            assert steps is not None
            nera = steps.nera

        if nepoch is None:
            assert steps is not None
            nepoch = steps.nepoch

        if beta_init is None:
            beta_init = self.beta_init

        if beta_final is None:
            beta_final = (
                self.beta_final
                if self.beta_final is not None
                else self.beta_init
            )

        self.betas = np.linspace(beta_init, beta_final, nera)
        total = steps.total if steps is not None else 1
        self._dbeta = (beta_final - beta_init) / total
        self.beta_dict = {
            str(era): self.betas[era] for era in range(nera)
        }

        return self.beta_dict


@dataclass
class Annealear:
    """Dynamically adjust annealing schedule during training."""
    schedule: AnnealingSchedule
    patience: int
    min_delta: Optional[float] = None

    def __post_init__(self):
        self.wait = 0
        self.best = np.Inf
        self._current_era = 0
        self._current_beta = self.schedule.beta_init
        self._epoch = 0
        self._count = 0
        self.betas = []
        self.loss = []
        self.losses = {}
        self._reset()

    def _reset(self):
        self.wait = 0

    def update(self, loss: float):
        self._epoch += 1
        self.loss.append(loss)

    @staticmethod
    def avg_diff(
            y: list[float],
            x: Optional[list[float]] = None,
            *,
            drop: Optional[int | float] = None,
    ) -> float:
        """Returns (1/n) ∑ [δy/δx]."""
        if x is not None:
            assert len(x) == len(y)

        if drop is not None:
            if isinstance(drop, int):
                # If passed as an int, we should interpret as num to drop
                if drop > 1:
                    y = y[drop:]
                    if x is not None:
                        x = x[drop:]
                else:
                    raise ValueError('Expected `drop` to be an int > 1')
            elif isinstance(drop, float):
                # If passed as a float, we should interpret as a percentage
                if drop < 1.:
                    frac = drop * len(y)
                    y = y[frac:]
                    if x is not None:
                        x = x[frac:]
                else:
                    raise ValueError('Expected `drop` to be a float < 1.')
            else:
                raise ValueError(
                    'Expected drop to be one of `int` or `float`.'
                )

        dyavg = np.subtract(y[1:], y[:-1]).mean()
        if x is not None:
            dxavg = np.subtract(x[1:], x[:-1]).mean()
            return dyavg / dxavg

        return dyavg

    def start_epoch(self, era: int, beta: float):
        self.losses[f'{era}'] = {
            'beta': beta,
            'loss': [],
        }
        self._prev_beta = self.betas[-1]
        self._current_era = era
        self._current_beta = beta

        self.betas.append(beta)

        self._prev_best = np.Inf
        if (era - 1) in self.losses.keys():
            self._prev_best = np.min(self.losses[str(era - 1)]['loss'])

    def end_epoch(
            self,
            losses: list[float],
            era: Optional[int] = None,
            beta: Optional[float] = None,
            drop: Optional[int | float] = None,
    ) -> float:
        current_era = self._current_era if era is None else era
        current_beta = self._current_beta if beta is None else beta
        prev_beta = self._prev_beta
        new_beta = current_beta + self.schedule._dbeta
        self.losses[f'{current_era}'] = {
            'beta': current_beta,
            'loss': losses,
        }
        new_best = np.min(losses)
        avg_slope = self.avg_diff(losses, drop=drop)
        if new_best < self._prev_best or avg_slope < 0:
            # Loss has improved from previous best, return new_beta (increase)
            return new_beta
        else:
            # Loss has NOT improved from previous best
            current_beta_count = Counter(self.betas).get(current_beta)
            if (
                    current_beta_count is not None
                    and isinstance(current_beta_count, int)
                    and current_beta_count > self.patience
            ):
                # If we've exhausted our patience
                # at the current_beta, return prev_beta (decrease)
                return prev_beta

            # If we're still being patient, return current_beta (no change)
            return current_beta


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


def get_experiment(
        overrides: Optional[list[str]] = None,
        build_networks: bool = True,
        keep: Optional[str | list[str]] = None,
        skip: Optional[str | list[str]] = None,
):
    cfg = get_config(overrides)
    if cfg.framework == 'pytorch':
        from l2hmc.experiment.pytorch.experiment import Experiment
        return Experiment(
            cfg,
            keep=keep,
            skip=skip,
            build_networks=build_networks,
        )
    elif cfg.framework == 'tensorflow':
        from l2hmc.experiment.tensorflow.experiment import Experiment
        return Experiment(
            cfg,
            keep=keep,
            skip=skip,
            build_networks=build_networks,
        )
    else:
        raise ValueError(
            f'Unexpected value for `cfg.framework: {cfg.framework}'
        )


defaults = [
    # {'backend': MISSING}
]

cs = ConfigStore.instance()
cs.store(
    name='experiment_config',
    node=ExperimentConfig,
)
