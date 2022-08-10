"""
pytorch/experiment.py

Implements ptExperiment, a pytorch-specific subclass of the
Experiment base class.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
from typing import Optional, Any

import horovod.torch as hvd
from omegaconf import DictConfig
import torch
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from l2hmc.configs import NetWeights

from l2hmc.dynamics.pytorch.dynamics import Dynamics
from l2hmc.experiment.experiment import BaseExperiment
from l2hmc.lattice.su3.pytorch.lattice import LatticeSU3
from l2hmc.lattice.u1.pytorch.lattice import LatticeU1
# from l2hmc.trainers.pytorch.trainer import Trainer
from l2hmc.trainers.pytorch.trainer import Trainer
from l2hmc.utils.rich import get_console

log = logging.getLogger(__name__)

# LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')

Tensor = torch.Tensor
SIZE = hvd.size()
RANK = hvd.rank()
LOCAL_RANK = hvd.local_rank()


class Experiment(BaseExperiment):
    def __init__(
            self,
            cfg: DictConfig,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
    ) -> None:
        super().__init__(cfg=cfg)
        self.trainer = self.build_trainer(keep=keep, skip=skip)

        self._rank = hvd.rank()
        self._local_rank = hvd.local_rank()
        run = None
        arun = None
        if self._rank == 0 and self.config.init_wandb:
            import wandb
            log.warning(
                f'Initialize WandB from {self._rank}:{self._local_rank}'
            )
            run = super()._init_wandb()
            run.watch(
                # self.trainer.dynamics,
                self.trainer.dynamics.networks,
                log='all',
                log_graph=True,
                criterion=self.trainer.loss_fn,
            )
            assert run is wandb.run
            run.config['SIZE'] = SIZE

        if self._rank == 0 and self.config.init_aim:
            log.warning(
                f'Initializing Aim from {self._rank}:{self._local_rank}'
            )
            arun = self.init_aim()
            arun['SIZE'] = SIZE
            if arun is not None:
                if torch.cuda.is_available():
                    arun['ngpus'] = SIZE
                else:
                    arun['ncpus'] = SIZE

        self.run = run
        self.arun = arun
        self._is_built = True
        assert callable(self.trainer.loss_fn)
        assert isinstance(self.trainer, Trainer)
        assert isinstance(self.trainer.dynamics, Dynamics)
        assert isinstance(self.trainer.lattice, (LatticeU1, LatticeSU3))
        # if not isinstance(self.cfg, ExperimentConfig):
        #     self.cfg = hydra.utils.instantiate(cfg)
        #     assert isinstance(self.config, ExperimentConfig)

    def set_net_weights(self, net_weights: NetWeights):
        from l2hmc.network.pytorch.network import Network
        for step in range(self.config.dynamics.nleapfrog):
            xnet0 = self.trainer.dynamics._get_xnet(step, first=True)
            xnet1 = self.trainer.dynamics._get_xnet(step, first=False)
            vnet = self.trainer.dynamics._get_vnet(step)
            assert isinstance(xnet0, Network)
            assert isinstance(xnet1, Network)
            assert isinstance(vnet, Network)
            xnet0.set_net_weight(net_weights.x)
            xnet1.set_net_weight(net_weights.x)
            vnet.set_net_weight(net_weights.v)

    def visualize_model(self, x: Optional[Tensor] = None):
        from torchviz import make_dot  # type: ignore
        if x is None:
            state = self.trainer.dynamics.random_state(1.)
            x = state.x
            v = state.v
        else:
            v = torch.rand_like(x)

        assert isinstance(x, Tensor)
        assert isinstance(v, Tensor)
        sx0, tx0, qx0 = 0.0, 0.0, 0.0
        sv, tv, qv = 0.0, 0.0, 0.0
        # for step in range(self.config.dynamics.nleapfrog):
        sx0, tx0, qx0 = self.trainer.dynamics._call_xnet(
            0, inputs=(x, v), first=True
        )
        sv, tv, qv = self.trainer.dynamics._call_vnet(
            0, inputs=(x, v),
        )
        xparams = dict(
            self.trainer.dynamics.xnet.named_parameters()
        )
        vparams = dict(
            self.trainer.dynamics.vnet.named_parameters()
        )
        outdir = Path(self._outdir).joinpath('network_diagrams')
        outdir.mkdir(exist_ok=True, parents=True)
        make_dot(sx0, params=xparams).render(
            outdir.joinpath('scale-xnet-0').as_posix(), format='png'
        )
        make_dot(tx0, params=xparams).render(
            outdir.joinpath('transl-xnet-0').as_posix(), format='png'
        )
        make_dot(qx0, params=xparams).render(
            outdir.joinpath('transf-xnet-0').as_posix(), format='png'
        )
        make_dot(sv, params=vparams).render(
            outdir.joinpath('scale-vnet-0').as_posix(), format='png'
        )
        make_dot(tv, params=vparams).render(
            outdir.joinpath('transl-vnet-0').as_posix(), format='png'
        )
        make_dot(qv, params=vparams).render(
            outdir.joinpath('transf-vnet-0').as_posix(), format='png'
        )

    def update_wandb_config(
            self,
            run_id: Optional[str] = None,
    ) -> None:
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        # size = 'DDP' if torch.cuda.device_count() > 1 else 'local'
        self._update_wandb_config(device=device, run_id=run_id)

    def build_accelerator(self, **kwargs):
        assert self.config.framework == 'pytorch'
        from accelerate import Accelerator
        # return Accelerator(**asdict(self.config.accelerator))
        # return Accelerator(log_with=['all'])
        return Accelerator(**kwargs)

    def build_trainer(
            self,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
    ) -> Trainer:
        return Trainer(self.cfg, skip=skip, keep=keep)

    def init_wandb(
            self,
    ):
        return super()._init_wandb()

    def get_summary_writer(self):
        # sdir = super()._get_summary_dir(job_type=job_type)
        # sdir = os.getcwd()
        return SummaryWriter(self._outdir)

    def build(
            self,
            init_wandb: bool = True,
            init_aim: bool = True,
    ):
        return self._build(
            init_wandb=init_wandb,
            init_aim=init_aim,
        )

    def _build(
            self,
            init_wandb: bool = True,
            init_aim: bool = True,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
    ):
        if self._is_built:
            # assert self.accelerator is not None
            # assert self.lattice is not None
            assert self.trainer is not None
            # assert self.dynamics is not None
            # assert self.optimizer is not None
            # assert self.loss_fn is not None
            return {
                # 'lattice': self.lattice,
                # 'loss_fn': self.loss_fn,
                # 'dynamics': self.dynamics,
                # 'optimizer': self.optimizer,
                'trainer': self.trainer,
                'run': getattr(self, 'run', None),
                'arun': getattr(self, 'arun', None),
            }

        rank = hvd.rank()
        local_rank = hvd.local_rank()
        self.trainer = self.build_trainer(
            keep=keep,
            skip=skip,
        )
        run = None
        arun = None
        if RANK == 0:
            if init_wandb:
                import wandb
                log.warning(f'Initialize WandB from {rank}:{local_rank}')
                run = self.init_wandb()
                assert run is wandb.run
                # run.watch(
                #     self.trainer.dynamics,
                #     log="all"
                # )
                run.config['SIZE'] = SIZE
            if init_aim:
                log.warning(f'Initializing Aim from {rank}:{local_rank}')
                arun = self.init_aim()
                if arun is not None:
                    if torch.cuda.is_available():
                        arun['ngpus'] = SIZE
                    else:
                        arun['ncpus'] = SIZE

        self.run = run
        self.arun = arun
        self._is_built = True
        assert callable(self.trainer.loss_fn)
        assert isinstance(self.trainer, Trainer)
        assert isinstance(self.trainer.dynamics, Dynamics)
        assert isinstance(self.trainer.lattice, (LatticeU1, LatticeSU3))
        return {
            'lattice': self.trainer.lattice,
            'loss_fn': self.trainer.loss_fn,
            'dynamics': self.trainer.dynamics,
            'optimizer': self.trainer.optimizer,
            'trainer': self.trainer,
            'run': self.run,
            'arun': self.arun,
        }

    def _assert_is_built(self):
        # assert self.accelerator is not None
        assert self.trainer is not None
        assert isinstance(self.trainer, Trainer)
        assert self._is_built
        # assert self.lattice is not None
        # assert self.trainer is not None
        # assert self.dynamics is not None
        # assert self.optimizer is not None
        # assert self.loss_fn is not None

    def train(
            self,
            nchains: Optional[int] = None,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            writer: Optional[Any] = None,
            nera: Optional[int] = None,
            nepoch: Optional[int] = None,
            beta: Optional[float | list[float] | dict[str, float]] = None,
    ):
        # nchains = 16 if nchains is None else nchains
        jobdir = self.get_jobdir(job_type='train')
        writer = None
        if RANK == 0:
            writer = self.get_summary_writer()

        # logfile = jobdir.joinpath(f'train-{RANK}.log')
        # with open(logfile.as_posix(), 'wt') as logfile:
        # console = Console(log_path=False, file=logfile)
        console = get_console(record=True)
        # console = Console(log_path=False, record=True, width=210)
        self.trainer.set_console(console)
        output = self.trainer.train(
            x=x,
            nera=nera,
            nepoch=nepoch,
            run=self.run,
            arun=self.arun,
            writer=writer,
            train_dir=jobdir,
            skip=skip,
            beta=beta,
        )
        # fname = f'train-{RANK}'
        # txtfile = jobdir.joinpath(f'{fname}.txt')
        # htmlfile = jobdir.joinpath(f'{fname}.html')
        # console.save_text(txtfile.as_posix(), clear=False)
        # console.save_html(htmlfile.as_posix())

        if self.trainer._is_chief:
            dset = self.save_dataset(
                output=output,
                nchains=nchains,
                job_type='train',
                outdir=jobdir
            )
            output['dataset'] = dset

        if writer is not None:
            writer.close()

        return output

    def evaluate(
            self,
            job_type: str,
            therm_frac: float = 0.1,
            nchains: Optional[int] = None,
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
            eval_steps: Optional[int] = None,
    ):
        """Evaluate model."""
        # if RANK != 0:
        if not self.trainer._is_chief:
            return

        assert job_type in ['eval', 'hmc']
        jobdir = self.get_jobdir(job_type)
        writer = self.get_summary_writer()
        console = get_console(record=True)
        self.trainer.set_console(console)
        output = self.trainer.eval(
            run=self.run,
            arun=self.arun,
            writer=writer,
            nchains=nchains,
            job_type=job_type,
            eps=eps,
            nleapfrog=nleapfrog,
            eval_steps=eval_steps,
        )

        output['dataset'] = self.save_dataset(
            output=output,
            job_type=job_type,
            outdir=jobdir,
            therm_frac=therm_frac,
        )
        if writer is not None:
            writer.close()

        return output
