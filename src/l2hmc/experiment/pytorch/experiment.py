"""
pytorch/experiment.py

Implements ptExperiment, a pytorch-specific subclass of the
Experiment base class.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
from os import PathLike
from pathlib import Path
from typing import Any, Optional

import horovod.torch as hvd
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from l2hmc.configs import NetWeights
from l2hmc.configs import ExperimentConfig
from l2hmc.dynamics.pytorch.dynamics import Dynamics as ptDynamics
from l2hmc.dynamics.pytorch.dynamics import Dynamics
from l2hmc.experiment.experiment import BaseExperiment
from l2hmc.lattice.su3.pytorch.lattice import LatticeSU3
from l2hmc.lattice.u1.pytorch.lattice import LatticeU1
from l2hmc.trainers.pytorch.trainer import Trainer
from l2hmc.utils.dist import setup_torch_distributed
from l2hmc.utils.rich import get_console

# log = logging.getLogger(__name__)
# log = get_pylogger(__name__)
log = logging.getLogger(__name__)

# LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')

Tensor = torch.Tensor
# SIZE = hvd.size()
# RANK = hvd.rank()
# LOCAL_RANK = hvd.local_rank()


class Experiment(BaseExperiment):
    def __init__(
            self,
            cfg: DictConfig,
            build_networks: bool = True,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
    ) -> None:
        super().__init__(cfg=cfg)
        if not isinstance(self.config, ExperimentConfig):
            self.config = instantiate(cfg)
        assert isinstance(self.config, ExperimentConfig)
        # assert isinstance(
        #     self.config,
        #     (ExperimentConfig,
        #      configs.ExperimentConfig)
        # )
        self.ckpt_dir = self.config.get_checkpoint_dir()
        self.trainer: Trainer = self.build_trainer(
            keep=keep,
            skip=skip,
            build_networks=build_networks,
            ckpt_dir=self.ckpt_dir,
        )
        dsetup = setup_torch_distributed(self.config.backend)
        self._size = dsetup['size']
        self._rank = dsetup['rank']
        self._local_rank = dsetup['local_rank']

        # self._rank = hvd.rank()
        # self._local_rank = hvd.local_rank()
        run = None
        arun = None
        if self._rank == 0 and self.config.init_wandb:
            import wandb
            log.warning(
                f'Initialize WandB from {self._rank}:{self._local_rank}'
            )
            run = super()._init_wandb()
            assert run is wandb.run
            run.watch(
                # self.trainer.dynamics,
                self.trainer.dynamics.networks,
                log='all',
                log_graph=True,
                criterion=self.trainer.loss_fn,
            )
            run.config['SIZE'] = self._size

        if self._rank == 0 and self.config.init_aim:
            log.warning(
                f'Initializing Aim from {self._rank}:{self._local_rank}'
            )
            arun = self.init_aim()
            arun['SIZE'] = self._size
            if arun is not None:
                if torch.cuda.is_available():
                    arun['ngpus'] = self._size
                else:
                    arun['ncpus'] = self._size

        self.run = run
        self.arun = arun
        self._is_built = True
        assert callable(self.trainer.loss_fn)
        assert isinstance(self.trainer, Trainer)
        assert isinstance(self.trainer.dynamics, (ptDynamics, Dynamics))
        assert isinstance(self.trainer.lattice, (LatticeU1, LatticeSU3))
        # if not isinstance(self.cfg, ExperimentConfig):
        #     self.cfg = hydra.utils.instantiate(cfg)
        #     assert isinstance(self.config, ExperimentConfig)

    def set_net_weights(self, net_weights: NetWeights):
        from l2hmc.network.pytorch.network import LeapfrogLayer
        for step in range(self.config.dynamics.nleapfrog):
            xnet0 = self.trainer.dynamics._get_xnet(step, first=True)
            xnet1 = self.trainer.dynamics._get_xnet(step, first=False)
            vnet = self.trainer.dynamics._get_vnet(step)
            assert isinstance(xnet0, LeapfrogLayer)
            assert isinstance(xnet1, LeapfrogLayer)
            assert isinstance(vnet, LeapfrogLayer)
            xnet0.set_net_weight(net_weights.x)
            xnet1.set_net_weight(net_weights.x)
            vnet.set_net_weight(net_weights.v)

    def visualize_model(self, x: Optional[Tensor] = None):
        # import graphviz
        # from torchview import draw_graph
        from torchviz import make_dot  # type: ignore
        device = self.trainer.device
        state = self.trainer.dynamics.random_state(1.)
        m, _ = self.trainer.dynamics._get_mask(0)
        # x = state.x.reshape((state.x.shape[0], -1)).to(device)
        beta = state.beta.to(device)
        m = m.to(device)
        x = state.x.to(device)
        v = state.v.to(device)

        outdir = Path(self._outdir).joinpath('network_diagrams')
        outdir.mkdir(exist_ok=True, parents=True)
        # fpxnet = outdir.joinpath('xnet.png')
        # fpvnet = outdir.joinpath('vnet.png')
        vnet = self.trainer.dynamics._get_vnet(0)
        xnet = self.trainer.dynamics._get_xnet(0, first=True)

        with torch.autocast(  # type:ignore
            # dtype=torch.float32,
            device_type='cuda' if torch.cuda.is_available() else 'cpu'
        ):
            force = self.trainer.dynamics.grad_potential(x, state.beta)
            sv, tv, qv = self.trainer.dynamics._call_vnet(0, (x, force))
            xm = self.trainer.dynamics.unflatten(
                m * self.trainer.dynamics.flatten(x)
            )
            sx, tx, qx = self.trainer.dynamics._call_xnet(
                0,
                (xm, v),
                first=True
            )

        outputs = {
            'v': {
                'scale': sv,
                'transl': tv,
                'transf': qv,
            },
            'x': {
                'scale': sx,
                'transl': tx,
                'transf': qx,
            },
        }
        for key, val in outputs.items():
            for kk, vv in val.items():
                net = xnet if key == 'x' else vnet
                net = net.to(vv.dtype)
                _ = make_dot(
                    vv,
                    params=dict(net.named_parameters()),
                    show_attrs=True,
                    show_saved=True
                ).save(f'{key}-{kk}.gv')
                # ).render(
                #     f'{key}-{k}.png',
                #     # outdir.joinpath(f'{key}net{k}.gv').as_posix(),
                #     # format='png'
                # )
        # sx0, tx0, qx0 = 0.0, 0.0, 0.0
        # sv, tv, qv = 0.0, 0.0, 0.0
        # # for step in range(self.config.dynamics.nleapfrog):
        # sx0, tx0, qx0 = self.trainer.dynamics._call_xnet(
        #     0, inputs=(x, v), first=True
        # )
        # sv, tv, qv = self.trainer.dynamics._call_vnet(
        #     0, inputs=(x, v),
        # )
        # xparams = dict(
        #     self.trainer.dynamics.xnet.named_parameters()
        # )
        # vparams = dict(
        #     self.trainer.dynamics.vnet.named_parameters()
        # )
        # outdir = Path(self._outdir).joinpath('network_diagrams')
        # outdir.mkdir(exist_ok=True, parents=True)
        # make_dot(sx0, params=xparams).render(
        #     outdir.joinpath('scale-xnet-0').as_posix(), format='png'
        # )
        # make_dot(tx0, params=xparams).render(
        #     outdir.joinpath('transl-xnet-0').as_posix(), format='png'
        # )
        # make_dot(qx0, params=xparams).render(
        #     outdir.joinpath('transf-xnet-0').as_posix(), format='png'
        # )
        # make_dot(sv, params=vparams).render(
        #     outdir.joinpath('scale-vnet-0').as_posix(), format='png'
        # )
        # make_dot(tv, params=vparams).render(
        #     outdir.joinpath('transl-vnet-0').as_posix(), format='png'
        # )
        # make_dot(qv, params=vparams).render(
        #     outdir.joinpath('transf-vnet-0').as_posix(), format='png'
        # )

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
            build_networks: bool = True,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
            ckpt_dir: Optional[PathLike] = None,
    ) -> Trainer:
        ckpt_dir = self.ckpt_dir if ckpt_dir is None else ckpt_dir
        return Trainer(
            self.cfg,
            build_networks=build_networks,
            skip=skip,
            keep=keep,
            ckpt_dir=ckpt_dir,
        )

    def init_wandb(
            self,
    ):
        return super()._init_wandb()

    def get_summary_writer(self):
        return SummaryWriter(self._outdir)
        # if job_type is None:
        #     return SummaryWriter(self._outdir)
        # # sdir = super()._get_summary_dir(job_type=job_type)
        # # sdir = os.getcwd()
        # return SummaryWriter(job_type)

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
        if self._rank == 0:
            if init_wandb:
                import wandb
                log.warning(f'Initialize WandB from {rank}:{local_rank}')
                run = self.init_wandb()
                assert run is wandb.run
                # run.watch(
                #     self.trainer.dynamics,
                #     log="all"
                # )
                run.config['SIZE'] = self._size
            if init_aim:
                log.warning(f'Initializing Aim from {rank}:{local_rank}')
                arun = self.init_aim()
                if arun is not None:
                    if torch.cuda.is_available():
                        arun['ngpus'] = self._size
                    else:
                        arun['ncpus'] = self._size

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
            nprint: Optional[int] = None,
            nlog: Optional[int] = None,
            beta: Optional[float | list[float] | dict[str, float]] = None,
            # rich: Optional[bool] = None,
    ):
        # nchains = 16 if nchains is None else nchains
        jobdir = self.get_jobdir(job_type='train')
        writer = None
        if self._rank == 0:
            writer = self.get_summary_writer()

        # logfile = jobdir.joinpath(f'train-{RANK}.log')
        # with open(logfile.as_posix(), 'wt') as logfile:
        # console = Console(log_path=False, file=logfile)
        console = get_console(record=True)
        # console = Console(log_path=False, record=True, width=210)
        self.trainer.set_console(console)
        if self.config.annealing_schedule.dynamic:
            output = self.trainer.train_dynamic(
                x=x,
                nera=nera,
                nepoch=nepoch,
                run=self.run,
                arun=self.arun,
                writer=writer,
                train_dir=jobdir,
                skip=skip,
                beta=beta,
                # nprint=nprint,
                # nlog=nlog
            )
        else:
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
                nprint=nprint,
                nlog=nlog
            )
        # if self.trainer._is_chief:
        #     summaryfile = jobdir.joinpath('summaries.txt')
        #     with open(summaryfile.as_posix(), 'w') as f:
        #         f.writelines(output['summaries'])
        # fname = f'train-{RANK}'
        # txtfile = jobdir.joinpath(f'{fname}.txt')
        # htmlfile = jobdir.joinpath(f'{fname}.html')
        # console.save_text(txtfile.as_posix(), clear=False)
        # console.save_html(htmlfile.as_posix())

        if self.trainer._is_chief:
            dset = self.save_dataset(
                # output=output,
                nchains=nchains,
                job_type='train',
                outdir=jobdir,
                tables=output.get('tables', None),
            )
            output['dataset'] = dset

        if writer is not None:
            writer.close()

        return output

    def evaluate(
            self,
            job_type: str,
            therm_frac: float = 0.1,
            beta: Optional[float] = None,
            nchains: Optional[int] = None,
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
            eval_steps: Optional[int] = None,
            nprint: Optional[int] = None,
    ) -> dict | None:
        """Evaluate model."""
        # if RANK != 0:
        if not self.trainer._is_chief:
            return None

        assert job_type in ['eval', 'hmc']
        jobdir = self.get_jobdir(job_type)
        writer = self.get_summary_writer()
        console = get_console(record=True)
        self.trainer.set_console(console)
        output = self.trainer.eval(
            beta=beta,
            run=self.run,
            arun=self.arun,
            writer=writer,
            nchains=nchains,
            job_type=job_type,
            eps=eps,
            nleapfrog=nleapfrog,
            eval_steps=eval_steps,
            nprint=nprint,
        )

        output['dataset'] = self.save_dataset(
            # output=output,
            job_type=job_type,
            outdir=jobdir,
            tables=output.get('tables', None),
            therm_frac=therm_frac,
        )
        if writer is not None:
            writer.close()

        return output
