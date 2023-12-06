"""
pytorch/experiment.py

Implements ptExperiment, a pytorch-specific subclass of the
Experiment base class.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
# import os
from os import PathLike
from pathlib import Path
from typing import Any, Optional

# import horovod.torch as hvd
from hydra.utils import instantiate
from omegaconf import DictConfig
import time
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from l2hmc.utils.history import BaseHistory
# from l2hmc import get_logger
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
from l2hmc.common import print_dict

log = logging.getLogger(__name__)
# log = get_logger(__name__)


Tensor = torch.Tensor

# from l2hmc.trainers.pytorch.trainer import Trainer

def train_step(
        x: torch.Tensor,
        beta: torch.Tensor,
        trainer: Trainer,
) -> tuple[torch.Tensor, dict]:
    xout, metrics = trainer.dynamics_engine((x, beta))
    mcstates = metrics.pop('mc_states')
    loss = trainer.calc_loss(
        xinit=mcstates.init.x,
        xprop=mcstates.proposed.x,
        acc=metrics['acc']
    )
    loss.register_hook(lambda grad: grad.nan_to_num())
    trainer.optimizer.zero_grad()
    loss.backward()
    # log.info(f'mcstates.init.x.grad')
    torch.nn.utils.clip_grad.clip_grad_norm(
        trainer.dynamics.parameters(),
        max_norm=0.1,
    )
    trainer.optimizer.step()
    metrics |= {'loss': loss.item()}
    print_dict(metrics, grab=False)
    return xout.detach(), metrics


def train(
        nsteps: int,
        trainer: Trainer,
        beta: float | torch.Tensor,
        nlog: int = 1,
        nprint: int = 1,
        x: Optional[torch.Tensor] = None,
        grab: Optional[bool] = None,
) -> tuple[torch.Tensor, dict]:
    beta = torch.tensor(beta) if isinstance(beta, float) else beta
    history = {}
    if x is None:
        state = exp.trainer.dynamics.random_state(beta)
        x = state.x
    assert x is not None
    for step in range(nsteps):
        log.info(f'STEP: {step}')
        x, metrics = train_step(x, beta=beta, trainer=trainer)
        if (step > 0 and step % nprint == 0):
            print_dict(metrics, grab=grab)
        if (step > 0 and step % nlog == 0):
            for key, val in metrics.items():
                try:
                    history[key].append(val)
                except KeyError:
                    history[key] = [val]
    return x, history


def evaluate(
        nsteps: int,
        exp: Experiment,
        beta: float | torch.Tensor,
        nlog: int = 1,
        nprint: int = 1,
        job_type: str = 'eval',
        eps: Optional[float] = None,
        nleapfrog: Optional[int] = None,
        x: Optional[torch.Tensor] = None,
        grab: Optional[bool] = None,
) -> tuple[torch.Tensor, BaseHistory]:
    # history = {}
    history = BaseHistory()
    beta_ = beta.item() if isinstance(beta, torch.Tensor) else beta
    if x is None:
        state = exp.trainer.dynamics.random_state(beta_)
        x = state.x
    assert x is not None
    log.info(f'Running {nsteps} steps of {job_type} at beta={beta:.4f}')
    if job_type.lower == 'hmc':
        log.info(f'Using nleapfrog={nleapfrog} steps w/ eps={eps:.4f}')
    for step in range(nsteps):
        log.info(f'STEP: {step}')
        if job_type.lower() == 'eval':
            x, metrics = exp.trainer.eval_step((x, beta_))
        elif job_type.lower() == 'hmc':
            x, metrics = exp.trainer.hmc_step(
                (x, beta),
                eps=eps,
                nleapfrog=nleapfrog,
            )
        else:
            raise ValueError(
                'Expected `job_type` to be one of [`eval`, `hmc`]'
            )
        if (step > 0 and step % nprint == 0):
            print_dict(metrics, grab=grab)
        if (step > 0 and step % nlog == 0):
            history.update(metrics)
    return x, history


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
        self.ckpt_dir = self.config.get_checkpoint_dir()
        dsetup = setup_torch_distributed(self.config.backend)
        self._size = dsetup['size']
        self._rank = dsetup['rank']
        self._local_rank = dsetup['local_rank']

        run = None
        arun = None
        if self._rank == 0 and self.config.init_wandb:
            import wandb
            log.warning(
                f'Initialize WandB from {self._rank}:{self._local_rank}'
            )
            run = super()._init_wandb() if wandb.run is None else wandb.run
            # assert run is wandb.run
            # run.watch(
            #     self.trainer.dynamics.networks,
            #     log='all',
            #     log_graph=True,
            #     criterion=self.trainer.loss_fn,
            # )
            run.config['SIZE'] = self._size
            # env = os.environ
            # _ = env.pop('LS_COLORS', None)
            # run.config['environment'] = env
            # ds_config = getattr(self.trainer, 'ds_config', None)
            # if ds_config is not None:
            #     run.config.update(ds_config)

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

        self.trainer: Trainer = self.build_trainer(
            keep=keep,
            skip=skip,
            build_networks=build_networks,
            ckpt_dir=self.ckpt_dir,
        )
        # if run is not None:
        #     import wandb
        #     logfreq = self.config.steps.log
        #     assert logfreq is not None
        #     wandb.watch(
        #         (
        #             self.trainer.dynamics.networks,
        #             self.trainer.dynamics.xeps,
        #             self.trainer.dynamics.veps,
        #         ),
        #         log='all',
        #         log_graph=True,
        #         log_freq=logfreq,
        #         # criterion=self.trainer.loss_fn,
        #     )
        #     ds_config = getattr(self.trainer, 'ds_config', None)
        #     if ds_config is not None:
        #         run.config['deepspeed_config'] = ds_config

        self.run = run
        self.arun = arun
        self._is_built = True
        assert callable(self.trainer.loss_fn)
        assert isinstance(self.trainer, Trainer)
        assert isinstance(self.trainer.dynamics, (ptDynamics, Dynamics))
        assert isinstance(self.trainer.lattice, (LatticeU1, LatticeSU3))

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
        vnet = self.trainer.dynamics._get_vnet(0)
        xnet = self.trainer.dynamics._get_xnet(0, first=True)

        with torch.autocast(  # type:ignore
            # dtype=torch.float32,
            device_type='cuda' if torch.cuda.is_available() else 'cpu'
        ):
            force = self.trainer.dynamics.grad_potential(x, beta)
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
                ).save(
                    outdir.joinpath(f'{key}-{kk}.gv').as_posix()
                )

    def update_wandb_config(
            self,
            run_id: Optional[str] = None,
    ) -> None:
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self._update_wandb_config(device=device, run_id=run_id)

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
            save_data: bool = True,
            # rich: Optional[bool] = None,
    ):
        jobdir = self.get_jobdir(job_type='train')
        writer = self.get_summary_writer() if self._rank == 0 else None
        # console = get_console(record=True)
        # self.trainer.set_console(console)
        tstart = time.time()
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
        # if self.trainer._is_chief and self.run is not None:
        #     self.run.log({
        #         f'Timers/training_total': time.time() - tstart
        #     })
        # if self.trainer._is_chief:
        #     summaryfile = jobdir.joinpath('summaries.txt')
        #     with open(summaryfile.as_posix(), 'w') as f:
        #         f.writelines(output['summaries'])
        # fname = f'train-{RANK}'
        # txtfile = jobdir.joinpath(f'{fname}.txt')
        # htmlfile = jobdir.joinpath(f'{fname}.html')
        # console.save_text(txtfile.as_posix(), clear=False)
        # console.save_html(htmlfile.as_posix())
        log.info(f'Training took: {time.time() - tstart:.4f}')

        if self.trainer._is_orchestrator:
            dset = self.save_dataset(
                # output=output,
                nchains=nchains,
                save_data=save_data,
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
            x: Optional[torch.Tensor] = None,
    ) -> dict | None:
        """Evaluate model."""
        # if RANK != 0:
        if not self.trainer._is_orchestrator:
            return None

        assert job_type in {'eval', 'hmc'}
        jobdir = self.get_jobdir(job_type)
        writer = self.get_summary_writer()
        console = get_console(record=True)
        # self.trainer.set_console(console)
        output = self.trainer.eval(
            beta=beta,
            x=x,
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
