"""
trainer.py
Implements methods for training L2HMC sampler
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional
from omegaconf import DictConfig

import tensorflow as tf
import tensorflow.python.framework.ops as ops
import horovod.tensorflow as hvd  # type: ignore

import aim
from aim import Distribution

import numpy as np
from rich.live import Live
from rich.table import Table
from rich import box
from tensorflow._api.v2.train import CheckpointManager

from tensorflow.python.keras import backend as K

from l2hmc.utils.rich import get_width, is_interactive
from l2hmc.configs import ExperimentConfig
from contextlib import nullcontext

from l2hmc.learning_rate.tensorflow.learning_rate import ReduceLROnPlateau

from l2hmc.dynamics.tensorflow.dynamics import Dynamics
from l2hmc.group.u1.tensorflow.group import U1Phase
from l2hmc.group.su3.tensorflow.group import SU3
from l2hmc.lattice.u1.tensorflow.lattice import LatticeU1
from l2hmc.lattice.su3.tensorflow.lattice import LatticeSU3
from l2hmc.loss.tensorflow.loss import LatticeLoss
from l2hmc.network.tensorflow.network import NetworkFactory
from l2hmc.trackers.tensorflow.trackers import update_summaries
from l2hmc.trainers.trainer import BaseTrainer
from l2hmc.utils.history import summarize_dict
from l2hmc.utils.rich import add_columns
from l2hmc.utils.step_timer import StepTimer


WIDTH = int(os.environ.get('COLUMNS', 150))

# tf.autograph.set_verbosity(0)
# os.environ['AUTOGRAPH_VERBOSITY'] = '0'
# JIT_COMPILE = (len(os.environ.get('JIT_COMPILE', '')) > 0)

log = logging.getLogger(__name__)

Tensor = tf.Tensor
Array = np.ndarray
TF_FLOAT = tf.keras.backend.floatx()
NP_INT = (np.int8, np.int16, np.int32, np.int64)
Model = tf.keras.Model
Optimizer = tf.keras.optimizers.Optimizer
TensorLike = tf.types.experimental.TensorLike


def plot_models(dynamics: Dynamics, logdir: os.PathLike):
    logdir = Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    if dynamics.config.use_separate_networks:
        networks = {
            'dynamics_vnet': dynamics._get_vnet(0),
            'dynamics_xnet0': dynamics._get_xnet(0, first=True),
            'dynamics_xnet1': dynamics._get_xnet(0, first=False),
        }
    else:
        networks = {
            'dynamics_vnet': dynamics._get_vnet(0),
            'dynamics_xnet': dynamics._get_xnet(0, first=True),
        }

    for key, val in networks.items():
        try:
            fpath = logdir.joinpath(f'{key}.png')
            tf.keras.utils.plot_model(
                val,
                show_shapes=True,
                to_file=fpath.as_posix()
            )
        except Exception:
            log.warning('Unable to plot dynamics networks, continuing!')
            pass


HVD_FP_MAP = {
    'fp16': hvd.Compression.fp16,
    'none': hvd.Compression.none
}


def reset_optimizer(optimizer: Optimizer):
    """Reset optimizer states when changing beta during training."""
    # NOTE: We don't want to reset iteration counter. From tf docs:
    # > The first value is always the iterations count of the optimizer,
    # > followed by the optimizer's state variables in the order they are
    # > created.
    weight_shapes = [x.shape for x in optimizer.get_weights()[1:]]
    optimizer.set_weights([
        tf.zeros_like(x) for x in weight_shapes
    ])
    return optimizer


def flatten(x: Tensor):
    return tf.reshape(x, (x.shape[0], -1))


def is_dist(z: Tensor | ops.EagerTensor | np.ndarray) -> bool:
    return len(z.shape) > 1 or (
        len(z.shape) == 1
        and z.shape[0] > 1  # type:ignore
    )


METRICS = {
    'step': 0,
    'beta': 0.,
    'dt': 0.0,
    'hmc_step': 0,
    'loss': 0.0,
    'dQint': 0.0,
    'dQsin': 0.0,
    'energy': 0.0,
    'logprob': 0.0,
    'eps': 0.0,
    'logdet': 0.0,
    'acc': 0.0,
    'acc_mask': 0.0,
    'sumlogdet': 0.0,
    'plaqs': 0.0,
    'sinQ': 0.0,
    'intQ': 0.0,
    'eps': 0.0,
    'acc': 0.0,
}

# Metric = tf.Tensor | ops.EagerTensor | list | np.ndarray | int | float | bool


# TODO: Replace arguments in __init__ call below with configs.TrainerConfig
class Trainer(BaseTrainer):
    def __init__(
            self,
            cfg: DictConfig | ExperimentConfig,
            build_networks: bool = True,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
    ) -> None:
        super(Trainer, self).__init__(cfg=cfg, keep=keep, skip=skip)
        # assert isinstance(self.config, ExperimentConfig)
        if self.config.compression in [True, 'fp16']:
            self.compression = hvd.Compression.fp16
        else:
            self.compression = hvd.Compression.none

        self._gstep = 0
        self.lattice = self.build_lattice()
        self.loss_fn = self.build_loss_fn()
        self.dynamics = self.build_dynamics(
            build_networks=build_networks
        )
        assert (
            self.dynamics is not None
            and isinstance(self.dynamics, Dynamics)
        )
        self.optimizer = self.build_optimizer()
        self.verbose = self.config.dynamics.verbose
        # compression = 'fp16'
        # self.verbose = not skip_tracking
        # self.compression = HVD_FP_MAP['fp16']
        # self.lr_schedule = self.build_lr_schedule()
        # skip_tracking = os.environ.get('SKIP_TRACKING', False)
        self.clip_norm = self.config.learning_rate.clip_norm
        self.reduce_lr = ReduceLROnPlateau(self.config.learning_rate)
        self.reduce_lr.set_model(self.dynamics)
        self.reduce_lr.set_optimizer(self.optimizer)
        self.rank = hvd.local_rank()
        self.global_rank = hvd.rank()
        self._is_chief = self.rank == 0 and self.global_rank == 0
        if self.config.dynamics.group == 'U1':
            self.g = U1Phase()
        elif self.config.dynamics.group == 'SU3':
            self.g = SU3()
        else:
            raise ValueError

    def draw_x(self):
        return flatten(self.g.random(
            list(self.config.dynamics.xshape)
        ))

    def reset_optimizer(self) -> None:
        if len(self.optimizer.variables()) > 0:
            log.warning('Resetting optimizer state!')
            for var in self.optimizer.variables():
                var.assign(tf.zeros_like(var))

    def build_lattice(self):
        group = str(self.config.dynamics.group).upper()
        kwargs = {
            'nchains': self.config.dynamics.nchains,
            'shape': list(self.config.dynamics.latvolume),
        }
        if group == 'U1':
            return LatticeU1(**kwargs)
        if group == 'SU3':
            c1 = (
                self.config.c1
                if self.config.c1 is not None
                else 0.0
            )

            return LatticeSU3(c1=c1, **kwargs)

        raise ValueError('Unexpected value in `config.dynamics.group`')

    def build_loss_fn(self) -> Callable:
        assert isinstance(self.lattice, (LatticeU1, LatticeSU3))
        return LatticeLoss(
            lattice=self.lattice,
            loss_config=self.config.loss,
        )

    def build_dynamics(
            self,
            build_networks: bool = True,
    ) -> Dynamics:
        input_spec = self.get_input_spec()
        net_factory = None
        if build_networks:
            net_factory = NetworkFactory(
                input_spec=input_spec,
                conv_config=self.config.conv,
                network_config=self.config.network,
                net_weights=self.config.net_weights,
            )
        return Dynamics(config=self.config.dynamics,
                        potential_fn=self.lattice.action,
                        network_factory=net_factory)

    def build_optimizer(
            self,
    ) -> Optimizer:
        # TODO: Expand method, re-build LR scheduler, etc
        # TODO: Replace `LearningRateConfig` with `OptimizerConfig`
        # TODO: Optionally, break up in to lrScheduler, OptimizerConfig ?
        return tf.keras.optimizers.Adam(self.config.learning_rate.lr_init)

    def get_lr(self) -> float:
        return K.get_value(self.optimizer.lr)

    def setup_CheckpointManager(self, outdir: os.PathLike):
        ckptdir = Path(outdir).joinpath('checkpoints')
        ckpt = tf.train.Checkpoint(dynamics=self.dynamics,
                                   optimizer=self.optimizer)
        manager = tf.train.CheckpointManager(ckpt,
                                             ckptdir.as_posix(),
                                             max_to_keep=5)
        if manager.latest_checkpoint:
            ckpt.restore(manager.latest_checkpoint)
            log.info(f'Restored checkpoint from: {manager.latest_checkpoint}')

            netdir = Path(outdir).joinpath('networks')
            if netdir.is_dir():
                log.info(f'Loading dynamics networks from: {netdir}')
                nets = self.dynamics.load_networks(netdir)
                self.dynamics.xnet = nets['xnet']
                self.dynamics.vnet = nets['vnet']
                self.dynamics.xeps = nets['xeps']
                self.dynamics.veps = nets['veps']
                log.info(f'Networks successfully loaded from {netdir}')

        return manager

    def save_ckpt(
            self,
            manager: CheckpointManager,
    ) -> os.PathLike | None:
        if not self._is_chief:
            return

        ckpt = manager.save()
        return ckpt

    def should_log(self, epoch):
        return (
            epoch % self.steps.log == 0
            and self._is_chief
        )

    def should_print(self, epoch):
        return (
            epoch % self.steps.print == 0
            and self._is_chief
        )

    def record_metrics(
            self,
            metrics: dict,
            job_type: str,
            step: Optional[int] = None,
            # record: Optional[dict] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            model: Optional[Model] = None,
            optimizer: Optional[Optimizer] = None,
    ):
        # record = {} if record is None else record
        assert job_type in ['train', 'eval', 'hmc']
        if step is None:
            timer = self.timers.get(job_type, None)
            if isinstance(timer, StepTimer):
                step = timer.iterations

        if step is not None:
            metrics.update({f'{job_type}_step': step})

        # record.update({
        #     'loss': metrics.get('loss', None),
        #     'dQint': metrics.get('dQint', None),
        #     'dQsin': metrics.get('dQsin', None),
        # })
        if job_type == 'train' and step is not None:
            metrics['lr'] = K.get_value(self.optimizer.lr)

        if job_type == 'eval':
            _ = metrics.pop('eps', None)

        if job_type in ['hmc']:
            _ = metrics.pop('xeps', None)
            _ = metrics.pop('veps', None)

        metrics.update(self.metrics_to_numpy(metrics))
        avgs = self.histories[job_type].update(metrics)
        summary = summarize_dict(avgs)

        # if writer is not None and self.verbose and step is not None:
        if (
                step is not None
                and writer is not None
        ):
            update_summaries(step=step,
                             model=model,
                             metrics=metrics,
                             prefix=job_type,
                             optimizer=optimizer)
            writer.flush()

        if self.config.init_wandb or self.config.init_aim:
            self.track_metrics(
                record=metrics,
                avgs=avgs,
                job_type=job_type,
                step=step,
                run=run,
                arun=arun,
            )

        return avgs, summary

    def track_metrics(
            self,
            record: dict[str, Tensor],
            avgs: dict[str, Tensor],
            job_type: str,
            step: Optional[int],
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
    ) -> None:
        if self.rank != 0:
            return
        dQdict = None
        dQint = record.get('dQint', None)
        if dQint is not None:
            dQdict = {
                f'dQint/{job_type}': {
                    'val': dQint,
                    'step': step,
                    'avg': tf.reduce_mean(dQint),
                }
            }

        if run is not None:
            run.log({f'wandb/{job_type}': record}, commit=False)
            run.log({f'avgs/wandb.{job_type}': avgs}, commit=False)
            if dQdict is not None:
                run.log(dQdict, commit=True)
        if arun is not None:
            kwargs = {
                'step': step,
                'job_type': job_type,
                'arun': arun
            }

            if dQdict is not None:
                self.aim_track({'dQint': dQint}, prefix='dQ', **kwargs)

            try:
                self.aim_track(avgs, prefix='avgs', **kwargs)
            except Exception as e:
                log.exception(e)
                log.warning('Unable to aim_track `avgs` !')
            try:
                self.aim_track(record, prefix='record', **kwargs)
            except Exception as e:
                log.exception(e)
                log.warning('Unable to aim_track `record` !')

    # @tf.function
    def hmc_step(
            self,
            inputs: tuple[Tensor, Tensor],
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> tuple[Tensor, dict]:
        xo, metrics = self.dynamics.apply_transition_hmc(
            inputs,
            eps=eps,
            nleapfrog=nleapfrog
        )
        xp = metrics.pop('mc_states').proposed.x
        loss = self.loss_fn(x_init=inputs[0], x_prop=xp, acc=metrics['acc'])
        if self.verbose:
            lmetrics = self.loss_fn.lattice_metrics(xinit=inputs[0], xout=xo)
            metrics.update(lmetrics)
        metrics.update({'loss': loss})

        return xo, metrics

    @tf.function
    def eval_step(
            self,
            inputs: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, dict]:
        """
        # @tf.function(experimental_follow_type_hints=True,
                       jit_compile=JIT_COMPILE)
        """
        assert self.dynamics is not None
        xout, metrics = self.dynamics(inputs, training=False)  # type:ignore
        xout = self.g.compat_proj(xout)
        xp = metrics.pop('mc_states').proposed.x
        loss = self.loss_fn(x_init=inputs[0], x_prop=xp, acc=metrics['acc'])
        if self.verbose:
            lmetrics = self.loss_fn.lattice_metrics(xinit=inputs[0], xout=xout)
            metrics.update(lmetrics)

        metrics.update({
            'beta': inputs[1],
            'loss': loss,
        })
        metrics.update({'loss': loss})
        assert isinstance(metrics, dict)

        return xout, metrics

    def get_context_manager(self, table: Table) -> Live | nullcontext:
        make_live = (
            int(get_width()) > 150    # make sure wide enough to fit table
            and self._is_chief
            and hvd.size() == 1       # not worth the trouble when distributed
            and not is_interactive()  # AND not in a jupyter / ipython kernel
        )
        if make_live:
            return Live(
                table,
                # screen=True,
                transient=True,
                # auto_refresh=False,
                console=self.console,
                vertical_overflow='visible'
            )

        return nullcontext()

    def _setup_eval(
            self,
            beta: Optional[Tensor | float] = None,
            eval_steps: Optional[int] = None,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            run: Optional[Any] = None,
            writer: Optional[Any] = None,
            job_type: Optional[str] = 'eval',
            nchains: Optional[int] = None,
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> dict:
        assert job_type in ['eval', 'hmc']

        if isinstance(skip, str):
            skip = [skip]

        if beta is None:
            beta = tf.constant(
                self.config.annealing_schedule.beta_final,
                dtype=TF_FLOAT
            )
        elif isinstance(beta, float):
            beta = tf.constant(beta, dtype=TF_FLOAT)

        if nleapfrog is None and str(job_type).lower() == 'hmc':
            nleapfrog = self.config.dynamics.nleapfrog
            assert isinstance(nleapfrog, int)
            if self.config.dynamics.merge_directions:
                nleapfrog *= 2

        if eps is None and str(job_type).lower() == 'hmc':
            eps = self.dynamics.config.eps_hmc
            log.warn(
                'Step size `eps` not specified for HMC! '
                f'Using default: {eps:.4f} for generic HMC'
            )

        if x is None:
            x = self.lattice.random()

        log.warning(f'x.shape (original): {x.shape}')
        if nchains is not None:
            if isinstance(nchains, int) and nchains > 0:
                x = x[:nchains]  # type: ignore

        assert isinstance(x, Tensor)
        log.warning(f'x[:nchains].shape: {x.shape}')

        if writer is not None:
            writer.set_as_default()

        table = Table(row_styles=['dim', 'none'], box=box.HORIZONTALS)
        eval_steps = self.steps.test if eval_steps is None else eval_steps
        assert isinstance(eval_steps, int)
        nprint = max(1, eval_steps // 20)
        nlog = max((1, min((10, eval_steps))))
        if nlog <= eval_steps:
            nlog = min(10, max(1, eval_steps // 100))

        if run is not None:
            run.config.update({
                job_type: {'beta': beta, 'xshape': x.shape.as_list()}
            })

        assert x is not None and isinstance(x, Tensor)
        assert beta is not None and isinstance(beta, Tensor)
        output = {
            'x': x,
            'eps': eps,
            'beta': beta,
            'nlog': nlog,
            'table': table,
            'nprint': nprint,
            'eval_steps': eval_steps,
            'nleapfrog': nleapfrog,
        }
        log.info(
            '\n'.join([
                f'{k} = {v}' for k, v in output.items()
                if k != 'x'
            ])
        )
        return output

    def eval(
            self,
            beta: Optional[Tensor | float] = None,
            eval_steps: Optional[int] = None,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            job_type: Optional[str] = 'eval',
            nchains: Optional[int] = None,
            eps: Optional[float] = None,
            nleapfrog: Optional[int] = None,
    ) -> dict:
        """Evaluate model."""
        assert job_type in ['hmc', 'eval']

        tables = {}
        summaries = []
        patience = 5
        stuck_counter = 0
        setup = self._setup_eval(
            x=x,
            run=run,
            skip=skip,
            beta=beta,
            eps=eps,
            writer=writer,
            nchains=nchains,
            job_type=job_type,
            eval_steps=eval_steps,
        )
        x = setup['x']
        eps = setup['eps']
        beta = setup['beta']
        table = setup['table']
        nleapfrog = setup['nleapfrog']
        eval_steps = setup['eval_steps']
        # assert eps is not None and isinstance(eps, (float, tf.Tensor))
        timer = self.timers.get(job_type, None)
        history = self.histories.get(job_type, None)
        assert (
            eval_steps is not None
            and timer is not None
            and history is not None
            and x is not None
            and beta is not None
        )

        def eval_fn(inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, dict]:
            if job_type == 'hmc':
                assert eps is not None
                return self.hmc_step(inputs, eps=eps, nleapfrog=nleapfrog)
            return self.eval_step(inputs)  # type:ignore

        with self.get_context_manager(table) as ctx:
            for step in range(eval_steps):
                timer.start()
                x, metrics = eval_fn((x, beta))  # type:ignore
                dt = timer.stop()
                if step % setup['nprint'] == 0 or step % setup['nlog'] == 0:
                    record = {
                        f'{job_type}_step': step,
                        'dt': dt,
                        'beta': beta,
                        'loss': metrics.pop('loss', None),
                        'dQsin': metrics.pop('dQsin', None),
                        'dQint': metrics.pop('dQint', None),
                    }
                    record.update(metrics)
                    if job_type == 'hmc':
                        acc = metrics.get('acc_mask', None)
                        record['eps'] = eps
                        if acc is not None and eps is not None:
                            acc_avg = tf.reduce_mean(acc)
                            if acc_avg < 0.66:
                                eps -= (eps / 10.)
                            else:
                                eps += (eps / 10.)

                    avgs, summary = self.record_metrics(run=run,
                                                        arun=arun,
                                                        step=step,
                                                        # record={},
                                                        writer=writer,
                                                        metrics=record,
                                                        job_type=job_type)
                    summaries.append(summary)

                    if (
                            # not isinstance(setup['ctx'], Live)
                            step % setup['nprint'] == 0
                    ):
                        log.info(summary)

                    if step == 0:
                        table = add_columns(avgs, table)
                    else:
                        table.add_row(*[f'{v}' for _, v in avgs.items()])

                    if avgs.get('acc', 1.0) <= 1e-5:
                        if stuck_counter < patience:
                            stuck_counter += 1
                        else:
                            self.console.log('Chains are stuck! Re-drawing x!')
                            x = self.lattice.random()
                            stuck_counter = 0

                if isinstance(ctx, Live):
                    ctx.console.clear_live()

        tables[str(0)] = table

        return {
            'timer': timer,
            'history': history,
            'summaries': summaries,
            'tables': tables,
        }

    @tf.function
    def train_step(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, dict]:
        """Implement a single training step (forward + backward) pass.

        - NOTE: Wrapper possibilities:
            ```python
            @tf.function(
                    experimental_follow_type_hints=True,
                    jit_compile=JIT_COMPILE
            )
            @tf.function(
                experimental_follow_type_hints=True,
                input_signature=[
                    tf.TensorSpec(shape=None, dtype=TF_FLOAT),
                    tf.TensorSpec(shape=None, dtype=TF_FLOAT),
                ],
            )
            ```
        """
        xinit, beta = inputs
        # xinit = self.dynamics.g.compat_proj(xinit)
        # should_clip = (
        #     (self.lr_config.clip_norm > 0)
        #     if clip_grads is None else clip_grads
        # )
        aw = self.config.loss.aux_weight
        assert (
            self.dynamics is not None
            # and isinstance(self.dynamics, Dynamics)
        )
        with tf.GradientTape() as tape:
            tape.watch(xinit)
            xout, metrics = self.dynamics(  # type:ignore
                (xinit, beta),
                training=True
            )
            xprop = metrics.pop('mc_states').proposed.x
            loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])

            if aw > 0:
                yinit = self.draw_x()
                _, metrics_ = self.dynamics(  # type:ignore
                    (yinit, beta),
                    training=True
                )
                yprop = metrics_.pop('mc_states').proposed.x
                aux_loss = aw * self.loss_fn(x_init=yinit,
                                             x_prop=yprop,
                                             acc=metrics_['acc'])
                loss += aw * aux_loss
                # loss = (loss + aux_loss) / (1. + self.aux_weight)

        # tape = hvd.DistributedGradientTape(
        #     tape,
        #     compression=self.compression
        # )
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, self.dynamics.trainable_variables)
        if self.clip_norm > 0.0:
            grads = [
                tf.clip_by_norm(grad, clip_norm=self.clip_norm)
                for grad in grads
            ]
        self.optimizer.apply_gradients(
            zip(grads, self.dynamics.trainable_variables)
        )
        if self.timers['train'].iterations == 0:
            hvd.broadcast_variables(self.dynamics.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        metrics['loss'] = loss
        if self.verbose:
            lmetrics = self.loss_fn.lattice_metrics(xinit=xinit, xout=xout)
            metrics.update(lmetrics)

        return xout, metrics

    def train_epoch(
            self,
            x: Tensor,
            beta: float | Tensor,
            era: Optional[int] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            nepoch: Optional[int] = None,
            writer: Optional[Any] = None,
            extend: Optional[int] = None,
    ) -> tuple[Tensor, dict]:
        rows = {}
        summaries = []
        extend = 1 if extend is None else extend
        record = {'era': 0, 'epoch': 0, 'beta': 0.0, 'dt': 0.0}
        table = Table(
            box=box.HORIZONTALS,
            row_styles=['dim', 'none'],
        )

        # nepoch = self.steps.nepoch * extend
        nepoch = self.steps.nepoch if nepoch is None else nepoch
        assert isinstance(nepoch, int)
        nepoch *= extend
        losses = []
        ctx = self.get_context_manager(table)
        with ctx:
            if isinstance(ctx, Live):
                tstr = ' '.join([
                    f'ERA: {era}/{self.steps.nera}',
                    f'BETA: {beta:.3f}',
                ])
                ctx.console.clear_live()
                ctx.console.rule(tstr)
                ctx.update(table)

            for epoch in range(nepoch):
                self.timers['train'].start()
                x, metrics = self.train_step((x, beta))  # type:ignore
                dt = self.timers['train'].stop()
                losses.append(metrics['loss'])
                self._gstep += 1
                # if (
                #         self._is_chief and (
                #             self.should_print(epoch)
                #             or self.should_log(epoch)
                #         )
                # ):
                if self.should_print(epoch) or self.should_log(epoch):
                    record = {
                        'era': era,
                        'epoch': epoch,
                        'train_step': self._gstep,
                        'dt': dt,
                        'beta': beta,
                        'loss': metrics.pop('loss', None),
                        'dQsin': metrics.pop('dQsin', None),
                        'dQint': metrics.pop('dQint', None)
                    }
                    record.update(metrics)
                    avgs, summary = self.record_metrics(
                        run=run,
                        arun=arun,
                        step=self._gstep,
                        writer=writer,
                        metrics=record,  # metrics from Dynamics
                        job_type='train',
                        model=self.dynamics,
                        optimizer=self.optimizer,
                    )
                    rows[self._gstep] = avgs
                    summaries.append(summary)

                    if (
                            self.should_print(epoch)
                            # and not isinstance(ctx, Live)
                    ):
                        log.info(summary)

                    if epoch == 0:
                        table = add_columns(avgs, table)
                    else:
                        table.add_row(*[f'{v}' for _, v in avgs.items()])

                    if avgs.get('acc', 1.0) < 1e-5:
                        self.reset_optimizer()
                        log.warning('Chains are stuck! Re-drawing x !')
                        x = self.draw_x()

        data = {
            'rows': rows,
            'table': table,
            'losses': losses,
            'summaries': summaries,
        }

        return x, data

    def _setup_training(
            self,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            train_dir: Optional[os.PathLike] = None,
            writer: Optional[Any] = None,
            nera: Optional[int] = None,
            nepoch: Optional[int] = None,
            beta: Optional[float | list[float] | dict[str, float]] = None,
    ) -> dict:
        skip = [skip] if isinstance(skip, str) else skip

        # -- Tensorflow specific
        if writer is not None:
            writer.set_as_default()

        if train_dir is None:
            train_dir = Path(os.getcwd()).joinpath(
                self._created, 'train'
            )
            train_dir.mkdir(exist_ok=True, parents=True)

        if x is None:
            x = flatten(self.g.random(list(self.xshape)))

        # -- Setup checkpoint manager for TensorFlow --------
        manager = self.setup_CheckpointManager(train_dir)
        self._gstep = K.get_value(self.optimizer.iterations)
        # ----------------------------------------------------
        # -- Setup Step information (nera, nepoch, etc). -----
        nera = self.config.steps.nera if nera is None else nera
        nepoch = self.config.steps.nepoch if nepoch is None else nepoch
        extend = self.config.steps.extend_last_era
        assert isinstance(nera, int)
        assert isinstance(nepoch, int)

        if beta is not None:
            assert isinstance(beta, (float, list))
            if isinstance(beta, list):
                assert len(beta) == nera, 'Expected len(beta) == nera'
            else:
                beta = nera * [beta]

            betas = {f'{i}': b for i, b in zip(range(nera), beta)}
        else:
            betas = self.config.annealing_schedule.setup(
                nera=nera,
                nepoch=nepoch,
            )

        beta_final = list(betas.values())[-1]
        assert beta_final is not None and isinstance(beta_final, float)
        return {
            'x': x,
            'nera': nera,
            'nepoch': nepoch,
            'extend': extend,
            'betas': betas,
            'manager': manager,
            'train_dir': train_dir,
            'beta_final': beta_final,
        }

    def train(
            self,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            train_dir: Optional[os.PathLike] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            nera: Optional[int] = None,
            nepoch: Optional[int] = None,
            beta: Optional[float | list[float] | dict[str, float]] = None,
    ) -> dict:
        """Perform training and return dictionary of results."""
        setup = self._setup_training(
            x=x,
            skip=skip,
            train_dir=train_dir,
            nera=nera,
            nepoch=nepoch,
            beta=beta,
        )
        era = 0
        # epoch = 0
        extend = 1
        x = setup['x']
        nera = setup['nera']
        betas = setup['betas']
        nepoch = setup['nepoch']
        extend = setup['extend']
        manager = setup['manager']
        train_dir = setup['train_dir']
        beta_final = setup['beta_final']
        assert x is not None
        assert nera is not None
        assert train_dir is not None
        for era in range(nera):
            b = tf.constant(betas.get(str(era), beta_final))
            if era == (nera - 1) and self.steps.extend_last_era is not None:
                extend = int(self.steps.extend_last_era)

            if self._is_chief:
                if era > 1 and str(era - 1) in self.summaries['train']:
                    esummary = self.histories['train'].era_summary(f'{era-1}')
                    log.info(f'Avgs over last era:\n {esummary}\n')

                self.console.rule(f'ERA: {era} / {nera}, BETA: {b:.3f}')

            epoch_start = time.time()
            x, edata = self.train_epoch(
                x=x,
                beta=b,
                era=era,
                run=run,
                arun=arun,
                writer=writer,
                extend=extend,
                nepoch=nepoch,
            )
            st0 = time.time()

            self.rows['train'][str(era)] = edata['rows']
            self.tables['train'][str(era)] = edata['table']
            self.summaries['train'][str(era)] = edata['summaries']
            losses = tf.stack(edata['losses'][1:])
            if self.config.annealing_schedule.dynamic:
                dy_avg = tf.reduce_mean(
                    losses[1:] - losses[:-1]  # type:ignore
                )
                if dy_avg > 0:
                    b -= (b / 10.)
                else:
                    b += (b / 10.)

            if (era + 1) == self.steps.nera or (era + 1) % 5 == 0:
                self.save_ckpt(manager)

            if self._is_chief:
                log.info(f'Saving took: {time.time() - st0:<5g}s')
                log.info(f'Era {era} took: {time.time() - epoch_start:<5g}s')

        return {
            'timer': self.timers['train'],
            'rows': self.rows['train'],
            'summaries': self.summaries['train'],
            'history': self.histories['train'],
            'tables': self.tables['train'],
        }

    def train_dynamic(
            self,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            train_dir: Optional[os.PathLike] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            nera: Optional[int] = None,
            nepoch: Optional[int] = None,
            beta: Optional[float | list[float] | dict[str, float]] = None,
    ) -> dict:
        """Perform training and return dictionary of results."""
        setup = self._setup_training(
            x=x,
            skip=skip,
            train_dir=train_dir,
            nera=nera,
            nepoch=nepoch,
            beta=beta,
        )
        era = 0
        # epoch = 0
        extend = 1
        x = setup['x']
        nera = setup['nera']
        betas = setup['betas']
        nepoch = setup['nepoch']
        extend = setup['extend']
        manager = setup['manager']
        train_dir = setup['train_dir']
        beta_final = setup['beta_final']
        b = tf.constant(betas.get(str(era), beta_final))
        assert x is not None
        assert nera is not None
        assert train_dir is not None
        while b < beta_final:
            b = tf.constant(betas.get(str(era), beta_final))
            if era == (nera - 1) and self.steps.extend_last_era is not None:
                extend = int(self.steps.extend_last_era)

            if self._is_chief:
                if era > 1 and str(era - 1) in self.summaries['train']:
                    esummary = self.histories['train'].era_summary(f'{era-1}')
                    log.info(f'Avgs over last era:\n {esummary}\n')

                self.console.rule(f'ERA: {era} / {nera}, BETA: {b:.3f}')

            epoch_start = time.time()
            x, edata = self.train_epoch(
                x=x,
                beta=b,
                era=era,
                run=run,
                arun=arun,
                writer=writer,
                extend=extend,
                nepoch=nepoch,
            )
            st0 = time.time()

            self.rows['train'][str(era)] = edata['rows']
            self.tables['train'][str(era)] = edata['table']
            self.summaries['train'][str(era)] = edata['summaries']
            losses = tf.stack(edata['losses'][1:])
            if self.config.annealing_schedule.dynamic:
                dy_avg = tf.reduce_mean(
                    losses[1:] - losses[:-1]  # type:ignore
                )
                if dy_avg > 0:
                    b -= (b / 10.)
                else:
                    b += (b / 10.)

            if (era + 1) == self.steps.nera or (era + 1) % 5 == 0:
                _ = self.save_ckpt(manager)

            if self._is_chief:
                log.info(f'Saving took: {time.time() - st0:<5g}s')
                log.info(f'Era {era} took: {time.time() - epoch_start:<5g}s')

            era += 1

        return {
            'timer': self.timers['train'],
            'rows': self.rows['train'],
            'summaries': self.summaries['train'],
            'history': self.histories['train'],
            'tables': self.tables['train'],
        }

    def metric_to_numpy(
            self,
            metric: Any,
            # key: str = '',
    ):
        """Consistently convert `metric` to np.ndarray."""
        if isinstance(metric, np.ScalarType):
            return metric

        if isinstance(metric, np.ndarray):
            return metric  # [~np.isnan(metric)]

        if (
                isinstance(metric, Tensor)
                and hasattr(metric, 'numpy')
                and isinstance(metric.numpy, Callable)
        ):
            # tmp = metric.numpy()
            # return tmp[~np.isnan(tmp)]
            return metric.numpy()

        elif isinstance(metric, list):
            if isinstance(metric[0], np.ndarray):
                # metric = np.stack(metric)
                # return metric[~np.isnan(metric)]
                metric = np.stack(metric)

            if isinstance(metric[0], Tensor):
                stack = tf.stack(metric)
                # stack = tf.stack(metric)
                # stack = stack[~tf.math.is_nan(stack)]
                if (
                        hasattr(stack, 'numpy')
                        and isinstance(stack.numpy, Callable)
                ):
                    return stack.numpy()
            else:
                return np.array(metric)
                # tmp = np.array(metric)
                # return tmp[~np.isnan(tmp)]

            # tmp = np.array(metric)
            # return tmp[~np.isnan(tmp)]
            return np.array(metric)

        else:
            raise ValueError(
                f'Unexpected type for metric: {type(metric)}'
            )

    def aim_track(
            self,
            metrics: dict,
            step: int,
            job_type: str,
            arun: aim.Run,
            prefix: Optional[str] = None,
    ) -> None:
        context = {'subset': job_type}
        dtype = getattr(step, 'dtype', None)
        if dtype is not None and dtype in NP_INT:
            try:
                step = step.item()  # type:ignore
            except AttributeError:
                pass

        for key, val in metrics.items():
            if prefix is not None:
                name = f'{prefix}/{key}'
            else:
                name = f'{key}'

            akws = {
                'step': step,
                'name': name,
                'context': context,
            }

            if isinstance(val, ops.EagerTensor):
                val = val.numpy()

            elif isinstance(val, dict):
                for k, v in val.items():
                    self.aim_track(
                        v,
                        step=step,
                        arun=arun,
                        job_type=job_type,
                        prefix=f'{name}/{k}',
                    )
            elif isinstance(val, (Tensor, ops.EagerTensor, np.ndarray)):
                if isinstance(val, ops.EagerTensor):
                    val = val.numpy()
                # check to see if we should track as Distribution
                if is_dist(val):
                    dist = Distribution(val)
                    arun.track(dist, **akws)
                    akws['name'] = f'{name}/avg'
                    avg = (
                        tf.reduce_mean(val) if isinstance(val, Tensor)
                        else val.mean()
                    )
                    arun.track(avg, **akws)
                else:
                    arun.track(val, **akws)

            else:
                arun.track(val, **akws)
