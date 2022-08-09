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
            tf.keras.utils.plot_model(val, show_shapes=True, to_file=fpath)
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
        and z.shape[0] > 1
    )


# TODO: Replace arguments in __init__ call below with configs.TrainerConfig
class Trainer(BaseTrainer):
    def __init__(
            self,
            cfg: DictConfig | ExperimentConfig,
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
        self.dynamics = self.build_dynamics()
        self.optimizer = self.build_optimizer()
        # self.lr_schedule = self.build_lr_schedule()
        assert isinstance(self.dynamics, Dynamics)
        self.verbose = self.config.dynamics.verbose
        # skip_tracking = os.environ.get('SKIP_TRACKING', False)
        # self.verbose = not skip_tracking
        self.clip_norm = self.config.learning_rate.clip_norm
        # compression = 'fp16'
        # self.compression = HVD_FP_MAP['fp16']
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

    def build_dynamics(self) -> Dynamics:
        input_spec = self.get_input_spec()
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
            train_dir: os.PathLike,
    ) -> None:
        if not self._is_chief:
            return

        manager.save()
        self.dynamics.save_networks(train_dir)

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
            record: Optional[dict] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            model: Optional[Model] = None,
            optimizer: Optional[Optimizer] = None,
    ):
        record = {} if record is None else record
        assert job_type in ['train', 'eval', 'hmc']
        if step is None:
            timer = self.timers.get(job_type, None)
            if isinstance(timer, StepTimer):
                step = timer.iterations

        if step is not None:
            record.update({f'{job_type}_step': step})

        record.update({
            'loss': metrics.get('loss', None),
            'dQint': metrics.get('dQint', None),
            'dQsin': metrics.get('dQsin', None),
        })
        if job_type == 'train' and step is not None:
            record['lr'] = K.get_value(self.optimizer.lr)

        if job_type in ['eval', 'hmc']:
            _ = record.pop('xeps', None)
            _ = record.pop('veps', None)

        record.update(self.metrics_to_numpy(metrics))
        avgs = self.histories[job_type].update(record)
        summary = summarize_dict(avgs)

        # if writer is not None and self.verbose and step is not None:
        if (
                step is not None
                and writer is not None
        ):
            update_summaries(step=step,
                             model=model,
                             metrics=record,
                             prefix=job_type,
                             optimizer=optimizer)
            writer.flush()

        if self.config.init_wandb or self.config.init_aim:
            self.track_metrics(
                record=record,
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

    @tf.function
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

    # @tf.function(experimental_follow_type_hints=True,
    #         jit_compile=JIT_COMPILE)
    @tf.function
    def eval_step(
            self,
            inputs: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, dict]:
        xout, metrics = self.dynamics(inputs, training=False)
        xout = self.g.compat_proj(xout)
        xp = metrics.pop('mc_states').proposed.x
        loss = self.loss_fn(x_init=inputs[0], x_prop=xp, acc=metrics['acc'])
        if self.verbose:
            lmetrics = self.loss_fn.lattice_metrics(xinit=inputs[0], xout=xout)
            metrics.update(lmetrics)

        metrics.update({'loss': loss})
        assert isinstance(metrics, dict)

        return xout, metrics

    def get_context_manager(self, table: Table):
        width = get_width()
        make_live = (
            int(width) > 150          # make sure wide enough to fit table
            and hvd.size() > 1        # not worth the trouble when distributed
            and self.rank == 0        # only display from (one) main rank
            and not is_interactive()  # AND not in a jupyter / ipython kernel
        )
        if make_live:
            return Live(
                table,
                # screen=True,
                console=self.console,
                vertical_overflow='visible'
            )

        return nullcontext()

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
        if isinstance(skip, str):
            skip = [skip]

        if beta is None:
            beta = tf.constant(
                self.config.annealing_schedule.beta_final,
                dtype=TF_FLOAT
            )

        if eps is None and str(job_type).lower() == 'hmc':
            # eps = tf.constant(0.1, dtype=TF_FLOAT)
            eps = tf.constant(self.dynamics.config.eps_hmc, dtype=TF_FLOAT)
            log.warn(
                'Step size `eps` not specified for HMC! '
                f'Using default: {self.dynamics.config.eps_hmc:.3f}'
            )

        assert job_type in ['eval', 'hmc']

        if x is None:
            r = self.g.random(list(self.xshape))
            # r = self.dynamics.g.random(list(self.xshape))
            x = tf.reshape(r, (r.shape[0], -1))

        if writer is not None:
            writer.set_as_default()

        def eval_fn(inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, dict]:
            if job_type == 'eval':
                return self.eval_step(inputs)  # type:ignore

            if job_type == 'hmc':
                return self.hmc_step(
                    inputs, eps=eps, nleapfrog=nleapfrog
                )  # type: ignore

            raise ValueError

        assert isinstance(x, Tensor)  # and x.dtype == TF_FLOAT

        tables = {}
        summaries = []
        table = Table(row_styles=['dim', 'none'], box=box.HORIZONTALS)
        eval_steps = self.steps.test if eval_steps is None else eval_steps
        assert isinstance(eval_steps, int)
        nprint = max(1, eval_steps // 20)
        nlog = max((1, min((10, eval_steps))))
        if nlog <= eval_steps:
            nlog = min(10, max(1, eval_steps // 100))

        assert job_type in ['eval', 'hmc']
        timer = self.timers[job_type]
        history = self.histories[job_type]

        log.warning(f'x.shape (original): {x.shape}')
        if nchains is not None:
            if isinstance(nchains, int) and nchains > 0:
                x = x[:nchains]  # type: ignore

        assert isinstance(x, Tensor)
        log.warning(f'x[:nchains].shape: {x.shape}')

        if run is not None:
            run.config.update({
                job_type: {'beta': beta, 'xshape': x.shape.as_list()}
            })

        assert x is not None and isinstance(x, Tensor)
        assert beta is not None and isinstance(beta, Tensor)
        ctx = self.get_context_manager(table)
        with ctx:
            for step in range(eval_steps):
                timer.start()
                x, metrics = eval_fn((x, beta))  # type:ignore
                dt = timer.stop()
                if step % nprint == 0 or step % nlog == 0:
                    record = {
                        'step': step, 'beta': beta, 'dt': dt,
                    }
                    avgs, summary = self.record_metrics(run=run,
                                                        arun=arun,
                                                        step=step,
                                                        record=record,
                                                        writer=writer,
                                                        metrics=metrics,
                                                        job_type=job_type)

                    if not isinstance(ctx, Live) and step % nprint == 0:
                        log.info(summary)

                    summaries.append(summary)
                    if step == 0:
                        table = add_columns(avgs, table)
                    else:
                        table.add_row(*[f'{v}' for _, v in avgs.items()])

                    if avgs.get('acc', 1.0) <= 1e-5:
                        self.console.log('Chains are stuck! Re-drawing x !')
                        assert isinstance(x, Tensor)
                        x = self.draw_x()

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
        with tf.GradientTape() as tape:
            tape.watch(xinit)
            xout, metrics = self.dynamics((xinit, beta), training=True)
            # xprop = self.dynamics.g.compat_proj(
            #     metrics.pop('mc_states').proposed.x
            # )
            xprop = metrics.pop('mc_states').proposed.x
            loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
            # xout = self.dynamics.g.compat_proj(xout)
            # xout = to_u1(xout)

            if aw > 0:
                # yinit = to_u1(self.draw_x())
                yinit = self.draw_x()
                _, metrics_ = self.dynamics((yinit, beta), training=True)
                # yprop = to_u1(metrics_.pop('mc_states').proposed.x)
                # yprop = self.dynamics.g.compat_proj(
                #     metrics_.pop('mc_states').proposed.x
                # )
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
            # lmetrics = self.lattice.calc_metrics(
            #     x=xout,
            #     xinit=xinit,
            # )
            # lmetrics = self.lattice.calc_loss(xinit)
            # lmetrics = self.loss_fn.lattice_metrics(xinit=inputs[0], xout=xo)
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
                        'era': era, 'epoch': epoch, 'beta': beta, 'dt': dt,
                    }
                    avgs, summary = self.record_metrics(
                        run=run,
                        arun=arun,
                        step=self._gstep,
                        writer=writer,
                        record=record,    # template w/ step info
                        metrics=metrics,  # metrics from Dynamics
                        job_type='train',
                        model=self.dynamics,
                        optimizer=self.optimizer,
                    )
                    rows[self._gstep] = avgs
                    summaries.append(summary)

                    if not isinstance(ctx, Live) and self.should_print(epoch):
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

        # ┏━------------------------------------━┓
        # ┃         MAIN TRAINING LOOP           ┃
        # ┗-------------------------------------━┛
        era = 0
        extend = 1
        assert x is not None
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

            # losses = edata['losses']
            # if losses[-1] < losses[0]:
            #     b += self.config.annealing_schedule._dbeta
            # else:
            #     b -= self.config.annealing_schedule._dbeta

            if (era + 1) == self.steps.nera or (era + 1) % 5 == 0:
                self.save_ckpt(manager, train_dir)

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

    def metric_to_numpy(
            self,
            metric: TensorLike | list | np.ndarray,
            # key: str = '',
    ) -> np.ndarray:
        """Consistently convert `metric` to np.ndarray."""
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
