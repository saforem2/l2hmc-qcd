"""
trainer.py
Implements methods for training L2HMC sampler
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from pathlib import Path
# import wandb
import time
from typing import Callable, Any, Optional

import tensorflow as tf
import tensorflow.python.framework.ops as ops
import horovod.tensorflow as hvd  # type: ignore

import aim
from aim import Distribution

import numpy as np
# from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich import box

from tensorflow.python.keras import backend as K

from l2hmc.configs import (
    AnnealingSchedule, DynamicsConfig, LearningRateConfig, Steps
)
from l2hmc.group.u1.tensorflow.group import U1Phase
from l2hmc.group.su3.tensorflow.group import SU3
from l2hmc.learning_rate.tensorflow.learning_rate import ReduceLROnPlateau
from l2hmc.loss.tensorflow.loss import LatticeLoss
from l2hmc.utils.history import summarize_dict
from l2hmc.trackers.tensorflow.trackers import update_summaries
from l2hmc.utils.step_timer import StepTimer
from l2hmc.dynamics.tensorflow.dynamics import Dynamics
# from l2hmc.utils.tensorflow.history import History
from l2hmc.utils.history import BaseHistory
from l2hmc.utils.rich import add_columns, get_console
from contextlib import nullcontext

# from rich.layout import Layout
# from rich.columns import  SpinnerColumn, BarColumn, TextColumn
# SpinnerColumn(),
# BarColumn(),
# TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),


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


# TODO: Replace arguments in __init__ call below with configs.TrainerConfig
class Trainer:
    def __init__(
            self,
            steps: Steps,
            dynamics: Dynamics,
            optimizer: Optimizer,
            schedule: AnnealingSchedule,
            lr_config: LearningRateConfig,
            rank: int = 0,
            loss_fn: Callable = LatticeLoss,
            aux_weight: float = 0.0,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
            compression: Optional[str] = 'none',
            evals_per_step: int = 1,
            dynamics_config: Optional[DynamicsConfig] = None,
            # compile: Optional[bool] = True,
    ) -> None:
        self.rank = rank
        self.steps = steps
        self.loss_fn = loss_fn
        self.dynamics = dynamics
        self.schedule = schedule
        self.lr_config = lr_config
        self.optimizer = optimizer
        self.aux_weight = aux_weight
        self.clip_norm = lr_config.clip_norm
        self.keep = [keep] if isinstance(keep, str) else keep
        self.skip = [skip] if isinstance(skip, str) else skip
        assert compression in ['none', 'fp16']
        self.compression = HVD_FP_MAP[compression]
        # self.compression = hvd.Compression.fp16
        self.reduce_lr = ReduceLROnPlateau(lr_config)
        # if compile:
        #     self.dynamics.compile(
        #             optimizer=self.optimizer,
        #             loss=self.loss_fn
        #     )

        self.reduce_lr.set_model(self.dynamics)
        self.reduce_lr.set_optimizer(self.optimizer)

        self.dynamics_config = (
            dynamics_config if dynamics_config is not None
            else self.dynamics.config
        )
        self.nlf = self.dynamics_config.nleapfrog
        self.xshape = self.dynamics_config.xshape
        self.verbose = self.dynamics_config.verbose
        if self.dynamics_config.group == 'U1':
            self.g = U1Phase()
        elif self.dynamics_config.group == 'SU3':
            self.g = SU3()
        else:
            raise ValueError('Unexpected value for dynamics_config.group')

        self.history = BaseHistory(steps=steps)
        self.timer = StepTimer(evals_per_step=self.nlf)
        self.histories = {
            'train': self.history,
            'eval': BaseHistory(),
            'hmc': BaseHistory()
        }
        self.timers = {
            'train': self.timer,
            'eval': StepTimer(evals_per_step=evals_per_step),
            'hmc': StepTimer(evals_per_step=evals_per_step)
        }
        self.console = get_console(record=False)

    def draw_x(self, shape: Optional[list[int]] = None) -> Tensor:
        """Draw `x` """
        shape = list(self.dynamics.xshape) if shape is None else shape
        x = self.dynamics.g.random(shape)
        x = tf.reshape(x, (x.shape[0], -1))

        return x

    def reset_optimizer(self) -> None:
        if len(self.optimizer.variables()) > 0:
            log.warning('Resetting optimizer state!')
            for var in self.optimizer.variables():
                var.assign(tf.zeros_like(var))

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

    def metric_to_numpy(
            self,
            metric: Tensor | list | np.ndarray,
            # key: str = '',
    ) -> np.ndarray:
        """Consistently convert `metric` to np.ndarray."""
        if isinstance(metric, np.ndarray):
            return metric

        if (
                isinstance(metric, Tensor)
                and hasattr(metric, 'numpy')
                and isinstance(metric.numpy, Callable)
        ):
            return metric.numpy()

        elif isinstance(metric, list):
            if isinstance(metric[0], np.ndarray):
                return np.stack(metric)

            if isinstance(metric[0], Tensor):
                stack = tf.stack(metric)
                if (
                        hasattr(stack, 'numpy')
                        and isinstance(stack.numpy, Callable)
                ):
                    return stack.numpy()
            else:
                return np.array(metric)

            return np.array(metric)

        else:
            raise ValueError(
                f'Unexpected type for metric: {type(metric)}'
            )

    def metrics_to_numpy(
            self,
            metrics: dict[str, Tensor | list | np.ndarray]
    ) -> dict:
        m = {}
        for key, val in metrics.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    m[f'{key}/{k}'] = self.metric_to_numpy(v)
            else:
                try:
                    m[key] = self.metric_to_numpy(val)
                except (ValueError, tf.errors.InvalidArgumentError):
                    log.warning(
                        f'Error converting metrics[{key}] to numpy. Skipping!'
                    )
                    continue

        return m

    def should_log(self, epoch):
        return (
            epoch % self.steps.log == 0
            and self.rank == 0
        )

    def should_print(self, epoch):
        return (
            epoch % self.steps.print == 0
            and self.rank == 0
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
                is_dist = (
                    len(val.shape) > 1
                    or (
                        len(val.shape) == 1
                        and val.shape[0] > 1
                    )
                )
                if is_dist:
                    dist = Distribution(val)
                    arun.track(dist, step=step, name=name, context=context)
                    aname = f'{name}/avg'
                    avg = (
                        tf.reduce_mean(val) if isinstance(val, Tensor)
                        else val.mean()
                    )
                    arun.track(avg, step=step, name=aname, context=context)
                else:
                    arun.track(val, step=step, name=name, context=context)

            else:
                arun.track(val,
                           name=name,
                           # step=step,
                           context=context)

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
        kwargs = {'step': step, 'job_type': job_type, 'arun': arun}
        history = self.histories[job_type]

        if step is not None:
            record.update({f'{job_type}_step': step})

        record.update({
            'loss': metrics.get('loss', None),
            'dQint': metrics.get('dQint', None),
            'dQsin': metrics.get('dQsin', None),
        })
        if job_type == 'train' and step is not None:
            record['lr'] = K.get_value(self.optimizer.lr)

        record.update(self.metrics_to_numpy(metrics))
        avgs = self.histories[job_type].update(record)
        summary = summarize_dict(avgs)
        # if step is not None:
        if writer is not None and self.verbose and step is not None:
            update_summaries(step=step,
                             prefix=job_type,
                             model=model,
                             metrics=record,
                             optimizer=optimizer)
            if writer is not None:
                writer.flush()

            if run is not None:
                run.log({f'wandb/{job_type}': record}, commit=False)
                run.log({f'avgs/wandb.{job_type}': avgs}, commit=False)
            if arun is not None:
                try:
                    self.aim_track(avgs, prefix='avgs', **kwargs)
                except Exception:
                    log.warning('Unable to aim_track `avgs` !')
                try:
                    self.aim_track(record, prefix='record', **kwargs)
                except Exception:
                    log.warning('Unable to aim_track `record` !')
                # try:
                #     self.aim_track(metrics, prefix='metrics', **kwargs)
                # except Exception:
                #     log.warning('Unable to aim_track `metrics` !')

        dQint = record.get('dQint', None)
        if dQint is not None:
            dQdict = {
                f'dQint/{job_type}': {
                    'val': dQint,
                    'step': step,
                    'avg': dQint.mean(),
                }
            }
            if run is not None:
                run.log(dQdict)
            if arun is not None:
                self.aim_track({'dQint': dQint}, **kwargs)

        return avgs, summary

    # @tf.function(experimental_follow_type_hints=True,
    #         jit_compile=JIT_COMPILE)
    @tf.function
    def hmc_step(
            self,
            inputs: tuple[Tensor, Tensor],
            eps: float,
            nleapfrog: Optional[int] = None,
    ) -> tuple[Tensor, dict]:
        # xi, beta = inputs
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

    def eval(
            self,
            beta: Optional[Tensor | float] = None,
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
            beta = tf.constant(self.schedule.beta_final, dtype=TF_FLOAT)

        if eps is None and str(job_type).lower() == 'hmc':
            # eps = tf.constant(0.1, dtype=TF_FLOAT)
            eps = tf.constant(self.dynamics.config.eps_hmc, dtype=TF_FLOAT)
            log.warn(
                'Step size `eps` not specified for HMC! '
                f'Using default: {self.dynamics.config.eps_hmc:.3f}'
            )

        assert job_type in ['eval', 'hmc']

        if x is None:
            r = self.dynamics.g.random(list(self.xshape))
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
        nprint = max(1, self.steps.test // 20)
        nlog = max((1, min((10, self.steps.test))))
        if nlog <= self.steps.test:
            nlog = min(10, max(1, self.steps.test // 100))

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

        # with Live(table) as live:
        assert x is not None and isinstance(x, Tensor)
        assert beta is not None and isinstance(beta, Tensor)
        with Live(
                table,
                console=self.console,
                vertical_overflow='visible',
        ) as live:
            for step in range(self.steps.test):
                timer.start()
                x, metrics = eval_fn((x, beta))
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
                    summaries.append(summary)
                    if step == 0:
                        table = add_columns(avgs, table)
                    else:
                        table.add_row(*[f'{v}' for _, v in avgs.items()])

                    if avgs.get('acc', 1.0) <= 1e-5:
                        live.console.log('Chains are stuck! Re-drawing x !')
                        assert isinstance(x, Tensor)
                        x = self.g.random(list(x.shape))

        tables[str(0)] = table

        return {
            'timer': timer,
            'history': history,
            'summaries': summaries,
            'tables': tables,
        }

    # @tf.function(experimental_follow_type_hints=True,
    #         jit_compile=JIT_COMPILE)
    @tf.function(
        experimental_follow_type_hints=True,
        # input_signature=[
        #     tf.TensorSpec(shape=None, dtype=TF_FLOAT),
        #     tf.TensorSpec(shape=None, dtype=TF_FLOAT),
        # ],
    )
    def train_step(
            self,
            inputs: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        # xinit = self.dynamics.g.compat_proj(xinit)
        # should_clip = (
        #     (self.lr_config.clip_norm > 0)
        #     if clip_grads is None else clip_grads
        # )
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

            if self.aux_weight > 0:
                # yinit = to_u1(self.draw_x())
                yinit = self.draw_x()
                _, metrics_ = self.dynamics((yinit, beta), training=True)
                # yprop = to_u1(metrics_.pop('mc_states').proposed.x)
                # yprop = self.dynamics.g.compat_proj(
                #     metrics_.pop('mc_states').proposed.x
                # )
                yprop = metrics_.pop('mc_states').proposed.x
                aux_loss = self.aux_weight * self.loss_fn(x_init=yinit,
                                                          x_prop=yprop,
                                                          acc=metrics_['acc'])
                loss += self.aux_weight * aux_loss
                # loss = (loss + aux_loss) / (1. + self.aux_weight)

        tape = hvd.DistributedGradientTape(tape, compression=self.compression)
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

    @staticmethod
    def update_record(
        record: dict,
        updates: dict[str, float | int | Tensor | Array | None],
        reset: Optional[bool] = False,
    ) -> dict:
        if reset:
            record = {}

        for key, val in updates.items():
            if val is not None:
                record[key] = val

        return record

    def train_epoch(
            self,
            beta: float | Tensor,
            x: Optional[Tensor] = None,
            step: Optional[int] = None,
            era: Optional[int] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
    ) -> dict:
        if x is None:
            x = self.draw_x()
        if isinstance(beta, float):
            beta = tf.constant(beta)

        gstep = 0
        avgs = {}
        summaries = []
        table = Table(expand=True)
        beta = tf.cast(beta, TF_FLOAT)
        record = {}
        for step in range(self.steps.nepoch):
            self.timers['train'].start()
            x, metrics = self.train_step((x, beta))  # type: ignore
            dt = self.timers['train'].stop()
            gstep += 1

            if self.should_print(step) or self.should_log(step):
                record = self.update_record(
                    record=record,
                    updates=(
                        {'era': era, 'epoch': step, 'beta': beta, 'dt': dt}
                    )
                )
                avgs_, summary = self.record_metrics(
                    run=run,
                    arun=arun,
                    step=gstep,
                    writer=writer,
                    record=record,      # template w/ step info, np.arr
                    metrics=metrics,    # metrics from Dynamics, Tensor
                    job_type='train',
                    model=self.dynamics,
                    optimizer=self.optimizer,
                )
                avgs[gstep] = avgs_
                summaries.append(summary)

                if step == 0:
                    table = add_columns(avgs_, table)
                else:
                    table.add_row(*[f'{v}' for _, v in avgs_.items()])

                if avgs_.get('acc', 1.0) <= 1e-5:
                    self.reset_optimizer()
                    log.warning('Chains are stuck! Re-drawing x !')
                    x = self.draw_x()

        return {'avgs': avgs, 'summaries': summaries}

    def train(
            self,
            xinit: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            train_dir: Optional[os.PathLike] = None,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None
    ) -> dict:
        """Train l2hmc Dynamics."""
        if isinstance(skip, str):
            skip = [skip]

        if writer is not None:
            writer.set_as_default()

        if train_dir is None:
            train_dir = Path(os.getcwd()).joinpath('train')

        try:
            model_dir = Path(os.getcwd()).joinpath('plots', 'models')
            model_dir.mkdir(exist_ok=True, parents=True)
            plot_models(self.dynamics, train_dir)
        except Exception:
            pass

        manager = self.setup_CheckpointManager(train_dir)
        gstep = K.get_value(self.optimizer.iterations)

        x = self.draw_x() if xinit is None else tf.constant(xinit, TF_FLOAT)
        assert isinstance(x, Tensor) and x.dtype == TF_FLOAT

        # inputs = (x, tf.constant(self.schedule.beta_init))
        # assert callable(self.train_step)
        # _ = self.train_step(inputs)

        era = 0
        epoch = 0
        rows = {}
        tables = {}
        metrics = {}
        summaries = []
        table = Table(expand=True)
        nepoch = self.steps.nepoch
        timer = self.timers['train']
        history = self.histories['train']
        extend = self.steps.extend_last_era
        nepoch_last_era = self.steps.nepoch
        record = {'era': 0, 'epoch': 0, 'beta': 0.0, 'dt': 0.0}
        if extend is not None and isinstance(extend, int) and extend > 1:
            nepoch_last_era *= extend

        for era in range(self.steps.nera):
            beta = tf.constant(self.schedule.betas[str(era)], dtype=TF_FLOAT)
            table = Table(
                box=box.HORIZONTALS,
                row_styles=['dim', 'none'],
            )

            if self.rank == 0:
                ctxmgr = Live(table,
                              console=self.console,
                              vertical_overflow='visible')
            else:
                ctxmgr = nullcontext()

            if era == self.steps.nera - 1:
                nepoch = nepoch_last_era

            with ctxmgr as live:
                if live is not None:
                    tstr = ' '.join([
                        f'ERA: {era}/{self.steps.nera}',
                        f'BETA: {beta.numpy():.3f}',
                    ])
                    live.console.clear_live()
                    live.console.rule(tstr)
                    live.update(table, refresh=True)

                epoch_start = time.time()
                for epoch in range(nepoch):
                    timer.start()
                    x, metrics = self.train_step(   # type:ignore
                        (x, beta)
                    )
                    dt = timer.stop()
                    gstep += 1
                    if self.should_print(epoch) or self.should_log(epoch):
                        record = {
                            'era': era, 'epoch': epoch, 'beta': beta, 'dt': dt,
                        }
                        avgs, summary = self.record_metrics(
                            run=run,
                            arun=arun,
                            step=gstep,
                            writer=writer,
                            record=record,      # template w/ step info, np.arr
                            metrics=metrics,    # metrics from Dynamics, Tensor
                            job_type='train',
                            model=self.dynamics,
                            optimizer=self.optimizer,
                        )
                        rows[gstep] = avgs
                        summaries.append(summary)

                        if epoch == 0:
                            table = add_columns(avgs, table)
                        else:
                            table.add_row(*[f'{v}' for _, v in avgs.items()])

                        if avgs.get('acc', 1.0) < 1e-5:
                            self.reset_optimizer()
                            log.warning('Chains are stuck! Re-drawing x !')
                            x = self.g.random(list(x.shape))

                tables[str(era)] = table
                self.reduce_lr.on_epoch_end((era + 1) * self.steps.nepoch, {
                    'loss': metrics.get('loss', tf.constant(np.Inf)),
                })
                if self.rank == 0:
                    # if writer is not None:
                    #     update_summaries(step=gstep,
                    #                      run=self.run,
                    #                      arun=self.arun,
                    #                      model=self.dynamics,
                    #                      optimizer=self.optimizer)
                    st0 = time.time()
                    manager.save()
                    if (era + 1) == self.steps.nera or (era + 1) % 5 == 0:
                        self.dynamics.save_networks(train_dir)

                    # ckptstr = '\n'.join([
                    if live is not None:
                        dts = time.time() - st0
                        savestr = ' '.join([
                            'Checkpoint saved to:',
                            f'{manager.latest_checkpoint}',
                            f'in {dts:<5f}s'
                        ])
                        estr = ' '.join([
                            f'Era {era} took:',
                            f'{time.time() - epoch_start:4f}s'
                        ])
                        ckptstr = '\n'.join([savestr, estr])
                        live.console.log(ckptstr)

        return {
            'timer': timer,
            'rows': rows,
            'summaries': summaries,
            'history': history,
            'tables': tables,
        }
