"""
trainer.py

Implements methods for training L2HMC sampler
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
from math import pi as PI
import os
from pathlib import Path
import time
from typing import Callable, Any, Optional

import horovod.tensorflow as hvd
import numpy as np
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich import box
import tensorflow as tf

from tensorflow.python.keras import backend as K

from l2hmc.configs import (
    AnnealingSchedule, DynamicsConfig, LearningRateConfig, Steps
)
from l2hmc.dynamics.tensorflow.dynamics import Dynamics, to_u1
from l2hmc.learning_rate.tensorflow.learning_rate import ReduceLROnPlateau
from l2hmc.loss.tensorflow.loss import LatticeLoss
from l2hmc.utils.console import console
from l2hmc.utils.history import summarize_dict
from l2hmc.trackers.tensorflow.trackers import update_summaries
from l2hmc.utils.step_timer import StepTimer
from l2hmc.utils.tensorflow.history import History

# import wandb
# from rich.panel import Panel
# from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

WIDTH = int(os.environ.get('COLUMNS', 150))

tf.autograph.set_verbosity(0)
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

log = logging.getLogger(__name__)

Tensor = tf.Tensor
TF_FLOAT = tf.keras.backend.floatx()
Optimizer = tf.keras.optimizers.Optimizer


def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="main"),
        Layout(name="footer", size=5),
    )
    return layout


def add_columns(avgs: dict, table: Table) -> Table:
    for key in avgs.keys():
        if key == 'loss':
            table.add_column(str(key),
                             justify='center',
                             style='green')
        elif key == 'dt':
            table.add_column(str(key),
                             justify='center',
                             style='red')

        elif key == 'acc':
            table.add_column(str(key),
                             justify='center',
                             style='magenta')
        elif key == 'dQint':
            table.add_column(str(key),
                             justify='center',
                             style='cyan')
        elif key == 'dQsin':
            table.add_column(str(key),
                             justify='center',
                             style='yellow')
        else:
            table.add_column(str(key),
                             justify='center')

    return table


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
            compression: bool = True,
            evals_per_step: int = 1,
            dynamics_config: Optional[DynamicsConfig] = None,
    ) -> None:
        self.rank = rank
        self.steps = steps
        self.dynamics = dynamics
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.schedule = schedule
        self.aux_weight = aux_weight
        self.keep = [keep] if isinstance(keep, str) else keep
        self.skip = [skip] if isinstance(skip, str) else skip
        if compression:
            self.compression = hvd.Compression.fp16
        else:
            self.compression = hvd.Compression.none

        self.reduce_lr = ReduceLROnPlateau(lr_config)
        self.dynamics.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.reduce_lr.set_model(self.dynamics)
        self.reduce_lr.set_optimizer(self.optimizer)
        self.dynamics_config = (
            dynamics_config if dynamics_config is not None
            else self.dynamics.config
        )
        self.history = History(steps=steps)
        evals_per_step = self.dynamics.config.nleapfrog
        self.timer = StepTimer(evals_per_step=evals_per_step)
        self.eval_timer = StepTimer(evals_per_step=evals_per_step)
        self.histories = {
            'train': self.history,
            'eval': History(),
            'hmc': History(),
        }
        self.timers = {
            'train': self.timer,
            'eval': StepTimer(evals_per_step=evals_per_step),
            'hmc': StepTimer(evals_per_step=evals_per_step),
        }
        # self.eval_history = History()
        # evals_per_step = self.dynamics.config.nleapfrog * steps.log

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

    def draw_x(self) -> Tensor:
        """Draw `x` """
        x = tf.random.uniform(self.dynamics.xshape,
                              *(-PI, PI), dtype=TF_FLOAT)
        x = tf.reshape(x, (x.shape[0], -1))

        return x

    # type: ignore
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

    def record_metrics(
            self,
            metrics: dict,
            job_type: str,
            step: Optional[int] = None,
            record: Optional[dict] = None,
            run: Optional[Any] = None,
            writer: Optional[Any] = None,
            history: Optional[History] = None,
    ):
        record = {} if record is None else record
        record.update({
            'loss': metrics.get('loss', None),
            'dQint': metrics.get('dQint', None),
            'dQsin': metrics.get('dQsin', None),
        })
        record.update(self.metrics_to_numpy(metrics))
        if history is not None:
            avgs = history.update(record)
        else:
            avgs = {k: v.mean() for k, v in record.items()}

        summary = summarize_dict(avgs)
        if writer is not None:
            assert step is not None
            update_summaries(step=step, prefix=job_type, metrics=record)
            writer.flush()
        if run is not None:
            run.log({f'wandb/{job_type}': record}, commit=False)
            run.log({f'avgs/wandb.{job_type}': avgs})

        return avgs, summary

    @tf.function(experimental_follow_type_hints=True)
    def hmc_step(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, dict]:
        xi, beta = inputs
        inputs = (to_u1(xi), tf.constant(beta))
        xo, metrics = self.dynamics.apply_transition_hmc(inputs)
        xo = to_u1(xo)
        xp = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xi, x_prop=xp, acc=metrics['acc'])
        lmetrics = self.loss_fn.lattice_metrics(xinit=xi, xout=xo)
        metrics.update(lmetrics)
        metrics.update({'loss': loss})

        return xo, metrics

    @tf.function(experimental_follow_type_hints=True)
    def eval_step(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, dict]:
        xi, beta = inputs
        inputs = (to_u1(xi), tf.constant(beta))
        xo, metrics = self.dynamics(inputs, training=False)
        xo = to_u1(xo)
        xp = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xi, x_prop=xp, acc=metrics['acc'])
        lmetrics = self.loss_fn.lattice_metrics(xinit=xi, xout=xo)
        metrics.update(lmetrics)
        metrics.update({'loss': loss})

        return xo, metrics

    def eval(
            self,
            beta: Optional[float] = None,
            x: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            run: Optional[Any] = None,
            writer: Optional[Any] = None,
            job_type: Optional[str] = 'eval',
            nchains: Optional[int] = None,
    ) -> dict:
        """Evaluate model."""
        if isinstance(skip, str):
            skip = [skip]

        if beta is None:
            beta = self.schedule.beta_final

        assert job_type in ['eval', 'hmc']

        if x is None:
            unif = tf.random.uniform(self.dynamics.xshape,
                                     *(-PI, PI), dtype=TF_FLOAT)
            x = tf.reshape(unif, (unif.shape[0], -1))
        else:
            x = tf.Variable(x, dtype=TF_FLOAT)

        if writer is not None:
            writer.set_as_default()

        if job_type == 'hmc':
            eval_fn = self.hmc_step
        else:
            eval_fn = self.eval_step

        assert isinstance(x, Tensor) and x.dtype == TF_FLOAT

        tables = {}
        rows = {}
        summaries = []
        table = Table(row_styles=['dim', 'none'], box=box.SIMPLE)
        nprint = max((20, self.steps.test // 20))
        nlog = max((1, min((10, self.steps.test))))
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

        with Live(table, screen=False, auto_refresh=False) as live:
            if WIDTH is not None and WIDTH > 0:
                live.console.width = WIDTH

            for step in range(self.steps.test):
                timer.start()
                x, metrics = eval_fn((x, beta))  # type: ignore
                dt = timer.stop()
                if step % nprint == 0 or step % nlog == 0:
                    record = {
                        'step': step, 'beta': beta, 'dt': dt,
                    }
                    avgs, summary = self.record_metrics(run=run,
                                                        step=step,
                                                        record=record,
                                                        writer=writer,
                                                        metrics=metrics,
                                                        history=history,
                                                        job_type=job_type)
                    rows[step] = avgs
                    summaries.append(summary)
                    if avgs.get('acc', 1.0) <= 1e-5:
                        log.warning('Chains are stuck! Re-drawing x !')
                        x = self.draw_x()

                    if step == 0:
                        table = add_columns(avgs, table)

                    if step % nprint == 0:
                        table.add_row(*[f'{v:5}' for _, v in avgs.items()])
                        live.refresh()

            tables[str(0)] = table

        return {
            'timer': timer,
            'history': history,
            'rows': rows,
            'summaries': summaries,
            'tables': tables,
        }

    @tf.function(experimental_follow_type_hints=True)
    def train_step(
            self,
            inputs: tuple[Tensor, Tensor],
            first_step: Optional[bool] = None,
    ) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        xinit = to_u1(xinit)
        # beta = tf.constant(beta)
        record = {'loss': None}
        with tf.GradientTape() as tape:
            tape.watch(xinit)
            xout, metrics = self.dynamics((xinit, beta), training=True)
            xprop = to_u1(metrics.pop('mc_states').proposed.x)
            loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])

            if self.aux_weight > 0:
                yinit = to_u1(self.draw_x())
                _, metrics_ = self.dynamics((yinit, beta), training=True)
                yprop = to_u1(metrics_.pop('mc_states').proposed.x)
                aux_loss = self.aux_weight * self.loss_fn(x_init=yinit,
                                                          x_prop=yprop,
                                                          acc=metrics_['acc'])
                loss = (loss + aux_loss) / (1. + self.aux_weight)

            record['loss'] = loss
            record.update(metrics)

        tape = hvd.DistributedGradientTape(tape, compression=self.compression)
        grads = tape.gradient(loss, self.dynamics.trainable_variables)
        updates = zip(grads, self.dynamics.trainable_variables)
        self.optimizer.apply_gradients(updates)
        if first_step:
            hvd.broadcast_variables(self.dynamics.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        # bconst = tf.cast(beta, dtype=TF_FLOAT)
        xout = to_u1(xout)
        lmetrics = self.loss_fn.lattice_metrics(xinit=xinit, xout=xout)
        record.update(lmetrics)

        return to_u1(xout), record

    def train(
            self,
            xinit: Optional[Tensor] = None,
            skip: Optional[str | list[str]] = None,
            train_dir: Optional[os.PathLike] = None,
            run: Optional[Any] = None,
            writer: Optional[Any] = None,
            # compile: bool = True,
            # jit_compile: bool = False,
            # save_x: bool = False,
    ) -> dict:
        """Train l2hmc Dynamics."""
        if isinstance(skip, str):
            skip = [skip]

        if writer is not None:
            writer.set_as_default()

        if train_dir is None:
            train_dir = Path(os.getcwd()).joinpath('train')

        manager = self.setup_CheckpointManager(train_dir)
        gstep = K.get_value(self.optimizer.iterations)

        x = self.draw_x() if xinit is None else tf.constant(xinit, TF_FLOAT)
        assert isinstance(x, Tensor) and x.dtype == TF_FLOAT

        inputs = (x, tf.constant(1.))
        _ = self.train_step(inputs, first_step=True)  # type: ignore

        era = 0
        epoch = 0
        tables = {}
        rows = {}
        metrics = {}
        summaries = []
        timer = self.timers['train']
        history = self.histories['train']
        tkwargs = {
            'box': box.SIMPLE,
            'row_styles': ['dim', 'none'],
        }
        for era in range(self.steps.nera):
            table = Table(**tkwargs)
            beta = tf.constant(self.schedule.betas[str(era)])
            if self.rank == 0:
                console.width = WIDTH
                console.rule(
                    f'ERA: {era} / {self.steps.nera}, BETA: {beta}',
                    style='black on white'
                )

            with Live(
                table,
                screen=False,
                console=console,
                # refresh_per_second=1,
            ) as live:
                if WIDTH is not None and WIDTH > 0:
                    live.console.width = WIDTH
                estart = time.time()
                for epoch in range(self.steps.nepoch):
                    timer.start()
                    x, metrics = self.train_step(  # type: ignore
                        (x, beta),
                        first_step=False
                    )
                    dt = timer.stop()
                    gstep += 1
                    if self.should_print(epoch) or self.should_log(epoch):
                        record = {
                            'era': era, 'epoch': epoch, 'beta': beta, 'dt': dt,
                        }
                        avgs, summary = self.record_metrics(run=run,
                                                            step=gstep,
                                                            writer=writer,
                                                            record=record,
                                                            metrics=metrics,
                                                            job_type='train',
                                                            history=history)
                        rows[gstep] = avgs
                        summaries.append(summary)

                        if avgs.get('acc', 1.0) <= 1e-5:
                            log.warning('Chains are stuck! Re-drawing x !')
                            x = self.draw_x()

                        if epoch == 0:
                            table = add_columns(avgs, table)

                        if self.should_print(epoch):
                            table.add_row(*[f'{v}' for _, v in avgs.items()])

            tables[str(era)] = table
            self.reduce_lr.on_epoch_end((era + 1) * self.steps.nepoch, {
                'loss': metrics.get('loss', tf.constant(np.Inf)),
            })
            if self.rank == 0:
                if writer is not None:
                    update_summaries(step=gstep,
                                     model=self.dynamics,
                                     optimizer=self.optimizer)
                log.info(
                    f'Era {era} took: {time.time() - estart:<5g}s'
                )
                log.info(
                    f'Avgs over last era:\n {self.history.era_summary(era)}',
                )
                log.info(
                    f'Saving checkpoint to: {manager.latest_checkpoint}'
                )
                manager.save()
                self.dynamics.save_networks(train_dir)

        return {
            'timer': timer,
            'rows': rows,
            'summaries': summaries,
            'history': history,
            'tables': tables,
        }

    """
    def train_interactive(
            self,
            xinit: Tensor = None,
            beta: float = 1.,
            compile: bool = True,
            jit_compile: bool = False,
            save_x: bool = False,
    ) -> dict:
        if xinit is None:
            x = tf.random.uniform(self.dynamics.xshape,
                                  *(-PI, PI), dtype=TF_FLOAT)
            x = tf.reshape(x, (x.shape[0], -1))
        else:
            x = tf.constant(xinit, dtype=TF_FLOAT)

        assert isinstance(x, Tensor) and x.dtype == TF_FLOAT

        if compile:
            self.dynamics.compile(optimizer=self.optimizer, loss=self.loss_fn)
            train_step = tf.function(self.train_step, jit_compile=jit_compile)
        else:
            train_step = self.train_step

        era = 0
        nera = self.steps.nera
        nepoch = self.steps.nepoch
        should_log = lambda epoch: (epoch % self.steps.log == 0)  # noqa
        should_print = lambda epoch: (epoch % self.steps.print == 0)  # noqa

        xarr = []
        table = None
        tables = {}
        summaries = []
        job_progress = Progress(
            "{task.description}",
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        era_task = job_progress.add_task("[blue]Era", total=nera)
        epoch_task = job_progress.add_task("[green]Epoch", total=nepoch)
        colors = 10 * ['red', 'yellow', 'green', 'blue', 'magenta', 'cyan']
        progress_table = Table.grid()
        progress_table.add_row(Panel.fit(job_progress))  # type: ignore
        layout = make_layout()
        layout['footer'].update(progress_table)
        with Live(layout, screen=False, auto_refresh=True):
            for era in range(self.steps.nera):
                table = Table(collapse_padding=True,
                              row_styles=['dim', 'none'])
                layout['root']['main'].update(table)
                # xdict[str(era)] = x
                job_progress.reset(epoch_task)
                estart = time.time()
                for epoch in range(self.steps.nepoch):
                    self.timer.start()
                    x, metrics = train_step((x, beta))  # type: ignore
                    dt = self.timer.stop()
                    if should_log(epoch) or should_print(epoch):
                        record = {'era': era, 'epoch': epoch, 'dt': dt}
                        if save_x:
                            xarr.append(x)
                        # Update metrics with train step metrics, tmetrics
                        record.update(self.metrics_to_numpy(metrics))
                        avgs = self.history.update(record)
                        summary = summarize_dict(avgs)
                        summaries.append(summary)
                        if epoch == 0:
                            for idx, key in enumerate(avgs.keys()):
                                table.add_column(str(key).upper(),
                                                 style=colors[idx],
                                                 justify='center')

                        if should_print(epoch):
                            table.add_row(*[f'{v:5}' for _, v in avgs.items()])
                        # if should_print(epoch):
                        #     layout['base'].update(row)

                        # row = list(map(str, avgs.values()))
                        # table.add_row(*row)
                        # data_table.add_row(*list(avgs.values()))

                    job_progress.advance(epoch_task)

                job_progress.advance(era_task)

                # live.console.rule()
                log.info(f'Era {era} took: {time.time() - estart:<3.2g}s')
                esumm = self.history.era_summary(era)
                log.info(f'Avgs over last era: {esumm}')
                # live.refresh()

            tables[str(era)] = table

        return {
            'xdict': xarr,
            'summaries': summaries,
            'history': self.history,
            'tables': tables,
        }
    """
