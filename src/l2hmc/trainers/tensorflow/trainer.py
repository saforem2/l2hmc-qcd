"""
trainer.py

Implements methods for training L2HMC sampler
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
import time
from typing import Callable

import horovod.tensorflow as hvd
import numpy as np
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
import tensorflow as tf

from tensorflow.keras.optimizers import Optimizer
from l2hmc.utils.hvd_init import RANK
from l2hmc.configs import AnnealingSchedule, Steps
from l2hmc.dynamics.tensorflow.dynamics import Dynamics, to_u1
from l2hmc.loss.tensorflow.loss import LatticeLoss
from l2hmc.utils.console import console
from l2hmc.utils.history import summarize_dict
from l2hmc.utils.step_timer import StepTimer
from l2hmc.utils.tensorflow.history import History
tf.autograph.set_verbosity(0)
os.environ['AUTOGRAPH_VERBOSITY'] = '0'


log = logging.getLogger(__name__)

TF_FLOAT = tf.keras.backend.floatx()
Tensor = tf.Tensor


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
        else:
            table.add_column(str(key),
                             justify='center')

    return table


class Trainer:
    def __init__(
            self,
            steps: Steps,
            dynamics: Dynamics,
            optimizer: Optimizer,
            schedule: AnnealingSchedule,
            loss_fn: Callable = LatticeLoss,
            aux_weight: float = 0.0,
            keep: str | list[str] = None,
            skip: str | list[str] = None,
            compression: bool = True,
            evals_per_step: int = 1,
    ) -> None:
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

        self.history = History(steps=steps)
        self.eval_history = History()
        # evals_per_step = self.dynamics.config.nleapfrog * steps.log
        self.timer = StepTimer(evals_per_step=evals_per_step)

    def draw_x(self) -> Tensor:
        x = tf.random.uniform(self.dynamics.xshape,
                              *(-np.pi, np.pi), dtype=TF_FLOAT)
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

    def eval_step(self, inputs: tuple[Tensor, float]) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        xout, metrics = self.dynamics((to_u1(xinit), tf.constant(beta)))
        xprop = to_u1(metrics.pop('mc_states').proposed.x)
        loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])
        metrics.update({'loss': loss})

        return to_u1(xout), metrics

    def eval(
            self,
            beta: float = None,
            x: Tensor = None,
            skip: str | list[str] = None,
            compile: bool = True,
            jit_compile: bool = False,
            width: int = 150,
    ) -> dict:
        """Evaluate model."""
        if isinstance(skip, str):
            skip = [skip]

        if beta is None:
            beta = self.schedule.beta_final

        if x is None:
            unif = tf.random.uniform(self.dynamics.xshape,
                                     *(-np.pi, np.pi), dtype=TF_FLOAT)
            x = tf.reshape(unif, (unif.shape[0], -1))
        else:
            x = tf.constant(x, dtype=TF_FLOAT)

        assert isinstance(x, Tensor) and x.dtype == TF_FLOAT

        if compile:
            self.dynamics.compile(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                experimental_run_tf_function=False,
            )
            eval_step = tf.function(self.eval_step, jit_compile=jit_compile)
        else:
            eval_step = self.eval_step

        xarr = []
        tables = {}
        summaries = []
        table = Table(collapse_padding=True, row_styles=['dim', 'none'])
        # console = get_console(width=width)
        with Live(table, console=console, screen=False) as live:
            if width is not None and width > 0:
                live.console.width = width

            for step in range(self.steps.test):
                self.timer.start()
                x, metrics = eval_step((x, beta))  # type: ignore
                dt = self.timer.stop()
                xarr.append(x)
                loss = metrics.pop('loss').numpy()
                record = {'step': step, 'dt': dt, 'loss': loss}
                record.update(self.metrics_to_numpy(metrics))
                avgs = self.eval_history.update(record)
                summary = summarize_dict(avgs)
                summaries.append(summary)
                if step == 0:
                    table = add_columns(avgs, table)

                if step % self.steps.print == 0:
                    table.add_row(*[f'{v:5}' for _, v in avgs.items()])

            tables[str(0)] = table

        return {
            'xarr': xarr,
            'history': self.eval_history,
            'summaries': summaries,
            'tables': tables,
        }

    def train_step(
            self,
            inputs: tuple[Tensor, float],
            first_step: bool = False
    ) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        with tf.GradientTape() as tape:
            x_out, metrics = self.dynamics((to_u1(xinit), beta))
            xprop = to_u1(metrics.pop('mc_states').proposed.x)
            loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])

            if self.aux_weight > 0:
                yinit = to_u1(self.draw_x())
                _, metrics_ = self.dynamics((yinit, beta))
                yprop = to_u1(metrics_.pop('mc_states').proposed.x)
                aux_loss = self.aux_weight * self.loss_fn(x_init=yinit,
                                                          x_prop=yprop,
                                                          acc=metrics_['acc'])
                loss = (loss + aux_loss) / (1. + self.aux_weight)

        tape = hvd.DistributedGradientTape(tape, compression=self.compression)
        grads = tape.gradient(loss, self.dynamics.trainable_variables)
        updates = zip(grads, self.dynamics.trainable_variables)
        self.optimizer.apply_gradients(updates)
        if first_step:
            hvd.broadcast_variables(self.dynamics.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        record = {
            'loss': loss,
        }
        for key, val in metrics.items():
            record[key] = val

        return to_u1(x_out), record

    def train(
            self,
            xinit: Tensor = None,
            skip: str | list[str] = None,
            compile: bool = True,
            jit_compile: bool = False,
            save_x: bool = False,
            width: int = 150,
    ) -> dict:
        """Train l2hmc Dynamics."""
        summaries = []
        if isinstance(skip, str):
            skip = [skip]
        if xinit is None:
            x = self.draw_x()
        else:
            x = tf.constant(xinit, dtype=TF_FLOAT)

        assert isinstance(x, Tensor) and x.dtype == TF_FLOAT

        if compile:
            self.dynamics.compile(optimizer=self.optimizer, loss=self.loss_fn)
            train_step = tf.function(self.train_step, jit_compile=jit_compile)
        else:
            train_step = self.train_step

        # nera = self.steps.nera
        # nepoch = self.steps.nepoch
        def should_log(epoch):
            return epoch % self.steps.log == 0 and RANK == 0

        def should_print(epoch):
            return epoch % self.steps.print == 0 and RANK == 0

        xarr = []
        tables = {}
        summaries = []
        # screen = (not is_interactive())
        # console = get_console(width=width)
        for era in range(self.steps.nera):
            beta = tf.constant(self.schedule.betas[str(era)])
            console.rule(f'ERA: {era}, BETA: {beta.numpy()}')
            table = Table(row_styles=['dim', 'none'])
            with Live(table, console=console, screen=False) as live:
                if width is not None and width > 0:
                    live.console.width = width
                estart = time.time()
                for epoch in range(self.steps.nepoch):
                    self.timer.start()
                    x, metrics = train_step((x, beta))  # type: ignore
                    dt = self.timer.stop()
                    if should_print(epoch) or should_log(epoch):
                        if save_x:
                            xarr.append(x.numpy())  # type: ignore

                        record = {
                            'era': era, 'epoch': epoch,
                            'beta': beta, 'dt': dt
                        }
                        record.update(self.metrics_to_numpy(metrics))
                        avgs = self.history.update(record)
                        summary = summarize_dict(avgs)
                        summaries.append(summary)
                        if epoch == 0:
                            table = add_columns(avgs, table)

                        if should_print(epoch):
                            table.add_row(*[f'{v}' for _, v in avgs.items()])

                live.console.log(
                    f'[{RANK}] :: '
                    f'Era {era} took: {time.time() - estart:<3.2g}s'
                )
                live.console.log(
                    f'[{RANK}] :: '
                    f'Avgs over last era:\n{self.history.era_summary(era)}'
                )

            tables[str(era)] = table

        return {
            'xarr': xarr,
            'summaries': summaries,
            'history': self.history,
            'tables': tables,
        }

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
                                  *(-np.pi, np.pi), dtype=TF_FLOAT)
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
                log.info('\n'.join([
                    f'Era {era} took: {time.time() - estart:<3.2g}s',
                    f'Avgs over last era:\n {self.history.era_summary(era)}',
                ]))
                # live.refresh()

            tables[str(era)] = table

        return {
            'xdict': xarr,
            'summaries': summaries,
            'history': self.history,
            'tables': tables,
        }
