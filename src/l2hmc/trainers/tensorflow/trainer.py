"""
trainer.py

Implements methods for training L2HMC sampler
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import time
from typing import Callable

import numpy as np
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer

from l2hmc.configs import Steps
from l2hmc.dynamics.tensorflow.dynamics import Dynamics, to_u1
from l2hmc.loss.tensorflow.loss import LatticeLoss
from l2hmc.utils.console import is_interactive
from l2hmc.utils.history import summarize_dict
from l2hmc.utils.step_timer import StepTimer
from l2hmc.utils.tensorflow.history import History


COLORS = 10 * ['red', 'yellow', 'green', 'blue', 'magenta', 'cyan']
log = logging.getLogger(__name__)


TF_FLOAT = tf.keras.backend.floatx()
Tensor = tf.Tensor


def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        # Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=5),
    )
    # layout["main"].split_row(
    #     # Layout(name="side"),
    #     Layout(name="body", minimum_size=80),
    # )
    # layout["side"].split(Layout(name="box1"), Layout(name="box2"))
    return layout


class Trainer:
    def __init__(
            self,
            steps: Steps,
            dynamics: Dynamics,
            optimizer: Optimizer,
            loss_fn: Callable = LatticeLoss,
            keep: str | list[str] = None,
            skip: str | list[str] = None,
    ) -> None:
        self.steps = steps
        self.dynamics = dynamics
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.history = History(steps=steps)

        evals_per_step = self.dynamics.config.nleapfrog * steps.log
        self.timer = StepTimer(evals_per_step=evals_per_step)

        self.keep = [] if keep is None else keep
        self.skip = [] if skip is None else skip
        if isinstance(self.keep, str):
            self.keep = [self.keep]
        if isinstance(self.skip, str):
            self.skip = [self.skip]

    def train_step(self, inputs: tuple[Tensor, float]) -> tuple[Tensor, dict]:
        xinit, beta = inputs
        with tf.GradientTape() as tape:
            x_out, metrics = self.dynamics((to_u1(xinit), tf.constant(beta)))
            xprop = to_u1(metrics.pop('mc_states').proposed.x)
            loss = self.loss_fn(x_init=xinit, x_prop=xprop, acc=metrics['acc'])

        grads = tape.gradient(loss, self.dynamics.trainable_variables)
        updates = zip(grads, self.dynamics.trainable_variables)
        self.optimizer.apply_gradients(updates)
        record = {
            'loss': loss,
        }
        for key, val in metrics.items():
            record[key] = val

        return to_u1(x_out), record

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

    def train(
            self,
            xinit: Tensor = None,
            beta: float = 1.,
            skip: str | list[str] = None,
            compile: bool = True,
            jit_compile: bool = False,
            save_x: bool = False,
            width: int = 0,
    ) -> dict:
        """Train l2hmc Dynamics."""
        summaries = []
        if isinstance(skip, str):
            skip = [skip]
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

        # nera = self.steps.nera
        # nepoch = self.steps.nepoch
        interactive = is_interactive()
        should_log = lambda epoch: (epoch % self.steps.log == 0)  # noqa
        should_print = lambda epoch: (epoch % self.steps.print == 0)  # noqa

        xarr = []
        tables = {}
        summaries = []
        for era in range(self.steps.nera):
            table = Table(collapse_padding=True,
                          safe_box=interactive,
                          row_styles=['dim', 'none'])
            with Live(table, screen=False) as live:
                if is_interactive() and width > 0:
                    live.console.width = width
                estart = time.time()
                for epoch in range(self.steps.nepoch):
                    self.timer.start()
                    x, metrics = train_step((x, beta))  # type: ignore
                    dt = self.timer.stop()
                    if should_print(epoch) or should_log(epoch):
                        if save_x:
                            xarr.append(x.numpy())  # type: ignore

                        record = {'era': era, 'epoch': epoch, 'dt': dt}
                        record.update(self.metrics_to_numpy(metrics))
                        avgs = self.history.update(record)
                        summary = summarize_dict(avgs)
                        summaries.append(summary)
                        if epoch == 0:
                            for idx, key in enumerate(avgs.keys()):
                                table.add_column(str(key).upper(),
                                                 justify='center',
                                                 style=COLORS[idx])

                        if should_print(epoch):
                            table.add_row(*[f'{v:5}' for _, v in avgs.items()])

                live.console.log(
                    f'Era {era} took: {time.time() - estart:<3.2g}s'
                )
                live.console.log(
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
