"""
trainer.py

Implements methods for training L2HMC sampler
"""
from __future__ import absolute_import, annotations, division, print_function
from contextlib import nullcontext
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional

import aim
from aim import Distribution
import horovod.tensorflow as hvd
import numpy as np
from omegaconf import DictConfig
from rich import box
from rich.live import Live
from rich.table import Table
import tensorflow as tf
from tensorflow._api.v2.train import CheckpointManager
import tensorflow.python.framework.ops as ops
from tensorflow.python.keras import backend as K

from l2hmc import configs
from l2hmc.common import ScalarLike
from l2hmc.configs import ExperimentConfig
from l2hmc.configs import CHECKPOINTS_DIR
from l2hmc.dynamics.tensorflow.dynamics import Dynamics
from l2hmc.group.su3.tensorflow.group import SU3
from l2hmc.group.u1.tensorflow.group import U1Phase
from l2hmc.lattice.su3.tensorflow.lattice import LatticeSU3
from l2hmc.lattice.u1.tensorflow.lattice import LatticeU1
from l2hmc.learning_rate.tensorflow.learning_rate import ReduceLROnPlateau
from l2hmc.loss.tensorflow.loss import LatticeLoss
from l2hmc.network.tensorflow.network import NetworkFactory
from l2hmc.trackers.tensorflow.trackers import update_summaries
from l2hmc.trainers.trainer import BaseTrainer
from l2hmc.utils.history import summarize_dict
from l2hmc.utils.rich import get_width, is_interactive
# from l2hmc.utils.logger import get_pylogger
from l2hmc.utils.step_timer import StepTimer
# tf.autograph.set_verbosity(0)
# os.environ['AUTOGRAPH_VERBOSITY'] = '0'
# JIT_COMPILE = (len(os.environ.get('JIT_COMPILE', '')) > 0)
# from tqdm.auto import trange
if is_interactive():
    from tqdm.rich import trange
else:
    from tqdm.auto import trange

# log = get_pylogger(__name__)
log = logging.getLogger(__name__)

Tensor = tf.Tensor
Array = np.ndarray
TF_FLOAT = tf.keras.backend.floatx()
NP_INT = (np.int8, np.int16, np.int32, np.int64)
Layer = tf.keras.layers.Layer
Model = tf.keras.Model
Optimizer = tf.keras.optimizers.Optimizer
TensorLike = tf.types.experimental.TensorLike

HVD_FP_MAP = {
    'fp16': hvd.Compression.fp16,
    'none': hvd.Compression.none
}


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


def reset_optimizer(optimizer: Optimizer):
    """Reset optimizer states when changing beta during training."""
    # > [!NOTE] Preserve Iterations
    # > We don't want to reset iteration counter. From tf docs:
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


from l2hmc.lattice.u1.tensorflow.lattice import plaq_exact

# TODO: Replace arguments in __init__ call below with configs.TrainerConfig
class Trainer(BaseTrainer):
    def __init__(
            self,
            cfg: DictConfig | ExperimentConfig,
            build_networks: bool = True,
            ckpt_dir: Optional[os.PathLike] = None,
            keep: Optional[str | list[str]] = None,
            skip: Optional[str | list[str]] = None,
    ) -> None:
        super().__init__(cfg=cfg, keep=keep, skip=skip)
        # assert isinstance(self.config, ExperimentConfig)
        if self.config.compression in [True, 'fp16']:
            self.compression = hvd.Compression.fp16
        else:
            self.compression = hvd.Compression.none

        assert isinstance(
            self.config,
            (configs.ExperimentConfig, ExperimentConfig)
        )

        self._gstep = 0
        self.size = hvd.size()
        self.rank = hvd.rank()
        self.local_rank = hvd.local_rank()
        self._is_chief = (self.local_rank == 0 and self.rank == 0)
        self.lattice = self.build_lattice()
        self.loss_fn = self.build_loss_fn()
        self.dynamics = self.build_dynamics(
            build_networks=build_networks
        )
        # Call dynamics to make sure everything initialized
        self.ckpt_dir = (
            Path(CHECKPOINTS_DIR).joinpath('checkpoints')
            if ckpt_dir is None
            else Path(ckpt_dir).resolve()
        )
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        # self.load_ckpt()
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
        # self.rank = hvd.local_rank()
        # self.global_rank = hvd.rank()
        # self._is_chief = self.rank == 0 and self.global_rank == 0
        if self.config.dynamics.group == 'U1':
            self.g = U1Phase()
        elif self.config.dynamics.group == 'SU3':
            self.g = SU3()
        else:
            raise ValueError

    def call_dynamics(
            self,
            x: Optional[Tensor] = None,
            beta: Optional[float] = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        beta = 1. if beta is None else beta
        if x is None:
            x = self.dynamics.lattice.random()
            # state = self.dynamics.random_state(beta)
            # x = state.x
        inputs = (x, tf.constant(beta, dtype=TF_FLOAT))
        return self.dynamics(inputs)

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

    def setup_CheckpointManager(self):
        log.info(f'Looking for checkpoints in: {self.ckpt_dir}')
        ckpt = tf.train.Checkpoint(dynamics=self.dynamics,
                                   optimizer=self.optimizer)
        manager = tf.train.CheckpointManager(
            ckpt,
            self.ckpt_dir.as_posix(),
            max_to_keep=5
        )
        if manager.latest_checkpoint and self.config.restore:
            ckpt.restore(manager.latest_checkpoint)
            log.warning(
                f'Restored checkpoint from: {manager.latest_checkpoint}'
            )
        else:
            log.info('No checkpoints found to load from. Continuing')

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
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            # model: Optional[Model | Layer] = None,
            optimizer: Optional[Optimizer] = None,
            # weights: Optional[dict] = None,
            log_weights: bool = True,
    ):
        # record = {} if record is None else record
        assert job_type in ['train', 'eval', 'hmc']
        if step is None:
            timer = self.timers.get(job_type, None)
            if isinstance(timer, StepTimer):
                step = timer.iterations

        if step is not None:
            metrics.update({f'{job_type[0]}step': step})

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
        if step is not None:
            update_summaries(
                    step=step,
                    # model=model,
                    # weights=weights,
                    metrics=metrics,
                    prefix=job_type,
                    optimizer=optimizer,
                    # job_type=job_type,
            )
            # if model is not None and job_type == 'train':
            #     update_summaries(
            #         step=step,
            #         model=self.dynamics,
            #         prefix='model',
            #         job_type=job_type,
            #     )
            # if job_type == 'train' and log_weights:
            #     weights = {
            #         f'{w.name}': w
            #         for w in self.dynamics.weights
            #     }
            #     log_dict(weights, step, prefix='debug-weights')

            if writer is not None:
                writer.flush()

        if self.config.init_wandb or self.config.init_aim:
            if job_type == 'train' and log_weights:
                self.track_weights(run=run)

            self.track_metrics(
                record=metrics,
                avgs=avgs,
                job_type=job_type,
                step=step,
                run=run,
                arun=arun,
                # log_weights=log_weights,
            )

        return avgs, summary

    def track_weights(
            self,
            run: Optional[Any] = None,
            # arun: Optional[Any] = None
    ) -> None:
        if not self._is_chief:
            return

        rename = lambda k: (
            f"model/{k.replace('/', '.').replace('dynamics', 'networks')}"
        )

        # f"model/{k.replace('/', '.').replace('dynamics', 'networks')}": v
        if run is not None:
            weights = {
                rename(k): v for k, v in self.dynamics.get_all_weights().items()
            }
            run.log(weights, commit=False)
            run.log(
                {f'{k}/avg': tf.reduce_mean(v) for k, v in weights.items()},
                commit=True
            )
        # if run is not None:
        #     weights = {
        #         k.replace('dynamics', 'networks'): v
        #         for k, v in weights.items()
        #     }
        #     run.log(weights, commit=True)

        # if run is not None:
        #     run.log({
        #         k.replace('/', '.').replace('dynamics', 'networks'): v
        #         for k, v in self.dynamics.get_all_weights()
        #     }, commit=True)
            # weights = {
            #     k.replace('dynamics', 'networks'): v
            #     for k, v in weights.items()
            # }
            # run.log(weights, commit=True)
            # run.log({f'wandb/{job_type}': record}, commit=False)
            # run.log({f'avgs/wandb.{job_type}': avgs}, commit=False)
            # if weights_dict is not None:
            #     # run.log({'Dynamics': weights}, commit=False)
            #     run.log(weights_dict, commit=False)
            # if dQdict is not None:
            #     run.log(dQdict, commit=True)


    def track_metrics(
            self,
            record: dict[str, TensorLike | ScalarLike],
            avgs: dict[str, TensorLike | ScalarLike],
            job_type: str,
            step: Optional[int],
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            # log_weights: bool = True,
    ) -> None:
        if not self._is_chief:
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

        # weights_dict = None
        # if job_type == 'train' and log_weights:
        #     # weights = self.dynamics.get_weights_dict()
        #     weights_dict = {
        #         f'model/{w.name}': w for w in self.dynamics.weights
        #     }

        if run is not None:
            run.log({f'wandb/{job_type}': record}, commit=False)
            run.log({f'avgs/wandb.{job_type}': avgs}, commit=False)
            # if weights_dict is not None:
            #     # run.log({'Dynamics': weights}, commit=False)
            #     run.log(weights_dict, commit=False)
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

            # if weights is not None:
            #     self.aim_track(
            #         weights,
            #         prefix='Dynamics',
            #         **kwargs
            #     )

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
        if inputs[0].shape != self.dynamics.xshape:
            x = tf.reshape(
                inputs[0],
                (inputs[0].shape[0],
                 *self.dynamics.xshape[1:]))
            inputs = (x, *inputs[1:])
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
        assert isinstance(metrics, dict)

        return xout, metrics

    def get_context_manager(self, table: Table) -> Live | nullcontext:
        make_live = (
            self._is_chief
            and self.size == 1       # not worth the trouble when distributed
            and not is_interactive()  # AND not in a jupyter / ipython kernel
            and int(get_width()) > 120    # make sure wide enough to fit table
        )
        if make_live:
            return Live(
                table,
                # screen=True,
                # transient=True,
                auto_refresh=False,
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
            nprint: Optional[int] = None,
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
        nprint = (
            max(1, min(50, eval_steps // 50)) if nprint is None else nprint
        )
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
            dynamic_step_size:  Optional[bool] = None,
            nprint: Optional[int] = None,
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
            nprint=nprint,
        )
        x = setup['x']
        assert isinstance(x, Tensor)
        xshape = x.shape
        if nchains is not None:
            if x.shape[0] != nchains:
                log.warning(f'x.shape: {xshape}')
                x = x[:nchains, ...]  # type:ignore
                log.warning(f'x[:nchains].shape: {xshape}')

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
        if job_type == 'hmc':
            assert eps is not None

        def eval_fn(inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, dict]:
            if job_type == 'hmc':
                return self.hmc_step(inputs, eps=eps, nleapfrog=nleapfrog)
            return self.eval_step(inputs)  # type:ignore

        ctx = self.get_context_manager(table)
        
        def refresh_view():
            if isinstance(ctx, Live):
                ctx.refresh()
                ctx.console.clear_live()

        x = self.warmup(beta)
        with self.get_context_manager(table) as ctx:
            # for step in trange(eval_steps):
            for step in trange(
                    eval_steps,
                    dynamic_ncols=True,
                    disable=(not self._is_chief),
            ):
                timer.start()
                x, metrics = eval_fn((x, beta))  # type:ignore
                dt = timer.stop()
                # if step % setup['nprint'] == 0 or step % setup['nlog'] == 0:
                if (
                        # step >= 0 and
                        (step % setup['nlog'] == 0
                         or step % setup['nprint'] == 0)
                ):
                    record = {
                        f'{job_type[0]}step': step,
                        'dt': dt,
                        'beta': beta,
                        'loss': metrics.pop('loss', None),
                        'dQsin': metrics.pop('dQsin', None),
                        'dQint': metrics.pop('dQint', None),
                    }
                    record.update(metrics)
                    avgs, summary = self.record_metrics(run=run,
                                                        arun=arun,
                                                        step=step,
                                                        writer=writer,
                                                        metrics=record,
                                                        job_type=job_type)
                    summaries.append(summary)
                    table = self.update_table(
                        table=setup['table'],
                        step=step,
                        avgs=avgs,
                    )

                    if (
                            # not isinstance(setup['ctx'], Live)
                            step % setup['nprint'] == 0
                    ):
                        log.info(summary)

                    refresh_view()

                    # if step == 0:
                    #     table = add_columns(avgs, table)
                    # else:
                    #     table.add_row(*[f'{v}' for _, v in avgs.items()])

                    if avgs.get('acc', 1.0) <= 1e-5:
                        if stuck_counter < patience:
                            stuck_counter += 1
                        else:
                            self.console.log('Chains are stuck! Re-drawing x!')
                            x = self.lattice.random()
                            stuck_counter = 0

                    if job_type == 'hmc' and dynamic_step_size:
                        acc = metrics.get('acc_mask', None)
                        record['eps'] = eps
                        if acc is not None and eps is not None:
                            acc_avg = tf.reduce_mean(acc)
                            if acc_avg < 0.66:
                                eps -= (eps / 10.)
                            else:
                                eps += (eps / 10.)

        if isinstance(ctx, Live):
            ctx.console.clear_live()

        tables[str(0)] = table

        return {
            'timer': timer,
            'history': history,
            'summaries': summaries,
            'tables': tables,
        }

    @tf.function(experimental_follow_type_hints=True)
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

        self._gstep += 1

        return xout, metrics

    def train_step_detailed(
            self,
            x: Optional[Tensor] = None,
            beta: Optional[Tensor | float] = None,
            era: int = 0,
            epoch: int = 0,
            run: Optional[Any] = None,
            arun: Optional[Any] = None,
            writer: Optional[Any] = None,
            rows: Optional[dict] = None,
            summaries: Optional[list] = None,
            verbose: bool = True,
    ) -> tuple[Tensor, dict]:
        # tstart = time.time()
        # xinit, beta = inputs
        # xinit = self.g.compat_proj(self.dynamics.unflatten(xinit))
        # beta = tf.constant(beta) if isinstance(beta, float) else beta
        if x is None:
            x = self.dynamics.lattice.random()
        if beta is None:
            beta = self.config.annealing_schedule.beta_init

        if isinstance(beta, float):
            beta = tf.constant(beta)

        self.timers['train'].start()
        xout, metrics = self.train_step((x, beta))  # type:ignore
        dt = self.timers['train'].stop()
        record = {
            'era': era,
            'epoch': epoch,
            'tstep': self._gstep,
            'dt': dt,
            'beta': beta,
            'loss': metrics.pop('loss', None),
            'dQsin': metrics.pop('dQsin', None),
            'dQint': metrics.pop('dQint', None),
            **metrics,
        }
        # record.update(metrics)
        avgs, summary = self.record_metrics(
            run=run,
            arun=arun,
            step=self._gstep,
            writer=writer,
            metrics=record,  # metrics from Dynamics
            job_type='train',
            log_weights=True,
            # model=self.dynamics,
            # prefix='Dynamics',
            # weights=self.dynamics.get_weights_dict(),
            optimizer=self.optimizer,
        )
        if rows is not None:
            rows[self._gstep] = avgs
        if summaries is not None:
            summaries.append(summary)
        if verbose:
            log.info(summary)

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
            nprint: Optional[int] = None,
            nlog: Optional[int] = None,
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

        log_freq = self.steps.log if nlog is None else nlog
        print_freq = self.steps.print if nprint is None else nprint
        assert log_freq is not None and print_freq is not None

        def should_print(epoch):
            return (self._is_chief and (epoch % print_freq == 0))

        def should_log(epoch):
            return (self._is_chief and (epoch % log_freq == 0))

        def refresh_view():
            if isinstance(ctx, Live):
                ctx.console.clear_live()
                ctx.update(table)
                ctx.refresh()


        # with self.get_context_manager(table) as ctx:
        with ctx:
            # if isinstance(ctx, Live):
            #     tstr = ' '.join([
            #         f'ERA: {era}/{self.steps.nera - 1}',
            #         f'BETA: {beta:.3f}',
            #     ])
            #     # ctx.console.clear()
            #     ctx.console.clear_live()
            #     ctx.console.rule(tstr)
            #     ctx.update(table)

            # for epoch in trange(
            # iterator = tqdm(
            #     range(nepoch),
            #     desc='Training',
            #     position=0,
            #     leave=True,
            #     dynamic_ncols=True,
            #     disable=(not self._is_chief)
            # )
            #         nepoch,
            #         dynamic_ncols=True,
            #         disable=(not self._is_chief)
            #     desc
            # ):
            for epoch in trange(
                    nepoch,
                    dynamic_ncols=True,
                    disable=(not self._is_chief)
            ):
                self.timers['train'].start()
                x, metrics = self.train_step((x, beta))  # type:ignore
                dt = self.timers['train'].stop()
                losses.append(metrics['loss'])
                if (should_print(epoch) or should_log(epoch)):
                    record = {
                        'era': era,
                        'epoch': epoch,
                        'tstep': self._gstep,
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
                        log_weights=True,
                        # model=self.dynamics,
                        # prefix='Dynamics',
                        # weights=self.dynamics.get_weights_dict(),
                        optimizer=self.optimizer,
                    )
                    rows[self._gstep] = avgs
                    summaries.append(summary)

                    if (
                            should_print(epoch)
                            # and not isinstance(ctx, Live)
                    ):
                        log.info(summary)

                    table = self.update_table(
                        table=table,
                        step=epoch,
                        avgs=avgs
                    )

                    if avgs.get('acc', 1.0) < 1e-5:
                        self.reset_optimizer()
                        log.warning('Chains are stuck! Re-drawing x !')
                        x = self.draw_x()

                    refresh_view()

                if isinstance(ctx, Live):
                    ctx.console.clear()
                    ctx.console.clear_live()

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

        # -- Setup checkpoint manager for TensorFlow --------------------------
        # manager = None
        # if self._is_chief:
        manager = self.setup_CheckpointManager()

        self._gstep = K.get_value(self.optimizer.iterations)
        # -- Setup Step information (nera, nepoch, etc). ----------------------
        nera = self.config.steps.nera if nera is None else nera
        nepoch = self.config.steps.nepoch if nepoch is None else nepoch
        # extend = self.config.steps.extend_last_era
        assert nera is not None and isinstance(nera, int)
        assert nepoch is not None and isinstance(nepoch, int)

        # -- Setup beta information -------------------------------------------
        if beta is None:
            betas = self.config.annealing_schedule.setup(
                nera=nera,
                nepoch=nepoch
            )
        elif isinstance(beta, (list, np.ndarray)):
            nera = len(beta)
            betas = {f'{i}': b for i, b in zip(range(nera), beta)}
        elif isinstance(beta, (int, float)):
            # nera = self.config.steps.nera if nera is None else nera
            betas = {f'{i}': b for i, b in zip(range(nera), nera * [beta])}
        elif isinstance(beta, dict):
            nera = len(list(beta.keys()))
            betas = {f'{i}': b for i, b in beta.items()}
        else:
            raise TypeError(
                'Expected `beta` to be one of: `float, int, list, dict`',
                f' received: {type(beta)}'
            )

        beta_final = list(betas.values())[-1]
        assert beta_final is not None and isinstance(beta_final, float)
        return {
            'x': x,
            'nera': nera,
            'nepoch': nepoch,
            'betas': betas,
            'manager': manager,
            'writer': writer,
            'train_dir': train_dir,
            'beta_final': beta_final,
        }

    def warmup(
            self,
            beta: float | Tensor,
            nsteps: int = 100,
            tol: float = 1e-3,
    ) -> Tensor:
        # state = self.dynamics.random_state(beta)
        x = self.dynamics.lattice.random()
        if not isinstance(beta, Tensor):
            beta = tf.constant(beta, dtype=TF_FLOAT)
        # btensor = tf.cast(beta, dtype=TF_FLOAT)
        pexact = plaq_exact(beta)
        for i in range(nsteps):
            x, metrics = self.dynamics((x, beta))
            plaqs = metrics.get('plaqs', None)
            if plaqs is not None:
                pdiff = tf.math.reduce_sum(tf.math.abs(plaqs - pexact))
                if pdiff < tol:
                    log.warning(
                        f'Chains thermalized!'
                        f' step: {i},'
                        f' plaq_diff: {pdiff:.4f}'
                    )
                    return x
                
        return x

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
            nprint: Optional[int] = None,
            nlog: Optional[int] = None,
            beta: Optional[float | list[float] | dict[str, float]] = None,
            # restore: bool = True,
    ) -> dict:
        """Perform training and return dictionary of results."""
        # _, _ = self.call_dynamics(
        #     beta=self.config.annealing_schedule.beta_init
        # )

        setup = self._setup_training(
            x=x,
            skip=skip,
            train_dir=train_dir,
            nera=nera,
            nepoch=nepoch,
            beta=beta,
            # restore=restore,
        )
        era = 0
        # epoch = 0
        extend = 1
        x = setup['x']
        nera = setup['nera']
        betas = setup['betas']
        nepoch = setup['nepoch']
        manager = setup['manager']
        train_dir = setup['train_dir']
        beta_final = setup['beta_final']
        writer = setup['writer']
        assert x is not None
        assert nera is not None
        assert train_dir is not None
        x = self.warmup(
            betas.get('0', beta_final)
        )
        for era in range(nera):
            b = tf.constant(betas.get(str(era), beta_final))
            if era == (nera - 1) and self.steps.extend_last_era is not None:
                extend = int(self.steps.extend_last_era)

            if self._is_chief:
                if era > 1 and str(era - 1) in self.summaries['train']:
                    esummary = self.histories['train'].era_summary(f'{era-1}')
                    log.info(f'Avgs over last era:\n {esummary}\n')

                self.console.rule(f'ERA: {era} / {nera}, BETA: {b:.3f}')

            x = self.warmup(b)

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
                nlog=nlog,
                nprint=nprint
            )

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

            if self._is_chief:
                st0 = time.time()
                self.save_ckpt(manager)
                log.info(f'Saving took: {time.time() - st0:<5g}s')
                log.info(f'Checkpoint saved to: {self.ckpt_dir}')
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

                self.console.rule(f'ERA: {era} / {nera - 1}, BETA: {b:.3f}')

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
        if isinstance(metric, (float, np.ScalarType)):
            return np.array(metric)

        if isinstance(metric, list):
            if isinstance(metric[0], (Tensor, tf.Variable)):
                return tf.stack(metric)
            if isinstance(metric[0], np.ndarray):
                return np.stack(metric)
        if (
                isinstance(metric, Tensor)
                and hasattr(metric, 'numpy')
                and isinstance(metric.numpy, Callable)
        ):
            return metric.numpy()

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
            if isinstance(step, int):
                step = step
            if callable(getattr(step, 'item', None)):
                step = step.item()  # type:ignore
            # try:
            #     step = step.item()
            # except AttributeError:
            #     pass

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
