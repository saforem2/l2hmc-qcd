"""
inference_utils.py

Collection of helper methods to use for running inference on trained model.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import time
import logging

from typing import Optional, List

from tqdm.autonotebook import tqdm
import tensorflow as tf
try:
    import horovod.tensorflow as hvd
    HAS_HOROVOD = True
    RANK = hvd.rank()
    IS_CHIEF = (RANK == 0)
    NUM_NODES = hvd.size()
except (ImportError, ModuleNotFoundError):
    HAS_HOROVOD = False
    RANK = 0
    IS_CHIEF = (RANK == 0)
    NUM_NODES = 1

import utils.file_io as io

from config import CBARS, GAUGE_LOGS_DIR, HMC_LOGS_DIR
from dynamics.gauge_dynamics import (build_dynamics, convert_to_angle,
                                     GaugeDynamics)
from utils.attr_dict import AttrDict
from utils.plotting_utils import plot_data
from utils.summary_utils import summarize_dict
from utils.data_containers import DataContainer

# pylint:disable=no-member

if IS_CHIEF:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
        stream=sys.stdout
    )
else:
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(asctime)s:%(levelname)s:%(message)s",
        stream=None
    )


SKIP = ['charges', 'sldf', 'sldb', 'Hf', 'Hb', 'Hwf', 'Hwb',
        'ldf_start', 'ldb_start', 'ldf_mid', 'ldf_end',
        'ldb_mid', 'ldb_end', 'Hf_start', 'Hf_mid', 'Hf_end',
        'Hb_start', 'Hb_mid', 'Hb_end']


def restore_from_train_flags(args):
    """Populate entries in `args` using the training `FLAGS` from `log_dir`."""
    train_dir = os.path.join(args.log_dir, 'training')
    flags = AttrDict(dict(io.loadz(os.path.join(train_dir, 'FLAGS.z'))))

    return flags


def load_and_run(
        args: AttrDict,
        x: tf.Tensor = None,
        runs_dir: str = None,
) -> (GaugeDynamics, DataContainer, tf.Tensor):
    """Load trained model from checkpoint and run inference."""
    if not IS_CHIEF:
        return None, None, None

    io.print_dict(args)
    ckpt_dir = os.path.join(args.log_dir, 'training', 'checkpoints')
    flags = restore_from_train_flags(args)
    eps_file = os.path.join(args.log_dir, 'training', 'train_data', 'eps.z')
    flags.eps = io.loadz(eps_file)[-1]
    dynamics = build_dynamics(flags)

    ckpt = tf.train.Checkpoint(dynamics=dynamics,
                               optimizer=dynamics.optimizer)
    manager = tf.train.CheckpointManager(ckpt, max_to_keep=5,
                                         directory=ckpt_dir)
    if manager.latest_checkpoint:
        io.log(f'Restored model from: {manager.latest_checkpoint}')
        status = ckpt.restore(manager.latest_checkpoint)
        status.assert_existing_objects_matched()
        xfile = os.path.join(args.log_dir, 'training',
                             'train_data', 'x_rank0.z')
        io.log(f'Restored x from: {xfile}.')
        x = io.loadz(xfile)

    dynamics, run_data, x = run(dynamics, args, x=x, runs_dir=runs_dir)

    return dynamics, run_data, x


# pylint:disable=too-many-locals
def setup(
        dynamics: GaugeDynamics,
        flags: AttrDict,
        x: Optional[tf.Tensor],
        dirs: Optional[AttrDict] = None,
        runs_dir: Optional[str] = None
):
    """Setup to run inference."""
    if isinstance(flags, dict):
        if not isinstance(flags, AttrDict):
            flags = AttrDict(**flags)

    if dirs is None:
        log_dir = flags.get('log_dir', None)
        if log_dir is None:
            log_dir = io.make_log_dir(flags)

    else:
        log_dir = dirs.get('log_dir', None)
        if log_dir is None:
            log_dir = flags.get('log_dir', None)
            if log_dir is None:
                io.make_log_dir(flags)

    if runs_dir is None:
        if dynamics.config.get('hmc', False):
            runs_dir = os.path.join(log_dir, 'inference_hmc')
        else:
            runs_dir = os.path.join(log_dir, 'inference')

    eps = dynamics.eps
    if hasattr(eps, 'numpy'):
        eps = eps.numpy()

    flags.eps = eps
    flags.logging_steps = 1  # Record after each accept/reject

    run_dir = io.make_run_dir(flags, runs_dir)
    data_dir = os.path.join(run_dir, 'run_data')
    summary_dir = os.path.join(run_dir, 'summaries')
    log_file = os.path.join(run_dir, 'run_log.txt')
    io.check_else_make_dir([runs_dir, run_dir, data_dir, summary_dir])
    writer = tf.summary.create_file_writer(summary_dir)
    writer.set_as_default()

    run_steps = flags.get('run_steps', 2000)
    print_steps = flags.get('print_steps', 5)
    beta = flags.get('beta', flags.get('beta_final', None))

    if x is None:
        x = convert_to_angle(tf.random.normal(shape=dynamics.x_shape))

    test_step = dynamics.test_step
    if flags.get('compile', True):
        test_step = tf.function(dynamics.test_step)
        io.log('Compiled `dynamics.test_step` using tf.function!')

    run_data = DataContainer(flags.run_steps)

    template = '\n'.join([f'beta: {beta}',
                          f'eps: {dynamics.eps.numpy():.4g}'])
    io.log(f'Running inference with:\n{template}')

    # Run 50 MD updates (w/o accept/reject) to ensure chains don't get stuck
    md_steps = flags.get('md_steps', 0)
    if md_steps > 0:
        for _ in range(md_steps):
            mc_states, _ = dynamics.md_update((x, beta), training=False)
            x = mc_states.out.x

    try:
        x, metrics = test_step((x, tf.constant(beta)))
    except Exception as exception:  # pylint:disable=broad-except
        io.log(f'Exception: {exception}')
        test_step = dynamics.test_step
        x, metrics = test_step((x, tf.constant(beta)))

    header = run_data.get_header(metrics,
                                 skip=SKIP,
                                 prepend=['{:^12s}'.format('step')])
    #  io.log(header)
    io.log(header.split('\n'), should_print=True)
    # -------------------------------------------------------------

    steps = tf.range(run_steps, dtype=tf.int64)
    if NUM_NODES == 1:
        ctup = (CBARS['red'], CBARS['green'], CBARS['red'], CBARS['reset'])
        steps = tqdm(steps, desc='running', unit='step',
                     bar_format=("%s{l_bar}%s{bar}%s{r_bar}%s" % ctup))

    output = AttrDict({
        'x': x,
        'test_step_fn': test_step,
        'run_data': run_data,
        'header': header,
        'steps': steps,
        'flags': flags,
        'runs_dir': runs_dir,
        'log_dir': log_dir,
        'log_file': log_file,
        'run_dir': run_dir,
        'data_dir': data_dir,
        'summary_dir': summary_dir,
        'writer': writer,
        'run_params': AttrDict({
            'eps': flags.eps,
            'beta': beta,
            'run_steps': run_steps,
            'plaq_weight': dynamics.config.get('plaq_weight', 0.),
            'charge_weight': dynamics.config.get('charge_weight', 0.),
            'lattice_shape': dynamics.config.get('lattice_shape', None),
            'num_steps': dynamics.config.get('num_steps', None),
            'net_weights': dynamics.net_weights,
            'input_shape': dynamics.x_shape,
            'print_steps': print_steps,
        }),
    })

    return output


# pylint:disable=too-many-arguments
def run(
        dynamics: GaugeDynamics,
        args: AttrDict,
        dirs: Optional[AttrDict] = None,
        x: Optional[tf.Tensor] = None,
        runs_dir: Optional[str] = None,
        save_x: Optional[bool] = False,

) -> (GaugeDynamics, DataContainer, tf.Tensor, List):
    """Run inference."""
    if not IS_CHIEF:
        return None, None, None

    config = setup(dynamics, args, x, dirs, runs_dir)
    run_params = config.get('run_params', None)

    run_data, x, x_arr = _run_dynamics(dynamics, config, save_x=save_x)

    run_data.flush_data_strs(config.log_file, mode='a')
    run_data.write_to_csv(config.log_dir, config.run_dir,
                          hmc=dynamics.config.hmc)
    io.save_inference(config.run_dir, run_data)
    if args.get('save_run_data', True):
        run_data.save_data(config.data_dir)

    config.update(dynamics.params)
    #  run_params.update(dynamics.params)
    io.save_params(run_params, config.run_dir, name='run_params')

    args.update({'logging_steps': 1})
    plot_data(run_data, config.run_dir, args,
              thermalize=True, params=run_params)

    return dynamics, run_data, x, x_arr


def _run_dynamics(
        dynamics: GaugeDynamics,
        config: AttrDict,
        save_x: bool = False,
) -> (DataContainer, tf.Tensor, list):
    """Run inference on trained dynamics."""
    if not IS_CHIEF:
        return None, None, None

    params = config.get('run_params', None)
    test_step = config.get('test_step_fn', None)
    run_data = config.get('run_data', None)
    steps = config.get('steps', None)

    x_arr = []

    def timed_step(x: tf.Tensor, beta: tf.Tensor):
        start = time.time()
        x, metrics = test_step((x, tf.constant(beta)))
        metrics.dt = time.time() - start
        if save_x:
            x_arr.append(x.numpy())

        return x, metrics

    for step in steps:
        x, metrics = timed_step(config.x, params.beta)
        run_data.update(step, metrics)

        if step % params.print_steps == 0:
            summarize_dict(metrics, step, prefix='testing')
            data_str = run_data.get_fstr(step, metrics, skip=SKIP)
            io.log(data_str, should_print=True)

        if (step + 1) % 1000 == 0:
            io.log(config.header, should_print=True)

    return run_data, x, x_arr


def run_hmc(
        args: AttrDict,
        dirs: Optional[AttrDict] = None,
        hmc_dir: Optional[str] = None,
        skip_existing: Optional[bool] = False,
) -> (GaugeDynamics, DataContainer, tf.Tensor):
    """Run HMC using `inference_args` on a model specified by `params`.

    NOTE:
    -----
    args should be a dict with the following keys:
        - 'hmc'
        - 'eps'
        - 'beta'
        - 'num_steps'
        - 'run_steps'
        - 'lattice_shape'
    """
    if not IS_CHIEF:
        return None, None, None

    log_dir = None
    if dirs is None:
        dirs = AttrDict({})
        log_dir = args.get('log_dir', None)

    if log_dir is None:
        log_dir = args.get('log_dir', None)

    if hmc_dir is None:
        #  root_dir = HMC_LOGS_DIR
        month_str = io.get_timestamp('%Y_%m')
        if dirs is not None:
            log_dir = dirs.get('log_dir', None)
            if log_dir is None:
                log_dir = args.get('log_dir', None)
        if log_dir is None:
            log_dir = HMC_LOGS_DIR

        hmc_dir = os.path.join(log_dir, month_str)

    io.check_else_make_dir(hmc_dir)
    dirs.update({
        'log_dir': log_dir,
        'hmc_dir': hmc_dir
    })

    def get_run_fstr(run_dir):
        _, tail = os.path.split(run_dir)
        fstr = tail.split('-')[0]
        return fstr

    if skip_existing:
        run_dirs = [os.path.join(hmc_dir, i) for i in os.listdir(hmc_dir)]
        run_fstrs = [get_run_fstr(i) for i in run_dirs]
        run_fstr = io.get_run_dir_fstr(args)
        if run_fstr in run_fstrs:
            io.log('ERROR:Existing run found! Skipping.')
            return None, None, None

    dynamics = build_dynamics(args)
    dynamics, run_data, x, x_arr = run(dynamics, args,
                                       dirs=dirs, runs_dir=hmc_dir)

    return dynamics, run_data, x, x_arr
