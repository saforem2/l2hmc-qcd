"""
inference_utils.py

Collection of helper methods to use for running inference on trained model.
"""
from __future__ import absolute_import, division, print_function

from models.gauge_model_new import GaugeModel, HEADER
import datetime
from utils.parse_args import parse_args
from utils.attr_dict import AttrDict
from eager.plotting import plot_data
from eager.file_io import get_run_str, make_log_dir
from plotters.data_utils import therm_arr
from dynamics.dynamics import DynamicsConfig
from network import NetworkConfig
from config import (NET_WEIGHTS_HMC, NET_WEIGHTS_L2HMC, NetWeights, NP_FLOAT,
                    PI, PROJECT_DIR, TF_FLOAT, TF_INT)
import utils.file_io as io
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import numpy as np
import os
import time
from .training_utils import build_model

import tensorflow as tf

if tf.__version__.startswith('1.'):
    TF_VERSION = '1.x'
elif tf.__version__.startswith('2.'):
    TF_VERSION = '2.x'

try:
    import horovod.tensorflow as hvd

    hvd.init()
    if TF_VERSION == '2.x':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(
                gpus[hvd.local_rank()], 'GPU'
            )
    elif TF_VERSION == '1.x':
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        tf.compat.v1.enable_eager_execution(config=config)

except ImportError:
    if TF_VERSION == '1.x':
        tf.compat.v1.enable_eager_execution()


#  from models.gauge_model_eager import GaugeModel


sns.set_palette('bright')


def get_run_num(run_dir):
    dirnames = [i for i in os.listdir(run_dir) if i.startwsith('run_')]
    if len(dirnames) == 0:
        return 1

    return sorted([int(i.split('_')) for i in dirnames])[-1] + 1


def run_inference_hmc(FLAGS):
    IS_CHIEF = (  # pylint:disable=invalid-name
        not FLAGS.horovod
        or FLAGS.horovod and hvd.rank() == 0
    )

    HFLAGS = AttrDict(dict(FLAGS))
    #  HFLAGS.log_dir = os.path.join(FLAGS.log_dir, 'HMC_START')
    HFLAGS.dropout_prob = 0.
    HFLAGS.hmc = True
    HFLAGS.net_weights = NET_WEIGHTS_HMC

    model, HFLAGS = build_model(HFLAGS, save_params=False)
    outputs, data_strs = model.run_eager(HFLAGS.run_steps,
                                         beta=HFLAGS.beta_final,
                                         save_run_data=True,
                                         ckpt_dir=None)
    if IS_CHIEF:
        log_dir = HFLAGS.log_dir
        runs_dir = os.path.join(log_dir, 'inference_HMC')
        io.check_else_make_dir(runs_dir)
        run_dir = os.path.join(runs_dir, f'run_{get_run_num(runs_dir)}')
        io.check_else_make_dir(run_dir)

        history_file = os.path.join(run_dir, 'inference_log.txt')
        with open(history_file, 'w') as f:
            f.write('\n'.join(data_strs))

        run_params = {
            'beta': HFLAGS.beta_final,
            'dynamics.eps': model.dynamics.eps.numpy(),
            'net_weights': model.dynamics.config.net_weights
        }
        io.save_dict(run_params, run_dir, 'run_params')

        outputs_dir = os.path.join(run_dir, 'outputs')
        io.check_else_make_dir(outputs_dir)
        for key, val in outputs.items():
            out_file = os.path.join(outputs_dir, f'{key}.z')
            io.savez(np.array(val), out_file, key)

        plot_data(outputs, run_dir, HFLAGS, thermalize=True)

    return model, outputs


def run_inference(FLAGS, model=None):
    IS_CHIEF = (  # pylint:disable=invalid-name
        not FLAGS.horovod
        or FLAGS.horovod and hvd.rank() == 0
    )

    if model is None:
        log_dir = FLAGS.log_dir
        fpath = os.path.join(FLAGS.log_dir, 'training', 'FLAGS.z')
        FLAGS = AttrDict(dict(io.loadz(fpath)))
        FLAGS.log_dir = log_dir
        model, FLAGS, ckpt_dir = build_model(FLAGS, save_params=False)
    else:
        dirname = os.path.join(FLAGS.log_dir, 'training', 'checkpoints')
        ckpt_dir = dirname if IS_CHIEF else None

    outputs, data_strs = model.run_eager(FLAGS.run_steps,
                                         beta=FLAGS.beta_final,
                                         save_run_data=True,
                                         ckpt_dir=ckpt_dir)

    if IS_CHIEF:
        log_dir = FLAGS.log_dir
        runs_dir = os.path.join(log_dir, 'inference')
        io.check_else_make_dir(runs_dir)
        run_dir = os.path.join(runs_dir, f'run_{get_run_num(runs_dir)}')
        io.check_else_make_dir(run_dir)
        history_file = os.path.join(run_dir, 'inference_log.txt')
        with open(history_file, 'w') as f:
            f.write('\n'.join(data_strs))

        outputs_dir = os.path.join(run_dir, 'outputs')
        io.check_else_make_dir(outputs_dir)
        for key, val in outputs.items():
            out_file = os.path.join(outputs_dir, f'{key}.z')
            io.savez(np.array(val), out_file, key)

        plot_data(outputs, run_dir, FLAGS, thermalize=True)

    return model, outputs
