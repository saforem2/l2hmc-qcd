import os
import sys
import pickle

from models.gauge_model import GaugeModel
from loggers.run_logger import RunLogger
from plotters.gauge_model_plotter import GaugeModelPlotter
from runners.gauge_model_runner import GaugeModelRunner


def load_model(log_dir):
    assert os.path.isdir(log_dir)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    assert os.path.isdir(checkpoint_dir)
    params_file = os.path.join(log_dir, 'parameters.pkl')
    with open(params_file, 'rb') as f:
        params = pickle.load(f)

    model = GaugeModel(params=params)
    run_logger = RunLogger(model, log_dir)
    plotter = GaugeModelPlotter(run_logger.figs_dir)

    return model, run_logger, plotter


def create_runner(sess, model, run_logger):
    return GaugeModelRunner(sess, model, run_logger)
