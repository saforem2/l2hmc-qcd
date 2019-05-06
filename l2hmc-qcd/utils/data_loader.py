"""
data_loader.py

Implements DataLoader class responsible for loading run_data.

Author: Sam Foreman (github: @saforem2)
Date: 05/03/2019
"""
import os
import pickle


def load_params_from_dir(d):
    params_file = os.path.join(d, 'params.pkl')
    with open(params_file, 'rb') as f:
        params = pickle.load(f)

    return params


class DataLoader:
    def __init__(self, run_dir=None):
        self.run_dir = None

    def load_pkl_file(self, pkl_file):
        with open(pkl_file, 'rb') as f:
            contents = pickle.load(f)

        return contents

    def load_observable(self, observable_str, run_dir=None):
        if run_dir is None:
            run_dir = self.run_dir

        obs_dir = os.path.join()
        obs_file = os.path.join(obs_dir, observable_str)

        return self.load_pkl_file(obs_file)

    def load_params(self, run_dir=None):
        if run_dir is None:
            run_dir = self.run_dir
        params_file = os.path.join(run_dir, 'parameters.pkl')

        return self.load_pkl_file(params_file)

    def load_run_data(self, run_dir=None):
        if run_dir is None:
            run_dir = self.run_dir

        run_data_file = os.path.join(run_dir, 'run_data.pkl')

        return self.load_pkl_file(run_data_file)

    def load_run_stats(self, run_dir=None):
        if run_dir is None:
            run_dir = self.run_dir

        run_stats_file = os.path.join(run_dir, 'run_stats.pkl')

        return self.load_pkl_file(run_stats_file)

    def load_observables(self, run_dir=None):
        if run_dir is None:
            run_dir = self.run_dir

        obs_dir = os.path.join(run_dir, 'observables')
        files = os.listdir(obs_dir)

        obs_names = [i.rstrip('.pkl') for i in files]
        obs_files = [os.path.join(obs_dir, i) for i in files]

        observables = {
            n: self.load_pkl_file(f) for (n, f) in zip(obs_names, obs_files)
        }

        return observables
