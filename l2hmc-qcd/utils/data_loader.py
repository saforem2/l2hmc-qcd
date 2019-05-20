"""
data_loader.py

Implements DataLoader class responsible for loading run_data.

Author: Sam Foreman (github: @saforem2)
Date: 05/03/2019
"""
import os
import pickle

import numpy as np


def load_params_from_dir(d):
    params_file = os.path.join(d, 'params.pkl')
    with open(params_file, 'rb') as f:
        params = pickle.load(f)

    return params


def get_run_dirs(root_dir):
    run_dirs = []
    root_dir = os.path.abspath(root_dir)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            keys = ['steps_', '_beta_', '_eps_']
            conditions = [key in dirname for key in keys]
            if np.alltrue(conditions):
                run_dirs.append(os.path.join(dirpath, dirname))

    return run_dirs

#  def make_fig_dirs(run_dirs):
#      for dirpath, dirnames, filenames in os.walk(root_dir):
#          for dirname in dirnames:
#              keys = ['steps_', '_beta_', '_eps_']
#              conditions = [key in dirname for key in keys]
#              if np.alltrue(conditions):
#                  run_dirs.append(os.path.join(dirpath, dirname))
#      return run_dirs


class DataLoader:
    def __init__(self, run_dir=None):
        self.run_dir = None

    def load_pkl_file(self, pkl_file):
        with open(pkl_file, 'rb') as f:
            contents = pickle.load(f)

        return contents

    def load_npz_file(self, npz_file):
        arr = np.load(npz_file)

        return arr.f.arr_0

    def load_observable(self, observable_str, run_dir=None):
        if run_dir is None:
            run_dir = self.run_dir

        if not observable_str.endswith('.pkl'):
            observable_str += '.pkl'

        obs_dir = os.path.join(run_dir, 'observables')
        obs_file = os.path.join(obs_dir, observable_str)

        return self.load_pkl_file(obs_file)

    def load_samples(self, run_dir):
        samples_file = os.path.join(run_dir, 'samples_out.npz')
        samples = self.load_npz_file(samples_file)

        return samples

    def load_leapfrogs(self, run_dir):
        lf_f_file = os.path.join(run_dir, 'lf_forward.npz')
        lf_b_file = os.path.join(run_dir, 'lf_backward.npz')
        lf_f = self.load_npz_file(lf_f_file)
        lf_b = self.load_npz_file(lf_b_file)

        return (lf_f, lf_b)

    def load_logdets(self, run_dir):
        logdets_f_file = os.path.join(run_dir, 'logdets_forward.npz')
        logdets_b_file = os.path.join(run_dir, 'logdets_backward.npz')

        logdets_f = self.load_npz_file(logdets_f_file)
        logdets_b = self.load_npz_file(logdets_b_file)

        return (logdets_f, logdets_b)

    def load_sumlogdets(self, run_dir):
        sumlogdet_f_file = os.path.join(run_dir, 'sumlogdet_forward.npz')
        sumlogdet_b_file = os.path.join(run_dir, 'sumlogdet_backward.npz')

        sumlogdet_f = self.load_npz_file(sumlogdet_f_file)
        sumlogdet_b = self.load_npz_file(sumlogdet_b_file)

        return (sumlogdet_f, sumlogdet_b)

    def load_plaqs(self, run_dir):
        obs_dir = os.path.join(run_dir, 'observables')
        plaqs_file = os.path.join(obs_dir, 'plaqs.pkl')

        return self.load_pkl_file(plaqs_file)

    def load_autocorrs(self, run_dir):
        obs_dir = os.path.join(run_dir, 'observables')
        autocorrs_file = os.path.join(obs_dir, 'charges_autocorrs.pkl')

        return self.load_pkl_file(autocorrs_file)

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
