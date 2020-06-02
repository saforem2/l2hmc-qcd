"""
gauge_inference.py

Runs inference on trained model using tensorflow.
"""
import os
import sys
import shlex
import shutil
import argparse

import utils.file_io as io

from runners.runner import RunnerTF, get_thermalized_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference on trained model using tensorflow.',
        fromfile_prefix_chars='@',
    )
    parser.add_argument('--beta', dest='beta', type=float,
                        required=False, default=None,
                        help=("""Value of `beta` at which to run
                              inference."""))

    parser.add_argument('--log_dir', dest='log_dir',
                        required=False, default=None,
                        help=("""Log dir containing saved model
                              checkpoints."""))

    parser.add_argument('--eps', dest='eps', type=float,
                        required=False, default=None,
                        help=("""Step size (`eps`) to use in leapfrog
                              integrator."""))

    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        required=False, default=None,
                        help=("""Batch size to use (# of chains to run in
                              parallel."""))

    parser.add_argument('--hmc', dest='hmc',
                        required=False, action='store_true',
                        help=("""Flag that when passed will run generic
                              HMC."""))

    parser.add_argument('--run_steps', dest='run_steps', type=int,
                        required=False, default=10000,
                        help=("""Number of inference steps to run."""))

    parser.add_argument('--plot_chains', dest='plot_chains', type=int,
                        required=False, default=None,
                        help=("""Number of chains to include when making
                              plots."""))

    parser.add_argument('--print_steps', dest='print_steps', type=int,
                        required=False, default=10,
                        help=("""Frequency with which to print data."""))

    parser.add_argument('--dont_save', dest='dont_save',
                        required=False, action='store_true',
                        help=("""Flag that when passed prevents run data from
                              being saved."""))

    parser.add_argument('--skip_existing', dest='skip_existing',
                        required=False, action='store_true',
                        help=("""Flag that when passed will prevent inference
                              from being run (again) if there exists a
                              (non-empty) directory containing inference data
                              for the given parameter values."""))

    parser.add_argument('--therm', dest='therm',
                        required=False, action='store_true',
                        help=("""FLag that when pased will initially run
                              generic HMC to get a thermalized configuration
                              which will then be used as the initial state for
                              L2HMC."""))

    if sys.argv[1].startswith('@'):
        args = parser.parse_args(shlex.split(open(sys.argv[1][1:]).read(),
                                             comments=True))
    else:
        args = parser.parse_args()

    return args


def check_existing(log_dir, run_str):
    """Check if there is a (non-empty) directory containing inference data."""
    runs_dir = os.path.join(log_dir, 'runs_tf')
    run_dirs = [
        os.path.join(runs_dir, i) for i in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, i))
    ]
    run_str = '_'.join(run_str.split('_')[:6])
    matched_dirs = [i for i in run_dirs if run_str in i]
    existing = False
    if len(matched_dirs) > 0:
        for matched_dir in matched_dirs:
            contents = os.listdir(matched_dir)
            if len(contents) > 0:
                existing = True

    return existing


def main(args):
    """Main method."""
    runner = RunnerTF(args)
    if args.skip_existing:
        if check_existing(runner.log_dir, runner.run_str):
            return None

    x = None
    if args.therm:
        x = get_thermalized_config(runner.log_dir, runner.beta)

    run_data = runner.inference(x=x, run_steps=args.run_steps)
    run_data.plot(runner.fig_dir, runner.title_str,
                  num_chains=args.plot_chains)
    _, _, fpaths = run_data.log_summary(runner.fig_dir, n_boot=10)
    if not args.dont_save:
        run_data.save(run_dir=runner.run_dir, save_samples=False)

    for fpath in fpaths:
        _ = shutil.copy2(fpath, runner.run_dir)

    return run_data


if __name__ == '__main__':
    FLAGS = parse_args()
    for key, val in FLAGS.__dict__.items():
        io.log(f'{key}: {val}\n')

    _ = main(FLAGS)
