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

from runners.runner_tf import RunData, RunnerTF


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

    if sys.argv[1].startswith('@'):
        args = parser.parse_args(shlex.split(open(sys.argv[1][1:]).read(),
                                             comments=True))
    else:
        args = parser.parse_args()

    return args


def main(FLAGS):
    """Main method."""
    runner = RunnerTF(FLAGS)
    run_data = runner.inference(FLAGS.run_steps)
    run_data.plot(runner.fig_dir, runner.title_str,
                  num_chains=FLAGS.plot_chains)
    out_file = os.path.join(runner.fig_dir, 'run_summary.txt')
    _, _ = run_data.log_summary(out_file=out_file, n_boot=10)

    if not FLAGS.dont_save:
        run_data.save(run_dir=runner.run_dir)

    _ = shutil.copy2(out_file, runner.run_dir)

    return run_data


if __name__ == '__main__':
    CLI_FLAGS = parse_args()
    for key, val in CLI_FLAGS.__dict__.items():
        io.log(f'{key}: {val}\n')

    _ = main(CLI_FLAGS)
