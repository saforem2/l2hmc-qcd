"""
parse_inference_args.py

Implements method for parsing command line arguments for `gauge_model.py`

Author: Sam Foreman (github: @saforem2)
Date: 04/09/2019
"""
import sys
import argparse
import shlex

#  from config import process_config
#  from attr_dict import AttrDict

DESCRIPTION = 'Run inference on trained L2HMC model.'


# =============================================================================
#  * NOTE:
#      - if action == 'store_true':
#          The argument is FALSE by default. Passing this flag will cause the
#          argument to be ''stored true''.
#      - if action == 'store_false':
#          The argument is TRUE by default. Passing this flag will cause the
#          argument to be ''stored false''.
# =============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        fromfile_prefix_chars='@',
    )
    ###########################################################################
    #                          Lattice parameters                             #
    ###########################################################################
    parser.add_argument('--log_dir',
                        dest='log_dir',
                        required=False,
                        default=None,
                        help=("""Path to `log_dir` containing trained model on
                              which to run inference."""))

    parser.add_argument('--hmc',
                        dest='hmc',
                        action='store_true',
                        required=False,
                        help="""Run generic HMC.""")

    parser.add_argument("--run_steps",
                        dest="run_steps",
                        type=int,
                        default=5000,
                        required=False,
                        help=("""Number of evaluation 'run' steps to perform
                              after training (i.e. length of desired chain
                              generate using trained L2HMC sample).
                              (Default: 5000)"""))

    parser.add_argument("--beta",
                        dest="beta",
                        type=float,
                        default=None,
                        required=False,
                        help=("""Flag specifying a singular value of beta at
                              which to run inference using the trained
                              L2HMC sampler. (Default: None"""))

    parser.add_argument("--eps",
                        dest="eps",
                        type=float,
                        default=None,
                        required=False,
                        help=("""Flag specifying value of `eps` to use."""))

    parser.add_argument('--save_samples',
                        dest='save_samples',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will set
                              `--save_samples=True`, and save the samples
                              generated during the `run` phase.
                              (Default: `--save_samples=False, i.e.
                              `--save_samples` is not passed).\n
                              WARNING!! This is very data intensive."""))

    if sys.argv[1].startswith('@'):
        args = parser.parse_args(shlex.split(open(sys.argv[1][1:]).read(),
                                             comments=True))
    else:
        args = parser.parse_args()

    return args
