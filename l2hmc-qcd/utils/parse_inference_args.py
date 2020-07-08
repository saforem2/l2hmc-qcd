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

DESCRIPTION = (
    """Run inference, either by loading a trained model from a checkpoint
    (specified by the `--log_dir` flag) or with generic HMC by creating a new
    `GaugeModel` instance."""
)


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
    parser.add_argument('--overwrite',
                        dest='overwrite',
                        required=False,
                        action='store_true',
                        help=("""Flag that when passed will overwrite existing
                              run directory with new inference data."""))

    parser.add_argument('--log_dir',
                        dest='log_dir',
                        required=False,
                        default=None,
                        help=("""Path to `log_dir` containing trained model on
                              which to run inference."""))

    parser.add_argument("--run_steps",
                        dest="run_steps",
                        type=int,
                        default=2000,
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

    #################################
    # Flags for running generic HMC
    #################################

    parser.add_argument('--hmc',
                        dest='hmc',
                        action='store_true',
                        required=False,
                        help="""Run generic HMC.""")

    parser.add_argument('--lattice_shape',
                        dest='lattice_shape',
                        type=lambda s: [int(i) for i in s.split(',')],
                        default="128, 16, 16, 2",
                        required=False,
                        help=("""Specifies the shape of our data, with:
                              lattice_shape =
                              (batch_size, time_size, space_size, dim)
                              Defaults to: (128, 16, 16, 2)"""))

    parser.add_argument("-n", "--num_steps",
                        dest="num_steps",
                        type=int,
                        default=1,
                        required=False,
                        help=("""Number of leapfrog steps to use in (augmented)
                              HMC sampler.\n(Default: 5)"""))

    if sys.argv[1].startswith('@'):
        args = parser.parse_args(shlex.split(open(sys.argv[1][1:]).read(),
                                             comments=True))
    else:
        args = parser.parse_args()

    return args
