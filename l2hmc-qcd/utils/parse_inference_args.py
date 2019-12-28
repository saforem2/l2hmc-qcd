"""
parse_inference_args.py

Implements method for parsing command line arguments for `gauge_model.py`

Author: Sam Foreman (github: @saforem2)
Date: 04/09/2019
"""
import os
import sys
import argparse
import shlex

import utils.file_io as io

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
    parser.add_argument('--params_file',
                        dest='params_file',
                        required=False,
                        default=None,
                        help=("""Path to `params.pkl` or `parameters.pkl` file
                              containing the model parameters needed to run
                              inference."""))

    parser.add_argument('--eps',
                        dest='eps',
                        type=float,
                        default=None,
                        required=False,
                        help=("""Step size to use during inference. If no value
                              is passed, `eps = None` and the optimal step size
                              (determined during training) will be used."""))

    parser.add_argument('--samples_init',
                        dest='samples_init',
                        type=str,
                        default='random',
                        required=False,
                        help=("""String specifying how to initialize samples
                              when running inference. Possible values are:
                              'zeros', 'ones', or 'random'.
                              (Default: 'random')"""))

    parser.add_argument('--bootstrap_iters',
                        dest='bootstrap_iters',
                        type=int,
                        default=100,
                        required=False,
                        help=("""Number of bootstrap replication iterations to
                              use for error analysis. Note that larger values
                              will improve statistics at the cost of noticeable
                              slowdown when running error analysis. (Default:
                              100)."""))

    parser.add_argument('--skip_acl',
                        dest='skip_acl',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will skip the
                              calculation of the autocorrelation spectrum for
                              the GaussianMixtureModel (since it is a source of
                              measurable slowdown, especially for long
                              runs."""))

    parser.add_argument('--run_hmc',
                        dest='run_hmc',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will run generic HMC
                              immediately following the initial inference run
                              by setting each of the net weights to zero,
                              using the same step size and number of steps for
                              both."""))

    parser.add_argument("--run_steps",
                        dest="run_steps",
                        type=int,
                        default=5000,
                        required=False,
                        help=("""Number of evaluation 'run' steps to perform
                              after training (i.e. length of desired chain
                              generate using trained L2HMC sample).
                              (Default: 5000)"""))

    parser.add_argument("--beta_inference",
                        dest="beta_inference",
                        type=float,
                        default=None,
                        required=False,
                        help=("""Flag specifying a singular value of beta at
                              which to run inference using the trained
                              L2HMC sampler. (Default: None"""))

    parser.add_argument('--plot_lf',
                        dest='plot_lf',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will set
                              `--plot_lf=True`, and will plot the 'metric'
                              distance between subsequent configurations, as
                              well as the determinant of the Jacobian of the
                              transformation for each individual leapfrog step,
                              as well as each molecular dynamics step (with
                              Metrpolis-Hastings accept/reject).\n
                              When plotting the determinant of the Jacobian
                              following the MD update, we actually calculate
                              the sum of the determinants from each individual
                              LF step since this is the quantity that actually
                              enters into the MH acceptance probability.
                              (Default: `--plot_lf=False`, i.e. `--plot_lf` is
                              not passed mostly just beause the plots are
                              extremely large (many LF steps during inference)
                              and take a while to actually generate.)"""))

    parser.add_argument('--loop_transl_weights',
                        dest='loop_transl_weights',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will loop over different
                              values for the `translation_weight` in
                              `net_weights`, which is believed to be causing
                              the discrepancy between the observed and expected
                              value of the average plaquette when running
                              inference."""))

    parser.add_argument('--v_scale_weight',
                        dest='v_scale_weight',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Specify the value of the `scale_weight`
                              parameter, a multiplicative weight that scales
                              the contribution of the `scale` (S) function when
                              performing the augmented L2HMC molecular dynamics
                              update."""))

    parser.add_argument('--v_translation_weight',
                        dest='v_translation_weight',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Specify the value of the `translation_weight`
                              parameter, a multiplicative weight that scales
                              the contribution of the `translation` (T)
                              function when performing the augmented L2HMC
                              molecular dynamics update."""))

    parser.add_argument('--v_transformation_weight',
                        dest='v_transformation_weight',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Specify the value of the
                              `transformation_weight` parameter, a
                              multiplicative weight that scales the
                              contribution of the `transformation` (Q) function
                              when performing the augmented L2HMC molecular
                              dynamics update."""))

    parser.add_argument('--x_scale_weight',
                        dest='x_scale_weight',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Specify the value of the `scale_weight`
                              parameter, a multiplicative weight that scales
                              the contribution of the `scale` (S) function when
                              performing the augmented L2HMC molecular dynamics
                              update."""))

    parser.add_argument('--x_translation_weight',
                        dest='x_translation_weight',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Specify the value of the `translation_weight`
                              parameter, a multiplicative weight that scales
                              the contribution of the `translation` (T)
                              function when performing the augmented L2HMC
                              molecular dynamics update."""))

    parser.add_argument('--x_transformation_weight',
                        dest='x_transformation_weight',
                        type=float,
                        default=1.,
                        required=False,
                        help=("""Specify the value of the
                              `transformation_weight` parameter, a
                              multiplicative weight that scales the
                              contribution of the `transformation` (Q) function
                              when performing the augmented L2HMC molecular
                              dynamics update."""))

    parser.add_argument('--loop_net_weights',
                        dest='loop_net_weights',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will loop over all 64
                              possible values of `net_weights`."""))

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

    parser.add_argument('--skip_existing',
                        dest='skip_existing',
                        action='store_true',
                        required=False,
                        help=("""Flag that when passed will skip inference run
                              if it has already been completed with the same
                              parameters."""))

    if sys.argv[1].startswith('@'):
        args = parser.parse_args(shlex.split(open(sys.argv[1][1:]).read(),
                                             comments=True))
    else:
        args = parser.parse_args()

    return args
